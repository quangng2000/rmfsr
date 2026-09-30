"""Deterministic held-out restoration from damaged audio, with DP diagnostics.

The clean waveform is used only as a reference and in the separately named
teacher-input DP diagnostic. Quality metrics always use an actual inference
trajectory beginning with damaged audio plus independent fixed pink noise.
"""
import json
from pathlib import Path
import warnings

import numpy as np
import soundfile as sf
import torch
from pystoi import stoi

from .flow import pink_noise_like, sample


def _positive_integer(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f'{name} must be a {"nonnegative" if allow_zero else "positive"} integer')
    return value


def _waveform(value, name):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 1 or not value.size:
        raise ValueError(f'{name} must be a nonempty mono waveform')
    if not np.isfinite(value).all():
        raise FloatingPointError(f'Nonfinite {name}')
    return value


def _stoi(reference, estimate, sr):
    # STOI's short-time analysis needs enough nonsilent frames. Short training
    # fixtures and silent crops must not contribute a meaningless dummy score.
    if len(reference) < sr or np.std(reference) < 1e-7 or np.std(estimate) < 1e-7:
        return None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        value = float(stoi(reference, estimate, sr, extended=False))
    if any('not enough' in str(item.message).lower() for item in caught):
        return None
    return value if np.isfinite(value) else None


def _errors(reference, estimate, mask):
    error = np.asarray(estimate, np.float64) - np.asarray(reference, np.float64)
    square = error * error
    return np.array([square.sum(), square[mask].sum(), square[~mask].sum()], np.float64)


def _masked_mse(total, count):
    return float(total / count) if count else None


@torch.no_grad()
def validate_restoration(ema, validation, spectral, cfg, device, output_dir=None, save_audio=False):
    """Return flat training metrics; optionally save a review report and audio.

    Every call regenerates the same held-out pairs with an isolated NumPy RNG.
    CPU-only Torch generators isolate both noise streams from training RNG and
    use identical restoration noise across NFE comparisons. ``chunk_frames=10``
    exercises persistent model caches; configure ``validation_chunk_frames=0``
    to evaluate whole-utterance inference. No audio is saved outside explicit
    reviews, and exports are capped by ``validation_audio_examples`` (default2).
    """
    count = _positive_integer(cfg.get('validation_examples', 1), 'validation_examples')
    audio_count = _positive_integer(cfg.get('validation_audio_examples', 2),
                                    'validation_audio_examples', allow_zero=True)
    chunk_frames = _positive_integer(cfg.get('validation_chunk_frames', 10),
                                     'validation_chunk_frames', allow_zero=True) or None
    nfes = cfg.get('inference_evaluations_to_compare', [1, 2, 5])
    if not isinstance(nfes, (list, tuple)) or not nfes:
        raise ValueError('inference_evaluations_to_compare must be a nonempty list')
    nfes = [_positive_integer(value, 'inference evaluations') for value in nfes]
    if len(set(nfes)) != len(nfes):
        raise ValueError('inference_evaluations_to_compare must contain unique values')
    if save_audio and output_dir is None:
        raise ValueError('Saving validation audio requires output_dir')
    sr = _positive_integer(cfg.get('sample_rate', spectral.sr), 'sample_rate')
    if sr != spectral.sr:
        raise ValueError('Validation spectral sample rate differs from configuration')
    seed = cfg['seed']
    diagnostic_rng = torch.Generator().manual_seed(seed + 2000)
    restoration_rng = torch.Generator().manual_seed(seed + 3000)
    original_rng = validation.rng
    original_mode = ema.training
    validation.rng = np.random.default_rng(seed + 1000)
    ema.eval()
    spectral_totals = np.zeros(2, np.float64)
    waveform_totals = {nfe: np.zeros(3, np.float64) for nfe in nfes}
    input_totals = np.zeros(3, np.float64)
    sample_counts = np.zeros(3, np.int64)
    stoi_scores = {nfe: [] for nfe in nfes}
    input_stoi = []
    examples = []
    audio = []
    try:
        for index in range(count):
            try:
                clean, damaged, missing, kinds = validation.one()
            except StopIteration as error:
                raise ValueError('Validation dataset ran out of examples') from error
            clean, damaged = _waveform(clean, 'clean target'), _waveform(damaged, 'damaged input')
            if clean.shape != damaged.shape:
                raise ValueError('Validation target and input lengths differ')
            missing = np.asarray(missing, dtype=bool)
            if missing.shape != clean.shape:
                raise ValueError('Validation gap mask does not match waveform')
            counts = np.array([len(clean), missing.sum(), (~missing).sum()], np.int64)
            sample_counts += counts
            input_totals += _errors(clean, damaged, missing)
            input_score = _stoi(clean, damaged, sr)
            if input_score is not None:
                input_stoi.append(input_score)
            x = spectral.encode(torch.from_numpy(clean)).to(device)
            y = spectral.encode(torch.from_numpy(damaged)).to(device)
            if x.shape != y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise FloatingPointError('Invalid encoded validation spectra')
            t = x.new_full((1,), .5)
            diagnostic_noise = torch.randn(x.shape, generator=diagnostic_rng, dtype=x.dtype)
            if cfg.get('validation_pink_noise', False):
                weights = torch.arange(x.shape[2], dtype=x.dtype).clamp_min(1).rsqrt()[None, None, :, None]
                diagnostic_noise *= weights / weights.square().mean().sqrt()
            prediction = ema(.5 * x + .5 * y + .15 * diagnostic_noise.to(device), y, t, t)
            if prediction.shape != x.shape or not torch.isfinite(prediction).all():
                raise FloatingPointError('Invalid diagonal validation prediction')
            diagnostic_errors = [(prediction - x).square().mean().item(), (y - x).square().mean().item()]
            if not np.isfinite(diagnostic_errors).all():
                raise FloatingPointError('Nonfinite diagonal validation metric')
            spectral_totals += diagnostic_errors
            # Draw once on CPU and reuse the exact sample for every NFE. The
            # sampler starts at y + .3*noise and receives no clean target.
            noise = pink_noise_like(y.detach().cpu(), restoration_rng).to(device)
            item_audio = {'clean': clean, 'damaged': damaged}
            item = dict(index=index, samples=len(clean), gap_samples=int(missing.sum()),
                        degradations=list(kinds), restoration={})
            for nfe in nfes:
                restored_spectrum = sample(ema, y, steps=nfe, noise=noise, chunk_frames=chunk_frames)
                if restored_spectrum.shape != y.shape or not torch.isfinite(restored_spectrum).all():
                    raise FloatingPointError(f'Invalid restoration spectrum at NFE {nfe}')
                decoded = spectral.decode(restored_spectrum, len(clean))
                if decoded.shape != (1, len(clean)):
                    raise ValueError('Decoded validation waveform does not match target length')
                wave = _waveform(decoded[0], f'restoration at NFE {nfe}')
                errors = _errors(clean, wave, missing)
                waveform_totals[nfe] += errors
                score = _stoi(clean, wave, sr)
                if score is not None:
                    stoi_scores[nfe].append(score)
                item['restoration'][str(nfe)] = dict(
                    mse=float(errors[0] / counts[0]), gap_mse=_masked_mse(errors[1], counts[1]),
                    intact_mse=_masked_mse(errors[2], counts[2]), stoi=score)
                if save_audio and index < audio_count:
                    item_audio[f'restoration_nfe{nfe}'] = wave
            examples.append(item)
            if save_audio and index < audio_count:
                audio.append((index, item_audio))
    finally:
        validation.rng = original_rng
        ema.train(original_mode)
    metrics = dict(validation_dp_mse=float(spectral_totals[0] / count),
                   validation_input_mse=float(spectral_totals[1] / count),
                   validation_examples=count, validation_waveform_samples=int(sample_counts[0]),
                   validation_gap_samples=int(sample_counts[1]), validation_intact_samples=int(sample_counts[2]),
                   validation_input_waveform_mse=float(input_totals[0] / sample_counts[0]),
                   validation_input_gap_mse=_masked_mse(input_totals[1], sample_counts[1]),
                   validation_input_intact_mse=_masked_mse(input_totals[2], sample_counts[2]),
                   validation_input_stoi=float(np.mean(input_stoi)) if input_stoi else None,
                   validation_input_stoi_examples=len(input_stoi))
    for nfe in nfes:
        errors = waveform_totals[nfe]
        metrics.update({f'validation_restoration_mse_nfe{nfe}': float(errors[0] / sample_counts[0]),
                        f'validation_gap_mse_nfe{nfe}': _masked_mse(errors[1], sample_counts[1]),
                        f'validation_intact_mse_nfe{nfe}': _masked_mse(errors[2], sample_counts[2]),
                        f'validation_stoi_nfe{nfe}': float(np.mean(stoi_scores[nfe])) if stoi_scores[nfe] else None,
                        f'validation_stoi_examples_nfe{nfe}': len(stoi_scores[nfe])})
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # One gain for all retained references, inputs, and outputs. PCM exports
        # remain directly comparable and cannot hide attenuation by normalizing
        # each restoration independently.
        peak = max((float(np.max(np.abs(wave))) for _, waves in audio for wave in waves.values()), default=0.)
        gain = min(1., .95 / max(peak, 1e-6))
        exports = []
        for index, waves in audio:
            directory = output_dir / f'example-{index:03d}'
            directory.mkdir(exist_ok=True)
            for name, wave in waves.items():
                path = directory / f'{name}.wav'
                sf.write(path, wave * gain, sr, subtype='PCM_16')
                exports.append(str(path.relative_to(output_dir)))
        report = dict(metrics=metrics, inference_evaluations=nfes, chunk_frames=chunk_frames or 0,
                      sample_rate=sr, seed=seed, playback_gain=gain if audio else None,
                      audio_examples=len(audio), audio_files=exports, examples=examples)
        (output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return metrics
