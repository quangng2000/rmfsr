"""Evaluate a LOCAL frozen EMA checkpoint; never accesses a trainer or cloud GPU.

SIG2024 has no clean waveform reference. Report learned quality predictors and
Whisper Turbo WER against its published transcripts. Human P.804 and the paper's
unspecified f_max estimator are deliberately not replaced with guessed scores.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch

from .flow import pink_noise_like, sample
from .model import model_from_checkpoint
from .spectral import Spectral


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def read_wave(path):
    wave, rate = sf.read(path, dtype='float32', always_2d=True)
    wave = wave.mean(axis=1)
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError(f'Empty or non-finite audio: {path}')
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        wave = resample_poly(wave, 16000 // divisor, rate // divisor)
    return np.ascontiguousarray(wave, dtype=np.float32)


class DNSMOSSIG:
    """Standard (non-personalized) P.835 SIG, using Microsoft's ONNX weights.

    Windowing, repetition and polynomial follow DNS-Challenge/dnsmos_local.py
    at 591184a9fcb2cbdec02520fed81a32bbbf9d73ff (license saved with inputs).
    The unrelated P.808 head is not needed for SIG.
    """
    def __init__(self, weights, threads=2):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(weights), sess_options=opts,
                                           providers=['CPUExecutionProvider'])

    def __call__(self, audio):
        if not len(audio):
            raise ValueError('DNSMOS requires nonempty audio')
        while len(audio) < 144160:
            audio = np.concatenate((audio, audio))
        hops = int(np.floor(len(audio) / 16000) - 9.01) + 1
        scores = []
        for index in range(hops):
            segment = audio[index * 16000:int((index + 9.01) * 16000)]
            if len(segment) < 144160:
                continue
            raw = self.session.run(None, {'input_1': segment[None].astype('float32')})[0][0]
            scores.append(float(np.polyval([-0.08397278, 1.22083953, 0.0052439], raw[0])))
        if not scores:
            raise ValueError('DNSMOS produced no windows')
        return float(np.mean(scores))


def word_counts(reference, hypothesis):
    import jiwer
    from whisper.normalizers import BasicTextNormalizer
    # SIG2024 includes languages beyond English. Preserve diacritics and avoid
    # applying English number/contraction rules to German, Spanish, Dutch, etc.
    normalize = BasicTextNormalizer(remove_diacritics=False, split_letters=False)
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref.strip():
        raise ValueError('Transcript has no reference words after normalization')
    result = jiwer.process_words(ref, hyp)
    words = result.hits + result.substitutions + result.deletions
    return {'reference_normalized': ref, 'hypothesis_normalized': hyp,
            'reference_words': words, 'substitutions': result.substitutions,
            'deletions': result.deletions, 'insertions': result.insertions,
            'wer': result.wer}


def aggregate(rows):
    result = {}
    for variant in sorted({row['variant'] for row in rows}):
        group = [row for row in rows if row['variant'] == variant]
        summary = {'clips': len(group)}
        for metric in ('distillmos', 'dnsmos_sig'):
            values = [r[metric] for r in group if metric in r]
            if values:
                summary[metric] = float(np.mean(values))
                summary[metric + '_clips'] = len(values)
        scored = [r['asr'] for r in group if 'asr' in r]
        if scored:
            words = sum(r['reference_words'] for r in scored)
            errors = sum(r['substitutions'] + r['deletions'] + r['insertions'] for r in scored)
            summary.update(wer=errors / words, wer_clips=len(scored),
                           reference_words=words, word_errors=errors)
        result[variant] = summary
    return result


def run(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    items = manifest['items'][:args.limit] if args.limit else manifest['items']
    if not items:
        raise ValueError('The evaluation manifest is empty')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['config']['sample_rate'] != 16000:
        raise ValueError('This evaluation currently expects a 16 kHz checkpoint')
    provenance = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'checkpoint_sha256': sha256(args.checkpoint), 'step': checkpoint['step'],
        'weights': 'ema', 'training_profile': checkpoint['config'].get('augmentation_profile'),
        'manifest_sha256': sha256(args.manifest), 'dataset': manifest,
        'clips': len(items), 'nfes': args.nfes, 'seed': args.seed,
        'sample_rate': 16000, 'device': 'cpu', 'threads': args.threads,
        'audio_policy': 'full recording; mono mean; scipy polyphase resample; no added gain or clipping; FLOAT WAV',
        'whisper': 'large-v3-turbo', 'whisper_sha256': sha256(args.whisper_weights),
        'asr_options': {'task': 'transcribe', 'temperature': 0.0,
                        'condition_on_previous_text': False, 'fp16': False},
        'wer_policy': 'Automatic language detection; multilingual Whisper BasicTextNormalizer preserving diacritics; corpus total S+D+I / reference words; UNKNOWN transcript tags removed, not wildcards',
        'dnsmos': 'standard non-personalized P.835 SIG; official windowing and polynomial',
        'dnsmos_sha256': sha256(args.dnsmos_weights),
        'versions': {p: importlib.metadata.version(p) for p in
                     ('torch', 'numpy', 'scipy', 'soundfile', 'distillmos',
                      'torchaudio', 'onnxruntime', 'openai-whisper', 'jiwer')},
        'source_sha256': {p: sha256(Path(__file__).parent / p) for p in
                          ('paper_evaluate.py', 'model.py', 'flow.py', 'spectral.py')},
        'limitations': [
            'A subset is preliminary and must not be compared directly to the published 500-clip average.',
            'Paper does not fully specify ASR decoding/normalization or DNSMOS variant; choices are recorded.',
            'No human ITU P.804 scores; no exact f_max implementation supplied by the paper.',
            'This is a frozen checkpoint measurement, not evidence of a trend across checkpoints.',
        ],
    }
    previous = out / 'provenance.json'
    if previous.exists() and json.loads(previous.read_text()) != provenance:
        raise ValueError('Output directory belongs to different evaluation inputs/settings; use a new output')
    save_json(previous, provenance)
    model = model_from_checkpoint(checkpoint).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['ema'])
    del checkpoint
    spectral = Spectral(16000)
    import distillmos
    quality = distillmos.ConvTransformerSQAModel().eval().requires_grad_(False)
    dns = DNSMOSSIG(args.dnsmos_weights, args.threads)
    rows = []

    def report(phase, complete=False):
        save_json(out / 'results.json', {'complete': complete, 'phase': phase,
                  'updated_at_unix': time.time(), 'checkpoint_step': provenance['step'],
                  'selected_clips': len(items), 'summary': aggregate(rows), 'rows': rows})

    for index, item in enumerate(items):
        audio = read_wave(item['audio'])
        encoded = spectral.encode(torch.from_numpy(audio))
        # Stable per-clip seed, identical noise for every NFE in this clip.
        clip_seed = args.seed + int(hashlib.sha256(item['id'].encode()).hexdigest()[:8], 16)
        noise = pink_noise_like(encoded, torch.Generator().manual_seed(clip_seed))
        for nfe in [0, *args.nfes]:
            variant = 'unprocessed' if nfe == 0 else f'nfe{nfe}'
            folder = out / variant
            folder.mkdir(exist_ok=True)
            row_path = folder / (item['id'] + '.json')
            wave_path = folder / (item['id'] + '.wav')
            if row_path.exists():
                rows.append(json.loads(row_path.read_text()))
                continue
            started = time.monotonic()
            with torch.inference_mode():
                restored = audio if not nfe else spectral.decode(
                    sample(model, encoded, steps=nfe, noise=noise), len(audio))[0].numpy()
                if not np.isfinite(restored).all():
                    raise ValueError(f'Non-finite model output: {item["id"]} {variant}')
                sf.write(wave_path, restored, 16000, subtype='FLOAT')
                row = {'id': item['id'], 'variant': variant, 'audio': str(wave_path),
                       'seconds': len(audio) / 16000, 'peak': float(np.abs(restored).max()),
                       'distillmos': float(quality(torch.from_numpy(restored)[None]).item()),
                       'dnsmos_sig': dns(restored), 'elapsed_seconds': time.monotonic() - started}
            save_json(row_path, row)
            rows.append(row)
            report('restoration_and_quality')
            print(f'QUALITY {index+1}/{len(items)} {variant} DistillMOS={row["distillmos"]:.3f} DNSMOS_SIG={row["dnsmos_sig"]:.3f} elapsed={row["elapsed_seconds"]:.1f}s', flush=True)
    del model, quality, dns
    import whisper
    asr = whisper.load_model(str(Path(args.whisper_weights).resolve()), device='cpu')
    transcripts = {item['id']: Path(item['transcript']).read_text().strip() for item in items}
    for index, row in enumerate(rows):
        if 'asr' not in row:
            started = time.monotonic()
            transcription = asr.transcribe(read_wave(row['audio']), verbose=None, **provenance['asr_options'])
            hypothesis = transcription['text']
            row['asr'] = word_counts(transcripts[row['id']], hypothesis)
            row['asr']['hypothesis'] = hypothesis
            row['asr']['detected_language'] = transcription['language']
            row['asr']['elapsed_seconds'] = time.monotonic() - started
            save_json(out / row['variant'] / (row['id'] + '.json'), row)
            report('whisper_wer')
            print(f'ASR {index+1}/{len(rows)} {row["variant"]} WER={row["asr"]["wer"]:.3f} elapsed={row["asr"]["elapsed_seconds"]:.1f}s', flush=True)
    report('complete', complete=True)
    print(json.dumps(aggregate(rows), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dnsmos-weights', required=True)
    parser.add_argument('--whisper-weights', required=True)
    parser.add_argument('--nfes', type=int, nargs='+', default=[5])
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.threads < 1 or any(n < 1 for n in args.nfes) or len(set(args.nfes)) != len(args.nfes):
        parser.error('threads/NFEs must be positive and NFEs unique')
    if args.limit is not None and args.limit < 1:
        parser.error('limit must be positive')
    run(args)


if __name__ == '__main__':
    main()
