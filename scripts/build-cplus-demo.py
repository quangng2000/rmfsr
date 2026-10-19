"""Add a verified C+ listening comparison without rebuilding the existing gallery."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from results_playback import digest, export_track, levels


def renderer():
    spec = importlib.util.spec_from_file_location('gallery', Path(__file__).with_name('build-results-page.py'))
    gallery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gallery)
    return gallery.spectrogram


def sources(source, report):
    names = ['input', 'clean'] + [f'output_nfe{n}' for n in report['nfes']]
    waves = {}
    for name in names:
        wave, rate = sf.read(source / f'{name}.wav', dtype='float32')
        if rate != 16000 or wave.shape != (64000,) or not np.isfinite(wave).all():
            raise ValueError(f'Invalid C+ audio: {name}')
        waves[name] = wave
    return waves


def original_track(name, wave, context):
    source, docs, gain, render = context
    stem = name.replace('output_', '')
    audio = Path(f'assets/audio/cplus/{stem}.wav')
    image = Path(f'assets/spectrograms/cplus/{stem}.png')
    (docs / audio).parent.mkdir(parents=True, exist_ok=True)
    sf.write(docs / audio, wave * gain, 16000, subtype='PCM_16')
    played, _ = sf.read(docs / audio, dtype='float32')
    if np.abs(played).max() > .9501 or np.abs(played - wave * gain).max() > 1 / 32768 + 1e-7:
        raise ValueError('Invalid original playback export')
    render(played, docs / image)
    audio_sha, image_sha = digest(docs / audio), digest(docs / image)
    return dict(audio=f'{audio}?v={audio_sha[:16]}', spectrogram=f'{image}?v={image_sha[:16]}',
                source_sha256=digest(source / f'{name}.wav'), audio_sha256=audio_sha,
                spectrogram_sha256=image_sha)


def sample_metadata(report):
    return dict(id='cplus', title='C+ paired example', description='125,000 updates · clean reference available',
                  duration_seconds=4., nfes=report['nfes'], references=[], outputs=[],
                  checkpoint=dict(step=report['training_step'], weights=report['weights'],
                      sha256=report['checkpoint_sha256'], conditioning_architecture=report['architecture']),
                  playback=dict(common_gain_db=report['common_playback_gain_db']),
                  inference={k: report[k] for k in ['device', 'noise_seed', 'input_sha256', 'cached_features_sha256']},
                  source_note='Synthetic paired C+ example. STOI is measured on this one clip, not the validation set. All outputs use the production EMA; no experimental FiLM.')


def build_sample(source, docs, report):
    waves, render, sample = sources(source, report), renderer(), sample_metadata(report)
    original_gain = 10 ** (report['common_playback_gain_db'] / 20)
    gains = {name: levels(waves['input'])['rms'] / levels(wave)['rms'] for name, wave in waves.items()}
    safety = min(1., .95 / max(levels(w)['peak'] * gains[n] for n, w in waves.items()))
    for name, wave in waves.items():
        track = original_track(name, wave, (source, docs, original_gain, render))
        track.update(stoi=report['metrics'][name]['stoi'])
        if name.startswith('output_'):
            track.update(nfe=int(name.removeprefix('output_nfe')), label=name.replace('output_nfe', 'NFE '))
        else:
            track.update(kind=name, label='Damaged input' if name == 'input' else 'Clean reference')
        export_track(track, wave, (docs, gains[name] * safety, render))
        sample['outputs' if 'nfe' in track else 'references'].append(track)
    values = [t['matched']['rms_dbfs'] for t in sample['references'] + sample['outputs']]
    sample['rms_matched'] = dict(reference='damaged input', common_safety_gain_db=float(20 * np.log10(safety)), maximum_rms_spread_db=max(values) - min(values))
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--docs', type=Path, default=Path(__file__).resolve().parents[1] / 'docs')
    args = parser.parse_args()
    report = json.loads((args.source / 'results.json').read_text())
    if not report['complete'] or report['training_step'] != 125000 or report['weights'] != 'EMA' or report['nfes'] != [1, 5, 20, 100]:
        raise ValueError('Expected the completed 125k EMA comparison at NFE 1/5/20/100')
    path = args.docs / 'results.json'
    manifest = json.loads(path.read_text())
    sample = build_sample(args.source, args.docs, report)
    manifest['samples'] = [sample] + [s for s in manifest['samples'] if s['id'] != 'cplus']
    manifest['playback']['rms_matched']['method'] = 'Match whole-clip RMS to the damaged input, then apply one common safety gain to every track within each example.'
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(samples=len(manifest['samples']), cplus_checkpoint=sample['checkpoint'], nfes=sample['nfes'])))


if __name__ == '__main__':
    main()
