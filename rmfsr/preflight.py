import argparse
import json
from pathlib import Path
import shutil
import subprocess
from .train import get_ffmpeg
from .gsm import library_path
from .data import assert_disjoint
from .paths import load_manifest, manifest_fingerprint


def check(config):
    cfg = dict(config) if isinstance(config, dict) else json.loads(Path(config).read_text())
    problems, info = [], {}
    expected = {'train_manifest': {f'p{i:03d}' for i in range(1, 100)},
                'validation_manifest': {f'p{i:03d}' for i in range(100, 104)}}
    if not cfg['pilot']:
        expected['test_manifest'] = {f'p{i:03d}' for i in range(104, 108)}
    existing = []
    for key, exact_speakers in expected.items():
        path = Path(cfg.get(key, '__missing__'))
        if not path.exists():
            problems.append(f'Missing {key}: {path}')
            continue
        rows = load_manifest(path)
        speakers = {r['speaker'] for r in rows}
        info[key] = dict(speakers=len(speakers), files=len(rows),
                         hours=sum(r['seconds'] for r in rows) / 3600,
                         fingerprint=manifest_fingerprint(path))
        if not cfg['pilot'] and speakers != exact_speakers:
            problems.append(f'{key}: expected exact documented speaker split')
        if any(not Path(r['path']).is_file() for r in rows):
            problems.append(f'{key}: missing audio files')
        if any(r['sr'] != cfg['sample_rate'] for r in rows):
            problems.append(f'{key}: sample rate mismatch')
        existing.append(path)
    try:
        assert_disjoint(*existing)
    except ValueError as exc:
        problems.append(str(exc))
    if not cfg['pilot']:
        for key in ('noise_dir', 'validation_noise_dir'):
            noise = Path(cfg.get(key, '__missing__'))
            if not noise.is_dir() or not any(p.suffix.lower() in ('.wav', '.flac') for p in noise.rglob('*')):
                problems.append(f'DNS noise not prepared: {key}')
        completion = Path(cfg.get('noise_completion', '__missing__'))
        if not completion.is_file():
            problems.append('Complete DNS corpus provenance missing')
        else:
            from .corpus import SHARDS
            from .download import digest
            inventory = json.loads(completion.read_text())
            if inventory.get('shards') != SHARDS:
                problems.append('DNS archive inventory differs from the configured corpus')
            for shard in SHARDS:
                index = completion.parent / (shard + '.json')
                if not index.is_file() or digest(index) != inventory.get('manifest_sha256', {}).get(shard):
                    problems.append(f'DNS shard manifest missing or changed: {shard}')
                elif any(not Path(row['path']).is_file() for row in load_manifest(index)):
                    problems.append(f'DNS shard audio missing: {shard}')
        ltas = Path(cfg['ltas_path'])
        if not ltas.exists() or not Path(str(ltas) + '.json').exists():
            problems.append('DAPS produced-speech LTAS and provenance not prepared')
    try:
        info['gsm_library'] = library_path()
    except RuntimeError as exc:
        problems.append(str(exc))
    ffmpeg = get_ffmpeg(cfg.get('ffmpeg'))
    info['ffmpeg'] = ffmpeg
    if not ffmpeg:
        problems.append('ffmpeg missing')
    else:
        try:
            encoders = subprocess.check_output([ffmpeg, '-hide_banner', '-encoders'], stderr=subprocess.DEVNULL, text=True)
            if 'libmp3lame' not in encoders:
                problems.append('ffmpeg lacks MP3 encoder')
        except (OSError, subprocess.CalledProcessError) as exc:
            problems.append(f'ffmpeg unusable: {exc}')
    info.update(problems=problems, ready=not problems, pilot=cfg['pilot'],
                free_disk_gib=shutil.disk_usage('.').free / 1024**3)
    return info


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output')
    a = p.parse_args()
    result = check(a.config)
    if a.output:
        Path(a.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['ready'] else 2)
