import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from .train import get_ffmpeg
from .gsm import library_path
from .data import assert_disjoint
from .paths import (FileVerifier, INTEGRITY_VERSION, verify_manifest,
                    noise_partition_fingerprints, ltas_fingerprint, _hash_json)


def startup_check(config):
    """Explicit manifest mode records declarations, never claims audio verification."""
    cfg = dict(config) if isinstance(config, dict) else json.loads(Path(config).read_text())
    mode = cfg.get('input_verification', 'full')
    if mode == 'full':
        return check(cfg)
    if mode != 'manifest':
        raise ValueError(f'Unknown input_verification: {mode}')
    print('Starting with saved manifest metadata; full corpus verification skipped.', flush=True)
    fingerprints = {'integrity_version': 'manifest-declared-v1', 'full_audio_verified': False}
    problems = []
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    try:
        from .model import model_options
        model_options(cfg)
        for key in ('train_manifest', 'validation_manifest', 'test_manifest'):
            if cfg.get(key):
                fingerprints[key] = digest(cfg[key])
        assert_disjoint(cfg['train_manifest'], cfg['validation_manifest'])
        if cfg.get('noise_completion'):
            completion = Path(cfg['noise_completion'])
            inventory = json.loads(completion.read_text())
            shards = {shard: digest(completion.parent / (shard + '.json')) for shard in inventory['shards']}
            fingerprints['noise_inventory'] = _hash_json({'completion': digest(completion), 'shards': shards})
        elif not cfg.get('pilot'):
            raise ValueError('Saved DNS index is required for manifest startup')
        for key in ('noise_dir', 'validation_noise_dir'):
            fingerprints[key] = cfg.get(key)
        if cfg.get('ltas_path'):
            fingerprints['ltas'] = digest(cfg['ltas_path'])
        elif not cfg.get('pilot'):
            raise ValueError('DAPS LTAS is required')
        library_path()
        if not get_ffmpeg(cfg.get('ffmpeg')):
            raise ValueError('ffmpeg missing')
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        problems.append(str(exc))
    return {'ready': not problems, 'problems': problems, 'input_fingerprints': fingerprints,
            'verification_mode': 'manifest', 'full_corpus_verified': False}


def check(config):
    cfg = dict(config) if isinstance(config, dict) else json.loads(Path(config).read_text())
    problems, info = [], {}
    from .model import model_options
    try:
        model_type, _ = model_options(cfg)
        info['model_type'] = model_type
    except (ValueError, TypeError) as exc:
        problems.append(f'Invalid model configuration: {exc}')
    pilot = cfg.get('pilot', False)
    def progress(label, completed, total):
        print(f'Verifying {label}: {completed}/{total}', flush=True)
    verifier = FileVerifier(workers=cfg.get('verification_workers', 8), progress=progress)
    fingerprints = {'integrity_version': INTEGRITY_VERSION}
    expected = {'train_manifest': {f'p{i:03d}' for i in range(1, 100)},
                'validation_manifest': {f'p{i:03d}' for i in range(100, 104)}}
    if not pilot:
        expected['test_manifest'] = {f'p{i:03d}' for i in range(104, 108)}
    existing, speech_content = [], []
    for key, exact_speakers in expected.items():
        path = Path(cfg.get(key) or '__missing__')
        if not path.is_file():
            problems.append(f'Missing {key}: {path}')
            continue
        try:
            verified = verify_manifest(path, require_hashes=not pilot,
                                       sample_rate=cfg['sample_rate'], verifier=verifier)
            speakers = {row['speaker'] for row in verified['rows']}
            info[key] = {field: verified[field] for field in
                         ('speakers', 'files', 'hours', 'fingerprint', 'computed_legacy_hashes')}
            fingerprints[key] = verified['fingerprint']
            if not pilot and speakers != exact_speakers:
                problems.append(f'{key}: expected exact documented speaker split')
            if not pilot and any(set(verified['content_hashes']) & other for other in speech_content):
                problems.append('Speech content leakage across data splits')
            speech_content.append(set(verified['content_hashes']))
            existing.append(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(f'{key}: {exc}')
    try:
        assert_disjoint(*existing)
    except ValueError as exc:
        problems.append(str(exc))

    provenance_rows = None
    if not pilot:
        completion = Path(cfg.get('noise_completion') or '__missing__')
        if not completion.is_file():
            problems.append('Complete DNS corpus provenance missing')
        else:
            from .corpus import SHARDS
            try:
                inventory = json.loads(completion.read_text())
                if not isinstance(inventory, dict) or not isinstance(inventory.get('manifest_sha256'), dict):
                    raise ValueError('DNS inventory must include a manifest_sha256 mapping')
                if inventory.get('shards') != SHARDS:
                    problems.append('DNS archive inventory differs from the configured corpus')
                if inventory.get('sample_rate') != cfg['sample_rate']:
                    problems.append('DNS archive inventory sample rate mismatch')
                shard_fingerprints, provenance_rows = {}, []
                for shard in SHARDS:
                    index = completion.parent / (shard + '.json')
                    try:
                        if verifier.digest(index) != inventory.get('manifest_sha256', {}).get(shard):
                            raise ValueError('manifest SHA-256 mismatch')
                        verified = verify_manifest(index, require_hashes=True,
                                                   sample_rate=cfg['sample_rate'], verifier=verifier)
                        shard_fingerprints[shard] = verified['fingerprint']
                        provenance_rows.extend(verified['rows'])
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        problems.append(f'DNS shard manifest/audio invalid: {shard}: {exc}')
                fingerprints['noise_inventory'] = _hash_json({
                    'completion_sha256': verifier.digest(completion), 'shards': shard_fingerprints})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                problems.append(f'DNS corpus provenance invalid: {exc}')

    if cfg.get('noise_dir') or not pilot:
        try:
            fingerprints.update(noise_partition_fingerprints(
                cfg.get('noise_dir', '__missing__'),
                cfg.get('validation_noise_dir') or cfg.get('noise_dir', '__missing__'),
                sample_rate=cfg['sample_rate'], verifier=verifier,
                provenance_rows=provenance_rows, require_hashes=not pilot))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(str(exc))
    if cfg.get('ltas_path') or not pilot:
        try:
            fingerprints['ltas'] = ltas_fingerprint(
                cfg.get('ltas_path', '__missing__'), sample_rate=cfg['sample_rate'],
                verifier=verifier, require_provenance=not pilot)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(str(exc))
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
    info.update(problems=problems, ready=not problems, pilot=pilot,
                input_fingerprints=fingerprints,
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
