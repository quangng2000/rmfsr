"""Download/convert public corpora on a data volume, without extracting archive paths."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tarfile
import urllib.request
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from .download import download, digest
from .prepare import prepare
from .ltas import make_ltas

DNS = 'https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset/noise_fullband/'
SHARDS = [f'datasets_fullband.noise_fullband.{source}_{i:03d}.tar.bz2'
          for source, count in [('audioset', 7), ('freesound', 2)] for i in range(count)]
DAPS = 'https://zenodo.org/records/4660670/files/daps.tar.gz'


def noise_split(member):
    bucket = int(hashlib.sha256(member.encode()).hexdigest()[:8], 16) % 100
    return 'train' if bucket < 90 else 'validation' if bucket < 95 else 'test'


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def convert_tar(archive, directory, source, kind, archive_digest):
    directory = Path(directory)
    rows = []
    seen = set()
    with tarfile.open(archive, 'r|*') as handle:
        for member in handle:
            if not member.isfile() or not member.name.lower().endswith('.wav'):
                continue
            if kind == 'daps' and 'produced' not in Path(member.name).parts:
                continue
            if shutil.disk_usage(directory).free < 8 * 1024**3:
                raise RuntimeError('Data conversion stopped: preserve 8 GiB of free space')
            # Never use extractall: a remote member name cannot write outside our root.
            key = hashlib.sha256(member.name.encode()).hexdigest()
            if key in seen:
                raise ValueError(f'Duplicate source member: {member.name}')
            seen.add(key)
            split = noise_split(member.name) if kind == 'dns' else 'produced'
            dest = directory / split / f'{key}.flac'
            dest.parent.mkdir(parents=True, exist_ok=True)
            raw = handle.extractfile(member).read()
            wave, sr = sf.read(io.BytesIO(raw), dtype='float32', always_2d=True)
            wave = wave.mean(1)
            common = math.gcd(sr, 16000)
            if sr != 16000:
                wave = resample_poly(wave, 16000 // common, sr // common)
            if not len(wave) or not np.isfinite(wave).all():
                raise ValueError(f'Invalid audio: {member.name}')
            sf.write(dest, wave, 16000, subtype='PCM_16')
            rows.append(dict(path=str(dest.relative_to(directory)), source=source,
                             source_member=member.name, archive_sha256=archive_digest,
                             audio_sha256=digest(dest), sr=16000, frames=len(wave),
                             seconds=len(wave) / 16000, split=split))
            if len(rows) % 500 == 0:
                print(kind, len(rows), 'files converted', flush=True)
    if not rows:
        raise ValueError(f'No matching {kind} files in {archive}; inspect source layout')
    return rows


def prepare_dns(root):
    directory = root / 'dns_noise'
    directory.mkdir(parents=True, exist_ok=True)
    for shard in SHARDS:
        index = directory / (shard + '.json')
        if index.exists():
            rows = json.loads(index.read_text())
            if rows and all((directory / row['path']).is_file() for row in rows):
                continue
        source = DNS + shard
        archive = download(source, root / 'downloads' / shard)
        rows = convert_tar(archive, directory, source, 'dns', digest(archive))
        atomic_json(index, rows)
        archive.unlink()
        print(shard, len(rows), 'complete', flush=True)
    atomic_json(directory / 'complete.json', dict(shards=SHARDS, sample_rate=16000,
                manifest_sha256={name: digest(directory / (name + '.json')) for name in SHARDS},
                source='https://github.com/microsoft/DNS-Challenge',
                note='Official DNS5 noise archives; no extra speech-content classifier applied.'))


def prepare_daps(root):
    directory = root / 'daps'
    directory.mkdir(parents=True, exist_ok=True)
    index = directory / 'manifest.json'
    rows = json.loads(index.read_text()) if index.exists() else []
    if not rows or any(not (directory / row['path']).is_file() for row in rows):
        archive = download(DAPS, root / 'downloads' / 'daps.tar.gz',
                           expected_size=16055782487,
                           checksum='md5:303c130b7ce2e02b59c7ca5cd595a89c')
        rows = convert_tar(archive, directory, DAPS, 'daps', digest(archive))
        atomic_json(index, rows)
        archive.unlink()
    make_ltas(directory / 'produced', root / 'daps_ltas.npy')


def inspect_sources():
    result = []
    for url in [DNS + shard for shard in SHARDS] + [DAPS]:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as response:
                result.append(dict(url=url, status=response.status, bytes=int(response.headers['Content-Length'])))
        except Exception as exc:
            result.append(dict(url=url, error=str(exc)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data')
    parser.add_argument('--corpus', choices=['ears', 'dns', 'daps', 'all', 'inspect'], required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.corpus == 'inspect':
        result = inspect_sources()
        atomic_json(root / 'source-access.json', result)
        print(json.dumps(result, indent=2))
        return
    if args.corpus in ('ears', 'all'):
        # Reproduction choices, not author-provided splits. Held-out speakers first.
        for split, speakers in [('validation', range(100, 104)), ('test', range(104, 108)), ('train', range(1, 100))]:
            prepare(root, speakers, split)
    if args.corpus in ('daps', 'all'):
        prepare_daps(root)
    if args.corpus in ('dns', 'all'):
        prepare_dns(root)


if __name__ == '__main__':
    main()
