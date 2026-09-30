"""Sequential EARS conversion; one source archive on disk at a time."""
import argparse
import io
import json
import math
import os
from pathlib import Path
import zipfile
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from .download import download, digest

BASE = 'https://github.com/facebookresearch/ears_dataset/releases/download/dataset/'


def prepare(root, speakers, split, sample_rate=16000):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for speaker in speakers:
        name = f'p{speaker:03d}'
        dest = root / 'ears' / name
        dest.mkdir(parents=True, exist_ok=True)
        index = dest / 'manifest.json'
        rows = None
        if index.exists():
            existing = json.loads(index.read_text())
            if any(row['sr'] != sample_rate for row in existing):
                raise ValueError('Existing sample rate differs')
            if existing and all((dest / Path(row['path']).name).is_file() for row in existing):
                rows = existing
        if rows is None:
            url = BASE + f'{name}.zip'
            archive = download(url, root / 'downloads' / f'{name}.zip')
            archive_digest = digest(archive)
            rows = []
            with zipfile.ZipFile(archive) as z:
                if z.testzip() is not None:
                    raise ValueError('Archive CRC failed')
                names = set()
                for member in sorted(z.namelist()):
                    if not member.lower().endswith('.wav'):
                        continue
                    filename = Path(member).stem + '.flac'
                    if filename in names:
                        raise ValueError(f'Duplicate EARS basename: {member}')
                    names.add(filename)
                    wave, sr = sf.read(io.BytesIO(z.read(member)), dtype='float32', always_2d=True)
                    wave = wave.mean(1)
                    gcd = math.gcd(sr, sample_rate)
                    wave = resample_poly(wave, sample_rate // gcd, sr // gcd)
                    if not len(wave) or not np.isfinite(wave).all():
                        raise ValueError(f'Invalid audio: {member}')
                    path = dest / filename
                    sf.write(path, wave, sample_rate, subtype='PCM_16')
                    rows.append(dict(path=filename, speaker=name, split=split, sr=sample_rate,
                                     frames=len(wave), seconds=len(wave) / sample_rate, source=url,
                                     source_member=member, archive_sha256=archive_digest,
                                     audio_sha256=digest(path), license='CC-BY-NC-4.0'))
            archive.unlink()
        for row in rows:
            row['path'] = Path(row['path']).name
            row['split'] = split
            if 'audio_sha256' not in row:
                row['audio_sha256'] = digest(dest / row['path'])
        temporary = index.with_suffix('.partial')
        temporary.write_text(json.dumps(rows, indent=2))
        os.replace(temporary, index)
        manifest.extend({**row, 'path': str((dest / row['path']).relative_to(root))} for row in rows)
        print(name, len(rows), 'files', round(sum(r['seconds'] for r in rows) / 3600, 3), 'hours', flush=True)
    out = root / f'{split}.json'
    temporary = out.with_suffix('.partial')
    temporary.write_text(json.dumps(manifest, indent=2))
    os.replace(temporary, out)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data')
    p.add_argument('--speakers', type=int, nargs='+', required=True)
    p.add_argument('--split', required=True)
    p.add_argument('--sample-rate', type=int, default=16000)
    a = p.parse_args()
    print(prepare(a.root, a.speakers, a.split, a.sample_rate))
