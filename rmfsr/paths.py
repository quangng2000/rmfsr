"""Portable manifests: relative audio paths are relative to the manifest file."""
import hashlib
import json
from pathlib import Path


def load_manifest(path):
    path = Path(path).resolve()
    rows = json.loads(path.read_text())
    for row in rows:
        audio = Path(row['path'])
        row['path'] = str(audio if audio.is_absolute() else path.parent / audio)
    return rows


def manifest_fingerprint(path):
    # Independent of mount path, sensitive to audio identity and split.
    rows = json.loads(Path(path).read_text())
    keys = ('speaker', 'split', 'sr', 'frames', 'source', 'source_member',
            'archive_sha256', 'audio_sha256')
    payload = [{k: r[k] for k in keys if k in r} for r in rows]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
