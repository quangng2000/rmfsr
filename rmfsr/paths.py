"""Portable manifests and verified, location-independent dataset identities.

Hash files once at process startup. A FileVerifier may be shared within that scan;
checkpoint writers should retain the resulting fingerprints instead of rescanning.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

_AUDIO_SUFFIXES = {'.wav', '.flac'}
_METADATA_KEYS = ('speaker', 'split', 'source', 'source_member', 'archive_sha256')
INTEGRITY_VERSION = 'verified-content-v1'


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value.lower())


class FileVerifier:
    """Per-scan cache; stat changes invalidate it, and no disk cache is trusted."""
    def __init__(self):
        self._digests = {}
        self._audio_info = {}

    @staticmethod
    def _identity(path):
        path = Path(path).resolve()
        stat = path.stat()
        if not path.is_file():
            raise ValueError(f'Not a regular file: {path}')
        return (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def digest(self, path):
        path = Path(path).resolve()
        before = self._identity(path)
        if before not in self._digests:
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            if self._identity(path) != before:
                raise ValueError(f'File changed during integrity verification: {path}')
            self._digests[before] = digest.hexdigest()
        return self._digests[before]

    def audio(self, row, *, sample_rate=None, require_hashes=False):
        path = Path(row['path']).resolve()
        identity = self._identity(path)
        if identity not in self._audio_info:
            try:
                header = sf.info(path)
            except (RuntimeError, OSError) as exc:
                raise ValueError(f'Unreadable audio header: {path}: {exc}') from exc
            if header.frames <= 0 or header.samplerate <= 0 or header.channels != 1:
                raise ValueError(f'Expected nonempty mono audio: {path}')
            self._audio_info[identity] = (header.samplerate, header.frames)
        rate, frames = self._audio_info[identity]
        if sample_rate is not None and rate != sample_rate:
            raise ValueError(f'Audio sample rate mismatch: {path}: {rate} != {sample_rate}')
        if 'sr' in row and row['sr'] != rate:
            raise ValueError(f'Manifest sample rate mismatch: {path}')
        if 'frames' in row and row['frames'] != frames:
            raise ValueError(f'Manifest frame count mismatch: {path}')
        if 'seconds' in row and (not np.isfinite(row['seconds']) or abs(row['seconds'] - frames / rate) > 1 / rate):
            raise ValueError(f'Manifest duration mismatch: {path}')
        recorded = row.get('audio_sha256')
        if require_hashes and recorded is None:
            raise ValueError(f'Missing recorded audio_sha256: {path}')
        if recorded is not None and not _valid_sha256(recorded):
            raise ValueError(f'Invalid recorded audio_sha256: {path}')
        actual = self.digest(path)
        if recorded is not None and actual != recorded.lower():
            raise ValueError(f'Audio SHA-256 mismatch: {path}')
        if self._identity(path) != identity:
            raise ValueError(f'File changed during integrity verification: {path}')
        return {'audio_sha256': actual, 'sr': rate, 'frames': frames}


def load_manifest(path):
    path = Path(path).resolve()
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or any(not isinstance(row, dict) or 'path' not in row for row in rows):
        raise ValueError(f'Invalid manifest: {path}')
    for row in rows:
        audio = Path(row['path'])
        row['path'] = str((audio if audio.is_absolute() else path.parent / audio).resolve())
    return rows


def verify_manifest(path, *, require_hashes=False, sample_rate=None, verifier=None):
    """Verify stored hashes/header metadata; legacy rows use actual byte hashes."""
    verifier = verifier or FileVerifier()
    rows = load_manifest(path)
    if not rows:
        raise ValueError(f'Empty manifest: {path}')
    payload, content_hashes, duration = [], [], 0.0
    for row in rows:
        actual = verifier.audio(row, sample_rate=sample_rate, require_hashes=require_hashes)
        payload.append({**{key: row[key] for key in _METADATA_KEYS if key in row}, **actual})
        content_hashes.append(actual['audio_sha256'])
        duration += actual['frames'] / actual['sr']
    return {'fingerprint': _hash_json({'version': INTEGRITY_VERSION, 'records': payload}),
            'files': len(rows), 'hours': duration / 3600,
            'speakers': len({row['speaker'] for row in rows if 'speaker' in row}),
            'content_hashes': content_hashes, 'rows': rows,
            'computed_legacy_hashes': sum(row.get('audio_sha256') is None for row in rows)}


def manifest_fingerprint(path, *, require_hashes=False, sample_rate=None, verifier=None):
    # Mount location is excluded; actual audio identity, metadata and order matter.
    return verify_manifest(path, require_hashes=require_hashes, sample_rate=sample_rate,
                           verifier=verifier)['fingerprint']


def noise_partition_fingerprints(train_dir, validation_dir, *, sample_rate=16000,
                                 verifier=None, provenance_rows=None, require_hashes=False):
    """Fingerprint exactly the selected noise files; reject cross-split content."""
    verifier = verifier or FileVerifier()
    provenance = {str(Path(row['path']).resolve()): row for row in provenance_rows or []}
    if len(provenance) != len(provenance_rows or []):
        raise ValueError('Duplicate noise provenance paths')
    result, contents = {}, []
    for label, directory in [('train', train_dir), ('validation', validation_dir)]:
        root = Path(directory).resolve()
        paths = sorted(path for path in root.rglob('*') if path.suffix.lower() in _AUDIO_SUFFIXES)
        if not root.is_dir() or not paths:
            raise ValueError(f'DNS noise not prepared: {label}: {directory}')
        payload, hashes = [], set()
        for path in paths:
            recorded = provenance.get(str(path.resolve()))
            if provenance_rows is not None and recorded is None:
                raise ValueError(f'Selected noise missing from verified provenance: {path}')
            if recorded is not None and recorded.get('split') != label:
                raise ValueError(f'Selected noise partition mismatch: {path}: expected {label}')
            actual = verifier.audio(recorded or {'path': path}, sample_rate=sample_rate,
                                    require_hashes=require_hashes)
            payload.append({'path': path.relative_to(root).as_posix(), **actual})
            hashes.add(actual['audio_sha256'])
        if any(hashes & other for other in contents):
            raise ValueError('Noise content leakage across train/validation partitions')
        contents.append(hashes)
        result['noise_' + label] = _hash_json({'version': INTEGRITY_VERSION, 'partition': label,
                                              'records': payload})
    return result


def ltas_fingerprint(path, *, sample_rate=16000, verifier=None, require_provenance=False):
    """Check LTAS shape/values, provenance metadata and any recorded output hash."""
    verifier = verifier or FileVerifier()
    path = Path(path)
    try:
        power = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f'Unreadable DAPS LTAS: {path}: {exc}') from exc
    if not isinstance(power, np.ndarray) or power.shape != (161,) or not np.issubdtype(power.dtype, np.number) or np.iscomplexobj(power):
        raise ValueError('DAPS LTAS must contain 161 real power-spectrum bins')
    if not np.isfinite(power).all() or np.any(power <= 0):
        raise ValueError('DAPS LTAS must contain finite, positive power')
    actual = verifier.digest(path)
    metadata_path = Path(str(path) + '.json')
    metadata_hash = None
    if metadata_path.is_file():
        provenance = json.loads(metadata_path.read_text())
        if not isinstance(provenance, dict) or provenance.get('sample_rate') != sample_rate:
            raise ValueError('DAPS LTAS provenance sample rate mismatch')
        files = provenance.get('files')
        if (not isinstance(files, list) or not files or
                any(not isinstance(row, dict) or not row.get('path') or not _valid_sha256(row.get('sha256')) for row in files)):
            raise ValueError('DAPS LTAS provenance must identify source files and SHA-256 hashes')
        if not provenance.get('source'):
            raise ValueError('DAPS LTAS provenance source missing')
        if ('normalization_dbfs' in provenance and
                (not np.isfinite(provenance['normalization_dbfs']) or provenance['normalization_dbfs'] != -25)):
            raise ValueError('DAPS LTAS provenance normalization must be -25 dBFS')
        for key in ('ltas_sha256', 'sha256'):
            if key in provenance and (not _valid_sha256(provenance[key]) or provenance[key].lower() != actual):
                raise ValueError('DAPS LTAS SHA-256 mismatch')
        metadata_hash = verifier.digest(metadata_path)
    elif require_provenance:
        raise ValueError('DAPS produced-speech LTAS provenance missing')
    return _hash_json({'version': INTEGRITY_VERSION, 'ltas_sha256': actual,
                       'provenance_sha256': metadata_hash})
