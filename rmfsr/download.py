"""Resumable public-data downloads with size checks and atomic publication."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request


def digest(path, algorithm='sha256'):
    value = hashlib.new(algorithm)
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def download(url, path, expected_size=None, checksum=None, reserve_gib=8):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.download')
    if expected_size is None:
        request = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(request, timeout=60) as response:
            expected_size = int(response.headers['Content-Length'])
    if not path.exists():
        remaining = expected_size - (partial.stat().st_size if partial.exists() else 0)
        if shutil.disk_usage(path.parent).free < max(0, remaining) + reserve_gib * 1024**3:
            raise RuntimeError(f'Insufficient space for {path.name}; leave {reserve_gib} GiB free')
        if remaining < 0:
            raise ValueError(f'Partial archive is larger than the source: {partial}')
        if remaining:
            subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                            '--retry', '5', '--retry-delay', '3', '--connect-timeout', '30',
                            '--speed-time', '120', '--speed-limit', '1024',
                            '--continue-at', '-', '--output', str(partial), url], check=True)
        if partial.stat().st_size != expected_size:
            raise ValueError(f'Incomplete archive: {partial}')
        if checksum:
            algorithm, expected = checksum.split(':', 1)
            if digest(partial, algorithm) != expected:
                raise ValueError(f'Archive checksum mismatch: {partial}')
        os.replace(partial, path)
    if path.stat().st_size != expected_size:
        raise ValueError(f'Cached archive size mismatch: {path}')
    if checksum:
        algorithm, expected = checksum.split(':', 1)
        if digest(path, algorithm) != expected:
            raise ValueError(f'Cached archive checksum mismatch: {path}')
    return path
