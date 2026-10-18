"""Atomic non-overwriting publication of a conditioning conversion."""
import hashlib
import os
from pathlib import Path
import tempfile

import torch

from .conditioning_migration_checks import MARKER, require


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def publish_conversion(source, destination, request):
    from .conditioning_migration import convert_payload
    source, destination = Path(source), Path(destination)
    require(source.resolve() != destination.resolve(), 'Source and destination must differ')
    require(source.is_file() and not destination.exists(), 'Source missing or destination already exists')
    require(file_sha256(source) == request.get('expected_source_sha256'), 'Source checkpoint digest mismatch')
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    converted = convert_payload(checkpoint, request, source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return save_conversion(converted, source, destination, request)


def save_conversion(checkpoint, source, destination, request):
    from .conditioning_migration import verify_payload
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + '.',
                                         suffix='.partial', delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        restored = torch.load(temporary, map_location='cpu', weights_only=False)
        verify_payload(restored)
        require(file_sha256(source) == request['expected_source_sha256'], 'Source changed during conversion')
        os.link(temporary, destination)
        return dict(restored[MARKER], output_checkpoint=dict(path=str(destination), sha256=file_sha256(destination)))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
