"""Explicit, checkpointed Fourier-width changes at a training process boundary.

The request changes only the two fixed buffers. The saved checkpoint, recipe,
optimizer, random generators and flow clock are never modified here.
"""
import copy
import hashlib
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import tempfile

import torch

from .flow_curriculum import config_digest


FREQUENCY_KEY = 'embedding.frequencies'
MARKER_KEY = 'frequency_width_transition'
REQUEST_FIELDS = {
    'version', 'transition_id', 'from_width', 'to_width',
    'expected_config_sha256', 'minimum_checkpoint_step',
    'source_frequency_sha256', 'target_frequency_sha256',
}


def tensor_sha256(tensor):
    """Hash exact contiguous tensor bytes without advancing any RNG."""
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _validate_request(request):
    if not isinstance(request, dict) or set(request) != REQUEST_FIELDS:
        raise ValueError('Invalid frequency transition request fields')
    if (type(request['version']) is not int or request['version'] != 1 or
            type(request['from_width']) is not int or request['from_width'] != 1 or
            type(request['to_width']) is not int or request['to_width'] != 4):
        raise ValueError('Only the explicit frequency width 1 to 4 transition is supported')
    if (not isinstance(request['transition_id'], str) or
            re.fullmatch('[a-z0-9][a-z0-9-]{0,79}', request['transition_id']) is None):
        raise ValueError('Invalid frequency transition id')
    if not _is_sha256(request['expected_config_sha256']):
        raise ValueError('Invalid frequency transition configuration digest')
    step = request['minimum_checkpoint_step']
    if type(step) is not int or step < 0:
        raise ValueError('Invalid frequency transition minimum checkpoint step')
    for field in ('source_frequency_sha256', 'target_frequency_sha256'):
        hashes = request[field]
        if (not isinstance(hashes, dict) or set(hashes) != {'model', 'ema'} or
                not all(_is_sha256(value) for value in hashes.values())):
            raise ValueError(f'Invalid frequency transition {field}')


def _checkpoint_step(checkpoint):
    step = checkpoint.get('step')
    if type(step) is not int or step < 0:
        raise ValueError('Invalid frequency transition checkpoint step')
    return step


def _frequency_buffers(model, ema, checkpoint, expected):
    buffers = {}
    for kind, module in (('model', model), ('ema', ema)):
        buffer = dict(module.named_buffers()).get(FREQUENCY_KEY)
        saved = checkpoint.get(kind, {}).get(FREQUENCY_KEY)
        if (not isinstance(buffer, torch.Tensor) or not isinstance(saved, torch.Tensor) or
                buffer.shape != (32,) or saved.shape != (32,) or
                not buffer.is_floating_point() or buffer.dtype != saved.dtype or
                not bool(torch.isfinite(buffer).all()) or not bool(torch.isfinite(saved).all())):
            raise ValueError(f'Invalid {kind} Fourier frequency buffer')
        if (tensor_sha256(saved) != expected[kind] or
                not torch.equal(buffer.detach().cpu(), saved.detach().cpu())):
            raise ValueError(f'{kind} Fourier frequency buffer differs from transition')
        buffers[kind] = buffer
    if (buffers['model'].dtype != buffers['ema'].dtype or
            not torch.equal(buffers['model'].detach().cpu(), buffers['ema'].detach().cpu())):
        raise ValueError('Raw and EMA Fourier frequency buffers differ')
    return buffers


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def publish_best_checkpoint(latest, best):
    """Publish a complete checkpoint atomically, retaining the old best on error."""
    latest, best = Path(latest), Path(best)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=best.parent, prefix=best.name + '.',
                                         suffix='.partial', delete=False) as handle:
            temporary = Path(handle.name)
            with latest.open('rb') as source:
                shutil.copyfileobj(source, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, best)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _has_target_frequencies(checkpoint, request):
    if not isinstance(checkpoint, dict):
        return False
    for kind in ('model', 'ema'):
        state = checkpoint.get(kind)
        frequencies = state.get(FREQUENCY_KEY) if isinstance(state, dict) else None
        if (not isinstance(frequencies, torch.Tensor) or frequencies.shape != (32,) or
                tensor_sha256(frequencies) != request['target_frequency_sha256'][kind]):
            return False
    return True


def _finite_score(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _is_selected_checkpoint(checkpoint, metric, score):
    history = checkpoint.get('history')
    return (isinstance(history, list) and bool(history) and isinstance(history[-1], dict) and
            history[-1].get('step') == checkpoint.get('step') and
            _finite_score(history[-1].get(metric)) and
            history[-1].get(metric) == score)


def _reconcile_best_checkpoint(checkpoint, marker, run, resume_path):
    """Repair the interrupted latest-to-best copy, never guess a lost best."""
    if marker['best_selection_pending']:
        return
    metric = checkpoint.get('validation_selection_metric')
    score = checkpoint.get('best_validation')
    if not isinstance(metric, str) or not _finite_score(score):
        raise ValueError('Checkpointed frequency transition has invalid best validation metadata')
    best = Path(run) / 'best-validation.pt'
    current = None
    if best.exists():
        try:
            current = torch.load(best, map_location='cpu', weights_only=False, mmap=True)
        except (OSError, RuntimeError, ValueError, EOFError, KeyError, IndexError, pickle.UnpicklingError):
            # A partial old copy is recoverable only from a fully selected latest.
            current = None
    if _has_target_frequencies(current, marker['request']):
        current_step = current.get('step')
        current_score = current.get('best_validation')
        if ((type(current_step) is int and current_step > checkpoint['step']) or
                (_finite_score(current_score) and current_score < score)):
            raise ValueError('Existing width-4 best contains newer or better history; refusing recovery')
        current_marker = current.get(MARKER_KEY)
        if (isinstance(current_marker, dict) and
                current_marker.get('request') == marker['request'] and
                current_marker.get('after_step') == marker['after_step'] and
                current_marker.get('nominal_frequency_width') == 4 and
                current_marker.get('best_selection_pending') is False and
                type(current_step) is int and marker['after_step'] <= current_step <= checkpoint['step'] and
                isinstance(current.get('config'), dict) and
                config_digest(current['config']) == marker['request']['expected_config_sha256'] and
                current.get('validation_selection_metric') == metric and current_score == score and
                _is_selected_checkpoint(current, metric, score)):
            return
    if not _is_selected_checkpoint(checkpoint, metric, score):
        raise ValueError('Cannot recover width-4 best: resume checkpoint is not its selected validation step')
    resume_path = Path(resume_path)
    if not resume_path.is_file():
        raise ValueError('Cannot recover width-4 best: resume checkpoint file is missing')
    source = torch.load(resume_path, map_location='cpu', weights_only=False, mmap=True)
    if (not _has_target_frequencies(source, marker['request']) or
            source.get(MARKER_KEY) != marker or source.get('step') != checkpoint['step'] or
            source.get('validation_selection_metric') != metric or source.get('best_validation') != score or
            not _is_selected_checkpoint(source, metric, score)):
        raise ValueError('Cannot recover width-4 best: resume checkpoint file differs from loaded state')
    if best.exists():
        digest = _file_sha256(best)
        archive = Path(run) / f'best-validation-before-frequency-recovery-{digest}.pt'
        if archive.exists() and _file_sha256(archive) != digest:
            raise ValueError('Frequency recovery archive differs from existing best')
        _write_archives([('previous_best_checkpoint', best, archive, digest)])
    publish_best_checkpoint(resume_path, best)


def _archive_plan(run, resume_path, transition_id, step):
    """Validate every existing archive before creating any new archive."""
    sources = [('source_checkpoint', Path(resume_path))]
    best = run / 'best-validation.pt'
    if best.exists():
        sources.append(('previous_best_checkpoint', best))
    plans = []
    for label, source in sources:
        if not source.is_file():
            raise ValueError(f'Frequency transition checkpoint is missing: {source}')
        archive = run / f'{label}-before-{transition_id}-step-{step:08d}.pt'
        digest = _file_sha256(source)
        if archive.exists() and (not archive.is_file() or _file_sha256(archive) != digest):
            raise ValueError(f'Frequency transition archive differs: {archive.name}')
        plans.append((label, source, archive, digest))
    return plans


def _write_archives(plans):
    archives = {}
    for label, source, archive, digest in plans:
        temporary = None
        try:
            if not archive.exists():
                with tempfile.NamedTemporaryFile(dir=archive.parent, prefix=archive.name + '.',
                                                 suffix='.partial', delete=False) as handle:
                    temporary = Path(handle.name)
                    with source.open('rb') as original:
                        shutil.copyfileobj(original, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                if _file_sha256(temporary) != digest:
                    raise ValueError('Frequency transition source changed while archiving')
                # Another activation must never silently replace its archive.
                try:
                    os.link(temporary, archive)
                except FileExistsError:
                    if _file_sha256(archive) != digest:
                        raise ValueError(f'Frequency transition archive differs: {archive.name}')
            archives[label] = {'path': archive.name, 'sha256': digest}
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return archives


def _restore_marker(model, ema, checkpoint, request, marker, run, resume_path):
    if (not isinstance(marker, dict) or type(marker.get('version')) is not int or
            marker['version'] != 1 or type(marker.get('nominal_frequency_width')) is not int or
            marker['nominal_frequency_width'] != 4 or
            marker.get('best_validation_reset') is not True or
            type(marker.get('best_selection_pending')) is not bool):
        raise ValueError('Invalid checkpointed frequency transition')
    saved_request = marker.get('request')
    _validate_request(saved_request)
    if request is not None and request != saved_request:
        raise ValueError('Frequency transition request differs from checkpoint')
    after_step = marker.get('after_step')
    if (type(after_step) is not int or
            not saved_request['minimum_checkpoint_step'] <= after_step <= _checkpoint_step(checkpoint)):
        raise ValueError('Checkpoint precedes its frequency transition')
    if config_digest(checkpoint['config']) != saved_request['expected_config_sha256']:
        raise ValueError('Checkpoint configuration differs from frequency transition')
    _frequency_buffers(model, ema, checkpoint, saved_request['target_frequency_sha256'])
    _reconcile_best_checkpoint(checkpoint, marker, run, resume_path)
    restored = copy.deepcopy(marker)
    model.frequency_width_transition = restored
    ema.frequency_width_transition = copy.deepcopy(restored)
    return restored, False


def apply_frequency_transition(model, ema, checkpoint, request, run, resume_path):
    """Return ``(persistent_marker, applied_now)`` after validating a resume.

    Call only after normal resume validation and state restoration. A marker in
    the checkpoint is authoritative even if its request file was removed. An
    unmarked checkpoint must match the source bytes; a marked one must match the
    target bytes. Replaying a crash therefore never multiplies twice.

    The caller resets its best score only when ``applied_now`` is true, persists
    ``model.frequency_width_transition`` in every new checkpoint, and clears its
    ``best_selection_pending`` flag when it selects a new width-4 best. The old
    best file remains available and is also archived; it must not be presented as
    a width-4 best while that flag is true. Once the flag is false, resumed runs
    verify the best file and repair an interrupted publication only from the
    checkpoint that actually achieved its saved validation score.
    """
    marker = checkpoint.get(MARKER_KEY)
    if marker is not None:
        return _restore_marker(model, ema, checkpoint, request, marker, run, resume_path)
    if request is None:
        return None, False
    _validate_request(request)
    step = _checkpoint_step(checkpoint)
    if step < request['minimum_checkpoint_step']:
        raise ValueError('Checkpoint precedes frequency transition minimum step')
    if config_digest(checkpoint['config']) != request['expected_config_sha256']:
        raise ValueError('Checkpoint configuration differs from frequency transition')
    buffers = _frequency_buffers(model, ema, checkpoint, request['source_frequency_sha256'])
    targets = {kind: buffer.detach().clone() * 4 for kind, buffer in buffers.items()}
    for kind, target in targets.items():
        if (not bool(torch.isfinite(target).all()) or
                tensor_sha256(target) != request['target_frequency_sha256'][kind]):
            raise ValueError(f'{kind} target Fourier frequency buffer differs from request')
    previous_best = checkpoint.get('best_validation', float('inf'))
    if (isinstance(previous_best, bool) or not isinstance(previous_best, (int, float)) or
            math.isnan(previous_best)):
        raise ValueError('Invalid previous frequency transition best validation score')
    run = Path(run)
    if not run.is_dir():
        raise ValueError('Frequency transition run directory is missing')
    plans = _archive_plan(run, resume_path, request['transition_id'], step)
    archives = _write_archives(plans)
    marker = dict(version=1, request=copy.deepcopy(request), after_step=step,
                  nominal_frequency_width=4, best_validation_reset=True,
                  best_selection_pending=True,
                  previous_best_validation=previous_best if math.isfinite(previous_best) else None,
                  archives=archives)
    # Replacing a registered buffer also protects callers whose state_dict
    # tensors still share storage with the module. Optimizer parameters retain
    # their identity; neither source tensor nor checkpoint is modified in place.
    for kind, module in (('model', model), ('ema', ema)):
        module.get_submodule('embedding').frequencies = targets[kind]
    model.frequency_width_transition = marker
    ema.frequency_width_transition = copy.deepcopy(marker)
    return marker, True
