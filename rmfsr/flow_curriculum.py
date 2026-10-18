"""Checkpointed, session-boundary changes to the flow clock only."""
import hashlib
import json
import math


def config_digest(config):
    base = {k: v for k, v in config.items() if k != 'flow_schedule_transition'}
    return hashlib.sha256(json.dumps(base, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def validate_transition(transition):
    if transition.get('version') != 1:
        raise ValueError('Unsupported flow schedule transition')
    for key in ('start_step', 'ramp_updates'):
        value = transition.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < (key == 'ramp_updates'):
            raise ValueError(f'Invalid flow schedule {key}')
    start, end = transition.get('start_progress'), transition.get('target_progress')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in (start, end)) or not 0 <= start < end <= 1:
        raise ValueError('Invalid flow schedule progress interval')
    restoration = transition.get('restoration')
    if restoration is not None:
        validate_restore_request(restoration.get('request'))
        step = restoration.get('after_step')
        if isinstance(step, bool) or not isinstance(step, int) or step < transition['start_step']:
            raise ValueError('Invalid flow schedule restoration step')


def validate_restore_request(request):
    if (not isinstance(request, dict) or
            set(request) != {'version', 'mode', 'expected_config_sha256'} or
            request['version'] != 1 or request['mode'] != 'original-gradual-clock' or
            not isinstance(request['expected_config_sha256'], str) or
            len(request['expected_config_sha256']) != 64 or
            any(c not in '0123456789abcdef' for c in request['expected_config_sha256'])):
        raise ValueError('Invalid flow schedule restoration request')


def training_flow_progress(step, config):
    transition = config.get('flow_schedule_transition')
    if transition is None:
        return step / config.get('schedule_steps', config['steps'])
    validate_transition(transition)
    if step < transition['start_step']:
        raise ValueError('Flow schedule transition predates its checkpoint')
    if transition.get('restoration') is not None:
        if step < transition['restoration']['after_step']:
            raise ValueError('Checkpoint precedes flow schedule restoration')
        return step / config.get('schedule_steps', config['steps'])
    fraction = min((step - transition['start_step']) / transition['ramp_updates'], 1.)
    blend = .5 * (1 - math.cos(math.pi * fraction))
    return transition['start_progress'] + blend * (
        transition['target_progress'] - transition['start_progress'])


def resolve_transition(config, previous, step, request=None, *, allowed_config_changes=(),
                       restore_request=None):
    """Return immutable policy; a resumed checkpoint always owns its clock."""
    saved = previous.get('flow_schedule_transition')
    supplied = config.get('flow_schedule_transition')
    if supplied is not None and supplied != saved:
        raise ValueError('Cannot replace a checkpointed flow schedule transition')
    if restore_request is not None:
        validate_restore_request(restore_request)
        if saved is None:
            raise ValueError('Flow schedule restoration requires an accelerated checkpoint')
    if saved is not None:
        validate_transition(saved)
        if request is not None and request != saved['request']:
            raise ValueError('Flow schedule request differs from checkpoint')
        bound_digest = saved.get('resume_config_sha256', saved['request']['expected_config_sha256'])
        current_digest = config_digest(config)
        restoration = saved.get('restoration')
        if restoration is not None:
            if restore_request is not None and restore_request != restoration['request']:
                raise ValueError('Restoration request differs from checkpoint')
            if step < restoration['after_step']:
                raise ValueError('Checkpoint precedes flow schedule restoration')
        elif restore_request is not None:
            # Restore only the sampling clock. Keep the accelerated policy as
            # provenance and reject simultaneous recipe changes at activation.
            if (config.get('flow_schedule') != 'figure1-cosine-approx' or
                    config.get('benchmark_flow_progress') is not None or
                    config_digest(previous) != bound_digest or current_digest != bound_digest or
                    restore_request['expected_config_sha256'] != bound_digest):
                raise ValueError('Restoration request is bound to a different configuration')
            saved = {**saved, 'restoration': dict(after_step=step, request=restore_request)}
        if current_digest != bound_digest:
            # The caller first validates the specific user-authorized migration.
            # Check every other field, so a data change cannot reset the flow
            # clock or silently alter optimizer, architecture or validation.
            ignored = {'flow_schedule_transition', *allowed_config_changes}
            old_base = {k: v for k, v in previous.items() if k not in ignored}
            new_base = {k: v for k, v in config.items() if k not in ignored}
            if (not allowed_config_changes or config_digest(previous) != bound_digest or
                    old_base != new_base):
                raise ValueError('Configuration differs from flow schedule request')
            migration = dict(after_step=step, from_sha256=bound_digest,
                             to_sha256=current_digest, fields=list(allowed_config_changes))
            saved = {**saved, 'resume_config_sha256': current_digest,
                     'config_migrations': [*saved.get('config_migrations', []), migration]}
        if step < saved['start_step']:
            raise ValueError('Checkpoint precedes flow schedule transition')
        return saved
    if request is None:
        return None
    if set(request) != {'version', 'expected_config_sha256', 'ramp_updates', 'target_progress'}:
        raise ValueError('Invalid flow schedule request fields')
    if config_digest(config) != request['expected_config_sha256'] or config_digest(previous) != request['expected_config_sha256']:
        raise ValueError('Flow schedule request is bound to a different configuration')
    if config.get('flow_schedule') != 'figure1-cosine-approx' or config.get('benchmark_flow_progress') is not None:
        raise ValueError('Flow schedule transition requires the full-training Figure 1 schedule')
    transition = dict(version=request['version'], start_step=step,
                      start_progress=training_flow_progress(step, previous),
                      ramp_updates=request['ramp_updates'], target_progress=request['target_progress'],
                      request=request)
    validate_transition(transition)
    return transition
