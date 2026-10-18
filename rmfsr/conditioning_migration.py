"""Offline, guarded conversion; never mutates a source or controls training."""
import copy
import math

import torch

from .conditioning import ARCHITECTURE, inspect
from .conditioning_migration_checks import (FREQUENCIES, MARKER, fresh_optimizer, require,
                                            validate_conditioning_transition, validate_source)
from .flow_curriculum import config_digest, resolve_transition, training_flow_progress
from .model import model_from_checkpoint, model_from_config


def migrate_weights(source, config):
    original = model_from_checkpoint(source)
    migrated = model_from_config(config)
    old_parameters, new_parameters = dict(original.named_parameters()), dict(migrated.named_parameters())
    added = set(new_parameters) - set(old_parameters)
    require(added and all(key.endswith(('.scale.weight', '.scale.bias')) for key in added),
            'Unexpected new learned parameter in conditioning architecture')
    require(set(old_parameters) <= set(new_parameters), 'Migration removed learned parameters')
    states = {}
    for kind in ('model', 'ema'):
        original.load_state_dict(source[kind], strict=True)
        states[kind] = transplant_state(source[kind], migrated.state_dict(), added)
    counts = dict(before=sum(p.numel() for p in old_parameters.values()),
                  after=sum(p.numel() for p in new_parameters.values()))
    counts['added'] = counts['after'] - counts['before']
    return migrated, states, counts, len(old_parameters), len(added)


def transplant_state(old, new, added):
    expected = added | {'_scm_architecture_version'}
    require(set(new) - set(old) == expected and set(old) <= set(new), 'Unexpected model state change')
    for key, value in old.items():
        require(value.shape == new[key].shape and value.dtype == new[key].dtype, f'State mismatch: {key}')
        require(not value.is_floating_point() or torch.isfinite(value).all(), f'Nonfinite source: {key}')
        if key != FREQUENCIES:
            new[key] = value.detach().clone()
    for key in added:
        expected_value = 0 if key.endswith('.weight') else 1
        require(torch.all(new[key] == expected_value), f'Invalid new scale initialization: {key}')
    return copy.deepcopy(new)


def migrated_config(source):
    config = copy.deepcopy(source['config'])
    config['conditioning_architecture'] = ARCHITECTURE
    transition = resolve_transition(config, source['config'], source['step'],
                                    allowed_config_changes=('conditioning_architecture',))
    if transition is not None:
        config['flow_schedule_transition'] = transition
    require(training_flow_progress(source['step'], config) == training_flow_progress(source['step'], source['config']),
            'Conditioning migration changed flow schedule progress')
    return config


def convert_payload(source, request, source_path):
    """Return new trusted-checkpoint payload, preserving training position and RNGs."""
    validate_source(source, request)
    config = migrated_config(source)
    with torch.random.fork_rng(devices=[]):
        model, states, counts, old_count, added_count = migrate_weights(source, config)
    marker = dict(version=1, transition_id=request['transition_id'], after_step=source['step'],
                  conditioning_architecture=ARCHITECTURE, source_checkpoint=dict(path=str(source_path),
                  sha256=request['expected_source_sha256']), source_config_sha256=config_digest(source['config']),
                  target_config_sha256=config_digest(config), source_architecture=copy.deepcopy(source['architecture']),
                  source_frequency_transition=copy.deepcopy(source['frequency_width_transition']),
                  optimizer_policy='reset_adamw_moments_preserve_group_options', best_validation_reset=True,
                  previous_best_validation=(source['best_validation'] if math.isfinite(source.get('best_validation', float('inf'))) else None), parameters=counts,
                  preserved_parameter_tensors=old_count, initialized_scale_tensors=added_count)
    result = dict(source, **states, config=config, architecture=model.architecture(),
                  optimizer=fresh_optimizer(model, source['optimizer'], old_count), best_validation=float('inf'))
    result.pop('frequency_width_transition', None)
    result[MARKER] = marker
    verify_payload(result)
    return result


def verify_payload(checkpoint):
    validate_conditioning_transition(checkpoint, checkpoint['config'])
    with torch.random.fork_rng(devices=[]):
        model = model_from_checkpoint(checkpoint)
        for kind in ('model', 'ema'):
            model.load_state_dict(checkpoint[kind], strict=True)
            inspect(model)
        optimizer = torch.optim.AdamW(model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer'])
    require(not optimizer.state, 'Converted checkpoint unexpectedly has AdamW moments')
    require(math.isinf(checkpoint['best_validation']), 'Converted checkpoint must reset best validation')


def convert_checkpoint(source, destination, request):
    """Publish a separately named checkpoint only after strict reload verification."""
    from .conditioning_migration_io import publish_conversion
    return publish_conversion(source, destination, request)
