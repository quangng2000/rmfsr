"""Strict identities for the explicitly authorized conditioning migration."""
import copy
import re

import torch

from .conditioning import ARCHITECTURE
from .flow_curriculum import config_digest
from .frequency_transition import tensor_sha256

MARKER = 'conditioning_architecture_transition'
FREQUENCIES = 'embedding.frequencies'
REQUEST_FIELDS = {'version', 'transition_id', 'expected_source_sha256',
                  'expected_config_sha256', 'expected_step'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_source(checkpoint, request):
    require(isinstance(request, dict) and set(request) == REQUEST_FIELDS, 'Invalid migration request fields')
    require(type(request['version']) is int and request['version'] == 1, 'Invalid migration request version')
    require(isinstance(request['transition_id'], str) and
            re.fullmatch('[a-z0-9][a-z0-9-]{0,79}', request['transition_id']), 'Invalid transition id')
    require(is_digest(request['expected_source_sha256']), 'Invalid source checkpoint digest')
    require(type(request['expected_step']) is int and request['expected_step'] >= 0 and
            checkpoint['step'] == request['expected_step'], 'Source checkpoint step mismatch')
    require(config_digest(checkpoint['config']) == request['expected_config_sha256'], 'Source configuration mismatch')
    require(not checkpoint.get('local_experiment') and MARKER not in checkpoint, 'Already migrated or experimental source')
    require(checkpoint['config'].get('conditioning_architecture') is None, 'Source must use original conditioning')
    required = {'model', 'ema', 'optimizer', 'architecture', 'torch_rng', 'numpy_rng', 'python_rng',
                'data_rng', 'history', 'input_fingerprints', 'validation_selection_metric'}
    require(required <= checkpoint.keys(), 'Incomplete production checkpoint')
    validate_source_frequencies(checkpoint)


def validate_source_frequencies(checkpoint):
    marker = checkpoint.get('frequency_width_transition', {})
    require(marker.get('nominal_frequency_width') == 4, 'Source must be verified width 4')
    hashes = marker.get('request', {}).get('target_frequency_sha256', {})
    for kind in ('model', 'ema'):
        value = checkpoint[kind].get(FREQUENCIES)
        require(isinstance(value, torch.Tensor) and value.shape == (32,), 'Missing Fourier frequency buffer')
        require(torch.isfinite(value).all() and tensor_sha256(value) == hashes.get(kind), 'Source frequency digest mismatch')
    require(torch.equal(checkpoint['model'][FREQUENCIES], checkpoint['ema'][FREQUENCIES]), 'Raw/EMA frequencies differ')


def fresh_optimizer(model, source, original_parameter_count):
    groups = source.get('param_groups', [])
    require(len(groups) == 1, 'Migration supports the production single AdamW parameter group')
    group = copy.deepcopy(groups[0])
    require(len(group['params']) == original_parameter_count and
            len(set(group['params'])) == original_parameter_count, 'Source optimizer parameter count mismatch')
    require(group.get('decoupled_weight_decay', True) is True, 'Source optimizer must be AdamW')
    group['params'] = list(range(len(list(model.parameters()))))
    if 'param_names' in group:
        group['param_names'] = list(dict(model.named_parameters()))
    state = {'state': {}, 'param_groups': [group]}
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.load_state_dict(state)
    require(not optimizer.state, 'Migration must reset AdamW moments')
    return state


def validate_conditioning_transition(checkpoint, config):
    """Validate a persisted transition on every normal process resume."""
    marker = checkpoint.get(MARKER)
    if config.get('conditioning_architecture') != ARCHITECTURE:
        require(marker is None, 'Conditioning transition on an original architecture')
        return None
    require(isinstance(marker, dict), 'Conditioned resume requires migration provenance')
    require(type(marker.get('version')) is int and marker['version'] == 1 and marker.get('conditioning_architecture') == ARCHITECTURE,
            'Invalid conditioning transition architecture')
    require(marker.get('target_config_sha256') == config_digest(config) == config_digest(checkpoint['config']),
            'Conditioning transition configuration mismatch')
    require(type(marker.get('after_step')) is int and marker['after_step'] <= checkpoint['step'],
            'Checkpoint precedes conditioning transition')
    require(checkpoint['architecture'].get('conditioning_architecture') == ARCHITECTURE,
            'Conditioning transition architecture mismatch')
    require('frequency_width_transition' not in checkpoint, 'Obsolete Fourier marker in conditioned checkpoint')
    require(is_digest(marker.get('source_checkpoint', {}).get('sha256')) and
            is_digest(marker.get('source_config_sha256')), 'Missing conditioning source provenance')
    require(marker.get('optimizer_policy') == 'reset_adamw_moments_preserve_group_options' and
            marker.get('best_validation_reset') is True, 'Invalid conditioning optimizer/validation policy')
    return copy.deepcopy(marker)
