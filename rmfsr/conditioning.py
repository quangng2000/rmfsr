"""Opt-in sCM-style conditioning; numerically identical to the audited 100-update test."""
import copy
import torch
from .conditioning_blocks import DoubleNormalizedConv, GROUP_EPSILON, PIXEL_EPSILON
from .conditioning_embedding import PositionalTimeEmbedding, angular_frequencies

ARCHITECTURE = 'scm-positional-adn-v1'

DESCRIPTION = {
    'version': 1,
    'name': 'positional identity-time embeddings and adaptive double normalization',
    'formula': 'FrameGroupNorm(x) * pnorm(scale(t,r)) + pnorm(bias(t,r))',
    'positional': '32 angular frequencies 10000**(-i/31) per time; sin then cos; no 2*pi or time multiplier',
    'embedding_reference': 'EDM SongUNet positional endpoint=True convention; sCM does not uniquely specify this convention',
    'time_input': 'Identity t and r; meanflow [0,1] times retained rather than TrigFlow times',
    'pixel_epsilon': PIXEL_EPSILON,
    'group_epsilon': GROUP_EPSILON,
    'normalization_axes': 'pnorm channels; GroupNorm group channels and frequency separately per audio frame',
    'group_count': 'largest channel divisor <= min(32, channels//4), minimum 1',
    'migration': 'Retain convolution, additive projection, and time MLP weights; new scale weight=0 and bias=1',
    'placement': 'Replace every ConditionedConv; existing surrounding ChannelNorm blocks retained',
    'adaptation': 'Causal frame-local GroupNorm, two-time conditioning, RMFSR placement; not the complete sCM network/objective',
    'caveat': 'An abrupt architecture/basis retrofit, not function preserving or sCM teacher pretraining',
    'derivatives': 'Full forward/backward differentiation through both normalization denominators; no detached tangent',
}


def replace_children(parent, conditioned_type):
    count = 0
    for name, child in list(parent.named_children()):
        if isinstance(child, conditioned_type):
            assert not child._forward_hooks and not child.condition._forward_hooks
            setattr(parent, name, DoubleNormalizedConv(child))
            count += 1
        else:
            count += replace_children(child, conditioned_type)
    return count


def install(model, conditioned_type):
    assert not hasattr(model, '_scm_architecture_version'), 'Architecture already installed'
    model.embedding = PositionalTimeEmbedding(model.embedding)
    count = replace_children(model, conditioned_type)
    assert count > 0, 'No conditioning layers found'
    model.register_buffer('_scm_architecture_version', torch.tensor(1, dtype=torch.int64))
    model._scm_conditioning = copy.deepcopy(DESCRIPTION)
    model.conditioning_architecture = ARCHITECTURE
    return model


def inspect(model):
    assert int(model._scm_architecture_version) == DESCRIPTION['version']
    assert isinstance(model.embedding, PositionalTimeEmbedding)
    actual = model.embedding.frequencies
    expected = angular_frequencies(actual.detach().cpu()).to(actual.device)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
    layers = {name: layer.norm.groups for name, layer in model.named_modules()
              if isinstance(layer, DoubleNormalizedConv)}
    assert layers, 'Double normalization is absent'
    assert all(not layer._forward_hooks and not layer._forward_pre_hooks
               for layer in model.modules()), 'Unexpected hidden hooks'
    return dict(enabled=True, layers=len(layers), groups=layers, description=copy.deepcopy(DESCRIPTION))
