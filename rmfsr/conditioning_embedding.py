"""Identity-time positional encoding following EDM SongUNet's convention."""
import torch
from torch import nn

POSITIONAL_COMPONENTS = 32
MAX_PERIOD = 10000


def angular_frequencies(reference):
    indices = torch.arange(POSITIONAL_COMPONENTS, device=reference.device, dtype=reference.dtype)
    return MAX_PERIOD ** (-indices / (POSITIONAL_COMPONENTS - 1))


class PositionalTimeEmbedding(nn.Module):
    """Two times retain a 128-value input to the checkpoint's existing MLP."""
    def __init__(self, original):
        super().__init__()
        self.net = original.net
        self.register_buffer('frequencies', angular_frequencies(original.frequencies))

    def encode(self, time, destination):
        phases = torch.stack((time, destination), -1)[..., None] * self.frequencies
        return torch.cat((phases.sin(), phases.cos()), -1).flatten(1)

    def forward(self, time, destination):
        return self.net(self.encode(time, destination))
