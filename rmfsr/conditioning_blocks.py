"""sCM double-normalization formula with explicitly causal speech statistics."""
import torch
from torch import nn
from torch.nn import functional as F

PIXEL_EPSILON = 1e-8
GROUP_EPSILON = 1e-5


def pixel_normalize(value):
    return value * torch.rsqrt(value.square().mean(1, keepdim=True) + PIXEL_EPSILON)


def group_count(channels):
    limit = max(1, min(32, channels // 4))
    return max(group for group in range(1, limit + 1) if channels % group == 0)


class FrameGroupNorm(nn.Module):
    """GroupNorm over group channels and frequency, independently at each frame."""
    def __init__(self, channels):
        super().__init__()
        self.groups = group_count(channels)

    def forward(self, value):
        batch, channels, frequencies, frames = value.shape
        grouped = value.reshape(batch, self.groups, channels // self.groups, frequencies, frames)
        centered = grouped - grouped.mean((2, 3), keepdim=True)
        normalized = centered * torch.rsqrt(centered.square().mean((2, 3), keepdim=True) + GROUP_EPSILON)
        return normalized.reshape_as(value)


class DoubleNormalizedConv(nn.Module):
    """Preserve convolution/bias weights; add unit-initialized conditional scale."""
    def __init__(self, original):
        super().__init__()
        self.history, self.freq_pad = original.history, original.freq_pad
        self.condition, self.conv = original.condition, original.conv
        self.norm = FrameGroupNorm(self.condition.out_features)
        self.scale = nn.Linear(self.condition.in_features, self.condition.out_features)
        self.scale.to(device=self.condition.weight.device, dtype=self.condition.weight.dtype)
        nn.init.zeros_(self.scale.weight)
        nn.init.ones_(self.scale.bias)

    def modulate(self, value, embedding):
        scale = pixel_normalize(self.scale(embedding))[:, :, None, None]
        bias = pixel_normalize(self.condition(embedding))[:, :, None, None]
        return self.norm(value) * scale + bias

    def padded(self, value, cache):
        if cache is None:
            return F.pad(value, (self.history, 0, self.freq_pad, self.freq_pad))
        previous = cache.get(self)
        if previous is None:
            previous = value.new_zeros(*value.shape[:-1], self.history)
        joined = torch.cat((previous, value), -1)
        if self.history:
            cache[self] = joined[..., -self.history:].detach()
        return F.pad(joined, (0, 0, self.freq_pad, self.freq_pad))

    def forward(self, value, embedding, cache=None):
        return self.conv(self.padded(self.modulate(value, embedding), cache))
