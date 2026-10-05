"""Experimental resource-budget heuristic, not the author's recovered model.

Version efficient-v1: grouped mixers, compact attention, and a folded temporal
bottleneck. Fixed input: (batch, 2, 161, frames), 16 kHz / 10 ms hop.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .model import (RMFSR, ConditionedConv, InvertedResidual,
                    ChannelNorm, SnakeBeta)


def validate_options(channels, groups, attention_rank, folded_width,
                     encoder_dilations, tcn_dilations):
    def positive_int(value):
        return isinstance(value, int) and not isinstance(value, bool) and value > 0
    if not positive_int(groups):
        raise ValueError('groups must be a positive integer')
    if (not isinstance(channels, (list, tuple)) or len(channels) != 5
            or any(not positive_int(c) or c % groups for c in channels)):
        raise ValueError('efficient-v1 needs five positive channel widths divisible by groups')
    if not positive_int(attention_rank) or attention_rank % 4:
        raise ValueError('attention_rank must be a positive multiple of four')
    if (not isinstance(folded_width, int) or isinstance(folded_width, bool)
            or folded_width < 0):
        raise ValueError('folded_width must be a nonnegative integer (zero keeps the original TCN)')
    for name, value, length in [('encoder_dilations', encoder_dilations, 5),
                                ('tcn_dilations', tcn_dilations, 4)]:
        if (not isinstance(value, (list, tuple)) or len(value) != length
                or any(not positive_int(d) for d in value)):
            raise ValueError(f'{name} must contain {length} positive integers')


def grouped(layer, groups):
    old = layer.conv
    g = math.gcd(groups, math.gcd(old.in_channels, old.out_channels))
    layer.conv = nn.Conv2d(old.in_channels, old.out_channels, old.kernel_size,
                          stride=old.stride, dilation=old.dilation, groups=g)
    return layer


def shuffle(x, groups):
    b, c, f, t = x.shape
    return x.reshape(b, groups, c // groups, f, t).transpose(1, 2).reshape(b, c, f, t)


class GroupedResidual(InvertedResidual):
    def __init__(self, cin, cout, kernel, dilation=1, stride=1, groups=8):
        super().__init__(cin, cout, kernel, dilation, stride)
        self.groups = math.gcd(groups, cin * 2)
        grouped(self.expand, groups)
        grouped(self.project, groups)
        if self.residual is not None:
            grouped(self.residual, groups)

    def forward(self, x, emb, cache=None):
        skip = x[:, :, ::self.stride, :]
        if self.residual is not None:
            skip = self.residual(skip, emb, cache)
        h = shuffle(self.a1(self.expand(self.norm(x), emb, cache)), self.groups)
        h = self.a2(self.depth(h, emb, cache))
        return skip + self.project(h, emb, cache)


class CompactFrequencyAttention(nn.Module):
    """Dense low-rank channel projections; frequency attention at every stage."""
    def __init__(self, channels, rank=16, heads=4):
        super().__init__()
        if heads < 1 or rank < 1 or rank % heads:
            raise ValueError('Attention rank must be positive and divisible by heads')
        self.rank, self.heads = rank, heads
        self.norm = ChannelNorm(channels)
        self.qkv = ConditionedConv(channels, 3 * rank)
        self.out = ConditionedConv(rank, channels)

    def forward(self, x, emb, cache=None):
        b, _, f, t = x.shape
        qkv = self.qkv(self.norm(x), emb, cache).reshape(b, 3, self.heads, self.rank // self.heads, f, t)
        q, k, v = qkv.permute(1, 0, 5, 2, 4, 3).unbind(0)
        weights = (q @ k.transpose(-1, -2) / math.sqrt(self.rank // self.heads)).softmax(-1)
        h = (weights @ v).permute(0, 2, 4, 3, 1).reshape(b, self.rank, f, t)
        return x + self.out(h, emb, cache)


class FoldedTemporalBottleneck(nn.Module):
    """Joint frequency/channel features at F=1, with full-rate causal time axis."""
    def __init__(self, channels=256, bins=6, width=448, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.channels, self.bins = channels, bins
        self.down = ConditionedConv(channels * bins, width)
        self.blocks = nn.ModuleList([InvertedResidual(width, width, (1, 11), d) for d in dilations])
        self.up = ConditionedConv(width, channels * bins)

    def forward(self, x, emb, cache=None):
        b, c, f, t = x.shape
        if (c, f) != (self.channels, self.bins):
            raise ValueError('Folded bottleneck requires 161-bin model input')
        h = self.down(x.reshape(b, c * f, 1, t), emb, cache)
        for block in self.blocks:
            h = block(h, emb, cache)
        return x + self.up(h, emb, cache).reshape(b, c, f, t)


class EfficientRMFSR(RMFSR):
    def __init__(self, groups=16, attention_rank=16, folded_width=448,
                 channels=(64, 64, 128, 256, 256),
                 encoder_dilations=(1, 2, 4, 8, 16), tcn_dilations=(1, 2, 4, 8)):
        validate_options(channels, groups, attention_rank, folded_width,
                         encoder_dilations, tcn_dilations)
        super().__init__(channels=channels, encoder_dilations=encoder_dilations,
                         tcn_dilations=tcn_dilations, decoder_before_upsample=True)
        self.groups, self.attention_rank, self.folded_width = groups, attention_rank, folded_width
        cin = self.channels[0]
        for i, (cout, d) in enumerate(zip(self.channels, self.encoder_dilations)):
            self.encoder[i] = GroupedResidual(cin, cout, (3, 3), d, stride=2, groups=groups)
            self.enc_attention[i] = CompactFrequencyAttention(cout, attention_rank)
            cin = cout
        for j, i in enumerate(reversed(range(len(self.channels)))):
            cout = self.channels[i]
            self.skip[j] = grouped(ConditionedConv(self.channels[i], cin), groups)
            self.decoder[j] = GroupedResidual(cin, cout, (3, 2), groups=groups)
            self.dec_attention[j] = CompactFrequencyAttention(cout, attention_rank)
            cin = cout
        if folded_width:
            self.tcn = nn.ModuleList()  # remove replaced weights completely
            self.folded = FoldedTemporalBottleneck(channels=self.channels[-1], width=folded_width,
                                                  dilations=self.tcn_dilations)
        else:
            self.folded = None  # original dense F=6 TCN, a conservative small model
        # Learn frequency-specific corrections AFTER final nearest upsampling.
        self.refine = ConditionedConv(cin, cin, (3, 1), groups=cin)
        self.refine_activation = SnakeBeta(cin)

    def architecture(self):
        return dict(super().architecture(), model_type='efficient-v1', groups=self.groups,
                    attention_rank=self.attention_rank, folded_width=self.folded_width,
                    frequency_bins=161, final_depthwise_frequency_refinement=True)

    def forward(self, x, y, t, r, cache=None):
        if x.ndim != 4 or x.shape[1:3] != (2, 161) or y.shape != x.shape:
            raise ValueError('Expected equal complex spectra (B, 2, 161, T)')
        emb = self.embedding(t, r)
        h = self.stem(torch.cat((x, y), 1), emb, cache)
        skips, sizes = [], []
        for block, attention in zip(self.encoder, self.enc_attention):
            sizes.append(h.shape[2])
            h = attention(block(h, emb, cache), emb, cache)
            skips.append(h)
        if self.folded is not None:
            h = self.folded(h, emb, cache)
        else:
            for block in self.tcn:
                h = block(h, emb, cache)
        for mapping, block, attention, skip, size in zip(self.skip, self.decoder, self.dec_attention, reversed(skips), reversed(sizes)):
            h = attention(block(h + mapping(skip, emb, cache), emb, cache), emb, cache)
            h = F.interpolate(h, size=(size, h.shape[-1]), mode='nearest')
        h = h + self.refine_activation(self.refine(h, emb, cache))
        return y + self.head(h, emb, cache)
