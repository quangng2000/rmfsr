"""Causal complex-spectral U-Net. All normalization/attention is frame-local.
Unspecified paper choices are recorded in README; this is not official code.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class SnakeBeta(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.log_beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
    def forward(self, x):
        return x + torch.sin(self.log_alpha.exp() * x).square() / (self.log_beta.exp() + 1e-8)

class ChannelNorm(nn.Module):
    """Normalize channels at one frequency/time position, never across time."""
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        return (x - mean) * torch.rsqrt((x - mean).square().mean(1, keepdim=True) + 1e-5) * self.weight + self.bias

class ConditionedConv(nn.Module):
    def __init__(self, cin, cout, kernel=(1, 1), dilation=1, stride=1, groups=1):
        super().__init__()
        self.history = (kernel[1] - 1) * dilation
        self.freq_pad = kernel[0] // 2
        self.condition = nn.Linear(128, cin)
        self.conv = nn.Conv2d(cin, cout, kernel, stride=(stride, 1),
                              dilation=(1, dilation), groups=groups)
    def forward(self, x, emb, cache=None):
        x = x + self.condition(emb)[:, :, None, None]
        if cache is None:
            x = F.pad(x, (self.history, 0, self.freq_pad, self.freq_pad))
        else:
            # A separate cache dictionary is required for each ODE evaluation index.
            previous = cache.get(self)
            if previous is None:
                previous = x.new_zeros(*x.shape[:-1], self.history)
            joined = torch.cat((previous, x), dim=-1)
            if self.history:
                cache[self] = joined[..., -self.history:].detach()
            x = F.pad(joined, (0, 0, self.freq_pad, self.freq_pad))
        return self.conv(x)

class InvertedResidual(nn.Module):
    def __init__(self, cin, cout, kernel, dilation=1, stride=1):
        super().__init__()
        hidden = cin * 2
        self.norm = ChannelNorm(cin)
        self.expand = ConditionedConv(cin, hidden)
        self.a1 = SnakeBeta(hidden)
        self.depth = ConditionedConv(hidden, hidden, kernel, dilation, stride, hidden)
        self.a2 = SnakeBeta(hidden)
        self.project = ConditionedConv(hidden, cout)
        self.residual = ConditionedConv(cin, cout) if cin != cout else None
        self.stride = stride
    def forward(self, x, emb, cache=None):
        skip = x[:, :, ::self.stride, :]
        if self.residual is not None:
            skip = self.residual(skip, emb, cache)
        h = self.a1(self.expand(self.norm(x), emb, cache))
        h = self.a2(self.depth(h, emb, cache))
        return skip + self.project(h, emb, cache)

class FrequencyAttention(nn.Module):
    """Multihead frequency attention, independently at every audio frame."""
    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads, self.channels = heads, channels
        self.norm = ChannelNorm(channels)
        self.qkv = ConditionedConv(channels, channels * 3)
        self.out = ConditionedConv(channels, channels)
    def forward(self, x, emb, cache=None):
        b, c, f, n = x.shape
        qkv = self.qkv(self.norm(x), emb, cache).reshape(b, 3, self.heads, c // self.heads, f, n)
        q, k, v = qkv.permute(1, 0, 5, 2, 4, 3).unbind(0)
        # Explicit matmul supports forward-mode AD on MPS; fused attention may not.
        weights = (q @ k.transpose(-1, -2) / math.sqrt(c // self.heads)).softmax(-1)
        h = (weights @ v).permute(0, 2, 4, 3, 1).reshape(b, c, f, n)
        return x + self.out(h, emb, cache)

class TimeEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('frequencies', torch.randn(32) * 16)
        self.net = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))
    def forward(self, t, r):
        phases = 2 * math.pi * torch.stack((t, r), -1)[..., None] * self.frequencies
        return self.net(torch.cat((phases.sin(), phases.cos()), -1).flatten(1))

class RMFSR(nn.Module):
    def __init__(self, channels=(64,64,128,256,256), encoder_dilations=(1,2,4,8,16), tcn_dilations=(1,2,4,8), decoder_layout='mirror-v2', bottleneck_attention=False, decoder_before_upsample=False):
        super().__init__()
        if decoder_layout not in ('mirror-v2', 'legacy-v1'):
            raise ValueError('Unknown decoder layout')
        self.channels = tuple(channels)
        self.encoder_dilations = tuple(encoder_dilations)
        self.tcn_dilations = tuple(tcn_dilations)
        self.decoder_layout = decoder_layout
        self.bottleneck_attention = bool(bottleneck_attention)
        self.decoder_before_upsample = bool(decoder_before_upsample)
        self.embedding = TimeEmbedding()
        self.stem = ConditionedConv(4, channels[0])
        self.encoder = nn.ModuleList()
        self.enc_attention = nn.ModuleList()
        cin = channels[0]
        for cout, dilation in zip(channels, encoder_dilations):
            self.encoder.append(InvertedResidual(cin, cout, (3,3), dilation, stride=2))
            self.enc_attention.append(FrequencyAttention(cout))
            cin = cout
        self.tcn = nn.ModuleList([InvertedResidual(cin, cin, (1,11), d) for d in tcn_dilations])
        # Optional capacity study, not a claim about the authors' attention placement.
        self.tcn_attention = nn.ModuleList([FrequencyAttention(cin) for _ in tcn_dilations]
                                           if bottleneck_attention else [])
        self.skip = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.dec_attention = nn.ModuleList()
        for i in reversed(range(len(channels))):
            # Paper decoder stage outputs mirror the encoder widths. The legacy
            # layout is retained only to evaluate the original pilot checkpoints.
            cout = channels[i] if decoder_layout == 'mirror-v2' else channels[max(i-1, 0)]
            self.skip.append(ConditionedConv(channels[i], cin))
            self.decoder.append(InvertedResidual(cin, cout, (3,2)))
            self.dec_attention.append(FrequencyAttention(cout))
            cin = cout
        self.head = ConditionedConv(cin, 2)
        # Small initial residual around degraded input avoids huge random waveforms.
        nn.init.zeros_(self.head.conv.weight)
        nn.init.zeros_(self.head.conv.bias)
        self.receptive_frames = 1 + 2*sum(encoder_dilations) + 10*sum(tcn_dilations) + len(channels)
    def architecture(self):
        return dict(decoder_layout=self.decoder_layout, bottleneck_attention=self.bottleneck_attention,
                    decoder_before_upsample=self.decoder_before_upsample,
                    channels=list(self.channels),
                    encoder_dilations=list(self.encoder_dilations),
                    tcn_dilations=list(self.tcn_dilations))

    def forward(self, x, y, t, r, cache=None):
        emb = self.embedding(t, r)
        h = self.stem(torch.cat((x,y), 1), emb, cache)
        skips, sizes = [], []
        for block, attention in zip(self.encoder, self.enc_attention):
            sizes.append(h.shape[2])
            h = attention(block(h, emb, cache), emb, cache)
            skips.append(h)
        for i, block in enumerate(self.tcn):
            h = block(h, emb, cache)
            if self.bottleneck_attention:
                h = self.tcn_attention[i](h, emb, cache)
        for mapping, block, attention, skip, size in zip(self.skip, self.decoder, self.dec_attention, reversed(skips), reversed(sizes)):
            h = h + mapping(skip, emb, cache)
            if self.decoder_before_upsample:
                h = attention(block(h, emb, cache), emb, cache)
                h = F.interpolate(h, size=(size,h.shape[-1]), mode='nearest')
            else:
                h = F.interpolate(h, size=(size,h.shape[-1]), mode='nearest')
                h = attention(block(h, emb, cache), emb, cache)
        return y + self.head(h, emb, cache)


def model_from_checkpoint(checkpoint):
    """Read trusted historical weights with their original decoder wiring."""
    architecture = checkpoint.get('architecture')
    layout = architecture['decoder_layout'] if architecture is not None else 'legacy-v1'
    model = RMFSR(channels=checkpoint['config']['channels'], decoder_layout=layout,
                  encoder_dilations=architecture.get('encoder_dilations', (1,2,4,8,16)) if architecture else (1,2,4,8,16),
                  tcn_dilations=architecture.get('tcn_dilations', (1,2,4,8)) if architecture else (1,2,4,8),
                  bottleneck_attention=architecture.get('bottleneck_attention', False) if architecture else False,
                  decoder_before_upsample=architecture.get('decoder_before_upsample', False) if architecture else False)
    if architecture is not None and architecture != model.architecture():
        raise ValueError('Checkpoint architecture differs from the supported model')
    return model
