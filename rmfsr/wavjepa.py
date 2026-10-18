"""Opt-in frozen WavJEPA conditioning; offline, not a streaming encoder.

Weights/code: https://huggingface.co/labhamlet/wavjepa-base at the pinned
revision below. Only the damaged waveform enters this branch. Features are
computed once outside MeanFlow's JVP and reused across all solver evaluations.
The external encoder is deliberately not part of the trainable model/EMA.
"""
import torch
from torch import nn
from torch.nn import functional as F


REPOSITORY = 'labhamlet/wavjepa-base'
REVISION = '6be4a5093f6c9adbbd55e00b6e5b8f067aa03345'
DEFAULT_OPTIONS = dict(version='frozen-offline-v1', repository=REPOSITORY,
                       revision=REVISION, adapter_channels=64)


def conditioning_options(config):
    supplied = config.get('wavjepa')
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_OPTIONS):
        raise ValueError('wavjepa must contain only version/repository/revision/adapter_channels')
    options = {**DEFAULT_OPTIONS, **supplied}
    for key in ('version', 'repository', 'revision'):
        if options[key] != DEFAULT_OPTIONS[key]:
            raise ValueError(f'Unsupported WavJEPA {key}; this integration pins the reviewed mono model')
    width = options['adapter_channels']
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError('WavJEPA adapter_channels must be a positive integer')
    if config.get('sample_rate', 16000) != 16000:
        raise ValueError('WavJEPA requires 16 kHz mono audio')
    if config.get('validation_chunk_frames', 10) != 0:
        raise ValueError('WavJEPA is offline: set validation_chunk_frames=0')
    return options


class BottleneckConditioning(nn.Module):
    """Concatenate pretrained context and CNN features, learn a residual mix.

    Only the final fusion projection is zero-initialized. Its input projection
    is random, allowing fusion to learn immediately and the projection to learn
    after the first update. Existing checkpoint predictions are unchanged at
    initialization, including for off-diagonal MeanFlow/JVP evaluations.
    """
    def __init__(self, channels, adapter_channels):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, adapter_channels), nn.GELU())
        self.fuse = nn.Conv2d(channels + adapter_channels, channels, 1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, h, features):
        if features.shape != (h.shape[0], 768, h.shape[-1]):
            raise ValueError('Expected time-aligned WavJEPA features [batch,768,STFT frames]')
        if features.device != h.device or features.dtype != h.dtype:
            raise ValueError('WavJEPA features must share the CNN device and dtype')
        context = self.project(features.transpose(1, 2)).transpose(1, 2)
        context = context[:, :, None].expand(-1, -1, h.shape[2], -1)
        return h + self.fuse(torch.cat((h, context), dim=1))


def align_features(features, positions, spectral, frames):
    """Linear interpolation at STFT window centers, with endpoint extension.

    Positions are physical sample indices, not a resize based on frame counts.
    Our STFT pads left by window-hop and includes final overlap-add tail frames.
    WavJEPA has its own valid convolutions and 2.01 s segment boundaries.
    """
    if features.ndim != 3 or positions.ndim != 1 or features.shape[1] != positions.numel():
        raise ValueError('Invalid frame features or positions')
    if positions.numel() < 1 or not bool((positions[1:] > positions[:-1]).all()):
        raise ValueError('Encoder feature positions must be nonempty and strictly increasing')
    centers = (torch.arange(frames, device=features.device, dtype=positions.dtype) * spectral.hop
               - (spectral.window_size - spectral.hop) + (spectral.window_size - 1) / 2)
    right = torch.searchsorted(positions, centers).clamp(max=positions.numel()-1)
    left = (right-1).clamp(min=0)
    weight = ((centers-positions[left]) / (positions[right]-positions[left]).clamp_min(1)).clamp(0, 1)
    aligned = features[:, left] * (1-weight)[None, :, None] + features[:, right] * weight[None, :, None]
    return aligned.transpose(1, 2).contiguous()


class FrozenWavJEPA(nn.Module):
    """Reviewed upstream encoder only; no predictor/teacher used for restoration.

    The upstream whole-waveform convenience method stretches timestamps and
    mixes integer 2 s lengths with 2.01 s segments. We call its segment encoder
    directly and align using the actual convolution geometry instead.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError('Install the wavjepa optional dependencies before enabling conditioning') from exc
        # Loading pretrained modules must not advance the training RNG.
        with torch.random.fork_rng(devices=[]):
            upstream = AutoModel.from_pretrained(REPOSITORY, revision=REVISION,
                                                 trust_remote_code=True, torch_dtype=torch.float32)
        self.core = upstream.model
        # The public bundle also contains pretraining-only modules (~196 M in
        # total). Keep only the encoder needed for audio features, not copies
        # of its teacher and predictor in GPU memory or restoration checkpoints.
        for name in ('teacher_encoder', 'decoder', 'decoder_to_encoder_mapper',
                     'encoder_to_decoder_mapper', 'mask_token', 'pos_encoding_decoder'):
            delattr(self.core, name)
        spec = self.core.extract_audio.conv_layers_spec
        if spec != [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)]:
            raise ValueError('Unexpected pinned WavJEPA frontend geometry')
        self.stride, self.receptive = 1, 1
        for _, kernel, stride in spec:
            self.receptive += (kernel-1) * self.stride
            self.stride *= stride
        self.segment_samples = self.core.target_length
        self.tokens = self.core.total_patches
        # Upstream int(16000 * 2.01) is 32159 due to float truncation.
        if self.segment_samples != 32159 or self.tokens != 200:
            raise ValueError('Unexpected pinned WavJEPA segment geometry')
        self.requires_grad_(False)
        self.eval()
        self.to(device)

    def train(self, mode=True):
        # Parent .train() calls must never enable dropout or encoder updates.
        return super().train(False)

    def metadata(self):
        return dict(repository=REPOSITORY, revision=REVISION, frozen=True,
                    encoder_parameters=sum(p.numel() for p in self.parameters()),
                    segment_samples=self.segment_samples, sample_rate=16000,
                    causal=False, alignment='frontend-centers-to-STFT-centers-v1')

    @torch.no_grad()
    def forward(self, damaged, spectral, frames):
        if spectral.sr != 16000 or damaged.ndim != 2 or damaged.shape[-1] < 1:
            raise ValueError('WavJEPA expects nonempty [batch,samples] mono audio at 16 kHz')
        if not bool(torch.isfinite(damaged).all()):
            raise ValueError('Nonfinite damaged audio')
        expected_frames = (damaged.shape[-1] + spectral.hop - 1) // spectral.hop + spectral.window_size // spectral.hop
        if frames != expected_frames:
            raise ValueError('STFT frame count does not match the damaged waveform')
        device = next(self.parameters()).device
        wave = damaged.detach().to(device=device, dtype=torch.float32)
        # Match the public feature extractor's per-clip -14 dBFS normalization;
        # only this encoder copy changes gain. CNN input/clean targets do not.
        rms = wave.square().mean(-1, keepdim=True).sqrt()
        wave = wave * torch.where(rms > 0, 10**(-14/20) / rms.clamp_min(1e-12), torch.ones_like(rms))
        features, positions = [], []
        token_start = torch.arange(self.tokens, device=device) * self.stride
        for start in range(0, wave.shape[-1], self.segment_samples):
            segment = wave[:, None, start:start+self.segment_samples]
            valid = segment.shape[-1]
            segment = F.pad(segment, (0, self.segment_samples-valid))
            # Upstream's released preprocessing normalizes each padded segment.
            segment = (segment-segment.mean((-2, -1), keepdim=True)) / (segment.std((-2, -1), keepdim=True)+1e-5)
            valid_tokens = token_start < valid
            mask = (~valid_tokens)[None].expand(wave.shape[0], -1)
            # Upstream returns inference tensors; clone outside its inference
            # context so trainable fusion layers can save features for backward.
            embedding = self.core._get_segment_representation(segment, mask).clone()
            features.append(embedding[:, valid_tokens])
            positions.append(start + token_start[valid_tokens] + (self.receptive-1)/2)
        result = align_features(torch.cat(features, 1), torch.cat(positions), spectral, frames)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError('WavJEPA produced nonfinite conditioning')
        return result


def build_conditioner(config, device):
    if config.get('wavjepa') is None:
        return None
    conditioning_options(config)
    return FrozenWavJEPA(device)
