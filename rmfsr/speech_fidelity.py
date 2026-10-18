"""Opt-in PESQ + SI-SDR supervision adapted from arXiv:2512.10382, Sec. 3.4.

Use the instantaneous (r=t) data estimate with MeanFlow: an off-diagonal
data-parameterized average velocity is not a clean endpoint estimate.
This is an adaptation, not the paper's NCSN++ / EDM reproduction.
"""
import math

import torch


def fidelity_config(config):
    value = config.get('speech_fidelity')
    if value is None:
        return None
    keys = {'version', 'pesq_weight', 'sisdr_weight'}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError('speech_fidelity requires version, pesq_weight and sisdr_weight')
    if value['version'] != 'paper1-pesq-sisdr-v1':
        raise ValueError('Unknown speech_fidelity version')
    for key in ('pesq_weight', 'sisdr_weight'):
        weight = value[key]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError(f'speech_fidelity {key} must be finite and positive')
    return dict(value)


def fidelity_resume_transition(previous, current):
    old, new = fidelity_config(previous), fidelity_config(current)
    if old == new:
        return None
    if old is not None or new is None or current.get('allow_speech_fidelity_on_resume') is not True:
        raise ValueError('Changing speech fidelity on resume requires an explicit one-time opt-in')
    return dict(previous=None, speech_fidelity=new, estimate='instantaneous-r-equals-t',
                reference='2512.10382v1-section-3.4', best_validation_reset=False)


def negative_si_sdr(prediction, reference):
    """Per-utterance negative SI-SDR, without mean removal (paper Sec. 3.4).

    A reference-relative numerical floor bounds perfect/zero predictions.
    Silent references are excluded by the caller; SI-SDR is undefined there.
    """
    energy = reference.square().sum(-1).clamp_min(1e-12)
    scale = (prediction * reference).sum(-1) / energy
    projected = scale[:, None] * reference
    floor = 1e-8 * energy
    ratio = (projected.square().sum(-1) + floor) / ((prediction - projected).square().sum(-1) + floor)
    return -10 * torch.log10(ratio)


class SpeechFidelityLoss(torch.nn.Module):
    def __init__(self, spectral, settings):
        super().__init__()
        self.settings = fidelity_config({'speech_fidelity': settings})
        if spectral.sr != 16000:
            raise ValueError('paper1 speech fidelity currently requires 16 kHz audio')
        try:
            from torch_pesq import PesqLoss
        except ImportError as exc:
            raise ImportError('Install the optional training losses with: pip install -e ".[fidelity]"') from exc
        self.spectral = spectral
        # Keep library scaling explicit: all outer weights live in the recipe.
        self.pesq = PesqLoss(factor=1.0, sample_rate=spectral.sr).requires_grad_(False)

    def forward(self, coefficients, reference):
        if reference.ndim != 2 or reference.shape[0] != coefficients.shape[0]:
            raise ValueError('Expected aligned [batch,samples] reference waveforms')
        if reference.shape[-1] < 8000:
            raise ValueError('PESQ supervision requires at least 0.5 seconds of audio')
        prediction = self.spectral.decode_differentiable(coefficients, reference.shape[-1])
        reference = reference.detach().to(device=prediction.device, dtype=prediction.dtype)
        if not torch.isfinite(prediction).all() or not torch.isfinite(reference).all():
            raise FloatingPointError('Nonfinite speech fidelity waveform')
        reference_power = reference.square().mean(-1)
        prediction_power = prediction.square().mean(-1)
        voiced = reference_power > 1e-12
        # torch-pesq level normalization divides by signal power. Do not feed
        # silent signals to it or conceal NaNs with nan_to_num; log exclusions.
        pesq_valid = voiced & (prediction_power > 1e-12)
        zero = prediction.sum() * 0
        sisdr = negative_si_sdr(prediction[voiced], reference[voiced]).mean() if voiced.any() else zero
        pesq = self.pesq(reference[pesq_valid].clone(), prediction[pesq_valid].clone()).mean() if pesq_valid.any() else zero
        if not torch.isfinite(sisdr) or not torch.isfinite(pesq):
            raise FloatingPointError('Nonfinite PESQ or SI-SDR loss')
        weighted_pesq = self.settings['pesq_weight'] * pesq
        weighted_sisdr = self.settings['sisdr_weight'] * sisdr
        total = weighted_pesq + weighted_sisdr
        stats = dict(fidelity_loss=total.detach().item(), pesq_loss=pesq.detach().item(),
                     negative_si_sdr_loss=sisdr.detach().item(),
                     weighted_pesq_loss=weighted_pesq.detach().item(),
                     weighted_si_sdr_loss=weighted_sisdr.detach().item(),
                     fidelity_reference_rms=reference_power.mean().sqrt().item(),
                     fidelity_prediction_rms=prediction_power.detach().mean().sqrt().item(),
                     fidelity_silent_references=(~voiced).sum().item(),
                     fidelity_pesq_skipped=(~pesq_valid).sum().item())
        return total, stats


def build_speech_fidelity(config, spectral, device):
    settings = fidelity_config(config)
    if settings is None:
        return None
    if torch.device(device).type not in ('cpu', 'cuda'):
        raise ValueError('paper1 speech fidelity requires CPU or CUDA, not MPS')
    if config['seconds'] < .5:
        raise ValueError('PESQ supervision requires training segments of at least 0.5 seconds')
    return SpeechFidelityLoss(spectral, settings).to(device)
