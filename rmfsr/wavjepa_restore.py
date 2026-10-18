"""Offline inference for a trusted conditioned checkpoint, with raw WAV export."""
import argparse
from pathlib import Path

import soundfile as sf
import torch

from .flow import pink_noise_like, sample
from .model import model_from_checkpoint
from .spectral import Spectral
from .wavjepa import build_conditioner


def restore(checkpoint_path, input_path, output_path, nfe=20, seed=1510001, device='cpu'):
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError('Output already exists; choose a new path')
    wave, rate = sf.read(input_path, dtype='float32')
    if wave.ndim != 1 or rate != 16000:
        raise ValueError('Expected mono 16 kHz input')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint['config'].get('wavjepa') is None:
        raise ValueError('Use a WavJEPA-conditioned checkpoint')
    model = model_from_checkpoint(checkpoint).to(device).eval()
    model.load_state_dict(checkpoint['ema'], strict=True)
    encoder = build_conditioner(checkpoint['config'], device)
    spectral = Spectral(rate)
    damaged = torch.from_numpy(wave)[None]
    y = spectral.encode(damaged).to(device)
    conditioning = encoder(damaged, spectral, y.shape[-1])
    noise = pink_noise_like(y.cpu(), torch.Generator().manual_seed(seed)).to(device)
    restored = sample(model, y, steps=nfe, noise=noise, conditioning=conditioning)
    result = spectral.decode(restored, len(wave))[0]
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError('Nonfinite restoration')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, result.numpy(), rate, subtype='FLOAT')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--nfe', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1510001)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    restore(args.checkpoint, args.input, args.output, args.nfe, args.seed, args.device)
