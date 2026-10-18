"""20 ms causal frames, 10 ms hop; complex compression with reversible WOLA.
CPU FFT preprocessing avoids the MPS complex-FFT limitation. Network tensors
are real [batch,2,frequency,frame] and may subsequently move to any device.
"""
import torch
import torch.nn.functional as F

class Spectral:
    def __init__(self, sample_rate=16000, window_ms=20, hop_ms=10, compression=.3):
        self.sr = sample_rate
        self.window_size = round(sample_rate * window_ms / 1000)
        self.hop = round(sample_rate * hop_ms / 1000)
        self.compression = compression
        self.window = torch.hann_window(self.window_size).sqrt()
        # sum(abs(window)) bounds every bin by one for abs(samples)<=1.
        # Paper calls this power normalization but does not specify the divisor.
        self.scale = self.window.sum()
    def encode(self, wave):
        if wave.ndim == 1: wave = wave[None]
        wave = wave.cpu()
        left = self.window_size - self.hop
        extra = (-wave.shape[-1]) % self.hop
        padded = F.pad(wave, (left, extra + self.window_size))
        frames = padded.unfold(-1, self.window_size, self.hop)
        z = torch.fft.rfft(frames * self.window, dim=-1) / self.scale
        mag = z.abs()
        z = z * mag.clamp_min(1e-10).pow(self.compression - 1)
        return torch.view_as_real(z).permute(0,3,2,1).contiguous()
    def decode(self, data, length):
        data = data.detach().cpu()
        z = torch.view_as_complex(data.permute(0,3,2,1).contiguous())
        z = z * z.abs().clamp_min(1e-10).pow(1/self.compression - 1)
        frames = torch.fft.irfft(z * self.scale, n=self.window_size, dim=-1) * self.window
        b, count, _ = frames.shape
        total = (count-1)*self.hop + self.window_size
        output = frames.new_zeros(b,total)
        weight = frames.new_zeros(total)
        for i in range(count):
            p = i*self.hop
            output[:,p:p+self.window_size] += frames[:,i]
            weight[p:p+self.window_size] += self.window.square()
        output /= weight.clamp_min(1e-8)
        start = self.window_size-self.hop
        return output[:,start:start+length]

    def decode_differentiable(self, data, length):
        """Same inverse/WOLA as decode, retaining gradients on CPU or CUDA.

        The inference decoder intentionally detaches and moves to CPU. This
        separate path is for waveform training losses, not playback export.
        """
        if data.ndim != 4 or data.shape[1:3] != (2, self.window_size // 2 + 1):
            raise ValueError('Expected [batch,2,frequency,frame] coefficients')
        if data.device.type not in ('cpu', 'cuda'):
            raise ValueError('Differentiable spectral decoding requires CPU or CUDA')
        if data.dtype not in (torch.float32, torch.float64):
            data = data.float()
        window = self.window.to(device=data.device, dtype=data.dtype)
        scale = self.scale.to(device=data.device, dtype=data.dtype)
        z = torch.view_as_complex(data.permute(0,3,2,1).contiguous())
        z = z * z.abs().clamp_min(1e-10).pow(1 / self.compression - 1)
        frames = torch.fft.irfft(z * scale, n=self.window_size, dim=-1) * window
        count = frames.shape[1]
        total = (count - 1) * self.hop + self.window_size
        start = self.window_size - self.hop
        if not isinstance(length, int) or length < 1 or start + length > total:
            raise ValueError('Invalid waveform length for supplied spectral frames')
        fold_args = dict(output_size=(1, total), kernel_size=(1, self.window_size),
                         stride=(1, self.hop))
        output = F.fold(frames.transpose(1, 2), **fold_args)[:, 0, 0]
        weights = window.square()[None, :, None].expand(1, -1, count)
        weight = F.fold(weights, **fold_args)[0, 0, 0]
        return (output / weight.clamp_min(1e-8))[:, start:start + length]
