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
