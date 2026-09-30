"""Bounded-state waveform streaming; no whole-utterance preprocessing.
Analysis and synthesis are CPU; the real-valued neural network runs on device.
"""
import numpy as np
import torch
from .spectral import Spectral
from .flow import pink_noise_like

class StreamingSTFT:
    def __init__(self,spec=None):
        self.spec=spec or Spectral();self.buffer=torch.zeros(self.spec.window_size-self.spec.hop)
        self.length=0
    def push(self,wave):
        wave=torch.as_tensor(wave,dtype=torch.float32).cpu().flatten();self.length+=len(wave)
        self.buffer=torch.cat((self.buffer,wave));frames=[]
        while self.buffer.numel()>=self.spec.window_size:
            frame=self.buffer[:self.spec.window_size]*self.spec.window
            z=torch.fft.rfft(frame)/self.spec.scale
            z=z*z.abs().clamp_min(1e-10).pow(self.spec.compression-1)
            frames.append(torch.view_as_real(z).T)
            self.buffer=self.buffer[self.spec.hop:]
        if not frames:return torch.empty(1,2,self.spec.window_size//2+1,0)
        return torch.stack(frames,-1)[None]
    def finish(self):
        n=self.length;result=self.push(torch.zeros((-n)%self.spec.hop+self.spec.window_size));self.length=n
        return result

class StreamingISTFT:
    def __init__(self,spec=None):
        self.spec=spec or Spectral();self.signal=torch.zeros(self.spec.window_size);self.weight=torch.zeros_like(self.signal)
        self.discard=self.spec.window_size-self.spec.hop
    def push(self,data):
        spec=self.spec;result=[]
        for frame in data.detach().cpu()[0].unbind(-1):
            z=torch.view_as_complex(frame.T.contiguous());z=z*z.abs().clamp_min(1e-10).pow(1/spec.compression-1)
            wave=torch.fft.irfft(z*spec.scale,n=spec.window_size)*spec.window
            self.signal+=wave;self.weight+=spec.window.square()
            ready=(self.signal[:spec.hop]/self.weight[:spec.hop].clamp_min(1e-8)).clone()
            self.signal=torch.cat((self.signal[spec.hop:],torch.zeros(spec.hop)))
            self.weight=torch.cat((self.weight[spec.hop:],torch.zeros(spec.hop)))
            discard=min(self.discard,len(ready));self.discard-=discard;ready=ready[discard:]
            if len(ready):result.append(ready)
        return torch.cat(result) if result else torch.empty(0)

class StreamingRestorer:
    def __init__(self,model,steps=5,sample_rate=16000,seed=903):
        if steps<1:raise ValueError('steps must be positive')
        self.model=model.eval();self.steps=steps;self.device=next(model.parameters()).device
        self.analysis=StreamingSTFT(Spectral(sample_rate));self.synthesis=StreamingISTFT(self.analysis.spec)
        self.caches=[{} for _ in range(steps)];self.generator=torch.Generator().manual_seed(seed)
        self.emitted=0;self.closed=False
    @torch.no_grad()
    def _frames(self,data):
        outputs=[]
        for frame in data.split(1,dim=-1):
            if frame.shape[-1]==0:continue
            y=frame.to(self.device);noise=pink_noise_like(frame,self.generator).to(self.device);x=y+.3*noise
            for i in range(self.steps):
                t=x.new_tensor([1-i/self.steps]);r=x.new_tensor([1-(i+1)/self.steps])
                prediction=self.model(x,y,t,r,cache=self.caches[i]);x=x-(t-r)*(x-prediction)/t
            outputs.append(self.synthesis.push(x))
        return torch.cat(outputs) if outputs else torch.empty(0)
    def push(self,wave):
        if self.closed:raise RuntimeError('Stream is finished')
        result=self._frames(self.analysis.push(wave));self.emitted+=len(result);return result
    def finish(self):
        if self.closed:raise RuntimeError('Stream is finished')
        self.closed=True;remaining=self.analysis.length-self.emitted
        return self._frames(self.analysis.finish())[:remaining]
