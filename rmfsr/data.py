import json,math,pathlib
from collections import OrderedDict
import numpy as np
import soundfile as sf
import torch
from scipy import signal
from .degradations import image_rir,level,rms,corrupt,target_process,spectral_augment
from .paths import load_manifest

def read_audio(path,sr):
    wave,rate=sf.read(path,dtype='float32',always_2d=True);wave=wave.mean(1)
    if sr!=rate:
        gcd=math.gcd(sr,rate);wave=signal.resample_poly(wave,sr//gcd,rate//gcd)
    return wave

class SpeechPairs:
    def __init__(self,manifest,seconds=4,sr=16000,seed=42,noise_dir=None,ltas_path=None,ffmpeg=None,pilot=False,augmentation_profile='legacy'):
        self.rows=load_manifest(manifest);self.n=round(seconds*sr);self.sr=sr
        self.rng=np.random.default_rng(seed);self.ffmpeg=ffmpeg;self.pilot=pilot
        if augmentation_profile not in ('legacy','figure2-v2'):raise ValueError('Unknown augmentation profile')
        self.figure2=augmentation_profile=='figure2-v2'
        self.files=[r['path'] for r in self.rows if r['frames']>=self.n]
        if not self.files:raise ValueError('No audio long enough for training crop')
        self.noise=sorted(p for p in pathlib.Path(noise_dir).rglob('*') if p.suffix.lower() in ('.wav','.flac')) if noise_dir else []
        self.ltas=np.load(ltas_path) if ltas_path else None
        if not pilot and (not self.noise or self.ltas is None or not ffmpeg):
            raise ValueError('Full data mode requires DNS noise, DAPS LTAS, and ffmpeg; no silent substitutes')
        self.cache=OrderedDict();self.cache_bytes=0;self.cache_limit=512*1024**2
    def segment(self,path):
        if path not in self.cache:
            wave=read_audio(path,self.sr)
            if not len(wave):raise ValueError(f'Empty audio: {path}')
            while self.cache and self.cache_bytes+wave.nbytes>self.cache_limit:
                _,old=self.cache.popitem(last=False);self.cache_bytes-=old.nbytes
            self.cache[path]=wave;self.cache_bytes+=wave.nbytes
        self.cache.move_to_end(path)
        x=self.cache[path]
        if len(x)<self.n:x=np.tile(x,math.ceil(self.n/max(1,len(x))))
        start=int(self.rng.integers(len(x)-self.n+1));return x[start:start+self.n].copy()
    def one(self):
        rng=self.rng
        room=rng.uniform([3,3,2.4],[9,8,4]);mic=rng.uniform(.2,room-.2)
        reflection=rng.uniform(.3,.85) if self.figure2 else None
        dry=np.zeros(self.n,np.float32);wet=np.zeros_like(dry)
        for talker in range(1+int(rng.random()<.2)):
            speech=self.segment(rng.choice(self.files))
            rir,direct,_,_,_=image_rir(rng,self.sr,room=room,mic=mic,reflection=reflection)
            gain=10**(rng.uniform(-6,0)/20)
            dry+=signal.fftconvolve(speech,direct)[:self.n]*gain
            reverberant=signal.fftconvolve(speech,rir)[:self.n]
            if self.figure2:reverberant=spectral_augment(reverberant,rng,self.sr)
            wet+=reverberant*gain
        if self.noise:noise=self.segment(rng.choice(self.noise))
        else:
            # Pilot only: explicitly not the DNS non-speech distribution.
            noise=signal.lfilter([1],[1,-.7],rng.normal(size=self.n)).astype(np.float32)
        if self.figure2:noise=spectral_augment(noise,rng,self.sr)
        snr=rng.normal(5,10);wet+=noise*(rms(wet)/(max(rms(noise),1e-6)*10**(snr/20)))
        wet=level(wet,rng.normal(-40,10))
        damaged,mask,kinds=corrupt(wet,rng,self.sr,self.ffmpeg,codec_enabled=bool(self.ffmpeg),extra_quantization=self.figure2)
        target=target_process(dry,self.sr,self.ltas)
        return target,damaged,mask,kinds
    def batch(self,size):
        rows=[self.one() for _ in range(size)]
        return torch.from_numpy(np.stack([r[0] for r in rows])),torch.from_numpy(np.stack([r[1] for r in rows]))

def assert_disjoint(*manifests):
    speakers=[]
    for path in manifests:
        rows=json.loads(pathlib.Path(path).read_text());current={r['speaker'] for r in rows}
        if any(current & prior for prior in speakers):raise ValueError('Speaker leakage across data splits')
        speakers.append(current)
