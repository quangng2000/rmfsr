"""Independent implementations of the paper's degradation families.
Probabilities/parameters not reported by authors are experimental assumptions.
"""
import math, pathlib, subprocess
import numpy as np
from scipy import signal

def rms(x):return float(np.sqrt(np.mean(x*x)+1e-12))
def level(x,db):return x*(10**(db/20)/max(rms(x),1e-6))

def image_rir(rng,sr,room=None,source=None,mic=None,order=2,reflection=None):
    """Finite shoebox image-source RIR (fractional-delay linear interpolation).
    Same-room callers share dimensions/mic; each has its own source position.
    Reflection order 2 is a deliberate low-cost approximation, not pyroomacoustics.
    """
    if room is None:room=rng.uniform([3,3,2.4],[9,8,4])
    if mic is None:mic=rng.uniform(.2,room-.2)
    if source is None:source=rng.uniform(.2,room-.2)
    direct_distance=np.linalg.norm(source-mic)
    if reflection is None:reflection=rng.uniform(.3,.85)
    impulse=np.zeros(round(sr*.7),np.float32)
    direct=np.zeros_like(impulse)
    for ix in range(-order,order+1):
      for iy in range(-order,order+1):
       for iz in range(-order,order+1):
        idx=np.array([ix,iy,iz]);n=int(abs(idx).sum())
        if n>order:continue
        image=idx*room+np.where(idx%2==0,source,room-source)
        distance=max(float(np.linalg.norm(image-mic)),.1)
        delay=distance/343*sr;start=int(delay);frac=delay-start
        amplitude=reflection**n/distance
        if start+1<len(impulse):
            impulse[start:start+2]+=amplitude*np.array([1-frac,frac])
            if n==0:direct[start:start+2]+=amplitude*np.array([1-frac,frac])
    return impulse,direct,room,mic,direct_distance

def spectral_augment(x,rng,sr):
    """Broad per-source EQ; +/-6 dB is a reproduction choice, not a paper setting.
    Symmetric offline FIR with compensated delay keeps target/input timing aligned.
    """
    frequencies=np.array([0,125,500,2000,sr/2],dtype=float)
    gains=10**(rng.uniform(-6,6,len(frequencies))/20)
    taps=signal.firwin2(65,frequencies,gains,fs=sr)
    return signal.fftconvolve(x,taps,mode='same')

def bandlimit(x,rng,sr):
    lo=rng.uniform(100,800);hi=rng.uniform(1500,.49*sr)
    kind=rng.choice(['butter','cheby1','cheby2','ellip','bessel','fir'])
    if kind=='fir':return signal.lfilter(signal.firwin(65,[lo,hi],pass_zero=False,fs=sr),[1],x)
    args=dict(N=int(rng.integers(2,6)),Wn=[lo,hi],btype='bandpass',fs=sr,output='sos')
    if kind in ('cheby1','ellip'):args['rp']=rng.uniform(.5,3)
    if kind in ('cheby2','ellip'):args['rs']=rng.uniform(15,40)
    return signal.sosfilt(getattr(signal,kind)(**args),x)

def codec_roundtrip(x,rng,sr,ffmpeg):
    """Decode to original sample rate. Trim codec delay via bounded correlation.
    Training alignment may use clean input; this is never used to align evaluation.
    """
    if not ffmpeg:raise RuntimeError('Codec augmentation requires ffmpeg')
    gsm=rng.random()<.5
    if gsm:
        from .gsm import roundtrip
        decoded=roundtrip(x,sr)
        corr=signal.correlate(decoded,x,mode='full',method='fft');lags=signal.correlation_lags(len(decoded),len(x))
        valid=(lags>=0)&(lags<min(sr//5,len(decoded)));delay=int(lags[valid][np.argmax(corr[valid])])
        return np.pad(decoded[delay:],(0,len(x)))[:len(x)]
    fmt='mp3'
    encoder=['-ar','8000','-c:a','libgsm'] if gsm else ['-c:a','libmp3lame','-b:a',str(int(rng.choice([16,24,32,48,64])))+'k']
    cmd=[ffmpeg,'-v','error','-f','f32le','-ar',str(sr),'-ac','1','-i','pipe:0']+encoder+['-f',fmt,'pipe:1']
    raw=subprocess.run(cmd,input=np.asarray(x,dtype='<f4').tobytes(),capture_output=True,check=True,timeout=20).stdout
    cmd=[ffmpeg,'-v','error','-f',fmt,'-i','pipe:0','-ar',str(sr),'-ac','1','-f','f32le','pipe:1']
    decoded=np.frombuffer(subprocess.run(cmd,input=raw,capture_output=True,check=True,timeout=20).stdout,dtype='<f4')
    corr=signal.correlate(decoded,x,mode='full',method='fft');lags=signal.correlation_lags(len(decoded),len(x))
    valid=(lags>=0)&(lags<min(sr//5,len(decoded)));delay=int(lags[valid][np.argmax(corr[valid])])
    return np.pad(decoded[delay:],(0,len(x)))[:len(x)]

def corrupt(x,rng,sr,ffmpeg=None,codec_enabled=True,extra_quantization=False):
    x=np.asarray(x,np.float64).copy();applied=[]
    if rng.random()<.55:x=bandlimit(x,rng,sr);applied.append('bandlimit')
    if rng.random()<.15:
        b,a=signal.iirnotch(rng.uniform(100,6000),rng.uniform(1,15),sr);x=signal.lfilter(b,a,x);applied.append('notch')
    if rng.random()<.3:
        drive=rng.uniform(1,8);scale=max(rms(x),1e-5);z=x/scale;kind=int(rng.integers(3))
        x=scale*(np.tanh(drive*z)/drive if kind==0 else np.maximum(z,-.2) if kind==1 else np.clip(z,-.4,.4));applied.append('nonlinear')
    if rng.random()<.15:
        x*=1+rng.uniform(.1,.9)*np.sin(2*np.pi*rng.uniform(1,40)*np.arange(len(x))/sr);applied.append('amplitude_modulation')
    if rng.random()<.15:
        a=rng.uniform(-.9,.9);x=signal.lfilter([a,1],[1,a],x);applied.append('phase_allpass')
    if rng.random()<.3:
        _,_,z=signal.stft(x,sr,nperseg=320,noverlap=160)
        if rng.random()<.5:
            for _ in range(int(rng.integers(1,5))):
                f=int(rng.integers(z.shape[0]));t=int(rng.integers(z.shape[1]));z[max(0,f-8):f+8,max(0,t-3):t+3]=0
            applied.append('spectral_bubbles')
        else:
            floor=np.quantile(abs(z),.25,axis=1,keepdims=True)
            z*=np.clip(1-rng.uniform(1.5,4)*floor/np.maximum(abs(z),1e-8),0,1);applied.append('noise_suppression')
        _,x2=signal.istft(z,sr,nperseg=320,noverlap=160);x=x2[:len(x)]
    if codec_enabled and rng.random()<.15:x=codec_roundtrip(x,rng,sr,ffmpeg);applied.append('codec')
    if extra_quantization and rng.random()<.1:
        bits=int(rng.integers(4,13));scale=max(float(np.max(abs(x))),1e-6)
        steps=2**(bits-1)-1;x=np.round(x/scale*steps)/steps*scale;applied.append('quantization')
    mask=np.zeros(len(x),bool)
    if rng.random()<.65:
        for _ in range(int(rng.integers(1,6))):
            length=round(rng.uniform(.01,.08)*sr);start=int(rng.integers(max(1,len(x)-length)))
            mask[start:start+length]=True
        x[mask]=0;applied.append('dropouts_10_80ms')
    return np.nan_to_num(x).astype(np.float32),mask,applied

def target_process(x,sr,reference_ltas=None):
    if reference_ltas is not None:
        # make_ltas normalizes each DAPS recording to -25 dBFS before computing
        # reference power. Match that level before EQ so gain limits constrain
        # spectral shape rather than arbitrary recording-volume differences.
        x=level(np.asarray(x,np.float64),-25)
        _,_,z=signal.stft(x,sr,nperseg=320,noverlap=160)
        power=(abs(z)**2).mean(axis=1)+1e-10
        # Smooth gain and constrain EQ to +/-6 dB (unspecified in paper).
        ratio=np.sqrt(reference_ltas/(power+1e-10));ratio=signal.savgol_filter(ratio,15,2).clip(.5,2)
        _,restored=signal.istft(z*ratio[:,None],sr,nperseg=320,noverlap=160);x=restored[:len(x)]
    x=level(x,-25)
    # Mild static soft-knee compression; paper provides no compressor settings.
    threshold=10**(-12/20);mag=np.abs(x)
    x=np.sign(x)*np.where(mag>threshold,threshold*(mag/threshold)**.75,mag)
    return level(x,-25).astype(np.float32)
