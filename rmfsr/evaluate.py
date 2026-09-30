"""Frozen-checkpoint audio evaluation. Never trains on comparison clips."""
import argparse,hashlib,json,pathlib,time
import numpy as np
import soundfile as sf
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pystoi import stoi
from .model import model_from_checkpoint
from .spectral import Spectral
from .flow import sample,pink_noise_like
from .train import device_for

def metrics(ref,est,missing,sr):
    ref=np.asarray(ref,np.float64);est=np.asarray(est,np.float64)
    error=est-ref
    centered=ref-ref.mean();pred=est-est.mean();project=centered*(pred@centered)/(centered@centered+1e-12)
    out=dict(stoi=float(stoi(ref,est,sr,extended=False)),rmse=float(np.sqrt(np.mean(error**2))),
             si_sdr=float(10*np.log10((project@project+1e-12)/(np.sum((pred-project)**2)+1e-12))))
    if missing.any():
        out.update(gap_rmse=float(np.sqrt(np.mean(error[missing]**2))),
                   gap_rms_ratio=float(np.sqrt(np.mean(est[missing]**2)/(np.mean(ref[missing]**2)+1e-12))))
        starts=np.flatnonzero(np.diff(np.r_[False,missing].astype(int))==1)
        ends=np.flatnonzero(np.diff(np.r_[missing,False].astype(int))==-1)+1
        out['gaps']=[]
        for start,end in zip(starts,ends):
            tail=(start+end)//2
            out['gaps'].append(dict(start_seconds=float(start/sr),end_seconds=float(end/sr),
                                   tail_rms_ratio=float(np.sqrt(np.mean(est[tail:end]**2)/(np.mean(ref[tail:end]**2)+1e-12))),
                                   rmse=float(np.sqrt(np.mean(error[start:end]**2)))))
    return out

@torch.no_grad()
def evaluate(checkpoint,baseline_dir,output,device='auto',steps=(1,2,5),chunk_frames=10):
    torch.set_num_threads(8);device=device_for(device);output=pathlib.Path(output);output.mkdir(parents=True,exist_ok=True)
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=ckpt['config']
    model=model_from_checkpoint(ckpt).to(device).eval();model.load_state_dict(ckpt['ema'])
    spec=Spectral(cfg['sample_rate']);report=dict(checkpoint=str(pathlib.Path(checkpoint).resolve()),
        checkpoint_sha256=hashlib.sha256(pathlib.Path(checkpoint).read_bytes()).hexdigest(),training_steps=ckpt['step'],
        pilot=cfg['pilot'],architecture=model.architecture(),quality_claim='Unvalidated experimental checkpoint; not a reproduction of published quality',
        device=str(device),chunk_frames=chunk_frames,results=[])
    for case_dir in sorted(pathlib.Path(baseline_dir).glob('opus_*ms')):
        if not case_dir.is_dir():continue
        ref,sr=sf.read(case_dir/'reference.wav',dtype='float32');damaged,_=sf.read(case_dir/'zero_filled.wav',dtype='float32')
        if sr!=cfg['sample_rate']:raise ValueError('Sample rate mismatch')
        mask=np.load(case_dir/'missing_mask.npy');y=spec.encode(torch.from_numpy(damaged)).to(device)
        gen=torch.Generator(device=device).manual_seed(903);noise=pink_noise_like(y,gen)
        audio={'reference':ref,'zero_filled':damaged}
        for name in ['opus_classic','opus_deep','tplc_s','tplc_l']:
            audio[name],rate=sf.read(case_dir/f'{name}.wav',dtype='float32')
            if rate!=sr or len(audio[name])!=len(ref):raise ValueError('Baseline alignment mismatch')
        timings={}
        for nfe in steps:
            start=time.perf_counter();prediction=sample(model,y,nfe,noise,chunk_frames=chunk_frames)
            if device.type=='mps':torch.mps.synchronize()
            elif device.type=='cuda':torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            wave=spec.decode(prediction,len(ref))[0].numpy()
            if not np.isfinite(wave).all():raise FloatingPointError('Nonfinite generated audio')
            name=f'rmfsr_nfe{nfe}';audio[name]=wave;timings[name]={'seconds':elapsed,'rtf':elapsed/(len(ref)/sr)}
        dest=output/case_dir.name;dest.mkdir(exist_ok=True)
        # One common playback gain for the whole comparison; no per-output normalization.
        peak=max(float(np.max(abs(x))) for x in audio.values());gain=min(1,.95/max(peak,1e-6))
        for name,wave in audio.items():
            sf.write(dest/f'{name}.wav',wave,sr,subtype='FLOAT')
            sf.write(dest/f'listen_{name}.wav',wave*gain,sr,subtype='PCM_16')
        result=dict(case=case_dir.name,playback_gain=gain,timing=timings,
                    metrics={name:metrics(ref,wave,mask,sr) for name,wave in audio.items() if name!='reference'})
        report['results'].append(result)
        if mask.any():
            start=np.flatnonzero(mask)[0];end=start
            while end<len(mask) and mask[end]:end+=1
            a=max(0,start-round(.04*sr));b=min(len(ref),end+round(.04*sr));axis=np.arange(a,b)/sr
            names=['reference','zero_filled','opus_deep','tplc_l',f'rmfsr_nfe{steps[-1]}']
            fig,axes=plt.subplots(len(names),1,figsize=(10,7),sharex=True,sharey=True)
            for ax,name in zip(axes,names):
                ax.plot(axis,audio[name][a:b],lw=.7);ax.axvspan(start/sr,end/sr,color='orange',alpha=.15);ax.set_ylabel(name,fontsize=8)
            axes[-1].set_xlabel('Audio time (seconds)');fig.suptitle(f'{case_dir.name}: first gap; RMFSR pilot only ({ckpt["step"]} steps)')
            fig.tight_layout();fig.savefig(dest/'gap_comparison.png',dpi=130);plt.close(fig)
        print(json.dumps({'case':case_dir.name,'metrics':result['metrics'][f'rmfsr_nfe{steps[-1]}']}),flush=True)
    (output/'report.json').write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--baseline-dir',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='auto');p.add_argument('--steps',type=int,nargs='+',default=[1,2,5]);p.add_argument('--chunk-frames',type=int,default=10)
    a=p.parse_args();evaluate(a.checkpoint,a.baseline_dir,a.output,a.device,a.steps,a.chunk_frames)
