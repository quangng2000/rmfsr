import argparse,json,time,statistics
import torch
from .model import RMFSR,FrequencyAttention
from .flow import meanflow_loss
from .train import device_for

def profile(device='auto',batch=1,seconds=4,repeat=5):
    torch.set_num_threads(8);device=device_for(device);model=RMFSR().to(device)
    x=torch.randn(batch,2,161,round(seconds*100),device=device)*.05;y=x+.02*torch.randn_like(x)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);durations=[]
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    for i in range(repeat+1):
        start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        # Always include the expensive off-diagonal JVP case.
        loss,_=meanflow_loss(model,x,y,t=x.new_full((batch,),.6),r=x.new_full((batch,),.2))
        loss.backward();optimizer.step()
        if device.type=='mps':torch.mps.synchronize()
        elif device.type=='cuda':torch.cuda.synchronize()
        if i:durations.append(time.perf_counter()-start)
    macs=[0];hooks=[]
    def conv_hook(m,args,out):macs[0]+=out.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
    def linear_hook(m,args,out):macs[0]+=out.numel()*m.in_features
    def attn_hook(m,args,out):
        b,c,f,n=out.shape;macs[0]+=2*b*n*f*f*c
    for m in model.modules():
        if isinstance(m,torch.nn.Conv2d):hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m,torch.nn.Linear):hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m,FrequencyAttention):hooks.append(m.register_forward_hook(attn_hook))
    with torch.no_grad():model(x[:1],y[:1],x.new_tensor([.5]),x.new_tensor([.3]))
    for hook in hooks:hook.remove()
    return dict(device=str(device),gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
                torch_version=torch.__version__,precision='float32',
                peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3 if device.type=='cuda' else None,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3 if device.type=='cuda' else None,
                batch=batch,seconds=seconds,parameters=sum(p.numel() for p in model.parameters()),
                training_step_seconds=durations,median_seconds=statistics.median(durations),
                estimated_300k_step_days=statistics.median(durations)*300000/86400,
                gmac_per_audio_second_per_evaluation=macs[0]/seconds/1e9,
                mac_note='Conv, Linear, attention matmuls; excludes elementwise ops, STFT, and augmentation')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='auto');p.add_argument('--batch',type=int,default=1)
    p.add_argument('--seconds',type=float,default=4);p.add_argument('--repeat',type=int,default=5);p.add_argument('--output',required=True)
    a=p.parse_args();r=profile(a.device,a.batch,a.seconds,a.repeat)
    from pathlib import Path
    Path(a.output).write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
