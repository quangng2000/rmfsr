import argparse,json,time,statistics
import torch
from .model import RMFSR,FrequencyAttention,model_from_config
from .efficient import CompactFrequencyAttention
from .flow import meanflow_loss
from .train import device_for

def _profile_model(config, bottleneck_attention=False, decoder_before_upsample=False):
    if config is not None:
        if bottleneck_attention or decoder_before_upsample:
            raise ValueError('Architecture flags cannot be combined with a model config')
        if config.get('sample_rate', 16000) != 16000:
            raise ValueError('Profiling requires the 16 kHz, 161-bin spectral configuration')
        return model_from_config(config)
    return RMFSR(bottleneck_attention=bottleneck_attention,
                 decoder_before_upsample=decoder_before_upsample)


def profile(device='auto',batch=1,seconds=4,repeat=5,bottleneck_attention=False,decoder_before_upsample=False,config=None):
    """Time synthetic optimizer updates; config selects the model architecture.

    Use model_inventory (or --inventory-only) for forward-only operation counts.
    """
    frames = round(seconds * 100)
    if frames < 1:
        raise ValueError('seconds must cover at least one 10 ms frame')
    audio_seconds = frames / 100
    torch.set_num_threads(8);device=device_for(device)
    model=_profile_model(config,bottleneck_attention,decoder_before_upsample).to(device)
    x=torch.randn(batch,2,161,frames,device=device)*.05;y=x+.02*torch.randn_like(x)
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
    inventory = model_inventory(model, seconds=audio_seconds)
    return dict(device=str(device),gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
                torch_version=torch.__version__,precision='float32',
                peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3 if device.type=='cuda' else None,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3 if device.type=='cuda' else None,
                batch=batch,seconds=audio_seconds,requested_seconds=seconds,frames=frames,
                parameters=sum(p.numel() for p in model.parameters()),
                training_step_seconds=durations,median_seconds=statistics.median(durations),
                estimated_300k_step_days=statistics.median(durations)*300000/86400,
                gmac_per_audio_second_per_evaluation=inventory['gmac_per_audio_second_per_evaluation'],
                architecture=model.architecture(),
                mac_note='Conv, Linear, attention matmuls; excludes elementwise ops, STFT, and augmentation')


def model_inventory(model, seconds=4):
    """Forward-only tensor/MAC inventory; does not estimate runtime."""
    device = next(model.parameters()).device
    frames = round(seconds * 100)
    if frames < 1:
        raise ValueError('seconds must cover at least one 10 ms frame')
    audio_seconds = frames / 100
    x = torch.zeros(1, 2, 161, frames, device=device)
    y = torch.zeros_like(x)
    cache = {}
    macs={};hooks=[]
    def add(stage, value):
        macs[stage] = macs.get(stage, 0) + value
    def hook_for(stage, kind):
        def count(module, args, out):
            if kind == 'conv':
                value = out.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
            elif kind == 'linear':
                value = out.numel()*module.in_features
            else:
                b,c,f,n = out.shape
                # Compact attention returns outer channels, but its QK/AV
                # products operate on the smaller projected channel rank.
                channels = module.rank if isinstance(module, CompactFrequencyAttention) else c
                value = 2*b*n*f*f*channels
            add(stage, value)
        return count
    for name, module in model.named_modules():
        stage = name.split('.')[0]
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(hook_for(stage, 'conv')))
        elif isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(hook_for(stage, 'linear')))
        elif isinstance(module, (FrequencyAttention, CompactFrequencyAttention)):
            hooks.append(module.register_forward_hook(hook_for(stage, 'attention')))
    try:
        with torch.no_grad():
            model(x,y,x.new_tensor([.5]),x.new_tensor([.3]),cache)
    finally:
        for hook in hooks:
            hook.remove()
    parameters = sum(p.numel() for p in model.parameters())
    cache_elements = sum(state.numel() for state in cache.values())
    return dict(architecture=model.architecture(), parameters=parameters,
                decoder_output_channels=[block.project.conv.out_channels for block in model.decoder],
                sample_rate=16000, hop_ms=10, seconds=audio_seconds, requested_seconds=seconds, frames=frames,
                gmac_per_audio_second_per_evaluation=sum(macs.values())/audio_seconds/1e9,
                gmac_by_stage={stage: count/audio_seconds/1e9 for stage, count in macs.items()},
                cache_elements_per_evaluation=cache_elements,
                fp16_weights_mb=parameters*2/1e6,
                fp16_cache_mb_per_evaluation=cache_elements*2/1e6,
                note='Conv, Linear, attention matmuls only; excludes nonlinear ops, norms, memory traffic and STFT. No runtime measurement.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='auto');p.add_argument('--batch',type=int,default=1)
    p.add_argument('--config',help='JSON config selecting the model architecture')
    p.add_argument('--decoder-before-upsample',action='store_true');p.add_argument('--bottleneck-attention',action='store_true');p.add_argument('--inventory-only',action='store_true',help='Count forward operations without optimizer updates');p.add_argument('--seconds',type=float,default=4);p.add_argument('--repeat',type=int,default=5);p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.config and (a.decoder_before_upsample or a.bottleneck_attention):
        p.error('--config cannot be combined with --decoder-before-upsample or --bottleneck-attention')
    from pathlib import Path
    config=json.loads(Path(a.config).read_text()) if a.config else None
    if a.inventory_only:
        torch.set_num_threads(8)
        model=_profile_model(config,a.bottleneck_attention,a.decoder_before_upsample).to(device_for(a.device))
        r=model_inventory(model,a.seconds)
    else:
        r=profile(a.device,a.batch,a.seconds,a.repeat,a.bottleneck_attention,a.decoder_before_upsample,config=config)
    Path(a.output).write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
