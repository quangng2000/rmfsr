import math
import torch

def expand(t): return t[:,None,None,None]

def pink_noise_like(x, generator=None):
    # Unit mean variance across bins; amplitude 1/sqrt(f), DC uses first-bin floor.
    f = torch.arange(x.shape[2], device=x.device, dtype=x.dtype).clamp_min(1)
    gain = f.rsqrt()
    gain = gain / gain.square().mean().sqrt()
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) * gain[None,None,:,None]

def schedule_values(progress, schedule='legacy'):
    p = min(max(progress,0),1)
    if schedule == 'legacy':
        gamma = .05 + .95*(1-math.cos(math.pi*p))/2
        diagonal_prob = .25 + .5/(1+math.exp(12*(p-.5)))
    elif schedule == 'figure1-cosine-approx':
        # Approximate the plotted milestones, not unpublished author code.
        # Figure 1: diagonal 1 -> .2 by 50%, span .05 -> 1 by 80%.
        diagonal_prob = .2 + .8*(1+math.cos(math.pi*min(p/.5,1)))/2
        gamma = .05 + .95*(1-math.cos(math.pi*min(p/.8,1)))/2
    else:
        raise ValueError(f'Unknown flow schedule: {schedule}')
    return diagonal_prob, gamma

def sample_times(batch, progress, device, schedule='legacy'):
    t = torch.sigmoid(torch.randn(batch,device=device)+.4).clamp_min(1e-4)
    diagonal_prob, gamma = schedule_values(progress, schedule)
    # Normalized span interpretation, explained in README (paper Eq14 ambiguous).
    r = t * torch.rand(batch,device=device).pow(gamma)
    r = torch.where(torch.rand(batch,device=device)<diagonal_prob,t,r)
    return t,r

def meanflow_loss(model, clean, degraded, progress=0, sigma_max=.3, sigma_min=1e-8, t=None, r=None, noise=None,schedule='legacy'):
    if t is None: t,r = sample_times(clean.shape[0],progress,clean.device,schedule)
    if noise is None: noise = pink_noise_like(clean)
    tt,rr = expand(t),expand(r)
    xt = (1-tt)*clean + tt*degraded + ((1-tt)*sigma_min + tt*sigma_max)*noise
    conditional_velocity = degraded-clean + (sigma_max-sigma_min)*noise
    with torch.no_grad():
        instantaneous = (xt-model(xt,degraded,t,t))/tt
        def velocity(z, end, start):
            return (z-model(z,degraded,start,end))/expand(start)
        _, jvp = torch.func.jvp(velocity, (xt,r,t),
                               (instantaneous,torch.zeros_like(r),torch.ones_like(t)))
        # t*(V-v_c) == target_data - predicted_data; JVP must be stop-gradient.
        target = xt - tt*conditional_velocity + tt*(tt-rr)*jvp
    prediction = model(xt,degraded,t,r)
    loss = (prediction-target.detach()).square().mean()
    return loss, {'dp_mse':(prediction.detach()-clean).square().mean().item(),
                  'target_rms':target.square().mean().sqrt().item()}

@torch.no_grad()
def sample(model, degraded, steps=5, noise=None, chunk_frames=None):
    if steps < 1: raise ValueError('steps must be positive')
    if noise is None: noise = pink_noise_like(degraded)
    def solve(y,e,caches=None):
        x = y + .3*e
        for i in range(steps):
            t = x.new_full((x.shape[0],),1-i/steps)
            r = x.new_full((x.shape[0],),1-(i+1)/steps)
            pred = model(x,y,t,r,cache=None if caches is None else caches[i])
            x = x - expand(t-r)*(x-pred)/expand(t)
        return x
    if chunk_frames is None: return solve(degraded,noise)
    caches = [{} for _ in range(steps)]
    return torch.cat([solve(degraded[...,i:i+chunk_frames],noise[...,i:i+chunk_frames],caches)
                      for i in range(0,degraded.shape[-1],chunk_frames)],-1)
