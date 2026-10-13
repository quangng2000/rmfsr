# MeanFlow upstream code and speech adapter boundary

This directory reuses the MeanFlow authors' PyTorch implementation from
[Gsunshine/py-meanflow](https://github.com/Gsunshine/py-meanflow), pinned to
commit `1f6d72d94247c8fdeb89489acca1a8007a6baf6c`. The upstream README identifies
it as a PyTorch/GPU reimplementation of the CIFAR-10 experiments in
*Mean Flows for One-step Generative Modeling*. This is code reuse, not an
official RMFSR implementation or an endorsement by the MeanFlow authors.

The pinned PyTorch repository is licensed under **Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International**, not the MIT license.
The unchanged license is in [LICENSE](LICENSE). The extracted `core.py` and
its modifications are also provided under CC-BY-NC-SA-4.0. Upstream material
is supplied as-is, without warranties, as set out in that license.

## What is executed

| File | Source and local changes | Runtime use |
| --- | --- | --- |
| `time_sampler.py` | Byte-for-byte copy of upstream `meanflow/models/time_sampler.py` | `rmfsr.flow.sample_times` calls its `logit_normal_timestep_sample` |
| `core.py` | Extracted from `MeanFlow.forward_with_loss`, upstream `meanflow/models/meanflow.py` lines 51–57 | Every off-diagonal RMFSR loss calls its JVP and detached velocity-target calculation |
| `upstream/meanflow.py.txt` | Byte-for-byte original source for inspection and differential tests | Audit snapshot; not imported in production |
| `upstream/README.md` | Byte-for-byte original project description | Attribution and provenance; not imported |

[UPSTREAM.json](UPSTREAM.json) records the immutable commit and SHA-256 of
every unchanged upstream file. There is no run-time download, floating Git
dependency, image model, or pretrained image checkpoint in this integration.

## Changes to the extracted core

The original JVP and detached velocity-target statements remain together;
the surrounding image-specific network, noise sampling, EMA and adaptive
loss are not copied into a second training framework. The extraction:

1. Accepts the velocity callback, path state, times and conditional velocity
   as arguments so the existing speech model and trainer can call it.
2. Accepts an optional spatial `tangent`. With no override, this is the
   original conditional velocity `v`. The speech adapter supplies its
   model-derived instantaneous velocity to preserve its IMF objective.
3. Disables autocast on the input tensor's device rather than hard-coding
   CUDA. This supports RMFSR's CPU, MPS and CUDA execution paths.
4. Returns predicted and detached target velocities without imposing the
   image implementation's squared-L2 sum or adaptive loss weighting.

Default-core differential tests use the untouched upstream class and compare
its loss and parameter gradients. Speech tests separately check that the
explicit tangent override preserves the existing data-prediction objective.

## Speech-specific code that remains outside this directory

`rmfsr/flow.py` retains the degraded-audio path, spectral pink noise,
sigma schedule, conditioning on degraded spectra, and conversion between
the speech network's clean-data prediction and interval-average velocity.
The loss remains plain data-space MSE. Replacing it with the image model's
conditional tangent or adaptive velocity-space loss would change the
training objective and invalidate the intended checkpoint semantics.

The speech time sampler reuses the upstream logit-normal draw with mean
0.4 and standard deviation 1, but retains the minimum time 1e-4 and RMFSR's
progress-dependent diagonal/span schedules. The upstream independent
sampling of `r` is therefore not used. Both existing schedule choices,
including their documented approximations, are preserved.

An all-diagonal batch retains the exact speech target
`clean + sigma_min * noise`, avoiding an unnecessary JVP and cancellation
of large degraded terms. For other batches, the adapter converts the
upstream detached velocity target using `xt - t * velocity_target`.
This is the same loss algebra as before; floating-point operation ordering
can change rounding and is not claimed to be bitwise identical.

The offline and streaming samplers share `rmfsr.flow.restoration_step`.
Their multiple intervals, noisy degraded prior, and separate causal caches
for each interval remain speech integration code. The upstream one-step
image sampler has neither degraded-audio conditioning nor these caches.
No network parameters, checkpoint keys, optimizer, or EMA implementation
are introduced by this extraction.
