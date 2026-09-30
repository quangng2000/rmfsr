# Architecture correction and educated study

The new default (`mirror-v2`) outputs decoder widths **[256,256,128,64,64]**, the reverse of encoder widths [64,64,128,256,256]. It replaces the old shifted sequence [256,128,64,64,64]. Skip mappings and attention widths follow the corrected stages. This is the literal stage-output reading of [paper section 3.2](https://arxiv.org/html/2605.16251v1#S3.SS2); exact author tensors and source are unavailable.

## Measured counts

[architecture-audit.json](architecture-audit.json) records forward-hook measurements at batch one, 16 kHz, 161 frequency bins and 400 frames (four seconds). MAC counts include convolutions, linear layers and attention matrix products; they exclude nonlinear functions, norms, memory traffic and audio transforms. These are counts, not speed or restoration-quality measurements.

| Variant | Parameters | GMAC/audio-second/NFE | Status |
|---|---:|---:|---|
| Mirrored default, decoder processes after upsampling | 6,112,710 | 7.156 | Default for new training |
| TCN attention, after upsampling | 7,431,622 | 7.792 | Capacity ablation |
| Mirrored decoder processes before upsampling | 6,112,710 | 4.837 | Efficiency ablation |
| Both optional changes | 7,431,622 | 5.473 | Educated candidate in `configs/architecture-study.json` |
| Published RMFSR | About 7.8M | 1.22 | Author result; not measured with our profiler |

Mirroring adds 691,904 parameters (12.8%) over the historical 5,420,806 model. The corrected default remains about 1.69M parameters below the paper's rounded count and has substantially higher compute. No inert tensors, arbitrary widening, or extra blocks are used to hit a count. Changing attention head count at fixed channel width or convolution dilation does not recover missing parameters.

## Why these guesses are reasonable

The paper states frequency attention, four TCN layers and mirrored decoder widths, but does not give exact attention placement or decoder resampling order. These two options explore those ambiguities while preserving all five requested stage widths, four temporal kernels, 2× depthwise expansion, time conditioning and causal temporal context. They are our experiments, not recovered author settings.

1. **Capacity at low frequency resolution:** `bottleneck_attention=true` inserts frame-local frequency attention after each of the four TCN blocks. At the six-bin bottleneck it adds 1,318,912 active parameters and 0.637 GMAC/s. The resulting 7.432M model is close in size to the paper, but closeness does not imply quality or architecture equivalence.
2. **Less work in the decoder:** `decoder_before_upsample=true` processes each decoder block and its attention at the corresponding encoder resolution, then upsamples. The mirrored default spends 2.555 GMAC/s in decoder convolution and 1.992 in decoder attention. This order change reduces total counted work by about 32% at unchanged parameter count. It changes the learned computation and must be trained/evaluated separately.

The combined candidate uses about 23.5% fewer counted MACs than the corrected default while adding capacity. It is still about 4.5× the paper's reported compute. Investigate exact author attention, block wiring, resampling and spectral resolution before claiming efficiency parity. A framework change alone cannot close a mathematical MAC-count gap.

## How we aim for useful restoration

Retain the mirrored default as a control. Screen all four choices with the same verified EARS/DNS/DAPS inputs, speaker splits, corruption/noise seeds, effective batch 16 and schedule. In separate run directories, qualify numerical behavior and CUDA memory/timing first, then use matched short learning runs to select one candidate for the proposed 300,000-update run. The existing 10k/50k/100k/200k/300k reviews inspect stability and held-out restoration. A short run is an early learning diagnostic, not evidence of convergence. Four complete 300k runs would require a separate compute budget; the current adviser proposal budgets one primary full run, not four.

Our educated starting budget remains 300k updates: 4.8M augmented four-second crops, or 5,333 mixture-duration hours. This improves the sampling coverage of expected loss; it does not determine achievable MOS or guarantee author performance. Preserve the diagonal-to-MeanFlow curriculum and EMA, and do not extend the budget solely because training loss declines. Choose the candidate using held-out quality and cost.

Evaluate NFE 1/2/5 on 10–80 ms gaps, codecs and bandwidth damage, and intact speech. Use waveform and spectral gap error, STOI and listening together. Keep 120 ms gaps as an out-of-range stress test. Check transcript errors and invented syllables; target the paper's SIG2024, DistillMOS, DNSMOS SIG and listening protocol for a later publication-level comparison. More steps or parameters alone cannot establish matching restoration. Full datasets, long training and those evaluations remain pending.

## Run and checkpoint isolation

```bash
# Forward-only counts; does not train.
python -m rmfsr.profile --inventory-only --device cpu --output runs/default-inventory.json
python -m rmfsr.profile --inventory-only --device cpu --bottleneck-attention --decoder-before-upsample --output runs/study-inventory.json

# Only after complete data preparation and GPU qualification.
python -m rmfsr.train --config configs/runpod-estimate.json --run runs/mirrored-control
python -m rmfsr.train --config configs/architecture-study.json --run runs/architecture-study
```

Set either optional flag independently in a copied config to run the individual ablations. Existing full/estimate configs retain the corrected mirrored default. Checkpoints and `run.json` record the architecture options; resume rejects older or different wiring before loading weights. Original pilot weights are still readable through `evaluate.py` with `legacy-v1` wiring, but cannot seed a new mirrored training run. All published historical pilot results and old Mac timing remain labeled as historical. No paid GPU or full training was started by this change.

PyTorch compilation, a JAX comparison and targeted Triton fusion are possible runtime studies after CUDA profiling; see [TRAINING.md](TRAINING.md).
