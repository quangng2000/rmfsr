# Architecture studies and experimental compute budget

The original mirrored control (`mirror-v2`) outputs decoder widths **[256,256,128,64,64]**, the reverse of encoder widths [64,64,128,256,256]. It replaces the old shifted sequence [256,128,64,64,64]. Skip mappings and attention widths follow the corrected stages. This is the literal stage-output reading of [paper section 3.2](https://arxiv.org/html/2605.16251v1#S3.SS2); exact author tensors and source are unavailable. Existing configurations still select this control or their explicitly configured ablation. The new `efficient-v1` models below are opt-in engineering experiments.

## Experimental efficient-v1

`efficient-v1` explores how to bring counted inference operations toward the paper's budget while preserving a causal five-stage spectral U-Net and data-prediction MeanFlow interface. It is a hypothesized engineering variant, not the recovered author architecture. It has no trained weights, validated restoration quality or measured real-time result.

The recommended configuration is [configs/efficient.json](configs/efficient.json): `model_type="efficient-v1"`, 16 groups, attention rank 16 and folded width 448. The smaller [configs/efficient-small.json](configs/efficient-small.json) uses eight groups and folded width 256. Both keep encoder widths [64,64,128,256,256], their mirrored decoder widths, 2× residual-block expansion and four temporal blocks. The configuration selector is handled by `rmfsr.model.model_from_config`; the implementation is `rmfsr.efficient.EfficientRMFSR`.

| Variant | Parameters | GMAC/audio-second/NFE, 4 s call | GMAC/audio-second/NFE, 10 ms call |
|---|---:|---:|---:|
| Efficient, 16 groups / width 448 | 7,742,598 | 0.910600064 | 1.192160000 |
| Efficient small, 8 groups / width 256 | 4,532,742 | 0.778861440 | 1.008940800 |
| Published RMFSR | About 7.8M | 1.22 reported, call length unspecified | Not reported separately |

These are forward-operation inventories at batch one and 161 frequency bins. They count convolutions, linear layers and attention matrix products, including the compact attention's projected rank. They exclude nonlinear functions, normalization, softmax, memory traffic, framework overhead, audio transforms and data augmentation. They do not measure wall-clock speed or training cost. The paper's counting protocol is not fully specified, so this is a budget comparison, not verified efficiency parity.

The call length matters: time-conditioning projections run once per network invocation, so a four-second call amortizes them over 400 frames. A one-frame call repeats that fixed work every 10 ms. The 10 ms column counts each invocation at one frame; it is neither a measured latency nor a hardware streaming benchmark. At multiple NFEs the model is evaluated repeatedly, and each flow step needs its own streaming state.

### Computation and tradeoffs

1. **Grouped stage mixing.** Encoder/decoder expansion and projection convolutions, shape-changing residual paths and skip mappings use grouped channel mixing. The effective group count divides both input and output widths. A channel shuffle follows expansion and SnakeBeta, before the depthwise convolution. This reduces expensive dense mixing, but each grouped layer can directly combine fewer channels. The shuffle and later dense projections offer communication across groups; quality still needs a matched training comparison.
2. **Compact frequency attention.** Attention remains after all five encoder and decoder stages and operates within each time frame. Dense projections map the current channel width to rank-16 queries, keys and values, with four heads, and project the result back. This preserves global frequency interaction at lower counted cost while constraining its representational rank. It is a deliberate departure from the control's full-width attention.
3. **Folded temporal bottleneck.** After frequency downsampling, six bins × 256 channels become 1,536 joint features at one frequency position. A conditioned projection maps these to width 448, followed by four dense inverted residual blocks with temporal kernel 11 and dilations [1,2,4,8]. A projection back to 1,536 features and reshape restores six × 256, with a residual connection. The small model uses width 256. These are active trainable layers, not unused tensors added to match a parameter total. Folding changes frequency weight sharing and compresses the joint representation; the resulting model is tied to the 161-bin input representation.
4. **Decoder work before upsampling.** Each decoder block and attention layer operates at the lower frequency resolution, then upsamples by nearest neighbor. A final depthwise frequency kernel (3,1), SnakeBeta and residual connection refine the full 161-bin output before the prediction head. This reduces high-resolution work, but may limit fine spectral detail relative to full decoder processing after upsampling. The refinement is a mitigation to evaluate, not evidence that the loss of capacity is harmless.

The model retains the existing channel-only normalization, Fourier time conditioning, causal temporal convolution and zero-initialized output residual around the degraded spectrum. It does not downsample time or add future-frame lookahead. Retaining this causal structure does not establish an end-to-end latency or real-time-performance result.

### Count, train and evaluate

```bash
mkdir -p runs
# These invocations count a forward pass and never update weights.
python -m rmfsr.profile --config configs/efficient.json --inventory-only --device cpu --seconds 4 --output runs/efficient-inventory-4s.json
python -m rmfsr.profile --config configs/efficient.json --inventory-only --device cpu --seconds 0.01 --output runs/efficient-inventory-10ms.json
python -m rmfsr.profile --config configs/efficient-small.json --inventory-only --device cpu --seconds 4 --output runs/efficient-small-inventory-4s.json
python -m rmfsr.profile --config configs/efficient-small.json --inventory-only --device cpu --seconds 0.01 --output runs/efficient-small-inventory-10ms.json

# After completing full data preparation and qualifying the device.
python -m rmfsr.preflight --config configs/efficient.json
python -m rmfsr.train --config configs/efficient.json --run runs/efficient
# Alternative separate fresh run; do not reuse the larger variant's weights.
python -m rmfsr.train --config configs/efficient-small.json --run runs/efficient-small

# After training, using the existing aligned opus_*ms comparison-bundle format.
python -m rmfsr.evaluate --checkpoint runs/efficient/best-validation.pt --baseline-dir /path/to/comparison-bundle --output runs/efficient/evaluation --steps 1 2 5 --chunk-frames 1
```

The two configurations inherit `runpod-estimate.json`'s full EARS/DNS/DAPS recipe, `figure2-v2` augmentation, `figure1-cosine-approx` schedule, four-second crops, effective batch 16 and proposed 300,000 updates; the device is `auto`. This schedule and update budget remain experimental choices. Merely adding the configurations does not prepare data, train a model or provision compute.

Each variant requires fresh training. Checkpoints record the architecture and the existing evaluation loader reconstructs it before loading weights. Resume rejects changes to the architecture, including model type, grouping, attention rank and folded width. Original checkpoints continue to use their original wiring; they cannot be loaded as efficient-model weights. Keep separate run directories, use the mirrored model as a control, and compare held-out restoration and listening results with identical data and corruption seeds. An operation-count target is only a screening criterion.

## Original mirrored-control counts

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
