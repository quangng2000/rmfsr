# RMFSR

This repository contains an **independent implementation attempt** and opt-in experimental architectures. It is not the authors' released model and its results have not been validated against the paper. No official weights or inference code were found in the [RMFSR demo repository](https://github.com/sebraun-msr/realtimemeanflowspeechrestoration). The source paper is [Braun, 2026, arXiv:2605.16251](https://arxiv.org/html/2605.16251v1).

## Current status, October 4, 2026

- Implemented the five-level causal U-Net, four-layer TCN, frequency attention, SnakeBeta, data-prediction MeanFlow loss, and separate streaming state for each flow step.
- Ran 100 training steps on the Mac **MPS GPU**, using one EARS training speaker and a different validation speaker. A third speaker is reserved for pilot testing. This is a plumbing/numerical pilot, not full training.
- The September 30 baseline passed 66 regression tests, covering training math, target EQ, verified datasets, actual restoration validation, causality, codecs, accumulation and checkpoint resume. A historical full-width (`legacy-v1`), effective-batch-16 MPS numerical smoke completed one update plus 1/2/5-step validation and review audio; this establishes execution, not restoration quality.
- Evaluated the checkpoint at 1/2/5 flow steps against the existing Opus/tPLCnet comparison inputs. **The pilot does not yet fill speech gaps**; its output is essentially the damaged input. Do not interpret this as evidence against the published method.
- New runs use decoder outputs **`[256,256,128,64,64]`** (`mirror-v2`). The 100-step pilot used the older shifted decoder; it remains a historical result and cannot resume into the new architecture.
- Added an optional capacity/efficiency study, with measured counts and a fair comparison plan in [ARCHITECTURE.md](ARCHITECTURE.md).
- Added `efficient-v1`, an experimental engineering variant aimed at the paper's compute budget. Its recommended configuration has 7,742,598 parameters and counts 0.911 GMAC/audio-second/NFE for four-second calls, or 1.192 for one-frame calls. These operation counts do not establish runtime or audio quality; the author architecture has not been recovered.
- All 76 tests pass after integration, including ten new checks for model selection, checkpoint round trips, legacy compatibility, compact attention accounting, causality, streaming, JVPs and a tiny save/resume/evaluation smoke test.
- The proposed full run has **not started**. Full dataset preparation is pending, and architecture parity remains unresolved.

## Paper-to-code audit

| Area | Implementation | Status |
|---|---|---|
| Backbone width | Encoder `[64,64,128,256,256]`; decoder outputs `[256,256,128,64,64]`; depthwise expansion 2 | Literal stage-output mirror; exact author tensors unavailable |
| Kernels | Encoder 3×3, decoder 3×2, TCN 1×11 | Matches stated kernels |
| Flow path | Degraded mean; sigma .3 → 1e-8; compressed complex spectrum, exponent .3 | Matches stated formulation |
| Training | Model-derived instantaneous JVP tangent; stop-gradient JVP; data-prediction loss | Independent equation implementation, analytically tested |
| Time sampling | Logit-normal mean .4, SD 1; sigmoid diagonal ratio and cosine span schedule | Schedule endpoints/shapes inferred where unspecified |
| Architecture size | **6,112,710 parameters**, measured **7.156 GMAC/audio-second/NFE** | Paper reports **7.8M / 1.22 GMAC/s**; substantial mismatch, not validated parity |
| Context | 218 frames; **2.17 s past context** at 10 ms hop | Paper reports 2.13 s; exact dilations unspecified |
| Spectrum | 16 kHz, 320-point FFT, sqrt-Hann, 20 ms window, 10 ms hop | Sample rate/hop/window details chosen for existing phone benchmark |
| Normalization | Divide FFT by window sum, inverse on synthesis | Explicit choice: bounds spectral magnitude for bounded audio; paper's “power normalization” divisor is unspecified |
| Attention/norm | Four-head, frame-local frequency attention; channel-only normalization; optional TCN attention study | Exact attention/normalization not specified by paper |
| Noise | Complex spectral Gaussian with 1/f energy, frequency-average unit variance | DC floor and absolute normalization are explicit choices |
| Optimizer/training length | AdamW, LR 1e-4, proposed 300k steps, EMA .999 | Our starting configuration; not reported paper hyperparameters |
| Initialization | Zero-initialized output residual around degraded input | Our stabilization choice; not reported by paper |
| Augmentation | Every listed family implemented; finite image-source RIR, local codec round trips, LTAS target EQ | Probabilities, reflection order, filter/compressor parameters and EQ limit are approximations |
| Pilot data | EARS p001 train, p002 validation, p003 test; synthetic colored background; no DAPS EQ | Explicit pilot deviations; cannot establish general restoration quality |
| Full data | EARS p001–p099 train, p100–p103 validation, p104–p107 test; DNS noise; DAPS produced LTAS | Proposed speaker-disjoint split; full assets not prepared |
| Paper evaluation | SIG2024, listening tests, MOS and WER | Not reproduced; current comparison is one held-out synthetic utterance |

The table describes the original mirrored backbone. The experimental `efficient-v1` family is a separate architecture study, selected explicitly below. Its active grouped layers, lower-rank attention and folded bottleneck change the computation; matching a parameter or MAC budget does not establish author parity. Do not quote the paper's compute or latency results as measurements of this implementation.

### Experimental efficient architecture

`configs/efficient.json` selects `model_type="efficient-v1"`, `groups=16`, `attention_rank=16` and `folded_width=448`. `configs/efficient-small.json` uses eight groups and folded width 256 for a smaller 4,532,742-parameter variant. Both retain the proposed full-data recipe from `runpod-estimate.json`: Figure 2 augmentation, the approximate Figure 1 flow schedule, four-second crops, effective batch 16 and 300,000 proposed updates. They choose the available device automatically. Existing configurations retain their original architecture.

```bash
# Forward-only operation counts; no dataset preparation or training.
mkdir -p runs
python -m rmfsr.profile --config configs/efficient.json --inventory-only --device cpu --seconds 4 --output runs/efficient-inventory-4s.json
python -m rmfsr.profile --config configs/efficient.json --inventory-only --device cpu --seconds 0.01 --output runs/efficient-inventory-10ms.json
python -m rmfsr.profile --config configs/efficient-small.json --inventory-only --device cpu --seconds 4 --output runs/efficient-small-inventory-4s.json

# After full EARS/DNS/DAPS preparation and device qualification; starts fresh training.
python -m rmfsr.preflight --config configs/efficient.json
python -m rmfsr.train --config configs/efficient.json --run runs/efficient
```

The small variant counts 0.779 GMAC/audio-second/NFE for four-second calls and 1.009 for one-frame calls. Long calls amortize conditioning computations across more frames, so quote the call length with each MAC count. One-frame counts still exclude framework overhead, normalization, nonlinear functions, memory traffic and the STFT. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design tradeoffs and comparison protocol.

These variants require new training and different weights. The checkpoint loader reconstructs the saved model type for evaluation; resume rejects a different architecture. To evaluate a newly trained checkpoint using the existing aligned comparison bundle:

```bash
python -m rmfsr.evaluate --checkpoint runs/efficient/best-validation.pt --baseline-dir /path/to/comparison-bundle --output runs/efficient/evaluation --steps 1 2 5 --chunk-frames 1
```

The comparison bundle must use the existing `opus_*ms` directory format with aligned references, damaged inputs, masks and baseline outputs. This command evaluates a checkpoint; it does not supply pretrained weights. No restoration-quality or measured real-time claim is supported for `efficient-v1`, and adding these configurations does not start training or provision compute.

### MeanFlow equations and choices

For clean data `x0`, degraded audio `y`, and spectral pink noise `e`:

```
xt = (1-t)*x0 + t*y + ((1-t)*sigma_min + t*sigma_max)*e
v_cond = y - x0 + (sigma_max-sigma_min)*e
u(xt,r,t) = (xt - model(xt,y,t,r)) / t
v_model = u(xt,t,t)
jvp = stopgrad(JVP(u; primals=(xt,r,t), tangent=(v_model,0,1)))
x_target = xt - t*v_cond + t*(t-r)*jvp
loss = mean((model(xt,y,t,r) - x_target)**2)
```

The `t²` weighting implicit in this x-space loss is the data-prediction interpretation of IMF. At `r=t`, it reduces to predicting `x0 + sigma_min*e`. The off-diagonal sign and model-derived tangent have an analytic regression test.

Paper Eq. 14 prints `U(0,t)^gamma` and then multiplies by `t` again. We interpret its span description using `r=t*U(0,1)^gamma`, making early samples concentrate near `r=t`. This is an explicit interpretation, not an exact transcription.

No GAN, pretrained vocoder, latent autoencoder, or clean target is used at inference. Evaluation does not train on or use the missing clean samples.

## Run locally

Clone and install from the repository root (Python 3.10+):

```bash
git clone https://github.com/quangng2000/rmfsr.git
cd rmfsr
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install jupyterlab nbformat ipykernel
python -m ipykernel install --user --name rmfsr --display-name "RMFSR (PyTorch)"
```

GSM needs `libgsm` (`brew install libgsm` on macOS; `sudo apt-get install libgsm1 libsndfile1` on Debian/Ubuntu). `imageio-ffmpeg` supplies the MP3 encoder. CUDA hosts should use the setup in [TRAINING.md](TRAINING.md).

Prepare a small, speaker-disjoint pilot and run it:

```bash
python -m unittest discover -s tests -v
python -m rmfsr.prepare --root data --split portable_pilot_train --speakers 1
python -m rmfsr.prepare --root data --split portable_pilot_validation --speakers 2
python -m rmfsr.prepare --root data --split portable_pilot_test --speakers 3
python -m rmfsr.preflight --config configs/pilot-figure2.json
python -m rmfsr.train --config configs/pilot-figure2.json --run runs/my-pilot
```

The pilot config selects Apple MPS. Set `device` to `cuda` or `cpu` in a copied config for other hosts. Data, checkpoints, generated audio and local results are excluded from Git. The notebook documents the original experiment; its result cells require the locally generated `runs/pilot` artifacts and external Opus/tPLCnet comparison files. A fresh clone does not include those results or pretrained weights. Open it with `jupyter lab rmfsr.ipynb`; use the commands above for a new pilot.

Checkpoint files include weights, EMA, optimizer, training step, data RNG, CPU/MPS/CUDA RNG, configuration, verified audio-content/partition fingerprints, and metric history. Checkpoints from before the verified-content format, corrected targets, or mirrored decoder require a fresh training run. Architecture flags are saved and checked before weights load; `evaluate.py` retains the original decoder when reading historical pilot weights. Only load trusted local checkpoint files. To extend training, increase `steps` in a copied config; preserve `schedule_steps` for a consistent schedule. Full mode cannot resume a pilot checkpoint as though it were full training.

## Full data preparation and training

**Not yet completed.** Fetch the [EARS dataset](https://github.com/facebookresearch/ears_dataset), [DNS non-speech noise](https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-5-noise-ir.sh), and [DAPS produced studio speech](https://zenodo.org/records/4660670). Do not use DAPS device recordings as the studio LTAS reference. Noise input must be the curated non-speech portion, not interfering speech.

The EARS helper processes one speaker archive at a time, verifies ZIP CRC, records SHA-256 provenance, resamples to 16 kHz FLAC, and removes only its own completed temporary archive. It reuses completed speaker conversions.

```bash
python -m rmfsr.prepare --root data --split train --speakers $(seq 1 99)
python -m rmfsr.prepare --root data --split validation --speakers $(seq 100 103)
python -m rmfsr.prepare --root data --split test --speakers $(seq 104 107)
python -m rmfsr.ltas --produced-dir /path/to/DAPS/produced --output data/daps_ltas.npy
# Put/extract curated DNS noise WAVs under data/dns_noise, or set noise_dir in config.
python -m rmfsr.preflight --config configs/full.json
python -m rmfsr.train --config configs/full.json --run runs/full
```

Full mode requires complete documented speaker splits, DNS noise, DAPS LTAS and codec tools. It fails instead of silently using pilot substitutes. Do not reuse the pilot p003 test results as held-out evaluation if a later full run includes p003 in training; use p104–p107 instead.

Raw downloads are large: EARS release assets total ~69 GB, DNS noise archives ~39 GB, and the complete DAPS archive is 16.1 GB. Use sequential conversion and a data volume sized for the converted datasets plus download headroom.

Historical `legacy-v1` measurement on this Mac: batch 4 × 4-second clips, full-width forward/JVP/backward/update, median **3.54 s/step** over three timed steps. For the original batch-4 configuration without gradient accumulation, extrapolation to 300k steps is **~12.3 days**, before augmentation, validation, I/O and thermal variability. This is a historical planning estimate, not timing for the corrected decoder or the paper's training requirement. Rebenchmark the chosen architecture before estimating GPU time. No paid cloud resources have been provisioned.

## Streaming and evaluation

`StreamingRestorer.push(samples)` / `.finish()` accepts arbitrary waveform chunks. It carries causal STFT, overlap-add synthesis, and separate convolution history for every flow step. No normalization uses future frames. The streaming tests compare both network outputs and waveform reconstruction against the same offline computation and identical prior noise.

`evaluate.py` uses 10-frame chunks (100 ms) by default for faster Python benchmarking; this batches input and adds buffering delay. This is **not** a 20 ms end-to-end latency measurement. The frame-by-frame streaming API avoids that batching, but has not demonstrated real-time hardware performance.

Saved comparisons include original, damaged, Opus Classic/Deep PLC, tPLCnet S/L, and RMFSR NFE 1/2/5. All receive the same audio and packet-loss mask. A common playback gain is used across outputs. Metrics include whole-clip STOI/SI-SDR plus gap RMSE, gap energy and second-half gap energy. Loudness alone is not correctness; phase-sensitive error alone is not perceptual quality. The 120 ms condition exceeds the paper's stated 10–80 ms training-dropout range and is labeled a stress test.

The current eager-Python/MPS NFE=5 comparison runs slower than real time. CUDA optimization, a statistically useful test set, gap listening tests, SIG2024/MOS/WER evaluation, and mobile deployment profiling remain future work. No claim that this checkpoint beats Opus or tPLCnet is supported.

## Data terms

EARS is **CC BY-NC 4.0**. Retain its attribution and license; these research checkpoints are not cleared for commercial deployment. DNS and DAPS have their own data terms. Downloaded audio is local and has not been uploaded or published.


## Cloud training preparation

See [TRAINING.md](TRAINING.md) for the Figure 2 implementation audit, portable EARS/DNS/DAPS preparation, Runpod GPU comparison, benchmark scripts, and checkpoint recovery. `configs/runpod-estimate.json` contains the latest proposed effective-batch-16 training recipe; `configs/runpod.json` retains the earlier batch-4 recipe. Both use `figure2-v2` augmentations. The training and benchmark scripts now default to the batch-16 estimate configs. Validation restores fixed damaged-only inputs at 1/2/5 steps, selects `best-validation.pt` by five-step waveform MSE, and saves gap/intact-region metrics plus review audio. Existing pilot results remain labeled as the legacy synthetic-noise pilot.
