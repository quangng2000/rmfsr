# RMFSR training preparation

This is an independent implementation of [RMFSR](https://arxiv.org/abs/2605.16251), not the authors' code or a pretrained RMFSR checkpoint. The existing 100-step pilot does not improve the packet-loss examples. The corrected mirrored decoder has 6.113M parameters and 7.156 GMAC/audio-second, compared with the paper's 7.8M and 1.22. The optional combined architecture study has 7.432M parameters and 5.473 GMAC/audio-second; neither matches the paper. See [ARCHITECTURE.md](ARCHITECTURE.md) for the hypotheses, measured 2×2 comparison and selection plan. The crop, batch, optimizer, 300,000-step budget, split, and unreported augmentation settings are our experiment choices.

## Author training duration: unresolved

Rechecked the paper PDF and rendered Figure 1 on 2026-09-30. The schedule plot has an x-axis labeled **Epoch**, extending from 0 to 10,000. This is a plotted schedule range, not an explicitly reported optimizer-update budget or confirmation of the duration of every reported training run. The paper does not specify batch size, examples/updates per epoch, or a total optimizer-step count, so 10,000 cannot be converted into training steps. Our 300,000 updates are an experimental setting, not a verified author setting or a guarantee of matching performance.

The plotted blue diagonal-sampling ratio starts near 1.0, falls to about 0.2 by epoch 5,000, and stays there; the orange span-shape curve reaches 1.0 around epoch 8,000. The selected `figure1-cosine-approx` configuration maps these endpoints to 50% and 80% of our update budget with smooth cosine interpolation. The earlier `legacy` schedule retains its 0.75-to-0.25 sigmoid for explicit legacy experiments. The paper's discussion of 75%/25% describes previous MF/IMF choices. Our endpoint approximation does not establish the authors' exact epoch definition or interpolation.

Primary evidence: [paper, Figure 1 and section 2.4](https://arxiv.org/html/2605.16251v1#S2.F1), [original vector figure](https://arxiv.org/html/2605.16251v1/r_schedule.svg). Needed author details: epoch definition, total updates, effective batch size, crop duration, optimizer/LR schedule, exact model configuration and augmentation settings. Judge quality on the reported evaluation protocol, not step count alone.

## Figure 2: what is implemented

On 2026-10-14 the user requested the balanced run to return from `demo-eight-v1`
to `figure2-v3-aligned` at its next natural six-hour session boundary. The queued
training recipe uses EARS, 1–2 talkers with separate direct/reverberant RIR paths,
DNS noise, the paper's stated noise/level distributions, the broad degradation
families below, and DAPS-based studio target processing. The existing validation
recipe is already `figure2-v3-aligned` and stays fixed. Resume retains weights,
EMA, AdamW, update count, learning-rate schedule, the previously activated flow
curriculum, and best-validation selection. It archives the pre-change latest and
best checkpoints. No signal, immediate trainer restart, corpus scan, benchmark,
controller replacement, pod restart or GPU release is required.

This is a Figure 2 reproduction, not the authors' original augmentation code.
Unreported effect probabilities, room/reflection settings, EQ/compressor settings
and split choices remain documented assumptions; the order-2 RIR and static
soft-knee compressor remain approximations. Noise/level normals interpret the
paper's second parameter as a 10 dB standard deviation. Bandpass upper cutoff
stops at 0.49 times the sample rate for numerical filter design near Nyquist.

| Block | Implementation and status |
|---|---|
| One or two clean talkers | EARS, sampled by `SpeechPairs.one`; two talkers in 20% of examples, a chosen probability. |
| Shared room / distinct RIRs | Same dimensions, microphone and reflection coefficient; different source position per talker. Finite order-2 image-source approximation, not an exact room simulator. |
| Direct RIR target branch | Direct path only, using the same per-talker gain and propagation delay as the degraded branch. |
| Per-source spectral/level augmentation | New `figure2-v2` profile applies broad EQ to each reverberant speech source and to noise; speech mixing gains, noise SNR and overall input level vary. EQ range and filter are our choices. |
| Studio processing | DAPS LTAS auto-EQ with matching -25 dBFS source/reference level before bounded EQ, mild compression, fixed -25 dBFS target level implemented. Actual DAPS corpus/LTAS must be prepared before full mode runs. |
| Microphone/recording chain | Six bandpass families, notch, static nonlinear distortions, level variation. Individual transfer functions are approximations. |
| Digital processing | MP3 and GSM encode/decode, suppression, spectral masks, phase/allpass, amplitude modulation, optional quantization, and 10–80 ms dropouts. |
| Noise source | DNS5 official noise archives; acquisition implemented and URLs checked. No additional classifier is used to screen residual speech in these archives. |

The previous pilot used the `legacy` profile, synthetic noise, and no DAPS EQ. The new full/cloud configurations use `figure2-v2`. Recorded old pilot outputs remain available locally. The corrected target processing and verified-content checkpoint format require a fresh run; checkpoints from before these fixes are rejected on resume. The mirrored decoder is a separate architecture change; old weights may be evaluated with their original wiring but cannot resume new training. These settings cover the diagram's functional blocks but cannot recreate unpublished author settings exactly.

## Data and storage

Only public datasets are used; no Workphone call recordings, logs or credentials are part of this bundle.

- [EARS](https://github.com/facebookresearch/ears_dataset): 107 speakers. Our full split is p001–p099 train, p100–p103 validation, p104–p107 test. This is an implementation choice, not an author-provided split. EARS is CC-BY-NC-4.0, so these checkpoints/data are for this noncommercial research experiment.
- [DNS5 noise download list](https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-5-noise-ir.sh): all nine noise archives. Files are deterministically assigned 90/5/5% train/validation/test by source-member hash. This is file-level separation, not verified source-recording-level separation. No DNS speech archive or measured RIR archive is required here.
- [DAPS](https://zenodo.org/records/4660670): select `produced` audio for studio LTAS only. Verify the published archive MD5; record SHA-256 and source members for provenance. It is not used as a speech training/test corpus.

Manifests use paths relative to their own directory. Archives download sequentially with resumable `.download` files, size checks, and atomic publication; only a successfully converted archive is removed. EARS ZIP CRC and DAPS published MD5 are checked. DNS SHA-256 values are recorded after download (no publisher checksum is claimed). Only audio members are decoded; archive paths are never extracted to the filesystem. Audio is mono 16 kHz FLAC. An 8 GiB free-space floor stops downloads/conversion safely.

Full source archives total approximately 125 GB of network transfer, but are not all retained simultaneously. This Mac has about 44 GiB free, so full acquisition is staged for a persistent data volume. The existing local EARS subset contains 3 speakers / 483 files / about 2.65 hours; that is a pilot, not the full training corpus. The new portable pilot manifests and augmentation examples are prepared locally.

Suggested volume: 200 GB persistent storage, plus a 30 GB container disk, at least 8 vCPUs and 32 GB host RAM (64 GB preferred). Prepare the full data on a CPU host/volume before renting a GPU where possible; downloading while a GPU sits idle still incurs GPU charges. A network volume must be in a data center where the selected GPU is available. Recheck capacity when provisioning.

## GPU choice and benchmark

Live Runpod catalog checked 2026-09-30, single GPU with a CUDA 12.8-compatible host:

| GPU | VRAM | Listed hourly GPU rate | Role |
|---|---:|---:|---|
| RTX 4090, Community | 24 GB | $0.34 | Lowest-cost candidate; training batch fit is not measured. Limited availability. |
| RTX 6000 Ada, Secure | 48 GB | $0.84 | Recommended first benchmark; extra activation/JVP memory headroom. Limited availability in US-KS-2. |
| L40S, Secure | 48 GB | $1.09 | Alternative if available in the chosen volume's region. |
| H100 SXM, Secure | 80 GB | $3.49 | Larger throughput/memory option; needs >4.15x Ada throughput to lower GPU cost per identical training step. |
| B200, Secure | 180 GB | $6.79 | Available catalog model; B100 was not listed. Needs >8.08x Ada throughput to lower GPU cost per identical step. |

Prices/availability vary; storage is extra. Source: live Runpod MCP catalog and [Runpod pricing](https://www.runpod.io/pricing). This is not a measured speed ranking. H100/B200 have more peak capability but this small convolutional model may be limited by utilization, forward-mode JVP, or CPU augmentation. Benchmark with the same batch, crop, precision and objective. A GPU with a newer compatible driver can run the CUDA 12.8 container.

Use official template `runpod-torch-v280`, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (verified in the current template catalog). Start with FP32. Do not assume BF16, compilation, or a pretrained CNN can be substituted without checking the MeanFlow derivatives and training quality.

Run `bash scripts/bootstrap-runpod.sh` inside the pod after unpacking this project at `/workspace/rmfsr`. It installs GSM/FFmpeg and the package, verifies PyTorch 2.8/CUDA, and runs tests. The Python environment defaults to `/opt/rmfsr-venv` on the local container disk to avoid network-volume metadata latency; set `RMFSR_VENV` consistently across scripts to override it. Re-run bootstrap after replacing the container. Data and checkpoints remain under `/workspace`. It does not create a pod.

```bash
# In a prepared Python environment, from the project root; can run on a CPU host:
bash scripts/prepare-data.sh

# On the GPU, after the data is complete:
bash scripts/benchmark-gpu.sh

# After reviewing benchmark results and authorizing the training budget:
bash scripts/train-runpod.sh
```

For an authorized continuous run, start `python -u -m rmfsr.continuous_train --config configs/efficient.json` from the prepared environment. It waits for a completed matching real-data qualification in `runs/qualification`, then starts a fresh full run under `runs/efficient`. Each six-hour checkpointed session automatically resumes until the configured update target is reached. It preserves a fixed copy of the configuration, rejects duplicate controllers with a file lock, and fails on unsuccessful or non-advancing sessions. It does not resume the benchmark checkpoint or stop the billed pod itself. The local supervisor is read-only: keep the GPU allocated through errors, stalls and completion, and release it only on explicit user instruction. Container or host restarts require restoring the environment and relaunching this controller against the persistent checkpoint.

### Prepare complete data before GPU rental

Mount the existing network volume on an authorized Linux CPU pod in the same data center. From `/workspace/rmfsr`, run `bash scripts/bootstrap-runpod.sh --cpu`, activate `/opt/rmfsr-venv/bin/activate`, then run `bash scripts/prepare-data.sh configs/efficient.json`. This reuses complete corpora and saved downloads, applies the DAPS metadata fix, and resumes remaining conversions. A shared-volume file lock prevents duplicate preparation jobs.

The CPU job must finish with `runs/data-preparation/status` equal to `ready`. `runs/data-readiness.json` verifies full EARS speaker partitions, all nine DNS archive manifests and audio hashes/headers, separate noise partitions, DAPS LTAS provenance, and codec dependencies. `runs/data-pipeline-check.json` exercises eight actual indexed training microbatches and two validation batches using real DNS noise, DAPS EQ, augmentation, codecs and spectral transforms at the configured crop/batch size. Eight input/target previews are saved under `runs/data-preparation/previews` for listening. Fixture tests do not establish actual corpus readiness, and the sample batches cannot exhaust every stochastic augmentation.

GPU qualification and benchmark launchers require both successful reports for the matching data recipe and a completed CPU job. They do not download or convert corpora. The trainer still verifies actual input integrity when starting. Passing CPU preparation establishes data readiness; GPU compatibility, memory use and update speed require qualification on the selected GPU. The 300,000-update target stays fixed.

The benchmark defaults to `configs/efficient-benchmark.json`: `efficient-v1`, 4-second crops, microbatch 4, four accumulation rounds. It first measures a complete synthetic off-diagonal JVP optimizer update (20 timed repeats). It records peak allocated/reserved CUDA memory and compute-only timing. It then runs 100 actual accumulated optimizer updates including data loading, augmentation, codecs, validation and checkpointing. Benchmark-only configs pin flow sampling at 90% schedule progress to exercise the late-training JVP workload; this override is rejected outside an explicitly marked benchmark. Both stages honor the same model config and accumulation count. `throughput.json` excludes ten warm-up updates and reports complete optimizer updates per second; individual update timings exclude validation/checkpointing, while the run wall time includes them. The launcher rejects long training configurations. Compare seconds/step, examples/second, peak memory and hourly-price × seconds/step / 3600. Reserve memory headroom; don't infer production throughput from the synthetic test alone. Local execution validates CPU/MPS logic, not CUDA performance.

## Educated-estimate recipe (separate configuration)

`configs/runpod-estimate.json` is a user-requested experimental estimate, not a claim about the authors' hidden configuration. Existing configurations and runs keep their previous settings.

- 300,000 **optimizer updates**, four-second crops, microbatch 4, four accumulation rounds: effective batch 16. Gradients are divided by the accumulation count; clipping, AdamW and EMA occur once per optimizer update. On resume, deterministic data indices advance in microbatches so no training samples are skipped or repeated due to prefetch.
- 4.8 million augmented crops; 300,000 × 16 × 4 / 3,600 = **5,333 hours of mixture-duration exposure**. This is roughly 53 duration-equivalents of a hypothetical 100-hour corpus, not 53 literal epochs, unique speech hours, or statistically independent observations. Our previous batch-4/300k budget was 1,333 hours; this proposal is four times the data exposure and roughly four times the microbatch work.
- AdamW at 1e-4, 5,000-update warmup, cosine decay to 1e-5; EMA .999; gradient norm clip 1. These are starting choices, not experimentally optimal settings.
- New `figure1-cosine-approx` schedule: diagonal-data-prediction probability falls smoothly from 1 to .2 during the first 50% of the update budget; span exponent rises from .05 to 1 by 80%. Smooth cosine interpolation approximates the figure's endpoints; it does not establish the authors' exact sigmoid, epoch definition, or schedule implementation. At the start, targets predominantly reduce to data prediction; later training introduces longer MeanFlow spans.
- Keep permanent quality-review checkpoints at 10k, 50k, 100k, 200k and 300k. At these updates the trainer also writes held-out restoration metrics and up to two audio comparisons under `runs/estimate/validation/step-NNNNNNNN/`. Every 1,000 updates, validation restores the same 32 examples from damaged audio plus fixed independent noise at 1/2/5 network evaluations. It reports waveform MSE, gap MSE, intact-region MSE, and STOI when enough nonsilent audio exists. `best-validation.pt` is selected by `validation_restoration_mse_nfe5`; the teacher-input diagonal MSE remains a separate diagnostic. Listening, transcript accuracy and hallucination assessment still require review.
- Consider an extension to 500k only if held-out quality is still improving. Keep the original 300k schedule and use its final learning rate for an extension, or design an explicitly separate continuation; resume rejects silent schedule changes.

Why this is reasonable: more independent corruptions improve the stochastic estimate of the expected training loss; accumulation reduces gradient sampling variance without requiring the entire effective batch in memory. It does not establish a power law from updates to MOS, eliminate model/augmentation approximation error, or guarantee published quality. Parameter count alone cannot determine the necessary number of updates. The default is the measured 6.113M-parameter mirrored version; `configs/architecture-study.json` opts into the 7.432M capacity/efficiency variant. Both keep the requested decoder widths and need fresh runs.

Runtime must be measured for the complete accumulated update. Total compute hours = updates × measured seconds/update / 3600; at 300k updates, 1 second/update means 83.3 hours and 2 seconds/update means 166.7 hours. These are conditional arithmetic examples, not H100/Ada speed predictions. Include data preparation, validation and checkpoint overhead separately.

On a prepared pod:

```bash
bash /workspace/rmfsr/scripts/benchmark-gpu.sh /workspace/rmfsr/configs/runpod-estimate-benchmark.json
bash /workspace/rmfsr/scripts/train-runpod.sh /workspace/rmfsr/configs/runpod-estimate.json /workspace/rmfsr/runs/estimate
```

The benchmark's synthetic stage now includes all configured accumulation rounds. Use its **real-data stage** to include augmentation, STFT, transfer and EMA overhead; this stage is still a throughput experiment, not a full quality-training run.

## Recent-iPhone inference estimate

A recent iPhone is a plausible deployment target after native conversion. This is an engineering estimate; the model has not been exported to Core ML or timed on an iPhone. The historical `legacy-v1` Python/MPS pilot took around twice audio duration at five evaluations on several comparison clips, so the current Python implementation does not demonstrate real-time operation.

Measured tensor inventory (16 kHz, batch one) gives 6,112,710 parameters and 930,432 persistent convolution-cache elements **per network evaluation index**. With FP16 storage: weights are 12.23 MB; one cache set is 1.86 MB; five sets are 9.30 MB. Around 21.53 MB covers shared weights plus those five state sets only; temporary activations, STFT/ISTFT, compiled model overhead and the host app need additional memory. Weight storage is not multiplied by the number of evaluation steps, provided the deployment shares weights.

| Network evaluations per audio chunk | Paper network, GMAC/audio-second | Our measured network, GMAC/audio-second |
|---|---:|---:|
| 1 | 1.22 | 7.16 |
| 2 | 2.44 | 14.31 |
| 5 | 6.10 | 35.78 |

Paper costs are from [Table 1](https://arxiv.org/html/2605.16251v1#S3.T1), scaled linearly by the evaluation count. Our figures count convolutions, linear layers and attention matrix products; exclude activation functions such as sine, normalization, memory traffic and audio transforms. Peak chip TOPS cannot be converted directly into achieved latency for this graph.

Deployment route: FP16 Core ML network with fixed streaming shapes and separate persistent state per flow step, native audio FFT outside the neural graph, no autograd/JVP during inference, serialized stream-state access. [Core ML stateful models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html) support state across predictions on iOS 18 and newer. Verify conversion and device placement for attention, normalization and SnakeBeta; full Neural Engine placement is not assumed. CPU/GPU fallback and invocation overhead may dominate a small graph.

A useful live-call engineering target is p95 total processing below 5 ms for each 10 ms hop (RTF below .5), with no sustained missed deadlines, measured with audio transforms and after at least a 30-minute thermal test. This is our headroom target, not a paper result. The 20 ms analysis window is only one part of total latency; audio I/O, batching, computation and queues also count. Historical context is cached rather than repeatedly processing two seconds of audio. Test 2 and 5 evaluations before deciding the quality/latency compromise. One evaluation must not be presumed to retain the paper's quality.

We can profile an exported fixed-shape model with current/random weights before expensive training; dense inference cost depends mainly on its graph and shapes. Final quality, numerical conversion parity and thermal behavior still require the trained checkpoint and an actual device. [Apple's Xcode performance workflow](https://developer.apple.com/documentation/coreml/analyzing-a-core-ml-model-s-performance-in-xcode).

## Training and recovery

- Deterministic per-step augmentation, four spawned CPU workers, bounded prefetch and per-worker audio cache. Worker scheduling/count cannot change a resumed example sequence.
- AdamW, warmup/cosine schedule, EMA, finite loss/gradient checks; full-shape JVP loss already tested locally.
- Speaker-separated validation, verified separate DNS noise partitions, fixed corruption/noise seeds. Validation noise generators and pair RNG are isolated from training; all NFEs receive identical noise. Actual restoration metrics use clean audio only as the comparison reference.
- Atomic `latest.pt`, `best-validation.pt`, and the latest three milestone checkpoints. Save model, EMA, optimizer, RNG states, dataset fingerprints and configuration. Actual logs stay in JSONL rather than growing each checkpoint indefinitely.
- Startup verifies audio bytes against recorded SHA-256 and checks actual audio headers, full-corpus provenance, LTAS validity, and selected noise partition membership/content separation. Hashless legacy pilot manifests receive computed byte fingerprints; full mode requires recorded hashes. The verification map is cached for checkpoint writes, avoiding a corpus scan every 1,000 updates. Resume verifies this map and the checkpoint selection configuration; metadata-only old checkpoints require a fresh run. Prefetched but unused batches are regenerated by step index. Checkpoints are trusted local artifacts; do not load untrusted pickle checkpoints.
- SIGINT/SIGTERM and a six-hour process limit finish the current step and checkpoint. **Process exit does not stop Runpod billing.** Stop/terminate the pod explicitly after preserving outputs; persistent volume charges continue. No pod or volume has been created by this preparation.
- Train on data volume; never put the only checkpoint on an ephemeral container disk.

Before a long run: confirm actual CUDA VRAM/time, inspect paired audio, overfit a small batch, then run a short learning experiment and compare held-out 20/60/80/120 ms losses at 1/2/5 evaluations. Include gap-only error/energy, full-clip intelligibility, preserved-audio distortion, listening and hallucination/transcription checks. Keep the existing tPLCnet/Opus examples outside training. A 120 ms gap is outside the paper's reported 10–80 ms training range and should be labeled a stress test.

## Deliverables

- `configs/runpod-estimate.json`: default 300,000-update, effective-batch-16 full-data run.
- `configs/runpod-estimate-benchmark.json`: default 100-update real-data benchmark with late-schedule flow sampling.
- `configs/runpod.json` and `configs/runpod-benchmark.json`: earlier explicit batch-4 options.
- `scripts/prepare-data.sh`: resumable full corpus preparation, then strict preflight.
- `scripts/make-bundle.py`: allowlisted source-only archive; excludes credentials, unrelated projects, raw data and checkpoints.
- `runs/cloud-preparation/`: local verification, source access metadata and data readiness report.

Full training is intentionally blocked until the complete EARS/DNS/DAPS inputs exist. Passing a pilot check must not be mistaken for full-data readiness.

## Training review fixes

- LTAS target EQ is invariant to a positive change in source volume: source/reference power is measured at a shared level before gain clipping.
- When every sample has `r=t`, the target is `clean + sigma_min*noise`; the expensive JVP and extra model passes are skipped. Mixed/off-diagonal batches retain the MeanFlow calculation. Regression tests compare losses and all parameter gradients.
- Best checkpoint selection uses real damaged-only restoration. Diagonal prediction error alone cannot choose the best restoration checkpoint. Review checkpoints include JSON metrics and bounded, comparable audio exports.
- Dataset fingerprints verify actual bytes and selected partitions. Verification happens once per process and does not repeatedly read all audio during checkpoint saves. Keep prepared datasets immutable during a process; a restart/resume verifies them again.

The update target stays at **300,000**, effective batch **16**, with the same AdamW/EMA and approximate schedule. The fixes do not establish author architecture parity or demonstrate learned restoration quality.

## Framework and optimization plan

Keep PyTorch 2.8 as the tested correctness baseline. On the selected CUDA GPU, first measure the complete accumulated update and isolate network forward, off-diagonal JVP, backward, data augmentation and validation time. Then benchmark `torch.compile`/Inductor at identical shapes and precision. Compilation overhead belongs in setup time; compare steady-state throughput, memory, outputs, loss and all parameter gradients against eager execution. Compilation and GPU-kernel fusion can improve utilization, but do not change this graph's mathematical MAC count.

[JAX `jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html) plus [`jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html) is a credible alternative for the MeanFlow derivatives. A port would require matching convolutions, padding, SnakeBeta, normalization, Fourier embeddings, random samples, stopped JVP targets, gradients and streaming state. Do not assume a speed gain or rewrite the data pipeline before a measured comparison.

[Triton](https://triton-lang.org/main/index.html) is a GPU-kernel language/compiler, rather than a replacement training framework. PyTorch supports [custom Triton kernels under `torch.compile`](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html). If profiling identifies memory-bound elementwise work, investigate fusing channel normalization, conditioning or SnakeBeta. Custom training kernels must retain tested forward-mode JVP and backward behavior; a fast inference-only kernel is insufficient for this loss. No CUDA compile, custom-kernel or JAX speedup has been measured yet.

## Direct continuous A100 run

The active A100 run uses `configs/efficient-a100.json` and starts with:

```sh
python -u -m rmfsr.continuous_train --config configs/efficient-a100.json --run runs/efficient --control runs/continuous --skip-qualification
```

This explicitly skips the separate throughput qualification and full corpus scan at the user's request. Startup reads saved manifest/index metadata and loads real audio as training consumes it. Checkpoints record `manifest-declared-v1` and `full_audio_verified: false`; cancelled verification is not represented as passed. Other configurations retain full verification by default.

The controller saves and resumes training across six-hour process sessions toward 300,000 optimizer updates. It preserves model, optimizer, EMA, learning-rate schedule, and random state. A training error is recorded and requires diagnosis; the controller never calls Runpod or releases GPU allocation. Checkpoints are also saved at updates 1 and 100, then at the configured regular intervals. Release this GPU only on explicit user instruction. The separately cancelled watch automation remains disabled.

On 2026-10-12, the user requested validation every 5,000 updates starting with the next planned session restart, with checkpoints still every 1,000 updates. Both `configs/efficient-a100.json` and the remote `runs/efficient/training-config.json` have this interval queued. The current trainer retains its in-memory 1,000-update validation interval until it finishes its normal six-hour session; the existing controller then starts the next trainer with the saved configuration. No signal, immediate process restart, or pod lifecycle action is needed. The change record is `runs/cloud-preparation/validation-cadence-change.json`. The active pod is the migrated L40S, despite the configuration's historical A100 filename.

The user also requested the FIR alignment correction at that same planned restart. The queued `figure2-v3-aligned` augmentation profile centers the symmetric 65-tap bandlimit FIR, removing its 32-sample (2 ms at 16 kHz) delay while retaining the full crop. Other filters and random sampling are unchanged. Existing `figure2-v2` workers retain their original behavior even with the new source files installed. The saved configuration explicitly enables the narrow v2-to-v3 resume migration; model weights, EMA, optimizer state, learning-rate schedule, update count and the 300,000-update target continue from the checkpoint.

The corrected profile applies to both training and held-out augmentation. At the transition, the trainer records `augmentation-transition-after-step-XXXXXXXX.json`, preserves the previous best checkpoint as `best-validation-before-alignment-step-XXXXXXXX.pt`, and starts a new best-score threshold for the corrected validation recipe. New metrics carry `augmentation_profile`; scores across this boundary are not exactly comparable. This is a bookkeeping reset, not a model reset. No full-corpus verification, benchmark, extra worker, pod restart, or GPU release is part of this change. Local deployment records are under `runs/cloud-preparation/alignment-fix`.

## Live training dashboard

Run `python3 scripts/live-training-dashboard.py` from this repository, then open
`http://127.0.0.1:8765` on the same computer. The dashboard reads the current pod
and SSH endpoint from `runs/cloud-preparation/active-run.json`, using the existing
SSH key and strict host-key checking. It refreshes every ten seconds while the
page is visible and shows optimizer updates, recent loss and timing, GPU usage,
checkpoint file information, and connection freshness.

The server is local and read-only: it cannot start, stop, or change training.
Closing the page stops its refresh requests; closing the server or sleeping the
Mac does not stop remote training. To reopen after restarting the Mac, run the
same command again. No recurring watch automation is required.

The initial remaining-time estimate uses recent update durations, which exclude
validation and checkpoint writes. After observing at least one minute of progress,
the estimate uses wall-clock pace over up to 120 refresh samples. This includes
pauses during that observation window but is still an estimate, not a completion
guarantee. A stale reading or live process without recent updates is labeled
explicitly rather than assumed to be healthy training.

## Separate evaluation with paper metrics

`python -m rmfsr.paper_evaluate` evaluates a local, frozen EMA checkpoint on
SIG2024 test recordings. Install `requirements-eval.txt` in a separate local
environment. The evaluator uses CPU only and never connects to Runpod or a
trainer. Copy an immutable `step-XXXXXXXX.pt` checkpoint once; inference and
scoring then run entirely on the evaluation computer.

The input manifest contains `items`, each with `id`, `audio`, and `transcript`
paths. The first recorded evaluation uses 20 clips selected with random seed 42
from the 500 public test recordings linked by the official SIG-Challenge README
(`ICASSP2023/test_data`, with the corresponding ICASSP2024 transcript directory).
The pinned source commits, selected IDs, downloaded blob hashes and licenses are
saved under `runs/paper-evaluation`.

```sh
python -m rmfsr.paper_evaluate \
  --checkpoint runs/paper-evaluation/step-00020000.pt \
  --manifest runs/paper-evaluation/manifest-20.json \
  --dnsmos-weights runs/paper-evaluation/models/sig_bak_ovr.onnx \
  --whisper-weights runs/paper-evaluation/large-v3-turbo.pt \
  --output runs/paper-evaluation/step-00020000-sig20 \
  --nfes 5 --threads 2
```

The evaluator compares unprocessed audio with restored audio using DistillMOS,
standard non-personalized DNSMOS P.835 SIG and Whisper Turbo WER. It processes
whole recordings at 16 kHz without extra gain or output clipping. Language
detection is automatic; Whisper's multilingual basic normalizer preserves
diacritics and removes transcript markup. `<UNKNOWN/>` annotations are removed,
not scored as wildcard words. Corpus WER is total substitutions, deletions and
insertions divided by total reference words, rather than mean per-clip WER.

Per-clip audio, scores and transcripts are saved incrementally. Reusing an output
directory resumes only when the recorded checkpoint, input manifest, evaluation
settings and code fingerprints match. `results.json` explicitly identifies an
incomplete run; `provenance.json` records the full setup. Other NFE choices can be
specified with `--nfes 1 2 5` in a separate output directory.

These are the paper's named automatic metrics using public implementations, not
an exact reproduction of its evaluation protocol: the paper does not completely
specify ASR decoding, text normalization or the DNSMOS variant. A 20-clip sample
cannot be directly compared with the published 500-clip averages. Human ITU P.804
ratings require listeners, and no substitute is reported for the paper's
unspecified maximum-bandwidth estimator. One checkpoint also cannot establish a
training trend. Source: https://arxiv.org/html/2605.16251v1#S4.SS1.
