import nbformat as nbf
from pathlib import Path
n=nbf.v4.new_notebook();cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s))
def code(s):cells.append(nbf.v4.new_code_cell(s))
md('''# RMFSR: full-width reproduction attempt

**Status: GPU pilot complete; full training and paper parity are not complete.**

This notebook runs our independent PyTorch implementation of [Real-time Speech Restoration using Data Prediction Mean Flows](https://arxiv.org/abs/2605.16251). It does not use pretrained RMFSR weights. The current 100-step checkpoint has **not learned meaningful speech-gap filling**.

The authors' public repository contains demonstration audio, but no model code or weights were available when checked. See `README.md` for the complete paper-to-code audit.

Select kernel **RMFSR (PyTorch / MPS)**. Running all cells displays the completed pilot and comparisons; the long training cell is disabled by default.''')
code(r'''from pathlib import Path
import sys, json, torch
from IPython.display import display, Markdown, Audio, Image
ROOT = Path.cwd()  # Start Jupyter from the repository root.
sys.path.insert(0, str(ROOT))
from rmfsr.model import RMFSR
from rmfsr.flow import meanflow_loss, sample
from rmfsr.streaming import StreamingRestorer
run = json.loads((ROOT/'runs/pilot/run.json').read_text())
profile = json.loads((ROOT/'runs/full-shape-profile.json').read_text())
report = json.loads((ROOT/'runs/pilot/evaluation/report.json').read_text())
print('PyTorch:', torch.__version__)
print('MPS GPU available:', torch.backends.mps.is_available())
print('Completed training steps:', run['completed_steps'])
print('Full training ready:', json.loads((ROOT/'runs/full-preflight.json').read_text())['ready'])''')
md('''## 1. Network and parity audit

The backbone uses all five reported channel widths, inverted residual blocks, a four-layer temporal bottleneck, frequency attention, and SnakeBeta. Temporal convolutions are causal; frequency downsampling never downsamples audio time.

Our choices for attention, normalization, dilation and block wiring are explicit assumptions. **The parameter and compute counts do not match the paper**, so this is a reproduction attempt, not a validated replica.''')
code(r'''model = RMFSR()
parameters = sum(p.numel() for p in model.parameters())
display(Markdown(f"""| Measure | Our implementation | Paper |
|---|---:|---:|
| Parameters | {parameters/1e6:.2f} million | 7.8 million |
| GMAC/audio-second/NFE | {profile['gmac_per_audio_second_per_evaluation']:.2f} | 1.22 |
| Past context per evaluation | {(model.receptive_frames-1)*.01:.2f} s | 2.13 s |
| STFT window | 20 ms | 20 ms |

The compute estimate excludes elementwise operations and STFT. No phone-performance claim is supported.
"""))''')
md(r'''## 2. What is learned?

The network predicts clean **complex compressed STFT coefficients**, conditioned on the damaged speech and two flow times. It is data-prediction MeanFlow, not direct score-function training.

$$
x_t=(1-t)x_0+t y+[(1-t)\sigma_{\min}+t\sigma_{\max}]\epsilon.
$$

$$
u_\theta(x_t,y,t,r)=\frac{x_t-\hat{x}_\theta(x_t,y,t,r)}{t}.
$$

Here, $t$ decreases from 1 to 0 during restoration. Gaussian spectral noise has a $1/f$ energy profile. Sampling starts around **the damaged input**, not around silence.

The instantaneous model velocity supplies the JVP direction. The derivative is detached before the parameter update:

$$
J=\operatorname{stopgrad}\!\left[\operatorname{JVP}(u_\theta;(v_\theta,0,1))\right].
$$

$$
x_{\mathrm{target}}=x_t-t v_{\mathrm{cond}}+t(t-r)J,\qquad
\mathcal{L}=\mathbb{E}\left[\|\hat{x}_\theta-x_{\mathrm{target}}\|^2\right].
$$

On the diagonal $r=t$, this becomes ordinary data prediction, up to the tiny residual noise $\sigma_{\min}$. There is no discriminator or GAN refinement in this implementation.''')
md('''## 3. Training data and limitations

Pilot: p001 training, p002 validation, p003 reserved testing, all from EARS. The existing synthetic speech comparison is never used to train. Full recipe: EARS speech, image-source room responses, DNS non-speech noise, codec/nonlinear/spectral degradations, 10–80 ms dropouts, and DAPS-derived target equalization.

The pilot substitutes synthetic colored background and omits DAPS EQ. Those substitutions are forbidden in full mode. EARS is CC BY-NC 4.0; this is a local research experiment.''')
code(r'''for split in ['pilot_train','pilot_validation','pilot_test']:
    rows = json.loads((ROOT/'data'/f'{split}.json').read_text())
    print(split, sorted({r['speaker'] for r in rows}), f"{sum(r['seconds'] for r in rows)/3600:.2f} hours")
print('\nFull-run prerequisites still missing:')
for problem in json.loads((ROOT/'runs/full-preflight.json').read_text())['problems']:
    print('-', problem)''')
md('''## 4. Pilot learning curve

One hundred steps validates gradient computation, optimizer updates and checkpointing. It does not establish convergence. Loss varies because degradations are generated afresh for each example. The fixed validation check measures data-prediction error; actual generated-speech evaluation follows below.''')
code(r'''import matplotlib.pyplot as plt
rows = [json.loads(line) for line in (ROOT/'runs/pilot/metrics.jsonl').read_text().splitlines()]
fig, ax = plt.subplots(figsize=(10,3))
ax.plot([r['step'] for r in rows], [r['loss'] for r in rows], alpha=.65, label='training loss')
validation = [r for r in rows if 'validation_dp_mse' in r]
ax.plot([r['step'] for r in validation], [r['validation_dp_mse'] for r in validation], 'o-', label='fixed validation DP MSE')
ax.set(xlabel='Training step', ylabel='Compressed spectral MSE', title='Full-width model: implementation pilot only')
ax.legend(); plt.show()''')
md('''## 5. Same-input comparison with Opus and tPLCnet

These are the existing 5.425-second synthetic-speech cases, with identical packet masks and baselines. This single utterance cannot establish a general ranking. **The pilot currently performs approximately like zero filling.** That says it is undertrained; it does not test the published RMFSR model.

120 ms gaps exceed the paper's stated 10–80 ms training range. Whole-clip STOI can hide empty gaps, so examine the waveform and listen as well.''')
code(r'''labels = ['zero_filled','opus_deep','tplc_l','rmfsr_nfe1','rmfsr_nfe2','rmfsr_nfe5']
lines = ['| Gap | ' + ' | '.join(labels) + ' |', '|---|' + '---:|'*len(labels)]
for case in report['results']:
    lines.append('| '+case['case']+' | '+' | '.join(f"{case['metrics'][name]['stoi']:.4f}" for name in labels)+' |')
display(Markdown('**Whole-clip STOI (higher is better)**\n\n'+'\n'.join(lines)))''')
code(r'''case_name = 'opus_120ms'
case_dir = ROOT/'runs/pilot/evaluation'/case_name
display(Image(filename=str(case_dir/'gap_comparison.png')))
for label, filename in [
    ('Original reference','reference'),
    ('Damaged input: zero-filled gaps','zero_filled'),
    ('Opus Deep PLC','opus_deep'),
    ('tPLCnet Large','tplc_l'),
    ('Our RMFSR pilot, NFE=5 — undertrained','rmfsr_nfe5')]:
    display(Markdown('**'+label+'**'))
    display(Audio(filename=str(case_dir/f'listen_{filename}.wav')))
print('All clips use the same playback gain. No per-model loudness normalization.')''')
md('''## 6. Causality, streaming and restart verification

The tests perturb future input and verify earlier outputs are unchanged. Separate history is maintained for every flow evaluation. Streaming STFT/overlap-add and waveform output agree with offline computation using the same prior noise.

The default comparison batches 100 ms of frames for speed. This adds buffering and **is not a 20 ms live-latency measurement**. The true waveform streaming API accepts hop-sized chunks. Neither mode has been validated for real-time phone deployment.''')
code(r'''print((ROOT/'runs/tests.log').read_text())
print('Checkpoint restart check:', json.loads((ROOT/'runs/resume-check.json').read_text()))
case = next(c for c in report['results'] if c['case']=='opus_120ms')
print('Eager MPS NFE=5 real-time factor:', round(case['timing']['rmfsr_nfe5']['rtf'],2))
print('Real-time factor > 1 means slower than real time.')''')
md('''## 7. Full-training plan

Resolve/document the architecture discrepancy, prepare full data, choose compute, then train and evaluate. Our proposed AdamW/300k-step schedule is a starting choice, not an author-provided recipe. The local measurement predicts roughly 12 days for network updates alone. Data preparation, augmentation and evaluation add time.

Use `README.md` for data preparation and prerequisites. Keep full-run test speakers p104–p107 separate. Future evaluation should include many speakers, 20/60/80 ms gaps plus 120 ms stress tests, listening, transcript errors, and the paper's SIG2024 benchmarks.''')
code(r'''print(json.dumps(profile, indent=2))
# Explicit opt-in: Run All will not launch a multi-day training job.
RUN_FULL_TRAINING = False
if RUN_FULL_TRAINING:
    import os
    from rmfsr.train import train
    cfg = json.loads((ROOT/'configs/full.json').read_text())
    previous_directory = Path.cwd()
    try:
        os.chdir(ROOT)
        train(cfg, ROOT/'runs/full')  # Fails clearly until all full-data prerequisites exist.
    finally:
        os.chdir(previous_directory)
else:
    print('Full training has not been launched. No cloud resources were provisioned.')''')
n.cells=cells;n.metadata.kernelspec={'display_name':'RMFSR (PyTorch / MPS)','language':'python','name':'rmfsr'}
n.metadata.language_info={'name':'python','version':'3.12'}
nbf.write(n,Path(__file__).resolve().with_name('rmfsr_reproduction.ipynb'))
