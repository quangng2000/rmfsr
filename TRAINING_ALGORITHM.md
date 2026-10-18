# RMFSR training algorithm

**For properly rendered equations, open the [typeset PDF](<output/pdf/RMFSR-training-algorithm.pdf>).** The [LaTeX source](<output/pdf/RMFSR-training-algorithm.tex>) is also editable. This Markdown preview may show raw math syntax.

**Implementation snapshot · verified 15 October 2026**

This document describes our `balanced-v1` run, checked against its live configuration and source at update **67,297**. The active data recipe is `figure2-v3-aligned`; the flow schedule is the restored gradual schedule. Training currently combines a data-parameterized MeanFlow objective with additional waveform PESQ and SI-SDR losses.

This is our implementation, not the authors' released training code. The core method follows [RMFSR, Sections 2–3](https://arxiv.org/html/2605.16251v1#S2). The additional waveform losses are our adaptation of [Investigating training objectives for flow matching-based speech enhancement, Section 3.4](https://arxiv.org/html/2512.10382v1#S3.SS4).

## Notation

- $s$: clean **target waveform**, after direct-path mixing and studio processing.
- $d$: corresponding **degraded waveform**, containing the same underlying speech.
- $c=\mathcal E(s)$ and $y=\mathcal E(d)$: compressed complex STFT coefficients, stored as real and imaginary channels.
- $x_t$: intermediate training state; $t$ is the current flow time and $r\le t$ is the destination time.
- $F_\theta(x_t,y,t,r)$: CNN output in compressed coefficient units.
- $\mathcal D$: inverse compression and waveform reconstruction.
- $\operatorname{sg}(\cdot)$: stop-gradient; the enclosed value is treated as constant during parameter backpropagation.

**Flow time runs from degraded/noisy at $t=1$ toward clean at $t=0$. It is separate from audio time and the optimizer-update counter.**

## 1. Generate a paired training example

Sample one or two EARS speech segments $s_j$, a shared room and microphone, and separate source positions. Generate full room impulse responses $h_j$ and their direct-path-only counterparts $h_j^{\mathrm{dir}}$.

Using the same mixing gains $a_j$ on both branches:

$$
s_{\mathrm{dir}}=\sum_{j=1}^{J}a_j(s_j*h_j^{\mathrm{dir}}),
\qquad
s_{\mathrm{wet}}=\sum_{j=1}^{J}a_j\,\operatorname{EQ}_j(s_j*h_j).
$$

The target and damaged waveforms are then

$$
\boxed{
s=\operatorname{Studio}(s_{\mathrm{dir}}),
\qquad
d=\operatorname{Corrupt}\!\left(
\operatorname{Level}(s_{\mathrm{wet}}+\beta\operatorname{EQ}_n(n),\ell)
\right).
}
$$

Here $n$ is DNS background noise; $\beta$ sets the sampled speech-to-noise ratio. Our code draws

$$
\mathrm{SNR}=5+10z_1\ \mathrm{dB},\qquad
\ell=-40+10z_2\ \mathrm{dBFS},\qquad z_1,z_2\sim\mathcal N(0,1).
$$

`Studio` applies DAPS-reference spectral EQ, mild compression, and a final target level of −25 dBFS. `Corrupt` samples bandwidth limitation, notch filtering, nonlinear distortion, modulation, phase distortion, spectral masking or suppression, MP3/GSM coding, quantization and 10–80 ms dropouts.

Our reproduction uses a 20% probability of a second source segment, order-2 image-source RIRs, and a static soft-knee compressor. These and unreported effect probabilities are implementation choices. Source segments are sampled independently; two segments are not guaranteed to have distinct speaker identities. The final damaged level can differ from $\ell$ because degradation follows level normalization.

## 2. Convert waveforms to coefficients

Use 16 kHz audio, a 320-sample square-root Hann window, and a 160-sample hop: **20 ms frames, 10 ms hop, 161 frequency bins**.

For a frame's Fourier coefficient $Z$, our transform applies

$$
\widetilde Z=\frac{Z}{\sum_n w[n]},
\qquad
\mathcal E:\ \widetilde Z\mapsto |\widetilde Z|^{0.3}e^{i\arg\widetilde Z}.
$$

The implementation floors magnitudes near zero for numerical stability. The CNN sees tensors shaped $[B,2,161,T]$, with real and imaginary parts in the two channels. Decoding reverses the compression, restores the scale, and performs inverse FFT and weighted overlap-add.

## 3. Sample flow times and construct the state

At zero-based optimizer update $k$, let $g=k/300000$. Independently for each example:

$$
t=\max\!\left(10^{-4},\operatorname{sigmoid}(z+0.4)\right),
\qquad z\sim\mathcal N(0,1),
$$

$$
r=
\begin{cases}
t,&\text{with probability }q(g),\\
tU^{\gamma(g)},&\text{otherwise},\quad U\sim\mathcal U(0,1).
\end{cases}
$$

The current gradual schedules are

$$
q(g)=0.2+0.4\left[1+\cos\!\left(\pi\min\!\left(\frac{g}{0.5},1\right)\right)\right],
$$

$$
\gamma(g)=0.05+0.475\left[1-\cos\!\left(\pi\min\!\left(\frac{g}{0.8},1\right)\right)\right].
$$

These approximate Figure 1's curves; mapping the paper's epoch axis onto 300,000 updates is our choice. The previously accelerated flow schedule is no longer active.

Sample pink coefficient noise $\epsilon$: independent Gaussian real/imaginary entries, with frequency-bin standard deviation proportional to $1/\sqrt{\max(f,1)}$, normalized to unit mean variance across frequency bins.

Set $\sigma_{\min}=10^{-8}$ and $\sigma_{\max}=0.3$, then construct

$$
\boxed{
x_t=(1-t)c+ty+\big[(1-t)\sigma_{\min}+t\sigma_{\max}\big]\epsilon.
}
$$

The corresponding conditional path velocity is

$$
v_{\mathrm{cond}}=y-c+(\sigma_{\max}-\sigma_{\min})\epsilon.
$$

The clean signal is available to construct this training state and its supervision. At inference, only the degraded signal and sampled noise are needed.

## 4. Compute the MeanFlow target and loss

Define the data-parameterized flow

$$
u_\theta(z,y,t,r)=\frac{z-F_\theta(z,y,t,r)}{t}.
$$

First obtain the detached **instantaneous tangent** from a diagonal network query:

$$
v_{\mathrm{diag}}=\operatorname{sg}\!\left[
\frac{x_t-F_\theta(x_t,y,t,t)}{t}
\right].
$$

Then evaluate the Jacobian–vector product at the sampled $(x_t,t,r)$:

$$
J_\theta=
D_z u_\theta(x_t,y,t,r)[v_{\mathrm{diag}}]
+\partial_tu_\theta(x_t,y,t,r).
$$

Equivalently, the code differentiates with respect to $(z,r,t)$ in direction $(v_{\mathrm{diag}},0,1)$. **The tangent comes from $r=t$; the differentiated flow is evaluated at the sampled $r,t$.** The condition $y$ and destination $r$ are held fixed in this directional derivative.

Build the detached coefficient target:

$$
\begin{aligned}
c_\star
&=\operatorname{sg}\!\left[x_t-tv_{\mathrm{cond}}+t(t-r)J_\theta\right]\\
&=\operatorname{sg}\!\left[c+\sigma_{\min}\epsilon+t(t-r)J_\theta\right].
\end{aligned}
$$

The implemented MeanFlow loss is

$$
\boxed{
\mathcal L_{\mathrm{MF}}
=\operatorname{mean}\!\left[
\big(F_\theta(x_t,y,t,r)-c_\star\big)^2
\right].
}
$$

The mean is over batch, real/imaginary channels, frequency and frame positions. This is coefficient MSE; algebraically it equals the mean squared velocity residual weighted by $t^2$. It must not be silently replaced with an unweighted velocity MSE.

When $r=t$, the correction vanishes and $c_\star=c+10^{-8}\epsilon\approx c$. If the entire microbatch is diagonal, the code skips the unnecessary JVP computation. For $r<t$, the main target includes the stopped JVP correction, so a low main loss is not directly a measurement of final restored-audio error.

## 5. Add the currently enabled waveform losses

Our additional supervision uses the **instantaneous clean estimate**:

$$
\widehat c_{\mathrm{diag}}=F_\theta(x_t,y,t,t),
\qquad
\widehat s_{\mathrm{diag}}=\mathcal D(\widehat c_{\mathrm{diag}}).
$$

This forward pass and its waveform decoder retain gradients. It is distinct from the detached diagonal query used to construct the JVP tangent.

The current total objective is

$$
\boxed{
\mathcal L_{\mathrm{total}}
=\mathcal L_{\mathrm{MF}}
+10^{-3}\mathcal L_{\mathrm{PESQ}}(s,\widehat s_{\mathrm{diag}})
+10^{-4}\mathcal L_{\mathrm{SI\text{-}SDR}}(\widehat s_{\mathrm{diag}},s).
}
$$

$\mathcal L_{\mathrm{PESQ}}$ is the differentiable `torch_pesq.PesqLoss` implementation with internal factor 1.0; it is a loss, not a reported PESQ score. The SI-SDR term is **negative** SI-SDR, averaged over eligible utterances:

$$
\alpha=\frac{\langle\widehat s,s\rangle}{\max(\|s\|^2,10^{-12})},
\quad
\delta=10^{-8}\max(\|s\|^2,10^{-12}),
$$

$$
\mathcal L_{\mathrm{SI\text{-}SDR}}
=-10\log_{10}\frac{\|\alpha s\|^2+\delta}
{\|\widehat s-\alpha s\|^2+\delta}.
$$

The code does not subtract waveform means. References with mean-square energy at or below $10^{-12}$ are excluded from both auxiliary losses; PESQ also excludes predictions below that threshold. Exclusions are logged. An empty eligible set contributes zero for that term.

**These waveform losses supervise a diagonal prediction from the sampled training state. We do not run the multi-step inference solver and backpropagate its final output during training.** The auxiliary terms are an adaptation added to our run, not a claim about the original RMFSR objective.

## 6. Complete optimizer-update algorithm

```text
Algorithm: one RMFSR optimizer update

Inputs:
    CNN parameters θ, EMA parameters θ̄, AdamW state, update index k
    Effective batch 16 = 4 microbatches × 4 examples

1. Set learning rate η(k), diagonal probability q(k/K), and span γ(k/K).
2. Clear accumulated parameter gradients.

3. Repeat for each of the 4 microbatches:
       a. Generate 4 paired target/degraded waveforms (s, d).
       b. Encode c = E(s), y = E(d).
       c. Sample t, r and pink noise ε independently for each example.
       d. Construct x_t and conditional velocity v_cond.

       e. Construct the stopped MeanFlow target:
              If all r = t: c_star = c + σ_min ε.
              Otherwise:
                  Evaluate the diagonal flow tangent without parameter gradients.
                  Compute JVP of u(z,y,t,r) in direction (v_diag, 0, 1).
                  c_star = stop_gradient(x_t − t v_cond + t(t−r) JVP).

       f. Predict p = F_θ(x_t, y, t, r) with parameter gradients.
       g. L_MF = mean((p − c_star)²).

       h. Obtain a differentiable diagonal estimate F_θ(x_t, y, t, t).
          Reuse p when the entire microbatch has r = t.
       i. Decode it and compute PESQ loss and negative SI-SDR loss.
       j. L = L_MF + 0.001 L_PESQ + 0.0001 L_SI-SDR.
       k. Backpropagate L / 4 into θ; accumulate gradients.

4. Clip the accumulated global gradient norm to 1.0.
5. Take ONE AdamW optimizer step; weight decay = 0.01.
6. Update EMA: θ̄ ← 0.999 θ̄ + 0.001 θ.
7. Record losses, update duration and sampling-schedule values.
8. Save / validate when due; increment the completed-update counter.
```

With $K=300000$ and zero-based $k$, the learning rate is

$$
\eta(k)=10^{-4}\min\!\left(1,\frac{k+1}{5000}\right)
\left[0.1+0.45\left(1+\cos\left(\pi\min(k/K,1)\right)\right)\right].
$$

Each example is a 4-second mono crop. AdamW uses PyTorch's default moment coefficients $(0.9,0.999)$ and numerical epsilon $10^{-8}$. The 300,000-update budget and optimizer settings are our experiment choices.

Checkpoints are saved every 1,000 updates, with additional saves for improvements and review milestones. Every 5,000 updates, and at designated review points, EMA weights restore a fixed 32-example validation set at NFE 1, 2 and 5 without backpropagation. Best-checkpoint selection uses NFE-5 waveform MSE. Six-hour training sessions resume the saved model, EMA, optimizer, schedule and update count.

## 7. Inference: how the trained model restores audio

Given only a degraded waveform $d$, encode $y=\mathcal E(d)$ and initialize

$$
x^{(0)}=y+0.3\epsilon.
$$

For $N$ network evaluations, use $t_i=1-i/N$ and $r_i=1-(i+1)/N$, for $i=0,\ldots,N-1$:

$$
p_i=F_{\bar\theta}(x^{(i)},y,t_i,r_i),
$$

$$
\boxed{
x^{(i+1)}=x^{(i)}-(t_i-r_i)\frac{x^{(i)}-p_i}{t_i}
=\frac{r_i}{t_i}x^{(i)}+\left(1-\frac{r_i}{t_i}\right)p_i.
}
$$

Return $\widehat s=\mathcal D(x^{(N)})$. No clean target, training loss or JVP is used during inference. With NFE 1, the coefficients returned are exactly $F_{\bar\theta}(y+0.3\epsilon,y,1,0)$. With more steps, the last update still returns that step's CNN prediction, but at a different intermediate state.

## Code references

- [Pair generation](rmfsr/data.py) and [degradations](rmfsr/degradations.py).
- [Spectral encoding and decoding](rmfsr/spectral.py).
- [Time sampling and MeanFlow objective](rmfsr/flow.py).
- [Waveform fidelity losses](rmfsr/speech_fidelity.py).
- [Gradient accumulation and training loop](rmfsr/train.py).
- [Current configuration](<configs/balanced-candidate.json>).

The remote and local versions of `flow.py`, `train.py`, `data.py`, `flow_curriculum.py` and `speech_fidelity.py` had matching SHA-256 hashes when this document was prepared. Earlier checkpoints used different recipes; this document records that phase of training. See the README for later conditioning changes.
