# Current restoration training

**Snapshot: 18 October 2026.** The `balanced-wavjepa-scm-v1` continuation uses
positional time embeddings, adaptive double normalization, frozen wavJEPA
conditioning, and the full data-prediction MeanFlow/JVP loss. PESQ and SI-SDR
auxiliary losses were disabled after update **108,502**, beginning at 108,503.
TEMF weighting and its boundary scheduler are not enabled in this snapshot.

## Inputs and learned parameters

- `c`: clean target waveform encoded into compressed complex STFT coefficients.
- `y`: matching damaged waveform encoded using the same transform.
- `h`: frozen wavJEPA features from the **damaged waveform only**, aligned to STFT frames.
- `x_t`: constructed training state at flow time `t`; `r <= t` is the destination time.
- `model(x_t, y, t, r, conditioning=h)`: trainable CNN, time conditioning, and feature adapter.

The encoder stays frozen; the adapter learns. Features `h` are computed once per
microbatch and remain fixed during every model query and JVP. Training does not
compare clean and damaged JEPA embeddings or update the pretrained encoder.

## One optimizer update

Python-style pseudocode; time scalars broadcast over each example's coefficient
dimensions. `conditioner` includes encoder preprocessing and feature alignment.
`no_grad()` disables reverse-mode parameter gradients; forward-mode `jvp` still works.

```python
optimizer.zero_grad()
diagnostics = initialize_microbatch_diagnostics()
for microbatch in range(accumulation_steps):
    clean_audio, damaged_audio = next_training_pair()
    c, y = encode(clean_audio), encode(damaged_audio)
    with no_grad():
        h = conditioner(damaged_audio)

    # Both y and h are captured explicitly by this wrapper.
    def predict(z, start, end):
        return model(z, y, start, end, conditioning=h)

    def velocity(z, end, start):
        return (z - predict(z, start, end)) / start

    t, r = sample_times(progress=completed_updates / 300_000,
                        schedule="figure1-cosine-approx")
    epsilon = pink_noise_like(c)
    sigma_min, sigma_max = 1e-8, 0.3
    sigma_t = (1 - t) * sigma_min + t * sigma_max
    x_t = (1 - t) * c + t * y + sigma_t * epsilon

    with no_grad():
        if all(r == t):
            target = c + sigma_min * epsilon
        else:
            path_velocity = y - c + (sigma_max - sigma_min) * epsilon
            instantaneous = velocity(x_t, t, t)
            _, J = jvp(velocity, (x_t, r, t),
                       (instantaneous, zeros_like(r), ones_like(t)))
            target = x_t - t * path_velocity + t * (t - r) * J

    prediction = predict(x_t, t, r)  # includes y and frozen h
    loss = mean((prediction - stop_gradient(target)) ** 2)
    (loss / accumulation_steps).backward()
    accumulate_microbatch_diagnostics(diagnostics, loss, target)

pre_clip_norm = clip_gradients(model.parameters(), max_norm=1.0)
check_stability_before_optimizer_step(diagnostics, pre_clip_norm)
optimizer.step()
update_EMA()
```
This omits learning-rate assignment, data loading, logging and checkpoint bookkeeping.
The loss contains no waveform-fidelity term, inference rollout, or TEMF weight.
The pre-step guard rejects unsafe updates before optimizer advancement.

## Why the JVP changes the clean-coefficient target

MeanFlow represents **average velocity across an interval**, while an instantaneous
velocity describes movement at one point. For a consistent interval-average field
along its trajectory, `u = v - (t-r) * du/dt`. Our coefficient prediction `p`
parameterizes the field through `u = (x_t - p) / t`.

Our implemented target therefore has the equivalent forms:

```text
target = stop_gradient(x_t - t * path_velocity + t * (t-r) * J)
       = stop_gradient(c + sigma_min * epsilon + t * (t-r) * J)
```
Here `J = D_x u[instantaneous] + partial_t u`; `r`, `y`, and `h` are held fixed.
Our implementation uses the **model-derived instantaneous velocity** as the JVP
direction, and the analytic conditional path velocity in the target's other term.
This is not the analytic-path JVP direction used in the TEMF comparison.

The correction teaches interval-flow consistency; it does not change the clean
reference waveform or align audio samples. At `r=t` it vanishes. At `r<t` it can
raise or lower individual target coefficients. A low loss measures agreement with
this detached, model-dependent target, not final restored-audio quality. Very large
derivatives can dominate the target, which is why derivative stability matters.

Sources: [MeanFlow and sampler](rmfsr/flow.py), [wavJEPA conditioning](rmfsr/wavjepa.py),
[training loop](rmfsr/train.py). The [15 October algorithm](TRAINING_ALGORITHM.md)
documents the earlier phase with waveform-fidelity supervision.
