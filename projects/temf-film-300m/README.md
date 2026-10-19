# TEMF + wavJEPA FiLM restoration

Status: project scope established; redesigned model is not implemented or trained.

## Agreed scope

- Target approximately **300 million trainable restoration parameters**.
- Keep the existing **89,280,768-parameter wavJEPA encoder frozen**.
- Use wavJEPA features to condition restoration through trainable **FiLM** adapters.
- Predict the **mean velocity directly**, rather than converting a clean-coefficient prediction into velocity.
- Implement and test TEMF's objective and time sampling against the pinned authors' code.

The combined inference model would contain approximately 389 million parameters.
The exact count depends on the backbone and adapter design, which remain to be specified.

## Proposed signal path

1. Encode degraded audio as compressed complex STFT coefficients.
2. Independently extract frozen wavJEPA features from the same degraded waveform.
3. Align those features to audio-frame positions and project them through trainable adapters.
4. Generate FiLM scale and shift for selected restoration blocks.
5. Condition the restoration backbone on the flow state, degraded coefficients,
   current/destination flow times, and the aligned wavJEPA context.
6. Predict mean velocity and integrate it to the clean endpoint.
7. Decompress coefficients and reconstruct the waveform with inverse STFT.

FiLM will start as an identity modulation: `(1 + scale) * features + shift`,
with zero-initialized final scale/shift projections. Placement and normalization
must be explicit experimental choices; they are not established quality improvements.

## TEMF fidelity requirements

Reference commit: `78c531cca978a23113b262cd0b6bede2197b1916`.

- Verify the actual `TEMFLoss` class and its inherited sampling/target helpers.
- Port temporal equilibrium weighting, boundary scheduling, JVP tangents,
  stop-gradient placement, and reduction conventions with numerical parity tests.
- Record the exact source configuration; distinguish released defaults from paper ablations.
- Resolve the Gaussian versus degraded-centered prior explicitly before implementation.
- Treat compressed STFT, wavJEPA, FiLM, and any prior change as speech-specific adaptations.
- Specify guidance behavior rather than silently omitting or adding classifier-free guidance.
- Keep full MeanFlow derivatives. Do not silently remove a derivative term for stability.

This is a TEMF-based speech-restoration redesign, not a reproduction of the paper's
ImageNet architecture or evidence that its image-generation results transfer to speech.

## Initialization and evaluation

The existing 125k checkpoint predicts coefficients. Its output head and optimizer
state cannot be reinterpreted as a direct velocity predictor. Choose and document
fresh initialization or an explicit transfer strategy before launching training.

First validate flow signs/endpoints, FiLM identity behavior, feature alignment,
frozen-encoder gradients, objective parity, finite JVPs, and actual parameter count.
Short runs establish execution and stability only. A quality experiment must compare
held-out restoration, intelligibility, loudness, and compute cost at fixed NFE values.
Training duration, hardware budget, and launch are separate decisions.

## Isolation

Branch: `codex/temf-film-300m`, based on committed `main` at `1ecfcdd`.
This worktree excludes the other checkout's uncommitted training changes.
Keep new model/configuration code opt-in and separate from production defaults.
Do not change, pause, or restart production training, its controller, or its pod as
part of project setup. No cloud resources or training jobs have been requested here.

## References

- [TEMF source at the pinned commit](https://github.com/yuanpengtu/TEMF/blob/78c531cca978a23113b262cd0b6bede2197b1916/src/training/loss.py)
- [TEMF paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Tu_Temporal_Equilibrium_MeanFlow_Bridging_the_Scale_Gap_for_One-Step_Generation_CVPR_2026_paper.pdf)
- [FiLM paper](https://arxiv.org/abs/1709.07871)
- [Pinned wavJEPA encoder](https://huggingface.co/labhamlet/wavjepa-base/tree/6be4a5093f6c9adbbd55e00b6e5b8f067aa03345)
