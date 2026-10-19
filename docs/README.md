# RMFSR listening results

Static GitHub Pages listening gallery with all **eight recordings** from the [paper authors' demo](https://sebraun-msr.github.io/realtimemeanflowspeechrestoration/) restored by our **145,000-update production EMA checkpoint** at **NFE 1, 5, 20, and 100**. Every example uses positional time embeddings and adaptive double normalization (`scm-positional-adn-v1`). This is a saved experimental gallery, not a live training dashboard or the paper authors' model. No experimental TEMF or FiLM weights were used.

## Preview

From the repository root:

```sh
python3 -m http.server 8770 --bind 127.0.0.1 --directory docs
```

Open <http://127.0.0.1:8770/>. The page uses relative URLs and needs no JavaScript packages or external CDN. Serve it through HTTP rather than opening `index.html` directly.

## Publication

GitHub Pages serves the `docs/` folder on `main`. The `.nojekyll` file keeps these files static. Only the files in this directory are published; no checkpoints, training processes, cloud credentials, or private run directories are required by the website.

## Audio and spectrograms

`results.json` records the checkpoint identity, inference settings, sample provenance, and asset paths. Each sample uses the same input and initial noise across its four NFE settings. The default **Original relative levels** mode preserves the existing shared gain and relative loudness.

Select **RMS-matched playback** to match the whole-clip RMS of each track to its example's damaged input. One additional safety gain is shared by all tracks in that example so no export clips. This is a playback-volume adjustment only: it changes neither inference nor the original files. RMS includes speech and noise, so matching it is not a quality score or exact perceived-loudness match. Each recording is matched separately. The manifest records every applied gain, output hash and measured RMS.

The images show linear-frequency, log-magnitude STFTs, with one fixed color scale across all samples and outputs. Switching playback mode also switches images and download links to the corresponding audio. Exact FFT, hop, window, normalization, dB reference, and clipping limits are recorded in the manifest. These display spectrograms are not the model's compressed coefficient tensors.

The eight recordings follow the order of the [source demo](https://sebraun-msr.github.io/realtimemeanflowspeechrestoration/); the wind recording retains the `#sample-07` link. These are the demo's degraded inputs, processed at the model's 16 kHz mono rate.

No clean references are available for the eight recordings, so they have no clean-target metric claims. No other model's restored output is included. Our inference uses offline WavJEPA conditioning and does not establish real-time performance.

## Updating the snapshot

`scripts/build-results-page.py` creates the audio, spectrograms, and manifest from saved local evaluation artifacts. Run its `--help` for supported options. Rebuilding requires the corresponding local experiment artifacts and compatible inference code; they are not distributed with this static website. The committed website files can be served directly without them.

To reproduce this snapshot locally, supply the trusted checkpoint and the separate artifact directory:

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=. .venv/bin/python scripts/build-results-page.py \
  --checkpoint runs/paper-evaluation/scm-145000-cplus-20261019/checkpoint.pt \
  --checkpoint-sha256 cb9f082041764fc0c751f1c17a1ae20a40d49eaa1200ac44ca06d68517805838 \
  --run-directory runs/paper-evaluation/scm-145000-author-eight-20261019/mps \
  --source-directory runs/paper-evaluation/author-demo-balanced-20261014-v1 \
  --reuse-features-from runs/paper-evaluation/width1-85000-author-eight-20261017 \
  --nfes 1,5,20,100 --device mps --infer-missing
```

All outputs for a custom snapshot are regenerated from that checkpoint; only validated input audio and frozen encoder features are reused. Content hashes in asset URLs prevent browsers from replaying an earlier snapshot from cache.

To add or rebuild only RMS-matched assets from existing raw WAVs, without model inference:

```sh
.venv/bin/python scripts/build-rms-playback.py \
  --source runs/paper-evaluation/scm-145000-author-eight-20261019/mps
```

The full page builder also exports both playback modes automatically.

Keep the checkpoint label and provenance in sync with the exported outputs. Review the generated manifest and audio, preview the page, then commit the updated assets. Publishing the page never starts or changes training.
