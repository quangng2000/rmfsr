# RMFSR listening results

Static GitHub Pages demo for the 85,000-update, frequency-width-1 EMA snapshot. It contains the eight degraded recordings from the [paper authors' demo](https://sebraun-msr.github.io/realtimemeanflowspeechrestoration/), each restored by our model at NFE 1, 20, 50, and 100, with playable audio and log-magnitude STFT images. The page shows the source inputs and our outputs only. It is a saved experimental snapshot, not a live training dashboard or the paper authors' model.

## Preview

From the repository root:

```sh
python3 -m http.server 8770 --bind 127.0.0.1 --directory docs
```

Open <http://127.0.0.1:8770/>. The page uses relative URLs and needs no JavaScript packages or external CDN. Serve it through HTTP rather than opening `index.html` directly.

## Publication

GitHub Pages serves the `docs/` folder on `main`. The `.nojekyll` file keeps these files static. Only the files in this directory are published; no checkpoints, training processes, cloud credentials, or private run directories are required by the website.

## Audio and spectrograms

`results.json` records the checkpoint identity, inference settings, sample provenance, and asset paths. Each sample uses the same input and initial noise across its four NFE settings. All playback files share one gain, recorded in the manifest; they are not independently loudness-normalized.

The images show linear-frequency, log-magnitude STFTs, with one fixed color scale across all samples and outputs. Exact FFT, hop, window, normalization, dB reference, and clipping limits are recorded in the manifest. These display spectrograms are not the model's compressed coefficient tensors.

The eight recordings follow the order of the [source demo](https://sebraun-msr.github.io/realtimemeanflowspeechrestoration/); the wind example is sample 7, labeled “wind noise, clipping, noise suppression.” These are the demo's degraded inputs, processed at the model's 16 kHz mono rate. The earlier synthetic C+ example is no longer part of this gallery.

No clean references are available for these eight recordings, so the page makes no clean-target metric claims. No other model's restored output is included. Our inference uses offline WavJEPA conditioning and does not establish real-time performance.

## Updating the snapshot

`scripts/build-results-page.py` creates the audio, spectrograms, and manifest from saved local evaluation artifacts. Run its `--help` for supported options. Rebuilding requires the corresponding local experiment artifacts and compatible inference code; they are not distributed with this static website. The committed website files can be served directly without them.

Keep the checkpoint label and provenance in sync with the exported outputs. Review the generated manifest and audio, preview the page, then commit the updated assets. Publishing the page never starts or changes training.
