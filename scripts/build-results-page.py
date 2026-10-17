#!/usr/bin/env python3
"""Build the static listening gallery from the pinned, local 85k evaluation.

    PYTHONPATH=. .venv/bin/python scripts/build-results-page.py --infer-missing

The optional inference is CPU-only and computes only NFE 50. It uses verified
local checkpoint/conditioning files; it never connects to a pod or downloads
models. Subsequent runs reuse the completed NFE 50 results. Without that flag,
missing inference artifacts are an error. The checked-in docs/ assets are all
that GitHub Pages needs; neither PyTorch nor the source checkpoint is published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from scipy.signal.windows import hann
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "runs/paper-evaluation"
DOCS = ROOT / "docs"
EXTRA = EVAL / "width1-85000-page-nfe50-20261017"
CHECKPOINT_SHA = "0ff452cbd2a8edd9808d80acb148138986e68fd050a9829245a64ed22f5c1f3a"
CHECKPOINT_DIR = EVAL / "width1-85000-nfe-20261017"
CHECKPOINT = CHECKPOINT_DIR / "checkpoint-00085000.pt"
SEED = 1510001
NFES = [1, 20, 50, 100]
SR = 16000
SPECS = {
    "cplus": {
        "title": "C+ · Moderate degradation",
        "description": "A four-second speech clip with moderate simulated degradation.",
        "evaluation": CHECKPOINT_DIR,
        "input": EVAL / "moderate-input-preview-cplus-v1/Cplus_raw.wav",
        "clean": EVAL / "moderate-input-preview-v1/raw/clean.wav",
        "clean_sha256": "57a56ee865125f2d2bf2f61ef6638383f9d12c287fbdb171a08d3c8cae2a7bb8",
        "cache": EVAL / "wavjepa-cplus-listening-20261016T141715Z/ema-and-features.pt",
        "input_sha256": "789545d22f1aedaf3774ffe0eb5cb21f9a0962c56ce7c580a946d28ef89cf661",
        "cache_sha256": "b32aeb7107dd4b6b496509607843622f4f88aca51c9c2a6dd5c150760e8480a6",
        "samples": 64000,
        "source_note": "Local C+ listening example; clean reference available. This is a single illustrative clip, not a dataset-wide evaluation.",
    },
    "wind": {
        "title": "Wind · Clipping and suppression",
        "description": "A real recording affected by wind noise, clipping and noise suppression.",
        "evaluation": EVAL / "width1-85000-wind-nfe-20261017",
        "input": EVAL / "wavjepa-author-demo07-step85000-20261016T145516Z/input.wav",
        "cache": EVAL / "wavjepa-author-demo07-step85000-20261016T145516Z/features.pt",
        "input_sha256": "55c551d7e921edf707d6952b38450c8a49342824af46d6f43a859e5ee2369363",
        "cache_sha256": "5e90afac71d5ac8d6dbca92aac54631f991febcdfc1deb4ed7d518644a082ff4",
        "samples": 145920,
        "source_note": "Wind input from the paper demo, sample 7 (NISQA recording); resampled to 16 kHz. No clean reference is available. All restoration outputs on this page are ours.",
        "source_url": "https://sebraun-msr.github.io/realtimemeanflowspeechrestoration/",
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def audio(path: Path, length: int) -> np.ndarray:
    wave, rate = sf.read(path, dtype="float32")
    require(rate == SR and wave.shape == (length,), f"Wrong audio format: {path.name}")
    require(bool(np.isfinite(wave).all()), f"Nonfinite audio: {path.name}")
    return wave


def verify_sources() -> None:
    require(sha(CHECKPOINT) == CHECKPOINT_SHA, "Pinned checkpoint digest mismatch")
    for spec in SPECS.values():
        require(sha(spec["input"]) == spec["input_sha256"], "Input digest mismatch")
        require(sha(spec["cache"]) == spec["cache_sha256"], "Conditioning digest mismatch")
        if "clean" in spec:
            require(sha(spec["clean"]) == spec["clean_sha256"], "Clean reference digest mismatch")
        report = read_json(spec["evaluation"] / "results.json")
        for key, expected in {
            "training_step": 85000, "weights": "EMA", "nominal_frequency_width": 1,
            "checkpoint_sha256": CHECKPOINT_SHA, "noise_seed": SEED, "complete": True,
        }.items():
            require(report.get(key) == expected, f"Evaluation provenance mismatch: {key}")
        require(report["cached_conditioning_sha256"] == spec["cache_sha256"], "Evaluation cache mismatch")
        require(np.array_equal(audio(spec["input"], spec["samples"]),
                               audio(spec["evaluation"] / "input.wav", spec["samples"])),
                "Evaluation input differs from pinned input")


def infer_missing() -> None:
    import torch

    sys.path.insert(0, str(ROOT))
    from rmfsr.flow import pink_noise_like, sample
    from rmfsr.model import model_from_checkpoint
    from rmfsr.spectral import Spectral

    pending = [key for key in SPECS if not (EXTRA / key / "results.json").exists()]
    if not pending:
        return
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    # Digest verification above precedes deserializing this trusted local file.
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    require(checkpoint["step"] == 85000, "Unexpected checkpoint step")
    model = model_from_checkpoint(checkpoint).cpu().eval().requires_grad_(False)
    model.load_state_dict(checkpoint["ema"], strict=True)
    frequency_rms = float(model.embedding.frequencies.double().square().mean().sqrt())
    require(abs(frequency_rms - 1.087568547424467) < 1e-7, "Not the width-1 EMA")
    require(all(parameter.device.type == "cpu" for parameter in model.parameters()), "CPU-only inference required")
    spectral = Spectral(SR)
    for key in pending:
        spec = SPECS[key]
        cache = torch.load(spec["cache"], map_location="cpu", weights_only=False)
        require(cache["input_sha256"] == spec["input_sha256"], "Conditioning input mismatch")
        for field in ("repository", "revision"):
            require(cache["encoder"][field] == checkpoint["config"]["wavjepa"][field], "Encoder mismatch")
        require(cache["encoder"]["alignment"] == "frontend-centers-to-STFT-centers-v1", "Encoder alignment mismatch")
        wave = audio(spec["input"], spec["samples"])
        y = spectral.encode(torch.from_numpy(wave))
        features = cache["conditioning"].cpu()
        require(tuple(features.shape) == (1, 768, y.shape[-1]) and bool(torch.isfinite(features).all()), "Invalid conditioning")
        noise = pink_noise_like(y, torch.Generator().manual_seed(SEED))
        started = time.monotonic()
        print(json.dumps({"sample": key, "nfe": 50, "status": "CPU inference"}), flush=True)
        with torch.inference_mode():
            predicted = sample(model, y, steps=50, noise=noise.clone(), conditioning=features)
            output = spectral.decode(predicted, spec["samples"])[0].numpy()
        require(output.shape == wave.shape and bool(np.isfinite(output).all()), "Invalid prediction")
        destination = EXTRA / key / "output_nfe50.wav"
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, output, SR, subtype="FLOAT")
        require(np.array_equal(audio(destination, spec["samples"]), output), "Float export changed output")
        report = {
            "complete": True, "checkpoint_sha256": CHECKPOINT_SHA, "training_step": 85000,
            "weights": "EMA", "frequency_width": 1, "frequency_rms": frequency_rms,
            "input_sha256": spec["input_sha256"], "conditioning_sha256": spec["cache_sha256"],
            "output_sha256": sha(destination), "nfe": 50, "noise_seed": SEED,
            "device": "cpu", "seconds": time.monotonic() - started,
            "samples": len(output), "sample_rate": SR, "chunk_frames": None,
            "input_gain_db": 0, "cached_model_weights_used": False,
        }
        write_json(destination.parent / "results.json", report)
        print(json.dumps({"sample": key, "nfe": 50, "seconds": report["seconds"], "complete": True}), flush=True)


def spectrogram(wave: np.ndarray, destination: Path) -> None:
    from matplotlib import colormaps

    # Explicit, common amplitude calibration: the magnitude of the complex STFT
    # divided by the periodic Hann window sum. A unit-amplitude bin-centered
    # sinusoid is about -6.02 dB on this two-sided-amplitude convention. No
    # per-file peak normalization, mel transform, or coefficient compression.
    n_fft, hop = 512, 128
    padded = np.pad(wave.astype(np.float64), (n_fft // 2, n_fft // 2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop]
    window = hann(n_fft, sym=False)
    magnitude = np.abs(np.fft.rfft(frames * window, axis=-1)) / window.sum()
    decibels = np.clip(20 * np.log10(np.maximum(magnitude, 1e-12)), -80, 0)
    pixels = colormaps["viridis"]((decibels.T[::-1] + 80) / 80, bytes=True)[..., :3]
    raster = Image.fromarray(pixels).resize((640, 240), Image.Resampling.BILINEAR)
    destination.parent.mkdir(parents=True, exist_ok=True)
    raster.save(destination, optimize=True)


def build() -> dict:
    sources = {}
    for key, spec in SPECS.items():
        sources[key] = {"input": spec["input"]}
        if "clean" in spec:
            sources[key]["clean"] = spec["clean"]
        for nfe in NFES:
            if nfe == 50:
                path = EXTRA / key / "output_nfe50.wav"
                result = read_json(path.parent / "results.json")
                for field, value in {
                    "complete": True, "training_step": 85000, "weights": "EMA",
                    "frequency_width": 1, "checkpoint_sha256": CHECKPOINT_SHA,
                    "input_sha256": spec["input_sha256"], "conditioning_sha256": spec["cache_sha256"],
                    "noise_seed": SEED, "nfe": 50,
                }.items():
                    require(result.get(field) == value, f"NFE 50 provenance mismatch: {field}")
                require(sha(path) == result["output_sha256"], "NFE 50 output digest mismatch")
            else:
                path = spec["evaluation"] / f"output_nfe{nfe}.wav"
            sources[key][f"nfe{nfe}"] = path
    waves = {(key, name): audio(path, SPECS[key]["samples"])
             for key, mapping in sources.items() for name, path in mapping.items()}
    maximum = max(float(np.max(np.abs(wave))) for wave in waves.values())
    gain = 1.0 if maximum < 1.0 else 0.95 / maximum
    manifest = {
        "schema_version": 1,
        "checkpoint": {"step": 85000, "weights": "EMA", "frequency_width": 1, "sha256": CHECKPOINT_SHA},
        "nfes": NFES, "sample_rate": SR,
        "spectrogram": {
            "n_fft": 512, "hop_length": 128, "min_db": -80, "max_db": 0,
            "window": "periodic Hann", "frequency_scale": "linear", "color_map": "viridis", "colormap": "viridis",
            "width": 640, "height": 240,
            "reference": "20 log10(abs(STFT) / sum(Hann window)), reference amplitude 1; common absolute -80 to 0 dB scale for all clips. No per-clip normalization. Spectrograms use the published PCM audio; 0 to 8 kHz vertically, time horizontally.",
        },
        "playback": {"format": "WAV PCM16", "common_gain_db": float(20 * np.log10(gain)), "independent_normalization": False},
        "inference": {"device": "cpu", "noise_seed": SEED, "same_initial_noise_per_sample": True, "chunk_frames": None,
                      "conditioning": "Frozen wavJEPA, offline aligned features", "causal_streaming_evaluation": False},
        "samples": [],
    }
    for key, spec in SPECS.items():
        sample = {"id": key, "title": spec["title"], "description": spec["description"],
                  "duration_seconds": spec["samples"] / SR, "references": [], "outputs": [],
                  "source_note": spec["source_note"], "conditioning_sha256": spec["cache_sha256"]}
        if "source_url" in spec:
            sample["source_url"] = spec["source_url"]
        for name, source in sources[key].items():
            relative_audio = f"assets/audio/{key}/{name}.wav"
            relative_image = f"assets/spectrograms/{key}/{name}.png"
            destination = DOCS / relative_audio
            destination.parent.mkdir(parents=True, exist_ok=True)
            sf.write(destination, waves[key, name] * gain, SR, subtype="PCM_16")
            published = audio(destination, spec["samples"])
            require(float(np.max(np.abs(published))) < 1, "Clipped PCM export")
            require(float(np.max(np.abs(published - waves[key, name] * gain))) <= 1 / 32768 + 1e-7,
                    "PCM conversion exceeds 16-bit quantization error")
            spectrogram(published, DOCS / relative_image)
            item = {"audio": relative_audio, "spectrogram": relative_image,
                    "source_sha256": sha(source), "audio_sha256": sha(destination),
                    "spectrogram_sha256": sha(DOCS / relative_image)}
            if name.startswith("nfe"):
                nfe = int(name[3:])
                item.update(nfe=nfe, label=f"NFE {nfe}")
                sample["outputs"].append(item)
            else:
                item.update(kind=name, label="Clean reference" if name == "clean" else "Damaged input")
                sample["references"].append(item)
        require([item["nfe"] for item in sample["outputs"]] == NFES, "Wrong output order")
        manifest["samples"].append(sample)
    write_json(DOCS / "results.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-missing", action="store_true", help="Generate missing NFE 50 outputs on local CPU")
    arguments = parser.parse_args()
    verify_sources()
    if arguments.infer_missing:
        infer_missing()
    manifest = build()
    files = [DOCS / item[field] for sample in manifest["samples"]
             for item in sample["references"] + sample["outputs"] for field in ("audio", "spectrogram")]
    print(json.dumps({"complete": True, "audio_clips": len(files) // 2,
                      "assets_bytes": sum(path.stat().st_size for path in files),
                      "common_gain_db": manifest["playback"]["common_gain_db"]}), flush=True)


if __name__ == "__main__":
    main()
