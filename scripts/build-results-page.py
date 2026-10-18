#!/usr/bin/env python3
"""Build a saved-checkpoint gallery on the paper's eight original demo inputs.

Cache features with the local transformers runtime (offline only):
  HF_HUB_OFFLINE=1 PYTHONPATH=. python scripts/build-results-page.py --prepare-features-only
Run resumable local restoration with the torch runtime (CPU by default):
  PYTHONPATH=. python scripts/build-results-page.py --infer-missing
Disjoint subsets can be run by separate processes, without building the page:
  PYTHONPATH=. python scripts/build-results-page.py --infer-missing --inference-only --samples 1,2,3,4
With no flags, rebuild public assets from completed local runs.
For a newer checkpoint, pass --checkpoint, --checkpoint-sha256 and
--run-directory. Use --reuse-features-from to reuse verified encoder caches
and --device mps to render on a local Apple GPU.

No download, pod connection, training change, author restoration output, or
clean reference is involved. Inference is saved after each NFE. Existing
artifacts must pass provenance and hash checks before reuse.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image
from scipy.signal import resample_poly
from scipy.signal.windows import hann
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "runs/paper-evaluation"
DOCS = ROOT / "docs"
SOURCE = EVAL / "author-demo-balanced-20261014-v1"
RUN = EVAL / "width1-85000-author-eight-20261017"
CHECKPOINT = EVAL / "width1-85000-nfe-20261017/checkpoint-00085000.pt"
CHECKPOINT_SHA = "0ff452cbd2a8edd9808d80acb148138986e68fd050a9829245a64ed22f5c1f3a"
CHECKPOINT_STEP = 85000
CONDITIONING_ARCHITECTURE = "gaussian-fourier-v1"
DEVICE = "cpu"
MANIFEST_SHA = "60f8cd95359d3d7f51c9d5227cd6776e9ef359904e4688992eff5f43fa139946"
ENCODER_REPOSITORY = "labhamlet/wavjepa-base"
ENCODER_REVISION = "6be4a5093f6c9adbbd55e00b6e5b8f067aa03345"
ALIGNMENT = "frontend-centers-to-STFT-centers-v1"
SEED, SR, NFES = 1510001, 16000, [1, 20, 50, 100]
FREQUENCY_RMS = 1.087568547424467
WIND_INPUT = EVAL / "wavjepa-author-demo07-step85000-20261016T145516Z/input.wav"
WIND_INPUT_SHA = "55c551d7e921edf707d6952b38450c8a49342824af46d6f43a859e5ee2369363"
WIND_CACHE = WIND_INPUT.parent / "features.pt"
WIND_CACHE_SHA = "5e90afac71d5ac8d6dbca92aac54631f991febcdfc1deb4ed7d518644a082ff4"
WIND_OUTPUTS = EVAL / "width1-85000-wind-nfe-20261017"
WIND_50 = EVAL / "width1-85000-page-nfe50-20261017/wind"
WIND_HASHES = {
    1: "8bb3bc5018f18a7d7490240bd2b0f3481519d096436d12292e111306ac286f0d",
    20: "caf17ddfca85d6872ce21dc7fcdedae019d37dc7e1685543fd5887db8bd210ba",
    50: "cddf527d3087a1f0ba4bb0ece4e6bf2089191eb77276cac7c3b56692aa79a470",
    100: "6c5c853e602945e4f264588ee8d3d3fb7f47ae78c659ca0d6c386888a49531cb",
}
EXPECTED_LENGTHS = [148557, 215481, 160000, 184000, 165888, 192000, 145920, 128000]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wave_sha(wave: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(wave, dtype="<f4").tobytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def audio(path: Path, length: int) -> np.ndarray:
    wave, rate = sf.read(path, dtype="float32")
    require(rate == SR and wave.shape == (length,), f"Wrong audio format: {path}")
    require(bool(np.isfinite(wave).all()), f"Nonfinite audio: {path}")
    return wave


def save_float(path: Path, wave: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".wav.tmp")
    sf.write(temporary, wave, SR, subtype="FLOAT", format="WAV")
    require(np.array_equal(audio(temporary, len(wave)), wave), "Float export changed samples")
    temporary.replace(path)


def load_specs() -> list[dict]:
    require(sha(SOURCE / "manifest.json") == MANIFEST_SHA, "Pinned source manifest mismatch")
    require(sha(CHECKPOINT) == CHECKPOINT_SHA, "Pinned checkpoint mismatch")
    manifest = read_json(SOURCE / "manifest.json")
    require([item["index"] for item in manifest["items"]] == list(range(1, 9)), "Expected exact eight inputs")
    specs = []
    for item, length in zip(manifest["items"], EXPECTED_LENGTHS):
        key = f"sample-{item['index']:02d}"
        original = SOURCE / key / "input.wav"
        require(sha(original) == item["files"]["input"]["sha256"], f"Source mismatch: {key}")
        wave, rate = sf.read(original, dtype="float32", always_2d=True)
        require(wave.shape[1] == 1 and rate in (16000, 24000), "Expected mono 16/24kHz source")
        wave = wave[:, 0]
        if rate != SR:
            divisor = math.gcd(rate, SR)
            wave = resample_poly(wave, SR // divisor, rate // divisor)
        wave = np.ascontiguousarray(wave, dtype=np.float32)
        require(wave.shape == (length,) and bool(np.isfinite(wave).all()), "Source audio length/data mismatch")
        require(np.array_equal(wave, audio(SOURCE / key / "model_input_16k.wav", length)),
                f"Reproducible source resampling mismatch: {key}")
        specs.append({"id": key, "index": item["index"], "source_id": item["id"],
                      "title": item["label"], "description": item["label"].split("(", 1)[1].rstrip(")"),
                      "original": original, "original_sha256": item["files"]["input"]["sha256"],
                      "source_url": item["files"]["input"]["url"], "source_sample_rate": rate,
                      "wave": wave, "samples": length, "directory": RUN / key,
                      "input_audio_sha256": wave_sha(wave)})
    return specs


def prepare_inputs(specs: list[dict]) -> None:
    for spec in specs:
        directory = spec["directory"]
        path, metadata = directory / "input.wav", directory / "input.json"
        if path.exists() or metadata.exists():
            require(path.exists() and metadata.exists(), f"Incomplete input artifact: {directory}")
            record = read_json(metadata)
            for key in ("original_sha256", "input_audio_sha256", "samples", "source_sample_rate"):
                require(record[key] == spec[key], f"Input metadata mismatch: {directory}/{key}")
            require(sha(path) == record["input_sha256"], f"Prepared input digest mismatch: {directory}")
            require(np.array_equal(audio(path, spec["samples"]), spec["wave"]), "Prepared input samples changed")
        else:
            save_float(path, spec["wave"])
            write_json(metadata, {key: spec[key] for key in
                                 ("original_sha256", "input_audio_sha256", "samples", "source_sample_rate")} |
                       {"input_sha256": sha(path), "sample_rate": SR, "input_gain_db": 0,
                        "resampling": "scipy.signal.resample_poly with default Kaiser window; 24kHz sources only"})
        spec["input"] = path
        spec["input_sha256"] = sha(path)


def verify_encoder(metadata: dict) -> None:
    for key, expected in {"repository": ENCODER_REPOSITORY, "revision": ENCODER_REVISION,
                          "alignment": ALIGNMENT, "frozen": True, "sample_rate": SR,
                          "causal": False, "segment_samples": 32159}.items():
        require(metadata.get(key) == expected, f"Encoder mismatch: {key}")


def feature_record(spec: dict, required: bool = True) -> dict | None:
    path, metadata = spec["directory"] / "features.pt", spec["directory"] / "features.json"
    if not path.exists() and not metadata.exists() and not required:
        return None
    require(path.exists() and metadata.exists(), f"Missing/incomplete features: {spec['id']}; run --prepare-features-only")
    record = read_json(metadata)
    for key in ("input_sha256", "input_audio_sha256", "original_sha256", "samples"):
        require(record.get(key) == spec[key], f"Feature provenance mismatch: {spec['id']}/{key}")
    require(record.get("complete") is True and record.get("device") == "cpu", "Incomplete/non-CPU feature cache")
    require(sha(path) == record["conditioning_sha256"], f"Feature digest mismatch: {spec['id']}")
    verify_encoder(record["encoder"])
    return record


def validate_features(spec: dict, spectral, record: dict):
    import torch

    cache = torch.load(spec["directory"] / "features.pt", map_location="cpu", weights_only=False)
    for key in ("input_sha256", "input_audio_sha256"):
        require(cache[key] == spec[key], f"Tensor cache input mismatch: {spec['id']}")
    verify_encoder(cache["encoder"])
    features = cache["conditioning"]
    frames = (spec["samples"] + spectral.hop - 1) // spectral.hop + spectral.window_size // spectral.hop
    require(tuple(features.shape) == (1, 768, frames) and list(features.shape) == record["shape"], "Feature shape mismatch")
    require(features.device.type == "cpu" and features.dtype == torch.float32 and bool(torch.isfinite(features).all()),
            "Invalid feature tensor")
    return features


def save_features(spec: dict, features, metadata: dict, reused_sha: str | None = None) -> None:
    import torch

    path = spec["directory"] / "features.pt"
    temporary = path.with_suffix(".pt.tmp")
    cache = {"conditioning": features.detach().cpu().contiguous(), "encoder": metadata,
             "input_sha256": spec["input_sha256"], "input_audio_sha256": spec["input_audio_sha256"]}
    torch.save(cache, temporary)
    temporary.replace(path)
    record = {key: spec[key] for key in ("input_sha256", "input_audio_sha256", "original_sha256", "samples")}
    record.update(complete=True, device="cpu", shape=list(features.shape), encoder=metadata,
                  conditioning_sha256=sha(path), reused_from_cache_sha256=reused_sha)
    write_json(spec["directory"] / "features.json", record)


def configure_torch_cpu() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)


def checkpoint_identity() -> dict:
    identity = {"step": CHECKPOINT_STEP, "weights": "EMA", "sha256": CHECKPOINT_SHA,
                "conditioning_architecture": CONDITIONING_ARCHITECTURE}
    if CONDITIONING_ARCHITECTURE == "gaussian-fourier-v1":
        identity["frequency_width"] = 1
    return identity


def configure_snapshot(checkpoint: Path, expected_sha: str, run_directory: Path) -> None:
    """Read a trusted local snapshot without changing training or model weights."""
    global CHECKPOINT, CHECKPOINT_SHA, CHECKPOINT_STEP, CONDITIONING_ARCHITECTURE, RUN
    import torch

    require(sha(checkpoint) == expected_sha, "Downloaded checkpoint digest mismatch")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    architecture = saved["config"].get("conditioning_architecture")
    require(architecture == "scm-positional-adn-v1", "Custom snapshots require positional/ADN conditioning")
    require(isinstance(saved["step"], int) and saved["step"] > 0 and "ema" in saved,
            "Incomplete checkpoint")
    require(saved["config"]["sample_rate"] == SR, "Checkpoint sample rate mismatch")
    for key, value in {"repository": ENCODER_REPOSITORY, "revision": ENCODER_REVISION}.items():
        require(saved["config"]["wavjepa"][key] == value, "Checkpoint encoder mismatch")
    CHECKPOINT, CHECKPOINT_SHA = checkpoint.resolve(), expected_sha
    CHECKPOINT_STEP, CONDITIONING_ARCHITECTURE = saved["step"], architecture
    RUN = run_directory.resolve()


def reuse_features(specs: list[dict], source: Path) -> None:
    for spec in specs:
        if feature_record(spec, required=False) is not None:
            continue
        original_input = source / spec["id"] / "input.wav"
        require(np.array_equal(audio(original_input, spec["samples"]), spec["wave"]),
                f"Cached input waveform mismatch: {spec['id']}")
        original_digest = sha(original_input)
        original = dict(spec, directory=source / spec["id"], input_sha256=original_digest)
        feature_record(original)
        # Preserve the exact source container too: equal float samples can have
        # different WAV-header bytes across libsndfile versions/write times.
        shutil.copyfile(original_input, spec["input"])
        spec["input_sha256"] = original_digest
        input_record = read_json(spec["directory"] / "input.json")
        input_record["input_sha256"] = original_digest
        write_json(spec["directory"] / "input.json", input_record)
        for name in ("features.pt", "features.json"):
            shutil.copyfile(original["directory"] / name, spec["directory"] / name)
        feature_record(spec)


def prepare_features(specs: list[dict]) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    configure_torch_cpu()
    import torch
    from rmfsr.spectral import Spectral

    spectral = Spectral(SR)
    pending = []
    for spec in specs:
        record = feature_record(spec, required=False)
        if record is not None:
            validate_features(spec, spectral, record)
        elif spec["index"] == 7:
            require(sha(WIND_CACHE) == WIND_CACHE_SHA and sha(WIND_INPUT) == WIND_INPUT_SHA, "Wind cache digest mismatch")
            require(np.array_equal(audio(WIND_INPUT, spec["samples"]), spec["wave"]), "Wind cache audio mismatch")
            cache = torch.load(WIND_CACHE, map_location="cpu", weights_only=False)
            require(cache["input_sha256"] == WIND_INPUT_SHA, "Wind cache input digest mismatch")
            verify_encoder(cache["encoder"])
            save_features(spec, cache["conditioning"], cache["encoder"], WIND_CACHE_SHA)
            validate_features(spec, spectral, feature_record(spec))
            print(json.dumps({"sample": spec["id"], "status": "reused verified wind features"}), flush=True)
        else:
            pending.append(spec)
    if pending:
        from rmfsr.wavjepa import FrozenWavJEPA

        print(json.dumps({"status": "loading pinned encoder from local cache", "remaining_samples": len(pending)}), flush=True)
        encoder = FrozenWavJEPA(device="cpu")
        verify_encoder(encoder.metadata())
        for spec in pending:
            started = time.monotonic()
            wave = torch.from_numpy(spec["wave"])[None]
            frames = (spec["samples"] + spectral.hop - 1) // spectral.hop + spectral.window_size // spectral.hop
            with torch.inference_mode():
                features = encoder(wave, spectral, frames).detach().cpu()
            save_features(spec, features, encoder.metadata())
            validate_features(spec, spectral, feature_record(spec))
            print(json.dumps({"sample": spec["id"], "status": "features complete", "seconds": time.monotonic() - started}), flush=True)
        del encoder, features
        gc.collect()


def output_record(spec: dict, nfe: int, required: bool = True) -> dict | None:
    path = spec["directory"] / f"output_nfe{nfe}.wav"
    metadata = path.with_suffix(".json")
    if not path.exists() and not metadata.exists() and not required:
        return None
    require(path.exists() and metadata.exists(), f"Missing/incomplete output: {spec['id']}/NFE{nfe}")
    record = read_json(metadata)
    expected = {"complete": True, "checkpoint_sha256": CHECKPOINT_SHA, "training_step": CHECKPOINT_STEP,
                "weights": "EMA",
                "input_sha256": spec["input_sha256"], "input_audio_sha256": spec["input_audio_sha256"],
                "conditioning_sha256": feature_record(spec)["conditioning_sha256"],
                "noise_seed": SEED, "nfe": nfe, "device": DEVICE, "samples": spec["samples"],
                "sample_rate": SR, "chunk_frames": None, "input_gain_db": 0}
    if CONDITIONING_ARCHITECTURE == "gaussian-fourier-v1":
        expected.update(frequency_width=1, frequency_rms=FREQUENCY_RMS)
    else:
        expected["conditioning_architecture"] = CONDITIONING_ARCHITECTURE
    for key, value in expected.items():
        require(record.get(key) == value, f"Output provenance mismatch: {spec['id']}/NFE{nfe}/{key}")
    require(sha(path) == record["output_sha256"], f"Output digest mismatch: {spec['id']}/NFE{nfe}")
    audio(path, spec["samples"])
    return record


def save_output_record(spec: dict, nfe: int, seconds: float, reused: bool) -> None:
    path = spec["directory"] / f"output_nfe{nfe}.wav"
    record = {"complete": True, "checkpoint_sha256": CHECKPOINT_SHA, "training_step": CHECKPOINT_STEP,
              "weights": "EMA", "conditioning_architecture": CONDITIONING_ARCHITECTURE,
              "input_sha256": spec["input_sha256"], "input_audio_sha256": spec["input_audio_sha256"],
              "conditioning_sha256": feature_record(spec)["conditioning_sha256"],
              "output_sha256": sha(path), "nfe": nfe, "noise_seed": SEED, "device": DEVICE,
              "seconds": seconds, "samples": spec["samples"], "sample_rate": SR,
              "chunk_frames": None, "input_gain_db": 0, "reused_verified_output": reused}
    if CONDITIONING_ARCHITECTURE == "gaussian-fourier-v1":
        record.update(frequency_width=1, frequency_rms=FREQUENCY_RMS)
    write_json(path.with_suffix(".json"), record)


def reuse_wind(spec: dict) -> None:
    record = read_json(WIND_OUTPUTS / "results.json")
    expected = {"training_step": 85000, "weights": "EMA", "nominal_frequency_width": 1,
                "frequency_rms": FREQUENCY_RMS, "checkpoint_sha256": CHECKPOINT_SHA,
                "cached_conditioning_sha256": WIND_CACHE_SHA, "input_sha256": WIND_INPUT_SHA,
                "noise_seed": SEED, "input_gain_db": 0, "chunk_frames": None, "complete": True}
    for key, value in expected.items():
        require(record.get(key) == value, f"Wind output provenance mismatch: {key}")
    require(feature_record(spec)["reused_from_cache_sha256"] == WIND_CACHE_SHA, "Wind features not from verified cache")
    require(np.array_equal(audio(WIND_INPUT, spec["samples"]), spec["wave"]), "Wind input mismatch")
    for nfe in NFES:
        if output_record(spec, nfe, required=False) is not None:
            continue
        source = (WIND_50 if nfe == 50 else WIND_OUTPUTS) / f"output_nfe{nfe}.wav"
        require(sha(source) == WIND_HASHES[nfe], f"Wind output changed: NFE{nfe}")
        audio(source, spec["samples"])
        if nfe == 50:
            extra = read_json(WIND_50 / "results.json")
            for key, value in {"checkpoint_sha256": CHECKPOINT_SHA, "training_step": 85000, "weights": "EMA",
                                "frequency_width": 1, "input_sha256": WIND_INPUT_SHA,
                                "conditioning_sha256": WIND_CACHE_SHA, "noise_seed": SEED, "nfe": 50,
                                "output_sha256": WIND_HASHES[50], "complete": True}.items():
                require(extra.get(key) == value, f"Wind NFE50 provenance mismatch: {key}")
            seconds = extra["seconds"]
        else:
            seconds = record["timings"][f"output_nfe{nfe}"]
        shutil.copyfile(source, spec["directory"] / source.name)
        save_output_record(spec, nfe, seconds, reused=True)
        print(json.dumps({"sample": spec["id"], "nfe": nfe, "status": "reused verified output"}), flush=True)


def infer_missing(specs: list[dict]) -> None:
    configure_torch_cpu()
    import torch
    from rmfsr.flow import pink_noise_like, sample
    from rmfsr.model import model_from_checkpoint
    from rmfsr.spectral import Spectral

    spectral = Spectral(SR)
    for spec in specs:
        validate_features(spec, spectral, feature_record(spec))
        if spec["index"] == 7 and CONDITIONING_ARCHITECTURE == "gaussian-fourier-v1" and DEVICE == "cpu":
            reuse_wind(spec)
    pending = [(spec, [nfe for nfe in NFES if output_record(spec, nfe, required=False) is None]) for spec in specs]
    pending = [(spec, nfes) for spec, nfes in pending if nfes]
    if not pending:
        return
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    require(DEVICE != "mps" or torch.backends.mps.is_available(), "Apple GPU inference is unavailable")
    require(checkpoint["step"] == CHECKPOINT_STEP, "Unexpected checkpoint step")
    model = model_from_checkpoint(checkpoint).cpu().eval().requires_grad_(False)
    model.load_state_dict(checkpoint["ema"], strict=True)
    if CONDITIONING_ARCHITECTURE == "gaussian-fourier-v1":
        rms = float(model.embedding.frequencies.double().square().mean().sqrt())
        require(abs(rms - FREQUENCY_RMS) < 1e-7, "Not the width-1 EMA")
    else:
        require(model.architecture()["conditioning_architecture"] == CONDITIONING_ARCHITECTURE,
                "Model conditioning architecture mismatch")
    for key, value in {"repository": ENCODER_REPOSITORY, "revision": ENCODER_REVISION}.items():
        require(checkpoint["config"]["wavjepa"][key] == value, "Checkpoint encoder mismatch")
    model.to(DEVICE)
    require(all(parameter.device.type == DEVICE for parameter in model.parameters()), "Inference device mismatch")
    del checkpoint
    for spec, nfes in pending:
        features = validate_features(spec, spectral, feature_record(spec)).to(DEVICE)
        y = spectral.encode(torch.from_numpy(spec["wave"]))
        noise = pink_noise_like(y, torch.Generator().manual_seed(SEED)).to(DEVICE)
        y = y.to(DEVICE)
        for nfe in nfes:
            started = time.monotonic()
            print(json.dumps({"sample": spec["id"], "nfe": nfe, "status": f"{DEVICE} inference"}), flush=True)
            with torch.inference_mode():
                predicted = sample(model, y, steps=nfe, noise=noise.clone(), conditioning=features)
                output = spectral.decode(predicted.cpu(), spec["samples"])[0].numpy()
            require(output.shape == spec["wave"].shape and bool(np.isfinite(output).all()), "Invalid prediction")
            save_float(spec["directory"] / f"output_nfe{nfe}.wav", output)
            seconds = time.monotonic() - started
            save_output_record(spec, nfe, seconds, reused=False)
            print(json.dumps({"sample": spec["id"], "nfe": nfe, "seconds": seconds, "complete": True}), flush=True)


def spectrogram(wave: np.ndarray, destination: Path) -> None:
    from matplotlib import colormaps

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


def build(specs: list[dict]) -> dict:
    require([spec["index"] for spec in specs] == list(range(1, 9)), "Page must contain all eight inputs")
    sources, waves = {}, {}
    for spec in specs:
        key = spec["id"]
        sources[key] = {"input": spec["input"]}
        for nfe in NFES:
            output_record(spec, nfe)
            sources[key][f"nfe{nfe}"] = spec["directory"] / f"output_nfe{nfe}.wav"
        for name, path in sources[key].items():
            waves[key, name] = audio(path, spec["samples"])
    maximum = max(float(np.max(np.abs(wave))) for wave in waves.values())
    gain = 1.0 if maximum < 1.0 else 0.95 / maximum
    manifest = {
        "schema_version": 1,
        "checkpoint": checkpoint_identity(),
        "nfes": NFES, "sample_rate": SR,
        "spectrogram": {
            "n_fft": 512, "hop_length": 128, "min_db": -80, "max_db": 0,
            "window": "periodic Hann", "frequency_scale": "linear", "color_map": "viridis", "colormap": "viridis",
            "width": 640, "height": 240,
            "reference": "20 log10(abs(STFT) / sum(Hann window)), reference amplitude 1; common absolute -80 to 0 dB scale for all clips. No per-clip normalization. Spectrograms use the published PCM audio; 0 to 8 kHz vertically, time horizontally.",
        },
        "playback": {"format": "WAV PCM16", "common_gain_db": float(20 * np.log10(gain)), "independent_normalization": False},
        "inference": {"device": DEVICE, "noise_seed": SEED, "same_initial_noise_per_sample": True,
                      "chunk_frames": None, "conditioning": "Frozen wavJEPA, offline aligned features",
                      "encoder_repository": ENCODER_REPOSITORY, "encoder_revision": ENCODER_REVISION,
                      "causal_streaming_evaluation": False},
        "samples": [],
    }
    for spec in specs:
        key = spec["id"]
        sample = {"id": key, "title": spec["title"], "description": spec["description"],
                  "duration_seconds": spec["samples"] / SR, "references": [], "outputs": [],
                  "source_note": "Original recording from the paper demo. No clean reference is available. All restoration outputs shown are ours.",
                  "source_url": spec["source_url"], "source_recording_id": spec["source_id"],
                  "source_sample_rate": spec["source_sample_rate"], "original_input_sha256": spec["original_sha256"],
                  "input_audio_sha256": spec["input_audio_sha256"],
                  "conditioning_sha256": feature_record(spec)["conditioning_sha256"]}
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
            audio_digest, image_digest = sha(destination), sha(DOCS / relative_image)
            item = {"audio": relative_audio + f"?v={audio_digest[:16]}",
                    "spectrogram": relative_image + f"?v={image_digest[:16]}",
                    "source_sha256": sha(source), "audio_sha256": audio_digest,
                    "spectrogram_sha256": image_digest}
            if name.startswith("nfe"):
                nfe = int(name[3:])
                item.update(nfe=nfe, label=f"NFE {nfe}")
                sample["outputs"].append(item)
            else:
                item.update(kind="input", label="Damaged input")
                sample["references"].append(item)
        require([item["nfe"] for item in sample["outputs"]] == NFES, "Wrong output order")
        manifest["samples"].append(sample)
    write_json(DOCS / "results.json", manifest)
    return manifest


def main() -> None:
    global DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-features-only", action="store_true", help="Prepare inputs and cache pinned wavJEPA features on CPU, then exit")
    mode.add_argument("--infer-missing", action="store_true", help="Resume missing NFE outputs locally using prepared features")
    parser.add_argument("--inference-only", action="store_true", help="Skip publishing after --infer-missing")
    parser.add_argument("--samples", help="Comma-separated indices 1..8, for feature preparation or inference-only")
    parser.add_argument("--checkpoint", type=Path, help="Trusted local production checkpoint")
    parser.add_argument("--checkpoint-sha256", help="Expected digest recorded when downloading the checkpoint")
    parser.add_argument("--run-directory", type=Path, help="Separate local artifacts directory for this snapshot")
    parser.add_argument("--reuse-features-from", type=Path, help="Existing eight-example directory with verified encoder caches")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu", help="Local inference device; no remote GPU use")
    arguments = parser.parse_args()
    DEVICE = arguments.device
    custom = (arguments.checkpoint, arguments.checkpoint_sha256, arguments.run_directory)
    require(not any(custom) or all(custom), "A custom snapshot requires checkpoint, SHA-256, and run directory")
    if arguments.checkpoint:
        configure_snapshot(*custom)
    require(not arguments.inference_only or arguments.infer_missing, "--inference-only requires --infer-missing")
    require(not arguments.samples or arguments.prepare_features_only or arguments.inference_only,
            "Subsets require --prepare-features-only or --inference-only; a published page contains all eight")
    sys.path.insert(0, str(ROOT))
    specs = load_specs()
    if arguments.samples:
        selected = [int(value) for value in arguments.samples.split(",")]
        require(len(selected) == len(set(selected)) and all(1 <= value <= 8 for value in selected), "Invalid/duplicate sample indices")
        specs = [spec for spec in specs if spec["index"] in selected]
    prepare_inputs(specs)
    if arguments.reuse_features_from:
        reuse_features(specs, arguments.reuse_features_from)
    if arguments.prepare_features_only:
        prepare_features(specs)
        print(json.dumps({"features_complete": True, "samples": len(specs)}), flush=True)
        return
    if arguments.infer_missing:
        infer_missing(specs)
    if arguments.inference_only:
        print(json.dumps({"inference_complete": True, "samples": [spec["index"] for spec in specs]}), flush=True)
        return
    manifest = build(specs)
    files = [DOCS / item[field].split("?", 1)[0] for sample in manifest["samples"]
             for item in sample["references"] + sample["outputs"] for field in ("audio", "spectrogram")]
    print(json.dumps({"complete": True, "audio_clips": len(files) // 2,
                      "assets_bytes": sum(path.stat().st_size for path in files),
                      "common_gain_db": manifest["playback"]["common_gain_db"]}), flush=True)


if __name__ == "__main__":
    main()
