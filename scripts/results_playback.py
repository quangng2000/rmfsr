"""Create optional RMS-matched playback without changing inference outputs."""
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def levels(wave):
    rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
    return {"rms_dbfs": float(20 * np.log10(max(rms, 1e-15))),
            "rms": rms, "peak": float(np.max(np.abs(wave)))}


def load_sources(sample, directories):
    docs, source = directories
    tracks = sample["references"] + sample["outputs"]
    waves = []
    for track in tracks:
        name = f"output_nfe{track['nfe']}.wav" if "nfe" in track else "input.wav"
        path = source / sample["id"] / name
        if digest(path) != track["source_sha256"]:
            raise ValueError(f"Raw audio provenance mismatch: {path}")
        wave, rate = sf.read(path, dtype="float32")
        if rate != 16000 or wave.ndim != 1 or not np.isfinite(wave).all() or levels(wave)["rms"] < 1e-12:
            raise ValueError(f"Invalid or silent source: {path}")
        for field in ("audio", "spectrogram"):
            if digest(docs / track[field].split("?", 1)[0]) != track[f"{field}_sha256"]:
                raise ValueError(f"Published asset digest mismatch: {track[field]}")
        waves.append(wave)
    return tracks, waves


def export_track(track, wave, context):
    docs, gain, render_spectrogram = context
    audio_path = Path(track["audio"].split("?", 1)[0])
    image_path = Path(track["spectrogram"].split("?", 1)[0])
    audio_path = audio_path.with_stem(audio_path.stem + "-rms-matched")
    image_path = image_path.with_stem(image_path.stem + "-rms-matched")
    played = wave.astype(np.float64) * gain
    sf.write(docs / audio_path, played, 16000, subtype="PCM_16")
    exported, _ = sf.read(docs / audio_path, dtype="float32")
    if levels(exported)["peak"] >= 1 or np.max(np.abs(exported - played)) > 1 / 32768 + 1e-7:
        raise ValueError(f"Invalid PCM export: {audio_path}")
    render_spectrogram(exported, docs / image_path)
    audio_sha, image_sha = digest(docs / audio_path), digest(docs / image_path)
    track["matched"] = {"audio": f"{audio_path}?v={audio_sha[:16]}",
                        "spectrogram": f"{image_path}?v={image_sha[:16]}", "audio_sha256": audio_sha,
                        "spectrogram_sha256": image_sha, "gain_db": float(20 * np.log10(gain)),
                        "source_rms_dbfs": levels(wave)["rms_dbfs"], **levels(exported)}


def match_sample(sample, directories, render_spectrogram):
    tracks, waves = load_sources(sample, directories)
    length = round(sample["duration_seconds"] * 16000)
    if len(sample["references"]) != 1 or sample["references"][0]["kind"] != "input" or any(len(w) != length for w in waves):
        raise ValueError(f"Unexpected input reference or duration: {sample['id']}")
    reference = levels(waves[0])["rms"]
    gains = [reference / levels(wave)["rms"] for wave in waves]
    safety = min(1.0, 0.95 / max(levels(w)["peak"] * g for w, g in zip(waves, gains)))
    for track, wave, gain in zip(tracks, waves, gains):
        export_track(track, wave, (directories[0], gain * safety, render_spectrogram))
    rms_values = [track["matched"]["rms_dbfs"] for track in tracks]
    sample["rms_matched"] = {"reference": "damaged input", "source_reference_rms_dbfs": levels(waves[0])["rms_dbfs"],
                             "common_safety_gain_db": float(20 * np.log10(safety)),
                             "maximum_rms_spread_db": max(rms_values) - min(rms_values)}
    if sample["rms_matched"]["maximum_rms_spread_db"] > 0.01:
        raise ValueError(f"RMS matching failed: {sample['id']}")


def add_rms_playback(manifest, docs, source, render_spectrogram):
    for sample in manifest["samples"]:
        match_sample(sample, (docs, source), render_spectrogram)
    manifest["playback"]["default_mode"] = "original"
    manifest["playback"]["rms_matched"] = {
        "method": "Match whole-clip RMS to each example's damaged input, then apply one common safety gain to all five clips.",
        "peak_limit": 0.95, "scope": "per example", "reference": "damaged input",
        "spectrograms": "Rendered from the selected playback audio, using the original absolute dB scale.",
        "note": "Playback gain only; inference and original audio remain unchanged. Equal RMS is not equal perceived loudness or a quality score."}
    return manifest
