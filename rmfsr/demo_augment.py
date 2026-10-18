"""User-approved moderate restoration treatments, version demo-eight-v1.

The eight families follow the author's demo labels, not unpublished author
parameters. Each family has probability 1/8. Bounds surround the listening
previews approved by the user; no independent chain of destructive effects.
The clean target is unchanged by this module. All random draws use the caller's
generator, preserving step-indexed sampling across workers and restarts.
"""
import subprocess

import numpy as np
from scipy import signal

from .degradations import image_rir

PROFILE = 'demo-eight-v1'
FAMILIES = ('bandwidth_noise', 'distance_airconditioner', 'bandwidth_codec',
            'reverb', 'distortion', 'frequency_imbalance',
            'wind_clipping_suppression', 'very_narrow_bandwidth')


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def band(x, lo, hi, sr):
    taps = signal.firwin(257, [lo, hi], pass_zero=False, fs=sr)
    return signal.fftconvolve(x, taps, mode='same')


def mix_noise(x, noise, snr):
    noise = np.asarray(noise, dtype=np.float64) - np.mean(noise)
    # Silent corpus segments stay silent rather than producing infinities.
    return x + noise * (rms(x) / (max(rms(noise), 1e-12) * 10 ** (snr / 20)))


def reverberate(x, rng, sr, distant=False, reverb_only=False):
    room = rng.uniform([4.5, 3.5, 2.6], [6.5, 5.5, 3.2])
    if distant:
        mic = room * rng.uniform([.10, .10, .40], [.17, .20, .50])
        source = room * rng.uniform([.70, .65, .40], [.82, .80, .55])
    else:
        mic = room * rng.uniform([.18, .18, .40], [.28, .30, .50])
        source = room * rng.uniform([.55, .50, .40], [.72, .72, .55])
    reflection = rng.uniform(.60, .70) if reverb_only else rng.uniform(.45, .58)
    order = 6 if reverb_only else 4
    rir, _, _, _, distance = image_rir(rng, sr, room=room, source=source,
                                       mic=mic, reflection=reflection, order=order)
    # Remove the integer propagation delay to keep the speech target aligned.
    delay = int(round(distance / 343 * sr))
    wave = signal.fftconvolve(x, rir)[delay:delay + len(x)]
    wave *= rms(x) / max(rms(wave), 1e-12)
    return wave, dict(room_m=room.tolist(), reflection=float(reflection),
                      source_distance_m=float(distance), direct_delay_trim=delay)


def mp3_roundtrip(x, sr, executable, bitrate):
    if not executable:
        raise ValueError('demo-eight-v1 requires ffmpeg for the codec family')
    # Avoid incidental PCM clipping inside the codec for high-crest-factor audio.
    gain = min(1., .9 / max(float(np.max(np.abs(x))), 1e-12))
    encoded_input = x * gain
    encoded = subprocess.run(
        [str(executable), '-v', 'error', '-threads', '1', '-f', 'f32le', '-ar', str(sr),
         '-ac', '1', '-i', 'pipe:0', '-c:a', 'libmp3lame', '-b:a', f'{bitrate}k',
         '-threads', '1', '-f', 'mp3', 'pipe:1'],
        input=np.asarray(encoded_input, dtype='<f4').tobytes(),
        capture_output=True, check=True, timeout=20).stdout
    decoded = subprocess.run(
        [str(executable), '-v', 'error', '-threads', '1', '-f', 'mp3', '-i', 'pipe:0',
         '-ar', str(sr), '-ac', '1', '-threads', '1', '-f', 'f32le', 'pipe:1'],
        input=encoded, capture_output=True, check=True, timeout=20).stdout
    wave = np.frombuffer(decoded, dtype='<f4').astype(np.float64)
    corr = signal.correlate(wave, encoded_input, mode='full', method='fft')
    lags = signal.correlation_lags(len(wave), len(encoded_input))
    valid = (lags >= 0) & (lags <= sr // 4)
    delay = int(lags[valid][np.argmax(corr[valid])]) if rms(x) > 1e-9 else 0
    if len(wave) - delay < len(x):
        raise ValueError('Codec alignment would truncate speech; refusing silent padding')
    return wave[delay:delay + len(x)] / gain, delay


def apply_treatment(clean, noise, rng, sr=16000, ffmpeg=None, family=None):
    """Produce one damaged input; optional family is for focused local checks."""
    if sr != 16000:
        raise ValueError('demo-eight-v1 is defined for 16 kHz speech')
    x = np.asarray(clean, dtype=np.float64).copy()
    noise = np.asarray(noise, dtype=np.float64)
    if x.ndim != 1 or not x.size or noise.shape != x.shape:
        raise ValueError('Speech and noise must be same-length nonempty mono audio')
    if not np.isfinite(x).all() or not np.isfinite(noise).all():
        raise ValueError('Nonfinite augmentation input')
    family = str(rng.choice(FAMILIES)) if family is None else family
    if family not in FAMILIES:
        raise ValueError('Unknown demo family')
    info = dict(profile=PROFILE, family=family)
    time = np.arange(len(x)) / sr
    if family == 'bandwidth_noise':
        lo, hi = rng.uniform(180, 300), rng.uniform(4300, 5500)
        snr = rng.uniform(8, 14)
        y = mix_noise(band(x, lo, hi, sr), noise, snr)
        info.update(bandpass_hz=[lo, hi], snr_db=snr, noise='DNS')
    elif family == 'distance_airconditioner':
        y, room = reverberate(x, rng, sr, distant=True)
        y = band(y, rng.uniform(80, 140), rng.uniform(4000, 5000), sr)
        cutoff, hum = rng.uniform(1000, 1600), float(rng.choice([50, 60]))
        fan = signal.sosfilt(signal.butter(2, cutoff, fs=sr, output='sos'), rng.normal(size=len(x)))
        fan += .035 * np.sin(2 * np.pi * hum * time) + .02 * np.sin(4 * np.pi * hum * time)
        snr = rng.uniform(8, 14)
        y = mix_noise(y, fan, snr)
        info.update(room=room, snr_db=snr, noise='synthetic_fan', hum_hz=hum)
    elif family == 'bandwidth_codec':
        lo, hi = rng.uniform(250, 350), rng.uniform(3000, 3800)
        bitrate = int(rng.choice([24, 32]))
        y, delay = mp3_roundtrip(band(x, lo, hi, sr), sr, ffmpeg, bitrate)
        info.update(bandpass_hz=[lo, hi], bitrate_kbps=bitrate, codec_delay_trim=delay)
    elif family == 'reverb':
        y, room = reverberate(x, rng, sr, reverb_only=True)
        info.update(room=room)
    elif family == 'distortion':
        factor = rng.uniform(2.5, 3.2)
        threshold = max(factor * rms(x), 1e-12)
        y = threshold * np.tanh(x / threshold)
        info.update(saturation_threshold_times_rms=factor)
    elif family == 'frequency_imbalance':
        frequencies = [0, 125, 500, 1800, 3500, sr / 2]
        gains = np.array([6, 6, 3, 0, -5, -9]) + rng.uniform(-1, 1, 6)
        # Some microphones emphasize highs instead; same bounded tilt severity.
        if rng.random() < .5:
            gains = -gains
        taps = signal.firwin2(257, frequencies, 10 ** (gains / 20), fs=sr)
        y = signal.fftconvolve(x, taps, mode='same')
        info.update(eq_gains_db=gains.tolist())
    elif family == 'wind_clipping_suppression':
        cutoff, gust_rate = rng.uniform(180, 280), rng.uniform(.45, .9)
        wind = signal.sosfilt(signal.butter(2, cutoff, fs=sr, output='sos'), rng.normal(size=len(x)))
        envelope = .15 + .85 * (.5 + .5 * np.sin(2 * np.pi * gust_rate * time + rng.uniform(-np.pi, np.pi))) ** 2
        snr = rng.uniform(8, 14)
        y = mix_noise(x, wind * envelope, snr)
        ceiling = max(rng.uniform(2.7, 3.2) * rms(y), 1e-12)
        info.update(clipped_fraction=float(np.mean(np.abs(y) > ceiling)))
        y = np.clip(y, -ceiling, ceiling)
        # Short silent fixtures also need a valid STFT window.
        window = min(320, len(y))
        _, _, z = signal.stft(y, sr, nperseg=window, noverlap=window // 2)
        power = np.abs(z) ** 2
        floor = np.quantile(power, .25, axis=1, keepdims=True)
        factor = rng.uniform(1.5, 2.)
        z *= np.sqrt(np.clip(1 - factor * floor / np.maximum(power, 1e-12), .22 ** 2, 1))
        _, y = signal.istft(z, sr, nperseg=window, noverlap=window // 2)
        y = y[:len(x)]
        info.update(snr_before_suppression_db=snr, noise='synthetic_wind', subtraction_factor=factor)
    else:
        lo, hi = rng.uniform(400, 600), rng.uniform(1600, 2000)
        y = band(x, lo, hi, sr)
        info.update(bandpass_hz=[lo, hi])
    # Center actual training levels on the approved -25 dBFS listening previews.
    dbfs = rng.uniform(-28, -22)
    y *= 10 ** (dbfs / 20) / max(rms(y), 1e-12)
    attenuation = min(1., .95 / max(float(np.max(np.abs(y))), 1e-12))
    y *= attenuation
    if y.shape != x.shape or not np.isfinite(y).all():
        raise ValueError('Invalid demo augmentation output')
    info.update(requested_rms_dbfs=dbfs, peak_attenuation=attenuation)
    return y.astype(np.float32), info
