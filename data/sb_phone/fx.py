"""sb_phone degradations (pure numpy/scipy + ffmpeg subprocess for codecs). All functions take/return 16 kHz mono
float32 and are deterministic given their arguments (randomness comes in via an explicit random.Random)."""
from __future__ import annotations

import math
import random
import subprocess

import numpy as np

SR = 16000
FRAME = 320  # 20 ms

# codec name -> (encoder args, container format for the encoded stream, sample rate of the coded signal)
CODECS = {
    "mulaw": (["-c:a", "pcm_mulaw"], "wav", 8000),
    "alaw": (["-c:a", "pcm_alaw"], "wav", 8000),
    "gsm": (["-c:a", "libgsm"], "gsm", 8000),
    "opus": (["-c:a", "libopus", "-application", "voip"], "ogg", 8000),  # bitrate added per clip (6-12 kbps)
    "g726": (["-c:a", "adpcm_g726"], "wav", 8000),  # bitrate added per clip (16-32 kbps)
    "speex": (["-c:a", "libspeex"], "ogg", 8000),
}


def ffmpeg_codec(x: np.ndarray, codec: str, bitrate_k: int | None = None) -> np.ndarray:
    """Encode with a real telephone/VoIP codec and decode back to 16 kHz (same length as the input)."""
    enc, fmt, sr = CODECS[codec]
    args = list(enc)
    if bitrate_k:
        args += ["-b:a", f"{bitrate_k}k"]
    pcm = np.clip(np.asarray(x, np.float32), -1, 1).tobytes()
    p1 = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
                         "-ar", str(sr), "-ac", "1", *args, "-f", fmt, "pipe:1"], input=pcm, capture_output=True)
    if p1.returncode != 0 or not p1.stdout:
        raise RuntimeError(f"ffmpeg encode {codec}: {p1.stderr.decode()[-400:]}")
    in_args = ["-f", "gsm", "-ar", str(sr)] if fmt == "gsm" else []
    p2 = subprocess.run(["ffmpeg", "-v", "error", *in_args, "-i", "pipe:0", "-f", "f32le", "-ar", str(SR), "-ac", "1",
                         "pipe:1"], input=p1.stdout, capture_output=True)
    if p2.returncode != 0:
        raise RuntimeError(f"ffmpeg decode {codec}: {p2.stderr.decode()[-400:]}")
    y = np.frombuffer(p2.stdout, dtype=np.float32).copy()
    # codecs add a small delay (opus pre-skip is handled by the decoder; gsm/g726 frame padding) -> align length
    if len(y) >= len(x):
        y = y[: len(x)]
    else:
        y = np.pad(y, (0, len(x) - len(y)))
    return y


def frame_db(x: np.ndarray) -> np.ndarray:
    k = len(x) // FRAME
    if k == 0:
        return np.array([10 * math.log10(np.mean(x ** 2) + 1e-12)])
    return 10 * np.log10(np.mean(x[: k * FRAME].reshape(k, FRAME) ** 2, axis=1) + 1e-12)


def active_mask(x: np.ndarray, rel_db: float = -30.0) -> np.ndarray:
    """Per-20ms-frame speech activity: frames within rel_db of the loudest 5% of frames."""
    e = frame_db(x)
    ref = np.percentile(e, 95)
    return e > ref + rel_db


def active_power(x: np.ndarray) -> float:
    m = active_mask(x)
    k = len(m)
    fr = x[: k * FRAME].reshape(k, FRAME) if k * FRAME <= len(x) and k > 0 else x.reshape(1, -1)
    sel = fr[m] if m.any() else fr
    return float(np.mean(sel ** 2) + 1e-12)


def fit_length(n: np.ndarray, L: int, rng: random.Random) -> np.ndarray:
    """Loop / crop a background signal to length L (random offset)."""
    n = np.asarray(n, np.float32)
    if len(n) == 0:
        return np.zeros(L, np.float32)
    if len(n) < L:
        n = np.tile(n, int(math.ceil(L / len(n))) + 1)
    off = rng.randrange(0, len(n) - L + 1)
    return n[off: off + L]


def mix_at_snr(x: np.ndarray, bg: np.ndarray, snr_db: float, rng: random.Random) -> np.ndarray:
    """Add a background so that (speech active power) / (background power) = snr_db."""
    b = fit_length(bg, len(x), rng)
    pb = float(np.mean(b ** 2) + 1e-12)
    ps = active_power(x)
    g = math.sqrt(ps / (pb * 10 ** (snr_db / 10)))
    return (x + g * b).astype(np.float32)


def apply_rir(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolve with a room impulse response, aligned to the direct path, at the input's active level."""
    from scipy.signal import fftconvolve

    h = np.asarray(h, np.float64)
    p = int(np.argmax(np.abs(h)))
    h = h[max(0, p - 16):]
    y = fftconvolve(x.astype(np.float64), h)[: len(x)]
    g = math.sqrt(active_power(x) / (active_power(y.astype(np.float32)) + 1e-12))
    return (y * g).astype(np.float32)


def dropouts(x: np.ndarray, frac: float, rng: random.Random, plc_prob: float = 0.0) -> tuple[np.ndarray, dict]:
    """Packet loss: zero out (or, with plc_prob, repeat the previous chunk) 20-60 ms chunks inside speech-active
    regions until `frac` of the active time is lost. Returns (audio, facts)."""
    y = x.copy()
    act = active_mask(x)
    idx = np.where(act)[0]
    if len(idx) == 0 or frac <= 0:
        return y, {"lost_frac": 0.0, "n_drops": 0}
    target = frac * len(idx) * FRAME
    lost = np.zeros(len(x), bool)
    n = 0
    tries = 0
    while lost.sum() < target and tries < 2000:
        tries += 1
        f = int(rng.choice(idx))
        L = int(rng.uniform(0.02, 0.06) * SR)
        a = f * FRAME + rng.randrange(0, FRAME)
        b = min(len(x), a + L)
        if lost[a:b].any():
            continue
        lost[a:b] = True
        if plc_prob and rng.random() < plc_prob and a >= (b - a):
            y[a:b] = y[a - (b - a): a]
        else:
            y[a:b] = 0.0
        n += 1
    act_samples = np.repeat(act, FRAME)[: len(x)]
    lf = float(lost[: len(act_samples)][act_samples].mean()) if act_samples.any() else 0.0
    return y, {"lost_frac": round(lf, 4), "n_drops": n}


def hard_clip(x: np.ndarray, clip_frac: float) -> tuple[np.ndarray, dict]:
    """Overdrive: scale up so that `clip_frac` of the active samples exceed full scale, then clip at +-1."""
    m = np.zeros(len(x), bool)
    rep = np.repeat(active_mask(x), FRAME)[: len(x)] if len(x) >= FRAME else np.ones(len(x), bool)
    m[: len(rep)] = rep
    a = np.abs(x[m])
    if len(a) == 0:
        a = np.abs(x)
    thr = float(np.quantile(a, 1 - clip_frac)) + 1e-9
    y = np.clip(x / thr, -1.0, 1.0).astype(np.float32)
    return y, {"clipped_frac": round(float(np.mean(np.abs(y) >= 0.999)), 4)}


def set_peak(x: np.ndarray, peak_db: float) -> np.ndarray:
    p = float(np.max(np.abs(x)) + 1e-9)
    return (x * (10 ** (peak_db / 20) / p)).astype(np.float32)


def bandlimit(x: np.ndarray, lo: float = 300.0, hi: float = 3400.0) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def render(x: np.ndarray, recipe: dict, bgs: dict, rng: random.Random) -> tuple[np.ndarray, dict]:
    """Apply a recipe (see labels.sample_recipe) in a physically sensible order:
    room (RIR) -> background (noise / music / talker) -> level / clipping -> codec -> packet loss -> output level.
    `bgs` maps 'noise' / 'music' / 'talker' / 'rir' to the arrays chosen for this clip."""
    facts = {}
    y = np.asarray(x, np.float32)
    if recipe.get("rir"):
        y = apply_rir(y, bgs["rir"])
    for kind in ("noise", "music", "talker"):
        if recipe.get(kind):
            y = mix_at_snr(y, bgs[kind], recipe[kind]["snr"], rng)
    y = set_peak(y, -3.0)
    if recipe.get("clip"):
        y, f = hard_clip(y, recipe["clip"])
        facts.update(f)
    if recipe.get("lowvol_db") is not None:
        y = set_peak(y, recipe["lowvol_db"])
    if recipe.get("codec"):
        y = ffmpeg_codec(y, recipe["codec"], recipe.get("bitrate_k"))
    if recipe.get("drop"):
        y, f = dropouts(y, recipe["drop"], rng, plc_prob=recipe.get("plc", 0.0))
        facts.update(f)
    if recipe.get("lowvol_db") is None:
        y = set_peak(y, recipe.get("peak_db", -3.0))
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 0.99:
        y = (y * (0.99 / peak)).astype(np.float32)
    facts["peak_dbfs"] = round(20 * math.log10(float(np.max(np.abs(y)) + 1e-9)), 2)
    return y.astype(np.float32), facts
