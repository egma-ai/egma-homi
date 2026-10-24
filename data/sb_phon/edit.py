"""sb_phon signal edits and measurements (16 kHz mono float32) with the WORLD vocoder via pyworld (MIT; WORLD is
modified-BSD). Every edit is measured back on the output audio; items whose edit is too weak are dropped."""
from __future__ import annotations

import math

import numpy as np

SR = 16000
FP = 5.0  # WORLD frame period (ms)


def _pw():
    import pyworld as pw  # imported lazily so the pure-logic tests do not need it

    return pw


def analyze(x):
    pw = _pw()
    x64 = np.asarray(x, np.float64)
    f0, t = pw.dio(x64, SR, f0_floor=60.0, f0_ceil=700.0, frame_period=FP)
    f0 = pw.stonemask(x64, f0, t, SR)
    sp = pw.cheaptrick(x64, f0, t, SR)
    ap = pw.d4c(x64, f0, t, SR)
    return f0, sp, ap


def synth(f0, sp, ap, n_out=None):
    y = _pw().synthesize(np.ascontiguousarray(f0), np.ascontiguousarray(sp), np.ascontiguousarray(ap), SR, FP)
    y = np.asarray(y, np.float32)
    if n_out is not None:
        y = y[:n_out] if len(y) >= n_out else np.pad(y, (0, n_out - len(y)))
    return y


def f0_track(x):
    pw = _pw()
    x64 = np.asarray(x, np.float64)
    f0, t = pw.dio(x64, SR, f0_floor=60.0, f0_ceil=700.0, frame_period=FP)
    return pw.stonemask(x64, f0, t, SR)


def fr(t_s: float) -> int:
    return int(round(t_s * 1000.0 / FP))


def window(n: int, s: int, e: int, ramp: int = 6) -> np.ndarray:
    """Per-frame weight: 1 inside [s, e), raised-cosine ramps of `ramp` frames outside."""
    w = np.zeros(n)
    s, e = max(0, s), min(n, e)
    w[s:e] = 1.0
    for k in range(1, ramp + 1):
        v = 0.5 * (1 + math.cos(math.pi * k / (ramp + 1)))
        if s - k >= 0:
            w[s - k] = max(w[s - k], v)
        if e - 1 + k < n:
            w[e - 1 + k] = max(w[e - 1 + k], v)
    return w


def _resample_frames(f0, sp, ap, pos):
    n = len(f0)
    i0 = np.clip(np.floor(pos).astype(int), 0, n - 1)
    i1 = np.clip(i0 + 1, 0, n - 1)
    a = (pos - i0)[:, None]
    sp2 = sp[i0] * (1 - a) + sp[i1] * a
    ap2 = ap[i0] * (1 - a) + ap[i1] * a
    v0, v1 = f0[i0] > 0, f0[i1] > 0
    both = v0 & v1
    lf = np.where(both, np.exp(np.log(np.maximum(f0[i0], 1e-3)) * (1 - a[:, 0]) + np.log(np.maximum(f0[i1], 1e-3)) * a[:, 0]), 0.0)
    near = f0[np.clip(np.round(pos).astype(int), 0, n - 1)]
    f02 = np.where(both, lf, near)
    return f02, np.ascontiguousarray(sp2), np.ascontiguousarray(ap2)


def flatten(f0, k: float = 0.5):
    """Remove the sentence's declination (linear trend of log pitch) and compress the remaining pitch movement by k,
    so that no word is prominent before the emphasis edit."""
    v = f0 > 0
    idx = np.where(v)[0]
    if len(idx) < 10:
        return f0
    lf = 12 * np.log2(f0[v])
    fit = np.polyfit(idx, lf, 1)
    new = float(np.median(lf)) + k * (lf - np.polyval(fit, idx))
    out = np.zeros_like(f0)
    out[v] = 2 ** (new / 12.0)
    return out


def resynth(x, flat: float | None = None):
    """WORLD analysis + synthesis without the word edit (optionally with the same flattening)."""
    f0, sp, ap = analyze(x)
    if flat is not None:
        f0 = flatten(f0, flat)
    return synth(f0, sp, ap, len(x))


def emphasis(x, s_t: float, e_t: float, st: float, db: float, stretch: float, flat: float | None = None):
    """Boost one word [s_t, e_t): pitch +st semitones, energy +db dB, duration x stretch.
    Returns (y, time_map) where time_map maps an input time (s) to the output time."""
    f0, sp, ap = analyze(x)
    if flat is not None:
        f0 = flatten(f0, flat)
    n = len(f0)
    s, e = fr(s_t), fr(e_t)
    w = window(n, s, e)
    f0n = np.where(f0 > 0, f0 * 2 ** (st * w / 12.0), 0.0)
    spn = sp * ((10 ** (db * w / 20.0)) ** 2)[:, None]
    apn = ap
    m = int(round((e - s) * stretch))
    if abs(stretch - 1.0) > 1e-3 and e > s:
        pos = np.concatenate([np.arange(0, s), s + np.arange(0, m) * ((e - s) / m), np.arange(e, n)]).astype(np.float64)
        f0n, spn, apn = _resample_frames(f0n, spn, ap, pos)
    n_out = len(x) + int(round((m - (e - s)) * FP / 1000 * SR))
    y = synth(f0n, spn, apn, n_out)
    d = (m - (e - s)) * FP / 1000.0
    s_q, e_q = s * FP / 1000.0, e * FP / 1000.0

    def tmap(t):
        if t <= s_q:
            return t
        if t <= e_q:
            return s_q + (t - s_q) * (m / max(e - s, 1))
        return t + d

    return y, tmap


def contour_shape(pattern: str, u: np.ndarray) -> np.ndarray:
    """Pitch pattern in units of the peak-to-peak size, over normalised voiced time u in [0, 1]."""
    if pattern == "falling":
        return 0.5 - u ** 1.3
    if pattern == "rising":
        return -0.25 + np.clip((u - 0.6) / 0.4, 0, 1) ** 1.3
    if pattern == "rise-fall":
        return np.sin(np.pi * u) - 0.5
    if pattern == "fall-rise":
        return 0.5 - np.sin(np.pi * u)
    raise ValueError(pattern)


def _detrended(f0):
    v = f0 > 0
    idx = np.where(v)[0]
    lf = 12 * np.log2(f0[v])
    deg = 3 if len(idx) > 40 else 1
    fit = np.polyfit(idx, lf, deg) if len(idx) > 3 else np.array([0.0, float(np.mean(lf))])
    resid = lf - np.polyval(fit, idx)
    return v, idx, lf, resid


def contour(x, pattern: str, size_st: float, resid_scale: float = 0.5):
    """Replace the sentence's global pitch movement by `pattern` (peak-to-peak size_st semitones), keeping a
    scaled copy of the local pitch detail."""
    f0, sp, ap = analyze(x)
    v, idx, lf, resid = _detrended(f0)
    if len(idx) < 20:
        raise ValueError("too few voiced frames")
    u = (idx - idx[0]) / max(idx[-1] - idx[0], 1)
    new = float(np.median(lf)) + resid_scale * resid + size_st * contour_shape(pattern, u)
    f0n = np.zeros_like(f0)
    f0n[v] = 2 ** (new / 12.0)
    return synth(f0n, sp, ap, len(x))


def word_glide(x, words_t, target: int, size_st: float, flat_scale: float = 0.15):
    """Flatten the sentence melody and put a rising glide of size_st semitones on word `target`
    (words_t: list of (start_s, end_s))."""
    f0, sp, ap = analyze(x)
    v, idx, lf, resid = _detrended(f0)
    if len(idx) < 20:
        raise ValueError("too few voiced frames")
    base = float(np.median(lf))
    new = base + flat_scale * resid
    s, e = fr(words_t[target][0]), fr(words_t[target][1])
    vin = idx[(idx >= s) & (idx < e)]
    if len(vin) < 6:
        raise ValueError("target word has too few voiced frames")
    a, b = vin[0], vin[-1]
    g = np.zeros(len(f0))
    g[a:b + 1] = np.linspace(-0.3, 0.7, b - a + 1) * size_st
    g[b + 1:] = 0.0  # back to the flat line after the word
    new = new + g[idx]
    f0n = np.zeros_like(f0)
    f0n[v] = 2 ** (new / 12.0)
    return synth(f0n, sp, ap, len(x))


# Measurements

def semitones(f0):
    return 12 * np.log2(np.maximum(f0, 1e-3) / 100.0)


def word_stats(y, words_t, f0=None):
    """Per word: median / p90 pitch (semitones re 100 Hz) over voiced frames, RMS dB, duration, rising glide
    (median of last 30% voiced minus first 30%)."""
    if f0 is None:
        f0 = f0_track(y)
    out = []
    for s_t, e_t in words_t:
        s, e = fr(s_t), fr(e_t)
        seg = f0[max(0, s):max(s + 1, e)]
        vv = seg[seg > 0]
        a, b = int(s_t * SR), int(e_t * SR)
        rms = float(np.mean(np.asarray(y[a:max(a + 1, b)], np.float64) ** 2) + 1e-12)
        st = semitones(vv) if len(vv) else np.array([np.nan])
        k = max(1, int(round(0.3 * len(st))))
        glide = float(np.median(st[-k:]) - np.median(st[:k])) if len(vv) >= 6 else float("nan")
        out.append({"st50": float(np.median(st)), "st90": float(np.percentile(st, 90)), "db": 10 * math.log10(rms),
                    "dur": e_t - s_t, "nv": int(len(vv)), "glide": glide})
    return out


def contour_class(y, margin: float = 2.0):
    """Measure the global pitch pattern: medians of the voiced pitch in three equal time thirds + final movement.
    Returns (class or None, stats)."""
    f0 = f0_track(y)
    idx = np.where(f0 > 0)[0]
    if len(idx) < 30:
        return None, {"nv": int(len(idx))}
    st = semitones(f0[idx])
    u = (idx - idx[0]) / max(idx[-1] - idx[0], 1)
    m1, m2, m3 = (float(np.median(st[(u >= a) & (u <= b)])) for a, b in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1)))
    fin = float(np.median(st[u >= 0.85]) - np.median(st[(u >= 0.6) & (u < 0.8)]))
    stats = {"m1": round(m1, 2), "m2": round(m2, 2), "m3": round(m3, 2), "final": round(fin, 2)}
    cls = None
    if m3 - m2 >= margin and fin >= margin and abs(m1 - m2) < margin:
        cls = "rising"
    elif m1 - m3 >= 1.5 * margin and fin <= -0.5 * margin and m1 >= m2 - 0.5:
        cls = "falling"
    elif m2 - m1 >= margin and m2 - m3 >= margin:
        cls = "rise-fall"
    elif m1 - m2 >= margin and m3 - m2 >= margin:
        cls = "fall-rise"
    return cls, stats


def unrelease(x, stop_start_s: float, keep_ms: float = 15.0, fade_ms: float = 12.0):
    """Remove the release of a word-final stop: keep the audio up to stop onset + keep_ms, fade out, then silence."""
    y = np.array(x, np.float32)
    a = int((stop_start_s + keep_ms / 1000) * SR)
    f = int(fade_ms / 1000 * SR)
    a = min(max(a, 0), len(y))
    b = min(a + f, len(y))
    y[a:b] *= np.linspace(1, 0, b - a, dtype=np.float32)
    y[b:] = 0.0
    return y


def burst_db(y, word_s: float, stop_start_s: float, end_s: float, gap: float = 0.025):
    """Highest 5-ms high-band (> 1.5 kHz) frame energy after stop onset + 25 ms, relative to the loudest
    5-ms full-band frame of the word (dB)."""
    from scipy.signal import butter, sosfilt

    sos = butter(4, 1500, "highpass", fs=SR, output="sos")
    h = sosfilt(sos, np.asarray(y, np.float64))
    F = int(0.005 * SR)

    def frames(sig, a, b):
        a, b = max(0, int(a * SR)), min(len(sig), int(b * SR))
        k = (b - a) // F
        if k <= 0:
            return np.array([1e-12])
        return np.mean(np.asarray(sig[a:a + k * F], np.float64).reshape(k, F) ** 2, axis=1) + 1e-12

    ref = float(np.max(frames(np.asarray(y, np.float64), word_s, stop_start_s)))
    tail = float(np.max(frames(h, stop_start_s + gap, end_s)))
    return 10 * math.log10(tail / ref)
