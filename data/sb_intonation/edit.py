"""WORLD (pyworld, MIT / WORLD modified-BSD) pitch-contour reshaping + independent measurement. 16 kHz mono float32.
The same analysis -> F0 replacement -> synthesis path is used for real (LibriTTS-R) and Kokoro clips, so WORLD
resynthesis is never a cue. Verification uses a different F0 tracker (Harvest) than the edit (DIO + StoneMask)."""
from __future__ import annotations

import numpy as np

import labels as L

SR = 16000
FP = 5.0  # ms


def _pw():
    import pyworld as pw

    return pw


def analyze(x, floor=55.0, ceil=600.0):
    pw = _pw()
    x64 = np.asarray(x, np.float64)
    f0, t = pw.dio(x64, SR, f0_floor=floor, f0_ceil=ceil, frame_period=FP)
    f0 = pw.stonemask(x64, f0, t, SR)
    sp = pw.cheaptrick(x64, f0, t, SR)
    ap = pw.d4c(x64, f0, t, SR)
    return f0, sp, ap


def synth(f0, sp, ap, n_out):
    y = _pw().synthesize(np.ascontiguousarray(f0), np.ascontiguousarray(sp), np.ascontiguousarray(ap), SR, FP)
    y = np.asarray(y, np.float32)
    return y[:n_out] if len(y) >= n_out else np.pad(y, (0, n_out - len(y)))


def harvest(y, floor, ceil):
    pw = _pw()
    y64 = np.asarray(y, np.float64)
    f0, t = pw.harvest(y64, SR, f0_floor=floor, f0_ceil=ceil, frame_period=FP)
    return f0


def _trend(f0):
    v = f0 > 0
    idx = np.flatnonzero(v)
    if len(idx) < 40:
        raise ValueError("too few voiced frames")
    lf = 12 * np.log2(f0[v])
    fit = np.polyfit(idx, lf, 3)
    trend = np.polyval(fit, idx)
    resid = np.clip(lf - trend, -3.0, 3.0)
    u = (idx - idx[0]) / max(idx[-1] - idx[0], 1)
    return v, idx, lf, trend, resid, u


def reshape(x, info: dict):
    """info: kind ('contour' | 'final'), pattern, size (st), resid (local-detail scale), c, p, u0.
    Returns (y, f0_median_hz)."""
    f0, sp, ap = analyze(x)
    v, idx, lf, trend, resid, u = _trend(f0)
    med = float(np.median(lf))
    if info["kind"] == "contour":
        new = med + info["resid"] * resid + info["size"] * L.contour_shape(info["pattern"], u, info["c"], info["p"])
    else:
        # keep the natural (cubic) melody up to u0, freeze it there, then glide up / down
        u0 = info["u0"]
        k0 = int(np.searchsorted(u, u0))
        tr = trend.copy()
        tr[k0:] = tr[min(k0, len(tr) - 1)]
        d = 1 if info["pattern"] == "final_rise" else -1
        new = tr + info["resid"] * resid + L.final_glide(u, u0, info["size"], d)
    f0n = np.zeros_like(f0)
    f0n[v] = 2 ** (new / 12.0)
    return synth(f0n, sp, ap, len(x)), 2 ** (med / 12.0)


def measure(y, f0_med):
    """Harvest F0 on the output (search range around the speaker's median), then the label classifier."""
    floor = max(45.0, f0_med / 2.6)
    ceil = min(800.0, f0_med * 2.6)
    f0 = harvest(y, floor, ceil)
    hop = int(SR * FP / 1000)
    yy = np.pad(np.asarray(y, np.float64), (200, 200))
    n = len(f0)
    e = np.array([10 * np.log10(np.mean(yy[i * hop:i * hop + 400] ** 2) + 1e-12) for i in range(n)])
    return L.classify_track(L.clean_track(f0, e))
