"""Call tones and line noise (ringback, busy, SIT, fax/modem, beep) and the sb_calls clip assembler.

`assemble` turns a plan entry plus its TTS segments into one 16 kHz clip and the facts the labels use."""
from __future__ import annotations

import numpy as np

SR = 16_000
MAX_S = 15.0
MIN_S = 1.0


def sil(s: float) -> np.ndarray:
    return np.zeros(max(0, int(round(s * SR))), np.float32)


def tone(freqs, dur: float, amp: float = 0.5, ramp_ms: float = 5.0, phase: float = 0.0) -> np.ndarray:
    n = max(0, int(round(dur * SR)))
    t = np.arange(n) / SR
    x = sum(np.sin(2 * np.pi * f * t + phase) for f in freqs) / max(1, len(freqs))
    x = (amp * x).astype(np.float32)
    r = min(n // 2, int(ramp_ms * SR / 1000))
    if r > 0:
        w = np.linspace(0, 1, r, dtype=np.float32)
        x[:r] *= w
        x[-r:] *= w[::-1]
    return x


def cadence(freqs, pattern, total: float, amp: float = 0.5) -> np.ndarray:
    """Repeat an on/off pattern [(on_s, off_s), ...] starting with 'on' until `total` seconds."""
    out, t = [], 0.0
    while t < total:
        for on, off in pattern:
            out.append(tone(freqs, on, amp))
            out.append(sil(off))
            t += on + off
            if t >= total:
                break
    x = np.concatenate(out) if out else np.zeros(0, np.float32)
    return x[: int(round(total * SR))]


RINGBACK = {  # name: (freqs, pattern, weight)
    "us": ([440, 480], [(2.0, 4.0)], 0.6),
    "uk": ([400, 450], [(0.4, 0.2), (0.4, 2.0)], 0.2),
    "eu": ([425], [(1.0, 4.0)], 0.2),
}
BUSY = {
    "us": ([480, 620], [(0.5, 0.5)], 0.5),
    "uk": ([400], [(0.375, 0.375)], 0.15),
    "eu": ([425], [(0.5, 0.5)], 0.2),
    "reorder": ([480, 620], [(0.25, 0.25)], 0.15),
}


def _choose(rng: np.random.Generator, table: dict) -> str:
    names = list(table)
    w = np.array([table[k][2] for k in names], float)
    return names[int(rng.choice(len(names), p=w / w.sum()))]


def _jit(rng, freqs, pct=0.01):
    return [f * (1 + rng.uniform(-pct, pct)) for f in freqs]


def ringback(rng, total: float) -> tuple[np.ndarray, str]:
    v = _choose(rng, RINGBACK)
    f, pat, _ = RINGBACK[v]
    return cadence(_jit(rng, f), pat, total), v


def busy(rng, total: float) -> tuple[np.ndarray, str]:
    v = _choose(rng, BUSY)
    f, pat, _ = BUSY[v]
    return cadence(_jit(rng, f), pat, total), v


def sit(rng) -> np.ndarray:
    """Special Information Tone: three rising tones (~914/985, ~1371/1429, 1777 Hz; 274 or 380 ms each)."""
    f1 = rng.choice([913.8, 985.2])
    f2 = rng.choice([1370.6, 1428.5])
    parts = []
    for f in (f1, f2, 1776.7):
        parts.append(tone([f], float(rng.choice([0.274, 0.380])), 0.5, ramp_ms=3))
        parts.append(sil(0.004))
    return np.concatenate(parts)


def fsk(rng, dur: float, f_mark=1650.0, f_space=1850.0, baud=300.0, flags=True) -> np.ndarray:
    """Continuous-phase binary FSK (V.21 channel 2, as used by T.30 fax). HDLC flags 0x7E if `flags`."""
    nbits = int(dur * baud) + 1
    if flags:
        bits = np.tile(np.array([0, 1, 1, 1, 1, 1, 1, 0]), nbits // 8 + 1)[:nbits]
    else:
        bits = rng.integers(0, 2, nbits)
    spb = SR / baud
    n = int(round(dur * SR))
    idx = np.minimum((np.arange(n) / spb).astype(int), nbits - 1)
    f = np.where(bits[idx] == 1, f_mark, f_space)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return (0.5 * np.sin(ph)).astype(np.float32)


def fax_modem(rng, total: float) -> tuple[np.ndarray, str]:
    if rng.random() < 0.6:  # fax: CED 2100 Hz answer tone, then V.21 HDLC flag preamble, repeated
        out, t = [], 0.0
        while t < total:
            ced = float(rng.uniform(2.6, 4.0))
            pre = float(rng.uniform(1.0, 2.5))
            out += [tone([2100 * (1 + rng.uniform(-0.005, 0.005))], ced, 0.5), sil(0.075), fsk(rng, pre), sil(0.3)]
            t += ced + 0.075 + pre + 0.3
        return np.concatenate(out)[: int(total * SR)], "fax"
    # modem: ANSam (2100 Hz, 15 Hz AM, phase reversals every 450 ms), then scrambled-data hiss
    ans = float(rng.uniform(2.5, 3.3))
    n = int(ans * SR)
    t = np.arange(n) / SR
    rev = np.pi * (np.floor(t / 0.45) % 2)
    x = (1 + 0.2 * np.sin(2 * np.pi * 15 * t)) * np.sin(2 * np.pi * 2100 * t + rev) * 0.4
    rest = max(0.0, total - ans - 0.1)
    hiss = _bandnoise(rng, int(rest * SR), 600, 3000) * 0.35
    return np.concatenate([x.astype(np.float32), sil(0.1), hiss])[: int(total * SR)], "modem"


def _bandnoise(rng, n: int, lo: float, hi: float) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, np.float32)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spec[(f < lo) | (f > hi)] = 0
    x = np.fft.irfft(spec, n)
    return (x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def beep(rng, dur: float) -> np.ndarray:
    return tone([1000.0 * (1 + rng.uniform(-0.01, 0.01))], dur, 0.5, ramp_ms=4)


def line_noise(rng, kind: str, n: int, dbfs: float) -> np.ndarray:
    if kind == "none" or n <= 0:
        return np.zeros(n, np.float32)
    if kind == "white":
        x = rng.standard_normal(n)
    elif kind == "pink":
        spec = np.fft.rfft(rng.standard_normal(n))
        f = np.fft.rfftfreq(n, 1 / SR)
        spec = spec / np.sqrt(np.maximum(f, 20.0))
        x = np.fft.irfft(spec, n)
    else:  # hum (50/60 Hz + harmonics) over a little white noise
        f0 = float(rng.choice([50.0, 60.0]))
        t = np.arange(n) / SR
        x = sum((0.6 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28)) for k in (1, 2, 3, 5))
        x = x + 0.15 * rng.standard_normal(n)
    x = x / (np.sqrt(np.mean(x ** 2)) + 1e-12)
    return (x * 10 ** (dbfs / 20)).astype(np.float32)


def trim_silence(x: np.ndarray, rel_db: float = 40.0, pad_s: float = 0.03) -> np.ndarray:
    """Trim leading/trailing near-silence (10 ms frames below peak-frame RMS - rel_db), keeping `pad_s`."""
    x = np.asarray(x, np.float32)
    hop = SR // 100
    n = len(x) // hop
    if n < 3:
        return x
    r = 20 * np.log10(np.sqrt((x[: n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1)) + 1e-9)
    on = np.flatnonzero(r > r.max() - rel_db)
    pad = int(pad_s * SR)
    a = max(0, on[0] * hop - pad)
    b = min(len(x), (on[-1] + 1) * hop + pad)
    return x[a:b]


def _peak(x):
    return float(np.abs(x).max()) if len(x) else 0.0


def _norm(x, peak=0.5):
    p = _peak(x)
    return x * (peak / p) if p > 0 else x


def assemble(plan: dict, segs: list[np.ndarray] | None, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """One clip from a plan and its 16 kHz TTS segments (None for tone-only classes) -> (audio, facts)."""
    cls, aug = plan["cls"], plan["aug"]
    parts: list[np.ndarray] = []
    t = 0.0
    facts = {"ring_prefix": False, "beep_present": False, "truncated": False, "variant": None}

    def put(x):
        nonlocal t
        parts.append(np.asarray(x, np.float32))
        t += len(x) / SR

    if aug["ring_prefix"]:
        rb, v = ringback(rng, float(rng.uniform(1.0, 4.5)))
        put(_norm(rb, 0.5))
        put(sil(float(rng.uniform(0.2, 0.6))))
        facts["ring_prefix"] = True
    put(sil(aug["lead_s"]))
    onset = t
    cue_end = onset + 1.2
    beep_span = None
    kept_texts: list[str] = []

    if cls in ("no_answer", "busy", "fax_modem"):
        if cls == "no_answer":
            ring_dur = float(rng.uniform(6.5, 11.0))
            ring_dur = min(ring_dur, MAX_S - t - 1.5)
            x, v = ringback(rng, ring_dur)
            put(_norm(x, 0.5))
            put(sil(float(rng.uniform(1.5, 3.5))))
        else:
            dur = float(rng.uniform(3.0, 10.0)) if cls == "busy" else float(rng.uniform(3.0, 12.0))
            dur = min(dur, MAX_S - t - aug["tail_s"])
            x, v = busy(rng, dur) if cls == "busy" else fax_modem(rng, dur)
            put(_norm(x, 0.5))
            put(sil(aug["tail_s"]))
        facts["variant"] = v
    else:
        assert segs is not None and len(segs) == len(plan["segments"])
        speech_level = 0.5
        if cls == "not_in_service":
            put(_norm(sit(rng), 0.5 * float(rng.uniform(0.5, 1.0))))
            cue_end = t + 0.3
            put(sil(float(rng.uniform(0.1, 0.5))))
        reserve = aug["tail_s"] + ((aug["beep_gap_s"] + aug["beep_s"]) if aug["beep"] else 0.0)
        for i, (seg, a) in enumerate(zip(plan["segments"], segs)):
            a = _norm(trim_silence(a), speech_level)
            if seg["kind"] == "followup":
                gap = float(rng.uniform(1.0, 3.5))
            elif i > 0:
                gap = float(rng.uniform(0.15, 0.6))
            else:
                gap = 0.0
            needed = gap + len(a) / SR
            if i > max(plan["cue_idx"], 0) and t + needed + reserve > MAX_S:
                break
            if gap:
                put(sil(gap))
            put(a)
            kept_texts.append(seg["text"])
            if i == plan["cue_idx"]:
                cue_end = max(cue_end, t)
        facts["segments_kept"] = len(kept_texts)
        facts["all_segments"] = len(kept_texts) == len(plan["segments"])
        if aug["beep"] and t + reserve <= MAX_S:
            put(sil(aug["beep_gap_s"]))
            b0 = t
            put(_norm(beep(rng, aug["beep_s"]), speech_level * float(rng.uniform(0.4, 1.0))))
            beep_span = (b0, t)
        put(sil(aug["tail_s"]))

    x = np.concatenate(parts)
    natural = len(x) / SR
    cut = None
    if aug["truncate"]:
        c = max(aug["cut_s"], cue_end + 0.15)
        if beep_span is not None and beep_span[0] - 0.05 < c < beep_span[1]:
            c = beep_span[0] - 0.05 if beep_span[0] - 0.05 >= cue_end + 0.15 else None
        if c is not None and c < natural - 0.05:
            cut = c
    if natural > MAX_S and (cut is None or cut > MAX_S):
        cut = MAX_S
    if cut is not None:
        x = x[: int(round(cut * SR))]
        facts["truncated"] = bool(aug["truncate"]) and cut < natural - 0.05
    if len(x) < MIN_S * SR:
        x = np.concatenate([x, sil(MIN_S - len(x) / SR + 0.01)])
    dur = len(x) / SR
    facts["beep_present"] = beep_span is not None and beep_span[1] <= dur + 1e-6
    if cls not in ("no_answer", "busy", "fax_modem"):
        facts["mentions_tone"] = any(("tone" in s.lower() or "beep" in s.lower()) for s in kept_texts)
    # level + line noise (same distributions for every class)
    x = _norm(x, aug["peak"])
    x = x + line_noise(rng, aug["noise"], len(x), aug["noise_dbfs"])
    p = _peak(x)
    if p > 0.99:
        x = x * (0.99 / p)
    facts.update({"duration_s": round(dur, 3), "onset_s": round(onset, 3), "cue_end_s": round(cue_end, 3),
                  "beep_span": [round(beep_span[0], 3), round(beep_span[1], 3)] if beep_span else None,
                  "natural_s": round(natural, 3)})
    return x.astype(np.float32), facts
