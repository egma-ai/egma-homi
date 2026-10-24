"""sb_voice label logic (numpy only, unit-tested): syllables, energy VAD and pauses, per-speaker tertiles, corpus
label maps, synthetic noise and SNR mixing, two-utterance pairs. F0 (librosa.pyin) is computed in build.py."""

from __future__ import annotations

import hashlib
import math
import re

import numpy as np

SR = 16_000

# Syllables
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def heuristic_syllables(word: str) -> int:
    """Vowel-group syllable counter for one English word (used when the word is not in CMUdict).

    Rules: lowercase letters only; words of <= 3 letters = 1 syllable; drop a silent final "e"/"es"/"ed"
    (but not "-le" as in "table", and not after t/d for "-ed" as in "wanted"); a leading "y" is a consonant;
    then count maximal runs of vowels [aeiouy]. Minimum 1.
    """
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    if w.endswith("le") and len(w) > 2 and w[-3] not in "aeiouy":
        base = 1  # "ta-ble": the final -le is its own syllable
        w = w[:-2]
    else:
        base = 0
        if w.endswith("es") and not w.endswith(("ses", "zes", "ces", "ges", "shes", "ches", "xes")):
            w = w[:-2]
        elif w.endswith("ed") and not w.endswith(("ted", "ded")):
            w = w[:-2]
        elif w.endswith("e") and not w.endswith(("ee", "ye")):
            w = w[:-1]
    w = re.sub(r"^y", "", w)
    groups = re.findall(r"[aeiouy]+", w)
    return max(1, len(groups) + base)


def count_syllables(text: str, cmu: dict | None = None) -> int:
    """Syllables in a transcript. Words found in CMUdict (`cmu`: word -> list of phone lists) count the
    phones carrying a stress digit (= vowels) of the first pronunciation; other words use
    heuristic_syllables. Digits are read one by one ("42" -> "four two", a slight undercount for
    "forty-two" but rare in these read-speech transcripts)."""
    total = 0
    for tok in re.findall(r"[A-Za-z']+|\d", text or ""):
        if tok.isdigit():
            tok = _ONES[int(tok)]
        w = tok.lower().strip("'")
        if not w:
            continue
        if cmu is not None and w in cmu:
            total += max(1, sum(1 for ph in cmu[w][0] if ph[-1].isdigit()))
        else:
            total += heuristic_syllables(w)
    return total


# Voice activity (frame energy)
VAD_FRAME, VAD_HOP = 400, 160  # 25 ms frames, 10 ms hop
VAD_REL_DB = 35.0  # frame is "active" if within 35 dB of the loudest frame ...
VAD_FLOOR_DB = -60.0  # ... and above -60 dBFS
MIN_RUN = 3  # active runs shorter than 30 ms are clicks, not speech
LONG_PAUSE_FRAMES = 30  # internal silence >= 300 ms = a long pause


def frame_db(x: np.ndarray, frame: int = VAD_FRAME, hop: int = VAD_HOP) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(x[idx] ** 2, axis=1))
    return 20.0 * np.log10(rms + 1e-10)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is True."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def activity(x: np.ndarray) -> dict | None:
    """Energy VAD. Returns span_s (first->last active frame = voice-activity-trimmed duration),
    pause_frac (share of that span in internal silences >= 300 ms), n_pauses, active_db (mean
    power of active frames, dBFS). None if the clip has no activity."""
    db = frame_db(x)
    thr = max(float(db.max()) - VAD_REL_DB, VAD_FLOOR_DB)
    act = db > thr
    for s, e in _runs(act):
        if e - s < MIN_RUN:
            act[s:e] = False
    if not act.any():
        return None
    idx = np.flatnonzero(act)
    first, last = int(idx[0]), int(idx[-1])
    span = last - first + 1
    inner = ~act[first : last + 1]
    pauses = [(s, e) for s, e in _runs(inner) if e - s >= LONG_PAUSE_FRAMES]
    p = 10 ** (db[act] / 10.0)
    return {
        "span_s": span * VAD_HOP / SR,
        "pause_frac": float(sum(e - s for s, e in pauses) / span),
        "n_pauses": len(pauses),
        "active_db": float(10 * np.log10(p.mean() + 1e-20)),
        "start_s": first * VAD_HOP / SR,
        "end_s": (last + 1) * VAD_HOP / SR + VAD_FRAME / SR,
    }


def f0_stats(f0: np.ndarray, voiced: np.ndarray) -> dict:
    """Median F0 (Hz) and robust F0 spread in semitones: (P90 - P10) / 2.563 of 12*log2(f0/median)
    over voiced frames (= the std for a normal distribution, but not thrown off by pYIN octave errors).
    NaN when fewer than 10 voiced frames."""
    f = np.asarray(f0, dtype=float)[np.asarray(voiced, bool)]
    f = f[np.isfinite(f) & (f > 0)]
    if len(f) < 10:
        return {"f0_med": float("nan"), "f0_spread_st": float("nan"), "n_voiced": int(len(f))}
    med = float(np.median(f))
    st = 12.0 * np.log2(f / med)
    p10, p90 = np.percentile(st, [10, 90])
    return {"f0_med": med, "f0_spread_st": float((p90 - p10) / 2.563), "n_voiced": int(len(f))}


def energy_gated_voicing(voiced: np.ndarray, rms: np.ndarray, rel_db: float = 25.0) -> np.ndarray:
    """Keep only voiced frames whose RMS is within rel_db of the clip's loudest frame (speech-level energy)."""
    db = 20.0 * np.log10(np.asarray(rms, dtype=float) + 1e-10)
    return np.asarray(voiced, bool) & (db > db.max() - rel_db)


# Per-speaker tertiles
MIN_CLIPS_PER_SPEAKER = 12
SOFT_BAND = 0.05  # within 5 percentile points of a tertile boundary -> soft 0.7 / 0.3 target


def percentile_ranks(values) -> np.ndarray:
    """Average-rank percentile in [0, 1] (ties share a rank). NaN values stay NaN."""
    v = np.asarray(values, dtype=float)
    out = np.full(len(v), np.nan)
    ok = np.flatnonzero(np.isfinite(v))
    if len(ok) == 0:
        return out
    if len(ok) == 1:
        out[ok] = 0.5
        return out
    vals = v[ok]
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(len(vals))
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    out[ok] = ranks / (len(vals) - 1)
    return out


def tertile_level(pr: float) -> int:
    """percentile rank -> 0 (bottom third), 1 (middle), 2 (top third)."""
    return 0 if pr < 1 / 3 else (2 if pr > 2 / 3 else 1)


def tertile_target(pr: float, band: float = SOFT_BAND) -> list[float]:
    """3-level target for a percentile rank: one-hot, except within `band` of a tertile boundary where
    the label is genuinely uncertain: 0.7 on the assigned level, 0.3 on the neighbouring one."""
    lvl = tertile_level(pr)
    t = [0.0, 0.0, 0.0]
    t[lvl] = 1.0
    for b, (lo, hi) in ((1 / 3, (0, 1)), (2 / 3, (1, 2))):
        if abs(pr - b) < band:
            other = hi if lvl == lo else lo
            t = [0.0, 0.0, 0.0]
            t[lvl], t[other] = 0.7, 0.3
    return t


def speaker_relative_levels(speakers, values, min_clips: int = MIN_CLIPS_PER_SPEAKER) -> np.ndarray:
    """Percentile rank of each clip's value within its own speaker (NaN if the speaker has fewer than
    `min_clips` valid values or the value is NaN)."""
    speakers = np.asarray(speakers)
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan)
    for spk in np.unique(speakers):
        m = np.flatnonzero(speakers == spk)
        if np.isfinite(values[m]).sum() < min_clips:
            continue
        out[m] = percentile_ranks(values[m])
    return out


def pause_level(pause_frac: float, n_pauses: int, spk_q67: float, min_many: float = 0.08) -> int:
    """0 = no long pause, 2 = many pauses (pause share >= the speaker's 67th percentile and >= 8 % of
    the speaking time), 1 = in between."""
    if n_pauses == 0:
        return 0
    if pause_frac >= max(spk_q67, min_many):
        return 2
    return 1


# Corpus label maps
ACCENT_GROUPS = {  # VCTK speaker-info accent -> group shown as an option (None = not used)
    "American": "American", "Canadian": "Canadian", "English": "English (England)",
    "Scottish": "Scottish", "Irish": "Irish", "NorthernIrish": "Irish", "Indian": "Indian",
    "Australian": "Australian or New Zealand", "NewZealand": "Australian or New Zealand",
    "SouthAfrican": "South African", "Welsh": None, "Unknown": None, "": None,
}


def accent_group(accent: str | None) -> str | None:
    if accent is None:
        return None
    return ACCENT_GROUPS.get(accent.strip().replace(" ", ""), None)


def so762_level(score: int) -> int:
    """speechocean762 sentence score 0..10 -> 0 (<= 5), 1 (6-7), 2 (>= 8)."""
    return 0 if score <= 5 else (1 if score <= 7 else 2)


def crema_age_bracket(age: int) -> int:
    """0 = under 30, 1 = 30-49, 2 = 50 or older."""
    return 0 if age < 30 else (1 if age < 50 else 2)


def child_label(age: int) -> bool | None:
    """speechocean762: child if <= 12, adult if >= 18, None (unused) for 13-17."""
    if age <= 12:
        return True
    if age >= 18:
        return False
    return None


def stable_frac(key: str, salt: str = "") -> float:
    """Deterministic pseudo-random number in [0, 1) from a string (for subsampling without state)."""
    h = hashlib.md5(f"{salt}:{key}".encode()).hexdigest()
    return int(h[:12], 16) / 16 ** 12


# Noise + mixing
NOISE_TYPES = ["white", "pink", "brown", "hum", "rumble"]


def colored_noise(n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Unit-RMS synthetic noise (CC0 by construction). pink = 1/f power, brown = 1/f^2,
    hum = 50/60 Hz mains harmonics + a little white noise, rumble = low-passed brown noise with a
    slow amplitude wobble (fan / traffic-like)."""
    if kind == "hum":
        t = np.arange(n) / SR
        f = rng.choice([50.0, 60.0])
        x = sum(rng.uniform(0.3, 1.0) / k * np.sin(2 * np.pi * f * k * t + rng.uniform(0, 2 * np.pi))
                for k in range(1, 8))
        x = x + 0.05 * rng.standard_normal(n)
    else:
        w = rng.standard_normal(n)
        if kind == "white":
            x = w
        else:
            spec = np.fft.rfft(w)
            freqs = np.fft.rfftfreq(n, 1 / SR)
            freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
            power = {"pink": 1.0, "brown": 2.0, "rumble": 2.0}[kind]
            spec = spec / freqs ** (power / 2)
            if kind == "rumble":
                spec = spec * (freqs < 400)
            x = np.fft.irfft(spec, n)
            if kind == "rumble":
                t = np.arange(n) / SR
                x = x * (1 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.2, 1.0) * t))
    x = np.asarray(x, dtype=np.float64)
    return (x / (np.sqrt(np.mean(x ** 2)) + 1e-12)).astype(np.float32)


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float, speech_power: float | None = None) -> np.ndarray:
    """speech + noise scaled so that (active speech power) / (noise power) = snr_db. Peak-limited to
    0.99 (scaled down only if needed)."""
    s = np.asarray(speech, dtype=np.float64)
    nz = np.asarray(noise, dtype=np.float64)[: len(s)]
    ps = speech_power if speech_power is not None else float(np.mean(s ** 2))
    pn = float(np.mean(nz ** 2)) + 1e-20
    y = s + nz * math.sqrt(ps / (pn * 10 ** (snr_db / 10.0)))
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0.99:
        y = y * (0.99 / peak)
    return y.astype(np.float32)


def peak_limit(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    return (x * (0.99 / peak)).astype(np.float32) if peak > 0.99 else np.asarray(x, dtype=np.float32)


def concat_pair(a: np.ndarray, b: np.ndarray, gap_s: float, level_a_db: float, level_b_db: float,
                target_db: float, jitter_a: float = 0.0, jitter_b: float = 0.0) -> np.ndarray:
    """Two utterances with a silent gap; each part is first set to the same active level target_db
    (+ a small random jitter, applied identically in same/different-speaker pairs) so loudness is not
    a shortcut cue. 10 ms fades at the joins."""
    def norm(x, lvl, jit):
        g = 10 ** ((target_db + jit - lvl) / 20.0)
        y = np.asarray(x, dtype=np.float64) * g
        f = min(160, len(y) // 4)
        if f > 0:
            ramp = np.linspace(0, 1, f)
            y[:f] *= ramp
            y[-f:] *= ramp[::-1]
        return y
    gap = np.zeros(int(round(gap_s * SR)))
    y = np.concatenate([norm(a, level_a_db, jitter_a), gap, norm(b, level_b_db, jitter_b)])
    return peak_limit(y)
