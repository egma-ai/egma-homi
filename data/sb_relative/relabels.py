"""Pure label logic for sb_relative: two-part recordings (normal voice, tone, target) and their relative labels.

Feature definitions are copied from data/sb_voice/labels.py so both builders measure the same thing.
"""

from __future__ import annotations

import hashlib
import math
import random
import re

import numpy as np

SR = 16_000

# Copied from data/sb_voice/labels.py (syllables, energy VAD, F0 statistics)
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def heuristic_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    if w.endswith("le") and len(w) > 2 and w[-3] not in "aeiouy":
        base = 1
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


VAD_FRAME, VAD_HOP = 400, 160  # 25 ms frames, 10 ms hop
VAD_REL_DB = 35.0
VAD_FLOOR_DB = -60.0
MIN_RUN = 3
LONG_PAUSE_FRAMES = 30


def frame_db(x: np.ndarray, frame: int = VAD_FRAME, hop: int = VAD_HOP) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(x[idx] ** 2, axis=1))
    return 20.0 * np.log10(rms + 1e-10)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def activity(x: np.ndarray) -> dict | None:
    """Energy VAD (sb_voice): span_s = first..last active frame, active_db = mean power of active frames."""
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
    inner = ~act[first: last + 1]
    pauses = [(s, e) for s, e in _runs(inner) if e - s >= LONG_PAUSE_FRAMES]
    p = 10 ** (db[act] / 10.0)
    return {"span_s": span * VAD_HOP / SR, "pause_frac": float(sum(e - s for s, e in pauses) / span),
            "n_pauses": len(pauses), "active_db": float(10 * np.log10(p.mean() + 1e-20))}


def f0_stats(f0: np.ndarray, voiced: np.ndarray) -> dict:
    """Median F0 (Hz) and robust F0 spread in semitones ((P90 - P10) / 2.563) over voiced frames."""
    f = np.asarray(f0, dtype=float)[np.asarray(voiced, bool)]
    f = f[np.isfinite(f) & (f > 0)]
    if len(f) < 10:
        return {"f0_med": float("nan"), "f0_spread_st": float("nan"), "n_voiced": int(len(f))}
    med = float(np.median(f))
    st = 12.0 * np.log2(f / med)
    p10, p90 = np.percentile(st, [10, 90])
    return {"f0_med": med, "f0_spread_st": float((p90 - p10) / 2.563), "n_voiced": int(len(f))}


def energy_gated_voicing(voiced: np.ndarray, rms: np.ndarray, rel_db: float = 25.0) -> np.ndarray:
    db = 20.0 * np.log10(np.asarray(rms, dtype=float) + 1e-10)
    return np.asarray(voiced, bool) & (db > db.max() - rel_db)


def stable_frac(key: str, salt: str = "") -> float:
    h = hashlib.md5(f"{salt}:{key}".encode()).hexdigest()
    return int(h[:12], 16) / 16 ** 12


def percentile_ranks(values) -> np.ndarray:
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
        ranks[order[i: j + 1]] = (i + j) / 2.0
        i = j + 1
    out[ok] = ranks / (len(vals) - 1)
    return out


# CREMA-D sentences: text + hand syllable count (copied from our evaluation-set builder)
CREMA_SENTENCES = {
    "IEO": ("It's eleven o'clock", 6), "TIE": ("That is exactly what happened", 8),
    "IOM": ("I'm on my way to the meeting", 8), "IWW": ("I wonder what this is about", 8),
    "TAI": ("The airplane is almost full", 7), "MTI": ("Maybe tomorrow it will be cold", 9),
    "IWL": ("I would like a new alarm clock", 8), "ITH": ("I think I have a doctor's appointment", 10),
    "DFA": ("Don't forget a jacket", 6), "ITS": ("I think I've seen this before", 7),
    "TSI": ("The surface is slick", 5), "WSI": ("We'll stop in a couple of minutes", 9),
}


def crema_syllables(key: str) -> int:
    """'1001_DFA_ANG_XX' -> 6 (hand count of the sentence)."""
    return CREMA_SENTENCES[key.split("_")[1]][1]


# Relative labels
FEATURES = ["loudness", "rate", "pitch", "pitch_variation"]
FEATURE_KEY = {"loudness": "loud_db", "rate": "rate", "pitch": "f0_med", "pitch_variation": "f0_spread_st"}
# (clear-difference threshold, "about the same" band, unit). Values between the two are ambiguous -> dropped.
#   loudness: dB difference of RMS power on voiced speech frames
#   rate: log ratio of syllables per second (12 % / 4 %)
#   pitch: semitone difference of median F0 (1.5 st / 0.5 st)
#   pitch_variation: log ratio of the robust F0 spread in semitones (40 % / 10 %)
THRESHOLDS = {
    "loudness": (3.0, 1.0, "dB"),
    "rate": (math.log(1.12), math.log(1.04), "log ratio"),
    "pitch": (1.5, 0.5, "semitones"),
    "pitch_variation": (math.log(1.4), math.log(1.10), "log ratio"),
}
PITCH_MAX_ABS_ST = 9.0  # larger median-F0 jumps are almost always pYIN octave errors -> not labelled
MIN_VOICED = {"pitch": 15, "pitch_variation": 30}  # voiced frames needed in both parts
CLASS_NAMES = ["lower", "same", "higher"]  # 0 / 1 / 2 (low -> high)


def _finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def rel_diff(feature: str, ref: dict, tgt: dict) -> float | None:
    """Second part minus first part, in the feature's unit (see THRESHOLDS). None if not measurable."""
    k = FEATURE_KEY[feature]
    a, b = ref.get(k), tgt.get(k)
    if not (_finite(a) and _finite(b)):
        return None
    if feature in MIN_VOICED and min(ref.get("n_voiced") or 0, tgt.get("n_voiced") or 0) < MIN_VOICED[feature]:
        return None
    if feature == "loudness":
        return float(b - a)
    if a <= 0 or b <= 0:
        return None
    if feature == "pitch":
        d = 12.0 * math.log2(b / a)
        return None if abs(d) > PITCH_MAX_ABS_ST else d
    return math.log(b / a)  # rate, pitch_variation


def classify(feature: str, d: float | None) -> int | None:
    """0 = lower/quieter/slower/flatter, 1 = about the same, 2 = higher/louder/faster/more varied, None = ambiguous."""
    if d is None or not math.isfinite(d):
        return None
    hi, same, _ = THRESHOLDS[feature]
    if d >= hi:
        return 2
    if d <= -hi:
        return 0
    if abs(d) <= same:
        return 1
    return None


def noul_answer(direction: str, cls: int) -> bool:
    """direction 'up' (louder/faster/higher/more varied?), 'down', 'same' -> yes/no for a class."""
    return {"up": cls == 2, "down": cls == 0, "same": cls == 1}[direction]


# Augmentation amounts (known, controlled changes of the target clip)
# feature -> (kind, max |amount| in the feature's unit)
AUG = {"loudness": ("gain_db", 12.0), "rate": ("tempo_log", math.log(1.45)), "pitch": ("shift_st", 4.0)}
FINAL_RANGE = {  # wanted final difference after augmentation, per class (uniform in the range, sign by class)
    "loudness": {2: (3.6, 10.0), 0: (-10.0, -3.6), 1: (-0.7, 0.7)},
    "rate": {2: (math.log(1.15), math.log(1.45)), 0: (math.log(1 / 1.45), math.log(1 / 1.15)), 1: (-0.025, 0.025)},
    "pitch": {2: (1.9, 4.5), 0: (-4.5, -1.9), 1: (-0.35, 0.35)},
}


def plan_amount(feature: str, d0: float, cls: int, rng: random.Random) -> float | None:
    """Amount that moves the natural difference d0 into the class range; None if it exceeds the allowed maximum."""
    lo, hi = FINAL_RANGE[feature][cls]
    want = rng.uniform(lo, hi)
    amt = want - d0
    if abs(amt) > AUG[feature][1]:
        return None
    return amt


# Two-part recording
GAP_S = 0.6
TONE_S = 0.15
TONE_HZ = 1000.0
TONE_AMP = 0.1  # -20 dBFS peak
MAX_TOTAL_S = 20.0


def tone(sr: int = SR) -> np.ndarray:
    n = int(round(TONE_S * sr))
    t = np.arange(n) / sr
    y = TONE_AMP * np.sin(2 * np.pi * TONE_HZ * t)
    f = int(0.01 * sr)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, f))
    y[:f] *= ramp
    y[-f:] *= ramp[::-1]
    return y


def fade(x: np.ndarray, n: int = 160) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64).copy()
    f = min(n, len(y) // 4)
    if f > 0:
        r = np.linspace(0, 1, f)
        y[:f] *= r
        y[-f:] *= r[::-1]
    return y


def two_part(ref: np.ndarray, tgt: np.ndarray, sr: int = SR) -> tuple[np.ndarray, dict]:
    """[ref] + [0.6 s silence, 0.15 s 1 kHz tone, 0.6 s silence] + [tgt]. The whole recording is scaled down
    together only if its peak exceeds 0.99, so the level difference between the two parts is preserved."""
    gap = np.zeros(int(round(GAP_S * sr)))
    y = np.concatenate([fade(ref), gap, tone(sr), gap, fade(tgt)])
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    scale = 0.99 / peak if peak > 0.99 else 1.0
    y = (y * scale).astype(np.float32)
    ref_end = len(ref) / sr
    info = {"ref_s": round(len(ref) / sr, 3), "tgt_start_s": round(ref_end + 2 * GAP_S + TONE_S, 3),
            "tgt_s": round(len(tgt) / sr, 3), "scale": round(scale, 5)}
    return y, info


def total_seconds(ref_s: float, tgt_s: float) -> float:
    return ref_s + 2 * GAP_S + TONE_S + tgt_s


# Training question wording (the EVAL_* evaluation questions below are not in these pools)
INTROS = [
    "This recording has two parts with a short beep between them: first the speaker's usual voice, then a second clip of the same person.",
    "You will hear the same speaker twice, separated by a tone. The first part is how they normally sound.",
    "The audio has two sections divided by a beep. Part one is a reference of this person's normal speaking voice; part two comes from later in the conversation.",
    "Before the tone you hear the speaker's normal voice; after the tone, the same speaker again.",
    "Two clips of one speaker are joined with a short beep. Treat the first clip as their baseline voice.",
    "First part: the caller talking normally. Then a beep. Second part: the same caller later on.",
    "The first half of this recording (before the beep) shows how the speaker usually talks; the second half is the clip to judge.",
]
INTRO_P = 0.8  # share of training questions that describe the two-part structure explicitly

SCORE_Q = {
    "loudness": ["Compared with the first part, how loud is the second part?",
                 "Is the speaker louder, about the same, or quieter in the second part than in the first?",
                 "How does the speaker's volume after the beep compare with their volume before it?",
                 "Relative to their normal voice in part one, how loudly is the person talking in part two?",
                 "How has the speaker's loudness changed in the second clip, if at all?",
                 "Rate the loudness of the second part relative to the first part."],
    "rate": ["Compared with the first part, how fast is the speaker talking in the second part?",
             "Is the speaker faster, about the same speed, or slower in the second part than in the first?",
             "How does the speaking pace after the beep compare with the pace before it?",
             "Relative to their normal pace in part one, how quickly is the person speaking in part two?",
             "How has the speaker's speaking rate changed in the second clip, if at all?",
             "Rate the speed of speech in the second part relative to the first part."],
    "pitch": ["Compared with the first part, how high is the speaker's voice in the second part?",
              "Is the speaker's pitch higher, about the same, or lower in the second part than in the first?",
              "How does the pitch of the voice after the beep compare with the pitch before it?",
              "Relative to their normal pitch in part one, where is the person's voice in part two?",
              "How has the speaker's pitch changed in the second clip, if at all?",
              "Rate the pitch of the second part relative to the first part."],
    "pitch_variation": ["Compared with the first part, how much does the speaker's pitch vary in the second part?",
                        "Is the speaker's intonation more varied, about the same, or flatter in the second part than in the first?",
                        "How does the melody of the voice after the beep compare with the melody before it?",
                        "Relative to their normal intonation in part one, how expressive is the person's voice in part two?",
                        "Has the speaker's intonation become flatter, stayed about the same, or become livelier in the second clip?",
                        "Rate the pitch variation of the second part relative to the first part."],
}
OPTION_SETS = {  # low -> high
    "loudness": [["quieter", "about the same", "louder"],
                 ["softer than in the first part", "about as loud as in the first part", "louder than in the first part"],
                 ["quieter than their normal voice", "same volume as normal", "louder than their normal voice"]],
    "rate": [["slower", "about the same", "faster"],
             ["slower than in the first part", "about the same pace as in the first part", "faster than in the first part"],
             ["slower than their normal pace", "their normal pace", "faster than their normal pace"]],
    "pitch": [["lower", "about the same", "higher"],
              ["lower than in the first part", "about the same pitch as in the first part", "higher than in the first part"],
              ["lower than their normal pitch", "their normal pitch", "higher than their normal pitch"]],
    "pitch_variation": [["flatter", "about the same", "more varied"],
                        ["flatter than in the first part", "about as varied as in the first part", "more varied than in the first part"],
                        ["more monotone than normal", "their normal intonation", "more expressive than normal"]],
}
NOUL_Q = {
    "loudness": {
        "up": ["Is the speaker louder in the second part than in the first?", "After the beep, is the person speaking more loudly than before?",
               "Is the speaker raising their voice in the second part compared with their normal voice?",
               "Is the second clip louder than the first one?", "Does the speaker get louder after the tone?"],
        "down": ["Is the speaker quieter in the second part than in the first?", "After the beep, is the person speaking more softly than before?",
                 "Is the second clip quieter than the first one?", "Does the speaker lower their voice in the second part?",
                 "Has the speaker become quieter after the tone?"],
        "same": ["Is the speaker's volume about the same in both parts?", "Does the speaker talk at roughly the same loudness before and after the beep?",
                 "Is the second clip about as loud as the first one?", "Is there no clear change in loudness between the two parts?",
                 "Is the speaker's loudness unchanged in the second part?"]},
    "rate": {
        "up": ["Is the speaker talking faster in the second part?", "After the beep, is the person speaking more quickly than before?",
               "Is the second clip spoken at a faster pace than the first?", "Does the speaker speed up after the tone?",
               "Compared with their normal pace in part one, is the speaker rushing in part two?"],
        "down": ["Is the speaker talking slower in the second part?", "After the beep, is the person speaking more slowly than before?",
                 "Is the second clip spoken at a slower pace than the first?", "Does the speaker slow down after the tone?",
                 "Compared with their normal pace in part one, is the speaker taking more time in part two?"],
        "same": ["Is the speaker talking at about the same speed in both parts?", "Is the pace of speech roughly unchanged after the beep?",
                 "Does the second clip have about the same speaking rate as the first?", "Is there no clear change in speaking speed between the two parts?",
                 "Does the speaker keep their usual pace in the second part?"]},
    "pitch": {
        "up": ["Is the speaker's voice higher in the second part?", "After the beep, is the person speaking at a higher pitch than before?",
               "Is the second clip higher-pitched than the first?", "Does the speaker's pitch go up after the tone?",
               "Compared with their normal voice in part one, is the speaker's pitch raised in part two?"],
        "down": ["Is the speaker's voice lower in the second part?", "After the beep, is the person speaking at a lower pitch than before?",
                 "Is the second clip lower-pitched than the first?", "Does the speaker's pitch drop after the tone?",
                 "Compared with their normal voice in part one, is the speaker's pitch deeper in part two?"],
        "same": ["Is the speaker's pitch about the same in both parts?", "Is the pitch of the voice roughly unchanged after the beep?",
                 "Does the second clip have about the same pitch as the first?", "Is there no clear change in pitch between the two parts?",
                 "Does the speaker keep their usual pitch in the second part?"]},
    "pitch_variation": {
        "up": ["Is the speaker's intonation more varied in the second part?", "After the beep, does the person sound more animated than before?",
               "Does the pitch move around more in the second clip than in the first?", "Does the speaker become more expressive after the tone?",
               "Compared with their normal intonation in part one, is the speaker livelier in part two?"],
        "down": ["Is the speaker's intonation flatter in the second part?", "After the beep, does the person sound more monotone than before?",
                 "Does the pitch move around less in the second clip than in the first?", "Does the speaker become flatter after the tone?",
                 "Compared with their normal intonation in part one, is the speaker more monotone in part two?"],
        "same": ["Is the speaker's intonation about equally varied in both parts?", "Is the melody of the voice roughly unchanged after the beep?",
                 "Does the second clip have about as much pitch variation as the first?", "Is there no clear change in expressiveness between the two parts?",
                 "Does the speaker keep their usual intonation in the second part?"]},
}


def score_question(feature: str, rng: random.Random) -> tuple[str, list[str], int]:
    """(question, options low -> high, option-set index)."""
    q = rng.choice(SCORE_Q[feature])
    if rng.random() < INTRO_P:
        q = f"{rng.choice(INTROS)} {q}"
    k = rng.randrange(len(OPTION_SETS[feature]))
    return q, list(OPTION_SETS[feature][k]), k


def noul_question(feature: str, direction: str, rng: random.Random) -> str:
    q = rng.choice(NOUL_Q[feature][direction])
    if rng.random() < INTRO_P:
        q = f"{rng.choice(INTROS)} {q}"
    return q


# Extra evaluation set (crema_relative_<feature>): one fixed wording per benchmark and qtype, never in the training pools
EVAL_INTRO = "This recording has two parts separated by a short tone: first the speaker's normal voice, then a second clip."
EVAL_SCORE = {
    "loudness": ("Compared with the first part, is the second part louder, about the same, or quieter?", ["quieter", "about the same", "louder"]),
    "rate": ("Compared with the first part, is the speaker talking faster, at about the same speed, or slower in the second part?",
             ["slower", "about the same", "faster"]),
    "pitch": ("Compared with the first part, is the speaker's voice higher, about the same, or lower in pitch in the second part?",
              ["lower", "about the same", "higher"]),
    "pitch_variation": ("Compared with the first part, is the speaker's intonation more varied, about the same, or flatter in the second part?",
                        ["flatter", "about the same", "more varied"]),
}
EVAL_NOUL = {
    "loudness": {"up": "Is the speaker talking louder in the second part than in the first?",
                 "same": "Is the second part about as loud as the first?"},
    "rate": {"up": "Is the speaker talking faster in the second part than in the first?",
             "same": "Is the speaker talking at about the same speed in the first and the second part?"},
    "pitch": {"up": "Is the speaker's voice higher in the second part than in the first?",
              "same": "Is the speaker's pitch about the same in the first and the second part?"},
    "pitch_variation": {"up": "Is the speaker's intonation more varied in the second part than in the first?",
                        "same": "Is the speaker's intonation about as varied in the second part as in the first?"},
}


def eval_questions() -> set[str]:
    out = set()
    for f in FEATURES:
        out.add(f"{EVAL_INTRO} {EVAL_SCORE[f][0]}")
        for d in EVAL_NOUL[f].values():
            out.add(f"{EVAL_INTRO} {d}")
    return out


def all_training_question_texts() -> set[str]:
    """Every exact question string the training pools can produce (for the leakage check)."""
    bases = []
    for f in FEATURES:
        bases += SCORE_Q[f]
        for d in NOUL_Q[f].values():
            bases += d
    out = set(bases)
    for b in bases:
        for i in INTROS:
            out.add(f"{i} {b}")
    return out


# Seeded, class-balanced sampling for the extra evaluation set (copied from our evaluation-set builder)
ADI_SEED = 0


class _NoShuffle:
    def shuffle(self, x):
        return None


NO_SHUFFLE = _NoShuffle()


def adi_take(pool: list[dict], n: int, rng: random.Random, label_of=None, strict: bool = False) -> list[dict]:
    """Seeded sample of n candidates; with `label_of` round-robin over labels; `strict` = exactly equal counts."""
    from collections import defaultdict

    pool = sorted(pool, key=lambda c: str(c["key"]))
    if label_of is None:
        return rng.sample(pool, min(n, len(pool)))
    by = defaultdict(list)
    for c in pool:
        by[label_of(c)].append(c)
    labels = sorted(by, key=str)
    for lab in labels:
        rng.shuffle(by[lab])
    if strict:
        n = min(n, len(labels) * min(len(v) for v in by.values()))
        n -= n % len(labels)
    out, i = [], 0
    while len(out) < n and any(i < len(by[lab]) for lab in labels):
        for lab in labels:
            if i < len(by[lab]) and len(out) < n:
                out.append(by[lab][i])
        i += 1
    rng.shuffle(out)
    return out


def adi_select(cands: list[dict], salt: str, cap: int, label_of=None, strict: bool = False) -> list[dict]:
    """Cap a benchmark at `cap` candidates, keeping each split's share (candidates carry "split" and "key")."""
    rng = random.Random(f"{ADI_SEED}:select:{salt}")
    n_total = len(cands)
    out = []
    for split in ("dev", "test"):
        pool = [c for c in cands if c["split"] == split]
        quota = len(pool) if n_total <= cap else round(cap * len(pool) / n_total)
        out += adi_take(pool, quota, rng, label_of, strict)
    return out
