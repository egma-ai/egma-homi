"""sb_intonation pure logic (no audio libs): contour shapes and the measured-contour classifier, pause-sentence
parsing, energy-VAD silences, question and option wording, balancing, MMSU string check."""
from __future__ import annotations

import collections
import math
import random
import re

import numpy as np

SR = 16000
BUILDER = "sb_intonation"
CLASSES = ["falling", "rising", "fall-rise", "rise-fall"]
FINALS = ["final_rise", "final_fall"]


class NoShuffle(random.Random):
    def shuffle(self, x, *a, **k):  # ordered "score" levels keep their natural order
        return None


# Contours

def contour_shape(pattern: str, u: np.ndarray, c: float = 0.5, p: float = 1.0) -> np.ndarray:
    """Global pitch movement in units of its peak-to-peak size over normalised voiced time u in [0, 1].
    c = turning point (rise-fall peak / fall-rise trough), p = curvature (falling / rising)."""
    u = np.clip(np.asarray(u, np.float64), 0.0, 1.0)
    if pattern == "falling":
        return 0.5 - u ** p
    if pattern == "rising":
        return -0.5 + u ** p
    left = np.sin(0.5 * np.pi * np.clip(u / c, 0, 1))
    right = np.sin(0.5 * np.pi * np.clip((1 - u) / (1 - c), 0, 1))
    hump = np.where(u <= c, left, right)  # 0 -> 1 (at c) -> 0
    if pattern == "rise-fall":
        return hump - 0.5
    if pattern == "fall-rise":
        return 0.5 - hump
    raise ValueError(pattern)


def final_glide(u: np.ndarray, u0: float, size: float, direction: int) -> np.ndarray:
    """0 before u0, then a glide to direction*size semitones at u = 1."""
    u = np.asarray(u, np.float64)
    r = np.clip((u - u0) / max(1e-6, 1 - u0), 0, 1) ** 1.2
    return direction * size * r


def thirds(st: np.ndarray, u: np.ndarray) -> tuple[float, float, float, float]:
    """Medians of the voiced pitch (semitones) in the three time thirds + final movement
    (median of u >= 0.85 minus median of 0.55 <= u < 0.75)."""
    st, u = np.asarray(st, np.float64), np.asarray(u, np.float64)
    m = [float(np.median(st[(u >= a) & (u <= b)])) for a, b in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0))]
    fin = float(np.median(st[u >= 0.85]) - np.median(st[(u >= 0.55) & (u < 0.75)]))
    return m[0], m[1], m[2], fin


# verification margins (semitones). Classes are mutually exclusive by construction (see test).
MARG = {"step": 1.0, "total": 2.5, "hump": 1.8, "rise_step2": 1.2, "rise_step1": 0.5}
FINAL_MIN = 2.0      # final_rise / final_fall clips: |final movement| >= 2 st with the right sign
FINAL_VIEW_MIN = 1.2  # 4-class clips: final-direction view only if |final| >= 1.2 st with the class's sign


def classify(m1: float, m2: float, m3: float, fin: float) -> str | None:
    M = MARG
    if m2 - m1 >= M["rise_step1"] and m3 - m2 >= M["rise_step2"] and m3 - m1 >= M["total"] and fin >= 0.5:
        return "rising"
    if m1 - m2 >= M["step"] and m2 - m3 >= M["step"] and m1 - m3 >= M["total"] and fin <= -0.5:
        return "falling"
    if m2 - m1 >= M["hump"] and m2 - m3 >= M["hump"]:
        return "rise-fall"
    if m1 - m2 >= M["hump"] and m3 - m2 >= M["hump"]:
        return "fall-rise"
    return None


def clean_track(f0: np.ndarray, energy_db: np.ndarray | None = None, rel_db: float = 30.0, min_run: int = 6,
                med: int = 5) -> np.ndarray:
    """Measurement hygiene: unvoice frames > rel_db below the loudest frame (quiet tails / breaths), drop voiced runs
    shorter than min_run frames, median-filter log-F0 inside each voiced run."""
    f0 = np.asarray(f0, np.float64).copy()
    if energy_db is not None:
        e = np.asarray(energy_db, np.float64)[: len(f0)]
        if len(e) < len(f0):
            e = np.pad(e, (0, len(f0) - len(e)), constant_values=-200.0)
        f0[e < e.max() - rel_db] = 0.0
    for s, e in runs(f0 > 0):
        if e - s < min_run:
            f0[s:e] = 0.0
        elif med > 1:
            lf = np.log(f0[s:e])
            h = med // 2
            pad = np.pad(lf, (h, h), mode="edge")
            f0[s:e] = np.exp(np.array([np.median(pad[i:i + med]) for i in range(e - s)]))
    return f0


def classify_track(f0: np.ndarray) -> tuple[str | None, dict]:
    """f0: per-frame Hz (0 = unvoiced). Returns (contour class or None, stats incl. final movement)."""
    f0 = np.asarray(f0, np.float64)
    idx = np.flatnonzero(f0 > 0)
    if len(idx) < 30:
        return None, {"nv": int(len(idx))}
    st = 12 * np.log2(f0[idx] / 100.0)
    u = (idx - idx[0]) / max(idx[-1] - idx[0], 1)
    m1, m2, m3, fin = thirds(st, u)
    stats = {"m1": round(m1, 2), "m2": round(m2, 2), "m3": round(m3, 2), "final": round(fin, 2), "nv": int(len(idx))}
    return classify(m1, m2, m3, fin), stats


def ends_rising(cls: str) -> bool:
    return cls in ("rising", "fall-rise", "final_rise")


# Pause sentences

def parse_placement(s: str) -> tuple[list[str], list[int]]:
    """'i met nora | lee and sam' -> (words, [index of the word before each pause])."""
    toks = s.split()
    words, pauses = [], []
    for t in toks:
        if t == "|":
            assert words, s
            pauses.append(len(words) - 1)
        else:
            words.append(t)
    return words, pauses


def norm_word(w: str) -> str:
    return re.sub(r"[^a-z]", "", w.lower())


def parse_entry(e: dict) -> dict:
    """Entry {type, q (question sentence?), placements: [(text with |, [meaning paraphrases])], foil}."""
    words0 = None
    pls = []
    for text, means in e["placements"]:
        words, pauses = parse_placement(text)
        if words0 is None:
            words0 = words
        assert [norm_word(w) for w in words] == [norm_word(w) for w in words0], (text, words0)
        assert pauses and all(0 <= p < len(words) - 1 for p in pauses), text
        assert len(means) >= 2, text
        pls.append({"pauses": sorted(pauses), "meanings": list(means)})
    assert 2 <= len(pls) <= 4, e
    keys = [tuple(p["pauses"]) for p in pls]
    assert len(set(keys)) == len(keys), e
    return {"type": e["type"], "words": words0, "placements": pls, "foil": e.get("foil"), "question": e.get("q", False)}


def tts_text(words: list[str], pauses: list[int] | None, question: bool) -> str:
    """TTS input: commas at the pause positions (pauses=None: no internal punctuation), capitalised, final mark."""
    out = []
    for i, w in enumerate(words):
        out.append(w + ("," if pauses and i in pauses else ""))
    s = " ".join(out)
    s = s[0].upper() + s[1:]
    return s + ("?" if question else ".")


# Energy VAD
VAD_FRAME, VAD_HOP = 400, 160
VAD_REL_DB = 35.0
VAD_FLOOR_DB = -60.0
MIN_RUN = 3
PAUSE_SIL_MIN = 0.25     # a measured silence >= this inside speech counts as a pause
KEY_MIN, KEY_MAX = 0.35, 1.0
MINOR_MIN, MINOR_MAX = 0.06, 0.16


def frame_db(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < VAD_FRAME:
        x = np.pad(x, (0, VAD_FRAME - len(x)))
    n = 1 + (len(x) - VAD_FRAME) // VAD_HOP
    idx = np.arange(VAD_FRAME)[None, :] + VAD_HOP * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(x[idx] ** 2, axis=1))
    return 20.0 * np.log10(rms + 1e-10)


def runs(mask) -> list[tuple[int, int]]:
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def active_frames(x: np.ndarray) -> np.ndarray:
    db = frame_db(x)
    act = db > max(float(db.max()) - VAD_REL_DB, VAD_FLOOR_DB)
    for s, e in runs(act):
        if e - s < MIN_RUN:
            act[s:e] = False
    return act


def silences(x: np.ndarray, min_s: float) -> list[tuple[float, float]]:
    """Low-energy stretches >= min_s strictly inside the speech span."""
    act = active_frames(x)
    if not act.any():
        return []
    idx = np.flatnonzero(act)
    first, last = idx[0], idx[-1]
    out = []
    for s, e in runs(~act[first:last + 1]):
        s0, e0 = (first + s) * VAD_HOP / SR, (first + e) * VAD_HOP / SR + (VAD_FRAME - VAD_HOP) / SR
        if e0 - s0 >= min_s:
            out.append((round(s0, 3), round(e0, 3)))
    return out


def match_pauses(sil: list[tuple[float, float]], inserted: dict) -> tuple[bool, dict]:
    """Every measured silence must contain exactly one key inserted interval (+-50 ms) and every key interval must be
    found. inserted: {boundary: (start_s, end_s, is_key)}. Returns (ok, {boundary: measured duration})."""
    keys = {b: v for b, v in inserted.items() if v[2]}
    found = {}
    for s, e in sil:
        hit = [b for b, (a, z, _) in keys.items() if s <= a + 0.05 and e >= z - 0.05]
        if len(hit) != 1 or hit[0] in found:
            return False, found
        found[hit[0]] = round(e - s, 3)
    return set(found) == set(keys), found


# Question wording
Q = {
    "inton_choice": [
        "Which description best fits the pitch contour of this sentence?",
        "How does the speaker's pitch move across the whole utterance?",
        "Listen to the melody of the voice from beginning to end. Which pattern is it?",
        "What is the overall intonation pattern of this sentence?",
        "Over the full sentence, which shape does the pitch trace?",
        "Ignoring the words, how would you describe the rise and fall of the voice?",
        "Which intonation contour does this utterance follow?",
        "Track the pitch from the first word to the last. Which option matches?",
        "What happens to the pitch of the voice over the course of the sentence?",
        "Pick the pitch pattern that matches the whole utterance.",
    ],
    "inton_noul": [
        "Does the pitch of this sentence follow a {d} pattern?",
        "Is the overall intonation of the utterance {d}?",
        "Would you describe the melody of this sentence as {d}?",
        "Taken as a whole, is the pitch contour here {d}?",
        "Does the voice follow a {d} contour from start to end?",
        "Is {d} a good description of how the pitch moves in this sentence?",
        "Listening to the whole utterance, is its pitch movement {d}?",
        "Over the entire sentence, is the speaker's intonation {d}?",
    ],
    "final_choice": [
        "At the very end of the sentence, does the pitch go up or down?",
        "How does the voice move in pitch on the last word?",
        "Does this utterance finish with a rise or a fall in pitch?",
        "Listen to the final syllables. Which way does the pitch go?",
        "What does the pitch do at the end of the utterance?",
        "Is the final pitch movement upward or downward?",
        "How does the sentence end, in terms of pitch?",
        "Which best describes the pitch at the close of the sentence?",
        "On the final word, which direction does the speaker's pitch take?",
    ],
    "final_noul": [
        "Does the pitch {v} at the very end of the sentence?",
        "Does the utterance end with a pitch {v}?",
        "Is there a final pitch {v} on the last word?",
        "At the end, does the speaker's voice {v} in pitch?",
        "Does the sentence close with a {v} in pitch?",
        "On the last word, does the pitch {v}?",
        "Is the last pitch movement of the sentence a {v}?",
        "Listening to the end of the utterance, does the pitch {v}?",
    ],
    "pm_meaning": [
        "Given where the speaker pauses, which meaning is intended?",
        "Taking the speaker's pause into account, what is being said?",
        "The placement of the pause decides the meaning here. Which interpretation fits?",
        "Which interpretation matches the way the speaker groups the words with the pause?",
        "Based on the speaker's phrasing, which reading is correct?",
        "Listen to where the clear break is. What does the speaker mean?",
        "How should the sentence be understood, judging by the speaker's pause?",
        "Considering the pause, which statement captures the speaker's meaning?",
        "The words could be read more than one way. Which reading does the pause point to?",
        "Which meaning does the speaker's pausing support?",
    ],
    "pm_word": [
        "Where does the speaker make the clear pause?",
        "Where in the sentence is the clearly longest break?",
        "At which point does the speaker pause noticeably?",
        "Where does the main break in this sentence come?",
        "Locate the speaker's most noticeable pause. Where is it?",
        "Where does the speaker insert a distinct pause?",
        "Which position has the clearly audible pause?",
        "Where does the speaker clearly stop for a moment?",
        "At which word boundary is the obvious pause?",
    ],
    "pm_meaning_noul": [
        "Given the pause, does the speaker mean this: {m}",
        "Is this what the speaker means, judging by the pause? {m}",
        "Judging by where the pause falls, is the following the intended meaning? {m}",
        "Does the speaker's phrasing support this reading? {m}",
        "Based on the pauses, is this interpretation correct? {m}",
        "Taking the pause into account, is it right to understand the sentence like this: {m}",
        "Do the words, grouped the way the speaker pauses, mean this? {m}",
        "Considering the speaker's pause, is this the intended meaning? {m}",
    ],
    "pm_word_noul": [
        "Is there a clear pause right after the word \"{w}\"?",
        "Does the speaker make a noticeable pause after \"{w}\"?",
        "Is \"{w}\" followed by a distinct break?",
        "Does a clear pause come directly after the word \"{w}\"?",
        "After saying \"{w}\", does the speaker pause noticeably?",
        "Is there an audible break immediately following \"{w}\"?",
        "Does the speaker stop clearly after \"{w}\" before going on?",
        "Right after \"{w}\", is there a clear pause?",
    ],
}

# option wordings for the 4 classes (one style per item; index = CLASSES order)
CLASS_STYLES = [
    ["falling", "rising", "fall-rise", "rise-fall"],
    ["falling intonation", "rising intonation", "falling, then rising", "rising, then falling"],
    ["the pitch goes down", "the pitch goes up", "the pitch dips and comes back up", "the pitch goes up and comes back down"],
    ["high to low", "low to high", "high, low, then high again", "low, high, then low again"],
    ["a steady fall", "a steady rise", "a dip in the middle", "a peak in the middle"],
    ["descending", "ascending", "descending then ascending", "ascending then descending"],
    ["downward", "upward", "down, then up", "up, then down"],
    ["falling contour", "rising contour", "fall-rise contour", "rise-fall contour"],
]
CLASS_NOUL_DESC = {
    "falling": ["falling", "steadily falling", "high-to-low"],
    "rising": ["rising", "steadily rising", "low-to-high"],
    "fall-rise": ["fall-rise", "falling-then-rising", "down-then-up"],
    "rise-fall": ["rise-fall", "rising-then-falling", "up-then-down"],
}
FINAL_STYLES = [["it rises", "it falls"], ["up", "down"], ["rising", "falling"], ["a final rise", "a final fall"],
                ["higher at the end", "lower at the end"], ["goes up", "goes down"], ["upward", "downward"]]
WORD_OPT_STYLES = ['after "{w}"', 'right after "{w}"', 'after the word "{w}"']


def boundary_label(words: list[str], b: int, style: int) -> str:
    """Option text for a pause after word b. Words that occur more than once are named by the pair around the pause."""
    w = words[b].strip(",.?")
    if sum(1 for x in words if norm_word(x) == norm_word(w)) > 1:
        return f'between "{w}" and "{words[b + 1].strip(",.?")}"'
    return WORD_OPT_STYLES[style].format(w=w)


def placement_label(words: list[str], pauses: list[int], style: int) -> str:
    return " and ".join(boundary_label(words, b, style if k == 0 else 0) for k, b in enumerate(pauses))


def all_templates() -> list[str]:
    return [q for v in Q.values() for q in v]


# Balancing and checks

def norm_q(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def mmsu_equal_count(questions: list[str], mmsu_questions: list[str]) -> int:
    """Number of item questions equal (exactly or after normalisation) to an MMSU question string."""
    exact = set(mmsu_questions)
    normed = {norm_q(x) for x in mmsu_questions}
    return sum(1 for q in questions if q in exact or norm_q(q) in normed)


RATIO = {"inton_choice": 1.15}  # tighter cap for the 4 contour classes (default 2x the rarest)


def balance(items: list[dict], seed: int) -> tuple[list[dict], dict]:
    """noul: exactly 50/50 per (source, qkind, split); choice/score: each answer class capped at RATIO (default 2) x
    the rarest per (source, qkind, split, meta.bal_group). answer class = meta['answer_class']."""
    rng = random.Random(f"{seed}:balance")
    groups = collections.defaultdict(list)
    for it in items:
        bg = it["meta"].get("bal_group", "") if it["meta"]["qtype"] != "noul" else ""
        groups[(it["source"], it["meta"]["qkind"], it["set"], bg)].append(it)
    out, report = [], {}
    for key in sorted(groups):
        g = groups[key]
        by = collections.defaultdict(list)
        for it in g:
            by[it["meta"]["answer_class"]].append(it)
        if g[0]["meta"]["qtype"] == "noul":
            k = min(len(by.get("yes", [])), len(by.get("no", [])))
            keep = []
            for c in ("yes", "no"):
                lst = list(by.get(c, []))
                rng.shuffle(lst)
                keep += lst[:k]
        else:
            m = min(len(v) for v in by.values())
            cap = int(math.floor(RATIO.get(key[1], 2.0) * m))
            keep = []
            for c in sorted(by):
                lst = list(by[c])
                rng.shuffle(lst)
                keep += lst[:cap]
        report["/".join(k for k in key if k)] = {"before": len(g), "after": len(keep),
                                 "classes": dict(collections.Counter(it["meta"]["answer_class"] for it in keep))}
        out += keep
    out.sort(key=lambda it: it["id"])
    return out, report


def cap_per_clip(items: list[dict], cap: int = 4) -> list[dict]:
    seen = collections.Counter()
    out = []
    for it in items:
        k = (it["audio"]["bank"], it["audio"]["key"])
        if seen[k] < cap:
            seen[k] += 1
            out.append(it)
    return out


def pearson(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def safe_round(x, k=3):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), k)
