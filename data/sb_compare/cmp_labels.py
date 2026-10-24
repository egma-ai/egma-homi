"""Label logic, margins and question wording for sb_compare (no audio I/O).

Families: pitch / speed / volume / range (2-4 segments), duration, pause and pause_meaning; labels come from measurement.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import random
import re

import numpy as np

SR = 16_000

# Energy VAD (same definitions as sb_voice / sb_relative: 25 ms frames, 10 ms hop)
VAD_FRAME, VAD_HOP = 400, 160
VAD_REL_DB = 35.0
VAD_FLOOR_DB = -60.0
MIN_RUN = 3


def frame_db(x: np.ndarray, frame: int = VAD_FRAME, hop: int = VAD_HOP) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(x[idx] ** 2, axis=1))
    return 20.0 * np.log10(rms + 1e-10)


def runs(mask) -> list[tuple[int, int]]:
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def active_frames(x: np.ndarray, ref_db: float | None = None, rel_db: float = VAD_REL_DB) -> np.ndarray:
    """Active = within rel_db of the loudest frame (or of ref_db) and above -60 dBFS; runs < 3 frames removed."""
    db = frame_db(x)
    top = float(db.max()) if ref_db is None else ref_db
    act = db > max(top - rel_db, VAD_FLOOR_DB)
    for s, e in runs(act):
        if e - s < MIN_RUN:
            act[s:e] = False
    return act


def speech_span(x: np.ndarray, ref_db: float | None = None) -> tuple[float, float] | None:
    """(start_s, end_s) of the first..last active frame."""
    act = active_frames(x, ref_db)
    if not act.any():
        return None
    idx = np.flatnonzero(act)
    return idx[0] * VAD_HOP / SR, (idx[-1] * VAD_HOP + VAD_FRAME) / SR


def silences(x: np.ndarray, min_s: float, ref_db: float | None = None) -> list[tuple[float, float]]:
    """Low-energy stretches >= min_s strictly inside the speech span (start_s, end_s)."""
    act = active_frames(x, ref_db)
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


def active_db(x: np.ndarray) -> float | None:
    """Mean power (dB) of the active frames (loudness measure for volume comparisons)."""
    db = frame_db(x)
    act = active_frames(x)
    if act.sum() < 10:
        return None
    p = 10 ** (db[act] / 10.0)
    return float(10 * np.log10(p.mean() + 1e-20))


def f0_stats(f0: np.ndarray, voiced: np.ndarray) -> dict:
    """Median F0 (Hz) and robust F0 spread in semitones ((P90 - P10) / 2.563) over voiced frames (sb_voice)."""
    f = np.asarray(f0, dtype=float)[np.asarray(voiced, bool)]
    f = f[np.isfinite(f) & (f > 0)]
    if len(f) < 10:
        return {"f0_med": None, "f0_spread_st": None, "n_voiced": int(len(f))}
    med = float(np.median(f))
    st = 12.0 * np.log2(f / med)
    p10, p90 = np.percentile(st, [10, 90])
    return {"f0_med": med, "f0_spread_st": float((p90 - p10) / 2.563), "n_voiced": int(len(f))}


# Syllables (sb_voice heuristic; CMUdict used when available)
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


# Comparison families: values -> levels with margins
# feature unit transforms: pitch in semitones (12 log2 Hz), speed in log(rate), volume in dB, range in log(spread)
def to_unit(family: str, v: float) -> float:
    if family == "pitch":
        return 12.0 * math.log2(v)
    if family in ("speed", "range"):
        return math.log(v)
    return float(v)  # volume (dB)


# minimum separation between any two segments for ranking / pattern labels (in the unit above)
SEP = {"pitch": 1.5, "speed": math.log(1.15), "volume": 3.0, "range": math.log(1.4)}
# "about the same" (2-segment score items): |difference| <= SAME
SAME = {"pitch": 0.5, "speed": math.log(1.04), "volume": 1.0, "range": math.log(1.08)}
PITCH_MAX_JUMP = 14.0  # semitones; larger median-F0 differences between segments = likely octave errors, not labelled


def rank_levels(units: list[float], family: str) -> list[int] | None:
    """Rank of each segment (0 = lowest) if every pair is separated by >= SEP[family], else None."""
    if any(u is None or not math.isfinite(u) for u in units):
        return None
    srt = sorted(units)
    if any(b - a < SEP[family] for a, b in zip(srt, srt[1:])):
        return None
    if family == "pitch" and srt[-1] - srt[0] > PITCH_MAX_JUMP:
        return None
    order = sorted(range(len(units)), key=lambda i: units[i])
    r = [0] * len(units)
    for k, i in enumerate(order):
        r[i] = k
    return r


def pair_class(u_first: float, u_second: float, family: str) -> int | None:
    """second vs first: 0 lower / 1 about the same / 2 higher, None if in the ambiguous zone."""
    d = u_second - u_first
    if abs(d) <= SAME[family]:
        return 1
    if d >= SEP[family]:
        return 2
    if d <= -SEP[family]:
        return 0
    return None


LEVEL_WORDS = {  # low -> high, per family, several vocabularies
    "pitch": [["low", "medium", "high"]],
    "speed": [["low", "medium", "high"], ["slow", "medium", "fast"]],
    "volume": [["low", "medium", "high"], ["soft", "medium", "loud"], ["quiet", "medium", "loud"]],
    "range": [["low", "medium", "high"], ["narrow", "medium", "wide"]],
}


def pattern_string(ranks: list[int], words: list[str], style: int) -> str:
    seq = [words[r] for r in ranks]
    if style == 0:
        return "-".join(seq)
    if style == 1:
        return ", ".join(seq)
    return seq[0] + ", then " + ", then ".join(seq[1:])


def pattern_options(ranks: list[int], n_opts: int, rng: random.Random) -> tuple[list[tuple[int, ...]], int]:
    """All permutations of the 3 levels; gold + (n_opts - 1) distractors (canonical order: sorted permutations)."""
    perms = list(itertools.permutations(range(3)))
    gold = tuple(ranks)
    others = [p for p in perms if p != gold]
    rng.shuffle(others)
    chosen = sorted([gold] + others[: n_opts - 1])
    return chosen, chosen.index(gold)


ORD = ["first", "second", "third", "fourth"]


# Duration family
DUR_BINNINGS = [  # (cut points in s, labels) — labels low -> high
    ([5.0, 10.0, 15.0], ["less than 5 seconds", "5 to 10 seconds", "10 to 15 seconds", "more than 15 seconds"]),
    ([4.0, 8.0, 12.0], ["under 4 seconds", "4 to 8 seconds", "8 to 12 seconds", "over 12 seconds"]),
    ([3.0, 6.0, 10.0], ["under 3 seconds", "3 to 6 seconds", "6 to 10 seconds", "over 10 seconds"]),
    ([6.0, 12.0], ["under 6 seconds", "6 to 12 seconds", "over 12 seconds"]),
]
DUR_MARGIN = 0.6  # s, around every cut point
GAP_BINNING = ([0.75, 1.5, 3.0], ["shorter than 0.75 seconds", "about 1 second (0.75 to 1.5 s)",
                                  "1.5 to 3 seconds", "longer than 3 seconds"])
GAP_MARGIN = 0.2


def bin_index(v: float, cuts: list[float], margin: float) -> int | None:
    for c in cuts:
        if abs(v - c) < margin:
            return None
    return sum(1 for c in cuts if v > c)


def numeric_options(true_v: float, rng: random.Random, n: int = 4, lo: float = 0.8) -> tuple[list[float], int] | None:
    """Numeric second values: gold = true_v rounded to 0.1 s; distractors differ from gold and from each other by
    >= max(1.2 s, 20 % of the larger value). The gold's rank among the options is drawn uniformly from the feasible
    ranks (short clips cannot have smaller distractors). Returns (ascending values, gold index)."""
    gold = round(true_v, 1)

    def sep(a, b):
        return abs(a - b) >= max(1.2, 0.2 * max(a, b)) - 1e-9

    max_below, v = 0, gold
    while max_below < n - 1:
        nxt = v - max(1.2, 0.2 * v)
        if nxt < lo:
            break
        max_below += 1
        v = nxt
    p = rng.randint(0, max_below)
    for _ in range(50):
        vals, v = [gold], gold
        for _k in range(p):
            d = max(1.2, 0.2 * v)
            c = round(v - d * rng.uniform(1.0, 1.5), 1)
            if c < lo or not sep(c, v):
                c = round(v - d - 0.05, 1)
            vals.append(c)
            v = c
        v = gold
        for _k in range(n - 1 - p):
            d = max(1.25, 0.26 * v)
            c = round(v + d * rng.uniform(1.0, 1.5), 1)
            vals.append(c)
            v = c
        vals = sorted(vals)
        if min(vals) >= lo and all(sep(a, b) for a, b in itertools.combinations(vals, 2)) and len(set(vals)) == n:
            return vals, vals.index(gold)
    return None


def fmt_seconds(v: float, style: int) -> str:
    s = f"{v:.1f}"
    return [s, f"{s} seconds", f"about {s} s"][style]


# Pause family (TTS with word timings)
PAUSE_MIN_S = 0.45      # an inserted pause is >= this long (measured)
NO_PAUSE_MAX_S = 0.22   # a "no pause" sentence has no internal silence longer than this (measured)
PAUSE_DUR_BINS = ([0.7, 1.4, 2.6], ["less than 0.7 seconds", "about one second", "about two seconds", "three seconds or more"])
PAUSE_DUR_DRAW = [(0.5, 0.6), (0.9, 1.15), (1.75, 2.25), (3.0, 4.0)]  # planned pause lengths per bin
PAUSE_DUR_MARGIN = 0.12


def pause_after_word(sil: tuple[float, float], word_ends: list[float], tol: float = 0.12) -> int | None:
    """Index of the word whose (rendered) end time is at the silence start (within tol), else None."""
    best = None
    for i, e in enumerate(word_ends):
        if abs(sil[0] - e) <= tol and (best is None or abs(sil[0] - e) < abs(sil[0] - word_ends[best])):
            best = i
    return best


def third_of(index: int, n_words: int) -> int:
    """0/1/2 = beginning / middle / end of the sentence for a pause after word `index` (0-based) of n_words."""
    pos = (index + 1) / n_words
    return 0 if pos <= 1 / 3 else (1 if pos <= 2 / 3 else 2)


def clean_words(text: str) -> list[str]:
    """Lower-case alphabetic words (apostrophes kept); used for TTS input without internal punctuation."""
    return [w for w in re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())]


# ambiguous sentences: the meaning depends on where the pause is. words are TTS input; each reading = (pauses after
# word index list, meaning). distractors are plausible wrong meanings. All written for this builder.
PAUSE_MEANING = [
    {"words": "please call {a} {b} and {c} tomorrow",
     "readings": [([1], "Call three people: {A}, {B} and {C}."), ([2], "Call two people: {A} {B} and {C}.")],
     "distractors": ["Call only {C}.", "{A} will call {C} tomorrow."], "slots": "names3"},
    {"words": "{a} said {b} is the best player",
     "readings": [([0, 2], "{B} said that {A} is the best player."), ([1], "{A} said that {B} is the best player.")],
     "distractors": ["Both of them are the best players.", "Nobody said anything about the best player."], "slots": "names2"},
    {"words": "we need eggs milk bread and butter",
     "readings": [([2, 3], "We need four things: eggs, milk, bread and butter."), ([2], "We need eggs, milk, and bread with butter on it.")],
     "distractors": ["We need only eggs.", "We already have eggs, milk, bread and butter."], "slots": None},
    {"words": "the manager said the driver was late again",
     "readings": [([1, 4], "The driver said that the manager was late again."), ([2], "The manager said that the driver was late again.")],
     "distractors": ["Both of them were late again.", "Nobody was late."], "slots": None},
    {"words": "i invited my parents {a} and {b}",
     "readings": [([3, 4], "Four or more people were invited: the parents, {A} and {B}."), ([2], "The speaker's parents are {A} and {B}.")],
     "distractors": ["Only {A} was invited.", "{A} invited {B}."], "slots": "names2"},
    {"words": "{a} told {b} the doctor would call",
     "readings": [([2], "{A} told {B} that the doctor would call."), ([0, 3], "{B}, who is the doctor, was told something by {A}; the doctor would call.")],
     "distractors": ["The doctor told {A} to call {B}.", "{B} called the doctor."], "slots": "names2"},
    {"words": "the meeting is on monday not tuesday morning",
     "readings": [([4], "The meeting is on Monday morning, not on Tuesday."), ([3], "The meeting is on Monday, and it is not on Tuesday morning.")],
     "distractors": ["The meeting is on Tuesday morning.", "There is no meeting this week."], "slots": None},
    {"words": "i want the red and blue shirts and the hat",
     "readings": [([5], "I want the shirts and also the hat."), ([2], "I want the red one, and the blue shirts and the hat.")],
     "distractors": ["I want only the hat.", "I don't want any shirts."], "slots": None},
    {"words": "we sold the old cars and trucks yesterday",
     "readings": [([4], "Both the cars and the trucks that were sold are old."), ([3], "We sold the old cars, and also some trucks that may not be old.")],
     "distractors": ["We bought new cars yesterday.", "Nothing was sold yesterday."], "slots": None},
    {"words": "{a} met {b} and {c} at the station",
     "readings": [([0], "Someone met three people: {A}, {B} and {C}."), ([2], "{A} met two people, {B} and {C}.")],
     "distractors": ["{B} met {A} at home.", "Nobody went to the station."], "slots": "names3"},
    {"words": "the price is two hundred and fifty dollars each",
     "readings": [([3], "Two hundred dollars, plus fifty dollars per item."), ([5], "Each item costs two hundred and fifty dollars.")],
     "distractors": ["Each item costs twenty five dollars.", "The total price is fifty dollars."], "slots": None},
    {"words": "{a} the new nurse will help you today",
     "readings": [([0], "The speaker is talking to {A}; a new nurse will help them."), ([3], "{A} is the new nurse who will help you.")],
     "distractors": ["Nobody can help today.", "The old nurse will help {A}."], "slots": "names1"},
    {"words": "the kids ate the cake {a} brought and left",
     "readings": [([5], "{A} brought the cake; the kids ate it and left."), ([4], "The kids ate the cake; {A} brought something and left.")],
     "distractors": ["{A} ate the cake.", "Nobody ate the cake."], "slots": "names1"},
    {"words": "i called my brother {a} and my friend",
     "readings": [([3], "The speaker called three people: a brother, {A} and a friend."), ([4], "The speaker called the brother, whose name is {A}, and a friend.")],
     "distractors": ["{A} called the speaker.", "The speaker called nobody."], "slots": "names1"},
    {"words": "send the forms to {a} {b} and the bank",
     "readings": [([3], "Send the forms to three places: {A}, {B} and the bank."), ([4], "Send the forms to {A} {B}, one person, and to the bank.")],
     "distractors": ["Keep the forms.", "Send the forms only to the bank."], "slots": "names2"},
    {"words": "the coach said the team is not ready",
     "readings": [([1, 4], "The team said that the coach is not ready."), ([2], "The coach said that the team is not ready.")],
     "distractors": ["The coach and the team are both ready.", "Nobody said anything."], "slots": None},
]
FIRST_NAMES = ["anna", "peter", "maria", "james", "laura", "david", "sophie", "daniel", "emma", "lucas", "grace", "oliver",
               "nina", "henry", "clara", "simon", "julia", "martin", "rose", "victor", "helen", "oscar", "alice", "felix"]
SURNAMES = ["smith", "brown", "taylor", "wilson", "clark", "lewis", "walker", "young", "allen", "king", "wright", "hill"]


def fill_meaning_template(t: dict, rng: random.Random) -> dict:
    """Fill name slots; for names3/names2-with-surname templates, slot b is a surname where the reading needs it."""
    names = rng.sample(FIRST_NAMES, 3)
    if t["words"].startswith("please call") or t["words"].startswith("send the forms"):
        names[1] = rng.choice(SURNAMES)
    sub = {"a": names[0], "b": names[1], "c": names[2]}
    words = t["words"].format(**sub).split()
    cap = {k.upper(): v.capitalize() for k, v in sub.items()}
    readings = [(p, m.format(**cap)) for p, m in t["readings"]]
    distractors = [d.format(**cap) for d in t["distractors"]]
    return {"words": words, "readings": readings, "distractors": distractors}


# Question wording (all written for this builder; >= 8 paraphrases per family)
INTRO = {
    "same_rep": ["The same sentence is spoken {n} times.", "You will hear one sentence repeated {n} times.",
                 "A speaker says the same words {n} times in a row.", "This recording has {n} versions of the same sentence."],
    "same_spk": ["One speaker says {n} different sentences.", "You will hear {n} sentences from the same speaker.",
                 "This recording has {n} parts, all from one speaker.", "The same person speaks {n} times."],
    "diff_spk": ["{N} different people speak one after another.", "You will hear {n} speakers in turn.",
                 "This recording has {n} parts, each from a different speaker.", "{N} voices speak one after the other."],
}
NUMW = {2: "two", 3: "three", 4: "four"}

FEAT_NOUN = {"pitch": ["pitch", "voice pitch", "pitch of the voice"],
             "speed": ["speaking speed", "speaking rate", "speed of speech", "tempo of the speech"],
             "volume": ["volume", "loudness", "sound level"],
             "range": ["vocal range", "pitch range", "amount of pitch variation", "melodic range of the voice"]}

PATTERN_Q = [
    "How does the {feat} change across the {n} parts? Pick the matching pattern.",
    "Listen to the {n} parts in order. Which sequence describes their {feat}?",
    "Which pattern of {feat} (first part to last part) fits this recording?",
    "Going from the first part to the last, which pattern matches the {feat}?",
    "Describe the {feat} of the {n} parts in order. Which option is right?",
    "Rank the {n} parts by {feat} in the order you hear them. Which pattern is correct?",
    "What is the {feat} pattern over the {n} segments, in order?",
    "In order of appearance, how would you label the {feat} of each part?",
    "Which option lists the {feat} of each segment correctly, from first to last?",
]
EXTREME_Q = {  # (question, which end) — 'max' asks for the highest level, 'min' for the lowest
    "pitch": [("Which part has the highest pitch?", "max"), ("In which segment is the voice pitched highest?", "max"),
              ("Which part is spoken with the lowest pitch?", "min"), ("Where is the voice deepest (lowest pitch)?", "min"),
              ("Which segment sounds highest in pitch?", "max"), ("Which of the parts has the lowest voice pitch?", "min"),
              ("Pick the part with the highest-pitched voice.", "max"), ("Pick the segment with the lowest pitch.", "min")],
    "speed": [("Which part is spoken the fastest?", "max"), ("In which segment does the speaker talk most quickly?", "max"),
              ("Which part is spoken the slowest?", "min"), ("Where is the speech slowest?", "min"),
              ("Which segment has the fastest speaking rate?", "max"), ("Which of the parts is the slowest?", "min"),
              ("Pick the part with the quickest delivery.", "max"), ("Pick the segment with the slowest speaking rate.", "min")],
    "volume": [("Which part is the loudest?", "max"), ("In which segment is the speech loudest?", "max"),
               ("Which part is the quietest?", "min"), ("Where is the voice softest?", "min"),
               ("Which segment has the highest volume?", "max"), ("Which of the parts has the lowest volume?", "min"),
               ("Pick the loudest part.", "max"), ("Pick the segment that is played most quietly.", "min")],
    "range": [("Which part has the widest pitch range?", "max"), ("In which segment does the pitch vary the most?", "max"),
              ("Which part sounds the most monotone (narrowest pitch range)?", "min"),
              ("Where does the voice move up and down the least?", "min"),
              ("Which segment is the most melodic, with the biggest pitch movement?", "max"),
              ("Which of the parts has the flattest intonation?", "min"),
              ("Pick the part with the most varied pitch.", "max"), ("Pick the segment with the smallest vocal range.", "min")],
}
EXTREME_OPTS = [["the first part", "the second part", "the third part", "the fourth part"],
                ["first", "second", "third", "fourth"], ["segment 1", "segment 2", "segment 3", "segment 4"]]

PAIR_NOUL = {  # "Is part {j} <more> than part {i}?" — answer yes iff measured unit(j) > unit(i)
    "pitch": ["Is the {j} part higher in pitch than the {i} part?", "Does the voice sound higher in the {j} part than in the {i} part?",
              "Compared with the {i} part, is the {j} part pitched higher?", "Is the pitch of segment {J} higher than that of segment {I}?"],
    "speed": ["Is the {j} part spoken faster than the {i} part?", "Does the speaker talk more quickly in the {j} part than in the {i} part?",
              "Compared with the {i} part, is the {j} part faster?", "Is segment {J} spoken at a higher speed than segment {I}?"],
    "volume": ["Is the {j} part louder than the {i} part?", "Does the {j} part sound louder than the {i} part?",
               "Compared with the {i} part, is the {j} part played at a higher volume?", "Is segment {J} louder than segment {I}?"],
    "range": ["Does the pitch vary more in the {j} part than in the {i} part?", "Is the {j} part more expressive in pitch than the {i} part?",
              "Compared with the {i} part, does the {j} part use a wider pitch range?", "Does segment {J} have a wider vocal range than segment {I}?"],
}
PAIR_SCORE = {  # two-part items; levels low -> high
    "pitch": (["Compared with the first part, how is the pitch of the second part?", "How does the second part's pitch compare with the first part's?",
               "Relative to the first segment, the voice in the second segment is pitched ...", "Is the second part lower, the same, or higher in pitch than the first?"],
              [["lower", "about the same", "higher"], ["lower pitch", "same pitch", "higher pitch"]]),
    "speed": (["Compared with the first part, how fast is the second part?", "How does the speaking speed of the second part compare with the first?",
               "Relative to the first segment, the second segment is spoken ...", "Is the second part slower, the same speed, or faster than the first?"],
              [["slower", "about the same speed", "faster"], ["more slowly", "at the same speed", "more quickly"]]),
    "volume": (["Compared with the first part, how loud is the second part?", "How does the volume of the second part compare with the first?",
                "Relative to the first segment, the second segment is ...", "Is the second part quieter, about as loud, or louder than the first?"],
               [["quieter", "about the same", "louder"], ["softer", "equally loud", "louder"]]),
    "range": (["Compared with the first part, how much does the pitch move in the second part?",
               "How does the pitch variation of the second part compare with the first?",
               "Relative to the first segment, the intonation of the second segment is ...",
               "Is the second part flatter, about as varied, or more varied in pitch than the first?"],
              [["flatter", "about the same", "more varied"], ["narrower range", "similar range", "wider range"]]),
}
SAME_NOUL = {
    "pitch": ["Are both parts spoken at about the same pitch?", "Is the pitch of the two parts roughly equal?"],
    "speed": ["Are both parts spoken at about the same speed?", "Is the speaking rate roughly equal in the two parts?"],
    "volume": ["Are the two parts about equally loud?", "Is the volume roughly the same in both parts?"],
    "range": ["Do both parts have about the same pitch variation?", "Is the vocal range roughly equal in the two parts?"],
}

DUR_TOTAL_Q = [
    "For how many seconds in total does someone speak in this recording?",
    "Add up all the speech in the clip. About how long is it, in seconds?",
    "Ignoring the silent gaps, how much speaking time does this audio contain?",
    "What is the combined duration of all the speech you hear?",
    "How long is the speech in total, not counting the silences?",
    "Estimate the total time during which a voice is talking.",
    "If you remove every silence, how long would the remaining speech last?",
    "Roughly how many seconds of talking are in this clip?",
    "Total up the speaking time in this recording. Which value is closest?",
]
DUR_BIN_Q = [
    "How much speech does this recording contain in total?",
    "Not counting pauses, how long does the talking last overall?",
    "Which range contains the total speaking time?",
    "How long is all of the speech together?",
    "What is the overall amount of speaking time in the clip?",
    "Summing every spoken part, how long is the speech?",
    "Which option best describes the total duration of speech?",
    "Leaving out the silent stretches, how long is the audio's speech?",
]
DUR_NOUL_Q = [
    "Is there more than {N} seconds of speech in total?",
    "Does the total speaking time exceed {N} seconds?",
    "Altogether, do the speakers talk for longer than {N} seconds?",
    "Is the combined speech longer than {N} seconds?",
    "Counting only the talking, does it add up to more than {N} seconds?",
    "Is the total time spent speaking above {N} seconds?",
    "Excluding silences, is the speech longer than {N} seconds?",
    "Does someone speak for more than {N} seconds overall?",
]
GAP_Q = [
    "How long is the silence between the first and the second sentence?",
    "How much time passes between the end of the first sentence and the start of the next one?",
    "What is the length of the gap after the first sentence?",
    "How long does the pause between the first two sentences last?",
    "After the first sentence ends, how long is it quiet before speech starts again?",
    "Estimate the duration of the break between sentence one and sentence two.",
    "How long is the gap separating the first and second spoken parts?",
    "What is the duration of the quiet stretch between the first two utterances?",
]
LONGER_Q = [
    "Which sentence lasts longer, the first or the second?", "Is the first or the second utterance longer in time?",
    "Which of the two spoken parts takes more time?", "Which part is longer in duration?",
    "Which takes longer to say: the first sentence or the second one?", "Of the two utterances, which one is longer?",
    "Which spoken segment has the longer duration?", "Which sentence goes on for longer?",
]

PAUSE_WORD_Q = [
    "After which word does the speaker pause? Choose 'no pause' if there is none.",
    "Which word comes right before a pause? If the speaker never pauses, pick 'no pause'.",
    "The speaker may stop briefly after one word. Which word is it? Select 'no pause' if there is no break.",
    "Where does the speaker take a break: after which of these words? (Or is there no pause?)",
    "Which of these words is immediately followed by a silence? Answer 'no pause' if none is.",
    "Listen for a gap in the sentence. After which word is it, if any?",
    "Identify the word that precedes the pause, or choose 'no pause'.",
    "Is there a word after which the speaker goes silent for a moment? Which one?",
    "Which word does the speaker hold a pause after? If the sentence flows without a break, choose 'no pause'.",
]
NO_PAUSE_OPTS = ["no pause", "No pause", "there is no pause", "none (no pause)"]
PAUSE_YN_Q = [
    "Does the speaker pause in the middle of the sentence?",
    "Is there a noticeable break inside the sentence?",
    "Does the speaker stop for a moment partway through?",
    "Is there a silent gap somewhere in the middle of this sentence?",
    "Does the sentence contain a clear pause?",
    "Is the sentence spoken without any break?",          # answer inverted (flag)
    "Does the speaker say the whole sentence in one go, without stopping?",  # inverted
    "Can you hear a pause between two of the words?",
]
PAUSE_YN_INVERTED = {5, 6}
PAUSE_COUNT_Q = [
    "How many pauses does the speaker make within the sentence?",
    "How many times does the speaker stop briefly in the middle of the sentence?",
    "Count the silent breaks inside the sentence. How many are there?",
    "How many noticeable gaps are there between words?",
    "How many times is the sentence interrupted by a pause?",
    "Count the pauses in this sentence.",
    "How many breaks does the speaker take while saying this sentence?",
    "How many silent moments are there between the words?",
]
PAUSE_DUR_Q = [
    "How long is the pause in the sentence?",
    "The speaker pauses once. How long does the pause last?",
    "What is the length of the silence in the middle of the sentence?",
    "How long is the break between the words?",
    "Estimate the duration of the speaker's pause.",
    "How much time does the pause take?",
    "For how long does the speaker stop talking mid-sentence?",
    "How long is the gap inside this sentence?",
]
PAUSE_WHERE_Q = [
    "Where in the sentence does the speaker pause?",
    "Is the pause near the beginning, in the middle, or near the end of the sentence?",
    "In which part of the sentence is the break?",
    "Where does the silent gap occur within the sentence?",
    "At what point in the sentence does the speaker stop briefly?",
    "Which part of the sentence contains the pause?",
    "Does the pause come early, midway, or late in the sentence?",
    "Locate the pause: beginning, middle, or end of the sentence?",
]
WHERE_OPTS = [["near the beginning", "in the middle", "near the end"], ["early", "midway", "late"],
              ["in the first third", "in the middle third", "in the last third"]]
PAUSE_LONGER_Q = [
    "The speaker pauses twice. Which pause is longer?",
    "There are two breaks in the sentence. Is the first or the second one longer?",
    "Which of the two pauses lasts longer?",
    "Of the two silences in the sentence, which is longer?",
    "Compare the two pauses. Which one is longer?",
    "Which gap is longer: the first pause or the second?",
    "The sentence has two pauses. Which lasts more time?",
    "Which break is the longer one, the earlier or the later?",
]
MEANING_Q = [
    "Judging by where the speaker pauses, what does the sentence mean?",
    "The pauses change the meaning of this sentence. Which meaning is intended?",
    "Given the speaker's phrasing and pauses, which interpretation is correct?",
    "Listen to how the words are grouped by the pauses. What is the speaker saying?",
    "Which reading of the sentence matches the speaker's pauses?",
    "Based on the way the sentence is broken up, what is meant?",
    "Taking the pause into account, which statement matches what was said?",
    "How should this sentence be understood, given where the speaker stops?",
    "The speaker's pause groups some words together. Which meaning follows?",
]


LEVEL_VOCAB = {w for fam in LEVEL_WORDS.values() for ws in fam for w in ws} | {"then"}


def is_generic_pattern(option: str) -> bool:
    """'high-low-medium' style labels (only level words): a generic answer format."""
    toks = [t for t in re.split(r"[^a-z]+", option.lower()) if t]
    return bool(toks) and all(t in LEVEL_VOCAB for t in toks)


def all_question_templates() -> list[str]:
    """Every question template string (for the MMSU exact-match check, templates are filled at item time)."""
    out = list(PATTERN_Q) + DUR_TOTAL_Q + DUR_BIN_Q + DUR_NOUL_Q + GAP_Q + LONGER_Q + PAUSE_WORD_Q + PAUSE_YN_Q
    out += PAUSE_COUNT_Q + PAUSE_DUR_Q + PAUSE_WHERE_Q + PAUSE_LONGER_Q + MEANING_Q
    for f in EXTREME_Q:
        out += [q for q, _ in EXTREME_Q[f]] + PAIR_NOUL[f] + PAIR_SCORE[f][0] + SAME_NOUL[f]
    return out


def stable_frac(key: str, salt: str = "") -> float:
    h = hashlib.md5(f"{salt}:{key}".encode()).hexdigest()
    return int(h[:12], 16) / 16 ** 12


class NoShuffle(random.Random):
    """rng for make_item on score items: keeps levels in natural order."""

    def shuffle(self, x, *a, **k):  # noqa: D401
        return None
