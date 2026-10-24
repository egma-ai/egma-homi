"""sb_speakers pure logic (numpy only). A speech segment is a stretch of speech in the whole clip with no silence
>= SEG_GAP_S (what a model could count without telling voices apart); a piece is one placed utterance.
"""
from __future__ import annotations

import hashlib
import math
import random

import numpy as np

SEG_GAP_S = 0.3
NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]

# timeline design constants (seconds)
DUR_LO, DUR_HI = 3.0, 24.0          # target total duration, drawn independently of N
PIECE_MIN, PIECE_MAX = 0.5, 6.0     # ordinary utterance length
SHORT_LO, SHORT_HI = 0.4, 1.2       # "says only a short phrase"
GAP_SHORT = (0.1, 0.24)             # speaker change without a segment break (< SEG_GAP_S)
GAP_LONG = (0.36, 1.0)              # pause (>= SEG_GAP_S): a new speech segment
SPLIT_PAUSE = (0.36, 0.9)           # pause inserted inside one speaker's turn
OVERLAP = (0.1, 0.3)


class NoShuffle(random.Random):
    """rng whose shuffle is a no-op (score items keep their natural level order)."""

    def shuffle(self, x, *a, **k):  # noqa: D401
        return None


def stable_frac(s: str, salt: str = "") -> float:
    h = hashlib.sha1(f"{salt}|{s}".encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def merge(iv, gap: float = 0.0):
    iv = sorted((float(a), float(b)) for a, b in iv if b > a)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def total(iv) -> float:
    return float(sum(b - a for a, b in iv))


def count_segments(intervals, gap: float = SEG_GAP_S, min_len: float = 0.1) -> int:
    """Speech segments of a clip: union of all speech intervals, silences < gap bridged, stretches < min_len dropped."""
    return sum(1 for a, b in merge(intervals, gap) if b - a >= min_len)


def pearson(x, y) -> float:
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


# Timeline design

def turn_sequence(counts: list[int], rng: random.Random) -> list[int] | None:
    """Order turns (counts[i] turns of speaker i) so that no speaker talks twice in a row (N >= 2).
    Returns the speaker index per turn, or None when impossible."""
    n = len(counts)
    if n == 1:
        return [0] * counts[0]
    P = sum(counts)
    if max(counts) > P - max(counts) + 1:
        return None
    for _ in range(30):
        left = list(counts)
        seq = []
        ok = True
        for pos in range(P):
            rem = P - pos
            cand = [i for i in range(n) if left[i] > 0 and (not seq or seq[-1] != i)]
            if not cand:
                ok = False
                break
            forced = [i for i in cand if left[i] > (rem - 1) - left[i] + 1]  # must take now or become infeasible
            pick = forced[0] if forced else rng.choices(cand, weights=[left[i] for i in cand])[0]
            seq.append(pick)
            left[pick] -= 1
        if ok:
            return seq
    return None


def waterfill(budget: float, n: int, rng: random.Random, lo: float = PIECE_MIN, hi: float = PIECE_MAX) -> list[float]:
    """n durations in [lo, hi] summing to `budget` as closely as the bounds allow (random Dirichlet shares)."""
    if n == 0:
        return []
    if budget <= n * lo:
        return [lo] * n
    if budget >= n * hi:
        return [hi] * n
    w = np.array([rng.gammavariate(2.0, 1.0) for _ in range(n)])
    d = np.full(n, lo, np.float64)
    free = np.ones(n, bool)
    rest = budget - n * lo
    for _ in range(20):
        if rest <= 1e-9 or not free.any():
            break
        share = w * free
        share = share / share.sum() * rest
        room = hi - d
        add = np.minimum(share, room)
        d += add
        rest -= add.sum()
        free = d < hi - 1e-9
    return [float(v) for v in d]


def design_timeline(rng: random.Random, N: int, T: float, short_phrase: bool, overlap: bool, seg_mode: str) -> dict:
    """Plan a clip of N distinct speakers with total duration ~T (drawn independently of N).

    Returns dict(seq=[speaker per piece], dur=[target speech s per piece], gaps=[gap after piece i (negative =
    overlap)], splits={piece: (frac, pause_s)}, lead, tail, short=[speaker indices that only say a short phrase],
    T_plan). seg_mode controls where segment breaks fall: 'free' (number of segments drawn independently of N),
    'natural' (most speaker changes are pauses)."""
    lead, tail = rng.uniform(0.1, 0.45), rng.uniform(0.1, 0.45)
    B = max(0.5, T - lead - tail)
    # speakers that only say a short phrase
    short = []
    if short_phrase and (N >= 3 or (N == 2 and B <= 12.0)):
        m = rng.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        short = rng.sample(range(N), min(m, N - 1))
    # turns per speaker (1-3; a lone speaker says 1-4 utterances)
    if N == 1:
        pmin = max(1, math.ceil(B / (PIECE_MAX + 1.0)))
        pmax = max(pmin, min(4, int((B + 0.36) / 1.1)))
        counts = [rng.randint(pmin, pmax)]
    else:
        counts = [1 if i in short else rng.choices([1, 2, 3], weights=[0.5, 0.32, 0.18])[0] for i in range(N)]
        normal = [i for i in range(N) if i not in short]

        def cap_speech(c):
            return sum(c[i] for i in normal) * PIECE_MAX + len(short) * SHORT_HI + (sum(c) - 1) * 0.6

        def min_speech(c):
            return sum(c[i] for i in normal) * PIECE_MIN + len(short) * SHORT_LO + (sum(c) - 1) * 0.12

        for _ in range(40):
            if cap_speech(counts) < B:
                up = [i for i in normal if counts[i] < 3]
                if not up:
                    break
                counts[rng.choice(up)] += 1
            elif min_speech(counts) > B:
                dn = [i for i in normal if counts[i] > 1]
                if not dn:
                    break
                counts[rng.choice(dn)] -= 1
            else:
                break
    seq = turn_sequence(counts, rng)
    while seq is None:  # e.g. counts (3, 1): too many turns for one speaker -> drop one
        j = int(np.argmax(counts))
        counts[j] -= 1
        seq = turn_sequence(counts, rng)
    P = len(seq)
    # where the segment breaks (pauses >= SEG_GAP_S) fall
    n_join = P - 1
    long_ok = [i for i in range(n_join)]
    if seg_mode == "natural":
        k_long = sum(1 for _ in range(n_join) if rng.random() < 0.75)
        s_target = 1 + k_long + (1 if rng.random() < 0.25 else 0)
    else:
        s_hi = max(1, min(10, int(T / 1.6)))
        s_target = rng.randint(1, s_hi)
        k_long = min(n_join, s_target - 1)
    long_set = set(rng.sample(long_ok, min(k_long, len(long_ok))))
    n_split = max(0, s_target - 1 - len(long_set))
    gaps = []
    for i in range(n_join):
        g = rng.uniform(*GAP_LONG) if i in long_set else rng.uniform(*GAP_SHORT)
        gaps.append(g)
    ov_join = None
    if overlap and N >= 2:
        cand = [i for i in range(n_join) if seq[i] != seq[i + 1] and seq[i] not in short and seq[i + 1] not in short]
        if cand:
            ov_join = rng.choice(cand)
            gaps[ov_join] = -rng.uniform(*OVERLAP)
    # durations
    split_pauses = [rng.uniform(*SPLIT_PAUSE) for _ in range(n_split)]
    budget = B - sum(g for g in gaps if g > 0) + sum(-g for g in gaps if g < 0) - sum(split_pauses)
    dur = [0.0] * P
    norm_idx = []
    for p, s in enumerate(seq):
        if s in short:
            dur[p] = rng.uniform(SHORT_LO, SHORT_HI)
        else:
            norm_idx.append(p)
    rest = budget - sum(dur)
    for p, d in zip(norm_idx, waterfill(rest, len(norm_idx), rng)):
        dur[p] = d
    if ov_join is not None:  # the overlap must be shorter than half of both pieces
        lim = 0.45 * min(dur[ov_join], dur[ov_join + 1])
        gaps[ov_join] = -min(-gaps[ov_join], max(0.1, lim))
    splits = {}
    cand = sorted([p for p in norm_idx if dur[p] >= 1.6], key=lambda p: -dur[p])
    rng.shuffle(cand)
    for p, ps in zip(cand, split_pauses):
        splits[p] = (rng.uniform(0.3, 0.7), ps)
    T_plan = lead + tail + sum(dur) + sum(gaps) + sum(v[1] for v in splits.values())
    return {"seq": seq, "dur": dur, "gaps": gaps, "splits": splits, "lead": lead, "tail": tail, "short": sorted(short),
            "T_plan": T_plan, "counts": counts, "overlap_join": ov_join, "s_target": s_target}


def draw_T(rng: random.Random, N: int) -> float:
    """Target duration U(3, 24) independent of N; redrawn only when physically impossible for N speakers
    (N x 0.45 s of speech + short gaps). For large N (whose shortest durations are impossible) a share q of clips is
    drawn just above the feasible minimum so that the mean duration does not grow with N."""
    tmin = N * 0.45 + (N - 1) * 0.12 + 0.3
    a = max(DUR_LO, tmin + 0.8)
    if a > DUR_LO:
        m_hi = (a + DUR_HI) / 2
        m_target = (DUR_LO + DUR_HI) / 2
        q = max(0.0, min(0.5, (m_hi - m_target) / max(1e-6, m_hi - (a + 1.5))))
        if rng.random() < q:
            return rng.uniform(a - 0.5, a + 3.0)
    for _ in range(100):
        T = rng.uniform(DUR_LO, DUR_HI)
        if T >= tmin:
            return T
    return max(tmin, DUR_LO)


# Audio helpers

def frame_db(x: np.ndarray, sr: int, hop: float = 0.02) -> np.ndarray:
    f = max(1, int(hop * sr))
    n = len(x) // f
    if n == 0:
        return np.zeros(0)
    return 20 * np.log10(np.sqrt((x[: n * f].reshape(n, f).astype(np.float64) ** 2).mean(1)) + 1e-9)


def trim_bounds(x: np.ndarray, sr: int, rel_db: float = 40.0, pad_s: float = 0.04):
    hop = sr // 100
    n = len(x) // hop
    if n < 3:
        return 0, len(x)
    r = 20 * np.log10(np.sqrt((x[: n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1)) + 1e-9)
    on = np.flatnonzero(r > r.max() - rel_db)
    pad = int(pad_s * sr)
    return max(0, int(on[0] * hop - pad)), min(len(x), int((on[-1] + 1) * hop + pad))


def energy_vad(x: np.ndarray, sr: int, hop: float = 0.02, bridge: float = 0.1):
    """Per-track energy VAD -> intervals (s). Active = > max(floor + 12 dB, p95 - 38 dB, -75 dB)."""
    db = frame_db(x, sr, hop)
    if len(db) == 0 or db.max() < -100:
        return []
    floor = np.percentile(db, 10)
    thr = max(floor + 12.0, np.percentile(db, 95) - 38.0, -75.0)
    act = db > thr
    iv, start = [], None
    for i, a in enumerate(act):
        if a and start is None:
            start = i
        elif not a and start is not None:
            iv.append((start * hop, i * hop))
            start = None
    if start is not None:
        iv.append((start * hop, len(act) * hop))
    iv = merge(iv, bridge)
    return [(round(s, 3), round(e, 3)) for s, e in iv if e - s >= 0.06]


def cut_points(x: np.ndarray, sr: int, hop: float = 0.01, rel_db: float = 28.0) -> list[float]:
    """Candidate cut times (s) at low-energy frames (word / phrase boundaries), always incl. 0 and the end."""
    f = int(hop * sr)
    n = len(x) // f
    if n < 5:
        return [0.0, len(x) / sr]
    db = 20 * np.log10(np.sqrt((x[: n * f].reshape(n, f).astype(np.float64) ** 2).mean(1)) + 1e-9)
    sm = np.convolve(db, np.ones(3) / 3, mode="same")
    thr = db.max() - rel_db
    pts = [0.0]
    for i in range(2, n - 2):
        if sm[i] < thr and sm[i] <= sm[i - 1] and sm[i] <= sm[i + 1]:
            t = (i + 0.5) * hop
            if t - pts[-1] >= 0.12:
                pts.append(t)
    pts.append(len(x) / sr)
    return pts


def best_window(pts: list[float], target: float, rng: random.Random, tol: float = 0.3):
    """(a, b) between cut points with b - a closest to target; random among those within tol of the best."""
    cands = []
    for i, a in enumerate(pts):
        for b in pts[i + 1:]:
            cands.append((abs((b - a) - target), a, b))
    if not cands:
        return None
    best = min(c[0] for c in cands)
    ok = [c for c in cands if c[0] <= best + min(tol, 0.15 * target)]
    _, a, b = rng.choice(ok)
    return a, b


def median_f0(x: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 420.0) -> float | None:
    """Median F0 (Hz) over voiced 40 ms frames (normalised autocorrelation peak > 0.55). None if < 10 voiced frames."""
    win, hop = int(0.04 * sr), int(0.01 * sr)
    n = (len(x) - win) // hop
    if n < 10:
        return None
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    fr = x[idx].astype(np.float64)
    fr = fr - fr.mean(1, keepdims=True)
    e = (fr ** 2).sum(1)
    edb = 10 * np.log10(e + 1e-12)
    keep = edb > np.percentile(edb, 90) - 25
    fr, e = fr[keep], e[keep]
    if len(fr) < 10:
        return None
    nfft = 1 << int(math.ceil(math.log2(2 * win)))
    F = np.fft.rfft(fr * np.hanning(win), nfft, axis=1)
    ac = np.fft.irfft(np.abs(F) ** 2, nfft, axis=1)[:, :win]
    ac = ac / (ac[:, :1] + 1e-12)
    lo, hi = int(sr / fmax), int(sr / fmin)
    seg = ac[:, lo:hi]
    k = seg.argmax(1)
    pk = seg[np.arange(len(seg)), k]
    v = pk > 0.55
    if v.sum() < 10:
        return None
    lag = (k[v] + lo).astype(np.float64)
    return float(np.median(sr / lag))


# Questions and labels

def count_window(n: int, lo: int, hi: int, rng: random.Random, width: int | None = None) -> list[int]:
    """Consecutive integers containing n (n at a uniformly random position), inside [lo, hi]."""
    w = width or rng.randint(3, 6)
    w = min(w, hi - lo + 1)
    first_lo, first_hi = max(lo, n - w + 1), min(n, hi - w + 1)
    a = rng.randint(first_lo, first_hi)
    return list(range(a, a + w))


def noul_count(n: int, kind: str, want_yes: bool, rng: random.Random, lo: int = 1, hi: int = 10):
    """Comparison value N for a yes/no question of `kind` (exactly / more / fewer / atleast) about count n with the
    requested truth value; near misses preferred (n - 1 and n + 1 most often). None if impossible."""
    cands = []
    for N in range(max(lo, n - 3), min(hi, n + 3) + 1):
        truth = {"more": n > N, "atleast": n >= N, "exactly": n == N, "fewer": n < N}[kind]
        if truth != want_yes:
            continue
        if kind in ("atleast", "fewer") and N < 2:
            continue
        cands.append(N)
    if not cands:
        return None
    w = [3.0 if abs(N - n) <= 1 else (1.5 if abs(N - n) == 2 else 0.7) for N in cands]
    return rng.choices(cands, weights=w)[0]


def noul_truth(n: int, kind: str, N: int) -> bool:
    return {"more": n > N, "atleast": n >= N, "exactly": n == N, "fewer": n < N}[kind]


def score_index(n: int) -> int:
    """one / two / three / four or more."""
    return min(n, 4) - 1


def mmsu_equal(questions, mmsu_questions) -> list[str]:
    """Item questions that equal (case / whitespace-insensitive) an MMSU question string."""
    norm = lambda s: " ".join(str(s).lower().split())  # noqa: E731
    m = {norm(q) for q in mmsu_questions}
    return sorted({q for q in questions if norm(q) in m})


def cap_ratio(counter: dict) -> float:
    v = [c for c in counter.values()]
    return max(v) / max(1, min(v)) if v else 0.0
