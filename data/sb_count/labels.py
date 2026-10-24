"""Label logic for sb_count (numpy only). A run is (start_s, end_s, speaker); a turn is one uninterrupted stretch by
one speaker (a pause inside one speaker's talk does not start a new turn). Speakers are named in order of first speech.
"""
from __future__ import annotations

import hashlib
import random

import numpy as np

NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
             "twelve", "thirteen", "fourteen", "fifteen"]
ORDINALS = ["zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]


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


def clip_iv(iv, t0: float, t1: float):
    """Intervals clipped to [t0, t1] and shifted so that t0 -> 0."""
    out = []
    for a, b in iv:
        a2, b2 = max(a, t0), min(b, t1)
        if b2 > a2:
            out.append((a2 - t0, b2 - t0))
    return out


def total(iv) -> float:
    return float(sum(b - a for a, b in iv))


def turn_sequence(runs):
    """runs [(s, e, spk)] -> list of turns [(spk, s, e)]: runs sorted by onset, consecutive runs of the same speaker
    merged (a pause inside one speaker's talk is not a new turn)."""
    out = []
    for s, e, k in sorted(runs, key=lambda r: (r[0], r[1])):
        if out and out[-1][0] == k:
            out[-1] = (k, out[-1][1], max(out[-1][2], e))
        else:
            out.append((k, s, e))
    return out


def count_turns(runs) -> int:
    return len(turn_sequence(runs))


def contained(runs, frac: float = 0.8) -> bool:
    """True if some run of one speaker lies (>= frac of its length) inside another speaker's run
    (a backchannel / full overlap -> turn count ambiguous)."""
    rs = sorted(runs)
    for i, (a, b, k) in enumerate(rs):
        for c, d, j in rs:
            if j == k:
                continue
            ov = min(b, d) - max(a, c)
            if ov > 0 and ov >= frac * (b - a):
                return True
    return False


def max_overlap(runs) -> float:
    m = 0.0
    for a, b, k in runs:
        for c, d, j in runs:
            if j != k:
                m = max(m, min(b, d) - max(a, c))
    return m


def speaker_order(runs) -> list:
    """Speakers in the order of their first onset."""
    seen, out = set(), []
    for s, e, k in sorted(runs, key=lambda r: (r[0], r[1])):
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def talk_time(runs) -> dict:
    by = {}
    for s, e, k in runs:
        by.setdefault(k, []).append((s, e))
    return {k: total(merge(v)) for k, v in by.items()}


def turns_per_speaker(runs) -> dict:
    c = {}
    for k, s, e in turn_sequence(runs):
        c[k] = c.get(k, 0) + 1
    return c


def argmax_margin(vals: dict, rel: float, abs_: float):
    """Key with the largest value if it beats the runner-up by >= abs_ and by a factor >= rel, else None."""
    if len(vals) < 2:
        return None
    srt = sorted(vals.items(), key=lambda kv: -kv[1])
    (k1, v1), (_, v2) = srt[0], srt[1]
    if v1 - v2 >= abs_ and v1 >= rel * v2:
        return k1
    return None


def count_window(n: int, lo: int, hi: int, rng: random.Random, width: int | None = None) -> list[int]:
    """Consecutive integers containing n (n at a uniformly random position), inside [lo, hi]."""
    w = width or rng.randint(4, 6)
    w = min(w, hi - lo + 1)
    first_lo, first_hi = max(lo, n - w + 1), min(n, hi - w + 1)
    a = rng.randint(first_lo, first_hi)
    return list(range(a, a + w))


def noul_count(n: int, want_yes: bool, rng: random.Random, lo: int = 1, hi: int = 14):
    """A yes/no comparison question about a count n with the requested truth value.
    -> (kind, N) with kind in {more, atleast, exactly, fewer}, or None when impossible."""
    kinds = ["more", "atleast", "exactly", "fewer"]
    rng.shuffle(kinds)
    for kind in kinds:
        cands = []
        for N in range(max(lo, n - 3), min(hi, n + 3) + 1):
            truth = {"more": n > N, "atleast": n >= N, "exactly": n == N, "fewer": n < N}[kind]
            if truth == want_yes:
                if kind == "more" and N < 1:
                    continue
                if kind in ("atleast", "fewer") and N < 2:
                    continue
                cands.append(N)
        if kind == "exactly" and not want_yes:
            cands = [N for N in cands if abs(N - n) <= 2]
        if cands:
            return kind, rng.choice(cands)
    return None


def bin_index(n: int, bins) -> int:
    """bins: [(label, lo, hi)] (inclusive) -> index of the bin containing n, or -1."""
    for i, (_, a, b) in enumerate(bins):
        if a <= n <= b:
            return i
    return -1


def pair_label(a: int, b: int) -> str:
    a, b = sorted((a, b))
    return f"{a} and {b}"


def components(groups: dict) -> list[set]:
    """groups: {group id: set(speakers)} -> connected components of group ids (groups sharing a speaker)."""
    parent = {g: g for g in groups}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner = {}
    for g, spk in groups.items():
        for s in spk:
            if s in owner:
                parent[find(g)] = find(owner[s])
            else:
                owner[s] = g
    comp = {}
    for g in groups:
        comp.setdefault(find(g), set()).add(g)
    return sorted(comp.values(), key=lambda c: sorted(c)[0])


def cal_groups(groups: dict, frac: float, rng: random.Random) -> set:
    """Whole speaker-closed components of groups for CAL until >= frac of groups (small components only)."""
    comps = components(groups)
    rng.shuffle(comps)
    cal, n = set(), len(groups)
    for c in comps:
        if len(cal) >= frac * n:
            break
        if len(c) <= max(2, 0.5 * frac * n) + 4:
            cal |= c
    return cal


def trim_bounds(x: np.ndarray, sr: int, rel_db: float = 40.0, pad_s: float = 0.04):
    """(a, b) sample bounds of x without leading / trailing near-silence (10 ms frames < peak frame - rel_db)."""
    hop = sr // 100
    n = len(x) // hop
    if n < 3:
        return 0, len(x)
    r = 20 * np.log10(np.sqrt((x[: n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1)) + 1e-9)
    on = np.flatnonzero(r > r.max() - rel_db)
    pad = int(pad_s * sr)
    return max(0, int(on[0] * hop - pad)), min(len(x), int((on[-1] + 1) * hop + pad))


def frame_db(x: np.ndarray, sr: int, hop: float = 0.02) -> np.ndarray:
    f = max(1, int(hop * sr))
    n = len(x) // f
    if n == 0:
        return np.zeros(0)
    return 20 * np.log10(np.sqrt((x[: n * f].reshape(n, f).astype(np.float64) ** 2).mean(1)) + 1e-9)


def energy_vad(x: np.ndarray, sr: int, hop: float = 0.02, bridge: float = 0.1):
    """Per-track energy VAD (sb_dialog's rule) -> intervals (s). Active = > max(floor + 12 dB, p95 - 38 dB, -75 dB)."""
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
    return [(round(s, 3), round(e, 3)) for s, e in iv if e - s >= 0.08]


def cap_balance(labels: list, cap_ratio: float = 2.0) -> list[int]:
    """Indices to keep so that every label value appears at most cap_ratio x the rarest value's count
    (keeps the first occurrences, input order assumed random)."""
    from collections import Counter

    c = Counter(labels)
    if not c:
        return []
    cap = int(cap_ratio * min(c.values()))
    seen, keep = Counter(), []
    for i, v in enumerate(labels):
        if seen[v] < cap:
            seen[v] += 1
            keep.append(i)
    return keep
