"""HarperValleyBank conversations: parsing, natural and re-laid layouts, rendering. Both 8 kHz channel wavs start at
the call start; a segment sits at `start_ms` in its channel (not `offset_ms`). A layout is a list of pieces
(role, src_start_s, src_end_s, dst_start_s) plus clip length and per-role intervals; `render_layout` mixes it."""
from __future__ import annotations

import math
import random
import re

import numpy as np

ROLES = ("agent", "caller")
NOISE_ONLY = re.compile(r"^(\s*\[[^\]]*\]\s*)*$")
PAD = 0.06  # seconds of channel audio kept around each re-laid turn


def parse_transcript(segs: list[dict]) -> list[dict]:
    """HVB transcript json -> sorted segments {role, s, e, words[(s,e)], speech(bool), text}."""
    out = []
    for g in segs:
        role = g.get("speaker_role")
        if role not in ROLES:
            continue
        s = g["start_ms"] / 1000.0
        e = s + g["duration_ms"] / 1000.0
        words = [(s + o / 1000.0, s + (o + d) / 1000.0)
                 for o, d in zip(g.get("word_offsets_ms") or [], g.get("word_durations_ms") or []) if d > 0]
        text = (g.get("human_transcript") or g.get("transcript") or "").strip()
        speech = bool(text) and not NOISE_ONLY.match(text)
        if speech and not words:
            words = [(s, e)]
        out.append({"role": role, "s": s, "e": e, "words": words if speech else [], "speech": speech, "text": text})
    out.sort(key=lambda g: (g["s"], g["e"]))
    return out


def intervals(segs, t0: float = 0.0, t1: float = 1e9):
    """Per-role speech (word) and inclusive (segment) intervals clipped to [t0, t1], shifted to t0."""
    from labels import clip_to
    sp = {r: [] for r in ROLES}
    inc = {r: [] for r in ROLES}
    for g in segs:
        sp[g["role"]] += g["words"]
        inc[g["role"]].append((g["s"], g["e"]))
    return ({r: clip_to(v, t0, t1) for r, v in sp.items()}, {r: clip_to(v, t0, t1) for r, v in inc.items()})


def natural_windows(segs, wav_len: float, rng: random.Random, max_len: float = 180.0, min_len: float = 20.0):
    """Whole call (lead <= 1.5 s, tail <= 1 s); calls > max_len are cut into windows at silent gaps."""
    sp = [g for g in segs if g["speech"]]
    if not sp:
        return []
    t0 = max(0.0, sp[0]["s"] - rng.uniform(0.4, 1.5))
    t_end = min(wav_len, max(g["e"] for g in segs) + rng.uniform(0.3, 1.0))
    if t_end - t0 <= max_len:
        return [(t0, t_end)] if t_end - t0 >= min_len else []
    # cut at gaps where nobody is inside a segment
    busy = sorted((g["s"], g["e"]) for g in segs)
    cuts, cur = [], -1.0
    for s, e in busy:
        if s - cur > 0.6 and cur > 0:
            cuts.append((cur + s) / 2)
        cur = max(cur, e)
    wins, a = [], t0
    while t_end - a > min_len:
        target = a + rng.uniform(45, 150)
        cands = [c for c in cuts if a + min_len <= c <= a + max_len]
        if not cands or t_end - a <= max_len:
            b = t_end if t_end - a <= max_len else (min(cands, key=lambda c: abs(c - target)) if cands else None)
        else:
            b = min(cands, key=lambda c: abs(c - target))
        if b is None:
            break
        wins.append((a, b))
        a = b
    return [w for w in wins if min_len <= w[1] - w[0] <= max_len]


def make_turns(segs):
    """Consecutive same-role segments -> turns {role, s, e, words(rel), incl(rel)}; noise-only segments are
    kept inside a turn of the same role but never start one."""
    turns = []
    for g in segs:
        if turns and turns[-1]["role"] == g["role"]:
            t = turns[-1]
        elif g["speech"]:
            t = {"role": g["role"], "s": g["s"], "e": g["e"], "segs": []}
            turns.append(t)
        else:
            continue
        t["segs"].append(g)
        t["e"] = max(t["e"], g["e"])
    for t in turns:
        t["words"] = [(a - t["s"], b - t["s"]) for g in t["segs"] for a, b in g["words"]]
        t["incl"] = [(g["s"] - t["s"], g["e"] - t["s"]) for g in t["segs"]]
        t["dur"] = t["e"] - t["s"]
        del t["segs"]
    return turns


def _last_run(t):
    from labels import runs
    r = runs(t["words"])
    return r[-1] if r else None


def _first_run(t):
    from labels import runs
    r = runs(t["words"])
    return r[0] if r else None


def _place(turns, gaps, overs, lead):
    """gaps[i] = silence before turn i (i >= 1); overs[i] = seconds turn i starts before the previous turn's
    last speech ends (talk-over) or None. Returns start times."""
    starts = [lead]
    last_end = {turns[0]["role"]: lead + turns[0]["dur"]}
    floor = lead + turns[0]["dur"]
    for i in range(1, len(turns)):
        t, p = turns[i], turns[i - 1]
        if overs.get(i) is not None:
            st = starts[i - 1] + _last_run(p)[1] - overs[i]
        else:
            st = floor + gaps[i]
        st = max(st, last_end.get(t["role"], -1e9) + 0.25)
        starts.append(st)
        last_end[t["role"]] = st + t["dur"]
        floor = max(floor, st + t["dur"])
    return starts


def relaid_layout(turns, rng: random.Random, recipe: dict | None = None):
    """Build one re-laid variant of a conversation. Returns layout dict or None.
    recipe keys (drawn when missing): start_turn, target_len, style, agent_over, caller_over, long_sil, slow."""
    from labels import runs
    n = len(turns)
    if n < 4:
        return None
    r = dict(recipe or {})
    r.setdefault("start_turn", 0 if rng.random() < 0.55 else rng.randint(1, max(1, n // 2)))
    u = rng.random()
    r.setdefault("target_len", rng.uniform(20, 45) if u < 0.35 else rng.uniform(45, 95) if u < 0.82 else rng.uniform(95, 160))
    r.setdefault("style", "tight" if rng.random() < 0.5 else "natural")
    r.setdefault("agent_over", rng.random() < 0.5)
    r.setdefault("caller_over", rng.random() < 0.3)
    r.setdefault("long_sil", rng.random() < 0.35)
    r.setdefault("slow", rng.random() < 0.3)
    r.setdefault("many_over", rng.random() < 0.15)  # heavy cross-talk: 3-5 talk-overs in both directions
    # window of turns
    s0 = min(r["start_turn"], n - 2)
    sel, acc = [], 0.0
    for t in turns[s0:]:
        sel.append(t)
        acc += t["dur"] + 0.7
        if acc >= r["target_len"]:
            break
    if acc < 22 and s0 > 0:
        sel, acc = [], 0.0
        for t in turns:
            sel.append(t)
            acc += t["dur"] + 0.7
            if acc >= r["target_len"]:
                break
    if len(sel) < 3 or acc < 20:
        return None
    m = len(sel)
    gaps = {}
    for i in range(1, m):
        g0 = sel[i]["s"] - sel[i - 1]["e"]
        gaps[i] = rng.uniform(0.15, 0.9) if r["style"] == "tight" or not (0.12 <= g0 <= 1.6) else g0
    lead = rng.uniform(0.3, 1.2)
    tent = _place(sel, gaps, {}, lead)
    T0 = tent[-1] + sel[-1]["dur"]
    used, overs, events = set(), {}, []

    def pick(cands, third):
        if not cands:
            return None
        inthird = [i for i in cands if min(2, int(3 * tent[i] / T0)) == third]
        pool = inthird or cands
        return pool[rng.randrange(len(pool))]

    for who, flag in (("agent", r["agent_over"]), ("caller", r["caller_over"])):
        if not flag:
            continue
        cands = []
        for i in range(1, m):
            if i in used or sel[i]["role"] != who or sel[i - 1]["role"] == who:
                continue
            lr, fr = _last_run(sel[i - 1]), _first_run(sel[i])
            if lr and fr and lr[1] - lr[0] >= 1.0 and fr[1] - fr[0] >= 0.6:
                cands.append(i)
        i = pick(cands, rng.randrange(3))
        if i is None:
            continue
        lr, fr = _last_run(sel[i - 1]), _first_run(sel[i])
        hi = min(2.0, (lr[1] - lr[0]) - 0.5)
        if hi < 0.5:
            continue
        overs[i] = rng.uniform(0.5, hi)
        used.add(i)
        events.append({"type": f"{who}_over", "turn": i, "ov": round(overs[i], 3)})
    if r["many_over"]:
        cands = []
        for i in range(1, m):
            if i in used or sel[i]["role"] == sel[i - 1]["role"]:
                continue
            lr, fr = _last_run(sel[i - 1]), _first_run(sel[i])
            if lr and fr and lr[1] - lr[0] >= 1.1 and fr[1] - fr[0] >= 0.7:
                cands.append(i)
        rng.shuffle(cands)
        for i in sorted(cands[: rng.randint(4, 6)]):
            lr, fr = _last_run(sel[i - 1]), _first_run(sel[i])
            overs[i] = rng.uniform(0.6, max(0.61, min(2.0, (lr[1] - lr[0]) - 0.45, (fr[1] - fr[0]) + 0.5)))
            used.add(i)
            events.append({"type": f"{sel[i]['role']}_over", "turn": i, "ov": round(overs[i], 3)})
    if r["long_sil"]:
        cands = [i for i in range(1, m) if i not in used]
        i = pick(cands, rng.randrange(3))
        if i is not None:
            gaps[i] = rng.uniform(3.7, 7.0)
            used.add(i)
            events.append({"type": "long_sil", "turn": i, "gap": round(gaps[i], 3)})
    if r["slow"]:
        cands = [i for i in range(1, m) if i not in used and sel[i]["role"] == "agent" and sel[i - 1]["role"] == "caller"]
        if cands:
            i = cands[rng.randrange(len(cands))]
            gaps[i] = rng.uniform(2.7, 5.0)
            used.add(i)
            events.append({"type": "slow", "turn": i, "gap": round(gaps[i], 3)})
    starts = _place(sel, gaps, overs, lead)
    T = max(st + t["dur"] for st, t in zip(starts, sel)) + rng.uniform(0.3, 1.0)
    if not (20.0 <= T <= 180.0):
        return None
    sp = {k: [] for k in ROLES}
    inc = {k: [] for k in ROLES}
    pieces = []
    for st, t in zip(starts, sel):
        sp[t["role"]] += [(st + a, st + b) for a, b in t["words"]]
        inc[t["role"]] += [(st + a, st + b) for a, b in t["incl"]]
        pieces.append((t["role"], t["s"] - PAD, t["e"] + PAD, st - PAD))
    return {"T": round(T, 3), "pieces": pieces, "speech": sp, "incl": inc, "recipe": r, "events": events,
            "bed": True}


def natural_layout(segs, t0: float, t1: float):
    sp, inc = intervals(segs, t0, t1)
    return {"T": round(t1 - t0, 3), "pieces": [(r, t0, t1, 0.0) for r in ROLES], "speech": sp, "incl": inc,
            "recipe": {"natural": True}, "events": [], "bed": False}


# Rendering (numpy / scipy)

def noise_bed(ch: np.ndarray, sr: int, segs_role, min_len: float = 0.5) -> np.ndarray:
    """The channel's own non-speech audio (outside every segment +- 0.3 s), concatenated."""
    mask = np.ones(len(ch), bool)
    for g in segs_role:
        a, b = max(0, int((g["s"] - 0.3) * sr)), min(len(ch), int((g["e"] + 0.3) * sr))
        mask[a:b] = False
    bed = ch[mask]
    if len(bed) < min_len * sr:
        return np.zeros(int(sr), np.float32)
    return bed.astype(np.float32)


def render_tracks(layout: dict, chans: dict, beds: dict, sr: int, rng: random.Random) -> dict:
    """-> {role: track} at the channel rate, length T."""
    n = int(round(layout["T"] * sr))
    tracks = {}
    for role in ROLES:
        if layout["bed"]:
            bed = beds[role]
            off = rng.randrange(len(bed))
            reps = math.ceil((n + off) / len(bed)) + 1
            tr = np.tile(bed, reps)[off:off + n].copy()
        else:
            tr = np.zeros(n, np.float32)
        tracks[role] = tr
    fade = max(1, int(0.01 * sr))
    for role, a, b, dst in layout["pieces"]:
        ch = chans[role]
        ia, ib, id_ = int(round(a * sr)), int(round(b * sr)), int(round(dst * sr))
        if id_ < 0:  # piece starts before the clip: drop its head
            ia -= id_
            id_ = 0
        ia = max(ia, 0)
        ib = min(ib, len(ch), ia + (n - id_))
        if ib <= ia:
            continue
        seg = ch[ia:ib].astype(np.float32)
        L = len(seg)
        w = np.ones(L, np.float32)
        if layout["bed"]:
            f = min(fade, L // 2)
            if f > 0:
                ramp = np.linspace(0, 1, f, dtype=np.float32)
                w[:f], w[L - f:] = ramp, ramp[::-1]
        tr = tracks[role]
        tr[id_:id_ + L] = tr[id_:id_ + L] * (1 - w) + seg * w
    return tracks


def frame_db(x: np.ndarray, sr: int, hop: float = 0.02) -> np.ndarray:
    f = max(1, int(hop * sr))
    n = len(x) // f
    if n == 0:
        return np.zeros(0)
    return 20 * np.log10(np.sqrt((x[: n * f].reshape(n, f).astype(np.float64) ** 2).mean(1)) + 1e-9)


def energy_vad(x: np.ndarray, sr: int, hop: float = 0.02):
    """Per-track energy VAD -> intervals (s). Active = > max(floor + 12 dB, p95 - 38 dB)."""
    from labels import merge
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
    iv = merge(iv, 0.1)
    return [(round(s, 3), round(e, 3)) for s, e in iv if e - s >= 0.08]
