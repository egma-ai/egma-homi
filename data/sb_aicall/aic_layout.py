"""Seeded two-party timelines for sb_aicall: scenario A (AI agent + live human) and E (two humans). A turn is
{"role": "agent" | "other", "dur", "words": speech intervals, "incl": speech + noise intervals, "src", "tts"},
with intervals relative to the turn start."""
from __future__ import annotations

import random

import labels as L  # data/sb_dialog/labels.py (merge, runs, gaps, talked_over, ...)

ROLES = ("agent", "other")


def runs_of(turn):
    return L.runs(turn["words"])


def last_run(turn, cut=None):
    r = runs_of(turn)
    if cut is not None:
        r = [(a, min(b, cut)) for a, b in r if a < cut]
    return r[-1] if r else None


def first_run(turn):
    r = runs_of(turn)
    return r[0] if r else None


def trim_turn(turn, keep_runs: int):
    """Keep only the first `keep_runs` speech runs of a turn (cut 0.12 s after the last kept run)."""
    r = runs_of(turn)
    if len(r) <= keep_runs:
        return turn
    end = min(r[keep_runs - 1][1] + 0.12, (r[keep_runs - 1][1] + r[keep_runs][0]) / 2)
    t = dict(turn)
    t["dur"] = end
    t["words"] = [(a, min(b, end)) for a, b in turn["words"] if a < end]
    t["incl"] = [(a, min(b, end)) for a, b in turn["incl"] if a < end]
    t["trimmed"] = True
    return t


def latency_profile(rng: random.Random) -> str:
    u = rng.random()
    return "fast" if u < 0.35 else "one_slow" if u < 0.60 else "mixed" if u < 0.85 else "slow"


def draw_latency(profile: str, rng: random.Random) -> float:
    """AI response latency after a caller turn. Mixed profile: 55% U(0.3,1.5), 25% U(1.5,3), 20% U(3,9)."""
    if profile == "fast" or profile == "one_slow":
        return rng.uniform(0.3, 1.5)
    if profile == "slow":
        return rng.uniform(1.5, 3.0) if rng.random() < 0.6 else rng.uniform(3.0, 7.0)
    u = rng.random()
    return rng.uniform(0.3, 1.5) if u < 0.55 else rng.uniform(1.5, 3.0) if u < 0.8 else rng.uniform(3.0, 9.0)


def draw_recipe(rng: random.Random, kind: str) -> dict:
    u = rng.random()
    return {
        "kind": kind,  # "A" (TTS agent) or "E" (human agent)
        "latency": latency_profile(rng),
        "dead": rng.random() < 0.30, "n_dead": 1 if rng.random() < 0.7 else 2,
        "barge": rng.random() < 0.30,
        "aiover": rng.random() < 0.25,
        "backch": rng.random() < 0.30,
        "share": "other_low" if u < 0.25 else "agent_low" if u < 0.55 else "normal",
        "target_len": (rng.uniform(20, 45) if (v := rng.random()) < 0.3 else rng.uniform(45, 100) if v < 0.82
                       else rng.uniform(100, 170)),
        "start0": rng.random() < 0.65,
        "gated": rng.random() < 0.70,
        "other_gain_db": rng.uniform(-6.0, 4.0),
        "peak": rng.uniform(0.4, 0.95),
    }


def select_turns(turns, rc, rng):
    """Window of consecutive turns (from turn 0 or a later turn) of roughly the target length."""
    n = len(turns)
    s0 = 0
    if not rc["start0"]:  # start later, preferably on an other-side turn (so "who spoke first" varies)
        cands = [i for i in range(1, max(2, n // 2)) if turns[i]["role"] == "other"]
        s0 = rng.choice(cands) if cands else rng.randint(1, max(1, n // 3))
    sel, acc = [], 0.0
    for t in turns[s0:]:
        sel.append(t)
        acc += t["dur"] + 1.0
        if acc >= rc["target_len"]:
            break
    if acc < 24 and s0 > 0:
        s0, sel, acc = 0, [], 0.0
        for t in turns:
            sel.append(t)
            acc += t["dur"] + 1.0
            if acc >= rc["target_len"]:
                break
    return s0, sel


def apply_share(sel, rc, rng):
    """Vary the talk share: 'other_low' trims other-side turns to 1-2 runs and drops some of them; 'agent_low'
    trims agent turns to their first run(s)."""
    out = []
    for i, t in enumerate(sel):
        if rc["share"] == "other_low" and t["role"] == "other":
            if 0 < i < len(sel) - 1 and rng.random() < 0.3:
                continue
            t = trim_turn(t, 1 if rng.random() < 0.6 else 2)
        elif rc["share"] == "agent_low" and t["role"] == "agent":
            t = trim_turn(t, 1 if rng.random() < 0.8 else 2)
        out.append(t)
    return out


def build_timeline(sel, rc, rng, backchannels=()):
    """Place turns with the recipe's controls -> {"T", "placed", "bc", "speech", "incl", "events", "recipe"} or None."""
    n = len(sel)
    if n < 3:
        return None
    trans = list(range(1, n))
    o2a = [i for i in trans if sel[i - 1]["role"] == "other" and sel[i]["role"] == "agent"]
    a2o = [i for i in trans if sel[i - 1]["role"] == "agent" and sel[i]["role"] == "other"]
    ev = {}
    events = []
    if rc["barge"]:
        cands = []
        for i in a2o:
            lr, fr = last_run(sel[i - 1]), first_run(sel[i])
            if lr and fr and lr[1] - lr[0] >= 1.2 and fr[1] - fr[0] >= 1.0:
                cands.append(i)
        k = 1 if rng.random() < 0.75 else 2
        for i in sorted(rng.sample(cands, min(k, len(cands)))):
            lr = last_run(sel[i - 1])
            room = (lr[1] - lr[0]) - 0.45  # keep >= 0.45 s of agent speech before the onset
            keep = rc["kind"] == "E" or rng.random() < 0.5
            if keep and room >= 2.1:  # agent keeps talking >= 2 s over the caller
                ov, stop = rng.uniform(2.0, min(3.0, room)), None
            else:  # onset U(0.3, 3) s before the run's end; the AI agent stops U(0.2, 1.0) s later
                ov = min(rng.uniform(0.3, 3.0), room)
                stop = rng.uniform(0.2, 1.0) if rc["kind"] == "A" else None
            if ov < 0.3:
                continue
            ev[i] = {"type": "barge", "ov": ov, "stop": stop}
    if rc["aiover"]:
        cands = []
        for i in o2a:
            if i in ev:
                continue
            lr, fr = last_run(sel[i - 1]), first_run(sel[i])
            if lr and fr and lr[1] - lr[0] >= 1.0 and fr[1] - fr[0] >= 0.6:
                cands.append(i)
        if cands:
            i = rng.choice(cands)
            lr = last_run(sel[i - 1])
            hi = min(2.0, (lr[1] - lr[0]) - 0.5)
            if hi >= 0.5:
                ev[i] = {"type": "aiover", "ov": rng.uniform(0.5, hi)}
    slow_i = rng.choice(o2a) if (rc["latency"] == "one_slow" and o2a) else None
    gaps = {}
    for i in trans:
        if i in o2a:
            gaps[i] = draw_latency(rc["latency"], rng)
            if i == slow_i:
                gaps[i] = rng.uniform(3.0, 9.0) if rng.random() < 0.65 else rng.uniform(1.5, 3.0)
        elif i in a2o:
            gaps[i] = rng.uniform(0.2, 1.1)
        else:
            gaps[i] = rng.uniform(0.3, 1.0)
    if rc["dead"]:
        cands = [i for i in trans if i not in ev]
        for i in rng.sample(cands, min(rc["n_dead"], len(cands))):
            gaps[i] = rng.uniform(4.0, 14.0)
            ev[i] = {"type": "dead", "gap": gaps[i]}
    lead = rng.uniform(0.3, 1.2)
    placed = []
    floor = lead
    last_end = {}
    for i, t in enumerate(sel):
        e = ev.get(i)
        if i == 0:
            st = lead
        elif e and e["type"] in ("barge", "aiover"):
            p = placed[-1]
            lr = last_run(p["turn"], p["cut"])
            st = p["st"] + lr[1] - e["ov"]
        else:
            st = floor + gaps[i]
        st = max(st, last_end.get(t["role"], -1e9) + 0.3)
        if e and e["type"] == "barge" and e["stop"] is not None:
            p = placed[-1]
            cut_rel = st + e["stop"] - p["st"]
            lr = last_run(p["turn"])
            if cut_rel < lr[1] - 0.05:
                p["cut"] = cut_rel
        if e:
            events.append({"turn": i, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()},
                           "at": round(st, 3)})
        placed.append({"turn": t, "st": st, "cut": None})
        last_end = {}  # recomputed from scratch: a cut may shorten an earlier turn
        for q in placed:
            q_end = q["st"] + (q["cut"] + 0.03 if q["cut"] is not None else q["turn"]["dur"])
            last_end[q["turn"]["role"]] = max(last_end.get(q["turn"]["role"], -1e9), q_end)
        floor = max(last_end.values())
    sp = {r: [] for r in ROLES}
    inc = {r: [] for r in ROLES}
    for q in placed:
        c = q["cut"]
        for key, dst in (("words", sp), ("incl", inc)):
            for a, b in q["turn"][key]:
                if c is not None:
                    if a >= c:
                        continue
                    b = min(b, c + 0.03)
                dst[q["turn"]["role"]].append((q["st"] + a, q["st"] + b))
    # backchannels: short other-side sounds inside long agent runs, away from other-side speech
    bcs = []
    if rc["backch"] and backchannels:
        a_runs = [r for r in L.runs(sp["agent"]) if r[1] - r[0] >= 2.2]
        rng.shuffle(a_runs)
        want = rng.randint(1, 3)
        for r0, r1 in a_runs:
            if len(bcs) >= want:
                break
            b = backchannels[rng.randrange(len(backchannels))]
            w0, w1 = b["word"]
            span = r1 - r0 - 1.2 - (w1 - w0)
            if span <= 0:
                continue
            at = r0 + 0.6 + rng.uniform(0, span)  # absolute time of the backchannel word start
            ws, we = at, at + (w1 - w0)
            if any(not (we + 0.6 <= s or e <= ws - 0.6) for s, e in sp["other"]):
                continue
            st = at - w0
            bcs.append({"bc": b, "st": st})
            sp["other"].append((ws, we))
            inc["other"].append((ws, we))
            events.append({"type": "backchannel", "at": round(ws, 3), "dur": round(we - ws, 3)})
    T = floor + rng.uniform(0.3, 1.0)
    sp = {r: L.merge(v) for r, v in sp.items()}
    inc = {r: L.merge(v) for r, v in inc.items()}
    return {"T": round(T, 3), "placed": placed, "bc": bcs, "speech": sp, "incl": inc, "events": events,
            "recipe": rc}


def make_layout(turns, rc, seed: str, backchannels=(), max_T: float = 179.0, min_T: float = 20.0):
    """select -> share -> timeline; drops trailing turns until T <= max_T. Deterministic given `seed`."""
    rng0 = random.Random(seed)
    s0, sel = select_turns(turns, rc, rng0)
    sel = apply_share(sel, rc, rng0)
    while len(sel) >= 3:
        lay = build_timeline(sel, rc, random.Random(seed + ":t"), backchannels)
        if lay is None:
            return None
        if lay["T"] <= max_T:
            if lay["T"] < min_T:
                return None
            lay["start_turn"] = s0
            lay["n_turns"] = len(sel)
            return lay
        sel = sel[:-1]
    return None
