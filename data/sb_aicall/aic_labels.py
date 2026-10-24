"""Label rules for sb_aicall: speech intervals (s) of the AI side ("agent") and the other side ("other") -> label
candidates {family: [(answer, param)]}, with margins. `param` is the threshold N, the talk-share binning id, or None;
ambiguous cases are not returned."""
from __future__ import annotations

import labels as L  # data/sb_dialog/labels.py

WAIT_NS = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8]
SIL_NS = [2, 3, 4, 5, 6, 8, 10]
WAIT_BINS = [(1.0, "under 1 second"), (3.0, "1 to 3 seconds"), (6.0, "3 to 6 seconds"), (None, "over 6 seconds")]
SIL_BINS = [(2.0, "under 2 seconds"), (5.0, "2 to 5 seconds"), (10.0, "5 to 10 seconds"), (None, "over 10 seconds")]
BIN_MARGIN = 0.3
SHARE_MARGIN = 0.03
SHARE_BINNINGS = {  # id -> (cut points, labels low -> high)
    "fifths": ([0.2, 0.4], ["less than 20%", "20% to 40%", "more than 40%"]),
    "quarters": ([0.25, 0.5], ["less than 25%", "25% to 50%", "more than 50%"]),
    "thirds": ([1 / 3, 0.5], ["less than a third", "between a third and a half", "more than half"]),
}


def margin(n: float) -> float:
    return max(0.4, 0.12 * n)


def bin_value(v: float, bins, m: float = BIN_MARGIN):
    """bins [(upper edge or None, label)] -> label, or None when v is within m of an edge."""
    lo = 0.0
    for hi, lab in bins:
        if hi is None or v < hi:
            if (lo > 0 and v < lo + m) or (hi is not None and v > hi - m):
                return None
            return lab
        lo = hi
    return None


# ---------------------------------------------------------------- waits (caller finished -> AI reply)

def wait_gaps(a_iv, o_iv, T: float):
    """-> (W, C): W = silences from the end of an other-side run to the next agent run; C = silences from an
    other-side run end to that side's next run (it spoke again before the AI answered). Trailing silence is ignored."""
    ra, ro = L.runs(a_iv), L.runs(o_iv)
    W, C = [], []
    for s, e in ro:
        if any(a0 < e < a1 for a0, a1 in ra) or any(a0 <= e and a1 >= e for a0, a1 in ra):
            continue
        na = min((a0 for a0, _ in ra if a0 >= e), default=None)
        no = min((o0 for o0, _ in ro if o0 > e), default=None)
        if na is None and no is None:
            continue
        if na is not None and (no is None or na <= no):
            W.append(na - e)
        else:
            C.append(no - e)
    return W, C


def waited_candidates(a_iv, o_iv, T: float):
    W, C = wait_gaps(a_iv, o_iv, T)
    if not W:
        return [], None
    w, c = max(W), max(C, default=0.0)
    out = []
    for n in WAIT_NS:
        m = margin(n)
        if w >= n + m:
            out.append(("yes", n))
        elif max(w, c) <= n - m:
            out.append(("no", n))
    lab = None
    if c <= w - BIN_MARGIN or (not C):
        lab = bin_value(w, WAIT_BINS)
    return out, lab


# ---------------------------------------------------------------- silences (nobody speaks)

def low_cover(gap, low_runs, frac=0.85):
    a, b = gap[0] + 0.15, gap[1] - 0.15
    if b <= a:
        return False
    cov = sum(max(0.0, min(b, e) - max(a, s)) for s, e in low_runs)
    return cov >= frac * (b - a)


def silence_candidates(inc_all, sp_all, T: float, low_runs):
    """yes@N: an inclusive gap >= N + m that is >= 85 % low-energy in the mix; no@N: every speech gap <= N - m and
    no low-energy stretch >= N - m. The bin label is given only when the speech-gap and energy views agree."""
    g_inc = L.gaps(inc_all, T)
    g_sp = L.gaps(sp_all, T)
    gi = max(g_inc, key=lambda g: g[1] - g[0]) if g_inc else (0.0, 0.0)
    li = gi[1] - gi[0]
    ls = max((e - s for s, e in g_sp), default=0.0)
    lo = max((e - s for s, e in low_runs), default=0.0) if low_runs is not None else None
    out = []
    for n in SIL_NS:
        m = margin(n)
        if li >= n + m and (low_runs is None or low_cover(gi, low_runs)):
            out.append(("yes", n))
        elif ls <= n - m and (lo is None or lo < n - m):
            out.append(("no", n))
    lab = None
    if ls - li < BIN_MARGIN:
        lab = bin_value(li, SIL_BINS)
        if lab is not None and low_runs is not None:
            if lo > li + 0.6 or (li >= 2.0 and not low_cover(gi, low_runs)):
                lab = None
            elif bin_value(lo, SIL_BINS, 0.0) not in (lab,) and li >= 2.0:
                lab = None
    return out, lab


# ---------------------------------------------------------------- caller cut-in / barge-in

def cutin_events(a_iv, o_iv):
    """Other-side runs starting inside an agent run -> [(t, prior, ov, dur, agent_run_end, kind)].
    kind: backchannel (run < 0.7 s), negligible (prior < 0.15 or ov < 0.15), strong (prior >= 0.4, ov >= 0.4,
    run >= 0.8 s), ambiguous otherwise."""
    ra, ro = L.runs(a_iv), L.runs(o_iv)
    out = []
    for s, e in ro:
        for b0, b1 in ra:
            if b0 < s < b1:
                prior, ov, dur = s - b0, min(e, b1) - s, e - s
                if dur < 0.7:
                    kind = "backchannel"
                elif prior < 0.15 or ov < 0.15:
                    kind = "negligible"
                elif prior >= 0.4 and ov >= 0.4 and dur >= 0.8:
                    kind = "strong"
                else:
                    kind = "ambiguous"
                out.append((s, prior, ov, dur, b1, kind))
                break
    return out


def cutin_labels(a_iv, o_iv):
    """-> (barge yes/no/None, count label/None, stopped yes/no/None)."""
    ev = cutin_events(a_iv, o_iv)
    kinds = [k for *_, k in ev]
    if "ambiguous" in kinds:
        return None, None, None
    strong = [x for x in ev if x[5] == "strong"]
    if not strong:
        return "no", "none", None
    cnt = "once" if len(strong) == 1 else "two or more times"
    stops = []
    for s, _p, _o, _d, b1, _k in strong:
        rem = b1 - s
        stops.append("yes" if rem <= 1.2 else "no" if rem >= 2.0 else None)
    stopped = stops[0] if stops[0] is not None and all(x == stops[0] for x in stops) else None
    return "yes", cnt, stopped


# ---------------------------------------------------------------- talk share

def share_value(a_iv, o_iv):
    ta, to = L.total(a_iv), L.total(o_iv)
    if ta + to <= 0:
        return None
    return to / (ta + to)


def share_bin(s: float, cuts, labels_, m: float = SHARE_MARGIN):
    for c in cuts:
        if abs(s - c) < m:
            return None
    k = sum(s >= c for c in cuts)
    return labels_[k]


def who_more(s: float):
    if s <= 0.40:
        return "agent"
    if s >= 0.60:
        return "other"
    if 0.45 <= s <= 0.55:
        return "equal"
    return None


# ---------------------------------------------------------------- all two-party labels

def two_party(sp: dict, inc: dict, T: float, low_runs=None):
    """sp/inc: {'agent': [...], 'other': [...]} -> {family: [(answer, param)]} + facts."""
    a, o = sp["agent"], sp["other"]
    out = {}
    wc, wlab = waited_candidates(a, o, T)
    if wc:
        out["waited"] = wc
    if wlab:
        out["wait_bin"] = [(wlab, None)]
    sc, slab = silence_candidates(inc["agent"] + inc["other"], a + o, T, low_runs)
    if sc:
        out["silence"] = sc
    if slab:
        out["sil_bin"] = [(slab, None)]
    barge, cnt, stopped = cutin_labels(a, o)
    if barge:
        out["barge"] = [(barge, None)]
    if cnt:
        out["barge_count"] = [(cnt, None)]
    if stopped:
        out["barge_stop"] = [(stopped, None)]
    ao, aw = L.talked_over(a, o, T)
    if ao:
        out["aiover"] = [(ao, None)]
    if aw:
        out["aiover_when"] = [(aw, None)]
    s = share_value(a, o)
    if s is not None:
        for bid, (cuts, labs) in SHARE_BINNINGS.items():
            b = share_bin(s, cuts, labs)
            if b:
                out.setdefault("share", []).append((b, bid))
        w = who_more(s)
        if w:
            out["more_talk"] = [(w, None)]
    fl = L.first_last({"agent": a, "other": o})
    if fl[0]:
        out["first"] = [(fl[0], None)]
    if fl[1]:
        out["last"] = [(fl[1], None)]
    facts = {"share": round(s, 3) if s is not None else None,
             "max_wait": round(max(wait_gaps(a, o, T)[0], default=0.0), 3),
             "agent_s": round(L.total(a), 2), "other_s": round(L.total(o), 2)}
    return out, facts


def agree(truth: dict, vad: dict) -> dict:
    """Keep a truth candidate unless the VAD-based labels give a different answer for the same param."""
    out = {}
    for fam, cands in truth.items():
        v = vad.get(fam)
        keep = []
        for ans, par in cands:
            if v is None:
                keep.append((ans, par))
                continue
            same_par = [va for va, vp in v if vp == par]
            if not same_par or ans in same_par:
                keep.append((ans, par))
        if keep:
            out[fam] = keep
    return out
