"""sb_dialog label logic: conversation-flow facts from per-speaker speech intervals in clip time (seconds). Each function
returns a label or None (ambiguous, not asked). A talk-over of B by A is strong when B had talked >= 0.4 s and both then
talk >= 0.4 s, negligible when either is < 0.15 s; anything in between is ambiguous."""
from __future__ import annotations

JOIN_GAP = 0.25
STRONG = 0.4
NEGLIGIBLE = 0.15
THIRDS = ["first third", "middle third", "last third"]
THIRD_MARGIN = 1.0  # an event must be >= 1 s away from a third boundary


def merge(intervals, join_gap: float = 0.0):
    """Sort + merge intervals whose gap is <= join_gap. Drops empty intervals."""
    iv = sorted((float(s), float(e)) for s, e in intervals if e > s)
    out = []
    for s, e in iv:
        if out and s - out[-1][1] <= join_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def clip_to(intervals, t0: float, t1: float):
    """Intersect intervals with [t0, t1] and shift to start at 0."""
    out = []
    for s, e in intervals:
        s2, e2 = max(s, t0), min(e, t1)
        if e2 > s2:
            out.append((s2 - t0, e2 - t0))
    return out


def total(intervals) -> float:
    return float(sum(e - s for s, e in merge(intervals)))


def runs(intervals):
    return merge(intervals, JOIN_GAP)


def talkover_events(a_iv, b_iv):
    """Events where A starts while B is talking. Returns [(t, prior, ov)] sorted by t."""
    ra, rb = runs(a_iv), runs(b_iv)
    ev = []
    for s, e in ra:
        for b0, b1 in rb:
            if b0 < s < b1:
                ev.append((s, s - b0, min(e, b1) - s))
                break
    return ev


def classify_event(prior: float, ov: float) -> str:
    if prior >= STRONG and ov >= STRONG:
        return "strong"
    if prior < NEGLIGIBLE or ov < NEGLIGIBLE:
        return "negligible"
    return "ambiguous"


def third_of(t: float, T: float):
    """Which third of [0, T] t lies in, or None if within THIRD_MARGIN of a boundary."""
    for k in (1, 2):
        if abs(t - k * T / 3) < THIRD_MARGIN:
            return None
    return min(2, int(3 * t / T))


def talked_over(a_iv, b_iv, T: float):
    """Did A talk over (interrupt) B? -> (yes/no/None, when-label/None).
    when-label in THIRDS + ["it never did"]; None when the yes/no is ambiguous or the first event is at a
    third boundary."""
    ev = talkover_events(a_iv, b_iv)
    kinds = [classify_event(p, o) for _, p, o in ev]
    if "ambiguous" in kinds:
        return None, None
    strong = [t for (t, _, _), k in zip(ev, kinds) if k == "strong"]
    if not strong:
        return "no", "it never did"
    k = third_of(strong[0], T)
    return "yes", (THIRDS[k] if k is not None else None)


def overlap_segments(iv_by_spk: dict):
    """Stretches where >= 2 speakers talk at once (speech intervals, not runs)."""
    pts = []
    for iv in iv_by_spk.values():
        for s, e in merge(iv):
            pts += [(s, 1), (e, -1)]
    pts.sort(key=lambda p: (p[0], p[1]))
    out, n, start = [], 0, None
    for t, d in pts:
        n += d
        if n >= 2 and start is None:
            start = t
        elif n < 2 and start is not None:
            if t > start:
                out.append((start, t))
            start = None
    return merge(out, 0.05)


def any_overlap(iv_by_spk: dict, T: float):
    """Do two people speak at the same time at any point? -> (yes/no/None, when/None, amount-level/None).
    yes: some overlap stretch >= 0.5 s; no: total overlap < 0.1 s and no stretch >= 0.1 s."""
    ov = overlap_segments(iv_by_spk)
    durs = [e - s for s, e in ov]
    tot = sum(durs)
    if not ov or (tot < 0.1 and max(durs) < 0.1):
        return "no", "never", "not at all"
    if max(durs) < 0.5:
        return None, None, None
    first_big = next(s for s, e in ov if e - s >= 0.5)
    earlier = [d for (s, e), d in zip(ov, durs) if s < first_big]
    when = None
    if all(d < 0.1 for d in earlier):
        k = third_of(first_big, T)
        when = THIRDS[k] if k is not None else None
    n_big = sum(d >= 0.5 for d in durs)
    amount = None
    if tot <= 2.0 and n_big <= 2:
        amount = "a little"
    elif tot >= 4.0 and n_big >= 3:
        amount = "a lot"
    return "yes", when, amount


def gaps(intervals, T: float):
    """Silent stretches in [0, T] (including leading / trailing) -> [(start, end)]."""
    m = merge(intervals)
    out, cur = [], 0.0
    for s, e in m:
        if s > cur:
            out.append((cur, s))
        cur = max(cur, e)
    if T > cur:
        out.append((cur, T))
    return out


def silence_facts(incl_iv, speech_iv, T: float, long_yes: float = 3.5, long_no: float = 2.5):
    """-> dict(long=yes/no/None, where=third label/None, gap=(s, e) of the longest inclusive gap, max_gap).
    long yes: an inclusive gap >= 3.5 s; no: every speech-only gap <= 2.5 s."""
    g_incl = gaps(incl_iv, T)
    g_sp = gaps(speech_iv, T)
    mx_incl = max((e - s for s, e in g_incl), default=0.0)
    mx_sp = max((e - s for s, e in g_sp), default=0.0)
    if mx_incl >= long_yes:
        long = "yes"
    elif mx_sp <= long_no:
        long = "no"
    else:
        long = None
    where, best = None, None
    if g_incl:
        srt = sorted(g_incl, key=lambda g: g[1] - g[0], reverse=True)
        best = srt[0]
        L = best[1] - best[0]
        second = srt[1][1] - srt[1][0] if len(srt) > 1 else 0.0
        # the longest speech-only gap must be the same stretch (no longer hidden gap elsewhere)
        sp_best = max(g_sp, key=lambda g: g[1] - g[0]) if g_sp else None
        same = sp_best is not None and sp_best[0] <= best[1] and best[0] <= sp_best[1]
        if L >= 2.0 and second <= 0.6 * L and same and (mx_sp - L) < 0.3 * L:
            k0, k1 = third_of(best[0] + 0.25, T), third_of(best[1] - 0.25, T)
            if k0 is not None and k0 == k1:
                where = THIRDS[k0]
    return {"long": long, "where": where, "gap": best, "max_gap": round(mx_incl, 3)}


def response_delays(resp_iv, prev_iv):
    """Gaps between the end of a `prev` run and the start of the next `resp` run when nobody speaks in
    between (turn transitions prev -> resp)."""
    rp, rr = runs(prev_iv), runs(resp_iv)
    out = []
    for s, e in rp:
        nxt_r = next((r for r in rr if r[0] >= e), None)
        if nxt_r is None:
            continue
        nxt_p = next((p for p in rp if p[0] > s), None)
        if nxt_p is not None and nxt_p[0] < nxt_r[0]:
            continue  # prev spoke again before resp answered
        if any(r[0] < e < r[1] for r in rr):
            continue  # resp was already talking (overlap), not a response
        out.append(nxt_r[0] - e)
    return out


def slow_response(resp_iv, prev_iv, yes_at: float = 2.5, no_below: float = 1.5):
    d = response_delays(resp_iv, prev_iv)
    if not d:
        return None
    if max(d) >= yes_at:
        return "yes"
    if max(d) <= no_below and len(d) >= 2:
        return "no"
    return None


def talk_share(a_iv, b_iv):
    """Share of speech time of A vs B -> 'a' / 'b' / 'equal' / None (margins 0.40-0.45, 0.55-0.60)."""
    ta, tb = total(a_iv), total(b_iv)
    if ta + tb <= 0:
        return None
    s = ta / (ta + tb)
    if s >= 0.60:
        return "a"
    if s <= 0.40:
        return "b"
    if 0.45 <= s <= 0.55:
        return "equal"
    return None


def first_last(iv_by_spk: dict, margin: float = 0.8, min_run: float = 0.3):
    """-> (first speaker, last speaker), each None when another speaker starts / ends within `margin`."""
    starts, ends = {}, {}
    for k, iv in iv_by_spk.items():
        r = [x for x in runs(iv) if x[1] - x[0] >= min_run]
        if r:
            starts[k], ends[k] = r[0][0], r[-1][1]
    if len(starts) < 2:
        return None, None
    s_sorted = sorted(starts.items(), key=lambda kv: kv[1])
    e_sorted = sorted(ends.items(), key=lambda kv: -kv[1])
    first = s_sorted[0][0] if s_sorted[1][1] - s_sorted[0][1] >= margin else None
    last = e_sorted[0][0] if e_sorted[0][1] - e_sorted[1][1] >= margin else None
    return first, last


def n_speakers(iv_by_spk: dict, min_total: float = 1.5, tiny: float = 0.2):
    """-> 'one' / 'two' / 'three or more' / None (a speaker with 0.2-1.5 s of speech is ambiguous)."""
    n = 0
    for iv in iv_by_spk.values():
        t = total(iv)
        if t >= min_total:
            n += 1
        elif t > tiny:
            return None
    if n == 0:
        return None
    return {1: "one", 2: "two"}.get(n, "three or more")


def dominant(iv_by_spk: dict):
    """Does one person do more than half of the talking? yes if top share >= 0.62, no if <= 0.45."""
    tots = sorted((total(iv) for iv in iv_by_spk.values()), reverse=True)
    tots = [t for t in tots if t > 0]
    if len(tots) < 2:
        return None
    s = tots[0] / sum(tots)
    if s >= 0.62:
        return "yes"
    if s <= 0.45:
        return "no"
    return None


def overlap_amount_2p(a_iv, b_iv):
    """Two-party overlap amount (score): 'not at all' / 'a little' / 'a lot' / None."""
    yn, _, amount = any_overlap({"a": a_iv, "b": b_iv}, 1e9)
    if yn == "no":
        return "not at all"
    return amount
