"""sb_dialog build logic: AMI / Earnings-21 timings, windows, splits, facts -> labels, balanced item selection and
validation. Modal orchestration lives in modal_build.py."""
from __future__ import annotations

import html
import random
import re
from collections import Counter, defaultdict

import numpy as np

import labels as L
import questions as Q

MAX_ITEMS_PER_CLIP = 4
PHONE_FRAC = {"hvb": 0.75, "ami": 0.5, "earnings21": 0.6}
CTX = {"hvb": "call", "ami": "recording", "earnings21": "recording"}


# AMI

W_RE = re.compile(r'<w [^>]*?starttime="([\d.]+)" endtime="([\d.]+)"([^>]*)>([^<]*)</w>')
VS_RE = re.compile(r'<(vocalsound|nonvocalsound)[^>]*?starttime="([\d.]+)" endtime="([\d.]+)"')
MEET_RE = re.compile(r'<meeting [^>]*observation="([^"]+)"[^>]*>(.*?)</meeting>', re.S)
SPK_RE = re.compile(r'<speaker [^>]*nxt_agent="([A-Z])"[^>]*global_name="([^"]+)"')


def parse_ami_words(xml: str):
    """-> (speech intervals, inclusive intervals) for one speaker's words file."""
    sp, inc = [], []
    for s, e, attrs, _txt in W_RE.findall(xml):
        s, e = float(s), float(e)
        if 'punc="true"' in attrs or e <= s:
            continue
        sp.append((s, e))
        inc.append((s, e))
    for _k, s, e in VS_RE.findall(xml):
        s, e = float(s), float(e)
        if e > s:
            inc.append((s, e))
    return L.merge(sp, 0.05), L.merge(inc, 0.05)


def parse_ami_meetings(xml: str) -> dict:
    """meetings.xml -> {meeting: {agent letter: global speaker id}}"""
    out = {}
    for obs, body in MEET_RE.findall(xml):
        out[obs] = {a: g for a, g in SPK_RE.findall(body)}
    return out


# Earnings-21 RTTM

def parse_rttm(text: str) -> dict:
    """-> {speaker: [(s, e)]}"""
    out = defaultdict(list)
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            s, d = float(p[3]), float(p[4])
            if d > 0:
                out[p[7]].append((s, s + d))
    return {k: L.merge(v) for k, v in out.items()}


# Windows over long recordings

def plan_windows(incl_by_spk: dict, dur: float, rng: random.Random, min_len=20.0, max_len=180.0):
    """Non-overlapping windows that start/end inside silent gaps (>= 0.3 s with nobody making a sound)."""
    allv = L.merge([iv for v in incl_by_spk.values() for iv in v])
    cuts = [(a + b) / 2 for a, b in L.gaps(allv, dur) if b - a >= 0.3]
    if len(cuts) < 2:
        return []
    cuts = np.array(cuts)
    wins = []
    a = cuts[0] + 0.0
    while True:
        u = rng.random()
        target = rng.uniform(20, 40) if u < 0.35 else rng.uniform(40, 90) if u < 0.8 else rng.uniform(90, 175)
        ok = cuts[(cuts >= a + min_len) & (cuts <= a + max_len)]
        if len(ok) == 0:
            nxt = cuts[cuts > a + max_len]
            if len(nxt) == 0:
                break
            a = float(nxt[0])
            continue
        b = float(ok[np.argmin(np.abs(ok - (a + target)))])
        wins.append((round(a, 3), round(b, 3)))
        skip = rng.uniform(0, 12)
        nxt = cuts[cuts >= b + skip]
        if len(nxt) == 0:
            break
        a = float(nxt[0])
    return wins


# Splits

def components(groups: dict) -> list[set]:
    """groups: {item: set(speakers)} -> connected components of items sharing speakers."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for it, spk in groups.items():
        for s in spk:
            parent[find(("s", s))] = find(("i", it))
        find(("i", it))
    comp = defaultdict(set)
    for it in groups:
        comp[find(("i", it))].add(it)
    return sorted(comp.values(), key=lambda c: sorted(c))


def cal_by_components(groups: dict, frac: float, rng: random.Random) -> set:
    comps = components(groups)
    rng.shuffle(comps)
    n_tot, cal = len(groups), set()
    for c in comps:
        if len(cal) >= frac * n_tot:
            break
        if len(c) <= max(2, 0.5 * frac * n_tot) + 4:
            cal |= c
    return cal


def hvb_speaker_split(conv_spk: dict, frac: float, rng: random.Random | None = None):
    """CAL speaker set S grown by the speaker pair that adds the most CAL conversations per conversation lost:
    conversations with both speakers in S -> CAL, with neither -> train, with exactly one -> dropped."""
    import itertools

    spk_all = sorted({s for v in conv_spk.values() for s in v})
    n = len(conv_spk)

    def stats(S):
        cal = sum(1 for v in conv_spk.values() if v <= S)
        drop = sum(1 for v in conv_spk.values() if (v & S) and not v <= S)
        return cal, drop

    S = set()
    while stats(S)[0] < frac * n and len(S) < max(2, len(spk_all) // 3):
        c0, d0 = stats(S)
        best = None
        for a, b in itertools.combinations(spk_all, 2):
            T = S | {a, b}
            if T == S:
                continue
            c, d = stats(T)
            sc = (c - c0) / (1 + d - d0)
            if best is None or sc > best[0]:
                best = (sc, a, b)
        if best is None or best[0] <= 0:
            break
        S |= {best[1], best[2]}
    split = {}
    for c, v in conv_spk.items():
        split[c] = "CAL" if v <= S else ("train" if not (v & S) else None)
    return split, S


# Facts -> labels

def _low_runs_cover(gap, low_runs, frac=0.85):
    if gap is None:
        return False
    a, b = gap[0] + 0.15, gap[1] - 0.15
    if b <= a:
        return False
    cov = sum(max(0.0, min(b, e) - max(a, s)) for s, e in low_runs)
    return cov >= frac * (b - a)


def silence_labels(incl_all, speech_all, T, low_runs):
    """Annotation-based silence facts + energy veto (low_runs = low-energy stretches of the mix >= 1 s)."""
    f = L.silence_facts(incl_all, speech_all, T)
    out = {}
    if low_runs is None:  # planning pass (no audio yet): annotation-only labels, no energy veto
        if f["long"]:
            out["long_sil"] = f["long"]
        if f["where"]:
            out["sil_where"] = f["where"]
        return out, f
    max_low = max((e - s for s, e in low_runs), default=0.0)
    if f["long"] == "yes" and _low_runs_cover(f["gap"], low_runs):
        out["long_sil"] = "yes"
    elif f["long"] == "no" and max_low < 3.0:
        out["long_sil"] = "no"
    if f["where"] is not None and _low_runs_cover(f["gap"], low_runs):
        # the longest low-energy stretch must be (inside) the annotated gap
        lr = max(low_runs, key=lambda r: r[1] - r[0]) if low_runs else None
        if lr and lr[0] < f["gap"][1] and f["gap"][0] < lr[1]:
            out["sil_where"] = f["where"]
    return out, f


def two_party_labels(sp: dict, inc: dict, T: float, ev: dict | None, low_runs):
    """sp/inc/ev: {'agent': [...], 'caller': [...]}; ev = per-track energy VAD (veto on contradiction)."""
    a, c = sp["agent"], sp["caller"]
    out = {}
    ao, aw = L.talked_over(a, c, T)
    co, cw = L.talked_over(c, a, T)
    if ev is not None:
        eao, eaw = L.talked_over(ev["agent"], ev["caller"], T)
        eco, ecw = L.talked_over(ev["caller"], ev["agent"], T)
        if eao is not None and eao != ao:
            ao, aw = None, None
        if eco is not None and eco != co:
            co, cw = None, None
        if eaw is not None and aw is not None and eaw != aw:
            aw = None
        if ecw is not None and cw is not None and ecw != cw:
            cw = None
    if ao:
        out["a_over"] = ao
    if co:
        out["c_over"] = co
    if aw:
        out["a_over_when"] = aw
    if cw:
        out["c_over_when"] = cw
    if ao and co:
        out["over_who"] = {("yes", "yes"): "both", ("yes", "no"): "a", ("no", "yes"): "b", ("no", "no"): "neither"}[(ao, co)]
    sh = L.talk_share(a, c)
    if sh and (ev is None or L.talk_share(ev["agent"], ev["caller"]) in (sh, None)):
        out["more_talk"] = sh
        out["agent_share"] = sh
    sr = L.slow_response(a, c)
    if sr and (ev is None or L.slow_response(ev["agent"], ev["caller"]) in (sr, None)):
        out["slow_resp"] = sr
    fl = L.first_last({"a": a, "b": c})
    efl = L.first_last({"a": ev["agent"], "b": ev["caller"]}) if ev is not None else (None, None)
    if fl[0] and efl[0] in (fl[0], None):
        out["first_spk"] = fl[0]
    if fl[1] and efl[1] in (fl[1], None):
        out["last_spk"] = fl[1]
    am = L.overlap_amount_2p(a, c)
    if am and (ev is None or L.overlap_amount_2p(ev["agent"], ev["caller"]) in (am, None)):
        out["ov_amount"] = am
    sil, f = silence_labels(inc["agent"] + inc["caller"], a + c, T, low_runs)
    out.update(sil)
    return out, {"max_gap": f["max_gap"], "agent_s": round(L.total(a), 2), "caller_s": round(L.total(c), 2)}


def multi_party_labels(sp: dict, inc: dict, T: float, low_runs):
    out = {}
    yn, when, amount = L.any_overlap(sp, T)
    if yn:
        out["mp_overlap"] = yn
    if when:
        out["mp_overlap_when"] = when
    if amount:
        out["ov_amount"] = amount
    n = L.n_speakers(sp)
    if n:
        out["n_spk"] = n
    if n in ("two", "three or more"):
        fl = L.first_last(sp)
        if fl[0] and fl[1]:
            out["same_first_last"] = "yes" if fl[0] == fl[1] else "no"
        d = L.dominant(sp)
        if d:
            out["dominant"] = d
    sil, f = silence_labels([iv for v in inc.values() for iv in v], [iv for v in sp.values() for iv in v], T, low_runs)
    out.update(sil)
    return out, {"max_gap": f["max_gap"], "n_active": sum(1 for v in sp.values() if L.total(v) > 0.2)}


# Selection + items

GROUPS = {  # at most one family per group on the same clip (they express the same fact)
    "a_over": "a", "a_over_when": "a", "c_over": "c", "c_over_when": "c", "over_who": "o", "ov_amount": "o",
    "mp_overlap": "o", "mp_overlap_when": "o", "more_talk": "t", "agent_share": "t", "dominant": "t",
    "long_sil": "s", "sil_where": "s", "slow_resp": "r", "first_spk": "f", "last_spk": "l",
    "same_first_last": "f", "n_spk": "n",
}
RARE = 0.03  # a choice/score label with < 3 % of its family's supply is exempt from the balance caps


def _caps(d: dict, skew: float) -> dict:
    """label -> cap for a choice/score family: skew x the rarest non-rare label; rare labels keep everything."""
    tot = sum(d.values())
    common_ = [n for n in d.values() if n >= RARE * tot]
    m = min(common_) if common_ else 0
    return {lab: (n if n < RARE * tot else min(n, int(max(m, 1) * skew))) for lab, n in d.items()}


def select_items(clips: list[dict], rng: random.Random, max_per_clip: int = MAX_ITEMS_PER_CLIP,
                 choice_skew: float = 2.0, batch: int = 25):
    """clips: [{key, source, split, labels: {family: label}}] -> [(clip, family, label)].
    Quotas per (source, split, family, label): noul = min(yes, no) (exact 50/50 after a final trim); choice /
    score: <= choice_skew x the rarest (non-rare) label. Assignment is round-robin over labels (`batch` items per
    label per round, scarcest labels first) so every label fills at the same pace; <= max_per_clip items and <= 1
    family per GROUPS group per clip. A final trim restores the balance caps."""
    supply = Counter()
    cands = defaultdict(list)
    for c in clips:
        for f, lab in c["labels"].items():
            k = (c["source"], c["split"], f, lab)
            supply[k] += 1
            cands[k].append(c)
    fam_labels = defaultdict(dict)
    for (src, sp, f, lab), n in supply.items():
        fam_labels[(src, sp, f)][lab] = n
    quota = {}
    for (src, sp, f), d in fam_labels.items():
        if len(d) < 2:
            continue  # a family with a single label value is never asked
        if Q.QTYPE[f] == "noul":
            m = min(d.get("yes", 0), d.get("no", 0))
            for lab in d:
                quota[(src, sp, f, lab)] = m
        else:
            for lab, cap in _caps(d, choice_skew).items():
                quota[(src, sp, f, lab)] = cap
    keys = sorted((k for k in quota if quota[k] > 0), key=lambda k: (-quota[k] / supply[k], k))
    for k in keys:
        rng.shuffle(cands[k])
    ptr = Counter()
    used = defaultdict(set)  # clip key -> used groups
    n_on = Counter()
    picked = []
    progress = True
    while progress:
        progress = False
        for k in keys:
            got = 0
            lst = cands[k]
            g = GROUPS[k[2]]
            while got < batch and quota[k] > 0 and ptr[k] < len(lst):
                c = lst[ptr[k]]
                ptr[k] += 1
                if n_on[c["key"]] >= max_per_clip or g in used[c["key"]]:
                    continue
                picked.append((c, k[2], k[3]))
                used[c["key"]].add(g)
                n_on[c["key"]] += 1
                quota[k] -= 1
                got += 1
            progress |= got > 0
    by = defaultdict(list)
    for p in picked:
        c, f, lab = p
        by[(c["source"], c["split"], f)].append(p)
    out = []
    for (src, sp, f), ps in sorted(by.items(), key=lambda kv: kv[0]):
        if Q.QTYPE[f] == "noul":
            ys = [p for p in ps if p[2] == "yes"]
            ns = [p for p in ps if p[2] == "no"]
            m = min(len(ys), len(ns))
            out += ys[:m] + ns[:m]
        else:
            cnt = Counter(p[2] for p in ps)
            if len(cnt) < 2:
                continue
            caps = _caps(dict(cnt), choice_skew)
            seen = Counter()
            for p in ps:
                if seen[p[2]] < caps[p[2]]:
                    out.append(p)
                    seen[p[2]] += 1
    return out


def make_items(picked, rng: random.Random, common):
    """picked [(clip, family, label)] -> items (common.make_item format)."""
    items = []
    per_clip = Counter()
    for c, f, lab in picked:
        src = c["source"]
        if src == "hvb":
            roles = rng.choice(Q.ROLE_SETS_AI) if rng.random() < 0.3 else rng.choice(Q.ROLE_SETS_BANK)
        else:
            roles = None
        q, opts, gold, qt = Q.build_question(f, lab, roles, CTX[src], rng)
        per_clip[c["key"]] += 1
        iid = f"sb_dialog/{src}/{c['key']}/{f}"
        meta = {"qtype": qt, "family": f, "label": lab, "clip": c["key"], "bank": c["bank"],
                "duration_s": round(c["dur"], 2), "licence": c["licence"], "source_id": c["source_id"],
                "speakers": c.get("speakers", []), "kind": c.get("kind"), "label_rule": "timing-exact"}
        if c.get("events"):
            meta["constructed_events"] = c["events"]
        shuf = Q.NoShuffle() if qt == "score" else rng
        channel = "phone" if rng.random() < PHONE_FRAC[src] else "clean"
        it = common.make_item(item_id=iid, set_name=c["split"], source=f"sb_dialog_{src}", question=q,
                              options=opts, target=Q.one_hot(len(opts), gold), gold=gold, rng=shuf,
                              audio={"bank": c["bank"], "key": c["key"]}, channel=channel, state_text=None,
                              meta=meta)
        items.append(it)
    return items


def validate(items, bank_index: dict, dur_by_key: dict) -> dict:
    """Schema / target / gold / audio-key / duration / balance checks. Raises on hard failures."""
    keys = {"id", "set", "source", "question", "options", "target", "gold", "audio", "channel", "state_text", "meta"}
    ids = set()
    bal = defaultdict(Counter)
    durs = []
    for it in items:
        assert keys <= set(it), it["id"]
        assert it["id"] not in ids, it["id"]
        ids.add(it["id"])
        assert 2 <= len(it["options"]) <= 20 and len(set(it["options"])) == len(it["options"]), it["id"]
        assert abs(sum(it["target"]) - 1) < 1e-6, it["id"]
        assert it["gold"] == int(np.argmax(it["target"])), it["id"]
        assert it["meta"]["qtype"] in ("choice", "noul", "score")
        if it["meta"]["qtype"] == "noul":
            assert sorted(it["options"]) == ["no", "yes"]
        a = it["audio"]
        assert a["key"] in bank_index[a["bank"]], it["id"]
        d = dur_by_key[a["key"]]
        assert 19.9 <= d <= 180.5, (it["id"], d)
        durs.append(d)
        bal[(it["source"], it["set"], it["meta"]["family"])][it["options"][it["gold"]]] += 1
    for (src, sp, f), cnt in bal.items():
        if Q.QTYPE[f] == "noul":
            assert cnt["yes"] == cnt["no"], (src, sp, f, cnt)
    return {"n": len(items), "dur_min": round(min(durs), 2), "dur_max": round(max(durs), 2),
            "dur_mean": round(float(np.mean(durs)), 2),
            "balance": {f"{s}|{sp}|{f}": dict(c) for (s, sp, f), c in sorted(bal.items())}}


def unescape(s: str) -> str:
    return html.unescape(s)
