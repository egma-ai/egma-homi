"""sb_aicall build logic: HVB turns, speaker split, balanced item selection, items and validation."""
from __future__ import annotations

import itertools
import random
from collections import Counter, defaultdict

import numpy as np

import aic_questions as Q

MAX_ITEMS_PER_CLIP = 5
SOURCE = {"A": "sb_aicall_ai_human", "B": "sb_aicall_voicemail", "C": "sb_aicall_automated",
          "D": "sb_aicall_nobody", "E": "sb_aicall_two_humans"}
GROUPS = {"live_in": "L", "live_out": "L", "live_choice": "L", "waited": "w", "wait_bin": "w", "silence": "s",
          "sil_bin": "s", "barge": "b", "barge_count": "b", "barge_stop": "bs", "aiover": "a", "aiover_when": "a",
          "share": "t", "more_talk": "t", "first": "f", "last": "l", "vm_left": "v", "tone": "tn", "auto_in": "au",
          "auto_out": "au"}
RARE = 0.03


def hvb_turns(segs):
    """sb_dialog hvb.make_turns + the segment texts (for re-voicing agent turns)."""
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
        t["texts"] = [g["text"] for g in t["segs"] if g["speech"]]
        del t["segs"]
    return turns


def hvb_speaker_split(conv_spk: dict, frac: float):
    """Copied from data/sb_dialog/build.py (pair greedy): CAL speaker set S; conversations with both speakers in
    S -> CAL, with neither -> train, with exactly one -> dropped."""
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
    split = {c: ("CAL" if v <= S else ("train" if not (v & S) else None)) for c, v in conv_spk.items()}
    return split, S


def _caps(d: dict, skew: float) -> dict:
    tot = sum(d.values())
    common_ = [n for n in d.values() if n >= RARE * tot]
    m = min(common_) if common_ else 0
    return {lab: (n if n < RARE * tot else min(n, int(max(m, 1) * skew))) for lab, n in d.items()}


def select_items(clips, rng: random.Random, max_per_clip: int = MAX_ITEMS_PER_CLIP, skew: float = 2.0,
                 batch: int = 20):
    """clips: [{key, split, labels: {family: [(answer, param)]}}] -> [(clip, family, answer, param)].
    Per (split, family, param): yes/no exactly equal, choice/score answers <= skew x the rarest. At most
    max_per_clip items and one family per GROUPS group per clip."""
    supply = Counter()
    cands = defaultdict(list)
    for c in clips:
        for f, lst in c["labels"].items():
            for ans, par in lst:
                k = (c["split"], f, ans, par)
                supply[k] += 1
                cands[k].append(c)
    unit = defaultdict(dict)
    for (sp, f, ans, par), n in supply.items():
        unit[(sp, f, par)][ans] = n
    quota = {}
    for (sp, f, par), d in unit.items():
        if len(d) < 2:
            continue
        if f in Q.NOUL:
            m = min(d.get("yes", 0), d.get("no", 0))
            for ans in d:
                quota[(sp, f, ans, par)] = m
        else:
            for ans, cap in _caps(d, skew).items():
                quota[(sp, f, ans, par)] = cap
    keys = sorted((k for k in quota if quota[k] > 0), key=lambda k: (-quota[k] / supply[k], str(k)))
    for k in keys:
        rng.shuffle(cands[k])
    ptr = Counter()
    used = defaultdict(set)
    n_on = Counter()
    picked = []
    progress = True
    while progress:
        progress = False
        for k in keys:
            got = 0
            lst = cands[k]
            g = GROUPS[k[1]]
            # yes/no families fill at 2x the pace (they compete with choice views of the same fact for a clip's slot)
            while got < (2 * batch if k[1] in Q.NOUL else batch) and quota[k] > 0 and ptr[k] < len(lst):
                c = lst[ptr[k]]
                ptr[k] += 1
                if n_on[c["key"]] >= max_per_clip or g in used[c["key"]]:
                    continue
                picked.append((c, k[1], k[2], k[3]))
                used[c["key"]].add(g)
                n_on[c["key"]] += 1
                quota[k] -= 1
                got += 1
            progress |= got > 0
    by = defaultdict(list)
    for p in picked:
        by[(p[0]["split"], p[1], p[3])].append(p)
    out = []
    for (sp, f, par), ps in sorted(by.items(), key=lambda kv: str(kv[0])):
        if f in Q.NOUL:
            ys = [p for p in ps if p[2] == "yes"]
            ns = [p for p in ps if p[2] == "no"]
            m = min(len(ys), len(ns))
            out += ys[:m] + ns[:m]
        else:
            cnt = Counter(p[2] for p in ps)
            if len(cnt) < 2:
                continue
            caps = _caps(dict(cnt), skew)
            seen = Counter()
            for p in ps:
                if seen[p[2]] < caps[p[2]]:
                    out.append(p)
                    seen[p[2]] += 1
    return out


def make_items(picked, rng: random.Random, common):
    items = []
    for c, f, ans, par in picked:
        roles = Q.roles_for(c["scenario"], c["direction"], rng)
        q, opts, gold, qt = Q.build_question(f, ans, par, roles, rng)
        meta = {"qtype": qt, "family": f, "answer": ans, "param": par, "scenario": c["scenario"], "sub": c["sub"],
                "direction": c["direction"], "clip": c["key"], "bank": c["bank"], "duration_s": round(c["dur"], 2),
                "licence": c["licence"], "source_id": c["source_id"], "speakers": c.get("speakers", []),
                "voices": c.get("voices", []), "roles": list(roles), "label_rule": "timing-exact (planned timings, "
                "energy-VAD veto)" if c["scenario"] in "AE" else "by construction"}
        if c.get("events"):
            meta["constructed_events"] = c["events"]
        if qt == "score":
            meta["levels"] = list(opts)
        shuf = Q.NoShuffle() if qt == "score" else rng
        iid = f"sb_aicall/{c['key']}/{f}"
        it = common.make_item(item_id=iid, set_name=c["split"], source=SOURCE[c["scenario"]], question=q,
                              options=opts, target=Q.one_hot(len(opts), gold), gold=gold, rng=shuf,
                              audio={"bank": c["bank"], "key": c["key"]}, channel="clean", state_text=None, meta=meta)
        items.append(it)
    return items


def validate(items, bank_index: dict, dur_by_key: dict, forbidden_questions=()) -> dict:
    keys = {"id", "set", "source", "question", "options", "target", "gold", "audio", "channel", "state_text", "meta"}
    ids = set()
    bal = defaultdict(Counter)
    per_clip = Counter()
    durs = []
    forb = {q.strip().lower() for q in forbidden_questions}
    for it in items:
        assert keys <= set(it), it["id"]
        assert it["id"] not in ids, it["id"]
        ids.add(it["id"])
        assert 2 <= len(it["options"]) <= 20 and len(set(it["options"])) == len(it["options"]), it["id"]
        assert abs(sum(it["target"]) - 1) < 1e-6, it["id"]
        assert it["gold"] == int(np.argmax(it["target"])), it["id"]
        assert it["meta"]["qtype"] in ("choice", "noul", "score")
        assert it["set"] in ("train", "CAL") and it["channel"] == "clean"
        assert it["question"].strip().lower() not in forb, it["question"]
        if it["meta"]["qtype"] == "noul":
            assert sorted(it["options"]) == ["no", "yes"]
        if it["meta"]["qtype"] == "score":
            assert it["options"] == it["meta"]["levels"]
        a = it["audio"]
        assert a["key"] in bank_index[a["bank"]], it["id"]
        d = dur_by_key[a["key"]]
        lo = 19.9 if it["meta"]["scenario"] in "AE" else 7.9
        assert lo <= d <= 180.5, (it["id"], d)
        durs.append(d)
        per_clip[a["key"]] += 1
        assert it["meta"]["qtype"] != "noul" or it["options"][it["gold"]] == it["meta"]["answer"]
        bal[(it["set"], it["meta"]["family"])][str(it["meta"]["answer"])] += 1
    assert max(per_clip.values()) <= MAX_ITEMS_PER_CLIP
    for (sp, f), cnt in bal.items():
        if f in Q.NOUL:
            assert cnt["yes"] == cnt["no"], (sp, f, cnt)
    return {"n": len(items), "clips": len(per_clip), "dur_min": round(min(durs), 2), "dur_max": round(max(durs), 2),
            "dur_mean": round(float(np.mean(durs)), 2),
            "balance": {f"{sp}|{f}": dict(c) for (sp, f), c in sorted(bal.items())}}
