"""Assemble sb_calltext: LLM results -> agreement filter -> items -> validation -> leakage check -> summary.json."""

from __future__ import annotations

import glob
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import build as B
import common

LIC = {
    "hvb": {"dataset": "Gridspace-Stanford Harper Valley Bank (whole calls)",
            "url": "https://github.com/cricketclub/gridspace-stanford-harper-valley",
            "revision": "0bd721e877c4a85d8c13ff837e68661ea6200a98", "licence": "CC BY 4.0 (GitHub licence field cc-by-4.0)"},
    "e22": {"dataset": "Earnings-22 (Rev.com speech-datasets)", "url": "https://github.com/revdotcom/speech-datasets",
            "revision": "c05ab6fd8b4b627d123c922a22a39e993dd37635",
            "licence": "CC BY-SA 4.0 (earnings22/LICENSE.md; share-alike, flagged)"},
    "ami": {"dataset": "AMI Meeting Corpus (Mix-Headset audio + manual word alignments 1.6.2)",
            "url": "https://groups.inf.ed.ac.uk/ami/corpus/", "revision": "ami_public_manual_1.6.2 (annotations); amicorpus mirror",
            "licence": "CC BY 4.0 (LICENCE.txt in the annotation zip; HF edinburghcstr/ami cardData cc-by-4.0)"},
}
SOURCE = {"hvb": "sb_calltext_hvb", "e22": "sb_calltext_e22", "ami": "sb_calltext_ami"}
AMI_ITEMS_PER_CLIP = 3
PRICE_H100 = 3.95  # $/h (Modal list price, approx.)
E22S_ITEMS_PER_CLIP = 1  # short second-pass earnings windows: one question each (keeps earnings from swamping HVB / AMI)
MIN_WORDS_PER_S = {"e22": 1.6, "ami": 1.0}  # windows below this have transcript gaps (unaligned tokens) -> skipped


def write_atomic_jsonl(path: str, rows: list[dict]) -> None:
    """Write to <path>.tmp then rename, so a reader never sees a half-written manifest."""
    common.write_jsonl(path + ".tmp", rows)
    os.replace(path + ".tmp", path)


def load_results(work: str) -> dict:
    res = {}
    for p in sorted(glob.glob(f"{work}/results_*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            res[r["rid"]] = r
    return res


def assemble(work: str, out: str, audio_dir: str | None = None) -> dict:
    audio_dir = audio_dir or f"{out}/audio"
    rng = random.Random(f"{B.SEED}:assemble")
    convs = [json.loads(l) for l in open(f"{work}/hvb_convs.jsonl")]
    wins = []
    for var in ("e22", "e22s"):
        if os.path.exists(f"{work}/{var}_windows.jsonl"):
            wins += [dict(json.loads(l), variant=var) for l in open(f"{work}/{var}_windows.jsonl")]
    # Earnings-22 CAL = whole earnings calls (file-disjoint), smallest hash first until >= 3% of the windows
    if wins:
        cal_files = B.choose_cal_groups([w["fid"] for w in wins], frac=B.CAL_FRAC, salt="sb_calltext_e22")
        for w in wins:
            w["split"] = "CAL" if w["fid"] in cal_files else "train"
    amiw = [json.loads(l) for l in open(f"{work}/ami_windows.jsonl")] if os.path.exists(f"{work}/ami_windows.jsonl") else []
    if amiw:  # AMI CAL = whole meeting series (same 4 participants), series-disjoint from train
        cal_series = B.choose_cal_groups([w["series"] for w in amiw], frac=B.CAL_FRAC, salt="sb_calltext_ami")
        for w in amiw:
            w["split"] = "CAL" if w["series"] in cal_series else "train"
    res = load_results(work)
    llm_meta = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(f"{work}/llm_meta_*.json"))]
    model = f"{llm_meta[0]['repo']}@{llm_meta[0]['revision']}" if llm_meta else "unknown"

    groups = {}  # (domain, group) -> info
    for c in convs:
        groups[("hvb", c["sid"])] = {"split": c["split"], "clips": c["clips"], "banks": c["banks"], "dur_s": c["dur_s"], "c": c}
    dropped_sparse = Counter()
    for w in list(wins) + list(amiw):
        dom = "ami" if "meeting" in w else "e22"
        w["wps"] = round(len(w["transcript"].split()) / max(1e-6, w["dur_s"]), 3)
        if w["wps"] < MIN_WORDS_PER_S[dom]:
            dropped_sparse[dom] += 1
    wins = [w for w in wins if w["wps"] >= MIN_WORDS_PER_S["e22"]]
    amiw = [w for w in amiw if w["wps"] >= MIN_WORDS_PER_S["ami"]]
    for w in wins:
        groups[("e22", w["key"])] = {"split": w["split"], "clips": [w["key"]], "banks": {w["key"]: w["bank"]},
                                     "dur_s": w["dur_s"], "c": w}

    for w in amiw:
        groups[("ami", w["key"])] = {"split": w["split"], "clips": [w["key"]], "banks": {w["key"]: w["bank"]},
                                     "dur_s": w["dur_s"], "c": w}

    agree_stats = defaultdict(Counter)
    cands = defaultdict(list)  # (domain, group) -> views
    for (dom, g), info in groups.items():
        labels = B.LABELS[dom]
        for key, (_, opts) in labels.items():
            r0, r1 = res.get(f"{dom}|{g}|{key}|0"), res.get(f"{dom}|{g}|{key}|1")
            st = agree_stats[f"{dom}:{key}"]
            if r0 is None or r1 is None:
                st["missing"] += 1
                continue
            n = len(opts)
            p0 = B.to_canonical(r0["probs"], B.run_order(n, 0))
            p1 = B.to_canonical(r1["probs"], B.run_order(n, 1))
            ok, top, mean = B.agree(p0, p1)
            st["n"] += 1
            st["same_top"] += int(np.argmax(p0) == np.argmax(p1))
            st["low_mass"] += int(min(r0["mass"], r1["mass"]) < 0.5)
            if not ok:
                continue
            st["agreed"] += 1
            st[f"top={opts[top]}"] += 1
            if dom == "hvb" and key == "reason":
                tasks = info["c"].get("tasks") or []
                if tasks and B.HVB_INTENTS[top] not in tasks:
                    st["reason_mismatch_metadata"] += 1
                    continue
            for v in B.views_for(dom, key, mean):
                v.update({"label_key": key, "domain": dom, "group": g, "split": info["split"],
                          "llm_p": [round(float(p0[top]), 4), round(float(p1[top]), 4)], "llm_top": opts[top]})
                cands[(dom, g)].append(v)
            st["used_label"] += 1

    # noul balance (per split), allocation to clips (<= 4 per clip), re-balance
    allc = [v for vs in cands.values() for v in vs]
    kept = []
    for split in ("train", "CAL"):
        kept += B.balance_noul([v for v in allc if v["split"] == split], random.Random(f"{B.SEED}:bal:{split}"))
    by_group = defaultdict(list)
    for v in kept:
        by_group[(v["domain"], v["group"])].append(v)
    slots = {k: groups[k]["clips"] for k in by_group}
    caps = {k: E22S_ITEMS_PER_CLIP for k in by_group if k[0] == "e22" and groups[k]["c"].get("variant") == "e22s"}
    caps.update({k: AMI_ITEMS_PER_CLIP for k in by_group if k[0] == "ami"})
    alloc = B.allocate(by_group, slots, rng, caps=caps)
    final = []
    for split in ("train", "CAL"):
        final += B.balance_noul([v for v in alloc if v["split"] == split], random.Random(f"{B.SEED}:bal2:{split}"))
    final.sort(key=lambda v: (v["domain"], v["group"], v["clip"], v["view"]))

    items = {"train": [], "CAL": []}
    for v in final:
        dom, g, info = v["domain"], v["group"], groups[(v["domain"], v["group"])]
        irng = random.Random(f"{B.SEED}:{dom}:{v['clip']}:{v['view']}")
        q = B.question_text(dom, v["view"], irng)
        channel = "phone" if irng.random() < B.PHONE_FRAC else "clean"
        st_text = irng.choice({"hvb": B.STATE_HVB, "e22": B.STATE_E22, "ami": B.STATE_AMI}[dom]) if irng.random() < 0.3 else None
        opts = list(v["options"])
        tgt = np.asarray(v["target"], float)
        gold = opts.index(v["label"])
        c = info["c"]
        meta = {"qtype": v["qtype"], "view": v["view"], "label_key": v["label_key"], "label": v["label"],
                "label_source": "llm_transcript", "llm": model, "llm_top_prob_runs": v["llm_p"],
                "agreement_rule": f"same top option in both option orders, each >= {B.AGREE_MIN}",
                "licence": LIC[dom]["licence"], "dataset_revision": LIC[dom]["revision"], "clip_dur_s": info["dur_s"]}
        if dom == "hvb":
            meta.update({"conversation": g, "call": g, "agent_speaker": c.get("agent_speaker"),
                         "caller_speaker": c.get("caller_speaker"), "hvb_tasks": c.get("tasks"),
                         "slue_split": c.get("slue_split"), "audio": "agent+caller channel mono mix (whole call)",
                         "baked_codec": None if v["clip"].endswith("_mix") else c.get("codec")})
        elif dom == "ami":
            meta.update({"meeting": c["meeting"], "series": c["series"], "window_start_s": c["start_s"],
                         "n_speakers_in_window": c.get("n_speakers"), "audio": "meeting window, Mix-Headset (mono)"})
        else:
            meta.update({"file": c["fid"], "window_start_s": c["start_s"], "n_speakers_in_window": c.get("n_speakers"),
                         "window_pass": c.get("variant"), "audio": "earnings-call window (mono)"})
        item = common.make_item(
            item_id=f"sb_calltext/{dom}/{v['clip']}/{v['view']}", set_name=v["split"], source=SOURCE[dom],
            question=q, options=opts, target=tgt, gold=gold,
            rng=B._NoShuffle() if v["qtype"] == "score" else irng,
            audio={"bank": info["banks"][v["clip"]], "key": v["clip"]}, channel=channel, state_text=st_text, meta=meta)
        items[v["split"]].append(item)

    write_atomic_jsonl(f"{out}/items_train.jsonl", items["train"])
    write_atomic_jsonl(f"{out}/items_cal.jsonl", items["CAL"])

    # Validation (reload what was written)
    val = validate(out, audio_dir)
    leak = leakage(convs, wins, items["train"] + items["CAL"], amiw)
    all_items = items["train"] + items["CAL"]
    used_clips = {(it["audio"]["bank"], it["audio"]["key"]) for it in all_items}
    banks = sorted({b for b, _ in used_clips})
    secs = 0.0
    for b in banks:
        idx = json.loads(Path(f"{audio_dir}/{b}.json").read_text())
        secs += sum(idx[k][1] for bb, k in used_clips if bb == b) / 16000.0
    cnt = Counter((it["source"], it["meta"]["view"], it["set"]) for it in all_items)
    examples = {}
    for src in SOURCE.values():
        ex = [it for it in items["train"] if it["source"] == src]
        r2 = random.Random(1)
        examples[src] = [{"question": e["question"], "options": e["options"], "gold": e["gold"],
                          "target": e["target"], "view": e["meta"]["view"]} for e in r2.sample(ex, min(3, len(ex)))]
    gpu_s = sum(m.get("seconds", 0) for m in llm_meta)
    agree_out = {}
    for k, st in sorted(agree_stats.items()):
        n = max(1, st["n"])
        agree_out[k] = {"n": st["n"], "same_top_rate": round(st["same_top"] / n, 4),
                        "agreement_rate": round(st["agreed"] / n, 4), "used": st["used_label"],
                        "label_counts": {kk[4:]: vv for kk, vv in st.items() if kk.startswith("top=")},
                        **{kk: vv for kk, vv in st.items() if kk in ("missing", "low_mass", "reason_mismatch_metadata")}}
    summary = {
        "builder": "sb_calltext",
        "counts": {"train": len(items["train"]), "CAL": len(items["CAL"]),
                   "per_source": dict(Counter(f"{it['source']}/{it['set']}" for it in all_items)),
                   "per_qtype": dict(Counter(f"{it['source']}/{it['meta']['qtype']}" for it in all_items)),
                   "per_view": {f"{s}/{v}/{sp}": n for (s, v, sp), n in sorted(cnt.items())},
                   "noul_balance": noul_balance(all_items),
                   "label_dist": label_dist(all_items),
                   "channel": dict(Counter(it["channel"] for it in all_items)),
                   "clips_used": len(used_clips), "max_items_per_clip": max(Counter((it["audio"]["bank"], it["audio"]["key"]) for it in all_items).values())},
        "audio_hours_used": round(secs / 3600, 2),
        "clip_duration_s": val["durations"],
        "sources": {SOURCE[k]: v for k, v in LIC.items()},
        "label_model": {"model": model, "licence": llm_meta[0]["licence"] if llm_meta else None,
                        "engine": "vLLM 0.31.0, greedy, first-token top-20 logprobs over option letters",
                        "runs": "2 per question (canonical and reversed option order)",
                        "keep_rule": f"same top option in both runs, each >= {B.AGREE_MIN}; target = mean distribution",
                        "requests": sum(m.get("n_requests", 0) for m in llm_meta),
                        "gpu_seconds_scoring": round(gpu_s, 1), "mean_letter_mass": [m.get("mean_mass") for m in llm_meta]},
        "agreement": agree_out,
        "windows_dropped_sparse_transcript": dict(dropped_sparse),
        "prep": {k: json.loads(Path(f"{work}/{k}_prep.json").read_text())
                 for k in ("hvb", "e22", "e22s", "ami") if os.path.exists(f"{work}/{k}_prep.json")},
        "examples": examples,
        "validation": val,
        "leakage_check": leak,
        "paths": {"train": f"{out}/items_train.jsonl", "cal": f"{out}/items_cal.jsonl", "audio": f"{audio_dir}/", "banks": banks},
    }
    Path(f"{out}/summary.json.tmp").write_text(json.dumps(summary, indent=1, default=str))  # written last
    os.replace(f"{out}/summary.json.tmp", f"{out}/summary.json")
    return {k: summary[k] for k in ("counts", "audio_hours_used", "validation", "leakage_check")}


def noul_balance(items):
    d = defaultdict(Counter)
    for it in items:
        if it["meta"]["qtype"] == "noul":
            d[f"{it['source']}/{it['meta']['view']}/{it['set']}"][it["options"][it["gold"]]] += 1
    return {k: dict(v) for k, v in sorted(d.items())}


def label_dist(items):
    d = defaultdict(Counter)
    for it in items:
        if it["meta"]["qtype"] != "noul":
            d[f"{it['source']}/{it['meta']['view']}"][it["meta"]["label"]] += 1
    return {k: dict(v) for k, v in sorted(d.items())}


def validate(out: str, audio_dir: str) -> dict:
    tr = common.read_jsonl(f"{out}/items_train.jsonl")
    ca = common.read_jsonl(f"{out}/items_cal.jsonl")
    errs = Counter()
    idx_cache = {}
    durs = []
    ids = set()
    for it in tr + ca:
        for e in B.validate_item(it):
            errs[e] += 1
        if it["id"] in ids:
            errs["duplicate id"] += 1
        ids.add(it["id"])
        b, k = it["audio"]["bank"], it["audio"]["key"]
        if b not in idx_cache:
            p = f"{audio_dir}/{b}.json"
            idx_cache[b] = json.loads(Path(p).read_text()) if os.path.exists(p) else {}
        if k not in idx_cache[b]:
            errs["audio key missing"] += 1
        else:
            durs.append(idx_cache[b][k][1] / 16000.0)
        if it["meta"]["qtype"] == "score" and it["source"].endswith("hvb") and it["options"] != B.SENTIMENT:
            errs["score order"] += 1
    # spot-load a few clips from the banks
    rng = random.Random(0)
    loaded = []
    for it in rng.sample(tr, min(20, len(tr))):
        bank = common.AudioBank(audio_dir, it["audio"]["bank"])
        x = bank.get(it["audio"]["key"])
        loaded.append({"key": it["audio"]["key"], "sec": round(len(x) / 16000, 1), "peak": round(float(np.max(np.abs(x))), 3),
                       "rms": round(float(np.sqrt(np.mean(x ** 2))), 4), "finite": bool(np.isfinite(x).all())})
    bank_sizes = {Path(p).stem: round(os.path.getsize(p) / 1e9, 3) for p in glob.glob(f"{audio_dir}/*.npy")}
    def grp(it):
        m = it["meta"]
        return m.get("conversation") or m.get("file") or m.get("series")

    cal_groups = {grp(it) for it in ca}
    tr_groups = {grp(it) for it in tr}
    d = np.array(durs) if durs else np.zeros(1)
    return {"n_train": len(tr), "n_cal": len(ca), "errors": dict(errs), "cal_share": round(len(ca) / max(1, len(tr) + len(ca)), 4),
            "cal_train_group_overlap": len(cal_groups & tr_groups), "spot_loaded": loaded, "bank_gb": bank_sizes,
            "durations": {"min": round(float(d.min()), 1), "mean": round(float(d.mean()), 1), "max": round(float(d.max()), 1),
                          "p10": round(float(np.percentile(d, 10)), 1), "p90": round(float(np.percentile(d, 90)), 1)}}


def leakage(convs, wins, items, amiw=()) -> dict:
    out = {}
    sids = {c["sid"] for c in convs}
    fids = {w["fid"] for w in wins} | {w["meeting"] for w in amiw}  # earnings file ids + AMI meeting ids
    our_q = {it["question"].strip().lower() for it in items}
    adi = "/vol/adi/items.jsonl"
    if os.path.exists(adi):
        adi_calls, adi_spk, adi_q, n = set(), set(), set(), 0
        fid_hits = Counter()
        for line in open(adi):
            d = json.loads(line)
            n += 1
            m = d.get("meta") or {}
            if m.get("call"):
                adi_calls.add(m["call"])
            for g in d.get("groups") or []:
                if g.startswith("hvb_call:"):
                    adi_calls.add(g.split(":", 1)[1])
                if g.startswith("hvb_speaker:"):
                    adi_spk.add(g.split(":", 1)[1])
            adi_q.add(str(d.get("question", "")).strip().lower())
            for f in fids:
                if f in line:
                    fid_hits[f] += 1
        our_spk = {str(c.get("agent_speaker")) for c in convs} | {str(c.get("caller_speaker")) for c in convs}
        out["adi"] = {"items": n, "hvb_call_overlap": len(sids & adi_calls), "e22_or_ami_id_mentions": dict(fid_hits),
                      "question_text_overlap": sorted(our_q & adi_q),
                      "hvb_speaker_overlap": len(our_spk & adi_spk),
                      "note": ("HVB has ~59 speakers who appear across many calls, so HVB leakage is checked at the "
                               "conversation level (no evaluation-set or SLUE-test conversation is used). Speaker ids overlap by design.")}
    ev = "/vol/data/eval_items.jsonl"
    if os.path.exists(ev):
        txt = open(ev).read()
        out["exp0_eval"] = {"hvb_call_mentions": sum(1 for s in sids if s in txt),
                            "e22_file_mentions": sum(1 for f in fids if f in txt)}
    out["ok"] = (out.get("adi", {}).get("hvb_call_overlap", 0) == 0 and not out.get("adi", {}).get("e22_or_ami_id_mentions")
                 and not out.get("adi", {}).get("question_text_overlap")
                 and out.get("exp0_eval", {}).get("hvb_call_mentions", 0) == 0
                 and out.get("exp0_eval", {}).get("e22_file_mentions", 0) == 0)
    return out
