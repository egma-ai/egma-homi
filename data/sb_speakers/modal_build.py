"""Build sb_speakers on Modal (CPU only); also ::smoke, ::tests (incl. the MMSU question test) and ::check.

    modal run --detach data/sb_speakers/modal_build.py::full
"""
from __future__ import annotations

import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
INFERENCE = HERE.parent.parent / "inference"
STAGEB = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(INFERENCE))

VOL = "/vol"
BUILDER = "sb_speakers"
WORK = f"{VOL}/stageb_work/{BUILDER}"
OUT = f"{VOL}/stageb/{BUILDER}"
N_SHARDS = 36
CLIPS_PER_SHARD = 205
MMSU_TRUTH = f"{VOL}/bench_mmsu/truth.jsonl"   # only its "question" field is read, for the non-equality check

volume = modal.Volume.from_name("egma-homi", create_if_missing=False)


def _code(img):
    img = img.add_local_file(INFERENCE / "common.py", "/root/common.py")
    img = img.add_local_file(INFERENCE / "answer_codes.json", "/root/answer_codes.json")
    img = img.add_local_file(STAGEB / "sb_phone" / "fx.py", "/root/fx.py")
    for f in ("build.py", "labels.py", "questions.py", "librispeech_speaker_gender.json"):
        img = img.add_local_file(HERE / f, f"/root/{f}")
    img = img.add_local_dir(HERE / "tests", "/root/tests")
    return img


cpu_image = _code(
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_pip_install("numpy==2.3.3", "scipy==1.16.2", "pyarrow==21.0.0", "soundfile==0.13.1",
                    "huggingface_hub[hf_xet]==0.35.3", "pytest==8.4.2")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
)
app = modal.App("sb-speakers-build")
secrets = [modal.Secret.from_name("huggingface")]

# LibriSpeech ORIGINAL shards: (path prefix, shard, tag, keep fraction, cap per speaker)
LS_TRAIN = [("clean/train.100/", i, "ls_c100", 0.35, 14) for i in (0, 3, 6, 9, 12)] + \
           [("other/train.500/", i, "ls_o500", 0.3, 12) for i in (0, 16, 32, 48)]
LS_CAL = [("clean/validation/", 0, "ls_dev", 0.5, 20)]


def _hf_env():
    import os
    for k in ("HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_HUB_TOKEN"):
        if os.environ.get(k) and not os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = os.environ[k]


def ls_tags():
    return [f"{t}_{i:02d}" for _, i, t, _, _ in LS_TRAIN], [f"{t}_{i:02d}" for _, i, t, _, _ in LS_CAL]


@app.function(image=cpu_image, volumes={VOL: volume}, secrets=secrets, timeout=3600, cpu=2, memory=8192, retries=1,
              max_containers=12)
def prep_ls(args: tuple) -> dict:
    _hf_env()
    import build
    r = build.prep_librispeech(*args)
    volume.commit()
    return r


@app.function(image=cpu_image, volumes={VOL: volume}, timeout=3600, cpu=2, memory=12288, retries=1, max_containers=40)
def render(args: tuple) -> dict:
    import build
    shard, n_clips, out_dir = args
    volume.reload()
    tr, cal = ls_tags()
    r = build.render_shard(shard, n_clips, out_dir, tr, cal)
    volume.commit()
    return r


# Assemble

def answer_key(it):
    m = it["meta"]
    if m["qtype"] == "noul":
        return it["options"][it["gold"]]
    if m["qtype"] == "score":
        return min(4, m["n_speakers"])
    return m["n_speakers"]


def equalize_duration(items, rng, tol: float = 0.15):
    """Drop whole clips so that the mean clip duration is the same (+-tol s) for every N (train only)."""
    import numpy as np
    from collections import defaultdict
    clip = {}
    for it in items:
        if it["set"] == "train":
            clip[it["audio"]["key"]] = (it["meta"]["n_speakers"], it["meta"]["dur_s"])
    byN = defaultdict(list)
    for k, (n, d) in clip.items():
        byN[n].append((d, k))
    mu = float(np.mean([d for _, d in clip.values()]))
    drop = set()
    for n, v in byN.items():
        v = sorted(v)
        lo, hi = 0, len(v)
        m = float(np.mean([d for d, _ in v]))
        while hi - lo > 50 and abs(m - mu) > tol:
            if m > mu:
                hi -= 1  # drop one of the longest
            else:
                lo += 1  # drop one of the shortest
            m = float(np.mean([d for d, _ in v[lo:hi]]))
        drop |= {k for _, k in v[:lo]} | {k for _, k in v[hi:]}
    return [it for it in items if it["audio"]["key"] not in drop], len(drop)


def assemble(shard_dir: str, out_dir: str, publish: bool, build_info: dict | None = None) -> dict:
    import gc
    import json
    import os
    import random
    import shutil
    import time
    from collections import Counter, defaultdict

    import numpy as np

    import build
    import common
    import labels as L

    t0 = time.time()
    sd, out = Path(shard_dir), Path(out_dir)
    items = []
    for p in sorted(sd.glob("items_*.jsonl")):
        items += common.read_jsonl(p)
    rng = random.Random(f"{common.SEED}:{BUILDER}:assemble")
    rng.shuffle(items)
    n_raw = len(items)
    # 1. mean duration must not change with N
    items, n_dur_drop = equalize_duration(items, rng)
    # 2. choice / score answers capped at 2x the rarest (per view and split)
    groups = defaultdict(list)
    for i, it in enumerate(items):
        if it["meta"]["qtype"] in ("choice", "score"):
            groups[(it["meta"]["view"], it["set"])].append(i)
    drop = set()
    for g, idx in groups.items():
        cnt = Counter(answer_key(items[i]) for i in idx)
        if len(cnt) < 2:
            continue
        cap = 2 * min(cnt.values())
        seen = Counter()
        for i in idx:
            k = answer_key(items[i])
            seen[k] += 1
            if seen[k] > cap:
                drop.add(i)
    items = [it for i, it in enumerate(items) if i not in drop]
    # 3. yes/no exactly 50/50 per question kind and split
    by = defaultdict(lambda: {"yes": [], "no": []})
    for i, it in enumerate(items):
        if it["meta"]["qtype"] == "noul":
            by[(it["meta"]["view"], it["set"])][it["options"][it["gold"]]].append(i)
    drop = set()
    for g, d in by.items():
        n = min(len(d["yes"]), len(d["no"]))
        for k in ("yes", "no"):
            drop |= set(d[k][n:])
    items = [it for i, it in enumerate(items) if i not in drop]
    items.sort(key=lambda it: it["id"])
    train = [it for it in items if it["set"] == "train"]
    cal = [it for it in items if it["set"] == "CAL"]

    # Validation
    banks = sorted({it["audio"]["bank"] for it in items})
    idx = {b: json.loads((sd / f"{b}.json").read_text()) for b in banks}
    bad = Counter()
    keys = {"id", "set", "source", "question", "options", "target", "gold", "audio", "channel", "state_text", "meta"}
    mmsu_q = [json.loads(line)["question"] for line in open(MMSU_TRUTH)]
    eq = L.mmsu_equal([it["question"] for it in items], mmsu_q)
    per_clip = Counter()
    spk_by_set, src_by_set = defaultdict(set), defaultdict(set)
    ids = Counter(it["id"] for it in items)
    bad["dup_ids"] = sum(1 for v in ids.values() if v > 1)
    for it in items:
        bad["schema"] += not keys <= set(it)
        bad["n_options"] += not (2 <= len(it["options"]) <= 20)
        bad["dup_options"] += len(set(it["options"])) != len(it["options"])
        bad["target_sum"] += abs(sum(it["target"]) - 1) > 1e-6
        bad["gold_argmax"] += it["gold"] != int(np.argmax(it["target"]))
        bad["audio_key_missing"] += it["audio"]["key"] not in idx[it["audio"]["bank"]]
        bad["bad_source_prefix"] += not it["source"].startswith(BUILDER + "_")
        bad["channel_not_clean"] += it["channel"] != "clean"
        m = it["meta"]
        bad["qtype"] += m.get("qtype") not in ("choice", "noul", "score")
        if m["qtype"] == "noul":
            bad["noul_options"] += sorted(it["options"]) != ["no", "yes"]
            truth = L.noul_truth(m["n_speakers"], m["view"].split("_", 1)[1], m["N_cmp"])
            bad["noul_label"] += it["options"][it["gold"]] != ("yes" if truth else "no")
        elif m["qtype"] == "score":
            bad["score_label"] += it["gold"] != L.score_index(m["n_speakers"])
        else:
            g = it["options"][it["gold"]]
            bad["choice_label"] += not (g.split()[0] in (str(m["n_speakers"]), L.NUM_WORDS[m["n_speakers"]]))
        per_clip[it["audio"]["key"]] += 1
        spk_by_set[it["set"]] |= set(map(str, m["speakers"]))
        src_by_set[it["set"]] |= set(map(str, m["source_ids"]))
    bad["items_per_clip_gt4"] = sum(1 for v in per_clip.values() if v > 4)
    bad["mmsu_question_equal"] = len(eq)

    # Anti-shortcut statistics (train)
    def corr(rows, f):
        return round(L.pearson([f(r) for r in rows], [r["meta"]["n_speakers"] for r in rows]), 4)

    clips_tr = {}
    for it in train:
        clips_tr.setdefault(it["audio"]["key"], it)
    ctr = list(clips_tr.values())
    stats = {
        "r_duration_N_items": corr(train, lambda r: r["meta"]["dur_s"]),
        "r_duration_N_clips": corr(ctr, lambda r: r["meta"]["dur_s"]),
        "r_segments_N_items": corr(train, lambda r: r["meta"]["n_segments"]),
        "r_segments_N_clips": corr(ctr, lambda r: r["meta"]["n_segments"]),
        "r_segments_gap0.2_N_clips": corr(ctr, lambda r: r["meta"]["n_segments_02"]),
        "r_segments_gap0.5_N_clips": corr(ctr, lambda r: r["meta"]["n_segments_05"]),
        "r_pieces_N_clips (placed utterances; >= N by definition, report only)": corr(ctr, lambda r: r["meta"]["n_pieces"]),
        "segment_definition": f"speech stretches of the clip (union of per-speaker-track energy VAD) separated by silences "
                              f">= {L.SEG_GAP_S} s; stretches < 0.1 s ignored",
    }
    byN = defaultdict(list)
    for c in ctr:
        byN[c["meta"]["n_speakers"]].append(c["meta"])
    stats["per_N"] = {n: {"clips": len(v), "dur_mean": round(float(np.mean([m["dur_s"] for m in v])), 2),
                          "dur_min": round(min(m["dur_s"] for m in v), 2), "dur_max": round(max(m["dur_s"] for m in v), 2),
                          "segments_mean": round(float(np.mean([m["n_segments"] for m in v])), 2),
                          "pieces_mean": round(float(np.mean([m["n_pieces"] for m in v])), 2),
                          "short_phrase_frac": round(float(np.mean([m["short_phrase_speakers"] > 0 for m in v])), 3)}
                      for n, v in sorted(byN.items())}
    multi = [c["meta"] for c in ctr if c["meta"]["n_speakers"] >= 2]
    ge3 = [c["meta"] for c in ctr if c["meta"]["n_speakers"] >= 3]
    sg_known = [m for m in multi if m["single_gender"] is not None and all(g is not None for g in m["genders"])]
    stats["short_phrase_frac_N_ge3"] = round(float(np.mean([m["short_phrase_speakers"] > 0 for m in ge3])), 3)
    stats["single_gender_frac_multi_all"] = round(float(np.mean([bool(m["single_gender"]) for m in multi])), 3)
    stats["single_gender_frac_multi_known_gender"] = round(float(np.mean([bool(m["single_gender"]) for m in sg_known])), 3)
    stats["similar_pitch_pair_frac_multi"] = round(float(np.mean([bool(m["similar_pitch_pair"]) for m in multi])), 3)
    stats["overlap_clip_frac"] = round(float(np.mean([m["overlap_s"] > 0.02 for m in multi])), 3)
    stats["overlap_max_s"] = round(max(m["overlap_s"] for m in multi), 3)
    stats["noise_clip_frac"] = round(float(np.mean([("noise" in c["meta"]) for c in ctr])), 3)
    stats["phone_fx_clip_frac"] = round(float(np.mean([("phone_fx" in c["meta"]) for c in ctr])), 3)
    stats["hvb_gender_note"] = "HVB has no gender metadata; HVB 'gender' is an F0 proxy (< 165 Hz -> male) used only to design single-gender clips"
    # Label balance
    bal = defaultdict(Counter)
    for it in items:
        m = it["meta"]
        bal[f"{m['view']}|{it['set']}"][str(answer_key(it))] += 1
    ratios = {k: round(L.cap_ratio(v), 3) for k, v in bal.items()}
    noul_ok = all(v["yes"] == v["no"] for k, v in bal.items() if k.startswith("noul"))
    choice_ok = all(r <= 2.0 + 1e-9 for k, r in ratios.items() if not k.startswith("noul") and k.endswith("train"))
    # Split and leakage
    adi = build.adi_ids()
    adi_hvb_spk = {x for src_, v in adi["speakers_by_source"].items() if "hvb" in src_.lower() for x in v}
    adi_libri_spk = {x for src_, v in adi["speakers_by_source"].items() if "libri" in src_.lower() for x in v}
    mine = spk_by_set["train"] | spk_by_set["CAL"]
    non_hvb_spk_overlap = sorted(s for s in mine if not s.startswith("hvb_spk:") and
                                 (s in adi["speakers"] or (s.startswith("libri:") and s.split(":")[-1] in adi_libri_spk)))
    hvb_spk_overlap = sorted(s for s in mine if s.startswith("hvb_spk:") and (s in adi["speakers"] or s.split(":")[-1] in adi_hvb_spk))
    all_src = src_by_set["train"] | src_by_set["CAL"]
    adi_id_hits = sorted(s for s in all_src if s in adi["ids"] or s.split(":", 1)[-1] in adi["ids"])
    slue = set(json.loads((build.RAW_HVB / "slue_ids.json").read_text())["test"])
    hvb_calls = {s.split(":", 1)[1] for s in all_src if s.startswith("hvb:")}
    ev_ids = set()
    evp = Path(VOL) / "data" / "eval_items.jsonl"
    if evp.exists():
        for line in open(evp):
            mm = json.loads(line).get("meta") or {}
            for k in ("call", "recording", "file", "source_id", "meeting", "clip", "speaker"):
                if mm.get(k) is not None:
                    ev_ids.add(str(mm[k]))
    leak = {"adi_items_file": str(Path(VOL) / "adi" / "items.jsonl"), "adi_ids_checked": len(adi["ids"]),
            "adi_speakers_checked": len(adi["speakers"]), "source_ids_in_adi": len(adi_id_hits), "examples": adi_id_hits[:5],
            "hvb_calls_used": len(hvb_calls), "hvb_calls_in_slue_test": len(hvb_calls & slue),
            "hvb_calls_in_adi": len(hvb_calls & adi["hvb_calls"]), "speakers_in_adi_non_hvb": non_hvb_spk_overlap,
            "hvb_speakers_also_in_adi": len(hvb_spk_overlap),
            "hvb_note": "HVB has ~60 voice actors and the HVB calls in our evaluation set cover most of them, so speaker-level "
                        "exclusion is impossible; as in sb_dialog / sb_count every SLUE-HVB test conversation (a superset of "
                        "those calls) is excluded.",
            "ids_in_exp0_eval": len(all_src & ev_ids) + len(mine & ev_ids)}
    split_chk = {"n_cal_train_speaker_overlap": len(spk_by_set["train"] & spk_by_set["CAL"]),
                 "cal_train_speaker_overlap": sorted(spk_by_set["train"] & spk_by_set["CAL"])[:20],
                 "n_cal_train_source_overlap": len(src_by_set["train"] & src_by_set["CAL"]),
                 "n_train_speakers": len(spk_by_set["train"]), "n_cal_speakers": len(spk_by_set["CAL"])}
    # Load a sample back
    bk = {b: common.AudioBank(sd, b) for b in banks}
    durs = []
    for it in random.Random(7).sample(items, min(400, len(items))):
        x = common.item_audio(it, bk)
        assert x is not None and np.isfinite(x).all() and float(np.abs(x).max()) <= 1.0
        assert abs(len(x) / 16000 - it["meta"]["dur_s"]) < 0.01
        durs.append(len(x) / 16000)
    all_durs = [v[1] / 16000 for b in banks for k, v in idx[b].items() if k in per_clip]
    hours = sum(all_durs) / 3600
    asserts = {
        "abs_r_duration_N_items<=0.10": abs(stats["r_duration_N_items"]) <= 0.10,
        "abs_r_duration_N_clips<=0.10": abs(stats["r_duration_N_clips"]) <= 0.10,
        "abs_r_segments_N_items<=0.5": abs(stats["r_segments_N_items"]) <= 0.5,
        "abs_r_segments_N_clips<=0.5": abs(stats["r_segments_N_clips"]) <= 0.5,
        "noul_50_50": noul_ok, "choice_score_cap_2x_train": choice_ok,
        "cal_speaker_disjoint": split_chk["n_cal_train_speaker_overlap"] == 0,
        "cal_source_disjoint": split_chk["n_cal_train_source_overlap"] == 0,
        "mmsu_question_nonequal": len(eq) == 0,
        "short_phrase_frac_N_ge3>=0.30": stats["short_phrase_frac_N_ge3"] >= 0.30,
        "single_gender_frac_multi>=0.40": stats["single_gender_frac_multi_known_gender"] >= 0.40,
        "no_item_errors": not any(bad.values()),
        "no_adi_leak": leak["source_ids_in_adi"] == 0 and not leak["speakers_in_adi_non_hvb"] and leak["hvb_calls_in_slue_test"] == 0
                       and leak["hvb_calls_in_adi"] == 0 and leak["ids_in_exp0_eval"] == 0,
    }
    ok = all(asserts.values())
    cnt = lambda f: dict(sorted(Counter(f(it) for it in items).items()))  # noqa: E731
    summary = {
        "builder": BUILDER, "built_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "n_items": len(items), "n_train": len(train),
        "n_cal": len(cal), "cal_frac": round(len(cal) / max(1, len(items)), 4), "n_raw_items": n_raw,
        "clips_dropped_for_duration_balance": n_dur_drop,
        "counts_by_source": cnt(lambda it: it["source"]),
        "counts_by_source_split": cnt(lambda it: f"{it['source']}|{it['set']}"),
        "counts_by_qtype": cnt(lambda it: it["meta"]["qtype"]),
        "counts_by_view": cnt(lambda it: it["meta"]["view"]),
        "counts_by_view_split": cnt(lambda it: f"{it['meta']['view']}|{it['set']}"),
        "counts_by_N": cnt(lambda it: it["meta"]["n_speakers"]),
        "counts_by_N_split": cnt(lambda it: f"{it['meta']['n_speakers']}|{it['set']}"),
        "clips_used": len(per_clip), "audio_hours_used": round(hours, 3),
        "clip_dur_s": {"min": round(min(all_durs), 2), "max": round(max(all_durs), 2), "mean": round(float(np.mean(all_durs)), 2)},
        "banks": {b: {"clips": len(idx[b]), "gb": round((sd / f"{b}.npy").stat().st_size / 1e9, 3)} for b in banks},
        "sources": build.SOURCES,
        "anti_shortcut": stats,
        "label_balance": {k: dict(v) for k, v in sorted(bal.items())}, "label_balance_max_min_ratio": ratios,
        "validation": {"ok": ok, "asserts": asserts, "errors": {k: v for k, v in bad.items() if v},
                       "sample_loaded": len(durs), "sample_dur_range": [round(min(durs), 2), round(max(durs), 2)],
                       "max_items_per_clip": max(per_clip.values()),
                       "mmsu_questions_checked": len(mmsu_q), "mmsu_question_equal": len(eq),
                       "mmsu_check_note": "only the 'question' field of /vol/bench_mmsu/truth.jsonl was read, only for this test",
                       "split": split_chk, "leakage": leak},
        "examples": {}, "build": build_info or {},
    }
    ex = defaultdict(list)
    for it in items:
        if len(ex[it["source"]]) < 3 and (not ex[it["source"]] or it["meta"]["view"] != ex[it["source"]][-1]["view"]):
            ex[it["source"]].append({"question": it["question"], "options": it["options"], "gold": it["gold"],
                                     "answer": it["options"][it["gold"]], "view": it["meta"]["view"],
                                     "n_speakers": it["meta"]["n_speakers"], "dur_s": it["meta"]["dur_s"], "audio": it["audio"]})
    summary["examples"] = dict(ex)
    out.mkdir(parents=True, exist_ok=True)
    bk.clear()
    gc.collect()
    if publish:
        assert ok, f"validation failed: {asserts} {dict(bad)} {split_chk} {leak} {stats}"
        assert not (out / "summary.json").exists(), "summary.json already exists; never touched again"
        (out / "audio").mkdir(parents=True, exist_ok=True)
        for b in banks:
            for ext in (".json", ".npy"):
                try:
                    os.rename(sd / f"{b}{ext}", out / "audio" / f"{b}{ext}")  # same volume: instant move
                except OSError:
                    shutil.copyfile(sd / f"{b}{ext}", out / "audio" / f"{b}{ext}")
    common.write_jsonl(out / "items_train.jsonl", train)
    common.write_jsonl(out / "items_cal.jsonl", cal)
    summary["elapsed_assemble_s"] = round(time.time() - t0, 1)
    if publish:
        volume.commit()
    (out / "summary.json.tmp").write_text(json.dumps(summary, indent=1, default=str))
    (out / "summary.json.tmp").rename(out / "summary.json")
    volume.commit()
    return {k: summary[k] for k in ("n_items", "n_train", "n_cal", "counts_by_source", "counts_by_view", "counts_by_N",
                                     "audio_hours_used", "clip_dur_s", "anti_shortcut")} | {"validation": summary["validation"]}


@app.function(image=cpu_image, volumes={VOL: volume}, secrets=secrets, timeout=3 * 3600, cpu=4, memory=32768)
def main(n_shards: int = N_SHARDS, clips: int = CLIPS_PER_SHARD, smoke: bool = False) -> dict:
    import json
    import time
    t0 = time.time()
    volume.reload()
    res_prep = [r if isinstance(r, dict) else repr(r)[:300] for r in prep_ls.map(LS_TRAIN + LS_CAL, order_outputs=False, return_exceptions=True)]
    print(json.dumps(res_prep)[:2000], flush=True)
    volume.reload()
    sdir = f"{WORK}/smoke/shards" if smoke else f"{WORK}/shards"
    res = []
    for r in render.map([(i, clips, sdir) for i in range(n_shards)], order_outputs=False, return_exceptions=True):
        res.append(r if isinstance(r, dict) else {"error": repr(r)[:400]})
        print(json.dumps(res[-1])[:800], flush=True)
    volume.reload()
    Path(sdir).mkdir(parents=True, exist_ok=True)
    (Path(sdir) / "render_report.json").write_text(json.dumps({"prep": res_prep, "render": res}, indent=1))
    volume.commit()
    drops = {}
    for r in res:
        for k, v in (r.get("drops") or {}).items():
            drops[k] = drops.get(k, 0) + v
    ok_res = [r for r in res if "error" not in r]
    info = {"render_shards": len(res), "render_errors": [r for r in res if "error" in r],
            "render_container_s": round(sum(r.get("elapsed_s", 0) for r in ok_res), 1), "render_cpu_per_container": 2,
            "clip_drops": drops, "librispeech_prep": res_prep,
            "pool_speakers": ok_res[0].get("pool_speakers") if ok_res else None,
            "noise_keys": ok_res[0].get("noise_keys") if ok_res else None,
            "noise_licence_report": ok_res[0].get("noise_licence_report") if ok_res else None,
            "hvb_excluded_convs": ok_res[0].get("hvb_excluded_convs") if ok_res else None,
            "modal_cost_note": "CPU only (2 vCPU render containers, 2 vCPU LibriSpeech prep, 4 vCPU assemble); no GPU"}
    out = assemble(sdir, f"{WORK}/smoke/out" if smoke else OUT, publish=not smoke, build_info=info)
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


@app.function(image=cpu_image, volumes={VOL: volume}, timeout=1200, cpu=2, memory=4096)
def tests_remote() -> str:
    import subprocess
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "/root/tests"], capture_output=True, text=True, cwd="/root",
                       env={"PYTHONPATH": "/root", "MMSU_TRUTH": MMSU_TRUTH, "PATH": "/usr/bin:/bin:/usr/local/bin"})
    return p.stdout[-4000:] + p.stderr[-2000:]


@app.local_entrypoint()
def smoke(n_shards: int = 1, clips: int = 60):
    import json
    print(json.dumps(main.remote(n_shards, clips, True), indent=1, default=str)[:12000])


@app.local_entrypoint()
def full(n_shards: int = N_SHARDS, clips: int = CLIPS_PER_SHARD):
    import json
    print(json.dumps(main.remote(n_shards, clips, False), indent=1, default=str)[:12000])


@app.local_entrypoint()
def tests():
    print(tests_remote.remote())


@app.function(image=cpu_image, volumes={VOL: volume}, timeout=1800, cpu=2, memory=8192)
def check_remote(out_dir: str = OUT) -> dict:
    """Reload the published items and banks and re-check them (read-only)."""
    import json
    import random
    from collections import Counter

    import numpy as np

    import common
    out = Path(out_dir)
    tr = common.read_jsonl(out / "items_train.jsonl")
    ca = common.read_jsonl(out / "items_cal.jsonl")
    items = tr + ca
    idx = {b: json.loads((out / "audio" / f"{b}.json").read_text()) for b in sorted({it["audio"]["bank"] for it in items})}
    bad = Counter()
    for it in items:
        bad["target_sum"] += abs(sum(it["target"]) - 1) > 1e-6
        bad["gold_argmax"] += it["gold"] != int(np.argmax(it["target"]))
        bad["n_options"] += not (2 <= len(it["options"]) <= 20)
        bad["key_missing"] += it["audio"]["key"] not in idx[it["audio"]["bank"]]
    banks = {b: common.AudioBank(out / "audio", b) for b in idx}
    durs = []
    for it in random.Random(11).sample(items, min(500, len(items))):
        x = common.item_audio(it, banks)
        assert np.isfinite(x).all() and float(np.abs(x).max()) <= 1.0
        durs.append(len(x) / 16000)
    spk = {s: set() for s in ("train", "CAL")}
    for it in items:
        spk[it["set"]] |= set(map(str, it["meta"]["speakers"]))
    summ = json.loads((out / "summary.json").read_text())
    return {"n_train": len(tr), "n_cal": len(ca), "errors": {k: v for k, v in bad.items() if v}, "loaded": len(durs),
            "dur_range": [round(min(durs), 2), round(max(durs), 2)], "speaker_overlap": len(spk["train"] & spk["CAL"]),
            "banks": len(idx), "summary_n_items": summ["n_items"], "summary_ok": summ["validation"]["ok"]}


@app.local_entrypoint()
def check(out_dir: str = OUT):
    import json
    print(json.dumps(check_remote.remote(out_dir), indent=1))
