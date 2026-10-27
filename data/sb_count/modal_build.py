"""Build sb_count on Modal (CPU; the Kokoro announcer phrases use one L4 GPU):
`modal run --detach data/sb_count/modal_build.py::prep_all`, then `::full` (or `::smoke` for one tiny shard).
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
WORK = f"{VOL}/stageb_work/sb_count"
OUT = f"{VOL}/stageb/sb_count"
N_SHARDS = 40
KOKORO_REPO, KOKORO_REV = "hexgrad/Kokoro-82M", "f3ff3571791e39611d31c381e3a41a3af07b4987"

volume = modal.Volume.from_name("homi", create_if_missing=False)


def _code(img):
    img = img.add_local_file(INFERENCE / "common.py", "/root/common.py")
    img = img.add_local_file(INFERENCE / "answer_codes.json", "/root/answer_codes.json")
    if (STAGEB / "forbidden_questions.json").exists():  # optional list of question strings items must not equal
        img = img.add_local_file(STAGEB / "forbidden_questions.json", "/root/forbidden_questions.json")
    img = img.add_local_file(STAGEB / "sb_phone" / "fx.py", "/root/fx.py")
    for f in ("build.py", "labels.py", "questions.py", "librispeech_speaker_gender.json"):
        img = img.add_local_file(HERE / f, f"/root/{f}")
    return img


cpu_image = _code(
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_pip_install("numpy==2.3.3", "scipy==1.16.2", "pyarrow==21.0.0", "soundfile==0.13.1", "huggingface_hub[hf_xet]==0.35.3")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
)
tts_image = _code(
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng", "git")
    .pip_install("torch==2.7.1", "kokoro==0.9.4", "misaki[en]==0.9.4", "transformers==4.51.3", "huggingface_hub==0.30.2",
                 "numpy==1.26.4", "scipy==1.13.1", "soundfile==0.13.1",
                 "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl")
    .env({"HF_HOME": f"{VOL}/hf_cache"})
)
app = modal.App("sb-count-build")
secrets = [modal.Secret.from_name("huggingface")]


def _hf_env():
    import os
    for k in ("HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_HUB_TOKEN"):
        if os.environ.get(k) and not os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = os.environ[k]


# LibriTTS-R pools and announcer phrases

@app.function(image=cpu_image, volumes={VOL: volume}, secrets=secrets, timeout=3600, cpu=2, memory=8192, retries=1,
              max_containers=40)
def prep(args: tuple) -> dict:
    _hf_env()
    import build
    r = build.prep_libri(*args)
    volume.commit()
    return r


@app.function(image=tts_image, gpu="L4", volumes={VOL: volume}, timeout=1800, retries=1)
def tts_numbers() -> dict:
    """Announcer phrases '<Noun> <number>.' for every announcer voice -> WORK/tts/announcer.npz"""
    import numpy as np
    from huggingface_hub import hf_hub_download
    from kokoro import KModel, KPipeline
    from scipy.signal import resample_poly

    import build
    import labels as L

    path = Path(f"{WORK}/tts/announcer.npz")
    if path.exists():
        return {"skipped": True}
    cfg = hf_hub_download(KOKORO_REPO, "config.json", revision=KOKORO_REV)
    mdl = hf_hub_download(KOKORO_REPO, "kokoro-v1_0.pth", revision=KOKORO_REV)
    model = KModel(config=cfg, model=mdl).to("cuda").eval()
    pipes = {lc: KPipeline(lang_code=lc, model=model) for lc in "ab"}
    arrays, fails = {}, []
    for v in build.ANN_VOICES_TRAIN + build.ANN_VOICES_CAL:
        vp = hf_hub_download(KOKORO_REPO, f"voices/{v}.pt", revision=KOKORO_REV)
        pipe = pipes["b" if v.startswith("b") else "a"]
        for noun in build.ANN_NOUNS:
            for n in range(1, 7):
                text = f"{noun.capitalize()} {L.NUM_WORDS[n]}."
                try:
                    ch = [r.audio.detach().cpu().numpy() for r in pipe(text, voice=vp, speed=1.0) if r.audio is not None]
                    y = resample_poly(np.concatenate(ch).astype(np.float64), 2, 3).astype(np.float32)
                    a, b = L.trim_bounds(y, 16000)
                    arrays[f"{v}|{noun}|{n}"] = y[a:b]
                except Exception as e:  # noqa: BLE001
                    fails.append([v, text, repr(e)[:120]])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("announcer.tmp.npz")
    np.savez(tmp, **arrays)
    tmp.rename(path)
    volume.commit()
    return {"phrases": len(arrays), "fails": fails}


def libri_plan():
    import build
    plan = []
    for i in range(18):
        plan.append(("train.clean.100", i, "tc100", 0.25, 24))
    n360 = len(build._hf_list("data/train.clean.360/"))
    n500 = len(build._hf_list("data/train.other.500/"))
    for i in sorted({int(round(j * (n360 - 1) / 9)) for j in range(10)}):
        plan.append(("train.clean.360", i, "tc360", 0.2, 20))
    for i in sorted({int(round(j * (n500 - 1) / 9)) for j in range(10)}):
        plan.append(("train.other.500", i, "to500", 0.15, 16))
    cal = [("dev.clean", i, "dev", 0.4, 40) for i in range(4)]
    return plan, cal


@app.function(image=cpu_image, volumes={VOL: volume}, secrets=secrets, timeout=3600, cpu=2, memory=4096)
def prep_all_remote() -> dict:
    _hf_env()
    import json
    plan, cal = libri_plan()
    t = tts_numbers.spawn()
    res = list(prep.map(plan + cal, order_outputs=False, return_exceptions=True))
    tr = t.get()
    volume.reload()
    out = {"prep": [r if isinstance(r, dict) else repr(r)[:200] for r in res], "tts": tr}
    Path(f"{WORK}/prep_report.json").write_text(json.dumps(out, indent=1))
    volume.commit()
    return out


def pool_tags():
    have = sorted(p.stem for p in Path(f"{WORK}/pool").glob("*.npy"))
    return [t for t in have if not t.startswith("dev")], [t for t in have if t.startswith("dev")]


# Render shards

@app.function(image=cpu_image, volumes={VOL: volume}, timeout=3600, cpu=2, memory=16384, retries=1, max_containers=40)
def render(args: tuple) -> dict:
    import build
    shard, scale, out_dir = args
    build.SHARDS = Path(out_dir)
    volume.reload()
    tr, cal = pool_tags()
    r = build.render_shard(shard, scale, f"{WORK}/tts/announcer.npz", tr, cal)
    volume.commit()
    return r


# Assemble, balance, validate, publish

def answer_key(it):
    m = it["meta"]
    fam, view = m["family"], m["view"]
    if fam == "turns":
        return m["n_turns"] - (1 if view == "changes" else 0)
    if fam == "speakers":
        return m["n_speakers"]
    if fam == "most":
        return m.get("winner")
    if fam == "same":
        return m.get("same")
    if fam == "spkid":
        return {"match": None if m.get("M") is None else f"K{m['K']}M{m['M']}", "pair": it["options"][it["gold"]].split()[-3:].__str__(), "count": it["options"][it["gold"]]}.get(view)
    return it["options"][it["gold"]]


def assemble(shard_dir: str, out_dir: str, publish: bool, build_info: dict | None = None) -> dict:
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
    rng = random.Random(f"{common.SEED}:sb_count:assemble")
    rng.shuffle(items)
    n_raw = len(items)
    # drop out-of-range counts (rare tails)
    keep = []
    for it in items:
        m = it["meta"]
        if m["family"] == "turns" and not (2 <= m["n_turns"] <= 12):
            continue
        if m["family"] == "speakers" and not (1 <= m["n_speakers"] <= 8):
            continue
        keep.append(it)
    items = keep
    # announcer voices: CAL voices only in CAL clips, train voices only in train clips
    n_ann_drop = 0
    keep = []
    for it in items:
        av = it["meta"].get("announcer_voice")
        if av and ((it["set"] == "train") != (av in build.ANN_VOICES_TRAIN)):
            n_ann_drop += 1
            continue
        keep.append(it)
    items = keep
    # choice answers capped at 2x the rarest (per family / view / split); spkid 'none' kept at its design rate
    groups = defaultdict(list)
    for i, it in enumerate(items):
        if it["meta"]["qtype"] == "choice":
            groups[(it["meta"]["family"], it["meta"]["view"], it["set"])].append(i)
    drop = set()
    for g, idx in groups.items():
        keyed = [(i, answer_key(items[i])) for i in idx]
        core = [(i, k) for i, k in keyed if k is not None]
        cnt = Counter(k for _, k in core)
        if len(cnt) < 2:
            continue
        cap = 2 * min(cnt.values())
        seen = Counter()
        for i, k in core:
            seen[k] += 1
            if seen[k] > cap:
                drop.add(i)
    items = [it for i, it in enumerate(items) if i not in drop]
    # noul exactly 50/50 per family / split
    by = defaultdict(lambda: {"yes": [], "no": []})
    for i, it in enumerate(items):
        if it["meta"]["qtype"] == "noul":
            by[(it["meta"]["family"], it["set"])][it["options"][it["gold"]]].append(i)
    drop = set()
    for g, d in by.items():
        n = min(len(d["yes"]), len(d["no"]))
        for k in ("yes", "no"):
            drop |= set(d[k][n:])
    items = [it for i, it in enumerate(items) if i not in drop]
    items.sort(key=lambda it: it["id"])
    train = [it for it in items if it["set"] == "train"]
    cal = [it for it in items if it["set"] == "CAL"]

    # validation
    banks = sorted({it["audio"]["bank"] for it in items})
    idx = {b: json.loads((sd / f"{b}.json").read_text()) for b in banks}
    bad = Counter()
    keys = {"id", "set", "source", "question", "options", "target", "gold", "audio", "channel", "state_text", "meta"}
    mmsu_q = set()
    mp = Path(VOL) / "bench_mmsu" / "truth.jsonl"
    for line in open(mp):
        mmsu_q.add(json.loads(line)["question"].strip().lower())
    fq = Path("/root/forbidden_questions.json")  # optional (data/forbidden_questions.json): e.g. your own test questions
    cbq = {q.strip().lower() for q in json.loads(fq.read_text())} if fq.exists() else set()
    per_clip = Counter()
    spk_by_set, ann_by_set, src_by_set = defaultdict(set), defaultdict(set), defaultdict(set)
    ids = Counter(it["id"] for it in items)
    bad["dup_ids"] = sum(1 for v in ids.values() if v > 1)
    for it in items:
        bad["schema"] += not keys <= set(it)
        bad["n_options"] += not (2 <= len(it["options"]) <= 20)
        bad["dup_options"] += len(set(it["options"])) != len(it["options"])
        bad["target_sum"] += abs(sum(it["target"]) - 1) > 1e-6
        bad["gold_argmax"] += it["gold"] != int(np.argmax(it["target"]))
        bad["audio_key_missing"] += it["audio"]["key"] not in idx[it["audio"]["bank"]]
        bad["mmsu_question_equal"] += it["question"].strip().lower() in mmsu_q
        bad["forbidden_question_equal"] += it["question"].strip().lower() in cbq
        m = it["meta"]
        bad["qtype"] += m.get("qtype") not in ("choice", "noul", "score")
        if m["qtype"] == "noul":
            bad["noul_options"] += sorted(it["options"]) != ["no", "yes"]
        per_clip[it["audio"]["key"]] += 1
        spk_by_set[it["set"]] |= set(map(str, m["speakers"]))
        src_by_set[it["set"]] |= set(map(str, m["source_ids"]))
        if m.get("announcer_voice"):
            ann_by_set[it["set"]].add(m["announcer_voice"])
    bad["items_per_clip_gt4"] = sum(1 for v in per_clip.values() if v > 4)
    # MMSU options overlap (report only; bare numbers are unavoidable)
    mmsu_opts = set()
    for line in open(mp):
        for o in json.loads(line)["options"]:
            mmsu_opts.add(str(o).strip().lower())
    opt_overlap = Counter(o for it in items for o in it["options"] if o.strip().lower() in mmsu_opts and not o.strip().isdigit()
                          and o not in ("yes", "no"))
    # leakage vs the evaluation items
    adi = build.adi_ids()
    adi_hvb_spk = {x for src_, v in adi["speakers_by_source"].items() if "hvb" in src_.lower() for x in v}
    adi_libri_spk = {x for src_, v in adi["speakers_by_source"].items() if "libri" in src_.lower() for x in v}
    mine = spk_by_set["train"] | spk_by_set["CAL"]
    hvb_spk_overlap = sorted(s for s in mine if s.startswith("hvb_spk:") and (s in adi["speakers"] or s.split(":")[-1] in adi_hvb_spk))
    non_hvb_spk_overlap = sorted(s for s in mine if not s.startswith("hvb_spk:") and
                                 (s in adi["speakers"] or (s.startswith("ltr:") and s.split(":")[-1] in adi_libri_spk)))
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
            "hvb_calls_in_adi": len(hvb_calls & adi["hvb_calls"]),
            "speakers_in_adi_non_hvb": non_hvb_spk_overlap, "hvb_speakers_also_in_adi": len(hvb_spk_overlap),
            "adi_sources_with_speakers": sorted(adi["speakers_by_source"]),
            "hvb_note": "HVB has ~60 voice actors and the evaluation set's HVB test calls cover most of them, so speaker-level "
                        "exclusion is impossible; as in sb_dialog we exclude every SLUE-HVB test conversation (a superset of "
                        "the evaluation set's HVB calls).",
            "ids_in_exp0_eval": len(all_src & ev_ids) + len((spk_by_set["train"] | spk_by_set["CAL"]) & ev_ids)}
    split_chk = {"cal_train_speaker_overlap": sorted(spk_by_set["train"] & spk_by_set["CAL"])[:20],
                 "n_cal_train_speaker_overlap": len(spk_by_set["train"] & spk_by_set["CAL"]),
                 "cal_train_source_overlap": len(src_by_set["train"] & src_by_set["CAL"]),
                 "cal_train_announcer_overlap": sorted(ann_by_set["train"] & ann_by_set["CAL"]),
                 "n_train_speakers": len(spk_by_set["train"]), "n_cal_speakers": len(spk_by_set["CAL"])}
    # load a sample back
    bk = {b: common.AudioBank(sd, b) for b in banks}
    durs = []
    for it in random.Random(7).sample(items, min(400, len(items))):
        x = common.item_audio(it, bk)
        assert x is not None and np.isfinite(x).all() and float(np.abs(x).max()) <= 1.0
        durs.append(len(x) / 16000)
    all_durs = [v[1] / 16000 for b in banks for k, v in idx[b].items() if k in per_clip]
    hours = sum(all_durs) / 3600
    bal = defaultdict(Counter)
    for it in items:
        m = it["meta"]
        if m["qtype"] == "noul":
            bal[f"{m['family']}|{it['set']}"][it["options"][it["gold"]]] += 1
        elif m["qtype"] in ("choice", "score"):
            bal[f"{m['family']}|{m['view']}|{m['qtype']}"][str(answer_key(it)) if m["qtype"] == "choice" else it["options"][it["gold"]]] += 1
    errors = {k: v for k, v in bad.items() if v}
    ok = (not errors and split_chk["n_cal_train_speaker_overlap"] == 0 and not split_chk["cal_train_announcer_overlap"]
          and leak["source_ids_in_adi"] == 0 and leak["hvb_calls_in_slue_test"] == 0 and leak["hvb_calls_in_adi"] == 0
          and not leak["speakers_in_adi_non_hvb"] and leak["ids_in_exp0_eval"] == 0)
    cnt = lambda f: dict(sorted(Counter(f(it) for it in items).items()))  # noqa: E731
    summary = {
        "builder": "sb_count", "built_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "n_items": len(items), "n_train": len(train),
        "n_cal": len(cal), "cal_frac": round(len(cal) / max(1, len(items)), 4), "n_raw_items": n_raw,
        "dropped_announcer_voice_mismatch": n_ann_drop,
        "counts_by_source": cnt(lambda it: it["source"]),
        "counts_by_source_split": cnt(lambda it: f"{it['source']}|{it['set']}"),
        "counts_by_qtype": cnt(lambda it: it["meta"]["qtype"]),
        "counts_by_family_qtype": cnt(lambda it: f"{it['meta']['family']}|{it['meta']['qtype']}"),
        "counts_by_family_view": cnt(lambda it: f"{it['meta']['family']}|{it['meta']['view']}"),
        "counts_by_corpus": cnt(lambda it: it["meta"]["corpus"]),
        "counts_by_clip_kind": cnt(lambda it: f"{it['meta']['clip_kind']}|{it['meta']['corpus']}"),
        "phone_fx_items": sum(1 for it in items if it["meta"].get("phone_fx")),
        "clips_used": len(per_clip), "audio_hours_used": round(hours, 3),
        "clip_dur_s": {"min": round(min(all_durs), 2), "max": round(max(all_durs), 2), "mean": round(float(np.mean(all_durs)), 2)},
        "banks": {b: {"clips": len(idx[b]), "gb": round((sd / f"{b}.npy").stat().st_size / 1e9, 3)} for b in banks},
        "sources": build.SOURCES,
        "label_balance": {k: dict(v) for k, v in sorted(bal.items())},
        "validation": {"errors": errors, "ok": ok, "sample_loaded": len(durs),
                       "sample_dur_range": [round(min(durs), 2), round(max(durs), 2)], "max_items_per_clip": max(per_clip.values()),
                       "mmsu_questions_checked": len(mmsu_q), "mmsu_question_equal": bad["mmsu_question_equal"],
                       "forbidden_questions_checked": len(cbq), "mmsu_option_strings_reused_non_numeric": dict(opt_overlap.most_common(10)),
                       "split": split_chk, "leakage": leak},
        "examples": {},
        "build": build_info or {},
    }
    ex = defaultdict(list)
    for it in items:
        if len(ex[it["source"]]) < 3 and (not ex[it["source"]] or it["meta"]["view"] != ex[it["source"]][-1]["view"]):
            ex[it["source"]].append({"question": it["question"], "options": it["options"], "gold": it["gold"],
                                     "answer": it["options"][it["gold"]], "view": it["meta"]["view"], "audio": it["audio"]})
    summary["examples"] = dict(ex)
    out.mkdir(parents=True, exist_ok=True)
    bk.clear()
    import gc
    gc.collect()
    if publish:
        assert ok, f"validation failed: {errors} {split_chk} {leak}"
        (out / "summary.json").unlink(missing_ok=True)
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
    (out / "summary.json.tmp").write_text(json.dumps(summary, indent=1))
    (out / "summary.json.tmp").rename(out / "summary.json")
    volume.commit()
    return {k: summary[k] for k in ("n_items", "n_train", "n_cal", "counts_by_source", "counts_by_qtype", "audio_hours_used",
                                     "clip_dur_s")} | {"validation": summary["validation"]}


@app.function(image=cpu_image, volumes={VOL: volume}, timeout=3 * 3600, cpu=4, memory=32768)
def main(scale: float = 1.0, n_shards: int = N_SHARDS, smoke: bool = False) -> dict:
    import json
    import time
    t0 = time.time()
    volume.reload()
    if not smoke:
        if not Path(f"{WORK}/tts/announcer.npz").exists():
            print(tts_numbers.remote(), flush=True)
        plan, cal = libri_plan()
        print([r for r in prep.map(plan + cal, order_outputs=False, return_exceptions=True) if not (isinstance(r, dict) and r.get("skipped"))], flush=True)
        volume.reload()
    sdir = f"{WORK}/smoke/shards" if smoke else f"{WORK}/shards"
    res = []
    for r in render.map([(i, scale, sdir) for i in range(n_shards)], order_outputs=False, return_exceptions=True):
        res.append(r if isinstance(r, dict) else {"error": repr(r)[:300]})
        print(json.dumps(res[-1])[:600], flush=True)
    volume.reload()
    Path(sdir).mkdir(parents=True, exist_ok=True)
    (Path(sdir) / "render_report.json").write_text(json.dumps(res, indent=1))
    volume.commit()
    drops = {}
    for r in res:
        for k, v in (r.get("drops") or {}).items():
            drops[k] = drops.get(k, 0) + v
    info = {"render_shards": len(res), "render_errors": [r for r in res if "error" in r],
            "render_container_s": round(sum(r.get("elapsed_s", 0) for r in res), 1), "render_cpu_per_container": 2,
            "clip_drops": drops, "scale": scale,
            "modal_cost_note": "CPU-only render/prep (2 vCPU containers) + one L4 for ~1-3 min of Kokoro announcer TTS"}
    out = assemble(sdir, f"{WORK}/smoke/out" if smoke else OUT, publish=not smoke, build_info=info)
    out["elapsed_s"] = round(time.time() - t0, 1)
    out["render_errors"] = [r for r in res if "error" in r]
    out["render_cpu_container_s"] = round(sum(r.get("elapsed_s", 0) for r in res), 1)
    return out


@app.local_entrypoint()
def prep_all():
    print(prep_all_remote.remote())


@app.local_entrypoint()
def smoke(scale: float = 0.06, n_shards: int = 1):
    import json
    print(json.dumps(main.remote(scale, n_shards, True), indent=1)[:6000])


@app.local_entrypoint()
def full(scale: float = 0.6, n_shards: int = N_SHARDS):
    import json
    print(json.dumps(main.remote(scale, n_shards, False), indent=1)[:6000])


# Read-only re-check of the published set

@app.function(image=cpu_image, volumes={VOL: volume}, timeout=1800, cpu=2, memory=8192)
def check_remote(out_dir: str = OUT) -> dict:
    """Reload the published items and banks with inference/common.py helpers and re-check them (read-only)."""
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
            "banks": len(idx), "bank_gb": round(sum((out / "audio" / f"{b}.npy").stat().st_size for b in idx) / 1e9, 2),
            "summary_n_items": summ["n_items"], "summary_ok": summ["validation"]["ok"]}


@app.local_entrypoint()
def check(out_dir: str = OUT):
    import json
    print(json.dumps(check_remote.remote(out_dir), indent=1))
