"""Build sb_phon on Modal: Kokoro TTS on L4 GPUs, CPU checks and WORLD edits, items, validation, summary.json last.

    modal run --detach data/sb_phon/modal_build.py::full
"""
from __future__ import annotations

import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
STAGEB = HERE.parent
INFERENCE = STAGEB.parent / "inference"
for p in (INFERENCE, STAGEB / "sb_phone", HERE):
    sys.path.insert(0, str(p))

VOL = "/vol"
OUT = f"{VOL}/stageb/sb_phon"
WORK = f"{VOL}/stageb_work/sb_phon"
KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_REV = "f3ff3571791e39611d31c381e3a41a3af07b4987"
CMU_SHA = "74790861f652b15e4ac49015a90074ad62a27690"
CMU_URL = f"https://raw.githubusercontent.com/cmusphinx/cmudict/{CMU_SHA}/cmudict.dict"
LIC = {
    "kokoro": f"{KOKORO_REPO}@{KOKORO_REV}: Apache-2.0 (HF cardData.license=apache-2.0, not gated; pinned as in "
              "data/sb_calls). All speech in this builder is Kokoro output.",
    "misaki": "misaki==0.9.4 G2P + its us_gold/gb_gold lexicons (Apache-2.0; used for TTS phonemes, heteronym "
              "noun/verb pronunciations and the vocabulary filter)",
    "cmudict": f"CMU Pronouncing Dictionary, github.com/cmusphinx/cmudict@{CMU_SHA}: BSD-2-clause style (repo "
               "LICENSE); syllable counts, lexical stress, phones for minimal pairs / sound items",
    "pyworld": "pyworld==0.3.5 (MIT, github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder LICENSE) wrapping WORLD "
               "(modified BSD, M. Morise); tool only, no data",
    "codecs": "ffmpeg (Debian build) codecs for the baked-in phone channel; tool only",
    "text": "all sentences / templates / questions written in code (data/sb_phon/lexicon.py, build.py)",
}

volume = modal.Volume.from_name("egma-homi", create_if_missing=False)
_files = {HERE: ["modal_build.py", "build.py", "lexicon.py", "edit.py"], STAGEB / "sb_phone": ["fx.py"],
          INFERENCE: ["common.py", "answer_codes.json"]}


def _add(img):
    for d, fs in _files.items():
        for f in fs:
            img = img.add_local_file(d / f, f"/root/{f}")
    return img


cpu_image = _add(
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "build-essential")
    .pip_install("numpy==2.1.3", "scipy==1.14.1", "cython==3.0.11", "setuptools==75.8.0")
    .pip_install("pyworld==0.3.5", "misaki==0.9.4", "soundfile==0.13.1")
    .env({"HF_HOME": f"{VOL}/hf_cache"})
)

tts_image = _add(
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng", "git")
    .pip_install(
        "torch==2.7.1", "kokoro==0.9.4", "misaki[en]==0.9.4", "transformers==4.51.3", "huggingface_hub==0.30.2",
        "numpy==1.26.4", "scipy==1.13.1", "soundfile==0.13.1",
        "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
    )
    .env({"HF_HOME": f"{VOL}/hf_cache"})
)

app = modal.App("sb-phon-build")
_TTS = {}


def _kokoro():
    if not _TTS:
        from huggingface_hub import hf_hub_download
        from kokoro import KModel, KPipeline

        cfg = hf_hub_download(KOKORO_REPO, "config.json", revision=KOKORO_REV)
        mdl = hf_hub_download(KOKORO_REPO, "kokoro-v1_0.pth", revision=KOKORO_REV)
        model = KModel(config=cfg, model=mdl).to("cuda").eval()
        _TTS["pipes"] = {lc: KPipeline(lang_code=lc, model=model) for lc in "ab"}
        _TTS["voices"] = {}
    return _TTS


@app.function(image=tts_image, gpu="L4", volumes={VOL: volume}, timeout=3600, retries=1, max_containers=2)
def tts_shard(args) -> dict:
    import json
    import time

    import numpy as np
    from huggingface_hub import hf_hub_download
    from scipy.signal import resample_poly

    shard_id, jobs, work = args
    p_npz = Path(f"{work}/tts/shard_{shard_id:03d}.npz")
    if p_npz.exists():
        return {"shard": shard_id, "skipped": True}
    t0 = time.time()
    K = _kokoro()
    arrays, meta, fails = {}, {}, []
    for key, voice, lang, speed, text in jobs:
        if voice not in K["voices"]:
            K["voices"][voice] = hf_hub_download(KOKORO_REPO, f"voices/{voice}.pt", revision=KOKORO_REV)
        try:
            chunks, toks, ps, pd, off = [], [], [], [], 0.0
            for r in K["pipes"][lang](text, voice=K["voices"][voice], speed=speed, split_pattern=None):
                if r.audio is None:
                    continue
                a = r.audio.detach().cpu().numpy().astype(np.float64)
                for t in r.tokens or []:
                    toks.append([t.text, t.phonemes, None if t.start_ts is None else t.start_ts + off,
                                 None if t.end_ts is None else t.end_ts + off])
                ps.append(r.phonemes)
                pd.append(r.output.pred_dur.detach().cpu().numpy().astype(int).tolist() if r.output is not None and
                          r.output.pred_dur is not None else None)
                chunks.append(a)
                off += len(a) / 24000.0
            y = resample_poly(np.concatenate(chunks), 2, 3).astype(np.float32)
            arrays[key] = y
            meta[key] = {"tokens": toks, "ps": ps[0] if len(ps) == 1 else None, "pred_dur": pd[0] if len(pd) == 1 else None,
                         "n_chunks": len(chunks)}
        except Exception as e:  # noqa: BLE001
            fails.append([key, repr(e)[:200]])
    p_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = p_npz.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    Path(f"{work}/tts/shard_{shard_id:03d}.json").write_text(json.dumps(meta))
    tmp.rename(p_npz)
    volume.commit()
    return {"shard": shard_id, "n": len(jobs), "ok": len(arrays), "fails": fails, "elapsed_s": round(time.time() - t0, 1)}


@app.function(image=cpu_image, volumes={VOL: volume}, timeout=3600, retries=1, cpu=2, memory=8192, max_containers=24)
def proc_shard(args) -> dict:
    import collections
    import json
    import time

    import numpy as np

    import build
    import fx

    shard_id, work = args
    done = Path(f"{work}/proc/shard_{shard_id:03d}.json")
    if done.exists():
        return json.loads(done.read_text())["stats"]
    t0 = time.time()
    volume.reload()
    specs = json.loads(Path(f"{work}/plan_shards/shard_{shard_id:03d}.json").read_text())
    meta = json.loads(Path(f"{work}/tts/shard_{shard_id:03d}.json").read_text())
    arrays, facts, drops, errors, drop_facts = {}, {}, collections.Counter(), [], []
    with np.load(f"{work}/tts/shard_{shard_id:03d}.npz") as z:
        for spec in specs:
            k = spec["key"]
            if k not in meta or k not in z.files:
                drops[f"{spec['fam']}:tts_fail"] += 1
                continue
            tts = {"audio": z[k], **meta[k]}
            try:
                y, f = build.process(spec, tts, fx)
            except Exception as e:  # noqa: BLE001
                errors.append([k, repr(e)[:200]])
                drops[f"{spec['fam']}:error"] += 1
                continue
            if y is None:
                drops[f"{spec['fam']}:{f['drop']}"] += 1
                if len(f) > 1 and len(drop_facts) < 40:
                    drop_facts.append([k, f])
                continue
            assert np.isfinite(y).all() and np.abs(y).max() <= 0.99 and 2.0 <= len(y) / 16000 <= 60.0, k
            arrays[k] = y
            facts[k] = f
    p = Path(f"{work}/proc/shard_{shard_id:03d}.npz")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.rename(p)
    stats = {"shard": shard_id, "kept": len(arrays), "drops": dict(drops), "errors": errors[:20], "drop_facts": drop_facts,
             "elapsed_s": round(time.time() - t0, 1)}
    done.write_text(json.dumps({"stats": stats, "facts": facts}))
    volume.commit()
    return stats


@app.function(image=cpu_image, volumes={VOL: volume}, timeout=3 * 3600, cpu=4, memory=32768)
def main(out_dir: str, work: str, scale: float = 1.0, n_shards: int = 16, stop_after_tts: bool = False) -> dict:
    import collections
    import json
    import random
    import time
    import urllib.request

    import numpy as np

    import build
    import common
    import lexicon as L

    t0 = time.time()
    note = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)  # noqa: E731
    out, wk = Path(out_dir), Path(work)
    assert not (out / "summary.json").exists(), "summary.json exists; final folder is write-once"
    wk.mkdir(parents=True, exist_ok=True)
    cmu_path = Path(f"{WORK}/cmudict_{CMU_SHA[:10]}.dict")
    if not cmu_path.exists():
        cmu_path.parent.mkdir(parents=True, exist_ok=True)
        cmu_path.write_bytes(urllib.request.urlopen(CMU_URL, timeout=60).read())
    cmu = L.parse_cmudict(cmu_path.read_text(encoding="utf-8", errors="replace"))
    import misaki

    mdir = Path(misaki.__file__).parent / "data"
    us_gold = json.loads((mdir / "us_gold.json").read_text())
    gb_gold = json.loads((mdir / "gb_gold.json").read_text())
    plan, pstats = build.plan(common.SEED, scale, cmu, us_gold, gb_gold)
    com = build.common_words(cmu)
    note("plan", len(plan), collections.Counter(c["fam"] for c in plan))
    shards = [plan[i::n_shards] for i in range(n_shards)]
    (wk / "plan_shards").mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(shards):
        (wk / "plan_shards" / f"shard_{i:03d}.json").write_text(json.dumps(s))
    volume.commit()

    tts_stats = []
    todo = [(i, [(c["key"], c["voice"], c["lang"], c["speed"], c["text"]) for c in s], work)
            for i, s in enumerate(shards) if not (wk / "tts" / f"shard_{i:03d}.npz").exists()]
    note("tts shards to run", len(todo))
    for r in (tts_shard.map(todo, order_outputs=False) if todo else []):
        note("tts", {k: v for k, v in r.items() if k != "fails"}, "fails", len(r.get("fails", [])))
        tts_stats.append(r)
    t_tts = time.time() - t0
    if stop_after_tts:
        return {"tts_only": True, "tts_s": round(t_tts), "fails": sum(len(r.get("fails", [])) for r in tts_stats)}
    proc_stats = []
    for r in proc_shard.map([(i, work) for i in range(n_shards)], order_outputs=False):
        note("proc", r["shard"], r["kept"], r["drops"], r["errors"][:2])
        for dfx in r.get("drop_facts", [])[:12]:
            note("dropfact", json.dumps(dfx))
        proc_stats.append(r)
    volume.reload()

    # banks: one train bank per shard, one CAL bank
    audio_dir = out / "audio"
    spec_by = {c["key"]: c for c in plan}
    facts, bank_of, recs, durs = {}, {}, [], []
    cal = {}
    bank_names = []
    for i in range(n_shards):
        fj = json.loads((wk / "proc" / f"shard_{i:03d}.json").read_text())["facts"]
        tr = {}
        with np.load(wk / "proc" / f"shard_{i:03d}.npz") as z:
            for k in z.files:
                (cal if spec_by[k]["split"] == "CAL" else tr)[k] = z[k]
                durs.append(len(z[k]) / 16000)
        name = f"sb_phon_train_{i:02d}"
        common.write_audio_bank(audio_dir, name, tr)
        bank_names.append(name)
        for k in tr:
            bank_of[k] = name
        facts.update(fj)
        note("bank", name, len(tr))
    common.write_audio_bank(audio_dir, "sb_phon_cal", cal)
    bank_names.append("sb_phon_cal")
    for k in cal:
        bank_of[k] = "sb_phon_cal"
    for k, b in bank_of.items():
        recs.append({**spec_by[k], "facts": facts[k], "bank": b, "com": com})
    recs.sort(key=lambda r: r["key"])

    items = []
    for r in recs:
        items += build.items_for(r, common.make_item, common.SEED)
    n_pre = collections.Counter((it["meta"]["qname"], it["set"]) for it in items)
    items = build.balance(items, common.SEED)
    train = [it for it in items if it["set"] == "train"]
    calit = [it for it in items if it["set"] == "CAL"]
    common.write_jsonl(out / "items_train.jsonl", train)
    common.write_jsonl(out / "items_cal.jsonl", calit)
    volume.commit()
    note("items", len(train), len(calit))

    # Validation (reload from disk)
    bank_index = {b: json.loads((audio_dir / f"{b}.json").read_text()) for b in bank_names}
    reloaded = common.read_jsonl(out / "items_train.jsonl") + common.read_jsonl(out / "items_cal.jsonl")
    mmsu_q = []
    mp = Path(f"{VOL}/bench_mmsu/truth.jsonl")
    if mp.exists():
        mmsu_q = [json.loads(line)["question"] for line in mp.read_text().splitlines() if line.strip()]
    report = build.validate(reloaded, bank_index, mmsu_q)
    report["mmsu_questions_checked"] = len(mmsu_q)
    for b in bank_names:
        arr = np.load(audio_dir / f"{b}.npy", mmap_mode="r")
        last = max([s + n for s, n in bank_index[b].values()] or [0])
        assert last == arr.shape[0], b
    used = {it["audio"]["key"] for it in reloaded}
    per_clip = collections.Counter(it["audio"]["key"] for it in reloaded)
    report["max_items_per_clip"] = max(per_clip.values())
    cal_voices = {spec_by[k]["voice"] for k in used if spec_by[k]["split"] == "CAL"}
    tr_voices = {spec_by[k]["voice"] for k in used if spec_by[k]["split"] == "train"}
    report["cal_train_voice_overlap"] = sorted(cal_voices & tr_voices)
    rng = random.Random(1)
    sample = rng.sample(reloaded, min(300, len(reloaded)))
    banks = {b: common.AudioBank(audio_dir, b) for b in bank_names}
    sd = [len(common.item_audio(it, banks)) / 16000 for it in sample]
    report["sample_loaded"] = len(sample)
    report["sample_dur_s"] = [round(min(sd), 2), round(float(np.mean(sd)), 2), round(max(sd), 2)]
    g = collections.defaultdict(collections.Counter)
    for it in reloaded:
        g[(it["meta"]["qname"], it["set"])][it["options"][it["gold"]] if it["meta"]["qtype"] == "noul"
                                            else it["meta"].get("answer_class")] += 1
    label_balance = {f"{q}|{s}": dict(c) for (q, s), c in sorted(g.items())}

    # edit strength (kept clips) + drop rates
    def q(vals):
        vals = [v for v in vals if v is not None]
        return {"n": len(vals), "p10": round(float(np.percentile(vals, 10)), 2), "median": round(float(np.median(vals)), 2),
                "p90": round(float(np.percentile(vals, 90)), 2)} if vals else {"n": 0}

    kept = {k: facts[k] for k in bank_of}
    fam_of = lambda k: spec_by[k]["fam"]  # noqa: E731
    edit_strength = {
        "emphasis_delta_pitch_st": q([f["d_st"] for k, f in kept.items() if fam_of(k) in ("emph", "reason")]),
        "emphasis_delta_energy_db": q([f["d_db"] for k, f in kept.items() if fam_of(k) in ("emph", "reason")]),
        "emphasis_prominence_margin": q([f["prom_margin"] for k, f in kept.items() if fam_of(k) in ("emph", "reason")]),
        "contour_measured_matches_planned": sum(fam_of(k) == "inton" for k in kept),
        "listrise_target_glide_st": q([max(f["glides"]) for k, f in kept.items() if fam_of(k) == "listrise"]),
        "release_burst_db_released": q([f.get("burst_db") for k, f in kept.items() if fam_of(k) == "release" and spec_by[k]["info"]["released"]]),
        "release_burst_db_unreleased": q([f.get("burst_db") for k, f in kept.items() if fam_of(k) == "release" and not spec_by[k]["info"]["released"]]),
        "rules": "emphasis: sentence pitch first flattened (declination removed, movement x0.5) in both the edited clip and "
                 "its control; kept iff measured pitch rise of the target word vs the control >= max(1.5 st, 0.5 x planned), "
                 "energy rise >= 2 dB (unless pitch+duration edit) and target prominence (p90 pitch + dB/3, re sentence "
                 "median) beats every other word by >= 1.0; contour kept iff the measured class (thirds medians, 2 st "
                 "margins) equals the planned one; list rise kept iff target glide >= 3 st and every other noun <= 1.5 st; "
                 "release kept iff high-band burst >= -32 dB re vowel peak (released) / <= -45 dB (unreleased).",
    }
    drops = collections.Counter()
    for r in proc_stats:
        drops.update(r["drops"])
    planned = collections.Counter(c["fam"] for c in plan)
    kept_f = collections.Counter(fam_of(k) for k in kept)

    # Leakage: no evaluation item may reference our banks or clip keys
    ours = set(bank_of) | set(bank_names)
    leak = {}
    for name, path in (("adi", f"{VOL}/adi/items.jsonl"), ("exp0_eval", f"{VOL}/data/eval_items.jsonl")):
        p = Path(path)
        if not p.exists():
            leak[name] = "file not present"
            continue
        hits, n = 0, 0
        for it in common.read_jsonl(p):
            n += 1
            a = it.get("audio") or {}
            vals = {str(a.get("bank")), str(a.get("key"))}
            hits += bool(vals & ours)
        leak[name] = {"items_checked": n, "items_referencing_our_audio": hits}

    def cnt(rows, f):
        return dict(sorted(collections.Counter(f(r) for r in rows).items()))

    examples = {}
    for s in sorted({it["source"] for it in train}):
        rows = [it for it in train if it["source"] == s]
        examples[s] = [{"question": it["question"], "options": it["options"], "gold": it["gold"],
                        "text": it["meta"]["text"]} for it in random.Random(s).sample(rows, min(3, len(rows)))]
    fam_examples = {}
    for fm in build.FAMS:
        rows = [it for it in train if it["meta"]["family"] == fm]
        if rows:
            fam_examples[fm] = [{"question": it["question"], "options": it["options"], "gold": it["gold"],
                                 "text": it["meta"]["text"]} for it in random.Random(fm).sample(rows, min(3, len(rows)))]
    hours = sum(bank_index[bank_of[k]][k][1] for k in used) / 16000 / 3600
    summary = {
        "builder": "sb_phon",
        "out_dir": out_dir,
        "n_items": {"train": len(train), "CAL": len(calit)},
        "n_clips_used": cnt([spec_by[k] for k in used], lambda c: c["split"]),
        "items_per_source": {s: cnt([it for it in reloaded if it["set"] == s], lambda it: it["source"]) for s in ("train", "CAL")},
        "items_per_family": {s: cnt([it for it in reloaded if it["set"] == s], lambda it: it["meta"]["family"]) for s in ("train", "CAL")},
        "items_per_question": {s: cnt([it for it in reloaded if it["set"] == s], lambda it: it["meta"]["qname"]) for s in ("train", "CAL")},
        "items_per_qtype": {s: cnt([it for it in reloaded if it["set"] == s], lambda it: it["meta"]["qtype"]) for s in ("train", "CAL")},
        "items_before_balance": {f"{a}|{b}": v for (a, b), v in sorted(n_pre.items())},
        "phone_fx_share_items": round(sum(it["meta"].get("phone_fx") is not None for it in reloaded) / len(reloaded), 3),
        "audio_hours_used": round(hours, 2),
        "audio_hours_banked": round(sum(durs) / 3600, 2),
        "clips": {"planned": dict(planned), "kept": dict(kept_f), "drops": dict(sorted(drops.items()))},
        "edit_strength": edit_strength,
        "label_balance": label_balance,
        "plan_stats": pstats,
        "banks": bank_names,
        "voices": {"train": L.TRAIN_VOICES, "CAL": L.CAL_VOICES},
        "split_rule": "CAL = clips voiced by held-out Kokoro voices (af_nova, am_eric, bf_lily); train uses the other 25",
        "licences": LIC,
        "licence_policy": "permissive only (Apache-2.0 / BSD / MIT tools + data); no MMSU audio or text used for data; "
                          "MMSU question strings read only for the equality check below",
        "examples_per_source": examples,
        "examples_per_family": fam_examples,
        "validation": report,
        "leakage_check": leak,
        "tts_stats": [{k: v for k, v in r.items() if k != "fails"} for r in tts_stats],
        "tts_fails": [f for r in tts_stats for f in r.get("fails", [])][:30],
        "proc_errors": [e for r in proc_stats for e in r.get("errors", [])][:30],
        "timing_s": {"tts": round(t_tts), "total": round(time.time() - t0)},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    volume.commit()
    note("DONE", json.dumps({k: summary[k] for k in ("n_items", "audio_hours_used", "validation", "timing_s")}))
    return summary


@app.local_entrypoint()
def smoke():
    s = main.remote(f"{WORK}/smoke/out", f"{WORK}/smoke", 0.02, 2)
    import json

    print(json.dumps({k: s[k] for k in ("n_items", "items_per_question", "clips", "edit_strength", "validation",
                                         "audio_hours_used", "timing_s")}, indent=1))


@app.local_entrypoint()
def smoke2():
    s = main.remote(f"{WORK}/smoke/out2", f"{WORK}/smoke", 0.02, 2)
    import json

    print(json.dumps({k: s[k] for k in ("n_items", "clips", "edit_strength", "validation", "proc_errors")}, indent=1))


@app.local_entrypoint()
def tts_full():
    print(main.remote(OUT, f"{WORK}/full", 1.0, 16, True))


@app.local_entrypoint()
def full():
    s = main.remote(OUT, f"{WORK}/full", 1.0, 16)
    print({k: s[k] for k in ("n_items", "validation", "audio_hours_used", "timing_s")})
