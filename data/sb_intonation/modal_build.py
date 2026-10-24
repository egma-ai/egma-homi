"""Build sb_intonation on Modal: Kokoro TTS on one L4, then CPU shards, items, validation; ::smoke for a small run.

    modal run --detach data/sb_intonation/modal_build.py::full
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
OUT = f"{VOL}/stageb/sb_intonation"
WORK = f"{VOL}/stageb_work/sb_intonation"
SBV_WORK = f"{VOL}/stageb/sb_voice/work2"
MMSU_TRUTH = f"{VOL}/bench_mmsu/truth.jsonl"  # only its "question" field is read, for the non-equality check
ADI_ITEMS = f"{VOL}/adi/items.jsonl"
EXP0_EVAL = f"{VOL}/data/eval_items.jsonl"
KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_REV = "f3ff3571791e39611d31c381e3a41a3af07b4987"
LTR_REV = "0d1718db351207d5a493135a62d5ed8cc4733830"
LIC = {
    "libritts_r": {"what": "LibriTTS-R dev-clean + train-clean-100 utterances (audio + transcripts) via sb_voice's "
                           "intermediate files in /vol/stageb/sb_voice/work2 (read-only)", "repo": "mythicinfinity/libritts_r",
                   "revision": LTR_REV, "licence": "CC BY 4.0"},
    "kokoro": {"what": "Kokoro-82M TTS: code (kokoro==0.9.4, github hexgrad/kokoro) Apache-2.0; weights "
                       f"{KOKORO_REPO}@{KOKORO_REV} Apache-2.0 (HF cardData.license=apache-2.0, not gated)",
               "repo": KOKORO_REPO, "revision": KOKORO_REV, "licence": "Apache-2.0 (code and weights)"},
    "misaki": {"what": "misaki==0.9.4 G2P (word timestamps come from its tokens)", "revision": "0.9.4",
               "licence": "Apache-2.0"},
    "pyworld": {"what": "pyworld==0.3.5 wrapping WORLD (M. Morise): DIO/StoneMask/CheapTrick/D4C analysis, synthesis, "
                        "Harvest for verification; tool only", "revision": "0.3.5",
                "licence": "MIT (pyworld) / modified BSD (WORLD)"},
    "codecs": {"what": "ffmpeg (Debian) mu-law / A-law / GSM / Opus for the baked-in phone channel (data/sb_phone/fx.py)",
               "licence": "tool only"},
    "text": {"what": "pause-meaning sentences, meanings, foils, all question wording: written for this builder "
                     "(data/sb_intonation/sentences.py, labels.py); intonation TTS text = LibriTTS-R transcripts",
             "licence": "own / CC BY 4.0"},
}

volume = modal.Volume.from_name("egma-homi", create_if_missing=False)
_files = {HERE: ["modal_build.py", "build.py", "labels.py", "edit.py", "sentences.py"], STAGEB / "sb_phone": ["fx.py"],
          INFERENCE: ["common.py", "answer_codes.json"]}


def _add(img):
    for d, fs in _files.items():
        for f in fs:
            img = img.add_local_file(d / f, f"/root/{f}")
    img = img.add_local_dir(HERE / "tests", "/root/tests")
    return img


cpu_image = _add(
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "build-essential")
    .pip_install("numpy==2.1.3", "scipy==1.14.1", "cython==3.0.11", "setuptools==75.8.0")
    .pip_install("pyworld==0.3.5", "misaki==0.9.4", "soundfile==0.13.1")
    .pip_install("pytest==8.3.4")
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

app = modal.App("sb-intonation-build")
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


@app.function(image=tts_image, gpu="L4", volumes={VOL: volume}, timeout=3600, retries=1, max_containers=1)
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
    for j in jobs:
        key, voice, lang, speed, text = j["key"], j["voice"], j["lang"], j["speed"], j["text"]
        if voice not in K["voices"]:
            K["voices"][voice] = hf_hub_download(KOKORO_REPO, f"voices/{voice}.pt", revision=KOKORO_REV)
        try:
            chunks, toks, off = [], [], 0.0
            for r in K["pipes"][lang](text, voice=K["voices"][voice], speed=speed, split_pattern=None):
                if r.audio is None:
                    continue
                a = r.audio.detach().cpu().numpy().astype(np.float64)
                for t in r.tokens or []:
                    toks.append([t.text, t.phonemes, None if t.start_ts is None else t.start_ts + off,
                                 None if t.end_ts is None else t.end_ts + off])
                chunks.append(a)
                off += len(a) / 24000.0
            y = resample_poly(np.concatenate(chunks), 2, 3).astype(np.float32)
            arrays[key] = y
            meta[key] = {"tokens": toks, "n_chunks": len(chunks)}
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
    import signal
    import time

    import numpy as np

    import build
    import common
    import fx
    import labels as L
    import sentences as S

    shard_id, work, sub = args  # sub: (k, m) = process every m-th clip starting at k (CPU parallelism within a shard)
    tag = f"{shard_id:03d}_{sub[0]:02d}"
    done = Path(f"{work}/proc/shard_{tag}.json")
    if done.exists():
        return json.loads(done.read_text())["stats"]
    t0 = time.time()
    volume.reload()
    specs = json.loads(Path(f"{work}/plan_shards/shard_{shard_id:03d}.json").read_text())["clips"][sub[0]::sub[1]]
    ents = {f"s{i:03d}": L.parse_entry(e) for i, e in enumerate(S.SENTENCES)}
    tts_meta, z = {}, None
    p_tts = Path(f"{work}/tts/shard_{shard_id:03d}.npz")
    if any(s["tts_key"] for s in specs):
        tts_meta = json.loads(Path(f"{work}/tts/shard_{shard_id:03d}.json").read_text())
        z = np.load(p_tts)
    banks = {}
    arrays, facts, drops, errors, drop_facts, kept_by = {}, {}, collections.Counter(), [], [], collections.Counter()
    planned_by = collections.Counter()

    def _alarm(*_):
        raise TimeoutError("clip watchdog")

    signal.signal(signal.SIGALRM, _alarm)
    for spec in specs:
        k = spec["key"]
        grp = f"{spec['fam']}:{spec['origin']}:{spec['info'].get('pattern', 'pause')}"
        planned_by[grp] += 1
        try:
            signal.alarm(60)
            if spec["tts_key"]:
                tk = spec["tts_key"]
                if tk not in tts_meta or tk not in z.files:
                    drops[f"{grp}:tts_fail"] += 1
                    continue
                x = np.asarray(z[tk], np.float32)
                toks = tts_meta[tk]["tokens"]
            else:
                r = spec["rec"]
                if r["bank"] not in banks:
                    banks[r["bank"]] = common.AudioBank(SBV_WORK, r["bank"])
                x = banks[r["bank"]].get(r["key"])
                toks = None
            if spec["fam"] == "pause":
                y, f = build.process_pause(spec, x, toks, ents[spec["info"]["sid"]], fx)
            else:
                y, f = build.process_contour(spec, x, fx)
        except Exception as e:  # noqa: BLE001
            errors.append([k, repr(e)[:200]])
            drops[f"{grp}:error"] += 1
            continue
        finally:
            signal.alarm(0)
        if y is None:
            drops[f"{grp}:{f['drop']}"] += 1
            if len(drop_facts) < 30:
                drop_facts.append([k, f])
            continue
        assert np.isfinite(y).all() and np.abs(y).max() <= 0.99 and 1.0 <= len(y) / 16000 <= 30.0, k
        arrays[k] = y
        facts[k] = f
        kept_by[grp] += 1
    p = Path(f"{work}/proc/shard_{tag}.npz")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.rename(p)
    stats = {"shard": tag, "kept": len(arrays), "planned_by": dict(planned_by), "kept_by": dict(kept_by),
             "drops": dict(drops), "errors": errors[:20], "drop_facts": drop_facts, "elapsed_s": round(time.time() - t0, 1)}
    done.write_text(json.dumps({"stats": stats, "facts": facts}, default=str))
    volume.commit()
    return stats


@app.function(image=cpu_image, volumes={VOL: volume}, timeout=3 * 3600, cpu=4, memory=32768)
def main(out_dir: str, work: str, scale: float = 1.0, n_shards: int = 12, n_sub: int = 2) -> dict:
    import collections
    import json
    import random
    import subprocess
    import time

    import numpy as np

    import build
    import common
    import labels as L
    import sentences as S

    t0 = time.time()
    note = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)  # noqa: E731
    out, wk = Path(out_dir), Path(work)
    assert not (out / "summary.json").exists(), "summary.json exists; final folder is write-once"
    wk.mkdir(parents=True, exist_ok=True)
    # Unit tests inside the container, where the MMSU check can read the real question strings
    sel = [] if scale >= 0.5 else ["-k", "not all_sentences_parse"]  # smoke may run before all sentences exist
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "/root/tests", *sel], capture_output=True, text=True, cwd="/root")
    note("pytest", r.returncode, r.stdout[-600:])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]

    recs = []
    for p in sorted(Path(SBV_WORK).glob("libritts_*.jsonl")):
        for line in p.read_text().splitlines():
            rr = json.loads(line)
            rr["bank"] = p.stem
            recs.append(rr)
    sbv_cal = set(json.loads(Path(f"{VOL}/stageb/sb_voice/cal_speakers.json").read_text()))
    clips, tts, pstats = build.plan(common.SEED, scale, recs, exclude_speakers=sbv_cal)
    note("plan", len(clips), collections.Counter((c["fam"], c["origin"]) for c in clips), "tts jobs", len(tts), pstats)
    shards_c, shards_t = build.shard_plan(clips, tts, n_shards)
    (wk / "plan_shards").mkdir(parents=True, exist_ok=True)
    for i in range(n_shards):
        (wk / "plan_shards" / f"shard_{i:03d}.json").write_text(json.dumps({"clips": shards_c[i]}))
    volume.commit()

    # LibriTTS-R-only work can start right away; shards with TTS start when their TTS shard is done
    handles, started = [], set()

    def start(i):
        if i not in started:
            started.add(i)
            for k in range(n_sub):
                handles.append(proc_shard.spawn((i, work, (k, n_sub))))

    todo = [(i, shards_t[i], work) for i in range(n_shards) if shards_t[i]]
    for i in range(n_shards):
        if not shards_t[i]:
            start(i)
    tts_stats = []
    for rr in tts_shard.map(todo, order_outputs=False):
        note("tts", {k: v for k, v in rr.items() if k != "fails"}, "fails", rr.get("fails", [])[:3])
        tts_stats.append(rr)
        start(rr["shard"])
    for i in range(n_shards):
        start(i)
    proc_stats = []
    for h in handles:
        st = h.get()
        note("proc", st["shard"], st["kept"], st["drops"], st["errors"][:2])
        for dfx in st.get("drop_facts", [])[:4]:
            note("dropfact", json.dumps(dfx, default=str)[:400])
        proc_stats.append(st)
    volume.reload()

    # Assemble
    ents = {f"s{i:03d}": L.parse_entry(e) for i, e in enumerate(S.SENTENCES)}
    spec_by = {c["key"]: c for c in clips}
    facts, audio = {}, {}
    for i in range(n_shards):
        for k in range(n_sub):
            tag = f"{i:03d}_{k:02d}"
            facts.update(json.loads((wk / "proc" / f"shard_{tag}.json").read_text())["facts"])
            with np.load(wk / "proc" / f"shard_{tag}.npz") as z:
                for kk in z.files:
                    audio[kk] = z[kk]
    bank_of = {}
    for k in audio:
        c = spec_by[k]
        bank_of[k] = f"{L.BUILDER}_cal" if c["split"] == "CAL" else (
            f"{L.BUILDER}_pause" if c["fam"] == "pause" else f"{L.BUILDER}_inton")
    items = []
    for k in sorted(audio):
        items += build.items_for(spec_by[k], facts[k], ents, common.make_item, bank_of[k])
    n_raw = len(items)
    items, bal = L.balance(items, common.SEED)
    items = L.cap_per_clip(items, 4)
    used = {it["audio"]["key"] for it in items}
    note("items", n_raw, "->", len(items), "clips used", len(used))

    # Write banks (only clips that have items)
    adir = out / "audio"
    bank_index, durations = {}, {}
    for b in sorted(set(bank_of.values())):
        clips_b = {k: audio[k] for k in sorted(audio) if bank_of[k] == b and k in used}
        common.write_audio_bank(adir, b, clips_b)
        bank_index[b] = json.loads((adir / f"{b}.json").read_text())
        for k, x in clips_b.items():
            durations[k] = len(x) / 16000
    # re-read the written banks for a sample
    rng = random.Random(1)
    reread_ok = 0
    sample = rng.sample(items, min(300, len(items)))
    bk = {b: common.AudioBank(adir, b) for b in bank_index}
    for it in sample:
        x = bk[it["audio"]["bank"]].get(it["audio"]["key"])
        reread_ok += int(np.allclose(x, audio[it["audio"]["key"]]))
    # independent re-measurement of a sample of written clips (contours + pauses)
    import edit

    recheck = collections.Counter()
    pool_rc = [i for i in items if i["meta"]["qkind"] in ("inton_choice", "pm_meaning")]
    for it in rng.sample(pool_rc, min(160, len(pool_rc))):
        x = bk[it["audio"]["bank"]].get(it["audio"]["key"])
        if it["meta"]["qkind"] == "inton_choice":
            cls, _ = edit.measure(x, facts[it["audio"]["key"]]["f0_med"])
            recheck["inton_same"] += int(cls == it["meta"]["contour"])
            recheck["inton_n"] += 1
        else:
            sil = L.silences(x, L.PAUSE_SIL_MIN)
            recheck["pause_same_count"] += int(len(sil) == len(it["meta"]["pause_boundaries"]))
            recheck["pause_n"] += 1

    mmsu_q = []
    if Path(MMSU_TRUTH).exists():
        for line in Path(MMSU_TRUTH).read_text().splitlines():
            q = json.loads(line).get("question")
            if isinstance(q, str):
                mmsu_q.append(q)
    sent_texts = [" ".join(e["words"]) for e in ents.values()]
    errs, mm = build.validate(items, bank_index, durations, mmsu_q, sent_texts)
    if reread_ok != len(sample):
        errs["bank_reread"] = len(sample) - reread_ok
    # Leakage against the evaluation items (speakers and source ids)
    leak = {"adi_items_exists": Path(ADI_ITEMS).exists()}
    ours_src = {s for it in items for s in it["meta"].get("source_ids") or []}
    ours_spk = {it["meta"].get("speaker") for it in items}
    for name, p in (("adi", ADI_ITEMS), ("exp0_eval", EXP0_EVAL)):
        hits_src, hits_spk = 0, 0
        if Path(p).exists():
            for line in Path(p).read_text().splitlines():
                m = json.loads(line).get("meta") or {}
                vals = {str(v) for kk, v in m.items() if isinstance(v, (str, int)) and any(
                    t in kk for t in ("source", "id", "file", "utt", "recording"))}
                for v in m.get("source_ids") or []:
                    vals.add(str(v))
                hits_src += len(vals & ours_src)
                sp = m.get("speaker")
                if sp is not None and str(sp) in ours_spk:
                    hits_spk += 1
        leak[f"{name}_source_id_hits"] = hits_src
        leak[f"{name}_speaker_hits"] = hits_spk
    if any(v for kk, v in leak.items() if kk.endswith("_hits")):
        errs["leakage"] = 1
    note("validation", errs, mm, leak, recheck)
    assert not errs, errs

    tr = [it for it in items if it["set"] == "train"]
    cal = [it for it in items if it["set"] == "CAL"]
    common.write_jsonl(out / "items_train.jsonl", tr)
    common.write_jsonl(out / "items_cal.jsonl", cal)

    # Summary
    plan_by = collections.Counter()
    kept_by = collections.Counter()
    drops = collections.Counter()
    for st in proc_stats:
        plan_by.update(st["planned_by"])
        kept_by.update(st["kept_by"])
        drops.update(st["drops"])
    f0_ver = {}
    for g in sorted(plan_by):
        if g.startswith(("inton", "final")):
            f0_ver[g] = {"planned": plan_by[g], "kept": kept_by[g],
                         "drop_rate": round(1 - kept_by[g] / max(1, plan_by[g]), 4)}
    pl_c = sum(v for g, v in plan_by.items() if g.startswith(("inton", "final")))
    kp_c = sum(v for g, v in kept_by.items() if g.startswith(("inton", "final")))
    pl_p = sum(v for g, v in plan_by.items() if g.startswith("pause"))
    kp_p = sum(v for g, v in kept_by.items() if g.startswith("pause"))
    cnt = collections.Counter((it["source"], it["meta"]["qkind"], it["set"]) for it in items)
    by_src = collections.defaultdict(lambda: collections.Counter())
    for it in items:
        by_src[it["source"]][it["set"]] += 1
    by_q = collections.Counter((it["meta"]["qtype"], it["set"]) for it in items)
    cls_bal = collections.Counter(it["meta"]["contour"] for it in items if it["meta"]["qkind"] == "inton_choice")
    cls_bal_src = collections.Counter((it["source"], it["meta"]["contour"]) for it in items if it["meta"]["qkind"] == "inton_choice")
    hours = sum(durations.values()) / 3600
    ex = {}
    for it in items:
        lst = ex.setdefault(it["source"], [])
        if len(lst) < 3 and it["meta"]["qkind"] not in [e["qkind"] for e in lst]:
            lst.append({"qkind": it["meta"]["qkind"], "question": it["question"], "options": it["options"],
                        "gold": it["gold"]})
    used_sent = collections.Counter(it["meta"]["sentence_id"] for it in items if it["meta"].get("sentence_id"))
    used_pl = {(it["meta"]["sentence_id"], it["meta"]["placement"]) for it in items if it["meta"].get("sentence_id")}
    summary = {
        "builder": L.BUILDER, "readme": "data/sb_intonation/README.md",
        "n_train": len(tr), "n_cal": len(cal), "cal_frac": round(len(cal) / max(1, len(items)), 4),
        "counts_by_source": {k: dict(v) for k, v in sorted(by_src.items())},
        "counts_by_source_qkind_split": {"/".join(k): v for k, v in sorted(cnt.items())},
        "counts_by_qtype_split": {"/".join(k): v for k, v in sorted(by_q.items())},
        "contour_class_balance_inton_choice": dict(cls_bal),
        "contour_class_balance_by_source": {"/".join(k): v for k, v in sorted(cls_bal_src.items())},
        "f0_verification": {"per_group": f0_ver, "planned": pl_c, "kept": kp_c,
                            "drop_rate": round(1 - kp_c / max(1, pl_c), 4)},
        "pause_verification": {"planned": pl_p, "kept": kp_p, "drop_rate": round(1 - kp_p / max(1, pl_p), 4)},
        "drop_reasons": dict(sorted(drops.items())),
        "pause_sentences": {"written": len(ents), "placements_written": sum(len(e["placements"]) for e in ents.values()),
                            "by_type": dict(collections.Counter(e["type"] for e in ents.values())),
                            "sentences_in_items": len(used_sent), "placements_in_items": len(used_pl),
                            "with_foil": sum(1 for e in ents.values() if e["foil"])},
        "audio_hours": round(hours, 3), "clips": len(used), "banks": {b: len(v) for b, v in bank_index.items()},
        "clip_duration_s": {"min": round(min(durations.values()), 2), "max": round(max(durations.values()), 2),
                            "median": round(float(np.median(list(durations.values()))), 2)},
        "phone_fx_clip_frac": round(sum(1 for k in used if spec_by[k]["phone"]) / max(1, len(used)), 3),
        "licences": LIC,
        "plan_stats": pstats, "tts": {"jobs": len(tts), "fails": sum(len(r.get("fails", [])) for r in tts_stats)},
        "balance_report_sample": dict(list(bal.items())[:40]),
        "validation": {"errors": errs, "mmsu": mm, "leakage": leak, "bank_reread_ok": f"{reread_ok}/{len(sample)}",
                       "final_bank_recheck": dict(recheck)},
        "examples": ex, "elapsed_s": round(time.time() - t0),
        "paths": {"out": str(out), "work": str(wk)},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    volume.commit()
    note("DONE", len(tr), len(cal), round(hours, 2))
    return {"n_train": len(tr), "n_cal": len(cal), "hours": round(hours, 2), "f0_drop": summary["f0_verification"]["drop_rate"]}


@app.local_entrypoint()
def smoke(tag: str = "smoke", scale: float = 0.03):
    print(main.remote(f"{WORK}/{tag}/out", f"{WORK}/{tag}", scale=scale, n_shards=3, n_sub=2))


@app.local_entrypoint()
def full():
    print(main.remote(OUT, f"{WORK}/full", scale=1.0, n_shards=12, n_sub=3))
