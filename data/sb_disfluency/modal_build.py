"""Build sb_disfluency on Modal: plan, Kokoro TTS on one L4, CPU post-processing shards, banks, items, validation.

    modal run --detach data/sb_disfluency/modal_build.py::full
"""
from __future__ import annotations

import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
STAGEB = HERE.parent
INFERENCE = STAGEB.parent / "inference"
for p in (INFERENCE, STAGEB / "sb_phone", STAGEB / "sb_phon", HERE):
    sys.path.insert(0, str(p))

VOL = "/vol"
OUT = f"{VOL}/stageb/sb_disfluency"
WORK = f"{VOL}/stageb_work/sb_disfluency"
KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_REV = "f3ff3571791e39611d31c381e3a41a3af07b4987"
CMU_SHA = "74790861f652b15e4ac49015a90074ad62a27690"
CMU_URL = f"https://raw.githubusercontent.com/cmusphinx/cmudict/{CMU_SHA}/cmudict.dict"
LIC = {
    "kokoro_weights": f"{KOKORO_REPO}@{KOKORO_REV} (kokoro-v1_0.pth + voices/*.pt): Apache-2.0 (HF cardData.license="
                      "apache-2.0, not gated; same pin as data/sb_calls and data/sb_phon). All audio is Kokoro output.",
    "kokoro_code": "kokoro==0.9.4 (PyPI, github.com/hexgrad/kokoro, Apache-2.0)",
    "misaki": "misaki==0.9.4 G2P + us_gold/gb_gold lexicons (Apache-2.0). The espeak-ng fallback G2P is disabled "
              "(g2p.fallback=None): out-of-lexicon words are not voiced and such clips are dropped, so no GPL "
              "component produces any audio.",
    "spacy_model": "en_core_web_sm 3.8.0 (MIT), used by misaki for POS tags",
    "cmudict": f"CMU Pronouncing Dictionary, github.com/cmusphinx/cmudict@{CMU_SHA}: BSD-2-clause style; syllable "
               "counts, lexical stress, rhyme keys",
    "codecs": "ffmpeg (Debian build) codecs for the baked-in phone channel via data/sb_phone/fx.py; tool only",
    "text": "all sentences, verse word banks, templates and questions written in code (data/sb_disfluency/lexbank.py, "
            "build.py); no poems, lyrics or benchmark text",
}

volume = modal.Volume.from_name("homi", create_if_missing=False)
_files = {HERE: ["modal_build.py", "build.py", "lexbank.py"], STAGEB / "sb_phon": ["lexicon.py"],
          STAGEB / "sb_phone": ["fx.py"], INFERENCE: ["common.py", "answer_codes.json"]}


def _add(img):
    for d, fs in _files.items():
        for f in fs:
            img = img.add_local_file(d / f, f"/root/{f}")
    return img


cpu_image = _add(
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("numpy==2.1.3", "scipy==1.14.1", "misaki==0.9.4", "soundfile==0.13.1")
    .env({"HF_HOME": f"{VOL}/hf_cache"})
)

tts_image = _add(
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.7.1", "kokoro==0.9.4", "misaki[en]==0.9.4", "transformers==4.51.3", "huggingface_hub==0.30.2",
        "numpy==1.26.4", "scipy==1.13.1", "soundfile==0.13.1",
        "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
    )
    .env({"HF_HOME": f"{VOL}/hf_cache"})
)

app = modal.App("sb-disfluency-build")
_TTS = {}


def _kokoro():
    if not _TTS:
        from huggingface_hub import hf_hub_download
        from kokoro import KModel, KPipeline

        cfg = hf_hub_download(KOKORO_REPO, "config.json", revision=KOKORO_REV)
        mdl = hf_hub_download(KOKORO_REPO, "kokoro-v1_0.pth", revision=KOKORO_REV)
        model = KModel(config=cfg, model=mdl).to("cuda").eval()
        pipes = {lc: KPipeline(lang_code=lc, model=model) for lc in "ab"}
        n_off = 0
        for p in pipes.values():
            if hasattr(p.g2p, "fallback"):
                p.g2p.fallback = None  # no espeak-ng (GPL) G2P: OOV words -> unk -> clip dropped
                n_off += 1
        _TTS["pipes"] = pipes
        _TTS["fallback_disabled"] = n_off
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
    for key, voice, lang, speed, text in jobs:
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
    return {"shard": shard_id, "n": len(jobs), "ok": len(arrays), "fails": fails, "elapsed_s": round(time.time() - t0, 1),
            "espeak_fallback_disabled_pipes": K["fallback_disabled"]}


@app.function(image=cpu_image, volumes={VOL: volume}, timeout=3600, retries=1, cpu=2, memory=8192, max_containers=16)
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
                if len(drop_facts) < 40:
                    drop_facts.append([k, spec["text"], f])
                continue
            assert np.isfinite(y).all() and np.abs(y).max() <= 0.99 and 1.0 <= len(y) / 16000 <= 30.0, k
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
def main(out_dir: str, work: str, scale: float = 1.0, n_shards: int = 8) -> dict:
    import collections
    import json
    import random
    import time
    import urllib.request

    import numpy as np

    import build
    import common
    import lexbank as LB
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
    plan, pstats, ctx = build.plan(common.SEED, scale, cmu, us_gold, gb_gold)
    note("plan", len(plan), collections.Counter(c["fam"] for c in plan), json.dumps(pstats)[:600])
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
    volume.reload()
    proc_stats = []
    for r in proc_shard.map([(i, work) for i in range(n_shards)], order_outputs=False):
        note("proc", r["shard"], r["kept"], r["drops"], r["errors"][:2])
        for dfx in r.get("drop_facts", [])[:6]:
            note("dropfact", json.dumps(dfx)[:400])
        proc_stats.append(r)
    volume.reload()

    audio_dir = out / "audio"
    spec_by = {c["key"]: c for c in plan}
    facts, bank_of, durs, cal, bank_names = {}, {}, [], {}, []
    for i in range(n_shards):
        fj = json.loads((wk / "proc" / f"shard_{i:03d}.json").read_text())["facts"]
        tr = {}
        with np.load(wk / "proc" / f"shard_{i:03d}.npz") as z:
            for k in z.files:
                (cal if spec_by[k]["split"] == "CAL" else tr)[k] = z[k]
                durs.append(len(z[k]) / 16000)
        name = f"sb_disfluency_train_{i:02d}"
        common.write_audio_bank(audio_dir, name, tr)
        bank_names.append(name)
        for k in tr:
            bank_of[k] = name
        facts.update(fj)
        note("bank", name, len(tr))
    common.write_audio_bank(audio_dir, "sb_disfluency_cal", cal)
    bank_names.append("sb_disfluency_cal")
    for k in cal:
        bank_of[k] = "sb_disfluency_cal"
    recs = sorted([{**spec_by[k], "facts": facts[k], "bank": b} for k, b in bank_of.items()], key=lambda r: r["key"])

    items = []
    for r in recs:
        items += build.items_for(r, common.make_item, common.SEED, ctx)
    n_pre = collections.Counter((it["meta"]["qname"], it["set"]) for it in items)
    items = build.balance(items, common.SEED)
    train = [it for it in items if it["set"] == "train"]
    calit = [it for it in items if it["set"] == "CAL"]
    common.write_jsonl(out / "items_train.jsonl", train)
    common.write_jsonl(out / "items_cal.jsonl", calit)
    volume.commit()
    note("items", len(train), len(calit))

    # validation, on the files reloaded from disk
    bank_index = {b: json.loads((audio_dir / f"{b}.json").read_text()) for b in bank_names}
    reloaded = common.read_jsonl(out / "items_train.jsonl") + common.read_jsonl(out / "items_cal.jsonl")
    mmsu_q = []
    mp = Path(f"{VOL}/bench_mmsu/truth.jsonl")
    if mp.exists():  # only the "question" field, for the non-equality check
        mmsu_q = [json.loads(line)["question"] for line in mp.read_text().splitlines() if line.strip()]
    report = build.validate(reloaded, bank_index, mmsu_q)
    mmn = {build.norm_q(x) for x in mmsu_q}
    tpl_hits = [t for v in build.Q.values() for t in v if t in set(mmsu_q) or build.norm_q(t) in mmn]
    report["mmsu_questions_checked"] = len(mmsu_q)
    report["mmsu_template_equal"] = len(tpl_hits)
    assert mmsu_q, "MMSU question list missing: non-equality check could not run"
    assert report.get("mmsu_question_equal", 0) == 0 and not tpl_hits
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
    assert not report["cal_train_voice_overlap"]
    rng = random.Random(1)
    sample = rng.sample(reloaded, min(400, len(reloaded)))
    banks = {b: common.AudioBank(audio_dir, b) for b in bank_names}
    sd = [len(common.item_audio(it, banks)) / 16000 for it in sample]
    report["sample_loaded"] = len(sample)
    report["sample_dur_s"] = [round(min(sd), 2), round(float(np.mean(sd)), 2), round(max(sd), 2)]
    g = collections.defaultdict(collections.Counter)
    for it in reloaded:
        g[(it["meta"]["qname"], it["set"])][it["options"][it["gold"]] if it["meta"]["qtype"] == "noul"
                                            else it["meta"].get("answer_class")] += 1
    label_balance = {f"{q}|{s}": dict(c) for (q, s), c in sorted(g.items())}
    bad_noul = {k: v for k, v in label_balance.items() if k.split("|")[0] in
                {it["meta"]["qname"] for it in reloaded if it["meta"]["qtype"] == "noul"} and
                len(set(v.values())) > 1}
    report["noul_unbalanced_groups"] = bad_noul
    assert not bad_noul and not {k: v for k, v in report.items() if isinstance(v, int) and k not in
                                 ("mmsu_questions_checked", "mmsu_template_equal", "max_items_per_clip", "sample_loaded")
                                 and v}, report

    def cnt(rows, f):
        return dict(sorted(collections.Counter(f(r) for r in rows).items()))

    dis = [it for it in reloaded if it["meta"]["family"] == "disfl"]
    ver = [it for it in reloaded if it["meta"]["family"] == "verse"]
    dclips = [spec_by[k] for k in used if spec_by[k]["fam"] == "disfl"]
    vclips = [spec_by[k] for k in used if spec_by[k]["fam"] == "verse"]
    vc = [it for it in ver if it["meta"]["qname"] == "verse_cont"]
    disfl_stats = {
        "clips_per_combination": cnt(dclips, lambda c: build.combo_key(c["info"]["types"])),
        "clips_per_n_types": cnt(dclips, lambda c: len(c["info"]["types"])),
        "combo_items_per_answer": cnt([it for it in dis if it["meta"]["qname"] == "disfl_combo"],
                                      lambda it: it["meta"]["combo"]),
        "combo_items_n_options": cnt([it for it in dis if it["meta"]["qname"] == "disfl_combo"],
                                     lambda it: len(it["options"])),
        "combo_name_styles": cnt([it for it in dis if it["meta"]["qname"] == "disfl_combo"],
                                 lambda it: it["meta"]["name_style"]),
        "fillers_used": dict(collections.Counter(f for c in dclips for f in c["info"]["fillers"])),
        "markers_used": dict(collections.Counter(m for c in dclips for m in c["info"]["markers"])),
        "edit_terms_used": dict(collections.Counter(c["info"]["edit_term"] for c in dclips if c["info"]["edit_term"])),
        "restart_kinds": dict(collections.Counter(c["info"]["restart"]["kind"] for c in dclips if c["info"]["restart"])),
        "restart_fragment_cut": sum(1 for c in dclips if facts[c["key"]].get("restart_cut")),
        "pauses_inserted": dict(collections.Counter(p[0] for c in dclips for p in facts[c["key"]].get("pauses", []))),
        "label_rules": {
            "filled": "a standalone 'uh' / 'um' / 'er' hesitation token (TTS token >= 0.1 s, vowel checked)",
            "marker": "'well' / 'so' / 'you know' (sentence-initial) or 'you know' / 'like' (mid-sentence) inserted as "
                      "a discourse marker; base sentences never contain these words",
            "restart": "speaker breaks off ('—', optional extra pause, 40% of fragments cut mid-word) and either "
                       "repeats the start of a phrase (1-2 words) or abandons a different start; no editing term",
            "editing": "reparandum + explicit editing term (I mean / sorry / or rather / rather / no wait / no, sorry / "
                       "sorry, I mean) + corrected word; the repair never repeats other words",
            "none": "the base sentence alone; plain pauses (0.15-0.45 s) are also inserted between fluent words in "
                    "~35% of all clips so silence alone is not a disfluency cue"},
    }
    verse_stats = {
        "clips_per_meter": cnt(vclips, lambda c: c["info"]["meter"]),
        "clips_per_family": cnt(vclips, lambda c: c["info"]["family"]),
        "alt_meter_kind": cnt(vclips, lambda c: c["info"]["alt_how"]),
        "slant_rhyme_distractor_share": round(sum(c["info"]["slant"] for c in vclips) / max(1, len(vclips)), 3),
        "cont_items_n_options": cnt(vc, lambda it: len(it["options"])),
        "cont_distractor_types": dict(collections.Counter(d for it in vc for d in it["meta"]["distractors"])),
        "rules": {
            "meters": {m: f"{f} {p}" for m, (f, p) in LB.METERS.items()},
            "stress": "line pattern = concatenation of word shapes: monosyllabic function words unstressed, monosyllabic "
                      "content words stressed, polysyllables = CMUdict stress digits (all pronunciations agree, no "
                      "secondary stress); spoken lines re-checked on Kokoro's phonemes (syllables + primary stress)",
            "rhyme": "rime from the last primary-stressed vowel; rhyme iff CMUdict, misaki-US and misaki-GB rimes all "
                     "agree; identical rhymes (same pre-vowel phone, e.g. light / delight) never paired; non-rhyme "
                     "iff all three rimes differ",
            "candidates": "2x2 design {M, M'} x {R, R'}: (M,R) correct; (M,R') rhyme broken; (M',R) meter broken "
                          "(M' = same syllable count with a different stress pattern, or a count differing by >= 2); "
                          "(M',R') both broken. 4-option items use all four cells, 3-option items the correct cell + "
                          "2 random cells, so no text-only 'odd one out' shortcut identifies the answer",
        },
    }
    examples = {}
    for s in sorted({it["source"] for it in train}):
        rows = [it for it in train if it["source"] == s]
        examples[s] = [{"question": it["question"], "options": it["options"], "gold": it["gold"],
                        "text": it["meta"]["text"]} for it in random.Random(s).sample(rows, min(3, len(rows)))]
    q_examples = {}
    for qn in sorted({it["meta"]["qname"] for it in train}):
        rows = [it for it in train if it["meta"]["qname"] == qn]
        q_examples[qn] = [{"question": it["question"], "options": it["options"], "gold": it["gold"],
                           "text": it["meta"]["text"]} for it in random.Random(qn).sample(rows, min(2, len(rows)))]
    leak = {}
    ours = set(bank_of) | set(bank_names)
    for name, path in (("adi", f"{VOL}/adi/items.jsonl"), ("exp0_eval", f"{VOL}/data/eval_items.jsonl")):
        p = Path(path)
        if not p.exists():
            leak[name] = "file not present"
            continue
        hits, n = 0, 0
        for it in common.read_jsonl(p):
            n += 1
            a = it.get("audio") or {}
            hits += bool({str(a.get("bank")), str(a.get("key"))} & ours)
        leak[name] = {"items_checked": n, "items_referencing_our_audio": hits}
    leak["note"] = "all audio is newly synthesised Kokoro speech from text written in code; no corpus audio is used"
    drops = collections.Counter()
    for r in proc_stats:
        drops.update(r["drops"])
    hours = sum(bank_index[bank_of[k]][k][1] for k in used) / 16000 / 3600
    summary = {
        "builder": "sb_disfluency",
        "out_dir": out_dir,
        "n_items": {"train": len(train), "CAL": len(calit)},
        "n_clips_used": cnt([spec_by[k] for k in used], lambda c: f"{c['fam']}|{c['split']}"),
        "items_per_source": {s: cnt([it for it in reloaded if it["set"] == s], lambda it: it["source"]) for s in ("train", "CAL")},
        "items_per_question": {s: cnt([it for it in reloaded if it["set"] == s], lambda it: it["meta"]["qname"]) for s in ("train", "CAL")},
        "items_per_qtype": {s: cnt([it for it in reloaded if it["set"] == s], lambda it: it["meta"]["qtype"]) for s in ("train", "CAL")},
        "items_per_source_qtype": {s: cnt([it for it in reloaded if it["set"] == s],
                                          lambda it: f"{it['source']}|{it['meta']['qtype']}") for s in ("train", "CAL")},
        "items_before_balance": {f"{a}|{b}": v for (a, b), v in sorted(n_pre.items())},
        "phone_fx_share_items": round(sum(it["meta"].get("phone_fx") is not None for it in reloaded) / len(reloaded), 3),
        "audio_hours_used": round(hours, 3),
        "audio_hours_banked": round(sum(durs) / 3600, 3),
        "clip_dur_s": {"min": round(min(durs), 2), "mean": round(float(np.mean(durs)), 2), "max": round(max(durs), 2)},
        "clips": {"planned": cnt(plan, lambda c: c["fam"]), "kept": cnt([spec_by[k] for k in bank_of], lambda c: c["fam"]),
                  "drops": dict(sorted(drops.items()))},
        "disfluency": disfl_stats,
        "verse": verse_stats,
        "label_balance": label_balance,
        "plan_stats": pstats,
        "banks": bank_names,
        "voices": {"train": L.TRAIN_VOICES, "CAL": L.CAL_VOICES},
        "split_rule": "CAL = clips voiced by held-out Kokoro voices (af_nova, am_eric, bf_lily); train uses the other 25",
        "licences": LIC,
        "licence_policy": "permissive only (Apache-2.0 / BSD / MIT code, weights and data); no corpus audio; no MMSU "
                          "audio or text used for data; MMSU 'question' strings read only for the non-equality check",
        "examples_per_source": examples,
        "examples_per_question": q_examples,
        "validation": report,
        "leakage_check": leak,
        "tts_stats": [{k: v for k, v in r.items() if k != "fails"} for r in tts_stats],
        "tts_fails": [f for r in tts_stats for f in r.get("fails", [])][:30],
        "proc_errors": [e for r in proc_stats for e in r.get("errors", [])][:30],
        "drop_examples": [d for r in proc_stats for d in r.get("drop_facts", [])][:20],
        "timing_s": {"tts": round(t_tts), "total": round(time.time() - t0)},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    volume.commit()
    note("done", json.dumps({k: summary[k] for k in ("n_items", "audio_hours_used", "validation", "timing_s")}))
    return summary


@app.local_entrypoint()
def smoke():
    import json

    s = main.remote(f"{WORK}/smoke/out", f"{WORK}/smoke", 0.04, 2)
    print(json.dumps({k: s[k] for k in ("n_items", "items_per_question", "clips", "disfluency", "verse", "validation",
                                         "audio_hours_used", "timing_s", "drop_examples", "examples_per_question")},
                     indent=1)[:20000])


@app.local_entrypoint()
def full():
    import json

    s = main.remote(OUT, f"{WORK}/full", 1.0, 8)
    print(json.dumps({k: s[k] for k in ("n_items", "items_per_source", "clips", "validation", "audio_hours_used",
                                         "timing_s")}, indent=1))
