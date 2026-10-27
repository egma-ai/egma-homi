"""Build sb_calls on Modal: Kokoro-82M TTS on GPU, clip assembly, items, validation and summary in one job.

    modal run --detach data/sb_calls/modal_build.py::full
"""
from __future__ import annotations

import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
INFERENCE = HERE.parent.parent / "inference"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(INFERENCE))

VOL = "/vol"
OUT = f"{VOL}/stageb/sb_calls"
WORK = f"{OUT}/work"
KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_REV = "f3ff3571791e39611d31c381e3a41a3af07b4987"  # pinned HF commit (apache-2.0)
KOKORO_REF = f"{KOKORO_REPO}@{KOKORO_REV} (Apache-2.0)"
BANK_CLIPS = 2500  # clips per audio bank (keeps each bank well under 4 GB)

volume = modal.Volume.from_name("homi", create_if_missing=False)

_local_files = ["modal_build.py", "build.py", "scripts.py", "audio_fx.py"]


def _add_code(img: modal.Image) -> modal.Image:
    img = img.add_local_file(INFERENCE / "common.py", "/root/common.py")
    img = img.add_local_file(INFERENCE / "answer_codes.json", "/root/answer_codes.json")
    for f in _local_files:
        if (HERE / f).exists():
            img = img.add_local_file(HERE / f, f"/root/{f}")
    return img


cpu_image = _add_code(
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy==2.3.3", "scipy==1.16.2", "pyarrow==21.0.0", "soundfile==0.13.1", "huggingface_hub==0.35.3")
    .env({"HF_HOME": f"{VOL}/hf_cache"})
)

tts_image = _add_code(
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng", "git")
    .pip_install(
        "torch==2.7.1",
        "kokoro==0.9.4",
        "misaki[en]==0.9.4",
        "transformers==4.51.3",
        "huggingface_hub==0.30.2",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "soundfile==0.13.1",
        "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
    )
    .env({"HF_HOME": f"{VOL}/hf_cache"})
)

app = modal.App("sb-calls-build")


@app.function(image=tts_image, gpu="L4", volumes={VOL: volume}, timeout=3600, retries=2, max_containers=12)
def tts_shard(args: tuple) -> dict:
    """TTS for a shard of jobs [(key, voice, speed, [texts])] -> <out>/work/tts/shard_XXX.npz ('<key>|<seg idx>')."""
    import time

    import numpy as np
    from huggingface_hub import hf_hub_download
    from kokoro import KModel, KPipeline
    from scipy.signal import resample_poly

    shard_id, jobs, out_dir = args
    path = Path(f"{out_dir}/work/tts/shard_{shard_id:03d}.npz")
    if path.exists():
        return {"shard": shard_id, "skipped": True}
    t0 = time.time()
    cfg = hf_hub_download(KOKORO_REPO, "config.json", revision=KOKORO_REV)
    mdl = hf_hub_download(KOKORO_REPO, "kokoro-v1_0.pth", revision=KOKORO_REV)
    model = KModel(config=cfg, model=mdl).to("cuda").eval()
    pipes = {lc: KPipeline(lang_code=lc, model=model) for lc in "ab"}
    vpaths = {}
    arrays, fails, secs = {}, [], 0.0
    for key, voice, speed, texts in jobs:
        if voice not in vpaths:
            vpaths[voice] = hf_hub_download(KOKORO_REPO, f"voices/{voice}.pt", revision=KOKORO_REV)
        pipe = pipes["b" if voice.startswith("b") else "a"]
        try:
            segs = []
            for t in texts:
                chunks = [r.audio.detach().cpu().numpy() for r in pipe(t, voice=vpaths[voice], speed=speed)
                          if r.audio is not None]
                x = np.concatenate(chunks).astype(np.float64)
                y = resample_poly(x, 2, 3).astype(np.float32)  # 24 kHz -> 16 kHz
                if len(y) < 1600:
                    raise ValueError(f"too short: {t!r}")
                segs.append(y)
            for i, y in enumerate(segs):
                arrays[f"{key}|{i}"] = y
                secs += len(y) / 16000
        except Exception as e:  # noqa: BLE001
            fails.append([key, repr(e)[:200]])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.rename(path)
    volume.commit()
    return {"shard": shard_id, "n": len(jobs), "fails": fails, "speech_s": round(secs, 1),
            "elapsed_s": round(time.time() - t0, 1)}


@app.function(image=cpu_image, volumes={VOL: volume}, timeout=4 * 3600, memory=32768, cpu=4)
def main(out_dir: str, scale: float = 1.0, n_shards: int = 12) -> dict:
    import collections
    import json
    import random
    import time

    import numpy as np

    import build
    import common
    from audio_fx import SR, assemble
    from scripts import CLASSES, CAL_VOICES, SPEECH_CLASSES, TRAIN_VOICES, build_plan

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("summary.json", "items_train.jsonl", "items_cal.jsonl"):
        (out / f).unlink(missing_ok=True)
    volume.commit()

    # 1. plan
    plan = build_plan(common.SEED, scale)
    plan_by_key = {p["key"]: p for p in plan}
    (out / "work").mkdir(parents=True, exist_ok=True)
    (out / "work" / "plan.json").write_text(json.dumps(plan))
    print("plan clips", len(plan), collections.Counter((p["split"], p["cls"]) for p in plan), flush=True)

    # 2. TTS on GPU shards
    jobs = [(p["key"], p["voice"], p["aug"]["speed"], [s["text"] for s in p["segments"]])
            for p in plan if p["voice"] is not None]
    shards = [(i, jobs[i::n_shards], out_dir) for i in range(n_shards)]
    tts_stats = []
    for r in tts_shard.map(shards, order_outputs=False):
        print("tts", {k: v for k, v in r.items() if k != "fails"}, "fails", len(r.get("fails", [])), flush=True)
        tts_stats.append(r)
    t_tts = time.time() - t0
    volume.reload()
    tts = {}
    for i in range(n_shards):
        with np.load(out / "work" / "tts" / f"shard_{i:03d}.npz") as z:
            for k in z.files:
                key, si = k.split("|")
                tts.setdefault(key, {})[int(si)] = z[k]
    print("tts loaded", len(tts), "clips", round(time.time() - t0), "s", flush=True)

    # 3. assemble clips -> banks
    audio_dir = out / "audio"
    clips_meta, bank_of, skipped = [], {}, []
    bank_names, cur, cur_name, bank_count = [], {}, None, collections.Counter()
    durations = []

    def flush_bank():
        nonlocal cur
        if cur:
            common.write_audio_bank(audio_dir, cur_name, cur)
            bank_names.append(cur_name)
            print("bank", cur_name, len(cur), flush=True)
        cur = {}

    for split in ("train", "CAL"):
        part = [p for p in plan if p["split"] == split]
        for j, p in enumerate(part):
            name = f"sb_calls_train_{j // BANK_CLIPS}" if split == "train" else "sb_calls_cal"
            if name != cur_name:
                flush_bank()
                cur_name = name
            segs = None
            if p["voice"] is not None:
                d = tts.get(p["key"])
                if d is None or len(d) != len(p["segments"]):
                    skipped.append(p["key"])
                    continue
                segs = [d[i] for i in range(len(p["segments"]))]
            rng = np.random.default_rng(p["tone_seed"])
            x, facts = assemble(p, segs, rng)
            assert 0.99 * 1.0 <= len(x) / SR <= 15.01 and np.isfinite(x).all() and np.abs(x).max() <= 0.99 + 1e-6
            cur[p["key"]] = x
            bank_of[p["key"]] = name
            durations.append(len(x) / SR)
            clips_meta.append({"key": p["key"], "cls": p["cls"], "split": split, "facts": facts})
        flush_bank()
    del tts
    print("assembled", len(clips_meta), "skipped", len(skipped), round(time.time() - t0), "s", flush=True)

    # 4. items
    items = []
    for split, cap_scale in (("train", scale), ("CAL", scale * 0.034)):
        cl = [c for c in clips_meta if c["split"] == split]
        alloc = build.allocate(cl, common.SEED + (0 if split == "train" else 1), cap_scale)
        items += build.make_items(cl, alloc, common.SEED, common.make_item, bank_of, plan_by_key, KOKORO_REF)
    train = [it for it in items if it["set"] == "train"]
    cal = [it for it in items if it["set"] == "CAL"]
    common.write_jsonl(out / "items_train.jsonl", train)
    common.write_jsonl(out / "items_cal.jsonl", cal)
    (out / "work" / "clips_facts.json").write_text(json.dumps(clips_meta))
    volume.commit()

    # 5. validation: reload from disk
    bank_index = {b: json.loads((audio_dir / f"{b}.json").read_text()) for b in bank_names}
    reloaded = common.read_jsonl(out / "items_train.jsonl") + common.read_jsonl(out / "items_cal.jsonl")
    report = build.validate(reloaded, bank_index)
    for b in bank_names:  # every bank loads and its index matches the array
        arr = np.load(audio_dir / f"{b}.npy", mmap_mode="r")
        last = max(s + n for s, n in bank_index[b].values())
        assert last == arr.shape[0], b
    bank = common.AudioBank(audio_dir, bank_names[0])
    k0 = next(iter(bank_index[bank_names[0]]))
    assert len(common.phone_channel(bank.get(k0))) == len(bank.get(k0))

    # Leakage: no evaluation item may point at our banks or clip keys
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
            vals = {str(a.get("bank")), str(a.get("key"))} | {str(v) for v in (it.get("meta") or {}).values()
                                                              if isinstance(v, (str, int))}
            hits += bool(vals & ours)
        leak[name] = {"items_checked": n, "items_referencing_our_audio": hits}

    # 6. summary (written last)
    def cnt(rows, f):
        return dict(sorted(collections.Counter(f(r) for r in rows).items()))

    by_src = collections.defaultdict(list)
    for it in train:
        by_src[it["source"]].append(it)
    examples = {}
    for s, rows in by_src.items():
        r = random.Random(s)
        examples[s] = [{"question": it["question"], "options": it["options"], "gold": it["gold"],
                        "class": it["meta"]["class"], "script": it["meta"]["script"]} for it in r.sample(rows, 3)]
    vxc = collections.defaultdict(collections.Counter)
    for c in clips_meta:
        v = plan_by_key[c["key"]]["voice"]
        if v:
            vxc[c["cls"]][v] += 1
    voice_spread = {cls: {"n_voices": len(v), "min": min(v.values()), "max": max(v.values())} for cls, v in vxc.items()}
    facts_tr = [c["facts"] for c in clips_meta]
    summary = {
        "builder": "sb_calls",
        "out_dir": out_dir,
        "n_items": {"train": len(train), "CAL": len(cal)},
        "n_clips": cnt(clips_meta, lambda c: c["split"]),
        "clips_per_class": {s: cnt([c for c in clips_meta if c["split"] == s], lambda c: c["cls"]) for s in ("train", "CAL")},
        "items_per_source": {s: cnt([it for it in items if it["set"] == s], lambda it: it["source"]) for s in ("train", "CAL")},
        "items_per_qtype": {s: cnt([it for it in items if it["set"] == s], lambda it: it["meta"]["qtype"]) for s in ("train", "CAL")},
        "items_per_question": {s: cnt([it for it in items if it["set"] == s], lambda it: it["meta"]["qname"]) for s in ("train", "CAL")},
        "items_per_class": {s: cnt([it for it in items if it["set"] == s], lambda it: it["meta"]["class"]) for s in ("train", "CAL")},
        "channel": cnt(items, lambda it: it["channel"]),
        "truncated_clip_share": round(sum(f["truncated"] for f in facts_tr) / len(facts_tr), 3),
        "beep_present_voicemail_clips": cnt([c for c in clips_meta if c["cls"] in ("personal_voicemail", "carrier_voicemail")],
                                            lambda c: c["facts"]["beep_present"]),
        "voices": {"train": TRAIN_VOICES, "CAL": CAL_VOICES},
        "voice_class_spread": voice_spread,
        "audio_hours": round(sum(durations) / 3600, 2),
        "duration_s": {"min": round(min(durations), 2), "mean": round(float(np.mean(durations)), 2),
                       "max": round(max(durations), 2)},
        "banks": bank_names,
        "licences": {
            "sb_calls_tts": f"speech synthesized with {KOKORO_REF}; scripts written combinatorially in code "
                            "(fictional names/businesses); tones synthesized with numpy",
            "sb_calls_tones": "synthetic (numpy-generated call-progress / fax / modem tones), no external data",
        },
        "examples": examples,
        "validation": report,
        "leakage_check": leak,
        "skipped_clips_tts_failure": skipped,
        "tts_stats": [{k: v for k, v in r.items() if k != "fails"} for r in tts_stats],
        "tts_fails": [f for r in tts_stats for f in r.get("fails", [])][:50],
        "timing_s": {"tts": round(t_tts), "total": round(time.time() - t0)},
        "split_rule": "CAL = held-out voices (af_nova, am_eric, bf_lily, bm_lewis) AND held-out script templates/names "
                      "(last entries of every template list in scripts.py); tone-only CAL clips use separate seeds.",
        "classes": CLASSES,
        "speech_classes": SPEECH_CLASSES,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    volume.commit()
    print(json.dumps({k: summary[k] for k in ("n_items", "n_clips", "items_per_question", "audio_hours", "timing_s")}))
    return summary


@app.local_entrypoint()
def smoke():
    s = main.remote(f"{WORK}/smoke", 0.012, 2)
    print({k: s[k] for k in ("n_items", "n_clips", "items_per_question", "validation", "audio_hours", "timing_s")})


@app.local_entrypoint()
def full():
    s = main.remote(OUT, 1.0, 12)
    print({k: s[k] for k in ("n_items", "n_clips", "items_per_question", "validation", "audio_hours", "timing_s")})
