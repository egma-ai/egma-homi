"""Build sb_spoken on Modal: HVB audio (CPU), Kokoro TTS (GPU) and assembly (CPU).

    modal run --detach data/sb_spoken/modal_build.py::main
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
INFERENCE = HERE.parent.parent / "inference"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(INFERENCE))

VOL = "/vol"
KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_REV = "f3ff3571791e39611d31c381e3a41a3af07b4987"  # pinned HF commit (apache-2.0)
HVB_REPO = "asapp/slue-phase-2"
HVB_MAIN_REV = "e2989c55a53593a8e39b8f8ebdb47ccaccbe484a"
HVB_PARQUET_REV = "c3e476fce94cbd006aafb52187125b748b2c037e"  # refs/convert/parquet commit (hvb/train/000{0,1}.parquet)
CSQA_REPO = "tau/commonsense_qa"
CSQA_REV = "94630fe30dad47192a8546eb75f094926d47e155"
JADE_REPO = "theunnecessarythings/JADE-Data"
JADE_REV = "ca75bcb94afd60c4e9c0c17133da4c731e527ea2"

SOURCES = {
    "hvb": {"repo": HVB_REPO, "config": "hvb", "split": "train", "revision": HVB_MAIN_REV,
            "parquet_revision": HVB_PARQUET_REV,
            "licence": "CC BY 4.0 (SLUE dataset card; original Gridspace-Stanford Harper Valley corpus is CC BY 4.0)"},
    "csqa": {"repo": CSQA_REPO, "split": "train", "revision": CSQA_REV, "licence": "MIT (HF cardData license: mit)"},
    "jade": {"repo": JADE_REPO, "file": "train.jsonl", "revision": JADE_REV, "licence": "Apache-2.0 (HF cardData + LICENSE)"},
    "kokoro": {"repo": KOKORO_REPO, "revision": KOKORO_REV, "licence": "Apache-2.0 (TTS model used to read the text)"},
}

volume = modal.Volume.from_name("homi", create_if_missing=False)
hf_secret = modal.Secret.from_name("huggingface")

_local_files = ["modal_build.py", "build.py", "voice_wer.json"]


def _add_code(img: modal.Image) -> modal.Image:
    img = img.add_local_file(INFERENCE / "common.py", "/root/common.py")
    img = img.add_local_file(INFERENCE / "answer_codes.json", "/root/answer_codes.json")
    for f in _local_files:
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

app = modal.App("sb-spoken-build")

# full-build sizes (smoke uses tiny ones)
FULL = {"csqa": (2200, 2600, 1600), "jade_scale": 1.0, "checks": (1800, 1800, 1400, 1000), "shard": 360, "hvb_max": None}
SMOKE = {"csqa": (6, 6, 6), "jade_scale": 0.01, "checks": (8, 8, 6, 6), "shard": 20, "hvb_max": 300}


def out_dir(smoke: bool) -> str:
    return f"{VOL}/stageb/sb_spoken_smoke" if smoke else f"{VOL}/stageb/sb_spoken"


# HVB audio bank
@app.function(image=cpu_image, volumes={VOL: volume}, timeout=3 * 3600, memory=32768, cpu=4, secrets=[hf_secret])
def hvb_bank(smoke: bool = False) -> dict:
    import io
    from collections import Counter

    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    import common as C

    t0 = time.time()
    OUT = out_dir(smoke)
    rows, clips, stats = [], {}, Counter()
    cols = ["issue_id", "speaker_id", "text", "utt_index", "channel", "role", "start_ms", "duration_ms", "intent", "dialog_acts"]
    for f in ("hvb/train/0000.parquet", "hvb/train/0001.parquet"):
        p = hf_hub_download(HVB_REPO, f, repo_type="dataset", revision=HVB_PARQUET_REV, cache_dir="/tmp/hf")
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=256):
            for r in batch.to_pylist():
                if smoke and len(rows) >= SMOKE["hvb_max"]:
                    break
                a = r.pop("audio")
                r = {k: r[k] for k in cols}
                r["dialog_acts"] = list(r["dialog_acts"] or [])
                key = f"{r['issue_id']}_{int(r['utt_index']):03d}"
                try:
                    x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=True)
                except Exception:
                    stats["decode_error"] += 1
                    continue
                stats[f"sr:{sr}"] += 1
                stats[f"nch:{x.shape[1]}"] += 1
                ch = int(r["channel"]) - 1 if x.shape[1] > 1 and 1 <= int(r["channel"]) <= x.shape[1] else 0
                x = C.resample(x[:, ch], sr, C.SAMPLE_RATE)
                if len(x) > 20 * C.SAMPLE_RATE:
                    stats["over_20s_skipped"] += 1
                    continue
                if len(x) < 0.05 * C.SAMPLE_RATE:
                    stats["empty"] += 1
                    continue
                pk = float(np.max(np.abs(x))) if len(x) else 0.0
                if pk > 0.99:
                    x = (x * (0.99 / pk)).astype(np.float32)
                    stats["peak_normalised"] += 1
                clips[key] = x.astype(np.float32)
                rows.append(r)
    C.write_audio_bank(f"{OUT}/audio", "sb_spoken_hvb", clips)
    work = Path(f"{OUT}/work")
    work.mkdir(parents=True, exist_ok=True)
    durs = {k: len(v) / C.SAMPLE_RATE for k, v in clips.items()}
    (work / "hvb_rows.json").write_text(json.dumps(rows))
    (work / "hvb_durations.json").write_text(json.dumps(durs))
    res = {"rows": len(rows), "hours": round(sum(durs.values()) / 3600, 3), "stats": dict(stats), "seconds": round(time.time() - t0, 1)}
    (work / "hvb_stats.json").write_text(json.dumps(res, indent=1))
    volume.commit()
    print("HVB", res, flush=True)
    return res


# TTS plan
@app.function(image=cpu_image, volumes={VOL: volume}, timeout=3600, memory=16384, secrets=[hf_secret])
def prepare_tts(smoke: bool = False) -> list[list[dict]]:
    import random
    from collections import Counter

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    import build as B

    cfg = SMOKE if smoke else FULL
    OUT = out_dir(smoke)
    p = hf_hub_download(CSQA_REPO, "data/train-00000-of-00001.parquet", repo_type="dataset", revision=CSQA_REV, cache_dir="/tmp/hf")
    csqa_rows = pq.read_table(p).to_pylist()
    csqa, cs_stats = B.csqa_specs(csqa_rows, *cfg["csqa"])
    # skip rows already used by the sb_jade text builder
    excl = set()
    for fn in ("items_train.jsonl", "items_cal.jsonl"):
        fp = Path(f"{VOL}/stageb/sb_jade/{fn}")
        if fp.exists():
            with fp.open() as f:
                for ln in f:
                    iid = json.loads(ln)["id"]
                    excl.add(iid.split("/", 1)[1] if iid.startswith("sb_jade/") else iid)
    jp = hf_hub_download(JADE_REPO, "train.jsonl", repo_type="dataset", revision=JADE_REV, cache_dir="/tmp/hf")
    jrows = []
    with open(jp) as f:
        for ln in f:
            r = json.loads(ln)
            if r.get("suite") in B.JADE_SUITES:
                jrows.append({k: r.get(k) for k in ("id", "family", "suite", "state", "question", "target")})
    if smoke:
        B.JADE_SUITES = {k: (max(2, int(v[0] * cfg["jade_scale"])), v[1]) for k, v in B.JADE_SUITES.items()}
    jade, j_stats = B.jade_specs(jrows, excl)
    specs = csqa + jade
    voice_wer = json.loads(Path("/root/voice_wer.json").read_text())
    voices = B.english_voices(voice_wer)
    train_v, cal_v = B.split_voices(voices)
    B.assign_voices(specs, train_v, cal_v)
    random.Random(f"{B.C.SEED}:shards").shuffle(specs)
    n = cfg["shard"]
    shards = [specs[i : i + n] for i in range(0, len(specs), n)]
    work = Path(f"{OUT}/work")
    work.mkdir(parents=True, exist_ok=True)
    info = {"n_specs": len(specs), "csqa": dict(cs_stats), "jade": dict(j_stats), "sb_jade_rows_excluded": len(excl),
            "jade_rows_in_suites": len(jrows), "voices_train": train_v, "voices_cal": cal_v,
            "by_source_variant": dict(Counter(f"{s['source']}|{s['variant']}|{'CAL' if s['cal'] else 'train'}" for s in specs)),
            "n_shards": len(shards)}
    (work / "tts_plan.json").write_text(json.dumps({"info": info, "shards": shards}))
    volume.commit()
    print("PLAN", json.dumps(info), flush=True)
    return shards


# TTS synthesis (GPU)
@app.function(image=tts_image, gpu=["L4", "A10G"], volumes={VOL: volume}, timeout=2 * 3600, secrets=[hf_secret],
              retries=1)
def tts_shard(idx: int, specs: list[dict], smoke: bool = False, n_asr: int = 6) -> dict:
    import re

    import numpy as np
    from huggingface_hub import hf_hub_download
    from kokoro import KModel, KPipeline
    from scipy.signal import resample_poly

    import common as C

    t0 = time.time()
    OUT = out_dir(smoke)
    cfg = hf_hub_download(KOKORO_REPO, "config.json", revision=KOKORO_REV)
    mdl = hf_hub_download(KOKORO_REPO, "kokoro-v1_0.pth", revision=KOKORO_REV)
    model = KModel(config=cfg, model=mdl).to("cuda").eval()
    pipes = {lc: KPipeline(lang_code=lc, model=model) for lc in "ab"}
    vpath = {}
    clips, durs, skipped = {}, {}, {}
    pad = np.zeros(int(0.15 * C.SAMPLE_RATE), np.float32)
    for s in specs:
        v = s["voice"]
        if v not in vpath:
            vpath[v] = hf_hub_download(KOKORO_REPO, f"voices/{v}.pt", revision=KOKORO_REV)
        lc = "b" if v.startswith("b") else "a"
        try:
            chunks = [r.audio.cpu().numpy() for r in pipes[lc](s["tts_text"], voice=vpath[v], speed=s["speed"]) if r.audio is not None]
        except Exception as e:  # noqa: BLE001
            skipped[s["clip"]] = f"error: {e!r}"[:200]
            continue
        if not chunks:
            skipped[s["clip"]] = "no audio"
            continue
        x = resample_poly(np.concatenate(chunks).astype(np.float64), 2, 3).astype(np.float32)
        x = np.concatenate([pad, x, pad])
        pk = float(np.max(np.abs(x)))
        if pk > 0.99:
            x = x * (0.99 / pk)
        if len(x) > 20 * C.SAMPLE_RATE:
            skipped[s["clip"]] = f"too long {len(x) / C.SAMPLE_RATE:.1f}s"
            continue
        clips[s["clip"]] = x.astype(np.float32)
        durs[s["clip"]] = len(x) / C.SAMPLE_RATE
    synth_s = time.time() - t0
    bank = f"sb_spoken_tts{idx:02d}"
    C.write_audio_bank(f"{OUT}/audio", bank, clips)
    # intelligibility spot check: Whisper on a few clips of this shard
    asr = []
    try:
        from transformers import pipeline as hf_pipeline

        rec = hf_pipeline("automatic-speech-recognition", model="openai/whisper-small.en", device=0)

        def norm(t):
            return re.sub(r"[^a-z0-9 ]", " ", t.lower()).split()

        def wer(ref, hyp):
            r, h = norm(ref), norm(hyp)
            dp = list(range(len(h) + 1))
            for i in range(1, len(r) + 1):
                prev, dp[0] = dp[0], i
                for j in range(1, len(h) + 1):
                    cur = min(dp[j] + 1, dp[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
                    prev, dp[j] = dp[j], cur
            return dp[len(h)] / max(1, len(r))

        keys = list(clips)[: n_asr] if not smoke else list(clips)
        text_of = {s["clip"]: s["text"] for s in specs}
        for k in keys:
            hyp = rec({"raw": clips[k], "sampling_rate": C.SAMPLE_RATE})["text"]
            asr.append({"clip": k, "ref": text_of[k], "hyp": hyp, "wer": round(wer(text_of[k], hyp), 3)})
    except Exception as e:  # noqa: BLE001
        asr.append({"error": repr(e)[:300]})
    res = {"idx": idx, "bank": bank, "durations": durs, "skipped": skipped, "asr": asr,
           "synth_seconds": round(synth_s, 1), "seconds": round(time.time() - t0, 1),
           "audio_seconds": round(sum(durs.values()), 1)}
    work = Path(f"{OUT}/work")
    work.mkdir(parents=True, exist_ok=True)
    (work / f"tts_result_{idx:02d}.json").write_text(json.dumps(res))
    volume.commit()
    print("SHARD", idx, len(clips), "clips", res["audio_seconds"], "s audio in", res["synth_seconds"], "s; skipped", len(skipped), flush=True)
    return {k: v for k, v in res.items() if k != "durations"}


# Assembly: items, CAL, leakage checks, validation, summary.json (written last)
def _norm_text(t: str) -> str:
    import re

    return " ".join(re.sub(r"[^a-z0-9 ]", " ", (t or "").lower()).split())


def _load_adi() -> tuple[list[dict], str]:
    p = Path(f"{VOL}/adi/items.jsonl")
    if p.exists():
        return [json.loads(ln) for ln in p.open()], str(p)
    rows = []
    for fp in sorted(Path(f"{VOL}/adi/parts").glob("*.jsonl")):
        rows += [json.loads(ln) for ln in fp.open()]
    return rows, f"{VOL}/adi/parts/*.jsonl (items.jsonl not built yet)"


@app.function(image=cpu_image, volumes={VOL: volume}, timeout=3600, memory=32768)
def assemble(smoke: bool = False, timings: dict | None = None) -> dict:
    import random
    from collections import Counter, defaultdict

    import build as B
    import common as C

    volume.reload()
    t0 = time.time()
    OUT = out_dir(smoke)
    cfg = SMOKE if smoke else FULL
    work = Path(f"{OUT}/work")
    plan = json.loads((work / "tts_plan.json").read_text())
    specs = [s for sh in plan["shards"] for s in sh]
    shard_of = {s["clip"]: i for i, sh in enumerate(plan["shards"]) for s in sh}
    results = {}
    for fp in sorted(work.glob("tts_result_*.json")):
        r = json.loads(fp.read_text())
        results[r["idx"]] = r
    missing_shards = [i for i in range(len(plan["shards"])) if i not in results]

    # evaluation-set metadata for the leakage checks
    adi, adi_src = _load_adi()
    adi_hvb = [it for it in adi if (it.get("source") == "hvb" or str(it.get("benchmark", "")).startswith("hvb"))]
    adi_calls = {str(it["meta"].get("call")) for it in adi_hvb if it.get("meta", {}).get("call")}
    adi_speakers = {str(it["meta"].get("speaker")) for it in adi_hvb if it.get("meta", {}).get("speaker") is not None}
    adi_texts = set()
    for it in adi:
        st = it.get("state_text")
        if st:
            adi_texts.add(_norm_text(st.split("\n")[0]))
            adi_texts.add(_norm_text(st))

    # HVB
    hvb_rows = json.loads((work / "hvb_rows.json").read_text())
    hvb_durs = json.loads((work / "hvb_durations.json").read_text())
    hvb_items, hvb_stats = B.build_hvb_items(hvb_rows, hvb_durs, adi_speakers)
    hvb_calls_used = {it["meta"]["call"] for it in hvb_items}
    hvb_spk_used = {it["meta"]["speaker"] for it in hvb_items}

    # spoken QA items
    items_b, clips_c, dropped = [], [], Counter()
    for s in specs:
        r = results.get(shard_of[s["clip"]])
        if r is None:
            dropped["shard_missing"] += 1
            continue
        dur = r["durations"].get(s["clip"])
        if dur is None:
            dropped["tts_skipped"] += 1
            continue
        if _norm_text(s["stem"]) in adi_texts:
            dropped["adi_text_overlap"] += 1
            continue
        items_b.append(B.qa_item(s, r["bank"], dur))
        clips_c.append({"clip": s["clip"], "text": s["text"], "cal": s["cal"], "voice": s["voice"], "row_id": s["row_id"],
                        "licence": s["licence"], "bank": r["bank"], "key": s["clip"], "duration_s": round(dur, 3)})

    # listening checks on the same clips
    vocab = sorted({w for c in clips_c for w in B.content_words(c["text"])})
    used = Counter(c["clip"] for c in clips_c)  # 1 QA item per clip so far
    items_c, chk_stats = B.plan_checks(clips_c, vocab, *cfg["checks"], used=used)

    items = hvb_items + items_b + items_c
    train = [it for it in items if it["set"] == "train"]
    cal = [it for it in items if it["set"] == "CAL"]
    strict = [it for it in train if not it["meta"].get("adi_speaker_overlap")]

    # validation against the written banks
    bank_index = {}
    for fp in Path(f"{OUT}/audio").glob("*.json"):
        bank_index[fp.stem] = json.loads(fp.read_text())
    val = B.validate(items, bank_index)
    # CAL disjointness
    tr_voices = {it["meta"].get("voice") for it in train if it["meta"].get("voice")}
    cal_voices = {it["meta"].get("voice") for it in cal if it["meta"].get("voice")}
    tr_spk = {it["meta"]["speaker"] for it in train if it["source"].startswith("hvb")}
    cal_spk = {it["meta"]["speaker"] for it in cal if it["source"].startswith("hvb")}
    tr_calls = {it["meta"]["call"] for it in train if it["source"].startswith("hvb")}
    cal_calls = {it["meta"]["call"] for it in cal if it["source"].startswith("hvb")}
    tr_rows = {it["meta"].get("row_id") or it["meta"].get("tts_source_row") for it in train if not it["source"].startswith("hvb")}
    cal_rows = {it["meta"].get("row_id") or it["meta"].get("tts_source_row") for it in cal if not it["source"].startswith("hvb")}
    tr_fams = {it["meta"].get("family") for it in train if it["meta"].get("family")}
    cal_fams = {it["meta"].get("family") for it in cal if it["meta"].get("family")}
    cal_check = {"voice_overlap": sorted(tr_voices & cal_voices), "hvb_speaker_overlap": sorted(tr_spk & cal_spk),
                 "hvb_call_overlap": len(tr_calls & cal_calls), "tts_row_overlap": len((tr_rows & cal_rows) - {None}),
                 "jade_family_overlap": len(tr_fams & cal_fams), "cal_share": round(len(cal) / max(1, len(items)), 4),
                 "cal_voices": sorted(cal_voices), "cal_hvb_speakers": sorted(cal_spk)}

    # hours of audio actually referenced
    used_clips = {(it["audio"]["bank"], it["audio"]["key"]) for it in items}
    hours = sum(bank_index[b][k][1] for b, k in used_clips if b in bank_index and k in bank_index[b]) / C.SAMPLE_RATE / 3600
    hours_by_part = defaultdict(float)
    for b, k in used_clips:
        if b in bank_index and k in bank_index[b]:
            hours_by_part["hvb" if b == "sb_spoken_hvb" else "tts"] += bank_index[b][k][1] / C.SAMPLE_RATE / 3600

    # write items, then summary.json last
    C.write_jsonl(f"{OUT}/items_train.jsonl", train)
    C.write_jsonl(f"{OUT}/items_cal.jsonl", cal)
    C.write_jsonl(f"{OUT}/items_train_strict.jsonl", strict)
    counts = defaultdict(Counter)
    for it in items:
        counts[it["source"]][f"{it['set']}|{it['meta']['qtype']}"] += 1
    checks_by_kind = Counter(f"{it['set']}|{it['meta']['check']}" for it in items_c)
    examples = {}
    rng = random.Random(C.SEED)
    for src in sorted(counts):
        pool = [it for it in items if it["source"] == src]
        examples[src] = [{"id": it["id"], "question": it["question"], "options": it["options"], "gold": it["gold"],
                          "transcript": it["meta"].get("transcript"), "audio": it["audio"]} for it in rng.sample(pool, min(3, len(pool)))]
    asr = [a for r in results.values() for a in r.get("asr", []) if "wer" in a]
    asr_err = [a for r in results.values() for a in r.get("asr", []) if "error" in a]
    tts_skipped = {k: v for r in results.values() for k, v in r.get("skipped", {}).items()}
    tts_gpu_s = sum(r.get("seconds", 0) for r in results.values())
    summary = {
        "builder": "sb_spoken",
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "smoke": smoke,
        "paths": {"items_train": f"{OUT}/items_train.jsonl", "items_cal": f"{OUT}/items_cal.jsonl",
                  "items_train_strict": f"{OUT}/items_train_strict.jsonl", "audio_dir": f"{OUT}/audio",
                  "banks": sorted(bank_index)},
        "n_items": {"train": len(train), "CAL": len(cal), "train_strict": len(strict)},
        "counts_by_source": {k: dict(v) for k, v in sorted(counts.items())},
        "listen_checks_by_kind": dict(sorted(checks_by_kind.items())),
        "audio_hours": {"total": round(hours, 3), **{k: round(v, 3) for k, v in hours_by_part.items()}},
        "sources": SOURCES,
        "examples": examples,
        "leakage": {
            "adi_items_file": adi_src, "adi_items": len(adi), "adi_hvb_items": len(adi_hvb),
            "hvb_issue_id_overlap_with_adi": sorted(hvb_calls_used & adi_calls),
            "hvb_speakers_used": len(hvb_spk_used),
            "hvb_speaker_overlap_with_adi": sorted(hvb_spk_used & adi_speakers),
            "hvb_items_with_adi_speaker": sum(1 for it in hvb_items if it["meta"]["adi_speaker_overlap"]),
            "note_speakers": ("HarperValleyBank has one pool of ~60 crowd-worker speakers who appear in both the SLUE "
                              "train and test conversations (SLUE splits by conversation). No evaluation-set call "
                              "(issue_id) is used, but most HVB speakers also occur in the evaluation set's HVB items. "
                              "items_train.jsonl keeps them (flag meta.adi_speaker_overlap); items_train_strict.jsonl "
                              "drops every item whose speaker is in the evaluation set."),
            "text_overlap_dropped": dropped.get("adi_text_overlap", 0),
            "text_sources_never_used": "VoiceBench, MMSU, Big Bench Audio, OpenBookQA, MMLU, BBH, ARC (only CSQA train + JADE-Data)",
        },
        "cal_disjointness": cal_check,
        "validation": val,
        "build_stats": {"hvb": hvb_stats, "tts_plan": plan["info"], "listen_checks": dict(chk_stats),
                        "dropped": dict(dropped), "tts_skipped": len(tts_skipped),
                        "tts_skipped_examples": dict(list(tts_skipped.items())[:10]), "missing_shards": missing_shards,
                        "hvb_bank": json.loads((work / "hvb_stats.json").read_text())},
        "tts_asr_check": {"n": len(asr), "mean_wer": round(sum(a["wer"] for a in asr) / max(1, len(asr)), 4),
                          "worst": sorted(asr, key=lambda a: -a["wer"])[:5], "errors": asr_err[:3]},
        "compute": {"tts_container_seconds": round(tts_gpu_s, 1), "timings": timings or {},
                    "est_cost_usd": round(tts_gpu_s / 3600 * 0.80 + (timings or {}).get("cpu_seconds", 0) / 3600 * 0.6, 2)},
        "assemble_seconds": round(time.time() - t0, 1),
    }
    (Path(OUT) / "summary.json").write_text(json.dumps(summary, indent=1))
    volume.commit()
    brief = {k: summary[k] for k in ("n_items", "counts_by_source", "audio_hours", "cal_disjointness", "validation", "tts_asr_check")}
    print("SUMMARY", json.dumps(brief, indent=1)[:6000], flush=True)
    print("LEAKAGE", json.dumps(summary["leakage"])[:1500], flush=True)
    return brief


@app.function(image=cpu_image, volumes={VOL: volume}, timeout=5 * 3600, memory=4096)
def orchestrate(smoke: bool = False) -> dict:
    t0 = time.time()
    h = hvb_bank.spawn(smoke)
    shards = prepare_tts.remote(smoke)
    t_plan = time.time() - t0
    res = list(tts_shard.starmap([(i, sh, smoke) for i, sh in enumerate(shards)], return_exceptions=True))
    errs = [repr(r)[:300] for r in res if isinstance(r, Exception)]
    if errs:
        print("TTS shard errors:", errs, flush=True)
    t_tts = time.time() - t0
    hv = h.get()
    t_hvb = time.time() - t0
    timings = {"plan_s": round(t_plan, 1), "tts_done_s": round(t_tts, 1), "hvb_done_s": round(t_hvb, 1),
               "hvb_seconds": hv.get("seconds"), "cpu_seconds": round((hv.get("seconds") or 0) * 4 + t_plan + 600, 1),
               "tts_shard_errors": errs}
    out = assemble.remote(smoke, timings)
    out["timings"] = timings
    return out


@app.local_entrypoint()
def main():
    print(json.dumps(orchestrate.remote(False), indent=1)[:4000])


@app.local_entrypoint()
def smoke():
    print(json.dumps(orchestrate.remote(True), indent=1)[:6000])


@app.local_entrypoint()
def reassemble(smoke: bool = False):
    print(json.dumps(assemble.remote(smoke), indent=1)[:4000])
