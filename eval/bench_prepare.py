"""Public benchmarks for evaluation only: VocalGrad (voice_pitch / speaking_speed / volume), VoxParadox (all 10 tasks)
and MSU-Bench (English, human-verified items). No model is run here.

Each benchmark is written in the layout mmsu_prepare() in eval/modal_eval.py uses for /vol/bench_mmsu:
    /vol/bench_<name>/flac/<cid>.flac            16 kHz mono FLAC, one per question
    /vol/bench_<name>/audio/bench_<name>.{npy,json}   audio bank (common.write_audio_bank)
    /vol/bench_<name>/truth.jsonl                one row per question (the benchmark's own question + options)
    /vol/bench_<name>/items<name>.jsonl          one item per question (same shape as mmsu_prepare's items)
    /vol/bench_<name>/summary.json               written last: source revision, licence, counts, skips, durations

    modal run eval/bench_prepare.py::main --smoke          # 20 items per benchmark -> /vol/bench_<name>_smoke
    modal run eval/bench_prepare.py::check --smoke         # read the files back with common.read_jsonl / load_banks
    modal run --detach eval/bench_prepare.py::main         # the three full builds, in parallel
    modal run eval/bench_prepare.py::check

Every Hugging Face download is pinned to the commit SHA in BENCHES.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

INFERENCE = Path(__file__).resolve().parent.parent / "inference"  # common.py + answer_codes.json
sys.path.insert(0, str(INFERENCE))

VOL = "/vol"
SR = 16000
LETTERS = "ABCDEFGHIJ"
PLACEHOLDERS = ("", "None", "null")  # empty option slots: mmsu_prepare drops "" / "None"; VoxParadox gender uses "null"

BENCHES = {
    "vocalgrad": {
        "repo": "ynklab/vocalgrad", "revision": "00e66e7aef76a453b3c98eeadd6c7ca7ced0cf78",
        "licence": "CC BY 4.0 (dataset card); source speech is CSTR VCTK 0.92, CC BY 4.0. Attribution required.",
        "paper": "arXiv 2610.23745 (NeurIPS 2026 E&D)"},
    "voxparadox": {
        "repo": "IHP-Lab/VoxParadox", "revision": "49e3119ff4ce73ad879ab8d8924f0b7a7532222e",
        "licence": "USC Research License (HF license: other / usc-research): educational, research and non-profit use only; "
                   "commercial use needs a licence from USC Stevens. Audio is commercial TTS (ElevenLabs, GPT-4o, Azure), "
                   "also subject to those vendors' terms.",
        "paper": "arXiv 2605.27772 (ICML 2026)"},
    "msubench": {
        "repo": "ASLP-lab/MSU-Benchmark", "revision": "2f3a07acd713a60cb105d8bb09d3c47867466802",
        "licence": "CC BY-NC 4.0, non-commercial academic research only (audio is third-party copyrighted film/TV, "
                   "telephone, meeting and podcast material).",
        "paper": "arXiv 2606.22868 (Interspeech 2026); eval protocol: github.com/ASLP-lab/MSU-Bench @ 5e57df21 "
                 "bench_evaluation_pipeline/step2_run_scoring/question_builder.py"},
}

app = modal.App("egma-homi-bench-prep")
volume = modal.Volume.from_name("egma-homi", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg", "ca-certificates")
         .uv_pip_install("numpy==2.5.3", "scipy==1.18.1", "soundfile==0.14.0", "pyarrow==25.0.1", "huggingface_hub==1.33.0")
         .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_XET_HIGH_PERFORMANCE": "1"})
         .add_local_file(INFERENCE / "common.py", "/root/common.py")
         .add_local_file(INFERENCE / "answer_codes.json", "/root/answer_codes.json"))  # common.py reads it at import
hub_secret = modal.Secret.from_name("huggingface")  # HF_TOKEN, only for download rate limits (the repos are open)


# ---------------------------------------------------------------------------------------------
# Helpers (run inside the container)
# ---------------------------------------------------------------------------------------------
def _log(*a) -> None:
    print(*a, flush=True)


def _hub_token_env() -> None:
    import os

    if not os.environ.get("HF_TOKEN"):
        for k in ("HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_HUB_TOKEN"):
            if os.environ.get(k):
                os.environ["HF_TOKEN"] = os.environ[k]
                break


def _hub_files(repo: str, revision: str, files: list[str], dest: Path, workers: int = 16) -> dict:
    """Download repo files at the pinned commit; returns {repo path: local path or None if it failed after retries}."""
    import random
    import time
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download

    def one(f):
        for attempt in range(9):
            try:
                return hf_hub_download(repo_id=repo, filename=f, repo_type="dataset", revision=revision, local_dir=str(dest))
            except Exception as e:  # noqa: BLE001  (rate limits, network errors: back off and retry)
                if attempt == 8:
                    _log("download failed", f, type(e).__name__, str(e)[:200])
                    return None
                time.sleep(min(60, 2 ** attempt) + random.random())

    out, t0 = {}, time.time()
    with ThreadPoolExecutor(workers) as ex:
        for i, (f, p) in enumerate(zip(files, ex.map(one, files))):
            out[f] = p
            if (i + 1) % 1000 == 0:
                _log(f"  downloaded {i + 1}/{len(files)} ({time.time() - t0:.0f}s)")
    return out


def _decode(data: bytes):
    """Same decoder as mmsu_prepare: ffmpeg -> 16 kHz mono float32."""
    import io
    import subprocess

    import soundfile as sf

    out = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", "-"],
                         input=data, capture_output=True, check=True).stdout
    x, _ = sf.read(io.BytesIO(out), dtype="float32")
    return x


def _sha256(path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pick(rows: list, key, n: int) -> list:
    """Smoke subset: deterministic round-robin over groups, so a few items of every kind are covered."""
    groups: dict = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    keys, out, i = sorted(groups), [], 0
    while len(out) < n and any(i < len(groups[k]) for k in keys):
        for k in keys:
            if i < len(groups[k]) and len(out) < n:
                out.append(groups[k][i])
        i += 1
    return out


# ---------------------------------------------------------------------------------------------
# Per-benchmark readers: source rows -> question specs (no audio yet)
#   spec = {cid, task, category, sub_category, question, options, gold, src (repo audio path), seg (None | (s, e)), extra}
# ---------------------------------------------------------------------------------------------
VG_CATS = ("voice_pitch", "speaking_speed", "volume")
VG_SUFFIX = ' Answer with only one word: "increase" or "decrease".'


def _vocalgrad(dl: Path, limit: int, skipped, notes: list) -> tuple[list, dict]:
    import csv
    from collections import Counter

    with open(dl / "metadata.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sub = [r for r in rows if r["category"] in VG_CATS]
    info = {"rows_in_source": len(rows), "rows_in_subset": len(sub),
            "subset": "categories voice_pitch, speaking_speed, volume (all 3 x 2,400 test clips); the six audio-quality "
                      "categories are out of scope",
            "excluded_by_subset": dict(Counter(r["category"] for r in rows if r["category"] not in VG_CATS))}
    if limit:
        sub = _pick(sub, lambda r: r["category"], limit)
    notes += ["question = the dataset's own `prompt` with its trailing output-format sentence "
              "' Answer with only one word: \"increase\" or \"decrease\".' removed (the eval harness adds its own answer "
              "instruction); the full original prompt is kept in truth.prompt_orig",
              "options = ['increase', 'decrease'] in the order the prompt and the card name them; gold = the row's `label` "
              "(card: direction=up -> increase, direction=down -> decrease)",
              "one clip per question; no multi-audio items"]
    specs, n_suffix = [], 0
    for r in sub:
        prompt = r["prompt"]
        if prompt.endswith(VG_SUFFIX):
            q, n_suffix = prompt[: -len(VG_SUFFIX)].strip(), n_suffix + 1
        else:
            q = prompt.strip()
        opts = ["increase", "decrease"]
        lab = r["label"].strip()
        if opts.count(lab) != 1:
            skipped["no_gold"] += 1
            continue
        stem = Path(r["file_name"]).stem  # <category>__<benchmark_clip_id>; the clip id alone repeats across categories
        specs.append({"cid": f"vocalgrad_{stem}", "task": r["category"], "category": r["category"],
                      "sub_category": r["tier"], "question": q, "options": opts, "gold": opts.index(lab),
                      "src": r["file_name"], "seg": None,
                      "extra": {"tier": r["tier"], "curve": r["curve"], "direction": r["direction"],
                                "attribute": r["attribute"], "benchmark_clip_id": r["benchmark_clip_id"],
                                "source_clip_id": r["source_clip_id"], "speaker_id": r["speaker_id"],
                                "prompt_orig": prompt, "file_name": r["file_name"]}})
    info["prompt_suffix_removed"] = n_suffix
    return specs, info


def _voxparadox(dl: Path, limit: int, skipped, notes: list) -> tuple[list, dict]:
    from collections import Counter

    rows = [json.loads(l) for l in open(dl / "metadata.jsonl", encoding="utf-8") if l.strip()]
    info = {"rows_in_source": len(rows), "rows_in_subset": len(rows), "subset": "all 10 tasks, all items",
            "excluded_by_subset": {}}
    if limit:
        rows = _pick(rows, lambda r: r["task_name"], limit)
    notes += ["question and options exactly as in metadata.jsonl (choice_a..choice_d in published order); placeholder "
              "options ('', 'None', 'null') dropped -- only gender_prediction has them (two 'null' slots per item), so "
              "gender is 2-way; truth.published_letters maps each kept option back to its original letter",
              "gold = the option equal to answer_gt; truth.adversarial_labels / adversarial_idx kept for the card's "
              "Adversarial-Label Agreement metric",
              "one audio file per item (speaker_identity_recognition and counting already ship as one combined file); "
              "no multi-audio items"]
    dropped = Counter()
    specs = []
    for r in rows:
        kept = [(L, str(r[f"choice_{L.lower()}"]).strip()) for L in "ABCD"
                if r.get(f"choice_{L.lower()}") is not None and str(r[f"choice_{L.lower()}"]).strip() not in PLACEHOLDERS]
        dropped[r["task_name"]] += 4 - len(kept)
        opts = [t for _, t in kept]
        ans = str(r["answer_gt"]).strip()
        if opts.count(ans) != 1:
            skipped["no_gold"] += 1
            continue
        adv = [str(a).strip() for a in (r.get("adversarial_labels") or [])]
        specs.append({"cid": f"voxparadox_{r['id']}", "task": r["task_name"], "category": r["task_name"],
                      "sub_category": None, "question": r["question"], "options": opts, "gold": opts.index(ans),
                      "src": r["file_name"], "seg": None,
                      "extra": {"orig_id": r["id"], "published_letters": [L for L, _ in kept],
                                "adversarial_labels": r.get("adversarial_labels"),
                                "adversarial_idx": [opts.index(a) for a in adv if a in opts]}})
    info["placeholder_options_dropped_per_task"] = {k: v for k, v in dropped.items() if v}
    return specs, info


def _first_registration_seg(meta) -> dict | None:
    """Same rule as the official question_builder._first_registration_seg: spk_a[0], else the first non-empty list."""
    if isinstance(meta, dict):
        arr = meta.get("spk_a")
        if isinstance(arr, list) and arr:
            return arr[0]
        for v in meta.values():
            if isinstance(v, list) and v:
                return v[0]
    return None


def _msubench(dl: Path, limit: int, skipped, notes: list) -> tuple[list, dict]:
    import difflib
    import re
    from collections import Counter

    rows = [json.loads(l) for l in open(dl / "data" / "test.jsonl", encoding="utf-8") if l.strip()]
    sub = [r for r in rows if r.get("language") == "en" and r.get("verified") is True]
    info = {"rows_in_source": len(rows), "rows_in_subset": len(sub),
            "subset": "language == 'en' and verified == true (human-reviewed, error-free)",
            "rows_in_subset_per_tier": dict(Counter(f"tier{r['tier']}" for r in sub)),
            "excluded_by_subset": dict(Counter(f"language={r.get('language')},verified={r.get('verified')}"
                                               for r in rows if not (r.get("language") == "en" and r.get("verified") is True)))}
    if limit:
        sub = _pick(sub, lambda r: (r["task"], r["question_type"]), limit)
    notes += [
        "question text exactly as in data/test.jsonl; options = the four `options` with the release's own 'A. '..'D. ' "
        "letter prefix removed (the harness re-letters them in the same order); truth.options_orig keeps the raw strings",
        "gold = the item's `answer` letter (the release's answer key). The release's `answer_text` is a separately "
        "translated string and often differs in wording from the option text; it is kept as truth.answer_text_orig, "
        "truth.answer_text is the gold option's text",
        "tier / category: the dataset's own `tier` field (= official task_taxonomy.py): Tier 1 includes Speaker Opinion "
        "Summary, Tier 2 includes Emotion Interaction",
        "audio follows the official question_builder.py: every index scheme except no_index -> the full source clip; "
        "no_index (level1, not verification) -> the target speaker's first registration snippet (speaker_meta.spk_a[0] "
        "start..end cut from the source clip; audio_mode = target_snippet)",
        "multi_audio skips: (a) no_index level2 items, whose protocol is snippet + full conversation as two separate "
        "audios while the question only says 'the audio below'; (b) no_index Speaker Verification items, whose protocol "
        "is two clips ('Clip 1:', 'Clip 2:') chosen by a `pair` field that data/test.jsonl does not carry, and whose "
        "question ('the two audio clips below') does not refer to the clips by order. No clips were concatenated."]
    specs, multi = [], Counter()
    for r in sub:
        raw = list(r.get("options") or [])
        kept, n_prefix_kept = [], 0
        for i, o in enumerate(raw):
            s = "" if o is None else str(o).strip()
            m = re.match(r"^([A-Za-z])\s*[\.\)、]\s*", s)
            if m and i < len(LETTERS) and m.group(1).upper() == LETTERS[i]:
                s = s[m.end():].strip()
            else:
                n_prefix_kept += 1
            if s in PLACEHOLDERS:
                continue
            kept.append((i, s))
        ans = str(r.get("answer", "")).strip().upper()
        ai = LETTERS.index(ans) if len(ans) == 1 and ans in LETTERS[: len(raw)] else None
        texts = [t for _, t in kept]
        pos = [j for j, (i, _) in enumerate(kept) if i == ai]
        if len(pos) != 1 or texts.count(texts[pos[0]]) != 1:
            skipped["no_gold"] += 1
            continue
        gi = pos[0]
        qt, lvl = r.get("question_type", ""), int(re.search(r"\d+", str(r.get("level", "level1"))).group())
        seg, mode = None, "full"
        if qt == "no_index":
            if r["task"] == "Speaker Verification":
                skipped["multi_audio"] += 1
                multi[f"{r['task']} / no_index (two clips from an unpublished pair ref)"] += 1
                continue
            if lvl >= 2:
                skipped["multi_audio"] += 1
                multi[f"{r['task']} / no_index level2 (snippet + full conversation)"] += 1
                continue
            s0 = _first_registration_seg(r.get("speaker_meta"))
            if s0 is None or s0.get("start_time") is None or s0.get("end_time") is None:
                skipped["no_anchor"] += 1
                continue
            seg, mode = (float(s0["start_time"]), float(s0["end_time"])), "target_snippet"
        at = str(r.get("answer_text", "")).strip()
        closest = max(range(len(texts)), key=lambda j: difflib.SequenceMatcher(None, at.lower(), texts[j].lower()).ratio())
        specs.append({"cid": f"msubench_{r['uid']}", "task": r["task"], "category": f"Tier {r['tier']}",
                      "sub_category": r.get("dimension"), "question": r["question"], "options": texts, "gold": gi,
                      "src": r["audio"], "seg": seg,
                      "extra": {"tier": r["tier"], "dimension": r.get("dimension"), "level": r.get("level"),
                                "question_type": qt, "scenario": r.get("scenario"), "media_type": r.get("media_type"),
                                "qa_length": r.get("qa_length"), "source_audio": r["audio"], "audio_mode": mode,
                                "snippet_start": seg[0] if seg else None, "snippet_end": seg[1] if seg else None,
                                "orig_uid": r["uid"], "answer_letter_orig": ans, "answer_text_orig": at,
                                "answer_text_orig_closest_idx": closest, "options_orig": raw,
                                "option_prefix_not_stripped": n_prefix_kept}})
    info["multi_audio_detail"] = dict(multi)
    info["answer_text_orig_differs_from_gold_option"] = sum(1 for s in specs if s["extra"]["answer_text_orig"] != s["options"][s["gold"]])
    info["answer_text_orig_closest_to_a_non_gold_option"] = sorted(
        s["cid"] for s in specs if s["extra"]["answer_text_orig_closest_idx"] != s["gold"])
    info["audio_mode"] = dict(Counter(s["extra"]["audio_mode"] for s in specs))
    return specs, info


READERS = {"vocalgrad": _vocalgrad, "voxparadox": _voxparadox, "msubench": _msubench}


# ---------------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------------
@app.function(image=image, volumes={VOL: volume}, secrets=[hub_secret], cpu=8, memory=49152, timeout=3 * 3600)
def prepare(name: str, limit: int = 0, out_dir: str = "") -> dict:
    """One benchmark -> /vol/bench_<name> (or out_dir). limit > 0: a smoke subset of that many source questions."""
    import shutil
    import statistics
    import time
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import soundfile as sf
    from huggingface_hub import HfApi

    import common

    _hub_token_env()
    cfg = BENCHES[name]
    repo, rev = cfg["repo"], cfg["revision"]
    root = Path(out_dir or f"{VOL}/bench_{name}")
    t0 = time.time()

    ds = HfApi().dataset_info(repo, revision=rev)
    if ds.sha != rev:
        raise RuntimeError(f"{repo}: asked for {rev}, hub returned {ds.sha}")
    repo_files = {s.rfilename for s in (ds.siblings or [])}
    _log(name, repo, rev, "files in repo:", len(repo_files))

    dl = Path(f"/tmp/hf_{name}")
    meta_files = {"vocalgrad": ["README.md", "metadata.csv", "metadata/summary.json"],
                  "voxparadox": ["README.md", "LICENSE", "metadata.jsonl", "eval.py"],
                  "msubench": ["README.md", "data/test.jsonl", "build_test_jsonl.py"]}[name]
    got = _hub_files(repo, rev, meta_files, dl, workers=4)
    if any(v is None for v in got.values()):
        raise RuntimeError(f"metadata download failed: {got}")
    meta_sha = {f: _sha256(p) for f, p in got.items()}

    skipped, notes = Counter({"no_gold": 0, "multi_audio": 0, "decode": 0, "download": 0}), []
    specs, info = READERS[name](dl, limit, skipped, notes)
    srcs = sorted({s["src"] for s in specs})
    missing = [s for s in srcs if s not in repo_files]
    _log(name, "questions:", len(specs), "audio files:", len(srcs), "missing in repo:", len(missing), "skipped so far:", dict(skipped))
    files = _hub_files(repo, rev, [s for s in srcs if s in repo_files], dl, workers=16)
    _log(name, f"downloaded audio in {time.time() - t0:.0f}s")

    def dec(src):
        p = files.get(src)
        if p is None:
            return src, None
        try:
            return src, _decode(Path(p).read_bytes())
        except Exception as e:  # noqa: BLE001
            return src, e

    with ThreadPoolExecutor(16) as ex:
        decoded = dict(ex.map(dec, srcs))
    _log(name, f"decoded {len(decoded)} files in {time.time() - t0:.0f}s")

    # remove earlier outputs of this function (an existing runs/ folder is kept)
    shutil.rmtree(root / "flac", ignore_errors=True)
    shutil.rmtree(root / "audio", ignore_errors=True)
    for f in ("summary.json", "truth.jsonl", f"items{name}.jsonl"):
        (root / f).unlink(missing_ok=True)
    (root / "flac").mkdir(parents=True, exist_ok=True)

    clips, truth, items, cache, seen, n_trunc = {}, [], [], {}, set(), 0
    for s in specs:
        x = decoded.get(s["src"])
        if x is None:
            skipped["download"] += 1
            continue
        if isinstance(x, Exception):
            skipped["decode"] += 1
            continue
        key = (s["src"], s["seg"])
        if key not in cache:
            if s["seg"] is None:
                y = x
            else:
                a, b = int(round(s["seg"][0] * SR)), int(round(s["seg"][1] * SR))
                n_trunc += b > len(x)
                y = x[a:b]
            if len(y) < 800:
                cache[key] = None
            else:
                peak = float(np.abs(y).max())
                cache[key] = (y / max(1e-6, peak) * 0.95 if peak > 0.99 else y).astype(np.float32)  # as mmsu_prepare
        y = cache[key]
        if y is None:
            skipped["decode"] += 1
            continue
        cid = s["cid"]
        if cid in seen:
            skipped["duplicate_id"] += 1
            continue
        seen.add(cid)
        opts, gi = s["options"], s["gold"]
        clips[cid] = y
        truth.append({"id": cid, "task": s["task"], "category": s["category"], "sub_category": s["sub_category"],
                      "question": s["question"], "options": opts, "gold": gi, "answer_text": opts[gi],
                      "dur": round(len(y) / SR, 2), **s["extra"]})
        items.append({"id": f"{cid}/answer", "set": f"BENCH/{name}", "source": f"bench_{name}", "question": s["question"],
                      "options": opts, "gold": gi, "target": [1.0 if i == gi else 0.0 for i in range(len(opts))],
                      "audio": {"bank": f"bench_{name}", "key": cid}, "state_text": None, "channel": "clean",
                      "meta": {"call": cid, "question": "answer", "keys": list(LETTERS[: len(opts)]), "has_truth": True,
                               "task": s["task"]}})
    del decoded

    def wr(kv):
        sf.write(root / "flac" / f"{kv[0]}.flac", kv[1], SR, format="FLAC")

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(wr, clips.items()))
    _log(name, f"wrote {len(clips)} flac in {time.time() - t0:.0f}s")
    common.write_audio_bank(root / "audio", f"bench_{name}", clips)
    common.write_jsonl(root / "truth.jsonl", truth)
    common.write_jsonl(root / f"items{name}.jsonl", items)
    _log(name, f"wrote bank + jsonl in {time.time() - t0:.0f}s")

    durs = [t["dur"] for t in truth]
    uniq = {(s["src"], s["seg"]) for s in specs if s["cid"] in clips}
    uniq_sec = sum(len(cache[k]) for k in uniq if cache.get(k) is not None) / SR
    summary = {
        "benchmark": name, "smoke": bool(limit), "evaluation_only": True,
        "source": {"hf_repo": repo, "url": f"https://huggingface.co/datasets/{repo}/tree/{rev}", "revision": rev,
                   "last_modified": str(getattr(ds, "last_modified", "")), "paper": cfg["paper"],
                   "metadata_sha256": meta_sha},
        "licence": cfg["licence"],
        "subset": info.pop("subset"), "rows_in_source": info.pop("rows_in_source"),
        "rows_in_subset": info.pop("rows_in_subset"), "excluded_by_subset": info.pop("excluded_by_subset"),
        "items": len(items), "skipped": dict(skipped),
        "items_per_task": dict(sorted(Counter(t["task"] for t in truth).items())),
        "items_per_category": dict(sorted(Counter(t["category"] for t in truth).items())),
        "hours": round(sum(durs) / 3600, 2), "unique_audio_hours": round(uniq_sec / 3600, 2),
        "clip_dur_mean_s": round(statistics.mean(durs), 2) if durs else None,
        "clip_dur_max_s": max(durs) if durs else None, "clip_dur_min_s": min(durs) if durs else None,
        "options": {str(k): v for k, v in sorted(Counter(len(t["options"]) for t in truth).items())},
        "gold_position": dict(sorted(Counter(LETTERS[t["gold"]] for t in truth).items())),
        "snippets_past_clip_end": n_trunc, "missing_audio_in_repo": missing,
        "handling": notes, **info,
        "files": {"truth": "truth.jsonl", "items": f"items{name}.jsonl", "bank": f"audio/bench_{name}.npy",
                  "flac": "flac/<id>.flac"},
        "truth_sha256": _sha256(root / "truth.jsonl"), "items_sha256": _sha256(root / f"items{name}.jsonl"),
    }
    if name == "msubench":
        summary["items_per_tier_task"] = {f"tier{k[0]} / {k[1]}": v for k, v in
                                          sorted(Counter((t["tier"], t["task"]) for t in truth).items())}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))  # last
    volume.commit()
    _log(name, f"done in {time.time() - t0:.0f}s:", len(items), "items")
    return summary


@app.function(image=image, volumes={VOL: volume}, cpu=4, memory=16384, timeout=3600)
def verify(root: str, name: str) -> dict:
    """Read a built benchmark back through common.read_jsonl / common.load_banks and check it is self-consistent."""
    import numpy as np
    import soundfile as sf

    import common

    volume.reload()
    root = Path(root)
    bank = f"bench_{name}"
    truth = common.read_jsonl(root / "truth.jsonl")
    items = common.read_jsonl(root / f"items{name}.jsonl")
    banks = common.load_banks(root, [bank])
    summary = json.loads((root / "summary.json").read_text())
    problems = []
    if bank not in banks:
        problems.append("bank missing")
        return {"problems": problems}
    if len(truth) != len(items) or summary["items"] != len(items):
        problems.append(f"counts: truth {len(truth)} items {len(items)} summary {summary['items']}")
    ids = [t["id"] for t in truth]
    if len(set(ids)) != len(ids):
        problems.append("duplicate truth ids")
    flacs = {p.stem for p in (root / "flac").glob("*.flac")}
    if flacs != set(ids):
        problems.append(f"flac set != truth ids ({len(flacs)} vs {len(ids)})")
    for k, (t, it) in enumerate(zip(truth, items)):
        o, g = t["options"], t["gold"]
        bad = []
        if it["id"] != f"{t['id']}/answer" or it["audio"] != {"bank": bank, "key": t["id"]}:
            bad.append("id/audio")
        if it["options"] != o or it["gold"] != g or it["question"] != t["question"] or not t["question"].strip():
            bad.append("q/options/gold")
        if not (2 <= len(o) <= len(LETTERS)) or not (0 <= g < len(o)) or t["answer_text"] != o[g] or len(set(o)) != len(o):
            bad.append("options")
        if it["target"] != [1.0 if i == g else 0.0 for i in range(len(o))] or it["meta"]["keys"] != list(LETTERS[: len(o)]):
            bad.append("target/keys")
        if it["set"] != f"BENCH/{name}" or it["source"] != bank or it["channel"] != "clean" or it["meta"]["call"] != t["id"]:
            bad.append("set/source/meta")
        x = common.item_audio(it, banks)
        if x is None or abs(len(x) / SR - t["dur"]) > 0.006 or not np.isfinite(x).all() or np.abs(x).max() > 1.0:
            bad.append("bank audio")
        info = sf.info(str(root / "flac" / f"{t['id']}.flac"))
        if info.samplerate != SR or info.channels != 1 or info.frames != len(x):
            bad.append("flac header")
        if k < 25:  # full decode of a few: FLAC (16-bit) must equal the bank clip
            y, _ = sf.read(str(root / "flac" / f"{t['id']}.flac"), dtype="float32")
            if len(y) != len(x) or float(np.abs(y - x).max()) > 2.0 / 32768:
                bad.append("flac != bank")
        if bad:
            problems.append(f"{t['id']}: {bad}")
    return {"root": str(root), "items": len(items), "truth": len(truth), "bank_keys": len(banks[bank].index),
            "problems": problems[:30], "n_problems": len(problems), "truth_sha256": _sha256(root / "truth.jsonl"),
            "summary_truth_sha256": summary.get("truth_sha256"), "example_truth": truth[0] if truth else None,
            "example_item": items[0] if items else None}


# ---------------------------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------------------------
def _root(name: str, smoke: bool) -> str:
    return f"{VOL}/bench_{name}_smoke" if smoke else f"{VOL}/bench_{name}"


@app.local_entrypoint()
def main(which: str = "vocalgrad,voxparadox,msubench", smoke: bool = False, limit: int = 20):
    """Spawn the builds in parallel. --smoke waits for the results; a full run returns at once (use --detach)."""
    calls = {}
    for name in which.split(","):
        calls[name] = prepare.spawn(name, limit if smoke else 0, _root(name, smoke))
        print(name, _root(name, smoke), calls[name].object_id, flush=True)
    if smoke:
        for name, c in calls.items():
            print(f"===== {name}\n" + json.dumps(c.get(), indent=1, ensure_ascii=False), flush=True)


@app.local_entrypoint()
def check(which: str = "vocalgrad,voxparadox,msubench", smoke: bool = False):
    calls = {name: verify.spawn(_root(name, smoke), name) for name in which.split(",")}
    for name, c in calls.items():
        print(f"===== {name}\n" + json.dumps(c.get(), indent=1, ensure_ascii=False), flush=True)
