"""
MMSU and the other audio benchmarks on Modal, one harness for every model (homi, the untouched Qwen3-Omni base,
Gemini, OpenAI): same audio, the benchmark's own question and options in the published order. Answers are scored
offline by eval/mmsu_report.py and eval/bench_report.py.

    modal run eval/modal_eval.py::mmsu --step prepare                     # MMSU -> /vol/bench_mmsu (CPU)
    modal run --detach eval/modal_eval.py::mmsu --step run --which homi,qwen-stock,gemini-3.8-flash
    modal run --detach eval/bench_prepare.py::main                        # VocalGrad, VoxParadox, MSU-Bench
    modal run --detach eval/modal_eval.py::bench --name vocalgrad --which homi,qwen-stock,gemini-3.8-flash,gpt-audio
    modal volume get homi /bench_mmsu ./bench_mmsu                   # then score locally:
    python eval/mmsu_report.py --dir ./bench_mmsu --out MMSU.md

Systems: "homi" = /vol/runs/pass2/adapter (from training/), or a downloaded kit with
--adapter-dir /vol/models/homi; "qwen-stock" = the untouched base (head = its own lm_head rows);
"gemini-<model>" needs a Modal secret "gemini" with GEMINI_API_KEY; "gpt-audio" needs a Modal secret "openai" with
OPENAI_API_KEY.

Benchmark folder (/vol/bench_<name>/): flac/<id>.flac, audio/ (bank), truth.jsonl, items<name>.jsonl, summary.json,
and runs/q<name>/<system>/{preds.jsonl, timing.json} (ours) or responses.jsonl (API models).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
INFERENCE = HERE.parent / "inference"
TRAINING = HERE.parent / "training"
sys.path.insert(0, str(INFERENCE))
sys.path.insert(0, str(TRAINING))

VOL = "/vol"
VOLUME_NAME = "homi"
MMSU_DIR = f"{VOL}/bench_mmsu"
MMSU_REPO, MMSU_REV = "ddwang2000/MMSU", "548e2283105825bf908a7db5c09c00dbcf42bd4c"
LETTERS = "ABCDEFGHIJ"
PER_ITEM_QSETS = {"mmsu", "vocalgrad", "voxparadox", "msubench"}  # public benchmarks, own question + options per item
OURS = {"homi": ("pass2", True), "qwen-stock": ("pass1", False)}  # system -> (run, use adapter)
QWEN_GPU = ["H200", "B200"]

app = modal.App("homi-eval")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.14.1", "torchvision==0.29.1", "transformers==5.18.0", "peft==0.21.2",
                    "accelerate==1.15.0", "huggingface_hub==1.33.0", "safetensors==0.8.0", "numpy==2.5.3",
                    "scipy==1.18.1", "soundfile==0.14.0", "pillow==12.3.0")
    .env({"HF_HOME": f"{VOL}/hf_cache", "HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(INFERENCE / "common.py", "/root/common.py")
    .add_local_file(INFERENCE / "readout.py", "/root/readout.py")
    .add_local_file(INFERENCE / "jev.py", "/root/jev.py")
    .add_local_file(INFERENCE / "answer_codes.json", "/root/answer_codes.json")
    .add_local_file(TRAINING / "recipe.py", "/root/recipe.py")
    .add_local_file(TRAINING / "model.py", "/root/model.py")
)
prep_image = (modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg", "ca-certificates")
              .uv_pip_install("numpy==2.5.3", "scipy==1.18.1", "soundfile==0.14.0", "pyarrow==25.0.1")
              .add_local_file(INFERENCE / "common.py", "/root/common.py")
              .add_local_file(INFERENCE / "answer_codes.json", "/root/answer_codes.json"))
gemini_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("google-genai", "numpy==2.5.3")
openai_image = modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg").uv_pip_install("openai", "numpy==2.5.3")


@app.function(image=prep_image, volumes={VOL: volume}, cpu=16, memory=65536, timeout=2 * 3600)
def mmsu_prepare() -> dict:
    """MMSU (ICLR 2026; MIT), all 5,000 questions, evaluation only. Each item keeps its own question and options in the
    published order (empty / "None" options dropped), lettered A, B, ...; gold = the option equal to answer_gt
    (the 19 questions whose answer_gt is not among the options are skipped)."""
    import io
    import subprocess
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf

    import common

    root = Path(MMSU_DIR)
    (root / "flac").mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(3):
        url = f"https://huggingface.co/datasets/{MMSU_REPO}/resolve/{MMSU_REV}/data/train-0000{k}-of-00003.parquet"
        with urllib.request.urlopen(url, timeout=600) as resp:
            rows += pq.read_table(io.BytesIO(resp.read())).to_pylist()

    def decode(audio: dict):
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", "-"],
                             input=audio["bytes"], capture_output=True, check=True).stdout
        x, _ = sf.read(io.BytesIO(out), dtype="float32")
        return x

    def one(r):
        try:
            return r, decode(r["audio"])
        except Exception as e:  # noqa: BLE001
            return r, e

    clips, truth, items, skipped = {}, [], [], {"decode": 0, "no_gold": 0}
    with ThreadPoolExecutor(16) as ex:
        for r, x in ex.map(one, rows):
            if isinstance(x, Exception) or len(x) < 800:
                skipped["decode"] += 1
                continue
            opts = [str(r[c]).strip() for c in ("choice_a", "choice_b", "choice_c", "choice_d")
                    if r.get(c) is not None and str(r[c]).strip() not in ("", "None")]
            ans = str(r["answer_gt"]).strip()
            if ans not in opts:
                skipped["no_gold"] += 1
                continue
            gi = opts.index(ans)
            cid = f"mmsu_{r['id']}"
            x = x / max(1e-6, float(np.abs(x).max())) * 0.95 if np.abs(x).max() > 0.99 else x
            clips[cid] = x.astype(np.float32)
            truth.append({"id": cid, "task": r["task_name"], "category": r["category"], "sub_category": r["sub-category"],
                          "discipline": r.get("linguistics_sub_discipline"), "question": r["question"], "options": opts,
                          "gold": gi, "answer_text": ans, "dur": round(len(x) / 16000, 2)})
            items.append({"id": f"{cid}/answer", "set": "BENCH/mmsu", "source": "bench_mmsu", "question": r["question"],
                          "options": opts, "gold": gi, "target": [1.0 if i == gi else 0.0 for i in range(len(opts))],
                          "audio": {"bank": "bench_mmsu", "key": cid}, "state_text": None, "channel": "clean",
                          "meta": {"call": cid, "question": "answer", "keys": list(LETTERS[: len(opts)]), "has_truth": True,
                                   "task": r["task_name"]}})
    for cid, x in clips.items():
        sf.write(root / "flac" / f"{cid}.flac", x, 16000, format="FLAC")
    common.write_audio_bank(root / "audio", "bench_mmsu", clips)
    common.write_jsonl(root / "truth.jsonl", truth)
    common.write_jsonl(root / "itemsmmsu.jsonl", items)
    from collections import Counter
    summary = {"rows": len(rows), "items": len(items), "skipped": skipped, "hours": round(sum(t["dur"] for t in truth) / 3600, 2),
               "by_category": dict(Counter((t["category"], t["sub_category"]) for t in truth).most_common()),
               "options": dict(Counter(len(t["options"]) for t in truth))}
    summary["by_category"] = {f"{a} / {b}": n for (a, b), n in summary["by_category"].items()}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    import shutil

    volume.commit()
    return summary


@app.function(image=gpu_image, volumes={VOL: volume}, gpu=QWEN_GPU, timeout=2 * 3600)
def bench_ours(system: str = "homi", qset: str = "mmsu", bench_dir: str = MMSU_DIR, bank: str = "bench_mmsu",
               adapter_dir: str = "") -> dict:
    """homi (or the untouched base) on every item: decision-head log-probabilities at T = 1 (the temperature does
    not change the top option). Timed after a warm-up, for the cost column."""
    import torch

    import common
    from model import load_qwen, run_eval

    run, use_adapter = OURS.get(system, (system, True))
    items = common.read_jsonl(f"{bench_dir}/items{qset}.jsonl")
    banks = common.load_banks(bench_dir, [bank])
    model, head, processor, codes, encode_fn, logprob_fn = load_qwen(run, adapter=use_adapter,
                                                                     adapter_dir=adapter_dir or None)
    with torch.inference_mode():  # warm-up
        run_eval(model, head, processor, items[:8], banks, encode_fn=encode_fn, logprob_fn=logprob_fn)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        preds = run_eval(model, head, processor, items, banks, encode_fn=encode_fn, logprob_fn=logprob_fn)
    torch.cuda.synchronize()
    seconds = time.time() - t0
    out = Path(bench_dir) / "runs" / f"q{qset}" / system
    out.mkdir(parents=True, exist_ok=True)
    common.write_jsonl(out / "preds.jsonl", preds)
    timing = {"system": system, "gpu": torch.cuda.get_device_name(), "inference_seconds": round(seconds, 2),
              "questions": len(items), "calls": len({it["meta"]["call"] for it in items})}
    (out / "timing.json").write_text(json.dumps(timing, indent=2))
    volume.commit()
    return timing


def _save_responses(d: Path, existing: dict, out: list) -> None:
    """Write the answers so far (earlier runs + this one), plus errors for ids without an answer."""
    merged = {**existing, **{r["id"]: r for r in out if "answers" in r}}
    errs = [r for r in out if "error" in r and r["id"] not in merged]
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "responses.jsonl.part"
    tmp.write_text("\n".join(json.dumps(r) for r in list(merged.values()) + errs) + "\n")
    tmp.replace(d / "responses.jsonl")
    volume.commit()


def _mmsu_prompt(t: dict) -> str:
    opts = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(t["options"]))
    keys = ", ".join(f'"{LETTERS[i]}": <number>' for i in range(len(t["options"])))
    return ("You are listening to an audio clip. Answer the multiple-choice question about the AUDIO. Give a probability "
            "for EVERY option (numbers between 0 and 1 that sum to 1) and the single best option.\n\n"
            f"Question: {t['question']}\nOptions:\n{opts}\n\n"
            'Reply with JSON only, in this shape: {"answer": "<letter>", "probabilities": {' + keys + "}}")


def _existing(d: Path) -> dict:
    existing = {}
    if (d / "responses.jsonl").exists():
        for line in (d / "responses.jsonl").read_text().splitlines():
            r = json.loads(line)
            if "answers" in r:
                existing[r["id"]] = r
    return existing


@app.function(image=gemini_image, volumes={VOL: volume}, secrets=[modal.Secret.from_name("gemini")], timeout=3 * 3600)
def bench_gemini(model_name: str = "gemini-3.8-flash", workers: int = 8, qset: str = "mmsu", bench_dir: str = MMSU_DIR,
                 sample: int = 0, seed: int = 13) -> dict:
    """Gemini on a per-item benchmark: audio first, then _mmsu_prompt. sample > 0 asks a seeded random sample.
    Questions already answered in responses.jsonl are not re-asked; stops if the first request is refused."""
    import os
    import random
    import re
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from google import genai
    from google.genai import types

    assert qset in PER_ITEM_QSETS, "API models are only run on public benchmarks"
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    truth = [json.loads(l) for l in Path(bench_dir, "truth.jsonl").read_text().splitlines()]
    if sample:
        truth = random.Random(seed).sample(truth, min(sample, len(truth)))
    d = Path(bench_dir) / "runs" / f"q{qset}" / f"gemini:{model_name}"
    existing = _existing(d)

    def ask(t):
        data = Path(bench_dir, "flac", f"{t['id']}.flac").read_bytes()
        for attempt in range(5):
            try:
                t0 = time.time()
                resp = client.models.generate_content(
                    model=model_name, contents=[types.Part.from_bytes(data=data, mime_type="audio/flac"), _mmsu_prompt(t)],
                    config=types.GenerateContentConfig(response_mime_type="application/json"))
                ms = (time.time() - t0) * 1000
                txt = resp.text or ""
                m = re.search(r"\{.*\}", txt, re.S)
                ans = json.loads(m.group(0)) if m else {}
                u = resp.usage_metadata
                details = {str(d.modality).split(".")[-1]: d.token_count for d in (u.prompt_tokens_details or [])}
                return {"id": t["id"], "answers": ans, "latency_ms": round(ms), "usage": {
                    "prompt": u.prompt_token_count, "output": u.candidates_token_count or 0,
                    "thinking": getattr(u, "thoughts_token_count", 0) or 0, "prompt_by_modality": details}}
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {str(e)[:300]}"
                time.sleep(5 * (attempt + 1))
        return {"id": t["id"], "error": err}

    todo = [t for t in truth if t["id"] not in existing]
    out = []
    if todo:
        first = ask(todo[0])
        if "error" in first and any(k in first["error"] for k in ("RESOURCE_EXHAUSTED", "429", "402", "404", "NOT_FOUND")):
            return {"model": model_name, "ok": 0, "errors": [first["error"]],
                    "stopped": "refused on the first request (quota or model unavailable)"}
        out = [first]
        with ThreadPoolExecutor(workers) as ex:  # partial results are saved as we go
            for k, fut in enumerate(as_completed([ex.submit(ask, t) for t in todo[1:]]), 1):
                out.append(fut.result())
                if k % 300 == 0:
                    _save_responses(d, existing, out)
    _save_responses(d, existing, out)
    if sample:
        (d / f"sample_{sample}_seed{seed}.json").write_text(json.dumps(sorted(t["id"] for t in truth)))
        volume.commit()
    ok = [r for r in out if "answers" in r]
    return {"model": model_name, "ok": len(ok), "errors": [r["error"] for r in out if "error" in r][:3],
            "prompt_tokens": sum(r["usage"]["prompt"] for r in ok), "output_tokens": sum(r["usage"]["output"] for r in ok),
            "thinking_tokens": sum(r["usage"]["thinking"] for r in ok)}


OPENAI_PRICES = {"gpt-audio": {"audio": 32.0, "text": 2.5, "output": 10.0}}  # $ / 1M tokens, checked 2026-10-21
# Without it, gpt-audio often replies that it cannot hear audio although the clip was received.
OPENAI_SYSTEM = ("You are an audio analysis assistant. Each user message contains an audio recording, which you can hear. "
                 "Listen to it and answer the question about it in the requested JSON format.")


@app.function(image=openai_image, volumes={VOL: volume}, secrets=[modal.Secret.from_name("openai")], timeout=8 * 3600)
def bench_openai(model_name: str = "gpt-audio", workers: int = 16, qset: str = "mmsu", bench_dir: str = MMSU_DIR,
                 sample: int = 0, seed: int = 13) -> dict:
    """OpenAI audio model (Chat Completions, audio in, text out) on a per-item benchmark with the Gemini prompt
    (_mmsu_prompt); responses are saved in the Gemini format. A reply without a parsable answer is re-asked up to
    twice, then kept as {"refused": true, ...}, which counts as wrong."""
    import base64
    import random
    import re
    import subprocess
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import openai

    assert qset in PER_ITEM_QSETS, "API models are only run on public benchmarks"
    client = openai.OpenAI(timeout=180, max_retries=0)  # reads OPENAI_API_KEY from the Modal secret
    truth = [json.loads(l) for l in Path(bench_dir, "truth.jsonl").read_text().splitlines()]
    if sample:
        truth = random.Random(seed).sample(truth, min(sample, len(truth)))
    d = Path(bench_dir) / "runs" / f"q{qset}" / f"openai:{model_name}"
    existing = _existing(d)

    def ask(t):
        wav = subprocess.run(["ffmpeg", "-v", "error", "-i", str(Path(bench_dir, "flac", f"{t['id']}.flac")), "-ac", "1",
                              "-ar", "16000", "-f", "wav", "-"], capture_output=True, check=True).stdout
        messages = [{"role": "system", "content": OPENAI_SYSTEM}, {"role": "user", "content": [  # audio first
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}},
            {"type": "text", "text": _mmsu_prompt(t)}]}]
        err, no_answer, spent, spent_audio = "no attempt", 0, None, 0
        for attempt in range(8):
            try:
                t0 = time.time()
                r = client.chat.completions.create(model=model_name, modalities=["text"], messages=messages)
                ms = (time.time() - t0) * 1000
                txt = r.choices[0].message.content or ""
                u = r.usage
                details = getattr(u, "prompt_tokens_details", None) if u else None
                audio = (getattr(details, "audio_tokens", 0) or 0) if details else 0
                usage = {"prompt": (u.prompt_tokens if u else 0) or 0, "output": (u.completion_tokens if u else 0) or 0,
                         "thinking": 0, "prompt_by_modality": {"AUDIO": audio}}
                m = re.search(r"\{.*\}", txt, re.S)
                try:
                    ans = json.loads(m.group(0)) if m else {}
                except ValueError:
                    ans = {}
                spent = {k: spent.get(k, 0) + usage[k] for k in ("prompt", "output")} if spent else dict(usage)
                spent_audio += audio
                total = {"prompt": spent["prompt"], "output": spent["output"], "thinking": 0,
                         "prompt_by_modality": {"AUDIO": spent_audio}}  # every attempt is billed
                if not isinstance(ans, dict) or not ans.get("answer"):
                    if no_answer < 2:  # no answer in the reply: ask again (at most twice)
                        no_answer += 1
                        continue
                    return {"id": t["id"], "answers": {}, "refused": True, "text": txt[:300], "latency_ms": round(ms),
                            "usage": total, "attempts": no_answer + 1}
                return {"id": t["id"], "answers": ans, "latency_ms": round(ms), "usage": total, "attempts": no_answer + 1}
            except openai.APIStatusError as e:
                err = f"HTTP {e.status_code}: {str(e)[:300]}"
                if e.status_code in (400, 401, 403, 404):
                    break
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {str(e)[:300]}"
            time.sleep(5 * (attempt + 1))
        return {"id": t["id"], "error": err}

    todo = [t for t in truth if t["id"] not in existing]
    out = []
    if todo:
        first = ask(todo[0])
        if "error" in first:
            return {"model": model_name, "ok": 0, "errors": [first["error"]], "stopped": "refused on the first request"}
        out = [first]
        with ThreadPoolExecutor(workers) as ex:
            for k, fut in enumerate(as_completed([ex.submit(ask, t) for t in todo[1:]]), 1):
                out.append(fut.result())
                if k % 300 == 0:
                    _save_responses(d, existing, out)
    _save_responses(d, existing, out)
    allr = {**existing, **{r["id"]: r for r in out if "answers" in r}}
    pr = OPENAI_PRICES.get(model_name, OPENAI_PRICES["gpt-audio"])
    cost = sum(((r["usage"]["prompt_by_modality"]["AUDIO"]) * pr["audio"] + (r["usage"]["prompt"] - r["usage"]["prompt_by_modality"]["AUDIO"])
                * pr["text"] + r["usage"]["output"] * pr["output"]) / 1e6 for r in allr.values() if r.get("usage"))
    return {"model": model_name, "qset": qset, "answered": sum(1 for r in allr.values() if not r.get("refused")),
            "refused": sum(1 for r in allr.values() if r.get("refused")), "errors": [r["error"] for r in out if "error" in r][:3],
            "n_errors": sum(1 for r in out if "error" in r), "cost_usd": round(cost, 2)}


def _spawn(system: str, qset: str, bench_dir: str, bank: str, workers: int, adapter_dir: str):
    if system.startswith("gemini"):
        return bench_gemini.spawn(system, workers, qset, bench_dir)
    if system.startswith("gpt"):
        return bench_openai.spawn(system, workers, qset, bench_dir)
    return bench_ours.spawn(system, qset, bench_dir, bank, adapter_dir)


@app.local_entrypoint()
def mmsu(step: str = "prepare", which: str = "homi,qwen-stock,gemini-3.8-flash", workers: int = 32, adapter_dir: str = ""):
    """--step prepare: build /vol/bench_mmsu. --step run: spawn one job per system and return (use --detach)."""
    if step == "prepare":
        print(json.dumps(mmsu_prepare.remote(), indent=2))
        return
    for s in [w.strip() for w in which.split(",") if w.strip()]:
        print("mmsu", s, _spawn(s, "mmsu", MMSU_DIR, "bench_mmsu", workers, adapter_dir).object_id)


@app.local_entrypoint()
def bench(name: str = "vocalgrad", which: str = "homi,qwen-stock,gemini-3.8-flash", workers: int = 16,
          adapter_dir: str = ""):
    """A benchmark prepared by eval/bench_prepare.py (vocalgrad | voxparadox | msubench)."""
    assert name in PER_ITEM_QSETS - {"mmsu"}, name
    for s in [w.strip() for w in which.split(",") if w.strip()]:
        print(name, s, _spawn(s, name, f"{VOL}/bench_{name}", f"bench_{name}", workers, adapter_dir).object_id)
