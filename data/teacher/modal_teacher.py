"""Soft labels from the text teacher for the text items of sb_jade and sb_tasksource, on Modal (H200).

    modal run --detach data/teacher/modal_teacher.py::main
"""

from __future__ import annotations

import collections
import glob
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import teacher_core as tc  # noqa: E402

app = modal.App("egma-homi-stageb-teacher")
volume = modal.Volume.from_name("egma-homi")
secrets = [modal.Secret.from_name("huggingface")]

OUT = "/vol/stageb/teacher"
PARTS = f"{OUT}/parts"
RUNS = f"{OUT}/runs"  # one stats file per GPU call
SOURCES = {
    "sb_jade": "/vol/stageb/sb_jade/items_train.jsonl",
    "sb_tasksource": "/vol/stageb/sb_tasksource/items_train.jsonl",
}
JADE_CAL = "/vol/stageb/sb_jade/items_cal.jsonl"
TS_SUMMARY = "/vol/stageb/sb_tasksource/summary.json"
TASK_START = 1791505680.0  # 2026-10-08 17:28 PDT; wait_hours for sb_tasksource counts from here
STABLE_SECONDS = 600  # sb_tasksource summary.json must be unchanged this long to count as final

GPU = "H200"
N_GPU = 2  # GPUs used at once
TOKEN_BUDGET = 32_768  # padded tokens per forward pass
MAX_ROWS = 256
COMMIT_EVERY = 300  # seconds

ENV = {
    "HF_HOME": "/vol/hf_cache",
    "HF_XET_HIGH_PERFORMANCE": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(  # the teacher's pinned requirements.txt without its server / training packages
        "torch==2.14.0",
        "torchvision==0.29.0",
        "transformers==5.17.0",
        "accelerate==1.15.0",
        "pillow==12.3.0",
        "kernels==0.16.1",
        "flash-linear-attention==0.5.2",
        "safetensors==0.8.0",
        "numpy==2.5.3",
        "huggingface-hub==1.31.0",
    )
    .env(ENV)
    .add_local_file(HERE / "teacher_core.py", "/root/teacher_core.py")
)
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface-hub==1.31.0")
    .env(ENV)
    .add_local_file(HERE / "teacher_core.py", "/root/teacher_core.py")
)


def _hf_env():
    for k in ("HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_HUB_TOKEN"):
        if os.environ.get(k) and not os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = os.environ[k]


def _read_jsonl(path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # a line cut by a preempted container
                pass
    return rows


def _log(msg: str, **kw):
    print(json.dumps({"t": time.strftime("%H:%M:%S"), "msg": msg, **kw}), flush=True)


# Teacher (GPU)


def _load_teacher():
    _hf_env()
    from huggingface_hub import snapshot_download

    t0 = time.time()
    ckpt = Path(snapshot_download(tc.REPO, revision=tc.REVISION))
    src = str(ckpt / "source" / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from autojev.model import DecisionModel

    model = DecisionModel(ckpt, device="cuda")
    _log("teacher loaded", seconds=round(time.time() - t0, 1), temperature=model.temperature,
         attention=model.attention_mode, pooling=model.pooling)
    return model


def _prompt_lengths(model, rows) -> list[int]:
    from autojev.model import decision_messages

    texts = [
        model.processor.apply_chat_template(
            decision_messages(r, model.codes), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for r in rows
    ]
    out = []
    for s in range(0, len(texts), 4096):
        out += [len(x) for x in model.processor.tokenizer(texts[s : s + 4096], add_special_tokens=False)["input_ids"]]
    return out


def _forward(model, rows) -> list[list[float]]:
    """DecisionModel.predict for one batch (softmax of logits / saved temperature); halves the batch on CUDA OOM."""
    import torch

    try:
        with torch.inference_mode():
            batch = model.prepare(rows)
            probs = (model(batch) / model.temperature).softmax(-1).cpu().tolist()
        return [p[:c] for p, c in zip(probs, batch.counts, strict=True)]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(rows) == 1:
            raise
        h = len(rows) // 2
        _log("OOM, splitting batch", rows=len(rows))
        return _forward(model, rows[:h]) + _forward(model, rows[h:])


def _score(model, items, yes_no_mode, on_batch=None, budget=TOKEN_BUDGET, max_rows=MAX_ROWS):
    """Score items -> {index: {"probs": ours, "n_tok": int}} plus a set of indices skipped as too long."""
    conv = [tc.to_teacher_row(it, yes_no_mode) for it in items]
    rows = [c[0] for c in conv]
    lengths = _prompt_lengths(model, rows)
    too_long = {i for i, n in enumerate(lengths) if n > tc.MAX_TOKENS}
    keep = [i for i in range(len(items)) if i not in too_long]
    batches = tc.make_batches([lengths[i] for i in keep], budget, max_rows)
    results = {}
    t0, done_tok = time.time(), 0
    for b_no, b in enumerate(batches):
        idx = [keep[j] for j in b]
        probs = _forward(model, [rows[i] for i in idx])
        new = {}
        for i, p in zip(idx, probs, strict=True):
            new[i] = {"probs": tc.map_back(p, conv[i][1]), "n_tok": lengths[i]}
            done_tok += lengths[i]
        results.update(new)
        if on_batch:
            on_batch(new)
        if b_no % 50 == 0 or b_no == len(batches) - 1:
            el = time.time() - t0
            _log("progress", batch=b_no + 1, of=len(batches), items=len(results), of_items=len(keep),
                 tok_per_s=round(done_tok / max(el, 1e-6)), elapsed_s=round(el))
    return results, too_long


gpu_kw = dict(
    image=gpu_image, gpu=GPU, volumes={"/vol": volume}, secrets=secrets, cpu=8, memory=65536, max_containers=N_GPU
)


@app.function(**gpu_kw, timeout=7200)
def smoke(n: int = 50) -> dict:
    """Accuracy and latency on n seeded sb_jade CAL items, then the whole CAL file batched (both yes/no mappings)."""
    import torch

    t_start = time.time()
    cal = [it for it in _read_jsonl(JADE_CAL) if tc.skip_reason(it) is None]
    sample = random.Random(0).sample(cal, n)
    model = _load_teacher()

    # 0. the model card's own example (expect "support")
    from autojev.model import answer

    q = {"type": "choice", "instructions": "Which team should handle this request?",
         "criteria": {"billing": "Charges and refunds", "support": "Technical integration errors",
                      "sales": "Questions about buying a product"}}
    card = answer(q, model.predict([{"state": "My integration keeps failing. Please help.", "question": q}])[0])

    # 1. one request at a time (latency)
    for it in sample[:3]:  # warm-up
        _forward(model, [tc.to_teacher_row(it)[0]])
    lat, single = [], []
    for it in sample:
        row, mapping = tc.to_teacher_row(it)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        p = _forward(model, [row])[0]
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
        single.append(tc.map_back(p, mapping))
    acc50 = sum(tc.argmax(p) == it["gold"] for p, it in zip(single, sample)) / n

    # 2. whole CAL file, batched, both yes/no mappings
    t0 = time.time()
    res_noul, _ = _score(model, cal, "noul")
    cal_s = time.time() - t0
    yn = [i for i, it in enumerate(cal) if tc.is_yes_no(it["options"])]
    res_choice, _ = _score(model, [cal[i] for i in yn], "choice")

    def acc(res, idxs, items):
        ok = [tc.argmax(res[i]["probs"]) == items[i]["gold"] for i in idxs if i in res]
        return round(sum(ok) / max(len(ok), 1), 4), len(ok)

    def nll(res, idxs, items):
        v = [-math.log(max(res[i]["probs"][items[i]["gold"]], 1e-9)) for i in idxs if i in res]
        return round(sum(v) / max(len(v), 1), 4)

    yn_items = [cal[i] for i in yn]
    by_suite = collections.defaultdict(list)
    for i, it in enumerate(cal):
        if i in res_noul:
            by_suite[it["meta"].get("suite", "?")].append(int(tc.argmax(res_noul[i]["probs"]) == it["gold"]))
    pos = {it["id"]: i for i, it in enumerate(cal)}
    batched_vs_single = max(
        max(abs(a - b) for a, b in zip(res_noul[pos[it["id"]]]["probs"], p)) for it, p in zip(sample, single)
    )
    tokens = sum(r["n_tok"] for r in res_noul.values())
    out = {
        "teacher": tc.REPO, "revision": tc.REVISION, "temperature": model.temperature,
        "model_card_example": card,
        "sample50": {"n": n, "accuracy": round(acc50, 4), "median_latency_ms": round(1000 * statistics.median(lat), 1),
                     "p90_latency_ms": round(1000 * sorted(lat)[int(0.9 * n)], 1)},
        "cal_all": {"n": len(res_noul), "accuracy": acc(res_noul, range(len(cal)), cal)[0],
                    "nll": nll(res_noul, range(len(cal)), cal),
                    "seconds": round(cal_s, 1), "items_per_s": round(len(res_noul) / cal_s, 2),
                    "tokens": tokens, "tokens_per_s": round(tokens / cal_s),
                    "by_suite": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_suite.items())}},
        "yes_no": {"n": len(yn), "noul_acc": acc(res_noul, yn, cal)[0], "noul_nll": nll(res_noul, yn, cal),
                   "choice_acc": acc(res_choice, range(len(yn)), yn_items)[0],
                   "choice_nll": nll(res_choice, range(len(yn)), yn_items)},
        "max_abs_diff_batched_vs_single": round(batched_vs_single, 5),
        "gpu_seconds": round(time.time() - t_start, 1),
    }
    Path(OUT).mkdir(parents=True, exist_ok=True)
    Path(f"{OUT}/smoke.json").write_text(json.dumps(out, indent=2) + "\n")
    volume.commit()
    _log("smoke done", **out)
    return out


@app.function(**gpu_kw, timeout=8 * 3600, retries=modal.Retries(max_retries=2, initial_delay=15.0))
def score_shard(source: str, shard: int, n_shards: int, yes_no_mode: str, round_tag: str) -> dict:
    t_start = time.time()
    volume.reload()
    items = [it for it in _read_jsonl(SOURCES[source]) if tc.skip_reason(it) is None]
    mine = [it for it in items if tc.shard_of(it["id"], n_shards) == shard]
    done = set()
    for p in glob.glob(f"{PARTS}/{source}/*.jsonl"):
        done |= {(r["id"], r["h"]) for r in _read_jsonl(p)}
    todo = [it for it in mine if (it["id"], tc.item_hash(it)) not in done]
    stats = {"source": source, "shard": shard, "round": round_tag, "eligible": len(mine), "todo": len(todo)}
    _log("shard start", **stats)
    if not todo:
        stats["gpu_seconds"] = round(time.time() - t_start, 1)
        _save_run(stats, t_start)
        return stats
    model = _load_teacher()
    part = Path(f"{PARTS}/{source}/{round_tag}-shard{shard}of{n_shards}.jsonl")
    part.parent.mkdir(parents=True, exist_ok=True)
    if part.exists() and part.stat().st_size and not part.read_bytes().endswith(b"\n"):
        with open(part, "a") as f:
            f.write("\n")
    last_commit = [time.time()]
    with open(part, "a") as f:

        def sink(new):
            for i, r in new.items():
                it = todo[i]
                f.write(json.dumps({"id": it["id"], "h": tc.item_hash(it), "probs": r["probs"],
                                    "n_tok": r["n_tok"], "mode": yes_no_mode}) + "\n")
            f.flush()
            if time.time() - last_commit[0] > COMMIT_EVERY:
                volume.commit()
                last_commit[0] = time.time()

        t0 = time.time()
        results, too_long = _score(model, todo, yes_no_mode, on_batch=sink)
        score_s = time.time() - t0
        for i in sorted(too_long):
            it = todo[i]
            f.write(json.dumps({"id": it["id"], "h": tc.item_hash(it), "skip": "too_long"}) + "\n")
    volume.commit()
    tokens = sum(r["n_tok"] for r in results.values())
    stats.update(scored=len(results), too_long=len(too_long), tokens=tokens, score_seconds=round(score_s, 1),
                 tokens_per_s=round(tokens / max(score_s, 1e-6)), gpu_seconds=round(time.time() - t_start, 1))
    _save_run(stats, t_start)
    _log("shard done", **stats)
    return stats


def _save_run(stats: dict, t_start: float):
    Path(RUNS).mkdir(parents=True, exist_ok=True)
    name = f"{stats['source']}-{stats['round']}-shard{stats['shard']}-{int(t_start)}.json"
    Path(f"{RUNS}/{name}").write_text(json.dumps(stats) + "\n")
    volume.commit()


# Orchestration and merge (CPU)

cpu_kw = dict(image=cpu_image, volumes={"/vol": volume}, secrets=secrets)


@app.function(**cpu_kw, cpu=8, memory=16384, timeout=3600)
def download() -> str:
    _hf_env()
    from huggingface_hub import snapshot_download

    t0 = time.time()
    path = snapshot_download(tc.REPO, revision=tc.REVISION)
    volume.commit()
    _log("downloaded", path=path, seconds=round(time.time() - t0))
    return path


def _ts_state() -> dict:
    """Current state of the sb_tasksource builder's outputs."""
    volume.reload()
    st = {"items": os.path.exists(SOURCES["sb_tasksource"]), "summary": os.path.exists(TS_SUMMARY)}
    if st["summary"]:
        raw = Path(TS_SUMMARY).read_bytes()
        st["summary_sha"] = hashlib.sha1(raw).hexdigest()[:16]
        try:
            s = json.loads(raw)
            st["not_final_flag"] = bool(s.get("final") is False or str(s.get("status", "")).lower()
                                        in ("partial", "in_progress", "building", "running"))
        except json.JSONDecodeError:
            st["not_final_flag"] = True
    if st["items"]:
        st["items_bytes"] = os.path.getsize(SOURCES["sb_tasksource"])
    return st


def _wait_for_tasksource(deadline: float) -> dict:
    """Poll until sb_tasksource's summary.json and items are final and unchanged for STABLE_SECONDS, or `deadline`."""
    seen, since = None, time.time()
    while True:
        st = _ts_state()
        key = (st.get("summary_sha"), st.get("items_bytes"))
        if key != seen:
            seen, since = key, time.time()
            if st["summary"] and st["items"]:  # unchanged since the files were last written
                since = max(os.path.getmtime(TS_SUMMARY), os.path.getmtime(SOURCES["sb_tasksource"]))
            _log("sb_tasksource state", **st, stable_since=time.strftime("%H:%M:%S", time.gmtime(since)))
        if st["summary"] and st["items"] and not st.get("not_final_flag") and time.time() - since >= STABLE_SECONDS:
            return {**st, "final": True, "waited_until": time.strftime("%H:%M:%S")}
        if time.time() >= deadline:
            return {**st, "final": False, "gave_up_at": time.strftime("%H:%M:%S")}
        time.sleep(60)


def _merge(meta: dict) -> dict:
    volume.reload()
    out_rows = []
    per_source = {}
    for source, path in SOURCES.items():
        if not os.path.exists(path):
            continue
        labels = {}
        for p in sorted(glob.glob(f"{PARTS}/{source}/*.jsonl")):
            for r in _read_jsonl(p):
                labels[(r["id"], r["h"])] = r
        if not labels:
            continue
        skipped = collections.Counter()
        n = acc = 0
        ent = ent_norm = 0.0
        by_q = collections.defaultdict(lambda: [0, 0])
        by_suite = collections.defaultdict(lambda: [0, 0])
        for it in _read_jsonl(path):
            reason = tc.skip_reason(it)
            if reason:
                skipped[reason] += 1
                continue
            r = labels.get((it["id"], tc.item_hash(it)))
            if r is None:
                skipped["not_scored"] += 1
                continue
            if r.get("skip"):
                skipped[r["skip"]] += 1
                continue
            probs = r["probs"]
            if len(probs) != len(it["options"]):
                skipped["length_mismatch"] += 1
                continue
            ok = int(tc.argmax(probs) == it["gold"])
            out_rows.append({"id": it["id"], "probs": probs, "teacher_acc_gold": ok})
            n += 1
            acc += ok
            e = tc.entropy(probs)
            ent += e
            ent_norm += e / math.log(len(probs))
            q = (it.get("meta") or {}).get("qtype", "?")
            by_q[q][0] += ok
            by_q[q][1] += 1
            m = it.get("meta") or {}
            suite = m.get("suite") or m.get("task") or m.get("dataset") or it.get("source", "?")
            by_suite[suite][0] += ok
            by_suite[suite][1] += 1
        per_source[source] = {
            "items_file": path,
            "scored": n,
            "skipped": dict(skipped),
            "accuracy_vs_gold": round(acc / max(n, 1), 4),
            "mean_entropy_nats": round(ent / max(n, 1), 4),
            "mean_entropy_normalised": round(ent_norm / max(n, 1), 4),
            "accuracy_by_qtype": {k: {"acc": round(a / t, 4), "n": t} for k, (a, t) in sorted(by_q.items())},
            "accuracy_by_suite": {k: {"acc": round(a / t, 4), "n": t} for k, (a, t) in sorted(by_suite.items())},
        }
    tmp = Path(f"{OUT}/soft_labels.jsonl.tmp")
    with open(tmp, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, f"{OUT}/soft_labels.jsonl")
    n_all = len(out_rows)
    summary = {
        "teacher": tc.REPO,
        "teacher_revision": tc.REVISION,
        "teacher_licence": "Apache-2.0",
        "inference": "teacher's own source/src/autojev/model.py (DecisionModel: noncausal full attention, last-token "
                     "pooling, 255-row readout), transformers 5.17.0, bf16, H200",
        "temperature_used": meta.get("temperature"),
        "yes_no_mapping": meta.get("yes_no_mode"),
        "max_options": tc.MAX_OPTIONS,
        "max_tokens": tc.MAX_TOKENS,
        "rows": n_all,
        "row_format": {"id": "item id", "probs": "teacher probability per option, in our option order",
                       "teacher_acc_gold": "1 if argmax(probs) == item gold"},
        "per_source": per_source,
        "accuracy_vs_gold_all": round(sum(r["teacher_acc_gold"] for r in out_rows) / max(n_all, 1), 4),
        **{k: v for k, v in meta.items() if k not in ("temperature", "yes_no_mode")},
    }
    tmp = Path(f"{OUT}/summary.json.tmp")
    tmp.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(tmp, f"{OUT}/summary.json")
    volume.commit()
    return summary


def _missing(source: str) -> int:
    volume.reload()
    path = SOURCES[source]
    if not os.path.exists(path):
        return 0
    done = set()
    for p in glob.glob(f"{PARTS}/{source}/*.jsonl"):
        done |= {(r["id"], r["h"]) for r in _read_jsonl(p)}
    return sum(1 for it in _read_jsonl(path) if tc.skip_reason(it) is None and (it["id"], tc.item_hash(it)) not in done)


@app.function(**cpu_kw, cpu=2, memory=16384, timeout=12 * 3600)
def orchestrate(yes_no_mode: str = "noul", wait_hours: float = 4.0, include_tasksource: bool = True) -> dict:
    t_start = time.time()
    download.remote()
    runs = []

    def run_round(source, tag):
        res = list(score_shard.starmap([(source, s, N_GPU, yes_no_mode, tag) for s in range(N_GPU)]))
        runs.extend(res)
        _log("round done", source=source, round=tag, results=res)

    run_round("sb_jade", "r1")
    if _missing("sb_jade"):
        run_round("sb_jade", "r2")

    ts = {"included": False}
    if include_tasksource:
        deadline = TASK_START + wait_hours * 3600
        st = _wait_for_tasksource(deadline)
        ts = {"included": st["items"], **st}
        if st["items"]:
            for k in range(1, 4):
                run_round("sb_tasksource", f"r{k}")
                st2 = _ts_state()
                if st2.get("summary_sha") != st.get("summary_sha") or st2.get("items_bytes") != st.get("items_bytes"):
                    _log("sb_tasksource changed while scoring; waiting for it to settle again")
                    st = _wait_for_tasksource(max(deadline, time.time() + STABLE_SECONDS + 60))
                    ts.update(st, rescored_after_change=True)
                if not _missing("sb_tasksource"):
                    break

    smoke_info = {}
    if os.path.exists(f"{OUT}/smoke.json"):
        smoke_info = json.loads(Path(f"{OUT}/smoke.json").read_text())
    volume.reload()
    runs = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(f"{RUNS}/*.json"))]
    gpu_s = sum(r.get("gpu_seconds", 0) for r in runs) + smoke_info.get("gpu_seconds", 0)
    cpu_h = (time.time() - t_start) / 3600
    cost = gpu_s / 3600 * tc.H200_PER_HOUR + gpu_s / 3600 * (8 * 0.0473 + 64 * 0.008) + cpu_h * (2 * 0.0473 + 16 * 0.008)
    meta = {
        "temperature": smoke_info.get("temperature", 1.0087417621345625),
        "yes_no_mode": yes_no_mode,
        "sb_tasksource_status": ts,
        "smoke_test": {k: smoke_info.get(k) for k in ("sample50", "cal_all", "yes_no", "max_abs_diff_batched_vs_single")},
        "runs": [r for r in runs if r.get("todo")],
        "cost": {"gpu_seconds_h200": round(gpu_s), "estimated_usd": round(cost, 2),
                 "note": "wall-clock estimate incl. model loads, CPU/memory and orchestration; excludes weight download"},
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    summary = _merge(meta)
    _log("done", rows=summary["rows"], acc=summary["accuracy_vs_gold_all"], cost=summary["cost"])
    return summary


@app.local_entrypoint()
def smoke_main(n: int = 50):
    download.remote()
    print(json.dumps(smoke.remote(n), indent=2))


@app.local_entrypoint()
def main(yes_no_mode: str = "noul", wait_hours: float = 4.0, include_tasksource: bool = True):
    print(json.dumps(orchestrate.remote(yes_no_mode, wait_hours, include_tasksource), indent=2)[:4000])
