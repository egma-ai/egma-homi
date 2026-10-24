"""
egma-homi training on Modal: base data, data builders (data/), audit, LoRA + decision head on 8 GPUs, temperature.

    modal run training/modal_train.py::prepare             # base items (CREMA-D, MInDS-14, ...) + Qwen3-Omni weights
    modal run training/modal_train.py::jade                # JADE-Data text decisions -> /vol/stageb/sb_jade
    (then every data builder in data/, see README.md)
    modal run training/modal_train.py::audit --run pass1   # CPU dry run: counts, fails if excluded data is left
    modal run --detach training/modal_train.py::train --run pass1       # 8 x H200 or B200 in one container
    modal run --detach training/modal_train.py::calibrate --run pass1   # predictions on the CAL split + temperature
    modal run --detach training/modal_train.py::train --run pass2       # continues from pass1
    modal run --detach training/modal_train.py::calibrate --run pass2
    modal run training/modal_train.py::smoke               # 10 steps of the pass-1 recipe on 2 GPUs

Data and runs live in the Modal Volume VOLUME_NAME (mounted at /vol; layout in training/recipe.py). The model is
/vol/runs/<run>/adapter/{adapter_model.safetensors, adapter_config.json, readout.safetensors} plus
/vol/runs/<run>/temperature_lora.json (shipped in the Hugging Face kit as temperature.json).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
INFERENCE = HERE.parent / "inference"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(INFERENCE))

import recipe  # noqa: E402
from recipe import (CKPT_EVERY, DATA_DIR, EFFECTIVE_BATCH, LR, SMOKE_CKPT_EVERY, SMOKE_TRAIN_STEPS,  # noqa: E402
                    VOL, all_banks, drop_excluded, fit_and_save_temperature, is_smoke, load_train_items, run_config,
                    run_dir, select_eval_items, stageb_items, valid_items)

VOLUME_NAME = "egma-homi"  # your Modal Volume (created on first use)
QWEN_GPU = ["H200", "B200"]
RETRIES = modal.Retries(max_retries=3, initial_delay=30.0)  # interrupted GPU steps restart (and resume)

app = modal.App("egma-homi-train")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.14.1", "torchvision==0.29.1", "transformers==5.18.0", "peft==0.21.2",
                    "accelerate==1.15.0", "huggingface_hub==1.33.0", "safetensors==0.8.0", "numpy==2.5.3",
                    "scipy==1.18.1", "pyarrow==25.0.1", "soundfile==0.14.0", "pillow==12.3.0")
    .env({"HF_HOME": f"{VOL}/hf_cache", "HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(INFERENCE / "common.py", "/root/common.py")
    .add_local_file(INFERENCE / "readout.py", "/root/readout.py")
    .add_local_file(INFERENCE / "jev.py", "/root/jev.py")
    .add_local_file(INFERENCE / "answer_codes.json", "/root/answer_codes.json")
    .add_local_file(HERE / "recipe.py", "/root/recipe.py")
    .add_local_file(HERE / "model.py", "/root/model.py")
    .add_local_file(HERE / "prepare_data.py", "/root/prepare_data.py")
)
# a fresh container per call always sees the latest committed Volume state
common_kwargs = dict(image=image, volumes={VOL: volume}, single_use_containers=True)


@app.function(**common_kwargs, cpu=4, memory=32768, timeout=3 * 3600)
def prepare_base(force: bool = False) -> dict:
    """Base manifests + audio banks (CPU only, training/prepare_data.py), then the Qwen3-Omni weights into the Volume."""
    import prepare_data
    from huggingface_hub import snapshot_download

    from model import QWEN_OMNI_ID, QWEN_OMNI_REV

    summary = prepare_data.main(DATA_DIR, force=force)
    volume.commit()
    t0 = time.time()
    snapshot_download(QWEN_OMNI_ID, revision=QWEN_OMNI_REV)  # ~70 GB, skipped if already cached
    volume.commit()
    print(f"[prepare] model weights ready ({time.time() - t0:.0f}s)")
    return {k: summary[k] for k in ("train_items_by_source", "train_items_total", "banking_share_of_train",
                                    "eval_items_by_set")}


JADE_TRAIN, JADE_CAL, JADE_MAX_OPTIONS = 40_000, 800, 40


@app.function(**common_kwargs, cpu=4, memory=16384, timeout=2 * 3600)
def build_jade() -> dict:
    """JADE-Data (Apache-2.0, synthetic) -> /vol/stageb/sb_jade/. Seeded sample spread over suites; probabilistic
    yes/no targets are kept as soft targets [p, 1 - p]."""
    import random
    import urllib.request
    from collections import Counter, defaultdict

    import common
    import prepare_data as P

    rows = []
    with urllib.request.urlopen(P.JADE_URL, timeout=1800) as resp:
        for line in resp:
            r = json.loads(line)
            if r.get("source", {}).get("input_tokens", 10**9) <= P.JADE_MAX_INPUT_TOKENS:
                rows.append(r)
    by_suite = defaultdict(list)
    for r in rows:
        by_suite[r["suite"]].append(r)
    rng = random.Random(f"{common.SEED}:jade-v1")
    for v in by_suite.values():
        rng.shuffle(v)
    picked, want = [], JADE_TRAIN + JADE_CAL
    while len(picked) < want and any(by_suite.values()):  # round-robin over suites
        for suite in sorted(by_suite):
            if by_suite[suite] and len(picked) < want:
                picked.append(by_suite[suite].pop())
    items = []
    for r in picked:
        q = r["question"]
        state = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False)
        if q["type"] == "noul":
            crit = q.get("criteria") or {}
            question = q["instructions"] + (f" (yes = {crit['true']}; no = {crit['false']})"
                                             if "true" in crit and "false" in crit else "")
            t = r["target"]
            p_yes = 1.0 if t is True else 0.0 if t is False else float(t)
            if not 0.0 <= p_yes <= 1.0:
                continue
            options, target = list(P.YES_NO), [p_yes, 1.0 - p_yes]
            if p_yes in (0.0, 1.0):
                target = list(common.smooth_one_hot(2, 0 if p_yes == 1.0 else 1, P.LABEL_SMOOTHING))
            gold = 0 if p_yes >= 0.5 else 1
            qtype = "noul"
        elif q["type"] == "choice":
            codes = sorted(q["criteria"])
            if not 2 <= len(codes) <= JADE_MAX_OPTIONS or r["target"] not in codes:
                continue
            question = q["instructions"]
            options = [q["criteria"][c] if q["criteria"][c] is not None else c for c in codes]
            options = [o if isinstance(o, str) else json.dumps(o, ensure_ascii=False) for o in options]
            gold = codes.index(r["target"])
            target, qtype = list(common.smooth_one_hot(len(options), gold, P.LABEL_SMOOTHING)), "choice"
        else:
            continue
        items.append(common.make_item(item_id=f"sb_jade/{r['id']}", set_name="train", source="jade",
                                      question=question if isinstance(question, str) else json.dumps(question),
                                      options=options, target=target, gold=gold, rng=rng, state_text=state,
                                      meta={"suite": r["suite"], "family": r.get("family"), "qtype": qtype,
                                            "licence": "Apache-2.0"}))
    # CAL = whole families (related examples share a family id)
    fams = sorted({it["meta"]["family"] for it in items})
    rng.shuffle(fams)
    cal_fams, n_cal = set(), 0
    fam_size = Counter(it["meta"]["family"] for it in items)
    for f in fams:
        if n_cal >= JADE_CAL:
            break
        cal_fams.add(f)
        n_cal += fam_size[f]
    train = [it for it in items if it["meta"]["family"] not in cal_fams]
    cal = [{**it, "set": "CAL", "target": [1.0 if i == it["gold"] else 0.0 for i in range(len(it["options"]))]
            if it["meta"]["qtype"] == "choice" else it["target"]} for it in items if it["meta"]["family"] in cal_fams]
    out = Path(recipe.STAGEB_DIR) / "sb_jade"
    out.mkdir(parents=True, exist_ok=True)
    common.write_jsonl(out / "items_train.jsonl", train)
    common.write_jsonl(out / "items_cal.jsonl", cal)
    summary = {"train": len(train), "cal": len(cal), "suites": dict(Counter(it["meta"]["suite"] for it in train)),
               "qtype": dict(Counter(it["meta"]["qtype"] for it in train)), "source": P.JADE_URL,
               "licence": "Apache-2.0"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "audio").mkdir(exist_ok=True)
    volume.commit()
    return {k: summary[k] for k in ("train", "cal", "qtype")} | {"n_suites": len(summary["suites"])}


@app.function(**common_kwargs, memory=65536, timeout=3600)
def data_audit(run: str = "pass1") -> dict:
    """recipe.audit, written to /vol/runs/<run>/data_audit.json."""
    audit = recipe.audit(run)
    (run_dir(run) / "data_audit.json").write_text(json.dumps(audit, indent=2))
    volume.commit()
    return audit


def _dp_worker(rank: int, world: int, run: str, port: int) -> None:
    """One data-parallel training process (one GPU)."""
    import os

    import torch
    import torch.distributed as dist

    import common
    from model import load_qwen, run_training

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        cfg = run_config(run)
        items = load_train_items(run)
        banks = all_banks(run)
        n_before = len(items)
        items, dropped = valid_items(items, banks)
        if rank == 0 and dropped:
            print(f"[train] dropped malformed items by source: {dropped}", flush=True)
        if len(items) < 0.9 * n_before:
            raise RuntimeError(f"only {len(items)} of {n_before} training items passed validation: {dropped}")
        torch.manual_seed(common.SEED)
        model, head, processor, codes, encode_fn, logprob_fn = load_qwen(run, for_training=True, device=rank)
        out = run_dir(run)
        out.mkdir(parents=True, exist_ok=True)
        log = run_training(model, head, processor, items, banks,
                           max_steps=SMOKE_TRAIN_STEPS if is_smoke(run) else None,
                           log_path=out / "train_log.jsonl", ckpt_dir=out / "ckpt",
                           ckpt_every=SMOKE_CKPT_EVERY if is_smoke(run) else CKPT_EVERY,
                           on_checkpoint=volume.commit if rank == 0 else None, epochs=cfg["epochs"],
                           crema_intent_mix=cfg["crema_intent_mix"], lr=cfg.get("lr", LR), encode_fn=encode_fn,
                           logprob_fn=logprob_fn, batch=cfg.get("batch", EFFECTIVE_BATCH), rank=rank, world=world,
                           ordinal_smooth=cfg.get("ordinal_smooth", 0.0))
        if rank == 0:
            adapter_dir = out / "adapter"
            model.save_pretrained(str(adapter_dir))
            head.save(adapter_dir / "readout.safetensors", codes)
            (out / "train_done.json").write_text(json.dumps({"steps": log[-1]["step"] if log else 0,
                                                             "gpus": world, "items": len(items)}))
            import shutil

            shutil.rmtree(out / "ckpt", ignore_errors=True)
            volume.commit()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@app.function(**common_kwargs, gpu=["B200:8", "H200:8"], memory=262144, timeout=16 * 3600, retries=RETRIES)
def train_dp(run: str = "pass1") -> dict:
    """Data-parallel LoRA + head training on all GPUs of this container. Resumes from /vol/runs/<run>/ckpt."""
    import random

    import torch
    import torch.multiprocessing as mp

    out = run_dir(run)
    if (out / "train_done.json").exists() and (out / "adapter" / "readout.safetensors").exists():
        print("[train] already finished for this run; skipping")
        return {"run": run, "skipped": True}
    recipe.freeze_builders(run)
    volume.commit()
    world = torch.cuda.device_count()
    print(f"[train] run {run!r} config: {run_config(run)}; {world} x {torch.cuda.get_device_name()}", flush=True)
    mp.spawn(_dp_worker, args=(world, run, random.randint(20000, 40000)), nprocs=world, join=True)
    volume.reload()
    return {"run": run, **json.loads((out / "train_done.json").read_text())}


@app.function(**common_kwargs, gpu=QWEN_GPU, timeout=6 * 3600, retries=RETRIES)
def calibrate_temperature(run: str = "pass1") -> dict:
    """Predictions on the base evaluation items and every builder's CAL items, then one temperature, saved as
    /vol/runs/<run>/temperature_lora.json."""
    import torch

    import common
    from model import load_qwen, run_eval

    items = select_eval_items(common.read_jsonl(f"{DATA_DIR}/eval_items.jsonl"), run)
    drop = run_config(run).get("cal_drop_sources", ())
    items = [it for it in items if not (it["set"] == "CAL" and it.get("source") in drop)]
    banks = all_banks(run)
    cal, _ = valid_items([{**it, "set": "CAL"} for it in drop_excluded(stageb_items("cal", run), run_config(run), "cal")],
                         banks)
    if is_smoke(run):
        cal = cal[:200]
    items += cal
    model, head, processor, codes, encode_fn, logprob_fn = load_qwen(run, adapter=True)
    t0 = time.time()
    with torch.inference_mode():
        preds = run_eval(model, head, processor, items, banks, encode_fn=encode_fn, logprob_fn=logprob_fn)
    out = run_dir(run)
    common.write_jsonl(out / "preds_lora.jsonl", preds)
    temperature = fit_and_save_temperature(out, "lora", items, preds)
    volume.commit()
    print(f"[eval] {run}: {len(preds)} items in {time.time() - t0:.0f}s; temperature T = {temperature:.4f}")
    return {"run": run, "items": len(preds), "temperature": round(temperature, 4)}


@app.local_entrypoint()
def prepare(force: bool = False):
    print(json.dumps(prepare_base.remote(force), indent=2))


@app.local_entrypoint()
def jade():
    print(json.dumps(build_jade.remote(), indent=2))


@app.local_entrypoint()
def audit(run: str = "pass1"):
    a = data_audit.remote(run)
    print(json.dumps({k: a[k] for k in ("train_passes", "distinct_items", "est_steps", "cal_items",
                                        "excluded_left_in_train", "excluded_left_in_cal")}, indent=2))
    if a["excluded_left_in_train"] or a["excluded_left_in_cal"]:
        raise SystemExit("STOP: excluded data survived the filters")


@app.local_entrypoint()
def train(run: str = "pass1"):
    """modal run --detach training/modal_train.py::train --run pass1  (then ::calibrate)"""
    print(json.dumps(train_dp.remote(run)))


@app.local_entrypoint()
def calibrate(run: str = "pass1"):
    print(json.dumps(calibrate_temperature.remote(run)))


@app.local_entrypoint()
def smoke(gpus: int = 2):
    """10 optimizer steps of the pass-1 recipe on `gpus` GPUs, then a small temperature fit."""
    print(json.dumps(train_dp.with_options(gpu=f"H200:{gpus}").remote(run="smoke-pass1"), indent=2))
    print(json.dumps(calibrate_temperature.remote("smoke-pass1"), indent=2))
