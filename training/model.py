"""
GPU code shared by training (modal_train.py) and evaluation (eval/modal_eval.py): load the Qwen3-Omni thinker with the
decision head (stock, with an adapter, or ready for LoRA training), score items, and the training loop.

Needs torch, transformers, peft and inference/{common,readout}.py plus training/recipe.py on the path.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

from recipe import (BRIER_WEIGHT, CKPT_EVERY, DATA_DIR, EFFECTIVE_BATCH, EVAL_MAX_BATCH, EVAL_TOKEN_BUDGET,
                    LOG_EVERY, LORA_DROPOUT, LR, TRAIN_TOKEN_BUDGET, WARMUP_FRAC, estimate_tokens, make_batches,
                    run_config, run_dir)

QWEN_OMNI_ID, QWEN_OMNI_REV = "Qwen/Qwen3-Omni-30B-A3B-Instruct", "26291f793822fb6be9555850f06dfe95f2d7e695"
QWEN_LORA_TARGETS = r".*model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"  # thinker attention only


def load_qwen(run: str, for_training: bool = False, adapter: bool = False, device: str | int = "cuda",
              adapter_dir: str | None = None):
    """Qwen3-Omni thinker (talker removed) + the decision head over its single-token answer codes.
    adapter=True loads /vol/runs/<run>/adapter (or `adapter_dir`, e.g. a downloaded Hugging Face kit).
    Returns (thinker_or_peft, head, processor, codes, encode_fn, logprob_fn)."""
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    import common
    import readout

    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN_OMNI_ID, revision=QWEN_OMNI_REV)
    processor.tokenizer.padding_side = "left"
    t0 = time.time()
    full = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        QWEN_OMNI_ID, revision=QWEN_OMNI_REV, dtype=torch.bfloat16,
        device_map="cuda" if device == "cuda" else {"": device})
    full.disable_talker()
    thinker = full.thinker
    print(f"[load] Qwen3-Omni thinker loaded in {time.time() - t0:.0f}s on {torch.cuda.get_device_name()}")
    # one head row per answer code the base evaluation items need (151), else all 255; the first N codes are the
    # same either way, so an item with N options gets the same prompt and scores
    eval_items = Path(f"{DATA_DIR}/eval_items.jsonl")
    max_options = max(len(it["options"]) for it in common.read_jsonl(eval_items)) if eval_items.exists() \
        else common.N_ANSWER_CODES
    codes, code_ids = common.build_answer_codes(processor.tokenizer, max_options)
    capture = {}
    # the text decoder's output (after its final norm) is what lm_head reads
    thinker.model.register_forward_hook(lambda _m, _a, out: capture.__setitem__(
        "hidden", (out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0])[:, -1]))
    if for_training:
        from peft import LoraConfig, get_peft_model

        thinker.config.use_cache = False
        thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        cfg = run_config(run)
        if cfg.get("init_from"):  # continue an earlier run's adapter + head (pass2 from pass1)
            from peft import PeftModel

            model = PeftModel.from_pretrained(thinker, cfg["init_from"], is_trainable=True)
            head = readout.DecisionHead.load(Path(cfg["init_from"]) / "readout.safetensors", device="cuda")
            print(f"[load] continuing from {cfg['init_from']}")
        else:
            model = get_peft_model(thinker, LoraConfig(r=cfg["lora_r"], lora_alpha=2 * cfg["lora_r"],
                                                       lora_dropout=LORA_DROPOUT, target_modules=QWEN_LORA_TARGETS,
                                                       bias="none"))
            head = readout.DecisionHead.from_lm_head(thinker.lm_head.weight, code_ids, None).to("cuda")
        model.print_trainable_parameters()
    elif adapter:
        from peft import PeftModel

        adapter_dir = Path(adapter_dir) if adapter_dir else run_dir(run) / "adapter"
        model = PeftModel.from_pretrained(thinker, str(adapter_dir)).eval()
        head = readout.DecisionHead.load(adapter_dir / "readout.safetensors", device="cuda").eval()
    else:
        model = thinker.eval()
        head = readout.DecisionHead.from_lm_head(thinker.lm_head.weight, code_ids, None).to("cuda").eval()

    def encode_fn(its, auds):
        texts = [processor.apply_chat_template(
            [{"role": "user", "content": ([{"type": "audio", "audio": "clip.wav"}] if a is not None else [])
              + [{"type": "text", "text": common.build_prompt_text(it, codes)}]}],
            add_generation_prompt=True, tokenize=False) for it, a in zip(its, auds)]
        present = [a for a in auds if a is not None]
        enc = processor(text=texts, audio=present or None, return_tensors="pt", padding=True)
        return {k: v.to("cuda", dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to("cuda")
                for k, v in enc.items()}

    def logprob_fn(enc, n_options):
        capture.clear()
        model(**enc, use_cache=False)
        return head(capture["hidden"], n_options)

    return model, head, processor, codes, encode_fn, logprob_fn


def run_eval(model, head, processor, items: list[dict], banks: dict, encode_fn=None, logprob_fn=None) -> list[dict]:
    """Decision-head log-probabilities for every item (no text generation)."""
    import torch

    import common
    import readout

    encode_fn = encode_fn or (lambda its, auds: common.encode_batch(processor, its, auds))
    logprob_fn = logprob_fn or (lambda enc, n: readout.decision_log_probs(model, head, enc, n))
    lengths = [estimate_tokens(it, processor.tokenizer, banks) for it in items]
    order = sorted(range(len(items)), key=lambda i: lengths[i])
    batches = make_batches(order, lengths, EVAL_TOKEN_BUDGET, EVAL_MAX_BATCH)
    out, t0 = {}, time.time()
    with torch.inference_mode():
        for b, idx in enumerate(batches, 1):
            its = [items[i] for i in idx]
            enc = encode_fn(its, [common.item_audio(it, banks) for it in its])
            n_opts = [len(it["options"]) for it in its]
            log_probs, _ = logprob_fn(enc, n_opts)
            for it, row, n in zip(its, log_probs.float().cpu().numpy(), n_opts):
                row = row[:n]
                out[it["id"]] = {"id": it["id"], "set": it["set"], "log_probs": [round(float(x), 5) for x in row],
                                 "probs": [round(float(x), 4) for x in common.apply_temperature(row, 1.0)],
                                 "pred": it["options"][int(row.argmax())], "gold": it["options"][it["gold"]]}
            if b % 20 == 0 or b == len(batches):
                print(f"[eval] batch {b}/{len(batches)}  items {len(out)}/{len(items)}  {time.time() - t0:.0f}s")
    return [out[it["id"]] for it in items]


def run_training(model, head, processor, items: list[dict], banks: dict, max_steps=None, log_path=None,
                 ckpt_dir=None, ckpt_every=CKPT_EVERY, on_checkpoint=None, epochs: int = 1,
                 crema_intent_mix: float = 0.0, lr: float = LR, encode_fn=None, logprob_fn=None,
                 batch: int = EFFECTIVE_BATCH, rank: int = 0, world: int = 1, ordinal_smooth: float = 0.0) -> list[dict]:
    """LoRA + decision-head training with soft-target cross-entropy at the answer position.
    Every `ckpt_every` steps the LoRA weights, head, optimizer and schedule are saved to `ckpt_dir`; an existing
    checkpoint there is resumed from.

    Data parallel (world > 1, torch.distributed initialised): each step takes `batch` items, rank r handles items
    r, r + world, ...; gradients are summed across ranks, so every rank applies the same update. Rank 0 logs and
    checkpoints."""
    import torch
    from transformers import get_cosine_schedule_with_warmup

    import common
    import readout

    import random

    encode_fn = encode_fn or (lambda its, auds: common.encode_batch(processor, its, auds))
    logprob_fn = logprob_fn or (lambda enc, n: readout.decision_log_probs(model, head, enc, n))
    torch.manual_seed(common.SEED)
    ordered = list(items)
    for e in range(2, int(math.ceil(epochs)) + 1):  # each extra pass in a new seeded order
        again = list(items)
        random.Random(f"{common.SEED}:epoch{e}").shuffle(again)
        if e > epochs:  # fractional last pass
            again = again[: int(round((epochs - (e - 1)) * len(items)))]
        ordered += again
    steps = [ordered[i : i + batch] for i in range(0, len(ordered), batch)]
    if max_steps:
        steps = steps[:max_steps]
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, round(WARMUP_FRAC * len(steps))), len(steps))
    model.train()
    head.train()
    log, window, t0 = [], [], time.time()
    start_step = 0
    ckpt_file = Path(ckpt_dir) / "state.pt" if ckpt_dir else None
    if ckpt_file and ckpt_file.exists():
        from peft import set_peft_model_state_dict

        state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, state["lora"])
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step, log = state["step"], state["log"]
        print(f"[train] rank {rank} resumed from checkpoint at step {start_step}")
    if world > 1:
        import torch.distributed as dist

        for p in params:  # same start on every rank (LoRA A matrices are random)
            dist.broadcast(p.data, src=0)
    say = print if rank == 0 else (lambda *a, **k: None)
    say(f"[train] {len(items)} items x {epochs} pass(es) -> {len(steps)} optimizer steps of {batch} items"
        + (f" ({world} GPUs)" if world > 1 else "")
          + (f"; emotion target = {1 - crema_intent_mix:.2f} x listener votes + {crema_intent_mix:.2f} x intended emotion"
             if crema_intent_mix else ""))
    for step, step_items in enumerate(steps, 1):
        if step <= start_step:
            continue  # done before the interruption
        mine = step_items[rank::world]
        lengths = [estimate_tokens(it, processor.tokenizer, banks) for it in mine]
        order = sorted(range(len(mine)), key=lambda i: -lengths[i])  # longest first
        for idx in make_batches(order, lengths, TRAIN_TOKEN_BUDGET, batch):
            its = [mine[i] for i in idx]
            enc = encode_fn(its, [common.item_audio(it, banks) for it in its])
            n_opts = [len(it["options"]) for it in its]
            log_probs, valid = logprob_fn(enc, n_opts)
            targets = torch.zeros_like(log_probs)
            for r, it in enumerate(its):
                t = torch.tensor(common.training_target(it, crema_intent_mix, ordinal_smooth), device=targets.device,
                                 dtype=targets.dtype)
                targets[r, : len(t)] = t
            loss = readout.decision_loss(log_probs, valid, targets, BRIER_WEIGHT)
            (loss * len(its) / len(step_items)).backward()  # micro-batches add up to the step mean
            with torch.no_grad():
                per_item = (-(targets * log_probs.masked_fill(~valid, 0.0)).sum(-1)).tolist()
            window += [(it["source"], v) for it, v in zip(its, per_item)]
        if world > 1:  # sum gradients across ranks in one all-reduce
            grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
            flat = torch.cat([g.reshape(-1).float() for g in grads])
            dist.all_reduce(flat)
            offset = 0
            for p, g in zip(params, grads):
                n = g.numel()
                p.grad = flat[offset : offset + n].view_as(p).to(p.dtype)
                offset += n
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if step % LOG_EVERY == 0 or step == len(steps):
            by_source = {}
            for s, v in window:
                by_source.setdefault(s, []).append(v)
            elapsed = time.time() - t0
            done_here = step - start_step
            row = {"step": step, "steps": len(steps), "loss": round(statistics.mean(v for _, v in window), 4),
                   "loss_by_source": {s: round(statistics.mean(v), 4) for s, v in sorted(by_source.items())},
                   "lr": scheduler.get_last_lr()[0], "grad_norm": round(grad_norm, 3),
                   "elapsed_s": round(elapsed), "eta_s": round(elapsed / done_here * (len(steps) - step))}
            say(f"[train] {json.dumps(row)}")
            log.append(row)
            window = []
            if log_path and rank == 0:
                common.write_jsonl(log_path, log)
        if ckpt_file and step % ckpt_every == 0 and step < len(steps) and rank == 0:
            from peft import get_peft_model_state_dict

            ckpt_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = ckpt_file.with_suffix(".tmp")
            torch.save({"lora": get_peft_model_state_dict(model), "head": head.state_dict(),
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "step": step, "log": log}, tmp)
            tmp.replace(ckpt_file)  # atomic
            if on_checkpoint:
                on_checkpoint()
            print(f"[train] checkpoint saved at step {step}")
    return log
