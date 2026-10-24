"""Torch-free logic for the text teacher: turn our items into requests for perplexity-ai/pplx-decider-v1.1-27b and
map its probabilities back to our option order. Yes/no items become teacher "noul" questions (teacher order
[false, true]); all others become "choice" questions with our options as criteria ("score" list form if duplicates)."""

from __future__ import annotations

import hashlib
import json
import math

REPO = "perplexity-ai/pplx-decider-v1.1-27b"
REVISION = "5cd25e3f8ae59ba65d1ebea74d8a06fa4a5f60fd"  # pinned HF commit (2026-10-08)
MAX_OPTIONS = 40  # items with more options are skipped
MAX_TOKENS = 8192  # the teacher's DecisionModel.prepare refuses longer inputs (no truncation)
H200_PER_HOUR = 4.54  # `modal billing rates`, 2026-10-08


def is_yes_no(options) -> bool:
    return len(options) == 2 and sorted(o.strip().lower() for o in options) == ["no", "yes"]


def skip_reason(item: dict) -> str | None:
    """Why an item is not scored (None = score it)."""
    if item.get("audio") is not None:
        return "audio"
    n = len(item.get("options") or [])
    if n > MAX_OPTIONS:
        return "too_many_options"
    if n < 2:
        return "too_few_options"
    return None


def to_teacher_row(item: dict, yes_no_mode: str = "noul") -> tuple[dict, list[int]]:
    """-> (teacher DecisionInput, mapping); mapping[i] indexes the teacher's probabilities for our option i."""
    if yes_no_mode not in ("noul", "choice"):
        raise ValueError(yes_no_mode)
    options = list(item["options"])
    state = item.get("state_text")
    state = "" if state is None else state
    instructions = item["question"]
    if yes_no_mode == "noul" and is_yes_no(options):
        question = {"type": "noul", "instructions": instructions}
        # teacher order is ["false", "true"] (autojev.model.options)
        mapping = [1 if o.strip().lower() == "yes" else 0 for o in options]
    elif len(set(options)) == len(options):
        question = {"type": "choice", "instructions": instructions, "criteria": {o: None for o in options}}
        mapping = list(range(len(options)))
    else:
        question = {"type": "score", "instructions": instructions, "criteria": options}
        mapping = list(range(len(options)))
    return {"state": state, "question": question}, mapping


def map_back(teacher_probs, mapping) -> list[float]:
    """Teacher probabilities -> one probability per option of ours (renormalised, rounded to 6 decimals)."""
    p = [max(float(teacher_probs[m]), 0.0) for m in mapping]
    s = sum(p)
    if not s > 0 or not math.isfinite(s):
        raise ValueError(f"bad teacher probabilities {teacher_probs}")
    p = [v / s for v in p]
    out = [round(v, 6) for v in p]
    # put the rounding residue on the largest entry so the vector still sums to 1
    k = max(range(len(out)), key=out.__getitem__)
    out[k] = round(out[k] + 1.0 - sum(out), 6)
    return out


def item_hash(item: dict) -> str:
    """Fingerprint of what the teacher sees plus the option order."""
    blob = json.dumps([item.get("state_text"), item["question"], item["options"]], ensure_ascii=False)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def shard_of(item_id: str, n_shards: int) -> int:
    return int(hashlib.sha1(item_id.encode()).hexdigest()[:8], 16) % n_shards


def entropy(p) -> float:
    return float(-sum(v * math.log(v) for v in p if v > 0))


def argmax(p) -> int:
    return max(range(len(p)), key=lambda i: p[i])


def make_batches(lengths: list[int], token_budget: int, max_rows: int) -> list[list[int]]:
    """Group item indices (sorted longest first) so that rows x longest_row <= token_budget (padded tokens)."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches, cur, cur_max = [], [], 0
    for i in order:
        m = max(cur_max, lengths[i])
        if cur and (len(cur) + 1 > max_rows or (len(cur) + 1) * m > token_budget):
            batches.append(cur)
            cur, m = [], lengths[i]
        cur.append(i)
        cur_max = m
    if cur:
        batches.append(cur)
    return batches
