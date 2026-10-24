"""Score multiple-choice label questions about a transcript with an open text LLM (vLLM, GPU).

Option probabilities come from the first generated token's top-20 logprobs, renormalised over the option letters.
"""

from __future__ import annotations

import json
import math
import time

LLM_CANDIDATES = [
    # (repo, pinned revision, licence); the first one that loads is used
    ("Qwen/Qwen3.6-35B-A3B-FP8", "95a723d08a9490559dae23d0cff1d9466213d989", "apache-2.0"),
    ("Qwen/Qwen3-30B-A3B-Instruct-2507", "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe", "apache-2.0"),
]
MAX_MODEL_LEN = 8192


def load_llm(candidates=LLM_CANDIDATES):
    import os

    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")  # DeepGEMM / FlashInfer JIT need a CUDA toolkit (nvcc) the image lacks
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM

    errors = []
    for repo, rev, lic in candidates:
        base = dict(model=repo, revision=rev, tokenizer_revision=rev, max_model_len=MAX_MODEL_LEN,
                    gpu_memory_utilization=0.90, enable_prefix_caching=True, max_logprobs=20, seed=0)
        tries = [dict(base, language_model_only=True, gdn_prefill_backend="triton"),
                 dict(base, language_model_only=True), dict(base, limit_mm_per_prompt={"image": 0, "video": 0}), base]
        for kw in tries:
            try:
                llm = LLM(**kw)
                return llm, {"repo": repo, "revision": rev, "licence": lic,
                             "engine_kwargs": {k: v for k, v in kw.items() if k not in ("model", "revision")}}
            except Exception as e:  # noqa: BLE001 - try the next configuration
                errors.append(f"{repo} {sorted(set(kw) - set(base))}: {type(e).__name__}: {str(e)[:300]}")
                print("load failed:", errors[-1], flush=True)
                if "out of memory" in str(e).lower():
                    break
    raise RuntimeError("no LLM could be loaded:\n" + "\n".join(errors))


def build_prompts(llm, requests: list[dict]) -> list[str]:
    tok = llm.get_tokenizer()
    out = []
    for r in requests:
        msgs = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}]
        try:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        out.append(p)
    return out


def letter_probs(top: dict, letters: list[str]) -> tuple[list[float], float]:
    """top: {decoded_token: logprob} for the first generated token -> (probabilities over `letters`, mass)."""
    p = {L: 0.0 for L in letters}
    for t, lp in top.items():
        key = t.strip()
        if key in p:
            p[key] += math.exp(lp)
    mass = sum(p.values())
    if mass <= 0:
        return [1.0 / len(letters)] * len(letters), 0.0
    return [p[L] / mass for L in letters], mass


def score(llm, requests: list[dict], batch: int = 4096) -> list[dict]:
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    res = []
    t0 = time.time()
    for i in range(0, len(requests), batch):
        chunk = requests[i : i + batch]
        prompts = build_prompts(llm, chunk)
        outs = llm.generate(prompts, sp, use_tqdm=False)
        for r, o in zip(chunk, outs):
            lp0 = o.outputs[0].logprobs[0] if o.outputs and o.outputs[0].logprobs else {}
            top = {}
            for tid, v in lp0.items():
                tok = v.decoded_token if v.decoded_token is not None else str(tid)
                top[tok] = max(top.get(tok, -1e9), v.logprob) if tok in top else v.logprob
            probs, mass = letter_probs(top, r["letters"])
            res.append({"rid": r["rid"], "probs": [round(x, 5) for x in probs], "mass": round(mass, 5),
                        "text": o.outputs[0].text if o.outputs else ""})
        print(f"scored {i + len(chunk)}/{len(requests)} in {time.time() - t0:.0f}s", flush=True)
    return res


def run_file(in_path: str, out_path: str, meta_path: str) -> dict:
    reqs = [json.loads(l) for l in open(in_path) if l.strip()]
    llm, info = load_llm()
    t0 = time.time()
    res = score(llm, reqs)
    with open(out_path, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    info.update({"n_requests": len(reqs), "seconds": round(time.time() - t0, 1),
                 "mean_mass": round(sum(r["mass"] for r in res) / max(1, len(res)), 4)})
    with open(meta_path, "w") as f:
        json.dump(info, f, indent=1)
    return info
