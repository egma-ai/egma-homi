"""Spot check, not part of the build: transcribe non-truncated speech clips with Whisper-small.en and compare them
with the script text. Run: modal run data/sb_calls/check_asr.py --out-dir /vol/stageb/sb_calls --n 60
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from modal_build import VOL, tts_image, volume  # noqa: E402

app = modal.App("sb-calls-asr-check")


def _norm(s):
    s = s.lower().replace("-", " ")
    return re.sub(r"[^a-z' ]", " ", s).split()


def _wer(ref, hyp):
    r, h = _norm(ref), _norm(hyp)
    dp = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(h) + 1):
            cur = min(dp[j] + 1, dp[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, dp[j] = dp[j], cur
    return dp[len(h)] / max(1, len(r))


@app.function(image=tts_image, gpu="L4", volumes={VOL: volume}, timeout=1800)
def check(out_dir: str, n: int = 60) -> dict:
    import random

    import numpy as np
    from transformers import pipeline as hf_pipeline

    import common

    volume.reload()
    out = Path(out_dir)
    items = common.read_jsonl(out / "items_train.jsonl") + common.read_jsonl(out / "items_cal.jsonl")
    seen, rows = set(), []
    for it in items:
        m = it["meta"]
        if m["voice"] and not m["truncated"] and it["audio"]["key"] not in seen:
            seen.add(it["audio"]["key"])
            rows.append(it)
    random.Random(0).shuffle(rows)
    rows = rows[:n]
    banks = {}
    asr = hf_pipeline("automatic-speech-recognition", model="openai/whisper-small.en", device=0)
    res = []
    for it in rows:
        b = it["audio"]["bank"]
        if b not in banks:
            banks[b] = common.AudioBank(out / "audio", b)
        x = banks[b].get(it["audio"]["key"])
        if it["channel"] == "phone":
            x = common.phone_channel(x)
        hyp = asr({"raw": x.astype(np.float32), "sampling_rate": 16000})["text"]
        ref = it["meta"]["script"].replace(" | ", " ")
        res.append({"cls": it["meta"]["class"], "voice": it["meta"]["voice"], "channel": it["channel"],
                    "wer": round(_wer(ref, hyp), 3), "ref": ref, "hyp": hyp})
    w = [r["wer"] for r in res]
    return {"n": len(res), "mean_wer": round(float(np.mean(w)), 3), "median_wer": float(np.median(w)),
            "n_wer_gt_0.3": sum(x > 0.3 for x in w), "worst": sorted(res, key=lambda r: -r["wer"])[:8]}


@app.local_entrypoint()
def run(out_dir: str = f"{VOL}/stageb/sb_calls", n: int = 60):
    r = check.remote(out_dir, n)
    print(json.dumps(r, indent=1))
    (HERE / "asr_check.json").write_text(json.dumps(r, indent=1))
