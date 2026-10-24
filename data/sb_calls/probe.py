"""One-off probes for sb_calls, not part of the build: data peeks (::hvb) and a Kokoro voice check with Whisper
(::voices). Run: modal run data/sb_calls/probe.py::voices
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from modal_build import KOKORO_REV, cpu_image, tts_image, volume, VOL  # noqa: E402

app = modal.App("sb-calls-probe")


@app.function(image=cpu_image, volumes={VOL: volume}, timeout=1800, memory=8192)
def peek_hvb():
    import collections
    import io
    import urllib.request

    import pyarrow.parquet as pq

    url = "https://huggingface.co/api/datasets/asapp/slue-phase-2/parquet/hvb/train/0.parquet"
    with urllib.request.urlopen(url, timeout=600) as r:
        t = pq.read_table(io.BytesIO(r.read()), columns=["issue_id", "speaker_id", "text", "utt_index", "channel",
                                                         "role", "start_ms", "duration_ms", "intent", "dialog_acts"])
    rows = t.to_pylist()
    print("rows", len(rows), "issues", len({r["issue_id"] for r in rows}))
    print("roles", collections.Counter(r["role"] for r in rows))
    print("intent", collections.Counter(r["intent"] for r in rows).most_common(40))
    print("acts", collections.Counter(a for r in rows for a in r["dialog_acts"]).most_common(40))
    print("nacts", collections.Counter(len(r["dialog_acts"]) for r in rows))
    d = sorted(r["duration_ms"] for r in rows)
    print("dur ms pct", [d[int(len(d) * q)] for q in (0.05, 0.25, 0.5, 0.75, 0.95)])
    by = collections.defaultdict(list)
    for r in rows:
        by[r["issue_id"]].append(r)
    for k in list(by)[:4]:
        for r in sorted(by[k], key=lambda r: r["utt_index"])[:6]:
            print(k, r["utt_index"], r["role"], r["speaker_id"], r["start_ms"], r["duration_ms"], r["intent"], r["dialog_acts"], r["text"])
    jade = Path(f"{VOL}/data/jade_items.jsonl")
    lines = jade.read_text().splitlines()
    print("jade items", len(lines))
    for ln in lines[:5]:
        print(ln[:700])


@app.function(image=tts_image, gpu="L4", volumes={VOL: volume}, timeout=1800)
def check_voices():
    import time

    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download, list_repo_files
    from kokoro import KModel, KPipeline
    from transformers import pipeline as hf_pipeline

    files = list_repo_files("hexgrad/Kokoro-82M", revision=KOKORO_REV)
    voices = sorted(f.split("/")[1][:-3] for f in files if f.startswith("voices/"))
    cfg = hf_hub_download("hexgrad/Kokoro-82M", "config.json", revision=KOKORO_REV)
    mdl = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth", revision=KOKORO_REV)
    model = KModel(config=cfg, model=mdl).to("cuda").eval()
    pipes = {lc: KPipeline(lang_code=lc, model=model) for lc in "ab"}
    asr = hf_pipeline("automatic-speech-recognition", model="openai/whisper-small.en", device=0)
    texts = ["Hi, you've reached Maria. I can't take your call right now, so please leave a message after the tone.",
             "Hello? Who's calling?",
             "Thank you for calling Northfield Dental. For appointments, press one. For billing, press two.",
             "The number you have dialed is not in service. Please check the number and try again."]
    import re

    def norm(s):
        return re.sub(r"[^a-z ]", " ", s.lower()).split()

    def wer(ref, hyp):
        r, h = norm(ref), norm(hyp)
        dp = list(range(len(h) + 1))
        for i in range(1, len(r) + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, len(h) + 1):
                cur = min(dp[j] + 1, dp[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
                prev, dp[j] = dp[j], cur
        return dp[len(h)] / max(1, len(r))

    out = {}
    t0 = time.time()
    total_audio = 0.0
    for v in voices:
        vp = hf_hub_download("hexgrad/Kokoro-82M", f"voices/{v}.pt", revision=KOKORO_REV)
        lc = "b" if v.startswith("b") else "a"
        ws = []
        for t in texts:
            chunks = [r.audio.cpu().numpy() for r in pipes[lc](t, voice=vp, speed=1.0) if r.audio is not None]
            x = np.concatenate(chunks)
            total_audio += len(x) / 24000
            from scipy.signal import resample_poly
            hyp = asr({"raw": resample_poly(x, 2, 3).astype("float32"), "sampling_rate": 16000})["text"]
            ws.append(wer(t, hyp))
        out[v] = round(float(np.mean(ws)), 3)
        print("VOICE", v, out[v], flush=True)
    print("elapsed", time.time() - t0, "audio s", total_audio)
    Path(f"{VOL}/stageb/sb_calls/work").mkdir(parents=True, exist_ok=True)
    Path(f"{VOL}/stageb/sb_calls/work/voice_wer.json").write_text(json.dumps(out, indent=1))
    volume.commit()
    return out


@app.local_entrypoint()
def hvb():
    peek_hvb.remote()


@app.local_entrypoint()
def voices():
    res = check_voices.remote()
    (HERE / "voice_wer.json").write_text(json.dumps(res, indent=1))
