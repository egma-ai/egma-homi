"""Build sb_phone on Modal (CPU only): `::prep` decodes the sources once, then `::build` writes the items.

    modal run --detach data/sb_phone/modal_build.py::build
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
INFERENCE = HERE.parent.parent / "inference"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(INFERENCE))

app = modal.App("homi-stageb-sb-phone")
volume = modal.Volume.from_name("homi")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_pip_install("numpy==2.3.3", "scipy==1.16.2", "pyarrow==21.0.0", "soundfile==0.13.1",
                    "huggingface_hub[hf_xet]==0.35.3")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "PYTHONUNBUFFERED": "1"})
    .add_local_file(INFERENCE / "common.py", "/root/common.py")
    .add_local_file(INFERENCE / "answer_codes.json", "/root/answer_codes.json")
    .add_local_file(HERE / "prep.py", "/root/prep.py")
    .add_local_file(HERE / "fx.py", "/root/fx.py")
    .add_local_file(HERE / "labels.py", "/root/labels.py")
    .add_local_file(HERE / "build.py", "/root/build.py")
)
secrets = [modal.Secret.from_name("huggingface")]
kw = dict(image=image, volumes={"/vol": volume}, secrets=secrets)


def _hf_env():
    for k in ("HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_HUB_TOKEN"):
        if os.environ.get(k) and not os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = os.environ[k]


def _run_prep(name: str) -> dict:
    _hf_env()
    import prep

    t0 = time.time()
    fn = {"apollo": prep.prep_apollo, "vystadial": prep.prep_vystadial, "minds": prep.prep_minds,
          "robocall": prep.prep_robocall, "musan": prep.prep_musan, "rirs": prep.prep_rirs,
          "codecs": prep.ffmpeg_codecs}[name]
    try:
        out = fn()
    except Exception as e:  # report, do not kill the other preps
        import traceback

        out = {"error": repr(e), "trace": traceback.format_exc()[-3000:]}
    out["seconds"] = round(time.time() - t0, 1)
    volume.commit()
    print(name, json.dumps(out, default=str)[:3000])
    return out


@app.function(**kw, cpu=4, memory=32768, timeout=3 * 3600)
def prep_one(name: str) -> dict:
    return _run_prep(name)


@app.function(**kw, cpu=1, memory=2048, timeout=3 * 3600)
def prep_all(names: list[str]) -> dict:
    calls = {n: prep_one.spawn(n) for n in names}
    res = {}
    for n, c in calls.items():
        res[n] = c.get()
    out = Path("/vol/stageb/sb_phone/work/prep_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update(res)
    out.write_text(json.dumps(old, indent=1, default=str))
    volume.commit()
    return res


@app.local_entrypoint()
def prep(names: str = "codecs,apollo,vystadial,minds,robocall,musan,rirs"):
    res = prep_all.remote(names.split(","))
    print(json.dumps(res, indent=1, default=str)[:20000])


# Build
@app.function(**kw, cpu=4, memory=16384, retries=1, timeout=3600)
def render_shard(args: tuple) -> dict:
    import build

    res = build.render_shard(*args)
    volume.commit()
    return res


@app.function(**kw, cpu=4, memory=32768, timeout=4 * 3600)
def build_all(smoke: bool = False) -> dict:
    import build

    return build.main(smoke=smoke, render_map=render_shard.map, commit=volume.commit, reload=volume.reload)


@app.local_entrypoint()
def build(smoke: bool = False):
    res = build_all.remote(smoke)
    print(json.dumps(res, indent=1, default=str)[:20000])


@app.function(**kw, cpu=2, memory=4096, timeout=600)
def codec_selftest() -> dict:
    import random as _r

    import numpy as np

    import fx

    t = np.arange(16000 * 3) / 16000
    x = (0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 2 * t) > 0)).astype(np.float32)
    out = {}
    for c in fx.CODECS:
        for br in ([None] if c not in ("opus", "g726") else ([6, 12] if c == "opus" else [24, 32])):
            t0 = time.time()
            try:
                y = fx.ffmpeg_codec(x, c, br)
                corr = float(np.corrcoef(x[2000:-2000], y[2000:-2000])[0, 1])
                lag = int(np.argmax(np.correlate(y[8000:12000], x[8000:12000], mode="full")) - 3999)
                out[f"{c}{br or ''}"] = {"len_ok": len(y) == len(x), "corr": round(corr, 3), "lag": lag,
                                         "ms": round(1000 * (time.time() - t0))}
            except Exception as e:
                out[f"{c}{br or ''}"] = {"error": repr(e)[:300]}
    return out


@app.local_entrypoint()
def selftest():
    print(json.dumps(codec_selftest.remote(), indent=1))


@app.function(**kw, cpu=2, memory=16384, timeout=1800)
def check_all() -> dict:
    """Re-read the written files and validate them (schema, targets, audio keys, durations, balance)."""
    import build

    out = build.OUT
    items = build.read_jsonl(out / "items_train.jsonl") + build.read_jsonl(out / "items_cal.jsonl")
    banks = sorted(p.stem for p in (out / "audio").glob("*.json"))
    return build.validate(out, items, banks)


@app.local_entrypoint()
def check():
    print(json.dumps(check_all.remote(), indent=1)[:6000])
