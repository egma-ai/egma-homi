"""Build sb_phone_pitch on Modal (CPU only).

    modal run --detach data/sb_phone_pitch/modal_build.py::main   # add --smoke for a 3% run
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

app = modal.App("homi-stageb-sb-phone-pitch")
volume = modal.Volume.from_name("homi")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_pip_install("numpy==2.3.3", "scipy==1.16.2", "soundfile==0.13.1", "librosa==0.11.0")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_file(INFERENCE / "common.py", "/root/common.py")
    .add_local_file(INFERENCE / "answer_codes.json", "/root/answer_codes.json")
    .add_local_file(HERE / "fx.py", "/root/fx.py")
    .add_local_file(HERE / "phone_labels.py", "/root/phone_labels.py")
    .add_local_file(HERE / "rel_labels.py", "/root/rel_labels.py")
    .add_local_file(HERE / "build.py", "/root/build.py")
)
kw = dict(image=image, volumes={"/vol": volume})


@app.function(**kw, cpu=4, memory=8192, timeout=3600)
def prep_hvb_fn() -> dict:
    import build

    out = build.prep_hvb()
    volume.commit()
    return out


@app.function(**kw, cpu=4, memory=16384, retries=1, timeout=3600)
def render_shard(args: tuple) -> dict:
    import build

    res = build.render_shard(*args)
    volume.commit()
    return res


@app.function(**kw, cpu=4, memory=32768, timeout=3 * 3600)
def build_all(smoke: bool = False) -> dict:
    import build

    t0 = time.time()
    prep = None
    if not (build.WORK / "raw" / "hvb_meta.json").exists():
        prep = prep_hvb_fn.remote()
        volume.reload()
        print("[prep_hvb]", json.dumps(prep)[:3000], flush=True)
    res = build.main(smoke=smoke, render_map=render_shard.map, commit=volume.commit, reload=volume.reload)
    res["prep_hvb"] = prep
    res["total_seconds"] = round(time.time() - t0)
    return res


@app.local_entrypoint()
def main(smoke: bool = False):
    res = build_all.remote(smoke)
    print(json.dumps(res, indent=1, default=str)[:30000])


@app.function(**kw, cpu=2, memory=8192, timeout=1800)
def peek() -> dict:
    """Quick look at inputs (no writes)."""
    import build

    recs, info = build.load_records()
    return {"records": info, "cal": sorted(build.cal_speakers())}


@app.local_entrypoint()
def inputs():
    print(json.dumps(peek.remote(), indent=1, default=str)[:20000])
