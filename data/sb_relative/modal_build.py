"""Build sb_relative on Modal (CPU only), plus the crema_relative_<feature> extra evaluation set in /vol/adi/extra/.

    modal run --detach data/sb_relative/modal_build.py::main
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
INFERENCE = HERE.parent.parent / "inference"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(INFERENCE))

app = modal.App("homi-stageb-sb-relative")
volume = modal.Volume.from_name("homi")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("rubberband-cli")
    .uv_pip_install("numpy", "scipy", "librosa", "soundfile")
    .env({"NUMBA_CACHE_DIR": "/tmp/numba"})
    .add_local_file(INFERENCE / "common.py", "/root/common.py")
    .add_local_file(INFERENCE / "answer_codes.json", "/root/answer_codes.json")
    .add_local_file(HERE / "relabels.py", "/root/relabels.py")
    .add_local_file(HERE / "build.py", "/root/build.py")
)
kw = dict(image=image, volumes={"/vol": volume}, timeout=3 * 3600)
N_AUG_SHARDS = 10


@app.function(**kw, cpu=4, memory=32768)
def plan(smoke: bool = False) -> dict:
    import build

    volume.reload()
    r = build.plan(smoke)
    volume.commit()
    return r


@app.function(**kw, cpu=8, memory=16384, retries=1)
def render_aug(shard: int, n_shards: int, smoke: bool = False) -> dict:
    import build

    volume.reload()
    r = build.render_aug(shard, n_shards, 8, smoke)
    volume.commit()
    return r


@app.function(**kw, cpu=4, memory=65536)
def assemble(smoke: bool = False) -> dict:
    import build

    volume.reload()
    s = build.assemble(smoke)
    volume.commit()
    summ = build.validate_and_summarize(smoke)
    volume.commit()
    return {"assemble": s, "validation_ok": summ["validation"]["ok"], "errors": summ["validation"]["errors"][:10],
            "counts": summ["counts"], "leakage": summ["validation"]["leakage"]}


@app.function(**kw, cpu=16, memory=32768)
def build_eval() -> dict:
    import build

    volume.reload()
    r = build.build_eval(16)
    volume.commit()
    return r


@app.function(**kw, cpu=2, memory=16384)
def validate_only(smoke: bool = False) -> dict:
    import build

    volume.reload()
    s = build.validate_and_summarize(smoke)
    volume.commit()
    return {"ok": s["validation"]["ok"], "errors": s["validation"]["errors"], "counts": s["counts"], "leakage": s["validation"]["leakage"]}


@app.function(**kw, cpu=2, memory=4096)
def smoke_aug() -> dict:
    """rubberband + feature timing check on a synthetic vowel-like signal."""
    import numpy as np

    import build

    t = np.arange(3 * 16000) / 16000
    x = (0.3 * np.sin(2 * np.pi * 150 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    out = {}
    t0 = time.time()
    out["base"] = build.clip_features(x, 9)
    out["t_feat"] = round(time.time() - t0, 2)
    for kind, amt in (("tempo_log", 0.2), ("shift_st", 2.0), ("gain_db", 5.0)):
        t0 = time.time()
        y = build.augment(x, kind, amt)
        f = build.clip_features(y, 9)
        out[kind] = {"len": len(y), "feat": f, "seconds": round(time.time() - t0, 2)}
    return out


@app.function(**kw, cpu=1, memory=2048)
def pipeline(smoke: bool = False, with_eval: bool = True) -> dict:
    t0 = time.time()
    out = {"plan": plan.remote(smoke)}
    print(f"[pipeline] plan {out['plan']} ({time.time() - t0:.0f}s)", flush=True)
    ev = build_eval.spawn() if with_eval and not smoke else None
    calls = [render_aug.spawn(i, N_AUG_SHARDS, smoke) for i in range(N_AUG_SHARDS)]
    out["render"] = []
    for c in calls:
        try:
            r = c.get()
        except Exception as e:
            r = {"error": repr(e)[:400]}
        print(f"[pipeline] render {r} ({time.time() - t0:.0f}s)", flush=True)
        out["render"].append(r)
    out["assemble"] = assemble.remote(smoke)
    print(f"[pipeline] assemble {out['assemble']} ({time.time() - t0:.0f}s)", flush=True)
    if ev is not None:
        try:
            r = ev.get()
            out["eval"] = {k: r[k] for k in ("n_items", "recordings", "by_benchmark_split_qtype", "validation")}
        except Exception as e:
            out["eval"] = {"error": repr(e)[:400]}
        print(f"[pipeline] eval {out['eval']} ({time.time() - t0:.0f}s)", flush=True)
    out["seconds"] = round(time.time() - t0)
    print(f"[pipeline] DONE in {out['seconds']}s", flush=True)
    return out


@app.local_entrypoint()
def main(smoke: bool = False, with_eval: bool = True):
    import json

    print(json.dumps(pipeline.remote(smoke, with_eval), indent=1, default=str)[:20000])


@app.local_entrypoint()
def check(smoke: bool = False):
    import json

    print(json.dumps(validate_only.remote(smoke), indent=1, default=str)[:20000])


@app.local_entrypoint()
def eval_only():
    import json

    print(json.dumps(build_eval.remote(), indent=1, default=str)[:20000])


@app.local_entrypoint()
def aug_check():
    import json

    print(json.dumps(smoke_aug.remote(), indent=1, default=str))


@app.function(**kw, cpu=2, memory=16384)
def spot_check_fn(smoke: bool = False) -> dict:
    import build

    volume.reload()
    r = build.spot_check(160, smoke)
    volume.commit()
    return r


@app.local_entrypoint()
def spot(smoke: bool = False):
    import json

    print(json.dumps(spot_check_fn.remote(smoke), indent=1, default=str))
