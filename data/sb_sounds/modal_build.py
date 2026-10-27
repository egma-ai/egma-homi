"""Build sb_sounds on Modal (CPU only).

    modal run --detach data/sb_sounds/modal_build.py::main    # --smoke for a small build in /vol/stageb/sb_sounds_smoke/
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
INFERENCE = HERE.parent.parent / "inference"
VOL = "/vol"

app = modal.App("sb-sounds-build")
volume = modal.Volume.from_name("homi", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "libsndfile1")
    .uv_pip_install("numpy==2.5.3", "scipy==1.18.1", "pyarrow==25.0.1", "soundfile==0.14.0",
                    "huggingface_hub==1.33.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_file(INFERENCE / "common.py", "/root/common.py")
    .add_local_file(INFERENCE / "answer_codes.json", "/root/answer_codes.json")
    .add_local_file(HERE / "labels.py", "/root/labels.py")
    .add_local_file(HERE / "build.py", "/root/build.py")
)

try:
    secrets = [modal.Secret.from_name("huggingface")]
except Exception:  # pragma: no cover
    secrets = []


@app.function(image=image, volumes={VOL: volume}, secrets=secrets, cpu=8, memory=49152, timeout=3 * 3600)
def run_build(smoke: bool = False) -> dict:
    import json
    import os
    import shutil

    sys.path.insert(0, "/root")
    import build

    if not os.environ.get("HF_TOKEN"):
        for k in ("HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_HUB_TOKEN"):
            if os.environ.get(k):
                os.environ["HF_TOKEN"] = os.environ[k]
    raw = Path("/tmp/raw")
    out_name = "sb_sounds_smoke" if smoke else "sb_sounds"
    final = Path(VOL) / "stageb" / out_name
    stage = Path("/tmp/out")  # build on local disk, then copy to the volume in one go
    status = build.fetch_all(raw, smoke=smoke)
    build.log("fetch", status)
    summary = build.build(raw, stage, smoke=smoke, fetch_status=status)
    # copy to a staging dir on the volume first, then swap, so readers never see a half-written dataset
    final.parent.mkdir(parents=True, exist_ok=True)
    staging, old = final.with_name(out_name + "_staging"), final.with_name(out_name + "_old")
    for p in (staging, old):
        if p.exists():
            shutil.rmtree(p)
    shutil.copytree(stage, staging)
    if final.exists():
        final.rename(old)
    staging.rename(final)
    if old.exists():
        shutil.rmtree(old)
    volume.commit()
    build.log("copied to", final)
    print(json.dumps({k: summary[k] for k in ("counts", "audio_hours_total", "checks", "leakage")}, indent=1)[:6000])
    return {k: summary[k] for k in ("counts", "audio_hours_total", "checks", "leakage")}


@app.local_entrypoint()
def main(smoke: bool = False):
    res = run_build.remote(smoke=smoke)
    print(res)


@app.function(image=image, volumes={VOL: volume}, cpu=2, memory=8192, timeout=1800)
def leak_check() -> dict:
    """Re-run the leakage check of the written items against the evaluation items; store it in summary.json["leakage"]."""
    import json

    sys.path.insert(0, "/root")
    import build
    import common

    volume.reload()
    out = Path(VOL) / "stageb" / "sb_sounds"
    items = common.read_jsonl(out / "items_train.jsonl") + common.read_jsonl(out / "items_cal.jsonl")
    res = build.leakage_check(items)
    summary = json.loads((out / "summary.json").read_text())
    summary["leakage"] = res
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    volume.commit()
    print(json.dumps(res, indent=1))
    return res


@app.local_entrypoint()
def check_leakage():
    print(leak_check.remote())


@app.function(image=image, volumes={VOL: volume}, cpu=2, memory=8192, timeout=1800)
def verify_written() -> dict:
    """Independent read-back of what is on the volume: schema, targets, gold, audio keys, durations, balance,
    CAL disjointness by group."""
    import collections
    import json

    sys.path.insert(0, "/root")
    import build
    import common

    volume.reload()
    out = Path(VOL) / "stageb" / "sb_sounds"
    train = common.read_jsonl(out / "items_train.jsonl")
    cal = common.read_jsonl(out / "items_cal.jsonl")
    res = build.validate(out, train + cal)

    def groups(items):
        g = set()
        for it in items:
            sp = it["meta"].get("speaker")
            g |= set(sp if isinstance(sp, list) else [sp])
        return g

    res["cal_train_group_overlap"] = sorted(groups(train) & groups(cal))[:20]
    res["cal_frac"] = len(cal) / (len(train) + len(cal))
    res["max_items_per_audio"] = max(collections.Counter(
        (it["audio"]["bank"], it["audio"]["key"]) for it in train + cal).values())
    res["sets"] = sorted({it["set"] for it in train}) + sorted({it["set"] for it in cal})
    print(json.dumps({k: v for k, v in res.items() if k != "answer_balance"}, indent=1))
    return res


@app.local_entrypoint()
def verify():
    import json

    print(json.dumps(verify_written.remote(), indent=1)[:5000])
