"""Build sb_sounds2 on Modal (CPU only).

    modal run --detach data/sb_sounds2/modal_build.py::main
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
INFERENCE = HERE.parent.parent / "inference"
VOL = "/vol"
WORK = Path(VOL) / "stageb_work" / "sb_sounds2"

app = modal.App("sb-sounds2-build")
volume = modal.Volume.from_name("homi", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "libsndfile1")
    .uv_pip_install("numpy==2.5.3", "scipy==1.18.1", "pyarrow==25.0.1", "soundfile==0.14.0", "pandas",
                    "requests", "huggingface_hub==1.33.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_file(INFERENCE / "common.py", "/root/common.py")
    .add_local_file(INFERENCE / "answer_codes.json", "/root/answer_codes.json")
    .add_local_file(HERE / "fsd_labels.py", "/root/fsd_labels.py")
    .add_local_file(HERE / "labels2.py", "/root/labels2.py")
    .add_local_file(HERE / "build.py", "/root/build.py")
)

try:
    secrets = [modal.Secret.from_name("huggingface")]
except Exception:  # pragma: no cover
    secrets = []


def _hf_env():
    import os

    if not os.environ.get("HF_TOKEN"):
        for k in ("HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_HUB_TOKEN"):
            if os.environ.get(k):
                os.environ["HF_TOKEN"] = os.environ[k]


@app.function(image=image, volumes={VOL: volume}, secrets=secrets, cpu=2, memory=8192, timeout=1800, retries=2,
              max_containers=40)
def fma_shard(i: int) -> dict:
    sys.path.insert(0, "/root")
    _hf_env()
    import build

    res = build.process_fma_shard(i, WORK)
    volume.commit()
    return {"shard": i, **res}


@app.function(image=image, volumes={VOL: volume}, cpu=8, memory=16384, timeout=3600, retries=1)
def openmic_prep(fma_ok: dict) -> dict:
    sys.path.insert(0, "/root")
    import build

    res = build.process_openmic(WORK, {int(k): v for k, v in fma_ok.items()})
    volume.commit()
    return res


@app.function(image=image, volumes={VOL: volume}, secrets=secrets, cpu=8, memory=98304, timeout=3 * 3600)
def run_build(smoke: bool = False) -> dict:
    import json
    import shutil

    sys.path.insert(0, "/root")
    _hf_env()
    import build

    raw = Path("/tmp/raw")
    meta_status = build.fetch_meta(raw)
    fma_ok, tree, fma_stats = build.load_fma_meta(raw)
    fma_stats["fetch"] = meta_status
    build.log("fma usable tracks", len(fma_ok), fma_stats)
    small = {k: {kk: v[kk] for kk in ("licence_class", "licence_url", "artist_id", "artist")} for k, v in fma_ok.items()}
    om_call = openmic_prep.spawn(small)
    shards = list(range(3)) if smoke else list(range(build.FMA_SHARDS))
    calls = [fma_shard.spawn(i) for i in shards]
    build.fetch_fsd_audio(raw, smoke=smoke)
    build.log("fsd audio fetched")
    shard_res = []
    for c in calls:
        try:
            shard_res.append(c.get())
        except Exception as e:  # a lost shard only shrinks the music part
            shard_res.append({"error": repr(e)})
    build.log("fma shards done", sum(r.get("kept", 0) for r in shard_res), "kept;",
              sum(1 for r in shard_res if "error" in r), "failed")
    om_res = om_call.get()
    build.log("openmic done", om_res)
    volume.reload()
    out_name = "sb_sounds2_smoke" if smoke else "sb_sounds2"
    stage = Path("/tmp/out")
    summary = build.build(raw, WORK, stage, fma_ok, tree, fma_stats, smoke=smoke)
    summary["extra"]["fma_shards"] = {"n": len(shard_res), "failed": [r for r in shard_res if "error" in r],
                                      "kept": sum(r.get("kept", 0) for r in shard_res),
                                      "rows": sum(r.get("rows", 0) for r in shard_res)}
    (stage / "summary.json").write_text(json.dumps(summary, indent=1))
    # smoke output stays outside /vol/stageb so training never picks it up
    final = (Path(VOL) / "stageb_work" / out_name) if smoke else (Path(VOL) / "stageb" / out_name)
    staging, old = final.with_name(out_name + "_staging"), final.with_name(out_name + "_old")
    for p in (staging, old):
        if p.exists():
            shutil.rmtree(p)
    # copy everything except summary.json first; summary.json is written last and marks the build complete
    shutil.copytree(stage, staging, ignore=shutil.ignore_patterns("summary.json"))
    if final.exists():
        final.rename(old)
    staging.rename(final)
    if old.exists():
        shutil.rmtree(old)
    shutil.copy(stage / "summary.json", final / "summary.json")
    volume.commit()
    build.log("copied to", final)
    keep = {k: summary[k] for k in ("counts", "audio_hours_total", "checks", "leakage")}
    print(json.dumps(keep, indent=1)[:8000])
    return keep


@app.function(image=image, volumes={VOL: volume}, secrets=secrets, cpu=2, memory=16384, timeout=3 * 3600)
def prep_all(with_openmic: bool = False) -> dict:
    """Pre-compute every FMA shard and OpenMIC (cached in WORK; the main build reuses them)."""
    sys.path.insert(0, "/root")
    _hf_env()
    import build

    raw = Path("/tmp/raw")
    build.fetch_meta(raw)
    fma_ok, _, _ = build.load_fma_meta(raw)
    small = {k: {kk: v[kk] for kk in ("licence_class", "licence_url", "artist_id", "artist")} for k, v in fma_ok.items()}
    om_call = openmic_prep.spawn(small) if with_openmic else None
    res = list(fma_shard.map(range(build.FMA_SHARDS), return_exceptions=True, order_outputs=False))
    ok = [r for r in res if isinstance(r, dict)]
    out = {"shards_ok": len(ok), "kept": sum(r.get("kept", 0) for r in ok), "errors": [repr(r) for r in res if not isinstance(r, dict)][:5]}
    if om_call is not None:
        try:
            out["openmic"] = om_call.get()
        except Exception as e:
            out["openmic_error"] = repr(e)
    print(out)
    return out


@app.local_entrypoint()
def prep(with_openmic: bool = False):
    print(prep_all.remote(with_openmic=with_openmic))


@app.local_entrypoint()
def main(smoke: bool = False):
    print(run_build.remote(smoke=smoke))


@app.function(image=image, volumes={VOL: volume}, cpu=4, memory=16384, timeout=1800)
def verify_written(name: str = "sb_sounds2") -> dict:
    """Independent read-back of what is on the volume."""
    import collections
    import json

    sys.path.insert(0, "/root")
    import build
    import common

    volume.reload()
    out = Path(VOL) / "stageb" / name
    train = common.read_jsonl(out / "items_train.jsonl")
    cal = common.read_jsonl(out / "items_cal.jsonl")
    res = build.validate(out, train + cal)
    g = lambda its: {it["meta"].get("speaker") for it in its}  # noqa: E731
    res["cal_train_group_overlap"] = sorted(g(train) & g(cal))[:20]
    res["cal_frac"] = len(cal) / (len(train) + len(cal))
    res["sets"] = sorted({it["set"] for it in train + cal})
    res["noul_yes_share_by_family"] = {}
    fam = collections.defaultdict(collections.Counter)
    for it in train + cal:
        if it["meta"]["qtype"] == "noul":
            fam[f"{it['source']}/{it['meta']['family']}"][it["options"][it["gold"]]] += 1
    res["noul_yes_share_by_family"] = {k: round(v["yes"] / (v["yes"] + v["no"]), 3) for k, v in fam.items()}
    res["summary_exists"] = (out / "summary.json").exists()
    print(json.dumps({k: v for k, v in res.items() if k != "answer_balance"}, indent=1))
    return res


@app.local_entrypoint()
def verify(name: str = "sb_sounds2"):
    import json

    print(json.dumps(verify_written.remote(name), indent=1)[:6000])


@app.function(image=image, volumes={VOL: volume}, cpu=4, memory=32768, timeout=1800)
def finalize_written() -> dict:
    import json

    sys.path.insert(0, "/root")
    import build

    volume.reload()
    res = build.finalize(Path(VOL) / "stageb" / "sb_sounds2")
    volume.commit()
    print(json.dumps(res, indent=1))
    return res


@app.local_entrypoint()
def finalize():
    import json

    print(json.dumps(finalize_written.remote(), indent=1)[:6000])
