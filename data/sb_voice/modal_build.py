"""Build sb_voice on Modal (CPU only).

    modal run --detach data/sb_voice/modal_build.py::main
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
INFERENCE = HERE.parent.parent / "inference"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(INFERENCE))

app = modal.App("egma-homi-stageb-sb-voice")
volume = modal.Volume.from_name("egma-homi")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy", "scipy", "librosa", "soundfile", "pyarrow", "huggingface_hub[hf_xet]", "cmudict")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "NUMBA_CACHE_DIR": "/tmp/numba"})
    .add_local_file(INFERENCE / "common.py", "/root/common.py")
    .add_local_file(INFERENCE / "answer_codes.json", "/root/answer_codes.json")
    .add_local_file(HERE / "labels.py", "/root/labels.py")
    .add_local_file(HERE / "build.py", "/root/build.py")
    .add_local_file(HERE / "librispeech_speaker_gender.json", "/root/librispeech_speaker_gender.json")
    .add_local_file(HERE / "crema_demographics.json", "/root/crema_demographics.json")
)
secrets = [modal.Secret.from_name("huggingface")]
kw = dict(image=image, volumes={"/vol": volume}, secrets=secrets, timeout=3 * 3600)


def _hf_env():
    for k in ("HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_HUB_TOKEN"):
        if os.environ.get(k) and not os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = os.environ[k]


def _done(name: str) -> bool:
    return Path(f"/vol/stageb/sb_voice/work2/{name}.jsonl").exists() and Path(f"/vol/stageb/sb_voice/work2/{name}.npy").exists()


@app.function(**kw, cpu=4, memory=16384, retries=1)
def vctk_shard(shard: int) -> dict:
    _hf_env()
    import build

    if _done(f"vctk_{shard:02d}"):
        return {"shard": shard, "skipped": "exists"}
    r = build.process_vctk(shard, 4)
    volume.commit()
    return r


@app.function(**kw, cpu=4, memory=16384, retries=1)
def libri_shard(split: str, shard: int, keep: float) -> dict:
    _hf_env()
    import build

    tag = "dev" if split == "dev.clean" else "tc100"
    if _done(f"libritts_{tag}_{shard:02d}"):
        return {"split": split, "shard": shard, "skipped": "exists"}
    r = build.process_libritts(split, shard, keep, 4)
    volume.commit()
    return r


@app.function(**kw, cpu=4, memory=16384, retries=1)
def crema_chunk(chunk: int, n_chunks: int) -> dict:
    import build

    if _done(f"crema_{chunk:02d}"):
        return {"chunk": chunk, "skipped": "exists"}
    r = build.process_crema(chunk, n_chunks, 4)
    volume.commit()
    return r


@app.function(**kw, cpu=4, memory=32768, retries=1)
def vocalsound() -> dict:
    _hf_env()
    import build

    if _done("vocalsound_00"):
        return {"skipped": "exists"}
    r = build.process_vocalsound(2)
    volume.commit()
    return r


@app.function(**kw, cpu=2, memory=8192, retries=1)
def so762() -> dict:
    _hf_env()
    import build

    if _done("so762_00"):
        return {"skipped": "exists"}
    r = build.process_so762(2)
    volume.commit()
    return r


@app.function(**kw, cpu=4, memory=65536)
def assemble() -> dict:
    import build

    volume.reload()
    s = build.assemble()
    volume.commit()
    rep = build.validate()
    volume.commit()
    return {"n_train": s["n_train"], "n_cal": s["n_cal"], "hours": s["total_audio_hours"], "validation_ok": rep["ok"],
            "errors": rep["errors"][:10], "checks": rep["checks"], "leakage": rep["leakage"]}


@app.function(**kw, cpu=2, memory=16384)
def validate_only() -> dict:
    import build

    volume.reload()
    rep = build.validate()
    volume.commit()
    return rep


@app.function(**kw, cpu=4, memory=16384)
def mmsu_check() -> dict:
    import json

    import build

    volume.reload()
    r = build.mmsu_content_check()
    p = build.OUT / "summary.json"
    s = json.loads(p.read_text())
    s.setdefault("validation", {}).setdefault("leakage", {})["mmsu_content"] = r
    p.write_text(json.dumps(s, indent=2))
    volume.commit()
    return r


@app.function(**kw, cpu=1, memory=2048)
def pipeline(skip_stage1: bool = False) -> dict:
    import build

    t0 = time.time()
    out = {"stage1": []}
    if not skip_stage1:
        calls = [vctk_shard.spawn(i) for i in range(build.VCTK_N_SHARDS)]
        calls += [libri_shard.spawn(*p) for p in build.LIBRI_PLAN]
        calls += [crema_chunk.spawn(i, 6) for i in range(6)]
        calls += [vocalsound.spawn(), so762.spawn()]
        for c in calls:
            try:
                r = c.get()
            except Exception as e:  # keep going: assemble uses whatever shards exist
                r = {"error": repr(e)[:500]}
            print(f"[pipeline] stage1 {r} ({time.time() - t0:.0f}s)", flush=True)
            out["stage1"].append(r)
    out["assemble"] = assemble.remote()
    try:
        out["mmsu_content"] = mmsu_check.remote()
    except Exception as e:
        out["mmsu_content"] = {"error": repr(e)[:300]}
    out["seconds"] = round(time.time() - t0)
    print(f"[pipeline] DONE {out['assemble']}", flush=True)
    return out


@app.function(**kw, cpu=2, memory=8192)
def smoke() -> dict:
    """Tiny end-to-end check of stage 1 code paths: one VCTK shard's first rows + feature speed."""
    _hf_env()
    import numpy as np

    import build

    x = np.sin(2 * np.pi * 150 * np.arange(48000) / 16000).astype(np.float32) * 0.3
    build._init_worker()
    t0 = time.time()
    f = build.clip_features(x, "Please call Stella.")
    return {"feat": f, "seconds_for_3s": round(time.time() - t0, 2)}


@app.local_entrypoint()
def main(skip_stage1: bool = False):
    print(pipeline.remote(skip_stage1=skip_stage1))


@app.local_entrypoint()
def check():
    import json

    print(json.dumps(validate_only.remote(), indent=2)[:6000])


@app.function(**kw, cpu=2, memory=8192)
def vs_list() -> dict:
    """Debug helper: list the non-audio files of the VocalSound release zip."""
    _hf_env()
    import collections
    import json
    import zipfile

    import build

    src = build.SOURCES["vocalsound"]
    z = zipfile.ZipFile(build._hf_file(src["repo"], src["rev"], src["file"]))
    names = z.namelist()
    other = [n for n in names if not n.endswith(".wav")]
    out = {"n": len(names), "other": other[:80], "dirs": collections.Counter("/".join(n.split("/")[:2]) for n in names).most_common(12)}
    for n in other:
        if n.endswith((".json", ".csv", ".txt")) and z.getinfo(n).file_size < 5_000_000:
            out[n] = z.read(n)[:400].decode("utf-8", "replace")
    return out


@app.local_entrypoint()
def mmsu():
    import json

    print(json.dumps(mmsu_check.remote(), indent=1))


@app.local_entrypoint()
def vslist():
    import json

    print(json.dumps(vs_list.remote(), indent=1)[:8000])
