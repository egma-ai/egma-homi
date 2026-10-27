"""Build sb_compare on Modal (Kokoro TTS on L4 GPUs, everything else on CPU).

    modal run --detach data/sb_compare/modal_build.py::main    # add --smoke for a ~3 % test build
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

KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_REV = "f3ff3571791e39611d31c381e3a41a3af07b4987"  # pinned (Apache-2.0), same as data/sb_calls

app = modal.App("homi-stageb-sb-compare")
volume = modal.Volume.from_name("homi")


def _code(img):
    img = img.add_local_file(INFERENCE / "common.py", "/root/common.py")
    img = img.add_local_file(INFERENCE / "answer_codes.json", "/root/answer_codes.json")
    for f in ("cmp_labels.py", "build.py"):
        img = img.add_local_file(HERE / f, f"/root/{f}")
    return img


cpu_image = _code(
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("rubberband-cli", "ffmpeg")
    .uv_pip_install("numpy==2.3.3", "scipy==1.16.2", "librosa==0.11.0", "soundfile==0.13.1")
    .env({"NUMBA_CACHE_DIR": "/tmp/numba"})
)
tts_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng", "git")
    .pip_install("torch==2.7.1", "kokoro==0.9.4", "misaki[en]==0.9.4", "transformers==4.51.3", "huggingface_hub==0.30.2",
                 "numpy==1.26.4", "scipy==1.13.1", "soundfile==0.13.1",
                 "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl")
    .env({"HF_HOME": "/vol/hf_cache"})
)
kw = dict(volumes={"/vol": volume}, timeout=3 * 3600)


@app.function(image=cpu_image, cpu=2, memory=8192, **kw)
def plan_fn(smoke: bool, extra: bool = False) -> dict:
    import build

    volume.reload()
    r = build.plan(smoke, extra)
    volume.commit()
    return r


@app.function(image=tts_image, gpu="L4", max_containers=2, retries=1, **kw)
def tts_fn(shard: int, jobs: list, work_dir: str) -> dict:
    """jobs: [(key, voice, speed, words)]. Writes tts_XX.npz (16 kHz audio) + tts_XX.json ({key: [[word, start_s, end_s]]})."""
    import json
    import re

    import numpy as np
    from huggingface_hub import hf_hub_download
    from kokoro import KModel, KPipeline
    from scipy.signal import resample_poly

    t0 = time.time()
    out = Path(work_dir) / "tts"
    out.mkdir(parents=True, exist_ok=True)
    cfg = hf_hub_download(KOKORO_REPO, "config.json", revision=KOKORO_REV)
    mdl = hf_hub_download(KOKORO_REPO, "kokoro-v1_0.pth", revision=KOKORO_REV)
    model = KModel(config=cfg, model=mdl).to("cuda").eval()
    pipes = {lc: KPipeline(lang_code=lc, model=model) for lc in "ab"}
    vpaths, arrays, timings, fails = {}, {}, {}, []
    for key, voice, speed, words in jobs:
        try:
            if voice not in vpaths:
                vpaths[voice] = hf_hub_download(KOKORO_REPO, f"voices/{voice}.pt", revision=KOKORO_REV)
            pipe = pipes["b" if voice.startswith("b") else "a"]
            chunks, toks, off = [], [], 0.0
            for r in pipe(" ".join(words), voice=vpaths[voice], speed=speed):
                if r.audio is None:
                    continue
                a = r.audio.detach().cpu().numpy().astype(np.float64)
                for t in r.tokens or []:
                    if re.search(r"[A-Za-z]", t.text or "") and t.start_ts is not None and t.end_ts is not None:
                        toks.append([t.text.lower(), round(off + float(t.start_ts), 4), round(off + float(t.end_ts), 4)])
                chunks.append(a)
                off += len(a) / 24000.0
            x = np.concatenate(chunks)
            y = resample_poly(x, 2, 3).astype(np.float32)
            arrays[key] = y
            timings[key] = toks
        except Exception as e:  # noqa: BLE001
            fails.append([key, repr(e)[:200]])
    np.savez(out / f"tts_{shard:02d}.npz", **arrays)
    (out / f"tts_{shard:02d}.json").write_text(json.dumps(timings))
    volume.commit()
    return {"shard": shard, "n": len(jobs), "ok": len(arrays), "fails": fails[:5], "n_fail": len(fails), "sec": round(time.time() - t0)}


@app.function(image=cpu_image, cpu=2, memory=6144, retries=1, **kw)
def render_fn(shard: int, n_shards: int, smoke: bool, extra: bool = False) -> dict:
    import build

    volume.reload()
    r = build.render_shard(shard, n_shards, 1, smoke, extra)
    volume.commit()
    return r


@app.function(image=cpu_image, cpu=2, memory=8192, retries=1, **kw)
def pause_fn(shard: int, n_shards: int, smoke: bool) -> dict:
    import build

    volume.reload()
    r = build.render_pause(shard, n_shards, smoke)
    volume.commit()
    return r


@app.function(image=cpu_image, cpu=4, memory=32768, **kw)
def assemble_fn(smoke: bool) -> dict:
    import build

    volume.reload()
    _, out = build.paths(smoke)
    if (out / "summary.json").exists():  # summary.json marks a finished build; remove it before re-assembling
        (out / "summary.json").unlink()
        volume.commit()
    r = build.assemble(smoke)
    volume.commit()
    return r


@app.function(image=cpu_image, cpu=1, memory=4096, **kw)
def pipeline(smoke: bool = False) -> dict:
    import json

    import build

    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)  # noqa: E731
    out = {"plan": plan_fn.remote(smoke)}
    log("plan", out["plan"])
    volume.reload()
    work, _ = build.paths(smoke)
    p = json.loads((work / "plan.json").read_text())
    jobs = [(t["key"], t["voice"], t["speed"], t["words"]) for t in p["tts"]]
    n_tts = build.N_TTS
    tts_calls = [tts_fn.spawn(i, jobs[i::n_tts], str(work)) for i in range(n_tts)]
    n_r = 8 if smoke else build.N_RENDER
    r_calls = [render_fn.spawn(i, n_r, smoke) for i in range(n_r)]
    out["tts"] = []
    for c in tts_calls:
        try:
            r = c.get()
        except Exception as e:  # noqa: BLE001
            r = {"error": repr(e)[:300]}
        log("tts", r)
        out["tts"].append(r)
    n_p = 2 if smoke else build.N_PAUSE
    p_calls = [pause_fn.spawn(i, n_p, smoke) for i in range(n_p)]
    out["render"], out["pause"] = [], []
    for name, calls in (("render", r_calls), ("pause", p_calls)):
        for c in calls:
            try:
                r = c.get()
            except Exception as e:  # noqa: BLE001
                r = {"error": repr(e)[:300]}
            log(name, r)
            out[name].append(r)
    out["assemble"] = assemble_fn.remote(smoke)
    log("assemble", json.dumps(out["assemble"], default=str)[:3000])
    out["seconds"] = round(time.time() - t0)
    return out


@app.function(image=cpu_image, cpu=1, memory=4096, **kw)
def topup_pipeline(n_shards: int = 20) -> dict:
    """Top-up: extra LibriTTS-R comparison/duration clips (new keys, same CAL speakers) -> re-assemble everything."""
    import json

    t0 = time.time()
    out = {"plan": plan_fn.remote(False, True)}
    print("plan_extra", out["plan"], flush=True)
    calls = [render_fn.spawn(i, n_shards, False, True) for i in range(n_shards)]
    out["render"] = []
    for c in calls:
        try:
            r = c.get()
        except Exception as e:  # noqa: BLE001
            r = {"error": repr(e)[:300]}
        print("render_extra", r, flush=True)
        out["render"].append(r)
    out["assemble"] = assemble_fn.remote(False)
    print("assemble", json.dumps(out["assemble"], default=str)[:4000], flush=True)
    out["seconds"] = round(time.time() - t0)
    return out


@app.local_entrypoint()
def topup():
    import json

    print(json.dumps(topup_pipeline.remote(), indent=1, default=str)[:20000])


@app.local_entrypoint()
def assemble_only(smoke: bool = False):
    import json

    print(json.dumps(assemble_fn.remote(smoke), indent=1, default=str)[:20000])


@app.local_entrypoint()
def main(smoke: bool = False):
    import json

    print(json.dumps(pipeline.remote(smoke), indent=1, default=str)[:20000])
