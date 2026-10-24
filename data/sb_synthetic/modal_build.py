"""Build sb_synthetic on Modal: real pool (CPU), TTS and copy-synthesis (CPU + one L4 GPU at a time), assembly (CPU).

    modal run --detach data/sb_synthetic/modal_build.py::run_prep   # then ::gpu and ::full (::full waits for ::gpu)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
INFERENCE = HERE.parent.parent / "inference"
SB_PHONE = HERE.parent / "sb_phone"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(INFERENCE))

VOL = "/vol"
WORK = f"{VOL}/stageb_work/sb_synthetic"
OUT = f"{VOL}/stageb/sb_synthetic"

volume = modal.Volume.from_name("egma-homi", create_if_missing=False)


def _code(img: modal.Image) -> modal.Image:
    img = img.add_local_file(INFERENCE / "common.py", "/root/common.py")
    img = img.add_local_file(INFERENCE / "answer_codes.json", "/root/answer_codes.json")
    img = img.add_local_file(SB_PHONE / "fx.py", "/root/fx.py")
    for f in ("synth.py", "build.py", "labels.py", "modal_build.py"):
        if (HERE / f).exists():
            img = img.add_local_file(HERE / f, f"/root/{f}")
    return img


ENV = {"HF_HOME": f"{VOL}/hf_cache", "TOKENIZERS_PARALLELISM": "false"}

cpu_image = _code(
    modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg")
    .uv_pip_install("numpy==2.3.3", "scipy==1.16.2", "soundfile==0.13.1", "huggingface_hub==0.35.3")
    .env(ENV))

kokoro_image = _code(
    modal.Image.debian_slim(python_version="3.11").apt_install("espeak-ng", "git")
    .pip_install("torch==2.7.1", "kokoro==0.9.4", "misaki[en]==0.9.4", "transformers==4.51.3", "huggingface_hub==0.30.2",
                 "numpy==1.26.4", "scipy==1.13.1", "soundfile==0.13.1",
                 "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl")
    .env(ENV))

piper_image = _code(
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("piper-tts==1.2.0", "piper-phonemize==1.1.0", "onnxruntime==1.19.2", "numpy==1.26.4", "scipy==1.13.1",
                 "soundfile==0.13.1", "huggingface_hub==0.30.2")
    .env(ENV))

torchcpu_image = _code(
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "torchaudio==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("transformers==4.46.3", "sentencepiece==0.2.0", "vocos==0.1.0", "encodec==0.1.1", "einops==0.8.0",
                 "pyyaml==6.0.2", "numpy==1.26.4", "scipy==1.13.1", "soundfile==0.13.1", "huggingface_hub==0.26.2")
    .env(ENV))

parler_image = _code(
    modal.Image.debian_slim(python_version="3.11").apt_install("git")
    .pip_install("torch==2.5.1", "torchaudio==2.5.1", "numpy==1.26.4", "scipy==1.13.1", "soundfile==0.13.1")
    .pip_install("parler_tts @ git+https://github.com/huggingface/parler-tts.git@d108732cd57788ec86bc857d99a6cabd66663d68")
    .env(ENV))

chatterbox_image = _code(
    modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg", "git")
    .pip_install("chatterbox-tts==0.1.2", "soundfile==0.13.1", "scipy==1.13.1")
    .env(ENV))

app = modal.App("sb-synthetic-build")


def _run(system: str, args: tuple, **kw) -> dict:
    import numpy as np

    import synth

    shard_id, jobs, out_dir = args
    path = Path(f"{out_dir}/{system}/shard_{shard_id:03d}.npz")
    if path.exists():
        return {"system": system, "shard": shard_id, "skipped": True}
    t0 = time.time()
    out, fails = synth.RUNNERS[system](jobs, **kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **out)
    tmp.rename(path)
    volume.commit()
    secs = sum(len(v) for v in out.values()) / 16000
    return {"system": system, "shard": shard_id, "n": len(jobs), "ok": len(out), "fails": fails,
            "speech_s": round(secs, 1), "elapsed_s": round(time.time() - t0, 1)}


@app.function(image=kokoro_image, volumes={VOL: volume}, timeout=3600, cpu=4, memory=8192, max_containers=16, retries=1)
def synth_kokoro(args: tuple) -> dict:
    return _run("kokoro", args)


@app.function(image=piper_image, volumes={VOL: volume}, timeout=3600, cpu=2, memory=4096, max_containers=16, retries=1)
def synth_piper(args: tuple) -> dict:
    return _run("piper", args)


@app.function(image=torchcpu_image, volumes={VOL: volume}, timeout=3600, cpu=4, memory=8192, max_containers=24, retries=1)
def synth_torchcpu(system: str, args: tuple) -> dict:
    return _run(system, args)


@app.function(image=parler_image, gpu="L4", volumes={VOL: volume}, timeout=3600, max_containers=1)
def synth_parler(args: tuple) -> dict:
    return _run("parler", args)


@app.function(image=chatterbox_image, gpu="L4", volumes={VOL: volume}, timeout=3600, max_containers=1)
def synth_chatterbox(args: tuple, time_budget_s: float = 1e9) -> dict:
    return _run("chatterbox", args, time_budget_s=time_budget_s)


@app.function(image=cpu_image, volumes={VOL: volume}, timeout=3600, cpu=8, memory=32768)
def prep_real(smoke: bool = False) -> dict:
    import build

    r = build.prep_real(smoke)
    volume.commit()
    return r


@app.function(image=cpu_image, volumes={VOL: volume}, timeout=3 * 3600, cpu=2, memory=8192)
def gpu_orchestrator(smoke: bool = False, chatterbox_budget_s: float = 1500.0) -> list:
    import build

    return build.gpu_chain(smoke, {"parler": synth_parler, "chatterbox": synth_chatterbox}, chatterbox_budget_s)


@app.function(image=cpu_image, volumes={VOL: volume}, timeout=4 * 3600, cpu=8, memory=65536)
def main(smoke: bool = False) -> dict:
    """CPU synthesis, assembly, items and summary, run in the cloud."""
    import build

    return build.pipeline(smoke, {"kokoro": synth_kokoro, "piper": synth_piper, "torchcpu": synth_torchcpu,
                                  "parler": synth_parler, "chatterbox": synth_chatterbox}, volume)


REF_KEY = "ls_2277-149896-0000"
PROBE_TEXTS = ["The weather was cold and the river had frozen over by the end of the week.",
               "Hi, I'm calling about my account, I think there's a charge I don't recognise."]


@app.local_entrypoint()
def probe(which: str = "kokoro,piper,speecht5,parler,chatterbox"):
    out_dir = f"{WORK}/probe"
    calls = []
    for s in which.split(","):
        jobs = []
        for i, t in enumerate(PROBE_TEXTS):
            j = {"key": f"probe_{s}_{i}", "text": t}
            if s == "kokoro":
                j.update(voice="af_heart" if i == 0 else "bm_george", params={"speed": 1.0})
            elif s == "piper":
                j.update(voice="en_US-libritts-high" if i == 0 else "en_GB-cori-high",
                         params={"speaker_id": 23 if i == 0 else None, "length_scale": 1.0, "noise_scale": 0.667, "noise_w": 0.8})
            elif s == "speecht5":
                j.update(voice="slt", params={"seed": i, "mix": [["slt", 0.7], ["bdl", 0.3]] if i else [["awb", 1.0]]})
            elif s == "parler":
                j.update(voice="Jon", params={"seed": i, "description": "Jon's voice is calm with a moderate pace and low pitch. "
                                                                         "The recording is very clear and close-sounding."})
            elif s == "chatterbox":
                j.update(voice="ls_ref", params={"seed": i, "ref_key": REF_KEY, "exaggeration": 0.5, "cfg_weight": 0.5,
                                                 "temperature": 0.8})
            elif s in ("vocos", "hifigan"):
                j = {"key": f"probe_{s}_{i}", "src_key": REF_KEY}
            jobs.append(j)
        if not jobs:
            continue
        args = (0, jobs, out_dir)
        if s == "kokoro":
            calls.append(synth_kokoro.spawn(args))
        elif s == "piper":
            calls.append(synth_piper.spawn(args))
        elif s == "speecht5":
            calls.append(synth_torchcpu.spawn("speecht5", args))
        elif s == "parler":
            calls.append(synth_parler.spawn(args))
        elif s == "chatterbox":
            calls.append(synth_chatterbox.spawn(args))
        elif s in ("vocos", "hifigan"):
            calls.append(synth_torchcpu.spawn(s, args))
    for c in calls:
        try:
            print(c.get())
        except Exception as e:  # noqa: BLE001
            print("PROBE FAIL", repr(e)[:2000])


@app.local_entrypoint()
def gpu(smoke: bool = False, chatterbox_budget_s: float = 1500.0):
    print(gpu_orchestrator.remote(smoke, chatterbox_budget_s))


@app.local_entrypoint()
def run_prep(smoke: bool = False):
    print(prep_real.remote(smoke))


@app.local_entrypoint()
def full(smoke: bool = False):
    s = main.remote(smoke)
    print({k: s.get(k) for k in ("n_items", "audio_hours", "timing_s")})
