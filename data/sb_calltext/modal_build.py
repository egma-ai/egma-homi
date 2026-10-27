"""Build sb_calltext on Modal: prep (CPU) -> LLM labels (1 x H100, vLLM) -> assemble; stages checkpoint to work/.

    modal run --detach data/sb_calltext/modal_build.py::full
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

VOL = "/vol"
OUT = f"{VOL}/stageb/sb_calltext"
WORK = f"{OUT}/work"

volume = modal.Volume.from_name("homi", create_if_missing=False)
_local_files = ["modal_build.py", "build.py", "llm_label.py", "prep.py", "assemble.py"]


def _add_code(img: modal.Image) -> modal.Image:
    img = img.add_local_file(INFERENCE / "common.py", "/root/common.py")
    img = img.add_local_file(INFERENCE / "answer_codes.json", "/root/answer_codes.json")
    for f in _local_files:
        if (HERE / f).exists():
            img = img.add_local_file(HERE / f, f"/root/{f}")
    return img


cpu_image = _add_code(
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_pip_install("numpy==2.3.3", "scipy==1.16.2", "pyarrow==21.0.0", "soundfile==0.13.1",
                    "huggingface_hub==0.35.3", "requests==2.32.5")
    .env({"HF_HOME": f"{VOL}/hf_cache"})
)

gpu_image = _add_code(
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.31.0")
    .env({"HF_HOME": f"{VOL}/hf_cache", "HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false",
          "VLLM_LOGGING_LEVEL": "WARNING", "VLLM_USE_DEEP_GEMM": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)

app = modal.App("sb-calltext-build")
hf_secret = [modal.Secret.from_name("huggingface")]


@app.function(image=gpu_image, gpu="H100", volumes={VOL: volume}, secrets=hf_secret, timeout=3 * 3600)
def label_gpu(in_path: str, out_path: str, meta_path: str) -> dict:
    import llm_label

    volume.reload()
    info = llm_label.run_file(in_path, out_path, meta_path)
    volume.commit()
    return info


@app.function(image=gpu_image, gpu="H100", volumes={VOL: volume}, secrets=hf_secret, timeout=3600)
def smoke_llm_remote() -> dict:
    import llm_label

    llm, info = llm_label.load_llm()
    sysmsg = "You label transcripts of customer-service phone calls. Answer with only the letter of the best option."
    tr = ("AGENT: hello this is harper valley bank how can i help you\n"
          "CALLER: hi i lost my card and this is the third time i'm calling, nobody has helped me, i'm really annoyed\n"
          "AGENT: i'm sorry about that let me replace it for you can i have your name\n"
          "CALLER: john smith\nAGENT: okay a new card is on its way\nCALLER: finally. thanks.")
    reqs = []
    for i, opts in enumerate([["positive", "neutral", "negative"], ["negative", "positive", "neutral"]]):
        letters = "ABC"
        u = (f"Transcript:\n{tr}\n\nQuestion: Overall, how did the caller feel during this call?\n"
             + "\n".join(f"{letters[j]}. {o}" for j, o in enumerate(opts)) + "\nAnswer:")
        reqs.append({"rid": f"s{i}", "system": sysmsg, "user": u, "letters": list(letters)})
    t0 = time.time()
    res = llm_label.score(llm, reqs * 64)
    info["seconds_128"] = round(time.time() - t0, 2)
    info["res"] = res[:2]
    info["prompt0"] = llm_label.build_prompts(llm, reqs[:1])[0][-400:]
    return info


@app.local_entrypoint()
def smoke_llm():
    print(json.dumps(smoke_llm_remote.remote(), indent=1))


# CPU stages
@app.function(image=cpu_image, volumes={VOL: volume}, secrets=hf_secret, timeout=3600, memory=16384, cpu=4)
def prep_hvb_fn(force: bool = False) -> dict:
    import prep

    volume.reload()
    if not force and Path(f"{WORK}/hvb_prep.json").exists():
        return json.loads(Path(f"{WORK}/hvb_prep.json").read_text())
    info = prep.prep_hvb(WORK, f"{OUT}/audio")
    volume.commit()
    return info


@app.function(image=cpu_image, volumes={VOL: volume}, timeout=1800, memory=4096, cpu=1, retries=2, max_containers=64)
def prep_e22_file_fn(fid: str, variant: str = "e22") -> dict:
    import prep

    try:
        info = prep.prep_e22_file(fid, WORK, variant)
    except Exception as e:  # noqa: BLE001 - one bad file must not stop the build
        info = {"fid": fid, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    volume.commit()
    return info


@app.function(image=cpu_image, volumes={VOL: volume}, timeout=3600, memory=8192, cpu=2)
def prep_e22_fn(force: bool = False, limit: int = 0, variant: str = "e22") -> dict:
    import prep

    volume.reload()
    if not force and Path(f"{WORK}/{variant}_prep.json").exists():
        return json.loads(Path(f"{WORK}/{variant}_prep.json").read_text())
    fids = prep.e22_file_ids()
    if limit:
        fids = fids[:limit]
    per_file = list(prep_e22_file_fn.starmap([(f, variant) for f in fids], order_outputs=True))
    volume.reload()
    info = prep.gather_e22(WORK, f"{OUT}/audio", variant)
    info["files_listed"] = len(fids)
    info["per_file_errors"] = [r for r in per_file if "error" in r]
    info["files_without_windows"] = sorted(r["fid"] for r in per_file if r.get("windows") == 0)
    Path(f"{WORK}/{variant}_prep.json").write_text(json.dumps(info, indent=1))
    volume.commit()
    return info


@app.function(image=cpu_image, volumes={VOL: volume}, timeout=1800, memory=4096, cpu=1, retries=2, max_containers=48)
def prep_ami_meeting_fn(info: dict) -> dict:
    import prep

    try:
        out = prep.prep_ami_meeting(info, WORK)
    except Exception as e:  # noqa: BLE001
        out = {"meeting": info["meeting"], "error": f"{type(e).__name__}: {str(e)[:300]}"}
    volume.commit()
    return out


@app.function(image=cpu_image, volumes={VOL: volume}, timeout=3600, memory=8192, cpu=2)
def prep_ami_fn(force: bool = False, limit: int = 0) -> dict:
    import prep

    volume.reload()
    if not force and Path(f"{WORK}/ami_prep.json").exists():
        return json.loads(Path(f"{WORK}/ami_prep.json").read_text())
    meetings = prep.ami_meetings(WORK)
    volume.commit()
    if limit:
        meetings = meetings[:limit]
    per = list(prep_ami_meeting_fn.map(meetings, order_outputs=True))
    volume.reload()
    info = prep.gather_ami(WORK, f"{OUT}/audio")
    info["meetings_listed"] = len(meetings)
    info["errors"] = [r for r in per if "error" in r]
    info["roles_known"] = sum(1 for m in meetings if m["roles"])
    Path(f"{WORK}/ami_prep.json").write_text(json.dumps(info, indent=1))
    volume.commit()
    return info


@app.function(image=cpu_image, volumes={VOL: volume}, timeout=1800, memory=8192)
def make_requests_fn(n_shards: int = 2, domains: str = "hvb,e22", tag: str = "") -> dict:
    import build as B

    volume.reload()
    reqs = []
    for line in (open(f"{WORK}/hvb_convs.jsonl") if "hvb" in domains else []):
        c = json.loads(line)
        for key, (q, opts) in B.HVB_LABELS.items():
            for run in (0, 1):
                reqs.append(B.label_request(f"hvb|{c['sid']}|{key}|{run}", "hvb", c["transcript"], q, opts, run))
    if "ami" in domains.split(",") and Path(f"{WORK}/ami_windows.jsonl").exists():
        for line in open(f"{WORK}/ami_windows.jsonl"):
            w = json.loads(line)
            for key, (q, opts) in B.AMI_LABELS.items():
                for run in (0, 1):
                    reqs.append(B.label_request(f"ami|{w['key']}|{key}|{run}", "ami", w["transcript"], q, opts, run))
    for var in ("e22", "e22s"):
        if var not in domains.split(",") or not Path(f"{WORK}/{var}_windows.jsonl").exists():
            continue
        for line in open(f"{WORK}/{var}_windows.jsonl"):
            w = json.loads(line)
            for key, (q, opts) in B.E22_LABELS.items():
                for run in (0, 1):
                    reqs.append(B.label_request(f"e22|{w['key']}|{key}|{run}", "e22", w["transcript"], q, opts, run))
    # contiguous shards (requests of one transcript stay together -> prefix-cache hits)
    per = (len(reqs) + n_shards - 1) // n_shards
    paths = []
    for k in range(n_shards):
        p = f"{WORK}/requests_{tag}{k}.jsonl"
        with open(p, "w") as f:
            for r in reqs[k * per : (k + 1) * per]:
                f.write(json.dumps(r) + "\n")
        paths.append(p)
    volume.commit()
    return {"n_requests": len(reqs), "shards": paths}


@app.function(image=cpu_image, volumes={VOL: volume}, timeout=3600, memory=16384, cpu=4)
def assemble_fn(staging: bool = False) -> dict:
    import assemble

    volume.reload()
    info = assemble.assemble(WORK, f"{WORK}/staging" if staging else OUT, audio_dir=f"{OUT}/audio")
    volume.commit()
    return info


@app.function(image=cpu_image, volumes={VOL: volume}, timeout=4 * 3600, memory=2048)
def pipeline_fn(stages: str = "prep,label,assemble", n_shards: int = 2, e22_limit: int = 0) -> dict:
    log = {}
    st = set(stages.split(","))
    t0 = time.time()
    if "prep" in st:
        h = prep_hvb_fn.spawn()
        e = prep_e22_fn.spawn(limit=e22_limit)
        log["hvb"] = h.get()
        log["e22"] = e.get()
        print("prep done", round(time.time() - t0), json.dumps(log)[:2000], flush=True)
    if "label" in st:
        r = make_requests_fn.remote(n_shards)
        log["requests"] = r
        calls = [label_gpu.spawn(p, p.replace("requests_", "results_"), p.replace("requests_", "llm_meta_").replace(".jsonl", ".json"))
                 for p in r["shards"]]
        log["label"] = [c.get() for c in calls]
        print("label done", round(time.time() - t0), json.dumps(log["label"])[:2000], flush=True)
    if "assemble" in st:
        log["assemble"] = assemble_fn.remote()
        print("assemble done", round(time.time() - t0), flush=True)
    return log


@app.function(image=cpu_image, volumes={VOL: volume}, timeout=4 * 3600, memory=2048)
def label_fn(domains: str, n_shards: int = 1, tag: str = "", prep_variant: str = "") -> dict:
    if prep_variant == "ami":
        print(json.dumps(prep_ami_fn.remote())[:2000], flush=True)
    elif prep_variant:
        prep_e22_fn.remote(variant=prep_variant)
    r = make_requests_fn.remote(n_shards, domains, tag)
    calls = [label_gpu.spawn(p, p.replace("requests_", "results_"), p.replace("requests_", "llm_meta_").replace(".jsonl", ".json"))
             for p in r["shards"]]
    return {"requests": r, "label": [c.get() for c in calls]}


@app.local_entrypoint()
def label(domains: str = "e22", n_shards: int = 1, tag: str = "e22_", prep_variant: str = ""):
    print(json.dumps(label_fn.remote(domains, n_shards, tag, prep_variant), indent=1)[:3000])


@app.local_entrypoint()
def full(stages: str = "prep,label,assemble", n_shards: int = 2, e22_limit: int = 0):
    print(json.dumps(pipeline_fn.remote(stages, n_shards, e22_limit), indent=1)[:5000])


@app.local_entrypoint()
def prep_e22(variant: str = "e22s"):
    print(json.dumps(prep_e22_fn.remote(variant=variant), indent=1)[:3000])


@app.local_entrypoint()
def assemble_only(staging: bool = True):
    print(json.dumps(assemble_fn.remote(staging), indent=1, default=str)[:6000])
