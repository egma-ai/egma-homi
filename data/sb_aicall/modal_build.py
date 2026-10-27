"""Build sb_aicall on Modal: Kokoro TTS on L4 GPUs, CPU rendering, labels, items, validation, summary.json last.

    modal run --detach data/sb_aicall/modal_build.py::full
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
STAGEB = HERE.parent
INFERENCE = STAGEB.parent / "inference"
for p in (INFERENCE, STAGEB / "sb_phone", STAGEB / "sb_calls", STAGEB / "sb_dialog", HERE):
    sys.path.insert(0, str(p))

VOL = "/vol"
OUT = f"{VOL}/stageb/sb_aicall"
WORK = f"{VOL}/stageb_work/sb_aicall"
RAW_HVB = f"{VOL}/stageb/sb_dialog/work/raw/hvb"
RAW_PHONE = f"{VOL}/stageb/sb_phone/work/raw"
KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_REV = "f3ff3571791e39611d31c381e3a41a3af07b4987"
HVB_SHA = "0bd721e877c4a85d8c13ff837e68661ea6200a98"
SLUE_REF = "asapp/slue-phase-2@e2989c55a53593a8e39b8f8ebdb47ccaccbe484a (hvb test split)"
ROBO_SHA = "5aa6f3bfa8563ce8c1c75ebf8a2271e6ff6b4272"
LIC = {
    "hvb": f"HarperValleyBank, github.com/cricketclub/gridspace-stanford-harper-valley@{HVB_SHA}: CC BY 4.0 "
           "(repo LICENSE; as used by data/sb_dialog)",
    "kokoro": f"{KOKORO_REPO}@{KOKORO_REV}: Apache-2.0 model output (HF cardData.license=apache-2.0, not gated; as "
              "pinned in data/sb_calls)",
    "robocall": f"FTC robocall audio, github.com/wspr-ncsu/robocall-audio-dataset@{ROBO_SHA}: audio public domain "
                "(repo README; as prepared + recorded by data/sb_phone/prep.py)",
    "tones": "ringback / beep / line noise synthesized with numpy (data/sb_calls/audio_fx.py)",
}
N_AGENT_A_PER_CONV = 5.0  # A variants per HVB conversation at scale 1
N_E_PER_CONV = 0.7
N_SYN = {"B": 900, "C": 600, "D": 300}  # synthetic clips at scale 1
SYN_CAL_FRAC = 0.04

volume = modal.Volume.from_name("homi", create_if_missing=False)
_files = {
    HERE: ["modal_build.py", "build.py", "aic_scripts.py", "aic_layout.py", "aic_labels.py", "aic_questions.py",
           "aic_synth.py"],
    STAGEB / "sb_dialog": ["hvb.py", "labels.py"],
    STAGEB / "sb_calls": ["scripts.py", "audio_fx.py"],
    STAGEB / "sb_phone": ["fx.py"],
    INFERENCE: ["common.py", "answer_codes.json"],
}
FORBIDDEN_FILE = STAGEB / "forbidden_questions.json"  # optional, see forbidden_questions()


def _add(img):
    for d, fs in _files.items():
        for f in fs:
            img = img.add_local_file(d / f, f"/root/{f}")
    if FORBIDDEN_FILE.exists():
        img = img.add_local_file(FORBIDDEN_FILE, "/root/forbidden_questions.json")
    return img


cpu_image = _add(
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_pip_install("numpy==2.3.3", "scipy==1.16.2", "soundfile==0.13.1", "modal")
    .env({"HF_HOME": f"{VOL}/hf_cache"})
)
tts_image = _add(
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng", "git")
    .pip_install("torch==2.7.1", "kokoro==0.9.4", "misaki[en]==0.9.4", "transformers==4.51.3",
                 "huggingface_hub==0.30.2", "numpy==1.26.4", "scipy==1.13.1", "soundfile==0.13.1", "modal",
                 "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/"
                 "en_core_web_sm-3.8.0-py3-none-any.whl")
    .env({"HF_HOME": f"{VOL}/hf_cache"})
)
app = modal.App("sb-aicall-build")


# TTS

@app.function(image=tts_image, gpu="L4", volumes={VOL: volume}, timeout=3600, retries=1, max_containers=4)
def tts_shard(args) -> dict:
    """args = (shard id, [(unit, [(key, voice, speed, text)])]). Writes WORK/tts/<unit>.npz (16 kHz float32)."""
    import time

    import numpy as np
    from huggingface_hub import hf_hub_download
    from kokoro import KModel, KPipeline
    from scipy.signal import resample_poly

    sid, units = args
    t0 = time.time()
    todo = []
    for u, jobs in units:
        p = Path(f"{WORK}/tts/{u}.npz")
        if p.exists():
            with np.load(p) as z:
                have = set(z.files)
            if {j[0] for j in jobs} <= have:
                continue
        todo.append((u, jobs))
    if not todo:
        return {"shard": sid, "skipped": len(units)}
    cfg = hf_hub_download(KOKORO_REPO, "config.json", revision=KOKORO_REV)
    mdl = hf_hub_download(KOKORO_REPO, "kokoro-v1_0.pth", revision=KOKORO_REV)
    model = KModel(config=cfg, model=mdl).to("cuda").eval()
    pipes = {lc: KPipeline(lang_code=lc, model=model) for lc in "ab"}
    vpaths, fails, secs, n = {}, [], 0.0, 0
    for u, jobs in todo:
        arrays = {}
        for key, voice, speed, text in jobs:
            if voice not in vpaths:
                vpaths[voice] = hf_hub_download(KOKORO_REPO, f"voices/{voice}.pt", revision=KOKORO_REV)
            pipe = pipes["b" if voice.startswith("b") else "a"]
            try:
                chunks = [r.audio.detach().cpu().numpy() for r in pipe(text, voice=vpaths[voice], speed=speed)
                          if r.audio is not None]
                x = np.concatenate(chunks).astype(np.float64)
                y = resample_poly(x, 2, 3).astype(np.float32)  # 24 kHz -> 16 kHz
                if len(y) < 1600:
                    raise ValueError("too short")
                arrays[key] = y
                secs += len(y) / 16000
                n += 1
            except Exception as e:  # noqa: BLE001
                fails.append([u, key, repr(e)[:120]])
        p = Path(f"{WORK}/tts/{u}.npz")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.rename(p)
    volume.commit()
    return {"shard": sid, "units": len(todo), "segments": n, "speech_h": round(secs / 3600, 2), "fails": fails,
            "elapsed_s": round(time.time() - t0, 1)}


def forbidden_questions(path: str = "/root/forbidden_questions.json") -> list[str]:
    """Question strings no item may equal (optional JSON list at data/forbidden_questions.json; empty by default)."""
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else []


# Rendering

def _low_runs(x, sr, min_len=1.0):
    """Low-energy stretches of the mix (sb_dialog's rule)."""
    import numpy as np

    import hvb
    db = hvb.frame_db(x, sr, 0.02)
    if len(db) == 0:
        return []
    thr = max(np.percentile(db, 10) + 10.0, np.percentile(db, 95) - 35.0)
    low = db < thr
    out, start = [], None
    for i, v in enumerate(np.append(low, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if (i - start) * 0.02 >= min_len:
                out.append((round(start * 0.02, 3), round(i * 0.02, 3)))
            start = None
    return out


def _fade(x, f_in, f_out):
    import numpy as np
    x = x.astype(np.float32).copy()
    L = len(x)
    a, b = min(f_in, L // 2), min(f_out, L // 2)
    if a > 0:
        x[:a] *= np.linspace(0, 1, a, dtype=np.float32)
    if b > 0:
        x[L - b:] *= np.linspace(1, 0, b, dtype=np.float32)
    return x


def _rms_over(x, iv, sr):
    import numpy as np
    seg = [x[int(s * sr):int(e * sr)] for s, e in iv]
    seg = [s for s in seg if len(s)]
    if not seg:
        return 1e-6
    v = np.concatenate(seg)
    return float(np.sqrt(np.mean(v.astype(np.float64) ** 2)) + 1e-9)


def tts_vad(y, sr, hop=0.02):
    """Speech intervals of a clean TTS segment: frames above max(p95 - 35 dB, -70 dBFS), gaps <= 0.1 s merged.
    A low-percentile floor would not work: TTS segments have almost no silent frames."""
    import numpy as np

    import hvb
    import labels as L
    db = hvb.frame_db(y, sr, hop)
    if len(db) == 0:
        return []
    act = db > max(np.percentile(db, 95) - 35.0, -70.0)
    iv, start = [], None
    for i, a in enumerate(np.append(act, False)):
        if a and start is None:
            start = i
        elif not a and start is not None:
            iv.append((start * hop, i * hop))
            start = None
    return [(round(s, 3), round(e, 3)) for s, e in L.merge(iv, 0.1) if e - s >= 0.08]


def _tts_turn(y, sr):
    import audio_fx
    y = audio_fx.trim_silence(y)
    words = tts_vad(y, sr)
    if not words:
        return None
    return y, words


def render_hvb(job, rng_seed_base):
    """job: {sid, split, voices, outbound_ok, variants: [{key, kind, vi, outbound}]} -> (clips, records, errors)."""
    import json
    import random

    import numpy as np
    import soundfile as sf

    import aic_labels as AL
    import aic_layout as LY
    import aic_scripts as S
    import build
    import common
    import hvb

    sr = common.SAMPLE_RATE
    sid = job["sid"]
    base = Path(RAW_HVB)
    segs = hvb.parse_transcript(json.loads((base / "transcript" / f"{sid}.json").read_text()))
    chans = {}
    for role in hvb.ROLES:
        x, r = sf.read(base / "audio" / role / f"{sid}.wav", dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        chans[role] = common.resample(x, r, sr)
    turns = build.hvb_turns(segs)
    tts = {}
    tp = Path(f"{WORK}/tts/hvb_{sid}.npz")
    if tp.exists():
        with np.load(tp) as z:
            tts = {k: z[k] for k in z.files}
    # backchannel snippets: single short words of the caller
    bcs = []
    for g in segs:
        if g["role"] != "caller" or not g["speech"] or len(g["words"]) != 1:
            continue
        w = S.clean_text(g["text"]).replace(".", "")
        (w0, w1), = g["words"]
        if w in S.BACKCHANNEL_WORDS and 0.2 <= w1 - w0 <= 0.6:
            bcs.append({"src": (w0 - 0.06, w1 + 0.06), "word": (0.06, 0.06 + (w1 - w0)), "text": w})
    caller_segs = [g for g in segs if g["role"] == "caller"]
    bed = hvb.noise_bed(chans["caller"], sr, caller_segs)
    clips, recs, errs = {}, [], []
    for v in job["variants"]:
        key = v["key"]
        try:
            kind = v["kind"]
            tlist = []
            for ti, t in enumerate(turns):
                if t["role"] == "agent" and kind == "A":
                    y = tts.get(f"v{v['vi']}_t{ti}")
                    if y is None:
                        continue
                    tt = _tts_turn(y, sr)
                    if tt is None:
                        continue
                    tlist.append({"role": "agent", "dur": len(tt[0]) / sr, "words": tt[1], "incl": tt[1],
                                  "src": ("tts", f"v{v['vi']}_t{ti}"), "audio": tt[0], "ti": ti})
                else:
                    tlist.append({"role": "agent" if t["role"] == "agent" else "other", "dur": t["dur"],
                                  "words": t["words"], "incl": t["incl"], "ti": ti,
                                  "src": ("ch", t["role"], t["s"])})
            outbound = bool(v.get("outbound")) and tlist and tlist[0]["role"] == "agent" and tlist[0]["ti"] == 0
            if outbound:
                y = tts.get(f"v{v['vi']}_open")
                tt = _tts_turn(y, sr) if y is not None else None
                if tt is None:
                    outbound = False
                else:
                    tlist[0] = {"role": "agent", "dur": len(tt[0]) / sr, "words": tt[1], "incl": tt[1],
                                "src": ("tts", f"v{v['vi']}_open"), "audio": tt[0], "ti": 0}
            rc = LY.draw_recipe(random.Random(f"{rng_seed_base}:{key}"), kind)
            if outbound:
                rc["start0"] = True
            lay = LY.make_layout(tlist, rc, f"{rng_seed_base}:{key}", bcs)
            if lay is None:
                errs.append([key, "no layout"])
                continue
            T = lay["T"]
            n = int(round(T * sr))
            rng = random.Random(f"{rng_seed_base}:{key}:render")
            a_tr = np.zeros(n, np.float32)
            if rc["gated"]:
                o_tr = np.zeros(n, np.float32)
            else:
                off = rng.randrange(len(bed))
                o_tr = np.tile(bed, int(np.ceil((n + off) / len(bed))) + 1)[off:off + n].astype(np.float32).copy()
            f10 = int(0.01 * sr)

            def put(tr, x, dst, crossfade):
                i = int(round(dst * sr))
                if i < 0:
                    x, i = x[-i:], 0
                L = min(len(x), n - i)
                if L <= 0:
                    return
                x = x[:L]
                if crossfade:
                    w = _fade(np.ones(L, np.float32), f10, f10)
                    tr[i:i + L] = tr[i:i + L] * (1 - w) + x * w
                else:
                    tr[i:i + L] += _fade(x, f10, f10)

            for q in lay["placed"]:
                t = q["turn"]
                if t["src"][0] == "tts":
                    x = t["audio"][:int(round(t["dur"] * sr))]
                    if q["cut"] is not None:
                        x = _fade(x[:int(round((q["cut"] + 0.03) * sr))], f10, int(0.03 * sr))
                    put(a_tr, x, q["st"], False)
                else:
                    _, ch, s0 = t["src"]
                    end = s0 + (q["cut"] + 0.03 if q["cut"] is not None else t["dur"])
                    x = chans[ch][max(0, int(round((s0 - hvb.PAD) * sr))):int(round((end + hvb.PAD) * sr))]
                    if t["role"] == "agent":
                        put(a_tr, x, q["st"] - hvb.PAD, False)
                    else:
                        put(o_tr, x, q["st"] - hvb.PAD, not rc["gated"])
            for b in lay["bc"]:
                s0, s1 = b["bc"]["src"]
                x = chans["caller"][int(round(s0 * sr)):int(round(s1 * sr))]
                put(o_tr, x, b["st"], not rc["gated"])
            ra = _rms_over(a_tr, lay["speech"]["agent"], sr)
            ro = _rms_over(o_tr, lay["speech"]["other"], sr)
            o_tr = o_tr * (ra / ro) * 10 ** (rc["other_gain_db"] / 20)
            mix = a_tr + o_tr
            pk = float(np.abs(mix).max())
            g = rc["peak"] / pk if pk > 0 else 1.0
            mix = np.clip(mix * g, -0.99, 0.99).astype(np.float32)
            ev = {"agent": hvb.energy_vad(a_tr * g, sr), "other": hvb.energy_vad(o_tr * g, sr)}
            sp_o = lay["speech"]["other"]
            cov = L_cov(sp_o, ev["other"])
            if cov < 0.75:
                errs.append([key, f"align {cov:.2f}"])
                continue
            low = _low_runs(mix, sr)
            truth, facts = AL.two_party(lay["speech"], lay["incl"], T, low)
            vad, _ = AL.two_party(ev, ev, T, low)
            labs = AL.agree(truth, vad)
            direction = "out" if outbound else "in"
            labs["live_out" if outbound else "live_in"] = [("yes", None)]
            labs["live_choice"] = [("live", None)]
            labs["auto_out" if outbound else "auto_in"] = [("no", None)]
            clips[key] = mix
            recs.append({"key": key, "split": job["split"], "scenario": kind, "direction": direction,
                         "sub": f"{kind}_{direction}", "labels": {f: [list(x) for x in c] for f, c in labs.items()},
                         "dur": len(mix) / sr, "licence": [LIC["hvb"]] + ([LIC["kokoro"]] if kind == "A" else []),
                         "source_id": f"hvb:{sid}", "speakers": job["speakers"],
                         "voices": [job["voices"][v["vi"]]] if kind == "A" else [], "events": lay["events"],
                         "facts": {**facts, "align": round(cov, 3), "T": T, "start_turn": lay["start_turn"],
                                   "n_turns": lay["n_turns"], "recipe": {k: (round(x, 3) if isinstance(x, float) else x)
                                                                         for k, x in rc.items()}}})
        except Exception as e:  # noqa: BLE001
            errs.append([key, repr(e)[:200]])
    return clips, recs, errs


def L_cov(truth_iv, vad_iv):
    import labels as L
    tot = L.total(truth_iv)
    if tot <= 0:
        return 1.0
    inter = [(max(s, a), min(e, b)) for s, e in truth_iv for a, b in vad_iv if min(e, b) > max(s, a)]
    return L.total(inter) / tot


def render_syn(plans, robo_bank):
    import random

    import numpy as np

    import aic_synth as SY
    import audio_fx
    import common
    import fx

    sr = common.SAMPLE_RATE
    clips, recs, errs = {}, [], []
    tts_cache = {}
    for p in plans:
        key = p["key"]
        try:
            u = p["unit"]
            if u not in tts_cache:
                tp = Path(f"{WORK}/tts/{u}.npz")
                with np.load(tp) as z:
                    tts_cache[u] = {k: z[k] for k in z.files}
            tts = tts_cache[u]
            aa = [tts.get(f"{key}|a{i}") for i in range(len(p["agent_lines"]))]
            oa = [tts.get(f"{key}|o{i}") for i in range(len(p["other_lines"]))]
            if any(x is None for x in aa + oa):
                errs.append([key, "tts missing"])
                continue
            aa = [audio_fx.trim_silence(x) for x in aa]
            oa = [audio_fx.trim_silence(x) for x in oa]
            robo = robo_bank.get(p["robo"]) if p.get("robo") else None
            a, o, T, facts = SY.assemble(p, aa, oa, robo)
            rng = random.Random(p["seed"] + 7)
            if p["phone_fx"]:
                codec = rng.choice(["mulaw", "alaw", "gsm", "opus"])
                o = fx.bandlimit(o)
                try:
                    o = fx.ffmpeg_codec(o, codec, rng.choice([8, 10, 12]) if codec == "opus" else None)
                except RuntimeError:
                    codec = "mulaw"
                    o = fx.ffmpeg_codec(o, codec)
                facts["codec"] = codec
            mix = a + o
            pk = float(np.abs(mix).max())
            mix = np.clip(mix * (rng.uniform(0.4, 0.95) / pk if pk > 0 else 1.0), -0.99, 0.99).astype(np.float32)
            labs = SY.labels(p, facts)
            clips[key] = mix
            lic = [LIC["kokoro"], LIC["tones"]] + ([LIC["robocall"]] if p.get("robo") else [])
            recs.append({"key": key, "split": p["split"], "scenario": p["scenario"], "direction": p["direction"],
                         "sub": p["sub"], "labels": {f: [list(x) for x in c] for f, c in labs.items()},
                         "dur": len(mix) / sr, "licence": lic,
                         "source_id": f"robocall:{p['robo']}" if p.get("robo") else f"synthetic:{key}",
                         "speakers": [f"robocall_case:{p['robo_case']}"] if p.get("robo") else [],
                         "voices": [p["agent_voice"]] + ([p["other_voice"]] if p["other_lines"] else []),
                         "events": [], "facts": {k: v for k, v in facts.items() if k != "speech"} |
                         {"speech": facts["speech"]}, "plan": {k: v for k, v in p.items() if k not in ("unit",)}})
        except Exception as e:  # noqa: BLE001
            errs.append([key, repr(e)[:200]])
    return clips, recs, errs


@app.function(image=cpu_image, volumes={VOL: volume}, timeout=2 * 3600, cpu=2, memory=8192, retries=1,
              max_containers=48)
def render_shard(args) -> dict:
    """args = (bank, out_dir, seed_base, hvb_jobs, syn_plans). Writes <out_dir>/audio/<bank>.{npy,json}."""
    import json

    import common

    bank, out_dir, seed_base, hvb_jobs, syn_plans = args
    done = Path(out_dir) / "work" / "render" / f"{bank}.json"
    if done.exists() and (Path(out_dir) / "audio" / f"{bank}.json").exists():
        return json.loads(done.read_text())
    clips, recs, errs = {}, [], []
    for job in hvb_jobs:
        c, r, e = render_hvb(job, seed_base)
        clips.update(c)
        recs += r
        errs += e
    if syn_plans:
        rb = common.AudioBank(RAW_PHONE, "robocall")
        c, r, e = render_syn(syn_plans, rb)
        clips.update(c)
        recs += r
        errs += e
    for r in recs:
        r["bank"] = bank
    common.write_audio_bank(Path(out_dir) / "audio", bank, clips)
    res = {"bank": bank, "records": recs, "errors": errs,
           "hours": round(sum(len(v) for v in clips.values()) / common.SAMPLE_RATE / 3600, 3)}
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(res))
    volume.commit()
    return res


# Main build

@app.function(image=cpu_image, volumes={VOL: volume}, timeout=5 * 3600, cpu=8, memory=32768)
def main(out_dir: str = OUT, scale: float = 1.0, smoke: bool = False) -> dict:
    import ast
    import hashlib
    import json
    import random
    import time
    from collections import Counter, defaultdict

    import numpy as np

    import aic_questions as Q
    import aic_scripts as S
    import aic_synth as SY
    import build
    import common
    import hvb

    t_start = time.time()
    log = []

    def note(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.append(msg)

    out = Path(out_dir)
    if out_dir == OUT:
        assert not (out / "summary.json").exists(), "summary.json already exists: the folder is frozen"
    (out / "work").mkdir(parents=True, exist_ok=True)
    seed_base = f"{common.SEED}:sb_aicall"

    # Forbidden questions (optional)
    forb = forbidden_questions()
    note("forbidden questions", len(forb))

    # Leakage sources
    adi_calls, adi_spk, adi_ids = set(), set(), set()
    adi_path = Path(VOL) / "adi" / "items.jsonl"
    adi_text = adi_path.read_text()
    for line in adi_text.splitlines():
        it = json.loads(line)
        m = it.get("meta") or {}
        if it.get("source") == "hvb":
            adi_calls.add(str(m.get("call")))
            adi_spk.add(str(m.get("speaker")))
        for k in ("call", "recording", "file", "source_id", "meeting", "clip"):
            if m.get(k) is not None:
                adi_ids.add(str(m[k]))
    ev_path = Path(VOL) / "data" / "eval_items.jsonl"
    eval_ids = set()
    eval_text = ev_path.read_text() if ev_path.exists() else ""
    for line in eval_text.splitlines():
        m = json.loads(line).get("meta") or {}
        for k in ("call", "recording", "file", "source_id", "meeting", "clip", "speaker"):
            if m.get(k) is not None:
                eval_ids.add(str(m[k]))
    slue_test = set(json.loads((Path(RAW_HVB) / "slue_ids.json").read_text())["test"])
    note("eval-set hvb calls", len(adi_calls), "| SLUE test calls", len(slue_test), "| eval-set ids", len(adi_ids))

    # HVB conversations, speaker split
    base = Path(RAW_HVB)
    sids = sorted(p.stem for p in (base / "transcript").glob("*.json"))
    conv_spk, excluded = {}, Counter()
    for sid in sids:
        if sid in adi_calls or sid in slue_test:
            excluded["adi_or_slue_test"] += 1
            continue
        md = json.loads((base / "metadata" / f"{sid}.json").read_text())
        conv_spk[sid] = {f"hvb_spk:{md['agent']['speaker_id']}", f"hvb_spk:{md['caller']['speaker_id']}"}
    hsplit, cal_spk = build.hvb_speaker_split(conv_spk, 0.03)
    note("HVB usable", len(conv_spk), "excluded", dict(excluded), "split", Counter(hsplit.values()),
         "CAL speakers", sorted(cal_spk))
    use = sorted(s for s, v in hsplit.items() if v is not None)
    if smoke:
        tr = [s for s in use if hsplit[s] == "train"][:10]
        ca = [s for s in use if hsplit[s] == "CAL"][:2]
        use = tr + ca

    rng = random.Random(seed_base)
    tts_units = []  # (unit, [(key, voice, speed, text)])
    hvb_jobs = []
    n_a = n_e = 0
    for sid in use:
        split = hsplit[sid]
        segs = hvb.parse_transcript(json.loads((base / "transcript" / f"{sid}.json").read_text()))
        turns = build.hvb_turns(segs)
        if len(turns) < 4:
            continue
        crng = random.Random(f"{seed_base}:conv:{sid}")
        pool = S.CAL_VOICES if split == "CAL" else S.TRAIN_VOICES
        voices = crng.sample(pool, 2)
        jobs = []
        for vi, voice in enumerate(voices):
            speed = round(crng.uniform(0.9, 1.1), 3)
            for ti, t in enumerate(turns):
                if t["role"] == "agent":
                    txt = S.turn_text(t["texts"])
                    if txt:
                        jobs.append((f"v{vi}_t{ti}", voice, speed, txt))
            jobs.append((f"v{vi}_open", voice, speed, S.fill(crng.choice(S.CALLBACK_OPEN), crng)))
        tts_units.append((f"hvb_{sid}", jobs))
        na = int(N_AGENT_A_PER_CONV * scale) + (crng.random() < (N_AGENT_A_PER_CONV * scale) % 1)
        ne = int(N_E_PER_CONV * scale) + (crng.random() < (N_E_PER_CONV * scale) % 1)
        if smoke:
            na, ne = 3, 1
        variants = []
        for j in range(na):
            variants.append({"key": f"aic_hvb_{sid}_a{j}", "kind": "A", "vi": j % 2,
                             "outbound": crng.random() < 0.3})
        for j in range(ne):
            variants.append({"key": f"aic_hvb_{sid}_e{j}", "kind": "E", "vi": 0, "outbound": False})
        n_a += na
        n_e += ne
        hvb_jobs.append({"sid": sid, "split": split, "voices": voices, "variants": variants,
                         "speakers": sorted(conv_spk[sid])})
    note("HVB convs", len(hvb_jobs), "A variants", n_a, "E variants", n_e)

    # Synthetic B/C/D
    robo_meta = json.loads((Path(RAW_PHONE) / "robocall_meta.json").read_text())
    robo_index = json.loads((Path(RAW_PHONE) / "robocall.json").read_text())
    robo_keys = sorted(k for k in robo_index if robo_index[k][1] >= 8 * 16000 and k in robo_meta)
    cases = sorted({robo_meta[k]["case"] for k in robo_keys}, key=lambda c: hashlib.md5(c.encode()).hexdigest())
    cal_cases, ncal = set(), 0
    for c in cases:
        if ncal >= 0.04 * len(robo_keys):
            break
        cal_cases.add(c)
        ncal += sum(1 for k in robo_keys if robo_meta[k]["case"] == c)
    robo_split = {k: ("CAL" if robo_meta[k]["case"] in cal_cases else "train") for k in robo_keys}
    note("robocalls >= 8 s", len(robo_keys), "CAL recordings", ncal, "CAL cases", len(cal_cases))
    syn = []
    srng = random.Random(f"{seed_base}:syn")
    for sc, n in N_SYN.items():
        n = max(6, round(n * scale)) if not smoke else 6
        n_cal = max(1, round(n * SYN_CAL_FRAC))
        for j in range(n):
            split = "CAL" if j < n_cal else "train"
            voices = S.CAL_VOICES if split == "CAL" else S.TRAIN_VOICES
            rk = [k for k in robo_keys if robo_split[k] == split]
            p = SY.plan_clip(f"aic_syn_{sc}{j:04d}", sc, split, srng, voices, rk)
            if p.get("robo"):
                p["robo_case"] = robo_meta[p["robo"]]["case"]
            syn.append(p)
    per_unit = 40
    for i in range(0, len(syn), per_unit):
        u = f"syn{'smoke' if smoke else ''}_{i // per_unit:03d}"
        jobs = []
        for p in syn[i:i + per_unit]:
            p["unit"] = u
            jobs += [(f"{p['key']}|a{k}", p["agent_voice"], p["agent_speed"], t) for k, t in enumerate(p["agent_lines"])]
            jobs += [(f"{p['key']}|o{k}", p["other_voice"], p["other_speed"], t) for k, t in enumerate(p["other_lines"])]
        tts_units.append((u, jobs))
    note("synthetic clips", len(syn), Counter((p["scenario"], p["sub"], p["split"]) for p in syn))

    # TTS
    n_sh = 4 if smoke else 16
    units_sorted = sorted(tts_units, key=lambda u: -sum(len(j[3]) for j in u[1]))
    shards = [[] for _ in range(n_sh)]
    load = [0] * n_sh
    for u in units_sorted:
        k = int(np.argmin(load))
        shards[k].append(u)
        load[k] += sum(len(j[3]) for j in u[1])
    t_tts = time.time()
    tts_fails = []
    for r in tts_shard.map(list(enumerate(shards)), order_outputs=False):
        tts_fails += r.get("fails", [])
        note("tts", {k: v for k, v in r.items() if k != "fails"}, "fails", len(r.get("fails", [])))
    note("TTS done", round(time.time() - t_tts), "s; fails", len(tts_fails), tts_fails[:5])
    volume.reload()

    # Render
    tag = "smoke_" if smoke else ""
    shard_args = []
    per_h = 4 if smoke else 22
    for i in range(0, len(hvb_jobs), per_h):
        shard_args.append((f"sb_aicall_{tag}hvb_{i // per_h:02d}", out_dir, seed_base, hvb_jobs[i:i + per_h], []))
    per_s = 160
    for i in range(0, len(syn), per_s):
        shard_args.append((f"sb_aicall_{tag}syn_{i // per_s:02d}", out_dir, seed_base, [], syn[i:i + per_s]))
    note("render shards", len(shard_args))
    t_r = time.time()
    recs, errors, hours = [], [], Counter()
    for res in render_shard.map(shard_args, order_outputs=False):
        recs += res["records"]
        errors += res["errors"]
        hours[res["bank"]] = res["hours"]
    note("rendered", len(recs), "clips in", round(time.time() - t_r), "s; errors", len(errors), errors[:8],
         "hours", round(sum(hours.values()), 2))
    volume.reload()

    # Items
    for r in recs:
        r["labels"] = {f: [tuple(x) for x in v] for f, v in r["labels"].items()}
    sup = Counter((f, a, str(p)) for r in recs for f, v in r["labels"].items() for a, p in v)
    note("label supply", sorted(sup.items()))
    picked = build.select_items(recs, random.Random(f"{seed_base}:sel"))
    items = build.make_items(picked, random.Random(f"{seed_base}:items"), common)
    items.sort(key=lambda it: it["id"])
    train = [it for it in items if it["set"] == "train"]
    cal = [it for it in items if it["set"] == "CAL"]
    random.Random(f"{seed_base}:order").shuffle(train)
    for name, rows in (("items_train.jsonl", train), ("items_cal.jsonl", cal)):
        common.write_jsonl(out / f"{name}.tmp", rows)
    note("items", len(items), "train", len(train), "cal", len(cal))

    # Validation (re-read from disk)
    reread = common.read_jsonl(out / "items_train.jsonl.tmp") + common.read_jsonl(out / "items_cal.jsonl.tmp")
    banks = sorted(hours)
    bank_index = {b: json.loads((out / "audio" / f"{b}.json").read_text()) for b in banks}
    dur_by_key = {r["key"]: r["dur"] for r in recs}
    val = build.validate(reread, bank_index, dur_by_key, forb)
    for b in banks:
        arr = np.load(out / "audio" / f"{b}.npy", mmap_mode="r")
        assert max((s + n for s, n in bank_index[b].values()), default=0) == arr.shape[0], b
    ab = {}
    for it in random.Random(1).sample(reread, min(200, len(reread))):
        b = it["audio"]["bank"]
        ab.setdefault(b, common.AudioBank(out / "audio", b))
        x = ab[b].get(it["audio"]["key"])
        assert np.isfinite(x).all() and 0.05 < float(np.abs(x).max()) <= 0.991, it["id"]
        assert abs(len(x) / 16000 - dur_by_key[it["audio"]["key"]]) < 0.01
    bank_sizes = {b: (out / "audio" / f"{b}.npy").stat().st_size for b in banks}
    assert max(bank_sizes.values()) < 4.0e9, bank_sizes
    # CAL / train disjointness: speakers (HVB speaker ids, robocall cases) and Kokoro voices
    used = {it["audio"]["key"] for it in reread}
    rec_by = {r["key"]: r for r in recs}
    spk = {s: {x for k in used if rec_by[k]["split"] == s for x in rec_by[k]["speakers"]} for s in ("train", "CAL")}
    vox = {s: {x for k in used if rec_by[k]["split"] == s for x in rec_by[k]["voices"]} for s in ("train", "CAL")}
    srcid = {s: {rec_by[k]["source_id"] for k in used if rec_by[k]["split"] == s} for s in ("train", "CAL")}
    split_check = {"speaker_overlap": sorted(spk["train"] & spk["CAL"]), "voice_overlap": sorted(vox["train"] & vox["CAL"]),
                   "source_id_overlap": len(srcid["train"] & srcid["CAL"])}
    assert not split_check["speaker_overlap"] and not split_check["voice_overlap"] and not split_check["source_id_overlap"]
    note("validation", {k: v for k, v in val.items() if k != "balance"}, split_check)

    # Leakage
    our_hvb = {r["source_id"].split(":", 1)[1] for r in recs if r["source_id"].startswith("hvb:") and r["key"] in used}
    our_ids = {r["source_id"].split(":", 1)[1] for r in recs if r["key"] in used} | used
    leak = {
        "hvb_calls_in_adi": len(our_hvb & adi_calls), "hvb_calls_in_slue_test": len(our_hvb & slue_test),
        "ids_in_adi_meta": len(our_ids & adi_ids), "ids_in_exp0_eval_meta": len(our_ids & eval_ids),
        "hvb_speakers_shared_with_adi": len({s.split(":")[1] for k in used for s in rec_by[k]["speakers"]
                                             if s.startswith("hvb_spk:")} & adi_spk),
        "banks_referenced_by_adi_or_eval": adi_text.count('"sb_aicall_') + eval_text.count('"sb_aicall_'),
        "private_paths_used": 0,
        "questions_equal_to_forbidden_questions": 0,
        "note": "HVB: every SLUE-HVB test conversation (a superset of the evaluation set's HVB calls) is excluded at "
                "conversation level, as in sb_dialog; HVB has ~60 voice actors, so speaker-level exclusion is "
                "impossible.",
    }
    assert leak["hvb_calls_in_adi"] == 0 and leak["hvb_calls_in_slue_test"] == 0 and leak["ids_in_adi_meta"] == 0
    assert leak["ids_in_exp0_eval_meta"] == 0 and leak["banks_referenced_by_adi_or_eval"] == 0
    note("leakage", leak)

    # Finalize files, summary.json last
    for name in ("items_train.jsonl", "items_cal.jsonl"):
        (out / f"{name}.tmp").rename(out / name)
    used_recs = [rec_by[k] for k in sorted(used)]
    by_src = defaultdict(Counter)
    for it in reread:
        by_src[it["source"]][f'{it["set"]}|{it["meta"]["qtype"]}|{it["meta"]["family"]}'] += 1
    examples = {}
    for fam in sorted({it["meta"]["family"] for it in reread}):
        exs = [it for it in reread if it["meta"]["family"] == fam and it["set"] == "train"] or \
              [it for it in reread if it["meta"]["family"] == fam]
        examples[fam] = [{"question": it["question"], "options": it["options"], "gold": it["gold"],
                          "source": it["source"], "key": it["audio"]["key"]}
                         for it in random.Random(fam).sample(exs, min(3, len(exs)))]
    scen = Counter(r["scenario"] for r in used_recs)
    summary = {
        "builder": "sb_aicall", "built_at": time.strftime("%Y-%m-%d %H:%M:%S"), "out_dir": out_dir,
        "n_items": len(reread), "n_train": sum(it["set"] == "train" for it in reread),
        "n_cal": sum(it["set"] == "CAL" for it in reread),
        "cal_frac": round(sum(it["set"] == "CAL" for it in reread) / max(1, len(reread)), 4),
        "counts_by_source": {s: sum(c.values()) for s, c in by_src.items()},
        "counts_by_source_split_qtype_family": {s: dict(sorted(c.items())) for s, c in by_src.items()},
        "counts_by_qtype": dict(Counter(it["meta"]["qtype"] for it in reread)),
        "counts_by_family_split": dict(sorted(Counter(f'{it["meta"]["family"]}|{it["set"]}' for it in reread).items())),
        "counts_by_split": dict(Counter(it["set"] for it in reread)),
        "clips_used": len(used), "clips_rendered": len(recs),
        "clips_by_scenario": dict(sorted(scen.items())),
        "clip_share_by_scenario": {k: round(v / len(used), 3) for k, v in sorted(scen.items())},
        "clips_by_sub_split": dict(sorted(Counter(f'{r["sub"]}|{r["split"]}' for r in used_recs).items())),
        "audio_hours_used": round(sum(r["dur"] for r in used_recs) / 3600, 2),
        "audio_hours_in_banks": round(sum(hours.values()), 2),
        "clip_duration_s": {"min": val["dur_min"], "max": val["dur_max"], "mean": val["dur_mean"]},
        "channel": dict(Counter(it["channel"] for it in reread)),
        "label_balance": val["balance"],
        "banks": {b: {"bytes": bank_sizes[b], "clips": len(bank_index[b])} for b in banks},
        "sources": {"hvb": LIC["hvb"], "kokoro": LIC["kokoro"], "robocall": LIC["robocall"], "tones": LIC["tones"]},
        "licences_by_item_source": {
            "sb_aicall_ai_human": "HVB caller audio CC BY 4.0 + Kokoro-82M Apache-2.0 agent voice",
            "sb_aicall_two_humans": "HVB CC BY 4.0", "sb_aicall_voicemail": "Kokoro Apache-2.0 + numpy tones",
            "sb_aicall_automated": "Kokoro Apache-2.0 + FTC robocall audio (public domain)",
            "sb_aicall_nobody": "Kokoro Apache-2.0 + numpy tones"},
        "split_rule": f"CAL = HVB conversations whose agent AND caller are CAL speakers ({sorted(cal_spk)}; pair "
                      f"greedy, mixed conversations dropped) + Kokoro voices {S.CAL_VOICES} (never used in train) + "
                      f"robocall recordings of held-out FTC cases ({len(cal_cases)} cases)",
        "split_check": split_check,
        "slue_exclusion": SLUE_REF,
        "leakage": leak,
        "validation": {k: v for k, v in val.items() if k != "balance"},
        "forbidden_questions_checked": len(forb),
        "n_agent_templates": S.n_agent_templates(),
        "tts_fails": tts_fails[:50], "render_errors": errors[:80], "n_render_errors": len(errors),
        "render_error_kinds": dict(Counter(e[1].split(" ")[0] for e in errors)),
        "examples": examples,
        "log": log,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    (out / "summary.json.tmp").write_text(json.dumps(summary, indent=1))
    (out / "summary.json.tmp").rename(out / "summary.json")
    volume.commit()
    return {k: summary[k] for k in ("n_items", "n_train", "n_cal", "clips_used", "clips_by_scenario",
                                     "counts_by_qtype", "audio_hours_used", "elapsed_s")}


@app.local_entrypoint()
def smoke():
    print(main.remote(f"{WORK}/smoke", 0.05, True))


@app.local_entrypoint()
def full(scale: float = 1.0):
    print(main.remote(OUT, scale, False))


# Independent read-only re-check

@app.function(image=cpu_image, volumes={VOL: volume}, timeout=3600, cpu=4, memory=16384)
def check_build(out_dir: str = OUT) -> dict:
    """Re-validate the written dataset from disk with inference/common.py helpers (read-only)."""
    import json
    import random
    from collections import Counter, defaultdict

    import numpy as np

    import aic_questions as Q
    import common

    out = Path(out_dir)
    items = common.read_jsonl(out / "items_train.jsonl") + common.read_jsonl(out / "items_cal.jsonl")
    summary = json.loads((out / "summary.json").read_text())
    rep = {"n_items": len(items)}
    keys = {"id", "set", "source", "question", "options", "target", "gold", "audio", "channel", "state_text", "meta"}
    bad = Counter()
    idx = {}
    for b in sorted({it["audio"]["bank"] for it in items}):
        idx[b] = json.loads((out / "audio" / f"{b}.json").read_text())
    bal = defaultdict(Counter)
    ids = Counter(it["id"] for it in items)
    bad["dup_ids"] = sum(1 for v in ids.values() if v > 1)
    forb = {q.strip().lower() for q in forbidden_questions()}
    slue = set(json.loads((Path(RAW_HVB) / "slue_ids.json").read_text())["test"])
    spk, vox, src = defaultdict(set), defaultdict(set), defaultdict(set)
    per_clip = Counter()
    for it in items:
        bad["schema"] += not keys <= set(it)
        bad["target_sum"] += abs(sum(it["target"]) - 1) > 1e-6
        bad["gold_argmax"] += it["gold"] != int(np.argmax(it["target"]))
        bad["n_options"] += not (2 <= len(it["options"]) <= 20)
        bad["audio_key_missing"] += it["audio"]["key"] not in idx[it["audio"]["bank"]]
        bad["forbidden_question"] += it["question"].strip().lower() in forb
        m = it["meta"]
        bad["qtype"] += m.get("qtype") not in ("choice", "noul", "score")
        if m["qtype"] == "noul":
            bad["noul_options"] += sorted(it["options"]) != ["no", "yes"]
            bal[(m["family"], it["set"])][it["options"][it["gold"]]] += 1
        sid = m["source_id"]
        bad["slue_test_conv"] += sid.startswith("hvb:") and sid.split(":", 1)[1] in slue
        spk[it["set"]] |= set(m["speakers"])
        vox[it["set"]] |= set(m["voices"])
        src[it["set"]].add(sid)
        per_clip[it["audio"]["key"]] += 1
    rep["errors"] = {k: v for k, v in bad.items()}
    rep["noul_balance"] = {f"{f}|{s}": dict(c) for (f, s), c in sorted(bal.items())}
    rep["noul_unbalanced"] = [k for k, c in rep["noul_balance"].items() if c.get("yes", 0) != c.get("no", 0)]
    rep["cal_train_speaker_overlap"] = sorted(spk["train"] & spk["CAL"])
    rep["cal_train_voice_overlap"] = sorted(vox["train"] & vox["CAL"])
    rep["cal_train_source_overlap"] = len(src["train"] & src["CAL"])
    rep["max_items_per_clip"] = max(per_clip.values())
    banks = {b: common.AudioBank(out / "audio", b) for b in idx}
    durs = []
    for it in random.Random(7).sample(items, min(400, len(items))):
        x = common.item_audio(it, banks)
        assert x is not None and np.isfinite(x).all() and float(np.abs(x).max()) <= 1.0
        durs.append(len(x) / 16000)
    rep["sample_loaded"] = len(durs)
    rep["sample_dur_range"] = [round(min(durs), 2), round(max(durs), 2)]
    rep["summary_n_items"] = summary["n_items"]
    rep["sizes_gb"] = round(sum((out / "audio" / f"{b}.npy").stat().st_size for b in idx) / 1e9, 2)
    return rep


@app.local_entrypoint()
def check(out_dir: str = OUT):
    import json
    print(json.dumps(check_build.remote(out_dir), indent=1))
