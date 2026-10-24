"""Container-side data preparation for sb_calltext (HVB whole calls, Earnings-22 windows). Runs on Modal CPU."""

from __future__ import annotations

import glob
import io
import json
import os
import random
import subprocess
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

import build as B

SR = 16_000
HVB_SHA = "0bd721e877c4a85d8c13ff837e68661ea6200a98"
HVB_ZIP = f"https://codeload.github.com/cricketclub/gridspace-stanford-harper-valley/zip/{HVB_SHA}"
SLUE_REPO, SLUE_REV = "asapp/slue-phase-2", "e2989c55a53593a8e39b8f8ebdb47ccaccbe484a"
E_SHA = "c05ab6fd8b4b627d123c922a22a39e993dd37635"
E22_RAW = f"https://raw.githubusercontent.com/revdotcom/speech-datasets/{E_SHA}/earnings22"
E22_MEDIA = f"https://media.githubusercontent.com/media/revdotcom/speech-datasets/{E_SHA}/earnings22/media"
MIN_S, MAX_S = 20.0, 180.0


def _get(url: str, path: str | None = None, tries: int = 4) -> bytes:
    import requests

    last = None
    for k in range(tries):
        try:
            r = requests.get(url, timeout=600, stream=path is not None)
            r.raise_for_status()
            if path is None:
                return r.content
            with open(path, "wb") as f:
                for chunk in r.iter_content(1 << 22):
                    f.write(chunk)
            return b""
        except Exception as e:  # noqa: BLE001
            last = e
            import time

            time.sleep(3 * (k + 1))
    raise RuntimeError(f"download failed {url}: {last}")


def resample(x, sr_in, sr_out=SR):
    import common

    return common.resample(x, sr_in, sr_out)


def ffmpeg_decode(path: str, sr: int = SR) -> np.ndarray:
    out = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def codec_roundtrip(x: np.ndarray, codec: str) -> np.ndarray:
    """Bake a narrow-band codec into a 16 kHz clip with ffmpeg (opus 8-12 kbps / GSM 06.10 / A-law)."""
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.f32")
        x.astype(np.float32).tofile(src)
        if codec.startswith("opus"):
            br = codec.split("_")[1]
            enc = os.path.join(d, "c.ogg")
            args = ["-ar", "16000", "-c:a", "libopus", "-b:a", br, "-application", "voip", enc]
        elif codec == "gsm":
            enc = os.path.join(d, "c.gsm")
            args = ["-ar", "8000", "-c:a", "libgsm", "-f", "gsm", enc]
        else:  # alaw
            enc = os.path.join(d, "c.wav")
            args = ["-ar", "8000", "-c:a", "pcm_alaw", enc]
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "f32le", "-ar", "16000", "-ac", "1", "-i", src] + args,
                       check=True, capture_output=True)
        y = ffmpeg_decode(enc)
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y[: len(x)].astype(np.float32)


def peak_norm(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    m = float(np.max(np.abs(x))) if len(x) else 0.0
    return (x * (peak / m)).astype(np.float32) if m > peak else x.astype(np.float32)


def active_rms(x: np.ndarray, frame: int = 1600) -> float:
    n = len(x) // frame
    if n == 0:
        return float(np.sqrt(np.mean(x ** 2)) + 1e-9)
    e = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1))
    thr = max(1e-4, np.percentile(e, 90) * 0.1)
    act = e[e > thr]
    return float(np.sqrt(np.mean(act ** 2)) if len(act) else np.sqrt(np.mean(e ** 2))) + 1e-9


class BankWriter:
    """Streams clips into banks of < max_bytes float32 each (names <prefix>_0, <prefix>_1, ...)."""

    def __init__(self, out_dir: str, prefix: str, max_bytes: float = 3.4e9):
        self.out_dir, self.prefix, self.max_bytes = out_dir, prefix, max_bytes
        self.cur, self.size, self.k, self.where = {}, 0, 0, {}

    def add(self, key: str, x: np.ndarray) -> str:
        if self.cur and self.size + 4 * len(x) > self.max_bytes:
            self.flush()
        self.cur[key] = np.asarray(x, np.float32)
        self.size += 4 * len(x)
        self.where[key] = f"{self.prefix}_{self.k}"
        return self.where[key]

    def flush(self):
        import common

        if self.cur:
            common.write_audio_bank(self.out_dir, f"{self.prefix}_{self.k}", self.cur)
            self.k, self.cur, self.size = self.k + 1, {}, 0


# Exclusions (evaluation set, eval items, SLUE-HVB test)
def excluded_hvb_calls() -> tuple[set, dict]:
    excl, info = set(), {}
    adi = "/vol/adi/items.jsonl"
    n_adi = 0
    if os.path.exists(adi):
        for line in open(adi):
            d = json.loads(line)
            if d.get("source") == "hvb" or "hvb" in str(d.get("set", "")):
                c = (d.get("meta") or {}).get("call")
                if c:
                    excl.add(c)
                for g in d.get("groups") or []:
                    if g.startswith("hvb_call:"):
                        excl.add(g.split(":", 1)[1])
                n_adi += 1
    info["adi_hvb_items"] = n_adi
    info["adi_hvb_calls"] = len(excl)
    ev = "/vol/data/eval_items.jsonl"
    n_ev = 0
    if os.path.exists(ev):
        for line in open(ev):
            if "hvb" in line.lower() or "harper" in line.lower():
                n_ev += 1
    info["exp0_eval_items_mentioning_hvb"] = n_ev
    # SLUE splits by issue_id
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    split_ids = {}
    for fn in ["hvb/test-00000-of-00001.parquet", "hvb/validation-00000-of-00001.parquet",
               "hvb/train-00000-of-00002.parquet", "hvb/train-00001-of-00002.parquet"]:
        p = hf_hub_download(SLUE_REPO, fn, repo_type="dataset", revision=SLUE_REV, cache_dir="/tmp/hf")
        t = pq.read_table(p, columns=["issue_id"])
        sp = fn.split("/")[1].split("-")[0]
        split_ids.setdefault(sp, set()).update(t.column("issue_id").to_pylist())
    info["slue_calls"] = {k: len(v) for k, v in split_ids.items()}
    test = split_ids.get("test", set())
    info["adi_calls_in_slue_test"] = len(excl & test)
    excl |= test
    info["excluded_total"] = len(excl)
    return excl, info, {k: sorted(v) for k, v in split_ids.items()}


# HVB
def prep_hvb(work: str, out_audio: str) -> dict:
    import soundfile as sf

    Path(work).mkdir(parents=True, exist_ok=True)
    excl, info, split_ids = excluded_hvb_calls()
    zpath = "/tmp/hvb.zip"
    _get(HVB_ZIP, zpath)
    z = zipfile.ZipFile(zpath)
    names = z.namelist()
    root = names[0].split("/")[0]
    tr_files = sorted(n for n in names if "/data/transcript/" in n and n.endswith(".json"))
    info["hvb_calls_in_repo"] = len(tr_files)
    slue_all = set().union(*[set(v) for v in split_ids.values()])
    rng = random.Random(f"{B.SEED}:hvb_codec")
    convs = []
    bw_mix, bw_codec = BankWriter(out_audio, "sb_calltext_hvb_mix"), BankWriter(out_audio, "sb_calltext_hvb_codec")
    stats = Counter()
    for tf in tr_files:
        sid = tf.rsplit("/", 1)[1][:-5]
        if sid in excl:
            stats["excluded"] += 1
            continue
        segs = json.loads(z.read(tf))
        meta = json.loads(z.read(f"{root}/data/metadata/{sid}.json"))
        turns = B.hvb_turns(segs)
        if len(turns) < 3 or not any(t["role"] == "caller" for t in turns) or not any(t["role"] == "agent" for t in turns):
            stats["too_few_turns"] += 1
            continue
        chans = {}
        for role in ("agent", "caller"):
            data, sr = sf.read(io.BytesIO(z.read(f"{root}/data/audio/{role}/{sid}.wav")), dtype="float32", always_2d=True)
            chans[role] = (resample(data.mean(axis=1), sr), sr)
        t0 = max(0, turns[0]["start_ms"] - 600)
        t1 = max(t["end_ms"] for t in turns) + 700
        dur = (t1 - t0) / 1000.0
        if dur < MIN_S or dur > MAX_S:
            stats["dur_out_of_range"] += 1
            continue
        n = int(round(dur * SR))
        y = np.zeros(n, np.float32)
        for role, (x, sr) in chans.items():
            shift = B.channel_shift_ms(segs, role)
            a = int(round((t0 + shift) * SR / 1000))
            seg = np.zeros(n, np.float32)
            lo, hi = max(0, a), min(len(x), a + n)
            if hi > lo:
                seg[lo - a : hi - a] = x[lo:hi]
            seg = seg * (0.1 / active_rms(seg))  # equal speech level on both sides (~ -20 dBFS)
            y += seg
        y = peak_norm(y * 0.9)
        cdc = rng.choice(["opus_8k", "opus_12k", "gsm", "alaw"])
        sid_split = "CAL" if B.is_cal(sid) else "train"
        b_mix = bw_mix.add(f"{sid}_mix", y)
        b_cod = bw_codec.add(f"{sid}_{cdc}", peak_norm(codec_roundtrip(y, cdc)))
        tasks = [t.get("task_type") for t in meta.get("tasks", []) if t.get("task_type")]
        convs.append({"sid": sid, "split": sid_split, "dur_s": round(dur, 2), "transcript": B.hvb_transcript(turns),
                      "n_turns": len(turns), "tasks": tasks, "agent_speaker": meta.get("agent", {}).get("speaker_id"),
                      "caller_speaker": meta.get("caller", {}).get("speaker_id"),
                      "slue_split": next((k for k, v in split_ids.items() if sid in set(v)), "none"),
                      "clips": [f"{sid}_mix", f"{sid}_{cdc}"], "codec": cdc,
                      "banks": {f"{sid}_mix": b_mix, f"{sid}_{cdc}": b_cod}})
        stats["kept"] += 1
    bw_mix.flush()
    bw_codec.flush()
    with open(f"{work}/hvb_convs.jsonl", "w") as f:
        for c in convs:
            f.write(json.dumps(c) + "\n")
    info.update({"stats": dict(stats), "hours_per_variant": round(sum(c["dur_s"] for c in convs) / 3600, 2),
                 "slue_split_of_kept": dict(Counter(c["slue_split"] for c in convs)),
                 "cal_convs": sum(c["split"] == "CAL" for c in convs)})
    Path(f"{work}/hvb_prep.json").write_text(json.dumps(info, indent=1))
    return info


# Earnings-22
def e22_file_ids() -> list[str]:
    import requests

    r = requests.get(f"https://api.github.com/repos/revdotcom/speech-datasets/contents/earnings22/media?ref={E_SHA}",
                     timeout=60)
    if r.ok:
        return sorted(x["name"][:-4] for x in r.json() if x["name"].endswith(".mp3"))
    meta = _get(f"{E22_RAW}/metadata.csv").decode()
    return sorted(line.split(",")[0] for line in meta.splitlines()[1:] if line.strip())


E22_VARIANTS = {  # name -> (window length lo, hi, gap between windows, key tag)
    "e22": (30.0, 100.0, (3.0, 30.0), ""),
    "e22s": (20.0, 45.0, (2.0, 20.0), "s"),  # second, independent pass with shorter windows (different boundaries)
}


def prep_e22_file(fid: str, work: str, variant: str = "e22") -> dict:
    """One earnings call -> windows (audio npy + jsonl) in work/<variant>/."""
    from urllib.parse import quote

    lo, hi, gap, tag = E22_VARIANTS[variant]
    d = Path(work) / variant
    d.mkdir(parents=True, exist_ok=True)
    if (d / f"{fid}.jsonl").exists() and (d / f"{fid}.npy").exists():
        return {"fid": fid, "cached": True}
    nlp = _get(f"{E22_RAW}/transcripts/force_aligned_nlp_references/{quote(fid)}.aligned.nlp").decode("utf-8", "replace")
    toks = B.parse_nlp(nlp)
    sents = B.nlp_sentences(toks)
    mp3 = f"/tmp/{fid}.mp3"
    _get(f"{E22_MEDIA}/{quote(fid)}.mp3", mp3)
    x = ffmpeg_decode(mp3)
    rng = random.Random(f"{B.SEED}:{variant}:{fid}")
    wins = B.make_windows(sents, rng, lo=lo, hi=hi, gap=gap)
    rows, chunks, off = [], [], 0
    split = "CAL" if B.is_cal(fid, frac=0.035, salt="sb_calltext_e22") else "train"
    for w in wins:
        a, b = int(w["start"] * SR), int(w["end"] * SR)
        if b > len(x) or b - a < min(20.0, lo) * SR:
            continue
        clip = peak_norm(x[a:b])
        if np.sqrt(np.mean(clip ** 2)) < 1e-3:
            continue
        key = f"{fid}_{tag}{int(w['start'] * 10):06d}"
        rows.append({"key": key, "fid": fid, "split": split, "start_s": round(w["start"], 2), "dur_s": round((b - a) / SR, 2),
                     "transcript": B.window_transcript(sents, w["i0"], w["i1"]), "off": off, "len": b - a,
                     "n_speakers": len({s["spk"] for s in sents[w["i0"] : w["i1"] + 1]})})
        chunks.append(clip)
        off += b - a
    np.save(d / f"{fid}.npy", np.concatenate(chunks) if chunks else np.zeros(0, np.float32))
    with open(d / f"{fid}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return {"fid": fid, "windows": len(rows), "hours": round(off / SR / 3600, 3), "sents": len(sents),
            "audio_h": round(len(x) / SR / 3600, 3)}


def gather_e22(work: str, out_audio: str, variant: str = "e22") -> dict:
    bw = BankWriter(out_audio, f"sb_calltext_{variant}")
    with open(f"{work}/{variant}_windows.jsonl", "w") as f:
        rows = []
        for jf in sorted(glob.glob(f"{work}/{variant}/*.jsonl")):
            arr = np.load(jf[:-6] + ".npy", mmap_mode="r")
            for line in open(jf):
                r = json.loads(line)
                r["bank"] = bw.add(r["key"], np.array(arr[r["off"] : r["off"] + r["len"]], dtype=np.float32))
                r = {k: v for k, v in r.items() if k not in ("off", "len")}
                f.write(json.dumps(r) + "\n")
                rows.append(r)
        bw.flush()
    info = {"windows": len(rows), "files": len({r["fid"] for r in rows}),
            "hours": round(sum(r["dur_s"] for r in rows) / 3600, 2),
            "cal_windows": sum(r["split"] == "CAL" for r in rows),
            "cal_files": sorted({r["fid"] for r in rows if r["split"] == "CAL"})}
    Path(f"{work}/{variant}_prep.json").write_text(json.dumps(info, indent=1))
    return info


# AMI meetings (CC BY 4.0): Mix-Headset audio + manual word alignments (NXT 1.6.2)
AMI_ANN = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
AMI_WAV = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.Mix-Headset.wav"
AMI_ROLES = {"PM": "Project Manager", "ME": "Marketing Expert", "UI": "User Interface Designer", "ID": "Industrial Designer"}
AMI_MAX_WINDOWS = 14


def ami_meetings(work: str) -> list[dict]:
    """Download the annotation zip to work/ami/ once; list meetings that have word alignments (+ role map)."""
    import re as _re

    d = Path(work) / "ami"
    d.mkdir(parents=True, exist_ok=True)
    zp = d / "ami_public_manual_1.6.2.zip"
    if not zp.exists():
        _get(AMI_ANN, str(zp))
    z = zipfile.ZipFile(zp)
    words = {}
    for n in z.namelist():
        m = _re.match(r"words/([A-Z]{2}\d{4}[a-z]?)\.([A-Z])\.words\.xml$", n)
        if m:
            words.setdefault(m.group(1), []).append(m.group(2))
    roles = {}
    mx = z.read("corpusResources/meetings.xml").decode("latin-1")
    for mm in _re.finditer(r'<meeting [^>]*observation="([^"]+)"[^>]*>(.*?)</meeting>', mx, _re.S):
        r = {}
        for sp in _re.finditer(r"<speaker ([^>]*)/>", mm.group(2)):
            ag = _re.search(r'nxt_agent="([A-Z])"', sp.group(1))
            ro = _re.search(r'role="([A-Z]+)"', sp.group(1))
            if ag and ro and ro.group(1) in AMI_ROLES:
                r[ag.group(1)] = AMI_ROLES[ro.group(1)]
        roles[mm.group(1)] = r
    return [{"meeting": m, "agents": sorted(a), "roles": roles.get(m, {})} for m, a in sorted(words.items())]


def prep_ami_meeting(info: dict, work: str) -> dict:
    import soundfile as sf

    m = info["meeting"]
    d = Path(work) / "ami"
    if (d / f"{m}.jsonl").exists() and (d / f"{m}.npy").exists():
        return {"meeting": m, "cached": True}
    z = zipfile.ZipFile(d / "ami_public_manual_1.6.2.zip")
    toks = []
    for a in info["agents"]:
        toks += B.ami_words(z.read(f"words/{m}.{a}.words.xml").decode("latin-1"), a)
    sents = B.ami_sentences(toks)
    wav = f"/tmp/{m}.wav"
    _get(AMI_WAV.format(m=m), wav)
    x, sr = sf.read(wav, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        x = resample(x, sr)
    rng = random.Random(f"{B.SEED}:ami:{m}")
    wins = B.make_windows(sents, rng, lo=30.0, hi=100.0, gap=(5.0, 40.0))
    if len(wins) > AMI_MAX_WINDOWS:
        wins = sorted(rng.sample(wins, AMI_MAX_WINDOWS), key=lambda w: w["start"])
    rows, chunks, off = [], [], 0
    for w in wins:
        a, b = int(w["start"] * SR), int(w["end"] * SR)
        if b > len(x) or b - a < 20 * SR:
            continue
        clip = peak_norm(x[a:b])
        if np.sqrt(np.mean(clip ** 2)) < 1e-3:
            continue
        key = f"{m}_{int(w['start'] * 10):06d}"
        rows.append({"key": key, "meeting": m, "series": m[:-1] if m[-1].isalpha() else m, "start_s": round(w["start"], 2),
                     "dur_s": round((b - a) / SR, 2), "transcript": B.ami_transcript(sents, w["i0"], w["i1"], info["roles"]),
                     "off": off, "len": b - a, "n_speakers": len({s["spk"] for s in sents[w["i0"] : w["i1"] + 1]})})
        chunks.append(clip)
        off += b - a
    np.save(d / f"{m}.npy", np.concatenate(chunks) if chunks else np.zeros(0, np.float32))
    with open(d / f"{m}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return {"meeting": m, "windows": len(rows), "hours": round(off / SR / 3600, 3)}


def gather_ami(work: str, out_audio: str) -> dict:
    bw = BankWriter(out_audio, "sb_calltext_ami")
    rows = []
    with open(f"{work}/ami_windows.jsonl", "w") as f:
        for jf in sorted(glob.glob(f"{work}/ami/*.jsonl")):
            arr = np.load(jf[:-6] + ".npy", mmap_mode="r")
            for line in open(jf):
                r = json.loads(line)
                r["bank"] = bw.add(r["key"], np.array(arr[r["off"] : r["off"] + r["len"]], dtype=np.float32))
                r = {k: v for k, v in r.items() if k not in ("off", "len")}
                f.write(json.dumps(r) + "\n")
                rows.append(r)
        bw.flush()
    info = {"windows": len(rows), "meetings": len({r["meeting"] for r in rows}),
            "hours": round(sum(r["dur_s"] for r in rows) / 3600, 2)}
    Path(f"{work}/ami_prep.json").write_text(json.dumps(info, indent=1))
    return info
