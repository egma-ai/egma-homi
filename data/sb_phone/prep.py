"""Download and decode each external source once into 16 kHz banks under /vol/stageb/sb_phone/work/raw/ (runs in
Modal). Sources and licences are listed in README.md."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
import re
import subprocess
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import common as C

OUT = Path("/vol/stageb/sb_phone")
RAW = OUT / "work" / "raw"
SR = 16000

APOLLO = ("BrunoHays/apollo-telephony-en", "b9d3a68a9faa550ae1f2069eb1eb7634cdc8966c")
VYSTADIAL = ("BrunoHays/vystadial-telephony-en", "b7454512a3547c829be201d9afceb64af8c28e61")
MINDS = ("PolyAI/minds14", "40ce77cb32a384e4d50a568e1ec39ac804019d33")
MINDS_LOCALES = ["cs-CZ", "de-DE", "es-ES", "fr-FR", "it-IT", "ko-KR", "nl-NL", "pl-PL", "pt-PT", "ru-RU", "zh-CN"]
ROBOCALL_SHA = "5aa6f3bfa8563ce8c1c75ebf8a2271e6ff6b4272"
ROBOCALL_URL = f"https://codeload.github.com/wspr-ncsu/robocall-audio-dataset/tar.gz/{ROBOCALL_SHA}"
MUSAN_URLS = ["https://openslr.elda.org/resources/17/musan.tar.gz", "https://www.openslr.org/resources/17/musan.tar.gz"]
RIRS_URLS = ["https://us.openslr.org/resources/28/rirs_noises.zip", "https://www.openslr.org/resources/28/rirs_noises.zip"]


def h01(s: str) -> float:
    """Deterministic hash of a string -> [0, 1)."""
    return int(hashlib.md5(s.encode()).hexdigest()[:12], 16) / float(16 ** 12)


def hf_file(repo: str, rev: str, fn: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, fn, repo_type="dataset", revision=rev, cache_dir="/tmp/hf")


def decode_bytes(b: bytes) -> tuple[np.ndarray, int]:
    """Any audio bytes -> (mono float32, sr). soundfile first (wav/flac/ogg-opus), ffmpeg as fallback."""
    import soundfile as sf

    try:
        x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
        return x.mean(axis=1).astype(np.float32), int(sr)
    except Exception:
        p = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1"],
                           input=b, capture_output=True, check=True)
        return np.frombuffer(p.stdout, dtype=np.float32).copy(), SR


def to16k(x: np.ndarray, sr: int) -> np.ndarray:
    x = C.resample(x, sr, SR) if sr != SR else x.astype(np.float32)
    return x


def trim_silence(x: np.ndarray, thr_db: float = -40.0, pad: float = 0.15) -> np.ndarray:
    """Trim leading/trailing low-energy frames (20 ms frames, threshold relative to the clip's max frame)."""
    n = len(x)
    f = 320
    if n < 2 * f:
        return x
    k = n // f
    e = 10 * np.log10(np.mean(x[: k * f].reshape(k, f) ** 2, axis=1) + 1e-12)
    on = np.where(e > e.max() + thr_db)[0]
    if len(on) == 0:
        return x
    a = max(0, on[0] * f - int(pad * SR))
    b = min(n, (on[-1] + 1) * f + int(pad * SR))
    return x[a:b]


def save_bank(name: str, clips: dict, meta: dict) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    C.write_audio_bank(RAW, name, clips)
    (RAW / f"{name}_meta.json").write_text(json.dumps(meta))


# Apollo
def prep_apollo(n_train: int = 9000, n_dev: int = 700) -> dict:
    import pyarrow.parquet as pq

    stats = {}
    clips, meta = {}, {}
    for split, files, n_keep in (("train", [f"data/train-0000{i}-of-00004.parquet" for i in range(4)], n_train),
                                 ("dev", [f"data/dev-0000{i}-of-00002.parquet" for i in range(2)], n_dev)):
        rows = []
        for fn in files:
            t = pq.read_table(hf_file(*APOLLO, fn))
            cols = [c for c in t.column_names if c != "audio"]
            d = t.select(cols).to_pylist()
            aud = t.column("audio").to_pylist()
            for r, a in zip(d, aud):
                r["_audio"] = a
                rows.append(r)
        stats[f"{split}_rows"] = len(rows)
        stats[f"{split}_columns"] = cols
        durs = np.array([float(r.get("dur") or 0) for r in rows])
        stats[f"{split}_dur_quantiles"] = np.quantile(durs, [0, .1, .25, .5, .75, .9, .99, 1]).round(2).tolist()
        ok = [r for r in rows if 2.0 <= float(r.get("dur") or 0) <= 30.0 and (r.get("transcription") or "").strip()
              and "<unintelligible>" not in (r.get("transcription") or "")]
        stats[f"{split}_eligible"] = len(ok)
        ok.sort(key=lambda r: h01("apollo:" + r["audio_id"]))
        for r in ok[:n_keep]:
            a = r["_audio"]
            b = a["bytes"] if isinstance(a, dict) else a
            x, sr = decode_bytes(b)
            x = to16k(x, sr)
            if len(x) < 2 * SR:
                continue
            key = f"apollo_{split}_{r['audio_id'].split('_')[-1]}"
            clips[key] = x[: 30 * SR]
            meta[key] = {"audio_id": r["audio_id"], "split": split, "text": r.get("transcription"),
                         "orig_sr": sr, **{k: r.get(k) for k in ("src", "sample_rate", "dur")}}
        stats[f"{split}_examples"] = [{"id": r["audio_id"], "text": r.get("transcription"), "dur": r.get("dur")} for r in ok[:15]]
    save_bank("apollo", clips, meta)
    stats["kept"] = len(clips)
    return stats


# Vystadial
def prep_vystadial(n_sessions: int = 7000) -> dict:
    """Utterances of one caller session (spk + date_time in call_id) are concatenated in order (silence-trimmed,
    0.3-1.0 s gaps) into clips of 4-20 s, so a clip sounds like one caller talking on a VoIP line."""
    import pyarrow.parquet as pq

    stats = {}
    rows = []
    for fn in ["data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet"]:
        t = pq.read_table(hf_file(*VYSTADIAL, fn))
        cols = [c for c in t.column_names if c != "audio"]
        d = t.select(cols).to_pylist()
        aud = t.column("audio").to_pylist()
        for r, a in zip(d, aud):
            r["_audio"] = a
            rows.append(r)
    stats["rows"] = len(rows)
    stats["columns"] = cols
    sess = defaultdict(list)
    for r in rows:
        cid = r["call_id"]  # jurcic_001_120912_124317_0001940_0002325
        parts = cid.split("_")
        skey = "_".join(parts[:4]) if len(parts) >= 6 else cid
        r["_t0"] = int(parts[4]) if len(parts) >= 6 and parts[4].isdigit() else 0
        sess[skey].append(r)
    stats["sessions"] = len(sess)
    stats["speakers"] = len({r["spk"] for r in rows})
    stats["utt_per_session"] = Counter(min(len(v), 10) for v in sess.values()).most_common()
    rng = random.Random(f"{C.SEED}:vystadial")
    skeys = sorted(sess, key=lambda s: h01("vys:" + s))
    clips, meta = {}, {}
    for skey in skeys:
        if len(clips) >= n_sessions:
            break
        utts = sorted(sess[skey], key=lambda r: r["_t0"])
        target = rng.uniform(4.0, 16.0)
        pieces, texts, ids, tot = [], [], [], 0.0
        for r in utts:
            a = r["_audio"]
            x, sr = decode_bytes(a["bytes"] if isinstance(a, dict) else a)
            x = trim_silence(to16k(x, sr))
            if len(x) < 0.3 * SR:
                continue
            if pieces:
                gap = np.zeros(int(rng.uniform(0.3, 1.0) * SR), np.float32)
                pieces.append(gap)
                tot += len(gap) / SR
            pieces.append(x)
            texts.append(r["transcription"])
            ids.append(r["audio_id"])
            tot += len(x) / SR
            if tot >= target or tot >= 20:
                break
        if tot < 2.0:
            continue
        y = np.concatenate(pieces)[: 20 * SR]
        key = f"vys_{skey}"
        clips[key] = y
        meta[key] = {"session": skey, "spk": utts[0]["spk"], "text": " ".join(texts), "audio_ids": ids,
                     "n_utts": len(ids)}
    stats["kept"] = len(clips)
    stats["kept_speakers"] = len({m["spk"] for m in meta.values()})
    stats["dur_quantiles"] = np.quantile([len(v) / SR for v in clips.values()], [0, .1, .5, .9, 1]).round(2).tolist()
    stats["examples"] = [m["text"] for m in list(meta.values())[:15]]
    save_bank("vystadial", clips, meta)
    return stats


# MInDS-14
def prep_minds() -> dict:
    import pyarrow.parquet as pq

    stats = {}
    clips, meta = {}, {}
    for loc in MINDS_LOCALES:
        t = pq.read_table(hf_file(*MINDS, f"{loc}/train-00000-of-00001.parquet"))
        cols = [c for c in t.column_names if c != "audio"]
        d = t.select(cols).to_pylist()
        aud = t.column("audio").to_pylist()
        n = 0
        for r, a in zip(d, aud):
            x, sr = decode_bytes(a["bytes"] if isinstance(a, dict) else a)
            x = trim_silence(to16k(x, sr), thr_db=-45)
            if len(x) < 2 * SR:
                continue
            path = r["path"]
            key = "minds_" + loc + "_" + re.sub(r"[^A-Za-z0-9]+", "_", Path(path).stem)
            if key in clips:
                key += "_" + hashlib.md5(path.encode()).hexdigest()[:6]
            clips[key] = x[: 30 * SR]
            meta[key] = {"path": path, "locale": loc, "text": r.get("transcription"),
                         "english_text": r.get("english_transcription"), "intent": r.get("intent_class")}
            n += 1
        stats[loc] = n
    stats["columns"] = cols
    stats["dur_quantiles"] = np.quantile([len(v) / SR for v in clips.values()], [0, .1, .5, .9, 1]).round(2).tolist()
    save_bank("minds", clips, meta)
    return stats


# FTC robocalls
def _http_stream(urls: list[str]):
    import urllib.request

    last = None
    for u in urls:
        try:
            return urllib.request.urlopen(u, timeout=120), u
        except Exception as e:  # try the next mirror
            last = e
    raise RuntimeError(f"all mirrors failed: {last}")


def prep_robocall() -> dict:
    resp, url = _http_stream([ROBOCALL_URL])
    tf = tarfile.open(fileobj=resp, mode="r|gz")
    wavs, meta_csv, lic = {}, None, None
    for m in tf:
        if not m.isfile():
            continue
        name = m.name.split("/", 1)[1] if "/" in m.name else m.name
        if name == "metadata.csv":
            meta_csv = tf.extractfile(m).read().decode("utf-8")
        elif name in ("LICENSE.md",):
            lic = tf.extractfile(m).read().decode("utf-8")[:300]
        elif name.startswith("audio-wav-16khz/") and name.endswith(".wav"):
            wavs[name] = tf.extractfile(m).read()
    rows = list(csv.DictReader(io.StringIO(meta_csv)))
    clips, meta = {}, {}
    durs = []
    for r in rows:
        fn = r["file_name"]
        if fn not in wavs or fn.endswith("_right.wav"):
            continue
        x, sr = decode_bytes(wavs[fn])
        x = trim_silence(to16k(x, sr), thr_db=-45)
        durs.append(len(x) / SR)
        if len(x) < 2 * SR:
            continue
        key = "robo_" + re.sub(r"[^A-Za-z0-9]+", "_", Path(fn).stem)
        clips[key] = x[: 40 * SR]
        meta[key] = {"file": fn, "language": r["language"], "case": r["case_details"], "dur": len(x) / SR}
    save_bank("robocall", clips, meta)
    return {"url": url, "rows": len(rows), "wavs": len(wavs), "kept": len(clips), "licence_head": lic,
            "cases": len({m["case"] for m in meta.values()}),
            "languages": Counter(m["language"] for m in meta.values()).most_common(),
            "dur_quantiles": np.quantile(durs, [0, .1, .25, .5, .75, .9, 1]).round(2).tolist()}


# MUSAN
def parallel_download(url: str, dest: Path, n_conn: int = 24) -> dict:
    """HTTP range requests in parallel threads (a single connection to the OpenSLR mirrors is ~1-13 MB/s)."""
    import time
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        size = int(r.headers["Content-Length"])
        ranges_ok = "bytes" in (r.headers.get("Accept-Ranges") or "")
    if not ranges_ok:
        raise RuntimeError("server does not accept ranges")
    t0 = time.time()
    with open(dest, "wb") as f:
        f.truncate(size)
    step = 64 << 20
    parts = [(a, min(size, a + step) - 1) for a in range(0, size, step)]

    def get(rg):
        a, b = rg
        for attempt in range(6):
            try:
                rq = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
                with urllib.request.urlopen(rq, timeout=120) as r:
                    data = r.read()
                if len(data) != b - a + 1:
                    raise IOError(f"short read {len(data)} != {b - a + 1}")
                with open(dest, "r+b") as f:
                    f.seek(a)
                    f.write(data)
                return len(data)
            except Exception:
                time.sleep(2 + 3 * attempt)
        raise RuntimeError(f"range {a}-{b} failed")

    with ThreadPoolExecutor(n_conn) as ex:
        got = sum(ex.map(get, parts))
    return {"bytes": got, "size": size, "seconds": round(time.time() - t0, 1)}


def prep_musan(music_s: float = 40.0, speech_s: float = 40.0, noise_s: float = 25.0) -> dict:
    """Download the MUSAN tarball (parallel ranges) and keep every LICENSE/README text and an excerpt of every wav
    (the per-file licence filter is applied at build time, from the recorded licence texts)."""
    dest = Path("/tmp/musan.tar.gz")
    dl, url = None, None
    for u in MUSAN_URLS:
        try:
            dl = parallel_download(u, dest)
            url = u
            break
        except Exception as e:
            dl = {"error": repr(e)}
    print("musan download", url, dl, flush=True)
    tf = tarfile.open(dest, mode="r|gz")
    texts = {}
    banks = {"noise": ({}, {}), "music": ({}, {}), "speech": ({}, {})}
    lim = {"noise": noise_s, "music": music_s, "speech": speech_s}
    for m in tf:
        if not m.isfile():
            continue
        name = m.name  # musan/music/fma/music-fma-0000.wav
        parts = name.split("/")
        if not name.endswith(".wav"):
            if m.size < 2_000_000:
                texts[name] = tf.extractfile(m).read().decode("utf-8", "replace")
            continue
        if len(parts) < 4 or parts[1] not in banks:
            continue
        kind = parts[1]
        x, sr = decode_bytes(tf.extractfile(m).read())
        x = to16k(x, sr)
        L = int(lim[kind] * SR)
        if len(x) > L:
            off = int(h01("musan:" + name) * (len(x) - L))
            x = x[off: off + L]
        key = Path(name).stem
        banks[kind][0][key] = x
        banks[kind][1][key] = {"path": name, "subdir": parts[2], "orig_s": float(m.size) / (2 * sr)}
    stats = {"url": url, "download": dl}
    for kind, (clips, meta) in banks.items():
        save_bank(f"musan_{kind}", clips, meta)
        stats[kind] = {"files": len(clips), "subdirs": Counter(v["subdir"] for v in meta.values()).most_common()}
    (RAW / "musan_texts.json").write_text(json.dumps(texts))
    stats["text_files"] = sorted(texts)
    return stats


# Room impulse responses (OpenSLR 28)
def rir_stats(h: np.ndarray, sr: int = SR) -> dict:
    """Direct-to-reverberant ratio (dB; direct = +-2.5 ms around the peak) and a T20-based RT60 estimate."""
    h = np.asarray(h, dtype=np.float64)
    p = int(np.argmax(np.abs(h)))
    w = int(0.0025 * sr)
    direct = np.sum(h[max(0, p - w): p + w] ** 2)
    rev = np.sum(h[p + w:] ** 2) + 1e-12
    drr = 10 * math.log10(direct / rev + 1e-12)
    edc = np.cumsum((h[p:] ** 2)[::-1])[::-1]
    edc_db = 10 * np.log10(edc / (edc[0] + 1e-12) + 1e-12)
    try:
        i5 = int(np.argmax(edc_db <= -5))
        i25 = int(np.argmax(edc_db <= -25))
        rt60 = 3.0 * (i25 - i5) / sr if i25 > i5 > 0 else float("nan")
    except Exception:
        rt60 = float("nan")
    return {"drr_db": round(drr, 2), "rt60_s": round(rt60, 3) if rt60 == rt60 else None}


def prep_rirs(per_size: int = 900) -> dict:
    import urllib.request

    tmp = Path("/tmp/rirs_noises.zip")
    last = None
    for u in RIRS_URLS:
        try:
            urllib.request.urlretrieve(u, tmp)
            url = u
            break
        except Exception as e:
            last = e
    else:
        raise RuntimeError(f"rirs download failed: {last}")
    zf = zipfile.ZipFile(tmp)
    names = zf.namelist()
    texts = {n: zf.read(n).decode("utf-8", "replace")[:20000] for n in names
             if not n.endswith(".wav") and not n.endswith("/") and zf.getinfo(n).file_size < 3_000_000}
    sims = [n for n in names if "/simulated_rirs/" in n and n.endswith(".wav")]
    by_size = defaultdict(list)
    for n in sims:
        by_size[n.split("/simulated_rirs/")[1].split("/")[0]].append(n)
    clips, meta = {}, {}
    for size, lst in sorted(by_size.items()):
        lst.sort(key=lambda n: h01("rir:" + n))
        for n in lst[:per_size]:
            x, sr = decode_bytes(zf.read(n))
            x = to16k(x, sr)[: SR]  # first 1 s is plenty
            key = "rir_" + size + "_" + Path(n).stem
            clips[key] = x
            room = n.split("/")[-2]
            meta[key] = {"path": n, "size": size, "room": f"{size}/{room}", **rir_stats(x)}
    save_bank("rirs", clips, meta)
    (RAW / "rirs_texts.json").write_text(json.dumps(texts))
    drr = np.array([m["drr_db"] for m in meta.values()])
    rt = np.array([m["rt60_s"] for m in meta.values() if m["rt60_s"] is not None])
    return {"url": url, "sim_total": len(sims), "sizes": {k: len(v) for k, v in by_size.items()}, "kept": len(clips),
            "drr_quantiles": np.quantile(drr, [0, .1, .25, .5, .75, .9, 1]).round(2).tolist(),
            "rt60_quantiles": np.quantile(rt, [0, .1, .25, .5, .75, .9, 1]).round(3).tolist() if len(rt) else None,
            "text_files": sorted(texts)}


def ffmpeg_codecs() -> dict:
    p = subprocess.run(["ffmpeg", "-hide_banner", "-codecs"], capture_output=True, text=True)
    want = ("gsm", "amr", "opus", "mulaw", "alaw", "g722", "g726", "speex")
    return {"version": subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0],
            "lines": [l for l in p.stdout.splitlines() if any(w in l for w in want)]}
