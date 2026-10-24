"""sb_synthetic builder: real pool, synthesis plan, rendering, duration matching, items, validation and summary.
Runs inside Modal containers (see modal_build.py)."""
from __future__ import annotations

import collections
import json
import math
import random
import re
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import common as C
import labels as L

SR = 16000
VOL = Path("/vol")
WORK = VOL / "stageb_work" / L.BUILDER
POOL = WORK / "pool"
OUT = VOL / "stageb" / L.BUILDER
RAW_HVB = VOL / "stageb" / "sb_dialog" / "work" / "raw" / "hvb"   # read-only
RAW_AMI = VOL / "stageb" / "sb_dialog" / "work" / "raw" / "ami"   # read-only
RAW_PHONE = VOL / "stageb" / "sb_phone" / "work" / "raw"          # read-only (apollo, musan, rirs)

LS_SETS = ["dev-clean", "test-clean", "dev-other", "test-other"]
LS_MIRRORS = ["https://openslr.elda.org/resources/12/{}.tar.gz", "https://us.openslr.org/resources/12/{}.tar.gz",
              "https://www.openslr.org/resources/12/{}.tar.gz"]
HVB_SHA = "0bd721e877c4a85d8c13ff837e68661ea6200a98"
SOURCES = {
    "librispeech": {"what": "LibriSpeech original audio (openslr 12: " + ", ".join(LS_SETS) + "), flac 16 kHz",
                    "licence": "CC BY 4.0", "revision": "openslr.org/resources/12/<set>.tar.gz"},
    "hvb": {"what": "HarperValleyBank per-role 8 kHz channels (raw copy /vol/stageb/sb_dialog/work/raw/hvb; SLUE-test and evaluation-set calls excluded)",
            "licence": "CC BY 4.0", "revision": f"github.com/cricketclub/gridspace-stanford-harper-valley@{HVB_SHA}"},
    "ami": {"what": "AMI Mix-Headset single-speaker excerpts (raw copy /vol/stageb/sb_dialog/work/raw/ami, manual word alignments)",
            "licence": "CC BY 4.0", "revision": "edinburghcstr/ami@46f28f2503e2ec48f8867a84eef356c70476beab (as sb_dialog)"},
    "apollo": {"what": "Apollo FSC-P4 mission-control 8 kHz telephony loops (raw bank of sb_phone; train partition -> train, dev -> CAL)",
               "licence": "CC BY 4.0", "revision": "BrunoHays/apollo-telephony-en@b9d3a68a9faa550ae1f2069eb1eb7634cdc8966c"},
    "musan": {"what": "MUSAN noise excerpts (sb_phone raw bank; per-file licence filter of sb_phone: NC/ND/no-licence dropped)",
              "licence": "CC BY 4.0 corpus / per-file CC BY, CC0, public domain", "revision": "openslr 17 musan.tar.gz"},
    "rirs": {"what": "OpenSLR 28 simulated RIRs (sb_phone raw bank)", "licence": "Apache-2.0", "revision": "openslr 28 rirs_noises.zip"},
}
REAL_SHARE = {"librispeech": 0.44, "hvb": 0.19, "ami": 0.17, "apollo": 0.20}
NARROW = {"hvb", "apollo"}
# share of the synthetic half (before failures / duration matching); kokoro <= 25 %
SYN_SHARE = {"kokoro": 0.20, "piper": 0.18, "speecht5": 0.13, "parler": 0.15, "chatterbox": 0.10,
             "vocos": 0.13, "hifigan": 0.11}
TTS_SYSTEMS = ["kokoro", "piper", "speecht5", "parler", "chatterbox"]
VOCODERS = ["vocos", "hifigan"]
N_CLASS = 3600       # target clips per class (train + CAL) before duration matching
N_PAIRS = 2400
OVERGEN = 1.25


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# Helpers
def merge(iv, gap=0.0):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2) + 1e-12))


def parallel_download(url: str, dest: Path, n_conn: int = 16) -> dict:
    """HTTP range requests in parallel threads (copied from sb_phone/prep.py)."""
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        size = int(r.headers["Content-Length"])
    t0 = time.time()
    with open(dest, "wb") as f:
        f.truncate(size)
    step = 32 << 20
    parts = [(a, min(size, a + step) - 1) for a in range(0, size, step)]

    def get(rg):
        a, b = rg
        for attempt in range(6):
            try:
                rq = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
                with urllib.request.urlopen(rq, timeout=120) as r:
                    data = r.read()
                if len(data) != b - a + 1:
                    raise IOError("short read")
                with open(dest, "r+b") as f:
                    f.seek(a)
                    f.write(data)
                return len(data)
            except Exception:  # noqa: BLE001
                time.sleep(2 + 3 * attempt)
        raise RuntimeError(f"range {a}-{b} failed")

    with ThreadPoolExecutor(n_conn) as ex:
        got = sum(ex.map(get, parts))
    return {"bytes": got, "seconds": round(time.time() - t0, 1)}


def adi_ids() -> dict:
    out = {"hvb_calls": set(), "ids": set(), "speakers": set()}
    p = VOL / "adi" / "items.jsonl"
    if not p.exists():
        return out
    for line in open(p):
        it = json.loads(line)
        m = it.get("meta") or {}
        if it.get("source") == "hvb" and m.get("call") is not None:
            out["hvb_calls"].add(str(m["call"]))
        for k in ("call", "recording", "file", "source_id", "meeting", "clip", "utt_id", "id", "path"):
            if m.get(k) is not None:
                out["ids"].add(str(m[k]))
        for k in ("speaker", "speaker_id", "actor"):
            if m.get(k) is not None:
                out["speakers"].add(str(m[k]))
    return out


# Real pool
def _ls_set(name: str, cap_per_spk: int):
    import io

    import soundfile as sf

    tgz = Path(f"/tmp/{name}.tar.gz")
    err = None
    for m in LS_MIRRORS:
        try:
            dl = parallel_download(m.format(name), tgz)
            log("downloaded", name, dl)
            break
        except Exception as e:  # noqa: BLE001
            err = e
    else:
        raise RuntimeError(f"librispeech {name}: {err!r}")
    flacs, trans, gender = {}, {}, {}
    with tarfile.open(tgz, "r:gz") as tf:
        for mem in tf:
            if not mem.isfile():
                continue
            n = mem.name
            if n.endswith(".flac"):
                flacs[n.rsplit("/", 1)[-1][:-5]] = tf.extractfile(mem).read()
            elif n.endswith(".trans.txt"):
                for line in tf.extractfile(mem).read().decode().splitlines():
                    if line.strip():
                        u, t = line.split(" ", 1)
                        trans[u] = t.strip()
            elif n.endswith("SPEAKERS.TXT"):
                for line in tf.extractfile(mem).read().decode().splitlines():
                    if line.startswith(";") or "|" not in line:
                        continue
                    f = [x.strip() for x in line.split("|")]
                    gender[f[0]] = {"F": "female", "M": "male"}.get(f[1])
    tgz.unlink(missing_ok=True)
    rows, clips, per = [], {}, Counter()
    for u in sorted(flacs, key=lambda u: L.h01("ls:" + u)):
        spk = u.split("-")[0]
        if per[spk] >= cap_per_spk or u not in trans:
            continue
        x, sr = sf.read(io.BytesIO(flacs[u]), dtype="float32")
        d = len(x) / sr
        if not (2.0 <= d <= 16.0):
            continue
        per[spk] += 1
        key = f"ls_{u}"
        clips[key] = x.astype(np.float32)
        rows.append({"key": key, "corpus": "librispeech", "speaker": f"librispeech:{spk}", "source_id": f"librispeech:{u}",
                     "text": L.tts_text(trans[u]), "gender": gender.get(spk), "ls_set": name,
                     "split": "CAL" if L.h01("lsspk:" + spk) < 0.08 else "train", "dur": round(d, 3)})
    return rows, clips


def _hvb(adi_calls: set, n_max: int):
    import soundfile as sf

    slue = set(json.loads((RAW_HVB / "slue_ids.json").read_text())["test"])
    conv_spk, role_spk, excluded = {}, {}, Counter()
    for p in sorted((RAW_HVB / "transcript").glob("*.json")):
        sid = p.stem
        if sid in slue or sid in adi_calls:
            excluded["slue_test_or_adi"] += 1
            continue
        md = json.loads((RAW_HVB / "metadata" / f"{sid}.json").read_text())
        role_spk[sid] = {r: f"hvb_spk:{md[r]['speaker_id']}" for r in ("agent", "caller")}
        conv_spk[sid] = set(role_spk[sid].values())
    # speaker-closed CAL: a conversation is CAL if both its speakers are CAL speakers, train if neither, else dropped
    spk_all = sorted({s for v in conv_spk.values() for s in v})
    cal_spk = {s for s in spk_all if L.h01("hvbspk:" + s) < 0.10}
    split = {c: ("CAL" if v <= cal_spk else ("train" if not (v & cal_spk) else None)) for c, v in conv_spk.items()}
    noise = re.compile(r"^(\s*\[[^\]]*\]\s*)*$")
    rows, clips, per = [], {}, Counter()
    sids = sorted((c for c, s in split.items() if s), key=lambda c: L.h01("hvbc:" + c))
    for sid in sids:
        if len(rows) >= n_max:
            break
        segs = json.loads((RAW_HVB / "transcript" / f"{sid}.json").read_text())
        segs = sorted([g for g in segs if g.get("speaker_role") in ("agent", "caller")], key=lambda g: g["start_ms"])
        # turns = consecutive same-role speech segments
        turns = []
        for g in segs:
            text = (g.get("human_transcript") or g.get("transcript") or "").strip()
            if not text or noise.match(text):
                continue
            s = g["start_ms"] / 1000.0
            e = s + g["duration_ms"] / 1000.0
            if turns and turns[-1]["role"] == g["speaker_role"] and s - turns[-1]["e"] < 0.8:
                turns[-1]["e"] = max(turns[-1]["e"], e)
                turns[-1]["text"] += " " + text
            else:
                turns.append({"role": g["speaker_role"], "s": s, "e": e, "text": text})
        audio = {}
        for t in turns:
            d = t["e"] - t["s"]
            spk = role_spk[sid][t["role"]]
            txt = L.tts_text(t["text"])
            if not (1.8 <= d + 0.2 <= 12.0) or txt is None or per[spk] >= 25:
                continue
            if t["role"] not in audio:
                x, sr = sf.read(str(RAW_HVB / "audio" / t["role"] / f"{sid}.wav"), dtype="float32", always_2d=True)
                audio[t["role"]] = C.resample(x.mean(1), sr, SR)
            x = audio[t["role"]][max(0, int((t["s"] - 0.1) * SR)): int((t["e"] + 0.1) * SR)]
            if len(x) < 1.5 * SR or rms(x) < 1e-4:
                continue
            per[spk] += 1
            key = f"hvb_{sid}_{t['role']}_{int(t['s'] * 100)}"
            clips[key] = x.astype(np.float32)
            rows.append({"key": key, "corpus": "hvb", "speaker": spk, "source_id": f"hvb:{sid}", "text": txt,
                         "gender": None, "split": split[sid], "dur": round(len(x) / SR, 3)})
    return rows, clips, {"excluded_convs": dict(excluded), "cal_speakers": len(cal_spk)}


def _ami(adi: dict, n_max: int):
    import soundfile as sf

    ann = json.loads((RAW_AMI / "ann.json").read_text())
    meetings = sorted(m for m in ann if (RAW_AMI / f"{m}.wav").exists())
    cal_m = {m for m in meetings if L.h01("amim:" + m) < 0.08}
    cal_spk = {g for m in cal_m for g in ann[m]}
    split = {}
    for m in meetings:
        if m in cal_m:
            split[m] = "CAL"
        elif set(ann[m]) & cal_spk:
            split[m] = None
        else:
            split[m] = "train"
    rows, clips, per = [], {}, Counter()
    for m in sorted(meetings, key=lambda m: L.h01("amio:" + m)):
        if split[m] is None or m in adi["ids"]:
            continue
        if len(rows) >= n_max:
            break
        try:
            info = sf.info(str(RAW_AMI / f"{m}.wav"))
        except Exception:  # noqa: BLE001
            continue
        D = info.duration
        n = int(D * 100) + 10
        act = {}
        for g, v in ann[m].items():
            a = np.zeros(n, bool)
            for s, e in v["inc"]:
                a[max(0, int(s * 100)): min(n, int(math.ceil(e * 100)))] = True
            act[g] = a
        cands = []
        for g, v in ann[m].items():
            others = np.zeros(n, bool)
            for h, a in act.items():
                if h != g:
                    others |= a
            for a, b in merge([tuple(x) for x in v["sp"]], 0.35):
                if not (1.8 <= b - a <= 10.0):
                    continue
                if others[max(0, int((a - 0.3) * 100)): int((b + 0.3) * 100) + 1].any():
                    continue
                cands.append((g, a, b))
        random.Random(f"ami:{m}").shuffle(cands)
        for g, a, b in cands[:70]:
            if per[g] >= 60:
                continue
            x, sr = sf.read(str(RAW_AMI / f"{m}.wav"), start=int((a - 0.1) * info.samplerate),
                            stop=int((b + 0.1) * info.samplerate), dtype="float32", always_2d=True)
            x = x.mean(1)
            if sr != SR:
                x = C.resample(x, sr, SR)
            if rms(x) < 1e-4:
                continue
            per[g] += 1
            key = f"ami_{m}_{g}_{int(a * 100)}"
            clips[key] = x.astype(np.float32)
            rows.append({"key": key, "corpus": "ami", "speaker": f"ami:{g}", "source_id": f"ami:{m}", "text": None,
                         "gender": "female" if g.startswith("F") else ("male" if g.startswith("M") else None),
                         "split": split[m], "dur": round(len(x) / SR, 3)})
    return rows, clips, {"cal_meetings": len(cal_m)}


def _apollo(n_max: int):
    meta = json.loads((RAW_PHONE / "apollo_meta.json").read_text())
    bank = C.AudioBank(RAW_PHONE, "apollo")
    rows, clips = [], {}
    for k in sorted(meta, key=lambda k: L.h01("ap:" + k)):
        if len(rows) >= n_max:
            break
        m = meta[k]
        s0, n = bank.index[k]
        d = n / SR
        txt = L.tts_text(m.get("text"))
        if not (2.0 <= d <= 16.0) or txt is None:
            continue
        x = bank.get(k)
        key = f"apollo_{k}"
        clips[key] = x
        rows.append({"key": key, "corpus": "apollo", "speaker": f"apollo:{m['audio_id']}", "source_id": f"apollo:{m['audio_id']}",
                     "text": txt, "gender": None, "split": "CAL" if m["split"] == "dev" else "train", "dur": round(d, 3)})
    return rows, clips


def prep_real(smoke: bool) -> dict:
    """Real pool -> /vol/stageb_work/sb_synthetic/pool/real_pool.{npy,json} + real_meta.json."""
    t0 = time.time()
    if (POOL / "real_meta.json").exists():
        return json.loads((POOL / "prep_report.json").read_text())
    adi = adi_ids()
    rows, clips, rep = [], {}, {}
    sets = LS_SETS[:1] if smoke else LS_SETS
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda s: _ls_set(s, 6 if smoke else 30), sets))
    for r, c in res:
        rows += r
        clips.update(c)
    log("librispeech", len(rows))
    r, c, rep["hvb"] = _hvb(adi["hvb_calls"], 150 if smoke else 2600)
    rows += r
    clips.update(c)
    log("hvb", len(r))
    r, c, rep["ami"] = _ami(adi, 150 if smoke else 2600)
    rows += r
    clips.update(c)
    log("ami", len(r))
    r, c = _apollo(150 if smoke else 2000)
    rows += r
    clips.update(c)
    log("apollo", len(r))
    pool = POOL
    C.write_audio_bank(pool, "real_pool", clips)
    (pool / "real_meta.json").write_text(json.dumps(rows))
    rep.update({"n": len(rows), "by": {f"{k[0]}|{k[1]}": v for k, v in sorted(Counter((r["corpus"], r["split"]) for r in rows).items())},
                "hours": round(sum(r["dur"] for r in rows) / 3600, 2), "seconds": round(time.time() - t0)})
    (pool / "prep_report.json").write_text(json.dumps(rep, indent=1))
    return rep


# MUSAN licence filter (as in sb_phone)
_BAD_LIC = re.compile(r"non-?commercial|noncommercial|\bnc\b|no-?deriv|noderiv|\bnd\b|all rights reserved|sampling", re.I)
_GOOD_LIC = re.compile(r"cc by|cc0|attribution|public domain", re.I)
_LIC_LINE = re.compile(r"cc by|cc0|attribution|public domain|licen[cs]e|sampling|non-?commercial|creative commons", re.I)
_MUSAN_ID = re.compile(r"^\s*((?:music|noise|speech)-[a-z-]+-\d{4})\s*$")


def musan_allowed_noise() -> dict:
    """{musan_noise key: licence} for files whose per-file licence is commercially usable (sb_phone's rule) and not
    share-alike."""
    texts = json.loads((RAW_PHONE / "musan_texts.json").read_text())
    meta = json.loads((RAW_PHONE / "musan_noise_meta.json").read_text())
    lic = {}
    for path, txt in texts.items():
        if not path.endswith("/LICENSE"):
            continue
        d = path.rsplit("/", 1)[0]
        blocks = re.split(r"\n\s*={10,}\s*\n?", txt)
        header, files, parsed = None, {}, []
        for b in blocks:
            lines = [l.strip() for l in b.splitlines() if l.strip()]
            ids = [m.group(1) for l in lines for m in [_MUSAN_ID.match(l)] if m]
            li = " / ".join(l for l in lines if _LIC_LINE.search(l) and not _MUSAN_ID.match(l) and not l.startswith("http"))
            parsed.append((ids, li))
            if li and not ids and header is None:
                header = li
        for ids, li in parsed:
            for i in ids:
                files[i] = li or header or ""
        any_ids = any(ids for ids, _ in parsed)
        lic[d] = {"dir": None if any_ids else " ".join(txt.split())[:300], "files": files}

    def ok(s):
        return bool(s) and bool(_GOOD_LIC.search(s)) and not _BAD_LIC.search(s) and not re.search(r"share-?alike|by-sa", s, re.I)

    out = {}
    for k, m in meta.items():
        d = "musan/" + m["path"].split("musan/", 1)[-1].rsplit("/", 1)[0]
        info = lic.get(d)
        if info is None:
            continue
        s = info["dir"] if info["dir"] is not None else info["files"].get(k, "")
        if ok(s):
            out[k] = s[:120]
    return out


# Plan
def _u(rng, a, b):
    return round(rng.uniform(a, b), 3)


def make_plan(rows: list[dict], smoke: bool) -> dict:
    """Deterministic plan: real-class selection, TTS jobs (text twins), vocoder jobs. Same output every call."""
    rng = random.Random(f"{C.SEED}:{L.BUILDER}:plan")
    scale = 0.04 if smoke else 1.0
    by = defaultdict(list)
    for r in sorted(rows, key=lambda r: L.h01("sel:" + r["key"])):
        by[(r["corpus"], r["split"])].append(r)
    n_tot = {c: len(by[(c, "train")]) + len(by[(c, "CAL")]) for c in REAL_SHARE}
    real_sel = []
    for c, sh in REAL_SHARE.items():
        q = int(round(N_CLASS * sh * 1.15 * scale))
        for sp in ("train", "CAL"):
            avail = by[(c, sp)]
            n = int(round(q * len(avail) / max(1, n_tot[c])))
            real_sel += avail[: min(n, int(0.8 * len(avail)))]  # keep >= 20 % unselected for vocoder sources / refs
    sel_keys = {r["key"] for r in real_sel}
    rest = [r for r in rows if r["key"] not in sel_keys]
    twins = {sp: defaultdict(list) for sp in ("train", "CAL")}
    for r in real_sel:
        twins[r["split"]][r["corpus"]].append(r)
    hvb_text = {sp: [r for r in rows if r["corpus"] == "hvb" and r["split"] == sp and r["text"]] for sp in ("train", "CAL")}
    refs = {sp: sorted([r for r in rows if r["corpus"] == "librispeech" and r["split"] == sp and 5.0 <= r["dur"] <= 12.0],
                       key=lambda r: L.h01("ref:" + r["key"])) for sp in ("train", "CAL")}
    refs = {"train": refs["train"][:80], "CAL": refs["CAL"][:8]}
    cal_frac = sum(r["split"] == "CAL" for r in real_sel) / max(1, len(real_sel))
    corpora = list(REAL_SHARE)
    weights = [REAL_SHARE[c] for c in corpora]
    jobs = defaultdict(list)
    import synth as S
    for system, share in SYN_SHARE.items():
        n = int(round(N_CLASS * share * OVERGEN * scale))
        for i in range(n):
            sp = "CAL" if (i % 100) < round(100 * cal_frac) else "train"
            corpus = rng.choices(corpora, weights)[0]
            key = f"{system}_{sp}_{i:05d}"
            if system in VOCODERS:
                pool = [r for r in (twins[sp][corpus] if rng.random() < 0.5 else rest) if r["corpus"] == corpus and r["split"] == sp]
                if not pool:
                    continue
                src = rng.choice(pool)
                jobs[system].append({"key": key, "split": sp, "src_key": src["key"], "corpus": corpus,
                                     "speaker": src["speaker"], "source_id": src["source_id"], "voice": None})
                continue
            if not twins[sp][corpus]:
                continue
            tw = rng.choice(twins[sp][corpus])
            text = tw["text"] if tw["text"] else rng.choice(hvb_text[sp])["text"]
            if len(text) > 260:
                cut = text[:260].rsplit(" ", 1)[0]
                text = cut.rstrip(",;:") + "."
            j = {"key": key, "split": sp, "text": text, "twin": tw["key"], "corpus": corpus, "twin_narrow": corpus in NARROW}
            cal = sp == "CAL"
            if system == "kokoro":
                j.update(voice=rng.choice(S.KOKORO_CAL if cal else S.KOKORO_TRAIN), params={"speed": _u(rng, 0.78, 1.08)})
            elif system == "piper":
                if rng.random() < 0.6:
                    v, spk = "en_US-libritts-high", rng.choice(S.PIPER_LIBRITTS_CAL_SPK if cal else S.PIPER_LIBRITTS_TRAIN_SPK)
                else:
                    v, spk = rng.choice(S.PIPER_SINGLE_CAL if cal else S.PIPER_SINGLE_TRAIN), None
                j.update(voice=v if spk is None else f"{v}#{spk}",
                         params={"voice_file": v, "speaker_id": spk, "length_scale": _u(rng, 0.92, 1.28),
                                 "noise_scale": _u(rng, 0.45, 0.8), "noise_w": _u(rng, 0.6, 1.0)})
            elif system == "speecht5":
                pool = S.SPEECHT5_CAL_SPK if cal else S.SPEECHT5_TRAIN_SPK
                if cal or rng.random() < 0.6:
                    mix = [[rng.choice(pool), 1.0]]
                else:
                    a, b = rng.sample(pool, 2)
                    w = _u(rng, 0.3, 0.7)
                    mix = [[a, w], [b, round(1 - w, 3)]]
                j.update(voice="+".join(m[0] for m in mix), params={"seed": rng.randrange(1 << 30), "mix": mix})
            elif system == "parler":
                name = rng.choice(S.PARLER_NAMES_CAL if cal else S.PARLER_NAMES_TRAIN)
                desc = (f"{name}'s voice is {rng.choice(['calm', 'expressive', 'monotone', 'slightly expressive', 'warm', 'animated'])} "
                        f"and {rng.choice(['fairly fast', 'moderate in pace', 'slightly slow', 'quick', 'measured'])}, with a "
                        f"{rng.choice(['low', 'moderate', 'slightly high', 'deep', 'high'])} pitch. The recording is "
                        f"{rng.choice(['very clear', 'clear', 'of good quality', 'slightly noisy', 'clean'])} and "
                        f"{rng.choice(['close-sounding', 'very close-sounding', 'slightly distant-sounding', 'close-up'])}.")
                j.update(voice=name, params={"seed": rng.randrange(1 << 30), "description": desc})
            elif system == "chatterbox":
                ref = rng.choice(refs[sp])
                j.update(voice=ref["speaker"], params={"seed": rng.randrange(1 << 30), "ref_key": ref["key"],
                                                       "exaggeration": _u(rng, 0.3, 0.8), "cfg_weight": _u(rng, 0.3, 0.7),
                                                       "temperature": _u(rng, 0.7, 0.9)})
            jobs[system].append(j)
    for js in jobs.values():  # piper jobs: the runner wants the voice file name
        for j in js:
            if "params" in j and "voice_file" in j["params"]:
                j["runner_voice"] = j["params"]["voice_file"]
    return {"real_sel": [r["key"] for r in real_sel], "jobs": dict(jobs), "scale": scale}


def shard(jobs, n):
    return [jobs[i::n] for i in range(n) if jobs[i::n]]


# Synthesis orchestration
N_SHARDS = {"kokoro": 20, "piper": 10, "speecht5": 24, "vocos": 6, "hifigan": 6, "parler": 3, "chatterbox": 4}


def synth_dir(smoke: bool) -> Path:
    return WORK / ("smoke" if smoke else "full") / "synth"


def shard_args(plan, system, smoke):
    n = N_SHARDS[system] if not smoke else (1 if system in ("parler", "chatterbox") else 2)
    return [(i, js, str(synth_dir(smoke))) for i, js in enumerate(shard(plan["jobs"].get(system, []), n))]


def gpu_chain(smoke: bool, fns: dict, chatterbox_budget_s: float) -> list:
    """Parler then Chatterbox, one GPU container at a time (max_containers=1 on both functions)."""
    plan = make_plan(json.loads((POOL / "real_meta.json").read_text()), smoke)
    stats = []
    for system in ("parler", "chatterbox"):
        args = shard_args(plan, system, smoke)
        t0 = time.time()
        try:
            if system == "chatterbox":
                per = chatterbox_budget_s / max(1, len(args))
                res = list(fns[system].map(args, kwargs={"time_budget_s": per}, return_exceptions=True))
            else:
                res = list(fns[system].map(args, return_exceptions=True))
        except Exception as e:  # noqa: BLE001
            res = [{"system": system, "error": repr(e)[:500]}]
        for r in res:
            r = r if isinstance(r, dict) else {"system": system, "error": repr(r)[:500]}
            log("gpu", {k: v for k, v in r.items() if k != "fails"}, "fails", len(r.get("fails", [])))
            stats.append({k: v for k, v in r.items() if k != "fails"} | {"n_fails": len(r.get("fails", [])),
                                                                          "fail_examples": r.get("fails", [])[:5]})
        log(system, "done in", round(time.time() - t0), "s")
    d = synth_dir(smoke)
    d.mkdir(parents=True, exist_ok=True)
    (d / "gpu_done.json").write_text(json.dumps(stats))
    return stats


def cpu_synth(plan, smoke, fns) -> list:
    from concurrent.futures import ThreadPoolExecutor

    def one(system):
        args = shard_args(plan, system, smoke)
        if system in ("speecht5", "vocos", "hifigan"):
            res = list(fns["torchcpu"].starmap([(system, a) for a in args], return_exceptions=True))
        else:
            res = list(fns[system].map(args, return_exceptions=True))
        out = []
        for r in res:
            r = r if isinstance(r, dict) else {"system": system, "error": repr(r)[:500]}
            out.append({k: v for k, v in r.items() if k != "fails"} | {"n_fails": len(r.get("fails", [])),
                                                                        "fail_examples": r.get("fails", [])[:5]})
        log("cpu synth", system, "done", sum(o.get("ok", 0) for o in out), "ok")
        return out

    with ThreadPoolExecutor(5) as ex:
        res = list(ex.map(one, ["kokoro", "piper", "speecht5", "vocos", "hifigan"]))
    return [r for rs in res for r in rs]


def load_synth(plan, smoke) -> dict:
    out = {}
    d = synth_dir(smoke)
    for system in SYN_SHARE:
        for p in sorted((d / system).glob("shard_*.npz")) if (d / system).exists() else []:
            if p.name.endswith(".tmp.npz"):
                continue
            with np.load(p) as z:
                for k in z.files:
                    out[k] = z[k]
    return out


# Rendering
R = {}  # banks for forked workers


def load_render_banks():
    R["pool"] = C.AudioBank(POOL, "real_pool")
    R["noise"] = C.AudioBank(RAW_PHONE, "musan_noise")
    allowed = musan_allowed_noise()
    keys = sorted(k for k, (s, n) in R["noise"].index.items() if n >= 3 * SR and k in allowed)
    R["noise_keys"] = {"train": [k for k in keys if L.h01("bg:" + k) >= 0.15], "CAL": [k for k in keys if L.h01("bg:" + k) < 0.15]}
    R["noise_lic"] = allowed
    R["rir"] = C.AudioBank(RAW_PHONE, "rirs")
    rm = json.loads((RAW_PHONE / "rirs_meta.json").read_text())
    rk = sorted(rm)
    R["rir_keys"] = {"train": [k for k in rk if L.h01("room:" + str(rm[k]["room"])) >= 0.15],
                     "CAL": [k for k in rk if L.h01("room:" + str(rm[k]["room"])) < 0.15]}
    R["rir_meta"] = rm


def trim(x, rng):
    import fx
    e = fx.frame_db(x)
    act = np.where(e >= e.max() - 40.0)[0]
    if len(act) == 0:
        return x
    a = max(0, act[0] * fx.FRAME - int(rng.uniform(0.05, 0.35) * SR))
    b = min(len(x), (act[-1] + 1) * fx.FRAME + int(rng.uniform(0.05, 0.35) * SR))
    y = x[a:b]
    if len(y) > 20 * SR:
        e2 = fx.frame_db(y[: 20 * SR])
        lo = int(17 * SR / fx.FRAME)
        c = lo + int(np.argmin(e2[lo:])) if len(e2) > lo else len(e2)
        y = y[: c * fx.FRAME]
    return y


def colored(n, rng, kind):
    w = np.random.default_rng(rng.randrange(1 << 30)).standard_normal(n)
    if kind == "white":
        return w.astype(np.float32)
    f = np.fft.rfft(w)
    k = np.arange(len(f)) + 1.0
    f = f / (np.sqrt(k) if kind == "pink" else k)
    y = np.fft.irfft(f, n)
    return (y / (np.std(y) + 1e-9)).astype(np.float32)


def render_one(spec: dict):
    """spec: key, split, wide, force_nb, src ('pool', key) or array. Same augmentation distribution for both classes."""
    import fx
    try:
        rng = random.Random(f"{C.SEED}:fx:{spec['key']}")
        x = spec["x"] if spec.get("x") is not None else R["pool"].get(spec["pool_key"])
        x = np.asarray(x, np.float32)
        x = x / (np.max(np.abs(x)) + 1e-9) * 0.5
        x = trim(x, rng)
        rec = {}
        if spec["wide"] and rng.random() < 0.20:
            k = rng.choice(R["rir_keys"][spec["split"]])
            x = fx.apply_rir(x, R["rir"].get(k))
            rec["rir"] = {"id": k, "rt60_s": R["rir_meta"][k].get("rt60_s"), "drr_db": R["rir_meta"][k].get("drr_db")}
        if rng.random() < 0.5:
            kind = rng.choice(["white", "pink", "brown"])
            snr = rng.uniform(35, 55)
            n = colored(len(x), rng, kind)
            p = fx.active_power(x)
            x = (x + n * math.sqrt(p / (10 ** (snr / 10)))).astype(np.float32)
            rec["floor"] = {"kind": kind, "snr_db": round(snr, 1)}
        if rng.random() < 0.25:
            k = rng.choice(R["noise_keys"][spec["split"]])
            snr = rng.uniform(10, 30)
            x = fx.mix_at_snr(x, R["noise"].get(k), snr, rng)
            rec["musan"] = {"id": k, "snr_db": round(snr, 1)}
        if spec["force_nb"]:
            x = fx.bandlimit(x, 250.0, 3500.0)
            codec = rng.choice(["mulaw", "alaw", "gsm", "opus"])
            x = fx.ffmpeg_codec(x / (np.max(np.abs(x)) + 1e-9) * 0.7, codec, 10 if codec == "opus" else None)
            rec["nb_match"] = codec
        if rng.random() < 0.25:
            codec = rng.choice(["mulaw", "alaw", "gsm", "opus", "g726", "speex"])
            br = {"opus": rng.choice([6, 8, 10, 12]), "g726": rng.choice([24, 32])}.get(codec)
            x = fx.ffmpeg_codec(x / (np.max(np.abs(x)) + 1e-9) * 0.7, codec, br)
            rec["phone_fx"] = {"codec": codec, "bitrate_k": br}
        t_db = rng.uniform(-32.0, -18.0)
        x = x * (10 ** (t_db / 20) / rms(x))
        pk = float(np.max(np.abs(x)))
        if pk > 0.99:
            x = x * (0.99 / pk)
        x = x.astype(np.float32)
        rec["target_rms_dbfs"] = round(t_db, 2)
        d = len(x) / SR
        if not (1.0 <= d <= 20.0) or not np.isfinite(x).all():
            return spec["key"], None, {"error": f"dur {d:.2f}"}
        return spec["key"], x, rec
    except Exception as e:  # noqa: BLE001
        return spec["key"], None, {"error": repr(e)[:200]}


# Pipeline
LIC_REAL = {c: f"{SOURCES[c]['licence']} — {SOURCES[c]['revision']}" for c in REAL_SHARE}


def generator_ref(system):
    import synth as S
    r = S.REPOS[system]
    return {"repo": r["weights"], "revision": r["revision"], "weights_licence": r["weights_licence"], "code": r["code"]}


def pipeline(smoke: bool, fns: dict, volume, gpu_wait_s: float = 2700.0) -> dict:
    t0 = time.time()
    rows = json.loads((POOL / "real_meta.json").read_text())
    by_key = {r["key"]: r for r in rows}
    plan = make_plan(rows, smoke)
    log("plan", {k: len(v) for k, v in plan["jobs"].items()}, "real_sel", len(plan["real_sel"]))
    (synth_dir(smoke)).mkdir(parents=True, exist_ok=True)

    # 1. CPU synthesis (GPU chain runs separately: modal_build.py::gpu)
    cpu_stats = cpu_synth(plan, smoke, fns)
    t_cpu = time.time() - t0
    # 2. wait for the GPU chain
    gdone = synth_dir(smoke) / "gpu_done.json"
    if smoke:
        gpu_wait_s = 0
    while not gdone.exists() and time.time() - t0 < gpu_wait_s:
        time.sleep(20)
        volume.reload()
    gpu_stats = json.loads(gdone.read_text()) if gdone.exists() else [{"error": "gpu chain not finished in time"}]
    volume.reload()
    syn = load_synth(plan, smoke)
    log("synth loaded", len(syn), "clips", round(time.time() - t0), "s")

    # 3. render specs (same augmentation distribution for both classes)
    specs, info = [], {}
    for k in plan["real_sel"]:
        r = by_key[k]
        specs.append({"key": f"real:{k}", "split": r["split"], "wide": r["corpus"] not in NARROW, "force_nb": False, "pool_key": k})
        info[f"real:{k}"] = {"label": "real", "kind": "real", "system": r["corpus"], "corpus": r["corpus"], "split": r["split"],
                             "speaker": r["speaker"], "source_id": r["source_id"], "pool_key": k, "text": r["text"],
                             "voice": None, "licence": LIC_REAL[r["corpus"]], "gender": r.get("gender")}
    for system, js in plan["jobs"].items():
        for j in js:
            if j["key"] not in syn:
                continue
            k = f"syn:{j['key']}"
            if system in VOCODERS:
                src = by_key[j["src_key"]]
                specs.append({"key": k, "split": j["split"], "wide": src["corpus"] not in NARROW, "force_nb": False,
                              "x": syn[j["key"]]})
                info[k] = {"label": "synthetic", "kind": "vocoded", "system": system, "corpus": src["corpus"], "split": j["split"],
                           "speaker": src["speaker"], "source_id": src["source_id"], "pool_key": j["src_key"], "text": src["text"],
                           "voice": None, "generator": generator_ref(system),
                           "licence": f"copy-synthesis of {LIC_REAL[src['corpus']]} through {system} ({generator_ref(system)['weights_licence']})"}
            else:
                specs.append({"key": k, "split": j["split"], "wide": not j["twin_narrow"], "force_nb": j["twin_narrow"],
                              "x": syn[j["key"]]})
                info[k] = {"label": "synthetic", "kind": "tts", "system": system, "corpus": j["corpus"], "split": j["split"],
                           "speaker": f"{system}:{j['voice']}", "source_id": f"{system}:{j['key']}", "twin": j["twin"],
                           "text": j["text"], "voice": j["voice"], "params": j.get("params"), "generator": generator_ref(system),
                           "licence": f"{system} output ({generator_ref(system)['weights_licence']}); text = transcript of "
                                      f"real clip {j['twin']} ({LIC_REAL[j['corpus']]}) or an HVB transcript for AMI twins"}
    del syn
    load_render_banks()
    import multiprocessing as mp
    rendered, recs, errs = {}, {}, []
    with mp.get_context("fork").Pool(8) as pool:
        for k, x, rec in pool.imap_unordered(render_one, specs, chunksize=16):
            if x is None:
                errs.append([k, rec.get("error")])
            else:
                rendered[k] = x
                recs[k] = rec
    del specs
    log("rendered", len(rendered), "errors", len(errs), round(time.time() - t0), "s")

    # 4. duration matching per split (histogram, 0.5 s bins)
    rng = random.Random(f"{C.SEED}:{L.BUILDER}:match")
    kept = {}
    for sp in ("train", "CAL"):
        re_ = sorted((k, len(x) / SR) for k, x in rendered.items() if info[k]["split"] == sp and info[k]["label"] == "real")
        sy = sorted((k, len(x) / SR) for k, x in rendered.items() if info[k]["split"] == sp and info[k]["label"] == "synthetic")
        kr, ks = L.duration_match(re_, sy, rng)
        kept[sp] = {"real": sorted(kr), "synthetic": sorted(ks)}
    for k in [k for k in rendered if k not in set(kept["train"]["real"] + kept["train"]["synthetic"] + kept["CAL"]["real"] + kept["CAL"]["synthetic"])]:
        del rendered[k]
    log("kept", {sp: {c: len(v) for c, v in d.items()} for sp, d in kept.items()})

    # 5. pairs (one real + one synthetic, 0.8 s gap; halves <= 12 s; prefer same-sentence twins)
    pairs = []
    prng = random.Random(f"{C.SEED}:{L.BUILDER}:pairs")
    n_pairs_total = int(round(N_PAIRS * plan["scale"]))
    n_train_clips = len(kept["train"]["real"])
    for sp in ("train", "CAL"):
        n_p = int(round(n_pairs_total * len(kept[sp]["real"]) / max(1, n_train_clips + len(kept["CAL"]["real"]))))
        reals = [k for k in kept[sp]["real"] if len(rendered[k]) <= 12 * SR]
        syns = [k for k in kept[sp]["synthetic"] if len(rendered[k]) <= 12 * SR]
        prng.shuffle(reals)
        prng.shuffle(syns)
        used = set()
        twin_of = defaultdict(list)
        for s in syns:
            tw = info[s].get("twin") or info[s].get("pool_key")
            twin_of[f"real:{tw}"].append(s)
        for r in reals:
            if len(pairs) >= n_p + sum(1 for p in pairs if p["split"] != sp) or len([p for p in pairs if p["split"] == sp]) >= n_p:
                break
            cand = [s for s in twin_of.get(r, []) if s not in used]
            if cand and prng.random() < 0.6:
                s = cand[0]
                same = True
            else:
                s = next((s for s in syns if s not in used), None)
                same = False
                if s is None:
                    break
            used.add(s)
            i = len([p for p in pairs if p["split"] == sp])
            synth_pos = i % 2
            ask = "synth" if (i // 2) % 2 == 0 else "real"
            pairs.append({"key": f"pair:{sp}:{i:05d}", "split": sp, "real": r, "synth": s, "synth_pos": synth_pos,
                          "ask": ask, "same_text": same or (info[s].get("text") == info[r].get("text"))})
    gap = np.zeros(int(0.8 * SR), np.float32)
    for p in pairs:
        a, b = (rendered[p["synth"]], rendered[p["real"]]) if p["synth_pos"] == 0 else (rendered[p["real"]], rendered[p["synth"]])
        rendered[p["key"]] = np.concatenate([a, gap, b]).astype(np.float32)
    log("pairs", Counter(p["split"] for p in pairs))

    # 6. items
    items = build_items(kept, pairs, info, recs, rendered)
    # CAL subsample to ~3 % of all items (balanced per qname / answer)
    tr = [it for it in items if it["set"] == "train"]
    cal_all = [it for it in items if it["set"] == "CAL"]
    n_cal = int(round(0.031 * len(tr) / 0.969))
    crng = random.Random(f"{C.SEED}:{L.BUILDER}:cal")
    groups = defaultdict(list)
    for it in cal_all:
        groups[(it["meta"]["qname"], it["options"][it["gold"]] if it["meta"]["qtype"] == "noul" else it["meta"]["label"])].append(it)
    cal = []
    per = max(1, n_cal // max(1, len(groups)))
    for g in sorted(groups):
        crng.shuffle(groups[g])
        cal += groups[g][:per]
    items = tr + cal
    used_keys = {it["audio"]["key"] for it in items}

    # 7. write banks + items (final dir, once), summary.json last
    out = OUT if not smoke else WORK / "smoke" / "out"
    for f in ("summary.json", "items_train.jsonl", "items_cal.jsonl"):
        (out / f).unlink(missing_ok=True)
    audio_dir = out / "audio"
    bank_of, bank_names = {}, []
    for sp in ("train", "CAL"):
        ks = sorted(k for k in used_keys if (info.get(k) or {}).get("split", k.split(":")[1] if k.startswith("pair:") else None) == sp)
        prefix = f"{L.BUILDER}_train" if sp == "train" else f"{L.BUILDER}_cal"
        for b in range(0, len(ks), 2500):
            name = f"{prefix}_{b // 2500:02d}"
            clip_d = {k.replace(":", "_"): rendered[k] for k in ks[b: b + 2500]}
            C.write_audio_bank(audio_dir, name, clip_d)
            bank_names.append(name)
            for k in ks[b: b + 2500]:
                bank_of[k] = name
            log("bank", name, len(clip_d))
    for it in items:
        k = it["audio"]["key"]
        it["audio"] = {"bank": bank_of[k], "key": k.replace(":", "_")}
    tr = [it for it in items if it["set"] == "train"]
    cal = [it for it in items if it["set"] == "CAL"]
    C.write_jsonl(out / "items_train.jsonl", tr)
    C.write_jsonl(out / "items_cal.jsonl", cal)
    volume.commit()
    summary = finish(out, smoke, tr, cal, bank_names, rendered, info, recs, kept, pairs, plan, cpu_stats, gpu_stats, errs,
                     t0, t_cpu, by_key)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    volume.commit()
    return summary


def _meta_single(k, info, recs, rendered):
    i = info[k]
    m = {kk: i.get(kk) for kk in ("label", "kind", "system", "corpus", "speaker", "source_id", "voice", "text", "twin",
                                  "pool_key", "gender", "licence", "generator", "params") if i.get(kk) is not None}
    m["clip"] = k
    m["fx"] = recs.get(k, {})
    x = rendered[k]
    m["dur_s"] = round(len(x) / SR, 3)
    m["rms_dbfs"] = round(20 * math.log10(rms(x)), 2)
    return m


def build_items(kept, pairs, info, recs, rendered) -> list[dict]:
    items = []
    yn = L.NoShuffle()
    for sp in ("train", "CAL"):
        for cls in ("real", "synthetic"):
            ks = sorted(kept[sp][cls], key=lambda k: L.h01("fam:" + k))
            for i, k in enumerate(ks):
                rng = random.Random(f"{C.SEED}:{L.BUILDER}:q:{k}")
                is_s = cls == "synthetic"
                base = _meta_single(k, info, recs, rendered)
                fam = "is_real" if i % 2 == 0 else "is_synth"
                q, ans = L.noul_view(is_s, fam, rng)
                opts = ["yes", "no"]
                g = opts.index(ans)
                items.append(C.make_item(item_id=f"{L.BUILDER}:{k}:noul", set_name=sp, source=L.SRC_NOUL, question=q,
                                         options=opts, target=[1.0 if j == g else 0.0 for j in range(2)], gold=g, rng=rng,
                                         audio={"bank": None, "key": k}, channel="clean",
                                         meta=base | {"qtype": "noul", "qname": fam, "view": "noul"}))
                if L.h01("choice:" + k) < 0.85:
                    q, opts, g = L.choice_view(is_s, rng)
                    items.append(C.make_item(item_id=f"{L.BUILDER}:{k}:choice", set_name=sp, source=L.SRC_CHOICE, question=q,
                                             options=opts, target=[1.0 if j == g else 0.0 for j in range(2)], gold=g, rng=rng,
                                             audio={"bank": None, "key": k}, channel="clean",
                                             meta=base | {"qtype": "choice", "qname": "real_or_synthetic", "view": "choice"}))
    for p in pairs:
        rng = random.Random(f"{C.SEED}:{L.BUILDER}:q:{p['key']}")
        q, opts, g = L.pair_view(p["synth_pos"], p["ask"], rng)
        si, ri = info[p["synth"]], info[p["real"]]
        meta = {"qtype": "choice", "qname": f"pair_{p['ask']}", "view": "pair", "label": f"synthetic_{'first' if p['synth_pos'] == 0 else 'second'}",
                "synth_pos": p["synth_pos"], "ask": p["ask"], "same_text": p["same_text"],
                "synth": _meta_single(p["synth"], info, recs, rendered), "real": _meta_single(p["real"], info, recs, rendered),
                "system": si["system"], "kind": si["kind"], "corpus": ri["corpus"],
                "speaker": [ri["speaker"], si["speaker"]], "source_id": [ri["source_id"], si["source_id"]],
                "licence": f"real: {ri['licence']} | synthetic: {si['licence']}", "gap_s": 0.8,
                "dur_s": round(len(rendered[p["key"]]) / SR, 3)}
        items.append(C.make_item(item_id=f"{L.BUILDER}:{p['key']}", set_name=p["split"], source=L.SRC_PAIR, question=q,
                                 options=opts, target=[1.0 if j == g else 0.0 for j in range(2)], gold=g, rng=yn,
                                 audio={"bank": None, "key": p["key"]}, channel="clean", meta=meta))
    return items


def finish(out, smoke, tr, cal, bank_names, rendered, info, recs, kept, pairs, plan, cpu_stats, gpu_stats, errs, t0, t_cpu,
           by_key) -> dict:
    audio_dir = out / "audio"
    idx = {b: json.loads((audio_dir / f"{b}.json").read_text()) for b in bank_names}
    items = C.read_jsonl(out / "items_train.jsonl") + C.read_jsonl(out / "items_cal.jsonl")
    keys = {"id", "set", "source", "question", "options", "target", "gold", "audio", "channel", "state_text", "meta"}
    per_clip = Counter()
    for it in items:
        assert keys <= set(it), it["id"]
        assert 2 <= len(it["options"]) <= 20 and len(it["options"]) == len(it["target"])
        assert abs(sum(it["target"]) - 1) < 1e-6
        assert it["gold"] == int(np.argmax(it["target"])) and max(it["target"]) == 1.0
        assert it["meta"]["qtype"] in ("noul", "choice", "score")
        assert it["source"].startswith(L.BUILDER + "_")
        b, k = it["audio"]["bank"], it["audio"]["key"]
        assert k in idx[b], (b, k)
        d = idx[b][k][1] / SR
        assert 1.0 <= d <= 30.0, (k, d)
        per_clip[(b, k)] += 1
    assert len({it["id"] for it in items}) == len(items)
    assert max(per_clip.values()) <= 4
    for b in bank_names:
        arr = np.load(audio_dir / f"{b}.npy", mmap_mode="r")
        assert max(s + n for s, n in idx[b].values()) == arr.shape[0]
        ks = list(idx[b])[:: max(1, len(idx[b]) // 40)]
        for k in ks:
            s, n = idx[b][k]
            x = np.asarray(arr[s: s + n])
            assert np.isfinite(x).all() and np.abs(x).max() <= 0.99 + 1e-5
    noul = L.noul_balance(items)
    for k, v in noul.items():
        assert abs(v["yes"] - v["no"]) <= max(2, 0.02 * (v["yes"] + v["no"])), (k, v)
    choice_bal = Counter((it["set"], it["meta"]["qname"], it["options"][it["gold"]] if it["meta"]["view"] == "pair"
                          else it["meta"]["label"]) for it in items if it["meta"]["qtype"] == "choice")
    pair_bal = Counter((it["set"], it["meta"]["synth_pos"], it["meta"]["ask"]) for it in items if it["meta"]["view"] == "pair")
    # duration / loudness balance on the single clips that are used by items
    singles = defaultdict(list)
    for it in items:
        if it["meta"]["view"] != "pair":
            singles[it["meta"]["label"]].append((it["audio"]["bank"], it["audio"]["key"], it["meta"]["dur_s"], it["meta"]["rms_dbfs"]))
    bal = {}
    for cls, v in singles.items():
        u = {(b, k): (d, r) for b, k, d, r in v}
        ds = np.array([d for d, r in u.values()])
        rs = np.array([10 ** (r / 20) for d, r in u.values()])
        bal[cls] = {"n_clips": len(u), "mean_dur_s": round(float(ds.mean()), 3), "median_dur_s": round(float(np.median(ds)), 3),
                    "mean_rms": round(float(rs.mean()), 5), "mean_rms_dbfs": round(float(20 * np.log10(rs.mean())), 2)}
    rel_dur = abs(bal["real"]["mean_dur_s"] / bal["synthetic"]["mean_dur_s"] - 1)
    rel_rms = abs(bal["real"]["mean_rms"] / bal["synthetic"]["mean_rms"] - 1)
    bal["rel_diff_mean_dur"] = round(rel_dur, 4)
    bal["rel_diff_mean_rms"] = round(rel_rms, 4)
    assert rel_dur < 0.10 and rel_rms < 0.10, bal
    # composition of the synthetic half (single clips used by items)
    comp = Counter()
    real_comp = Counter()
    seen = set()
    for it in items:
        if it["meta"]["view"] == "pair" or (it["audio"]["key"] in seen):
            continue
        seen.add(it["audio"]["key"])
        if it["meta"]["label"] == "synthetic":
            comp[it["meta"]["system"]] += 1
        else:
            real_comp[it["meta"]["corpus"]] += 1
    ns = sum(comp.values())
    share = {k: round(v / ns, 4) for k, v in comp.most_common()}
    assert smoke or share.get("kokoro", 0) <= 0.25, share
    nr = sum(real_comp.values())
    real_share = {k: round(v / nr, 4) for k, v in real_comp.most_common()}
    # CAL disjointness (speakers / voices / source recordings)
    def ids(rows_, f):
        s = set()
        for it in rows_:
            v = it["meta"].get(f)
            s |= set(v) if isinstance(v, list) else {v}
        return s - {None}
    trs = [it for it in items if it["set"] == "train"]
    cas = [it for it in items if it["set"] == "CAL"]
    spk_overlap = sorted(ids(trs, "speaker") & ids(cas, "speaker"))
    src_overlap = sorted(ids(trs, "source_id") & ids(cas, "source_id"))
    assert not spk_overlap, spk_overlap[:5]
    # MMSU non-equality (reads only the 'question' field)
    mm_path = VOL / "bench_mmsu" / "truth.jsonl"
    mmsu = L.check_mmsu_nonequality([it["question"] for it in items], mm_path) if mm_path.exists() else "truth file absent"
    # leakage vs our evaluation items
    adi = adi_ids()
    ours_spk = ids(items, "speaker")
    ours_src = ids(items, "source_id")
    raw_ids = {s.split(":", 1)[1] for s in ours_src if ":" in s}
    leak = {"adi_items_file": (VOL / "adi" / "items.jsonl").exists(),
            "speaker_hits": sorted(ours_spk & adi["speakers"])[:10],
            "source_id_hits": sorted((ours_src | raw_ids) & adi["ids"])[:10],
            "hvb_call_hits": sorted({s.split(":", 1)[1] for s in ours_src if s.startswith("hvb:")} & adi["hvb_calls"])[:10]}
    ev = VOL / "data" / "eval_items.jsonl"
    if ev.exists():
        banks = set(bank_names)
        hits = 0
        for it in C.read_jsonl(ev):
            a = it.get("audio") or {}
            hits += a.get("bank") in banks
        leak["exp0_eval_items_referencing_our_banks"] = hits
    assert not leak["speaker_hits"] and not leak["source_id_hits"] and not leak["hvb_call_hits"], leak
    # summary
    def cnt(rows_, f):
        return dict(sorted(Counter(f(r) for r in rows_).items()))
    examples = {}
    for s in sorted({it["source"] for it in items}):
        rs = [it for it in items if it["source"] == s and it["set"] == "train"]
        r = random.Random(s)
        examples[s] = [{"question": it["question"], "options": it["options"], "gold": it["gold"], "label": it["meta"]["label"],
                        "system": it["meta"]["system"]} for it in r.sample(rs, min(3, len(rs)))]
    used = {(it["audio"]["bank"], it["audio"]["key"]) for it in items}
    hours = sum(idx[b][k][1] for b, k in used) / SR / 3600
    import synth as S
    systems_used = sorted(comp)
    summary = {
        "builder": L.BUILDER, "out_dir": str(out), "smoke": smoke,
        "n_items": {"train": len(trs), "CAL": len(cas)},
        "items_per_source": {sp: cnt([it for it in items if it["set"] == sp], lambda it: it["source"]) for sp in ("train", "CAL")},
        "items_per_qtype": {sp: cnt([it for it in items if it["set"] == sp], lambda it: it["meta"]["qtype"]) for sp in ("train", "CAL")},
        "items_per_view": {sp: cnt([it for it in items if it["set"] == sp], lambda it: it["meta"]["view"]) for sp in ("train", "CAL")},
        "items_per_qname": {sp: cnt([it for it in items if it["set"] == sp], lambda it: it["meta"]["qname"]) for sp in ("train", "CAL")},
        "items_per_label": {sp: cnt([it for it in items if it["set"] == sp], lambda it: it["meta"]["label"]) for sp in ("train", "CAL")},
        "items_per_system": {sp: cnt([it for it in items if it["set"] == sp], lambda it: it["meta"]["system"]) for sp in ("train", "CAL")},
        "synthetic_half_share_by_system (single clips)": share,
        "synthetic_clips_by_system": dict(comp.most_common()),
        "real_half_share_by_corpus (single clips)": real_share,
        "real_clips_by_corpus": dict(real_comp.most_common()),
        "pairs": {"n": len(pairs), "same_text_share": round(sum(p["same_text"] for p in pairs) / max(1, len(pairs)), 3),
                  "balance": {f"{a}|{b}|{c}": v for (a, b, c), v in sorted(pair_bal.items())}},
        "noul_balance": noul, "choice_balance": {"|".join(map(str, k)): v for k, v in sorted(choice_bal.items())},
        "duration_rms_balance": bal,
        "fx_share": {cls: {f: round(np.mean([bool(recs[k].get(f)) for k in ks]), 3) for f in ("rir", "floor", "musan", "phone_fx", "nb_match")}
                     for cls, ks in (("real", [k for sp in kept for k in kept[sp]["real"]]),
                                     ("synthetic", [k for sp in kept for k in kept[sp]["synthetic"]]))},
        "audio_hours": round(hours, 2), "n_audio_clips": len(used), "banks": bank_names,
        "licences": {"real_sources": SOURCES, "generators": {s: S.REPOS[s] for s in systems_used},
                     "generators_planned_but_unused": sorted(set(SYN_SHARE) - set(systems_used))},
        "voices": {"kokoro": {"train": S.KOKORO_TRAIN, "CAL": S.KOKORO_CAL},
                   "piper": {"voices": S.PIPER_VOICES, "libritts_train_spk": S.PIPER_LIBRITTS_TRAIN_SPK,
                             "libritts_cal_spk": S.PIPER_LIBRITTS_CAL_SPK, "single_train": S.PIPER_SINGLE_TRAIN, "single_cal": S.PIPER_SINGLE_CAL},
                   "speecht5": {"train": S.SPEECHT5_TRAIN_SPK, "CAL": S.SPEECHT5_CAL_SPK},
                   "parler": {"train": S.PARLER_NAMES_TRAIN, "CAL": S.PARLER_NAMES_CAL},
                   "chatterbox": "zero-shot clones of LibriSpeech reference utterances (train refs from train speakers, CAL refs from CAL speakers)"},
        "examples": examples,
        "validation": {"schema_targets_gold_audio_keys": "ok", "max_items_per_clip": max(per_clip.values()),
                       "cal_speaker_overlap": spk_overlap, "cal_source_recording_overlap": src_overlap[:10],
                       "cal_share": round(len(cas) / len(items), 4), "mmsu_question_nonequality": mmsu,
                       "render_errors": len(errs), "render_error_examples": errs[:10]},
        "leakage_check": leak,
        "synth_stats": {"cpu": cpu_stats, "gpu": gpu_stats},
        "plan_jobs": {k: len(v) for k, v in plan["jobs"].items()},
        "timing_s": {"cpu_synth": round(t_cpu), "total": round(time.time() - t0)},
        "split_rule": "CAL = LibriSpeech CAL speakers (8 % hash), HVB conversations whose both speakers are CAL speakers, AMI CAL "
                      "meetings (train meetings never share a CAL speaker), Apollo dev partition; synthetic CAL = held-out TTS voices "
                      "(Kokoro af_nova/am_eric/bf_lily/bm_lewis, Piper john + 4 libritts speakers, SpeechT5 rms x-vectors, Parler "
                      "Anna/Bruce/Emily/Tom, Chatterbox refs from CAL speakers) and copy-synthesis of CAL recordings; disjoint MUSAN "
                      "files and RIR rooms.",
    }
    return summary
