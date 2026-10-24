"""Build sb_dialog on Modal: fetch, plan, render banks in parallel, label, write items, check, write summary.json.

    modal run --detach data/sb_dialog/modal_build.py::full
"""
from __future__ import annotations

import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
INFERENCE = HERE.parent.parent / "inference"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(INFERENCE))

VOL = "/vol"
OUT = f"{VOL}/stageb/sb_dialog"
RAW = f"{VOL}/stageb/sb_dialog/work/raw"

HVB_SHA = "0bd721e877c4a85d8c13ff837e68661ea6200a98"
HVB_REF = f"github.com/cricketclub/gridspace-stanford-harper-valley@{HVB_SHA} (CC BY 4.0, repo LICENSE)"
SLUE_REPO, SLUE_REV = "asapp/slue-phase-2", "e2989c55a53593a8e39b8f8ebdb47ccaccbe484a"
E21_SHA = "c05ab6fd8b4b627d123c922a22a39e993dd37635"
E21_REF = (f"github.com/revdotcom/speech-datasets@{E21_SHA} earnings21/ (CC BY-SA 4.0 per earnings21/README badge + "
           "LICENSE.md; RTTM = reference diarization), share-alike")
AMI_ANN = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
AMI_WAV = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.Mix-Headset.wav"
AMI_REF = ("AMI Meeting Corpus, official mirror groups.inf.ed.ac.uk/ami (Mix-Headset wavs + ami_public_manual_1.6.2 "
           "word alignments; CC BY 4.0, HF mirror edinburghcstr/ami@46f28f2503e2ec48f8867a84eef356c70476beab "
           "cardData.license=cc-by-4.0)")
LICENCE = {"hvb": "CC BY 4.0", "ami": "CC BY 4.0", "earnings21": "CC BY-SA 4.0"}
BUILD_TAG = "v2"  # bank-name suffix, so a rebuild never overwrites banks that are already published

volume = modal.Volume.from_name("egma-homi", create_if_missing=False)
_local = ["modal_build.py", "build.py", "labels.py", "questions.py", "hvb.py"]


def _add_code(img):
    img = img.add_local_file(INFERENCE / "common.py", "/root/common.py")
    img = img.add_local_file(INFERENCE / "answer_codes.json", "/root/answer_codes.json")
    for f in _local:
        img = img.add_local_file(HERE / f, f"/root/{f}")
    return img


image = _add_code(
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "curl")
    .uv_pip_install("numpy==2.3.3", "scipy==1.16.2", "pyarrow==21.0.0", "soundfile==0.13.1",
                    "huggingface_hub==0.35.3", "requests==2.32.5")
    .env({"HF_HOME": f"{VOL}/hf_cache"})
)
app = modal.App("sb-dialog-build")


# Fetch

@app.function(image=image, volumes={VOL: volume}, timeout=3600, cpu=4, memory=8192)
def fetch_hvb() -> dict:
    import json
    import shutil
    import subprocess
    import tempfile

    dst = Path(RAW) / "hvb"
    if not (dst / "DONE").exists():
        tmp = Path(tempfile.mkdtemp())
        subprocess.run(f"curl -sSL https://codeload.github.com/cricketclub/gridspace-stanford-harper-valley/tar.gz/"
                       f"{HVB_SHA} | tar xz -C {tmp}", shell=True, check=True)
        src = next(tmp.glob("gridspace-stanford-harper-valley-*")) / "data"
        dst.mkdir(parents=True, exist_ok=True)
        for sub in ("audio", "metadata", "transcript"):
            if (dst / sub).exists():
                shutil.rmtree(dst / sub)
            shutil.copytree(src / sub, dst / sub)
        lic = next(tmp.glob("gridspace-stanford-harper-valley-*")) / "LICENSE"
        shutil.copy(lic, dst / "LICENSE")
        (dst / "DONE").write_text("ok")
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    ids = {}
    for split, fn in (("test", "hvb/test-00000-of-00001.parquet"),):
        p = hf_hub_download(SLUE_REPO, fn, repo_type="dataset", revision=SLUE_REV)
        t = pq.read_table(p, columns=["issue_id"])
        ids[split] = sorted(set(t.column("issue_id").to_pylist()))
    (dst / "slue_ids.json").write_text(json.dumps(ids))
    volume.commit()
    n = len(list((dst / "transcript").glob("*.json")))
    return {"conversations": n, "slue_test": len(ids["test"])}


def ami_annotations(zp: Path) -> dict:
    """ami_public_manual_1.6.2.zip -> {meeting: {global speaker id: {sp, inc, letter}}}"""
    import io
    import zipfile

    import build

    z = zipfile.ZipFile(io.BytesIO(zp.read_bytes()))
    meet_spk = build.parse_ami_meetings(z.read("corpusResources/meetings.xml").decode("latin-1"))
    ann = {}
    for name in z.namelist():
        if not name.startswith("words/") or not name.endswith(".words.xml"):
            continue
        m, letter = Path(name).name.split(".")[:2]
        sp, inc = build.parse_ami_words(z.read(name).decode("latin-1"))
        gid = meet_spk.get(m, {}).get(letter, f"{m}.{letter}")
        ann.setdefault(m, {})[gid] = {"sp": sp, "inc": inc, "letter": letter}
    return ann


@app.function(image=image, volumes={VOL: volume}, timeout=3 * 3600, cpu=8, memory=8192)
def fetch_ami(max_meetings: int = 999) -> dict:
    import io
    import json
    import subprocess
    import zipfile
    from concurrent.futures import ThreadPoolExecutor

    import requests

    import build

    dst = Path(RAW) / "ami"
    dst.mkdir(parents=True, exist_ok=True)
    zp = dst / "ami_public_manual_1.6.2.zip"
    if not zp.exists():
        r = requests.get(AMI_ANN, timeout=600)
        r.raise_for_status()
        zp.write_bytes(r.content)
    ann = ami_annotations(zp)
    meetings = sorted(ann)[:max_meetings]

    def get(m):
        p = dst / f"{m}.wav"
        if p.exists() and p.stat().st_size > 1000:
            return m, True
        tmp = p.with_suffix(".part")
        rc = subprocess.run(["curl", "-sSfL", "--retry", "3", "-o", str(tmp), AMI_WAV.format(m=m)]).returncode
        if rc == 0 and tmp.exists() and tmp.stat().st_size > 1000:
            tmp.rename(p)
            return m, True
        return m, False

    with ThreadPoolExecutor(16) as ex:
        ok = dict(ex.map(get, meetings))
    have = [m for m in meetings if ok[m]]
    volume.commit()
    return {"meetings_annotated": len(ann), "downloaded": len(have), "missing": [m for m in meetings if not ok[m]]}


@app.function(image=image, volumes={VOL: volume}, timeout=3600, cpu=1, memory=2048, max_containers=30)
def ami_get(meetings: list) -> dict:
    """Download a few Mix-Headset wavs (unique temp names, so it can run next to fetch_ami)."""
    import subprocess
    import uuid

    dst = Path(RAW) / "ami"
    got = []
    for m in meetings:
        p = dst / f"{m}.wav"
        if p.exists() and p.stat().st_size > 1000:
            continue
        tmp = dst / f"{m}.{uuid.uuid4().hex[:8]}.part2"
        rc = subprocess.run(["curl", "-sSfL", "--retry", "3", "-o", str(tmp), AMI_WAV.format(m=m)]).returncode
        if rc == 0 and tmp.exists() and tmp.stat().st_size > 1000 and not p.exists():
            tmp.rename(p)
            got.append(m)
            volume.commit()
        elif tmp.exists():
            tmp.unlink()
    return {"got": got}


@app.function(image=image, volumes={VOL: volume}, timeout=3600, cpu=4, memory=4096)
def fetch_earnings() -> dict:
    import csv
    import io
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    import requests

    base = f"https://raw.githubusercontent.com/revdotcom/speech-datasets/{E21_SHA}/earnings21"
    dst = Path(RAW) / "earnings21"
    dst.mkdir(parents=True, exist_ok=True)
    meta = requests.get(f"{base}/earnings21-file-metadata.csv", timeout=60).text
    (dst / "file-metadata.csv").write_text(meta)
    (dst / "speaker-metadata.csv").write_text(requests.get(f"{base}/speaker-metadata.csv", timeout=60).text)
    (dst / "LICENSE.md").write_text(requests.get(f"{base}/LICENSE.md", timeout=60).text)
    ids = [r["file_id"] for r in csv.DictReader(io.StringIO(meta))]

    def get(fid):
        ok = True
        for sub, ext in (("rttms", "rttm"), ("media", "mp3")):
            p = dst / f"{fid}.{ext}"
            if p.exists() and p.stat().st_size > 100:
                continue
            tmp = p.with_suffix(".part")
            rc = subprocess.run(["curl", "-sSfL", "--retry", "3", "-o", str(tmp), f"{base}/{sub}/{fid}.{ext}"]).returncode
            if rc == 0:
                tmp.rename(p)
            else:
                ok = False
        return fid, ok

    with ThreadPoolExecutor(8) as ex:
        res = dict(ex.map(get, ids))
    volume.commit()
    return {"files": len(ids), "ok": sum(res.values()), "failed": [k for k, v in res.items() if not v]}


# Render

def _low_runs(x, sr, min_len=1.0):
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


def _finish(x, rng):
    import numpy as np
    pk = float(np.max(np.abs(x))) if len(x) else 0.0
    if pk > 0:
        x = x * (rng.uniform(0.4, 0.95) / pk)
    return np.clip(x, -0.99, 0.99).astype(np.float32)


@app.function(image=image, volumes={VOL: volume}, timeout=2 * 3600, cpu=2, memory=12288, retries=1, max_containers=48)
def render_shard(args) -> dict:
    """args = (bank, source, out_dir, jobs). hvb jobs: (sid, [(key, layout)]); ami/earnings jobs: (file id,
    [(key, t0, t1)]). Writes <out_dir>/audio/<bank>.{npy,json}; returns per-clip stats."""
    import json
    import random
    import subprocess

    import numpy as np
    import soundfile as sf

    import common
    import hvb
    import labels as L

    bank, source, out_dir, jobs = args
    done = Path(out_dir) / "work" / "render" / f"{bank}.json"
    if done.exists() and (Path(out_dir) / "audio" / f"{bank}.json").exists():
        return json.loads(done.read_text())
    rng = random.Random(f"{common.SEED}:{bank}")
    clips, stats, errors = {}, {}, []
    for fid, items in jobs:
        try:
            if source == "hvb":
                chans, segs_by_role = {}, {}
                base = Path(RAW) / "hvb"
                segs = hvb.parse_transcript(json.loads((base / "transcript" / f"{fid}.json").read_text()))
                for role in hvb.ROLES:
                    x, sr = sf.read(base / "audio" / role / f"{fid}.wav", dtype="float32")
                    if x.ndim > 1:
                        x = x.mean(1)
                    chans[role] = x
                    segs_by_role[role] = [g for g in segs if g["role"] == role]
                beds = {r: hvb.noise_bed(chans[r], sr, segs_by_role[r]) for r in hvb.ROLES}
                for key, lay in items:
                    tracks = hvb.render_tracks(lay, chans, beds, sr, rng)
                    ev = {r: hvb.energy_vad(tracks[r], sr) for r in hvb.ROLES}
                    mix = tracks["agent"] + tracks["caller"]
                    y = common.resample(mix, sr, common.SAMPLE_RATE)
                    y = _finish(y, rng)
                    sp_all = [iv for r in hvb.ROLES for iv in lay["speech"][r]]
                    ev_all = [iv for r in hvb.ROLES for iv in ev[r]]
                    cov = L.total(L.merge([(max(s, a), min(e, b)) for s, e in sp_all for a, b in ev_all
                                           if min(e, b) > max(s, a)])) / max(1e-6, L.total(sp_all))
                    clips[key] = y
                    stats[key] = {"dur": len(y) / common.SAMPLE_RATE, "ev": ev, "low": _low_runs(y, common.SAMPLE_RATE),
                                  "align": round(cov, 3)}
            else:
                if source == "ami":
                    x, sr = sf.read(Path(RAW) / "ami" / f"{fid}.wav", dtype="float32")
                else:
                    p = Path(RAW) / "earnings21" / f"{fid}.mp3"
                    try:
                        x, sr = sf.read(p, dtype="float32")
                    except Exception:  # noqa: BLE001
                        raw = subprocess.run(["ffmpeg", "-v", "quiet", "-i", str(p), "-ac", "1", "-ar", "16000",
                                              "-f", "f32le", "-"], capture_output=True, check=True).stdout
                        x, sr = np.frombuffer(raw, np.float32).copy(), 16000
                if x.ndim > 1:
                    x = x.mean(1)
                if sr != common.SAMPLE_RATE:
                    x = common.resample(x, sr, common.SAMPLE_RATE)
                    sr = common.SAMPLE_RATE
                for key, t0, t1 in items:
                    seg = x[int(t0 * sr): int(t1 * sr)]
                    if len(seg) < 19.9 * sr:
                        errors.append([key, "short"])
                        continue
                    y = _finish(seg, rng)
                    clips[key] = y
                    stats[key] = {"dur": len(y) / sr, "low": _low_runs(y, sr)}
        except Exception as e:  # noqa: BLE001
            errors.append([fid, repr(e)[:300]])
    common.write_audio_bank(Path(out_dir) / "audio", bank, clips)
    res = {"bank": bank, "stats": stats, "errors": errors,
           "hours": round(sum(len(v) for v in clips.values()) / common.SAMPLE_RATE / 3600, 3)}
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(res))
    volume.commit()
    return res


# Main

@app.function(image=image, volumes={VOL: volume}, timeout=5 * 3600, cpu=8, memory=65536)
def main(out_dir: str = OUT, scale: float = 1.0, hvb_variants: int = 13, fetch: bool = True) -> dict:
    import csv
    import json
    import random
    import time
    from collections import Counter, defaultdict

    import soundfile as sf

    import build
    import common
    import hvb
    import labels as L

    t_start = time.time()
    log = []

    def note(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.append(msg)

    out = Path(out_dir)
    (out / "work").mkdir(parents=True, exist_ok=True)
    smoke = scale < 1.0
    if fetch:
        fh, fa, fe = fetch_hvb.spawn(), fetch_ami.spawn(), fetch_earnings.spawn()
        rh, ra, re_ = fh.get(), fa.get(), fe.get()
        note("fetch hvb", rh)
        note("fetch ami", {k: v for k, v in ra.items() if k != "missing"}, "missing", ra["missing"][:10])
        note("fetch earnings", re_)
    volume.reload()

    # leakage sources
    adi_calls, adi_spk, adi_ids = set(), set(), set()
    adi_path = Path(VOL) / "adi" / "items.jsonl"
    for line in adi_path.open():
        it = json.loads(line)
        m = it.get("meta") or {}
        if it.get("source") == "hvb":
            adi_calls.add(str(m.get("call")))
            adi_spk.add(str(m.get("speaker")))
        for k in ("call", "recording", "file", "source_id", "meeting", "clip"):
            if m.get(k) is not None:
                adi_ids.add(str(m[k]))
    eval_ids = set()
    ev_path = Path(VOL) / "data" / "eval_items.jsonl"
    if ev_path.exists():
        for line in ev_path.open():
            it = json.loads(line)
            m = it.get("meta") or {}
            for k in ("call", "recording", "file", "source_id", "meeting", "clip", "speaker"):
                if m.get(k) is not None:
                    eval_ids.add(str(m[k]))
    slue_test = set(json.loads((Path(RAW) / "hvb" / "slue_ids.json").read_text())["test"])
    note("eval-set hvb calls", len(adi_calls), "speakers", len(adi_spk), "| SLUE test calls", len(slue_test))

    rng = random.Random(f"{common.SEED}:sb_dialog")
    plan = []  # dicts: key, source, split, fid, layout | (t0, t1), speakers, events, kind

    # HVB
    base = Path(RAW) / "hvb"
    sids = sorted(p.stem for p in (base / "transcript").glob("*.json"))
    conv_spk, excluded = {}, Counter()
    for sid in sids:
        if sid in adi_calls or sid in slue_test:
            excluded["adi_or_slue_test"] += 1
            continue
        md = json.loads((base / "metadata" / f"{sid}.json").read_text())
        conv_spk[sid] = {f"hvb_spk:{md['agent']['speaker_id']}", f"hvb_spk:{md['caller']['speaker_id']}"}
    if smoke:
        conv_spk = dict(list(conv_spk.items())[:40])
    hsplit, cal_spk = build.hvb_speaker_split(conv_spk, 0.025)
    note("HVB conversations usable", len(conv_spk), "excluded", dict(excluded), "split", Counter(hsplit.values()),
         "CAL speakers", sorted(cal_spk))
    hvb_jobs = defaultdict(list)
    for sid, sp_ in hsplit.items():
        if sp_ is None:
            continue
        segs = hvb.parse_transcript(json.loads((base / "transcript" / f"{sid}.json").read_text()))
        if not segs:
            continue
        wl = min(sf.info(str(base / "audio" / r / f"{sid}.wav")).duration for r in hvb.ROLES)
        crng = random.Random(f"{common.SEED}:hvb:{sid}")
        lays = []
        for i, (t0, t1) in enumerate(hvb.natural_windows(segs, wl, crng)):
            lays.append((f"hvb_{sid}_n{i}", hvb.natural_layout(segs, t0, t1), "natural"))
        turns = hvb.make_turns(segs)
        nv = hvb_variants
        for j in range(nv * 2):
            if sum(1 for x in lays if x[2] == "relaid") >= nv:
                break
            lay = hvb.relaid_layout(turns, crng)
            if lay is not None:
                lays.append((f"hvb_{sid}_v{j}", lay, "relaid"))
        for key, lay, kind in lays:
            plan.append({"key": key, "source": "hvb", "split": sp_, "fid": sid, "layout": lay, "kind": kind,
                         "speakers": sorted(conv_spk[sid]), "events": lay["events"], "dur": lay["T"]})
            hvb_jobs[sid].append((key, lay))

    # AMI
    ami_ann = ami_annotations(Path(RAW) / "ami" / "ami_public_manual_1.6.2.zip")
    ami_missing = sorted(m for m in ami_ann if not (Path(RAW) / "ami" / f"{m}.wav").exists())
    ami_ann = {m: v for m, v in ami_ann.items() if m not in ami_missing}
    if smoke:
        ami_ann = dict(sorted(ami_ann.items())[:6])
    note("AMI meetings without Mix-Headset wav (skipped)", len(ami_missing), ami_missing[:20])
    ami_groups = {m: set(v) for m, v in ami_ann.items()}
    ami_cal = build.cal_by_components(ami_groups, 0.03, random.Random(f"{common.SEED}:ami_split"))
    cal_spk_ami = {s for m in ami_cal for s in ami_groups[m]}
    ami_split = {m: ("CAL" if m in ami_cal else ("train" if not (ami_groups[m] & cal_spk_ami) else None)) for m in ami_ann}
    note("AMI meetings", len(ami_ann), "split", Counter(ami_split.values()), "CAL", sorted(ami_cal))
    win_jobs = defaultdict(list)
    for m, spk in sorted(ami_ann.items()):
        if ami_split[m] is None:
            continue
        try:
            dur = sf.info(str(Path(RAW) / "ami" / f"{m}.wav")).duration
        except Exception as e:  # noqa: BLE001
            note("ami info fail", m, repr(e)[:100])
            continue
        inc = {g: [tuple(x) for x in v["inc"]] for g, v in spk.items()}
        sp = {g: [tuple(x) for x in v["sp"]] for g, v in spk.items()}
        for i, (t0, t1) in enumerate(build.plan_windows(inc, dur, random.Random(f"{common.SEED}:ami:{m}"))):
            key = f"ami_{m}_{i:03d}"
            plan.append({"key": key, "source": "ami", "split": ami_split[m], "fid": m, "t0": t0, "t1": t1,
                         "kind": "window", "speakers": sorted(f"ami_spk:{g}" for g in spk),
                         "sp": {g: L.clip_to(v, t0, t1) for g, v in sp.items()},
                         "inc": {g: L.clip_to(v, t0, t1) for g, v in inc.items()}, "dur": t1 - t0, "events": []})
            win_jobs[("ami", m)].append((key, t0, t1))

    # Earnings-21
    e_dir = Path(RAW) / "earnings21"
    e_ids = [r["file_id"] for r in csv.DictReader((e_dir / "file-metadata.csv").open())
             if (e_dir / f"{r['file_id']}.rttm").exists() and (e_dir / f"{r['file_id']}.mp3").exists()]
    if smoke:
        e_ids = e_ids[:3]
    e_cal = set(random.Random(f"{common.SEED}:e21_split").sample(e_ids, max(1, round(0.03 * len(e_ids)))))
    durs = {r["file_id"]: float(r["audio_length"]) for r in csv.DictReader((e_dir / "file-metadata.csv").open())}
    for fid in e_ids:
        spk = build.parse_rttm((e_dir / f"{fid}.rttm").read_text())
        for i, (t0, t1) in enumerate(build.plan_windows(spk, durs[fid] - 0.5, random.Random(f"{common.SEED}:e21:{fid}"))):
            key = f"e21_{fid}_{i:03d}"
            sp = {g: L.clip_to(v, t0, t1) for g, v in spk.items()}
            plan.append({"key": key, "source": "earnings21", "split": "CAL" if fid in e_cal else "train", "fid": fid,
                         "t0": t0, "t1": t1, "kind": "window", "speakers": sorted(f"e21_spk:{fid}:{g}" for g in spk),
                         "sp": sp, "inc": sp, "dur": t1 - t0, "events": []})
            win_jobs[("earnings21", fid)].append((key, t0, t1))
    note("earnings21 files", len(e_ids), "CAL", sorted(e_cal))

    # pre-filter (annotation-only labels; energy veto comes after rendering)
    def labels_for(p, st):
        T = st["dur"] if st else p["dur"]
        low = st["low"] if st else None
        if p["source"] == "hvb":
            lay = p["layout"]
            ev = st.get("ev") if st else None
            return build.two_party_labels(lay["speech"], lay["incl"], T, ev, low)
        return build.multi_party_labels(p["sp"], p["inc"], T, low)

    # planning-pass selection: only clips that the (annotation-only) selection would use are rendered
    pre = []
    for p in plan:
        lab, _ = labels_for(p, None)
        if len(lab) >= 2:
            pre.append({"key": p["key"], "source": p["source"], "split": p["split"], "labels": lab})
    pre_pick = []
    for src in ("hvb", "ami", "earnings21"):  # per-source rngs: a source's clips do not depend on the others
        pre_pick += build.select_items([c for c in pre if c["source"] == src], random.Random(f"{common.SEED}:pre:{src}"))
    keep_keys = {c["key"] for c, _, _ in pre_pick}
    kept = [p for p in plan if p["key"] in keep_keys]
    note("planned clips", len(plan), "with >=2 labels", len(pre), "kept after pre-selection", len(kept),
         Counter(p["source"] for p in kept), "hours", round(sum(p["dur"] for p in kept) / 3600, 1),
         "pre-selected items", len(pre_pick))

    # shards
    shard_args = []
    hv_items = [(sid, [(k, lay) for k, lay in v if k in keep_keys]) for sid, v in sorted(hvb_jobs.items())]
    hv_items = [x for x in hv_items if x[1]]
    n_h = max(1, min(60, round(sum(len(v) for _, v in hv_items) / 300)))
    for i in range(n_h):
        shard_args.append((f"sb_dialog_hvb_{BUILD_TAG}_{i:02d}", "hvb", out_dir, hv_items[i::n_h]))
    for src, per in (("ami", 380), ("earnings21", 380)):
        jobs = [(fid, [x for x in v if x[0] in keep_keys]) for (s, fid), v in sorted(win_jobs.items()) if s == src]
        jobs = [x for x in jobs if x[1]]
        n_s = max(1, min(40, round(sum(len(v) for _, v in jobs) / per)))
        tag = f"{'ami' if src == 'ami' else 'e21'}_{BUILD_TAG}"
        for i in range(n_s):
            shard_args.append((f"sb_dialog_{tag}_{i:02d}", src, out_dir, jobs[i::n_s]))
    note("render shards", len(shard_args))
    stats, bank_of, errors, hours = {}, {}, [], Counter()
    for res in render_shard.map(shard_args, order_outputs=False):
        for k, v in res["stats"].items():
            stats[k] = v
            bank_of[k] = res["bank"]
        errors += res["errors"]
        btag = res["bank"].split("_")[2]
        hours[btag] += res["hours"]
    note("rendered clips", len(stats), "errors", len(errors), errors[:5], "hours", dict(hours))
    volume.reload()

    # final labels (with energy veto) -> clips
    clips = []
    align_drop = 0
    for p in kept:
        st = stats.get(p["key"])
        if st is None:
            continue
        if p["source"] == "hvb" and st.get("align", 1.0) < 0.75:
            align_drop += 1
            continue
        lab, facts = labels_for(p, st)
        if not lab:
            continue
        clips.append({"key": p["key"], "source": p["source"], "split": p["split"], "labels": lab,
                      "bank": bank_of[p["key"]], "dur": st["dur"], "licence": LICENCE[p["source"]],
                      "source_id": p["fid"], "speakers": p["speakers"], "kind": p["kind"],
                      "events": p.get("events") or [], "facts": facts})
    note("clips with labels", len(clips), "hvb align drops", align_drop)
    sup = Counter((c["source"], f, lab) for c in clips for f, lab in c["labels"].items())
    note("label supply", sorted(sup.items()))

    items = []
    for src in ("hvb", "ami", "earnings21"):
        picked = build.select_items([c for c in clips if c["source"] == src], random.Random(f"{common.SEED}:sel:{src}"))
        items += build.make_items(picked, random.Random(f"{common.SEED}:items:{src}"), common)
    train = [it for it in items if it["set"] == "train"]
    cal = [it for it in items if it["set"] == "CAL"]
    for name, rows in (("items_train.jsonl", train), ("items_cal.jsonl", cal)):
        common.write_jsonl(out / f"{name}.tmp", rows)
        (out / f"{name}.tmp").rename(out / name)
    note("items", len(items), "train", len(train), "cal", len(cal))

    # validation
    bank_index = {}
    for b in sorted(set(bank_of.values())):
        bank_index[b] = json.loads((out / "audio" / f"{b}.json").read_text())
    dur_by_key = {k: v["dur"] for k, v in stats.items()}
    val = build.validate(items, bank_index, dur_by_key)
    # spot-check audio loads for a sample
    srng = random.Random(1)
    banks = {}
    sample = srng.sample(items, min(60, len(items)))
    for it in sample:
        b = it["audio"]["bank"]
        if b not in banks:
            banks[b] = common.AudioBank(out / "audio", b)
        x = banks[b].get(it["audio"]["key"])
        assert len(x) > 19.9 * 16000 and float(abs(x).max()) <= 0.991 and float(abs(x).max()) > 0.05, it["id"]
    bank_sizes = {b: (out / "audio" / f"{b}.npy").stat().st_size for b in bank_index}
    assert max(bank_sizes.values()) < 4.2e9, bank_sizes

    # leakage
    our_hvb = {c["source_id"] for c in clips if c["source"] == "hvb"}
    our_ids = {c["source_id"] for c in clips} | {c["key"] for c in clips}
    used_spk = {s for c in clips for s in c["speakers"]}
    leak = {
        "hvb_calls_in_adi": len(our_hvb & adi_calls),
        "hvb_calls_in_slue_test": len(our_hvb & slue_test),
        "ids_in_adi_meta": len(our_ids & adi_ids),
        "ids_in_exp0_eval_meta": len(our_ids & eval_ids),
        "hvb_speakers_shared_with_adi": len({s.split(":")[1] for s in used_spk if s.startswith("hvb_spk:")} & adi_spk),
        "note": "HVB has ~60 voice actors in total and the evaluation set's SLUE-test calls cover 50 of them, so "
                "speaker-level exclusion is impossible; every conversation (issue_id) of the SLUE test split (a "
                "superset of the evaluation calls) is excluded. The evaluation HVB tasks are intent / dialog act.",
        "banks_referenced_by_adi_or_eval": 0,
    }
    for pth in (adi_path, ev_path):
        if pth.exists():
            txt = pth.read_text()
            leak["banks_referenced_by_adi_or_eval"] += txt.count('"sb_dialog_')
    assert leak["hvb_calls_in_adi"] == 0 and leak["hvb_calls_in_slue_test"] == 0 and leak["ids_in_adi_meta"] == 0
    assert leak["banks_referenced_by_adi_or_eval"] == 0
    note("leakage", leak)

    # summary.json (written last)
    used_keys = {it["audio"]["key"] for it in items}
    hours_used = sum(dur_by_key[k] for k in used_keys) / 3600
    by_src = defaultdict(Counter)
    for it in items:
        by_src[it["source"]][f'{it["set"]}|{it["meta"]["qtype"]}|{it["meta"]["family"]}'] += 1
    examples = {}
    for src in sorted({it["source"] for it in items}):
        exs = [it for it in train if it["source"] == src]
        examples[src] = [{"question": it["question"], "options": it["options"], "gold": it["gold"],
                          "key": it["audio"]["key"]} for it in random.Random(2).sample(exs, min(3, len(exs)))]
    chan = Counter((it["source"], it["channel"]) for it in items)
    summary = {
        "builder": "sb_dialog",
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_items": len(items), "n_train": len(train), "n_cal": len(cal),
        "cal_frac": round(len(cal) / max(1, len(items)), 4),
        "counts_by_source": {s: sum(c.values()) for s, c in by_src.items()},
        "counts_by_source_split_qtype_family": {s: dict(sorted(c.items())) for s, c in by_src.items()},
        "counts_by_qtype": dict(Counter(it["meta"]["qtype"] for it in items)),
        "counts_by_split": dict(Counter(it["set"] for it in items)),
        "channel": {f"{s}|{c}": n for (s, c), n in sorted(chan.items())},
        "clips_used": len(used_keys), "audio_hours_used": round(hours_used, 2),
        "audio_hours_in_banks": round(sum(hours.values()), 2), "hours_by_source_banks": dict(hours),
        "clip_duration_s": {"min": val["dur_min"], "max": val["dur_max"], "mean": val["dur_mean"]},
        "kinds": {f"{s}|{k}": n for (s, k), n in Counter((c["source"], c["kind"]) for c in clips
                                                         if c["key"] in used_keys).items()},
        "balance": val["balance"],
        "banks": {b: {"bytes": bank_sizes[b], "clips": len(bank_index[b])} for b in sorted(bank_sizes)},
        "sources": {
            "sb_dialog_hvb": {"ref": HVB_REF, "licence": "CC BY 4.0",
                              "exclusion": f"SLUE-HVB test conversations ({SLUE_REPO}@{SLUE_REV}) incl. all evaluation-set calls"},
            "sb_dialog_ami": {"ref": AMI_REF, "licence": "CC BY 4.0"},
            "sb_dialog_earnings21": {"ref": E21_REF, "licence": "CC BY-SA 4.0 (share-alike)"},
        },
        "skipped": {
            "earnings22": "not used; Earnings-21 reference RTTMs cover the long-call conference-call slice",
            "constructed_from_other_two_channel_sources": "only HVB has separate per-speaker channels here",
        },
        "leakage": leak,
        "render_errors": errors[:50],
        "examples": examples,
        "log": log,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    (out / "summary.json.tmp").write_text(json.dumps(summary, indent=1))
    (out / "summary.json.tmp").rename(out / "summary.json")
    volume.commit()
    return {k: summary[k] for k in ("n_items", "n_train", "n_cal", "counts_by_source", "audio_hours_used", "elapsed_s")}


@app.local_entrypoint()
def full(fetch: int = 1):
    print(main.remote(OUT, 1.0, 13, bool(fetch)))


@app.function(image=image, volumes={VOL: volume}, timeout=3600, cpu=2, memory=4096)
def ami_boost(start: int = 60, per: int = 5) -> dict:
    ann = ami_annotations(Path(RAW) / "ami" / "ami_public_manual_1.6.2.zip")
    ms = sorted(ann)[start:]
    groups = [ms[i:i + per] for i in range(0, len(ms), per)]
    got = []
    for r in ami_get.map(groups):
        got += r["got"]
    return {"requested": len(ms), "got": len(got)}


@app.local_entrypoint()
def boost(start: int = 60):
    print(ami_boost.remote(start))


@app.local_entrypoint()
def prefetch():
    calls = [fetch_hvb.spawn(), fetch_ami.spawn(), fetch_earnings.spawn()]
    for c in calls:
        print(c.get())


@app.local_entrypoint()
def smoke():
    print(main.remote(f"{OUT}/work/smoke", 0.05, 9, False))
