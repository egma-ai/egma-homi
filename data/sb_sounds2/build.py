"""sb_sounds2 builder: hard-negative sound questions on FSD50K dev clips, and music questions on licence-clean FMA
tracks (genre, tempo, level, live) and OpenMIC-2018 clips (instruments, vocals). Runs inside Modal (modal_build.py).
"""

from __future__ import annotations

import collections
import csv
import glob
import hashlib
import io
import json
import math
import os
import random
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import fsd_labels as FL  # noqa: E402
import labels2 as L2  # noqa: E402

BUILDER = "sb_sounds2"
SR = common.SAMPLE_RATE

FSD_META_REPO, FSD_META_REV = "Fhrozen/FSD50k", "ccf1acaff12f3f4a4c10052dddafbb6d22152c9b"
FSD_AUDIO_REPO, FSD_AUDIO_REV = "quinnlue/FSD50K-16k", "2a60d475f4e2f0db624a902881af2df79d656145"
ONTOLOGY_URL = "https://raw.githubusercontent.com/audioset/ontology/d417d32bf59c711abb5910fd2f76a0eb44697991/ontology.json"
FMA_REPO, FMA_REV, FMA_SHARDS = "benjamin-paine/free-music-archive-large", "d4cb3e133a7c5a007ddd20458aa30aef8968cf27", 171
FMA_META_ZIP = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
OPENMIC_URL = "https://zenodo.org/api/records/1432913/files/openmic-2018-v1.0.0.tgz/content"
OPENMIC_MD5 = "e4ccf187e2bb5ab2e115416e8aafe7f4"
SDD_LIC_URL = "https://zenodo.org/records/10072001/files/audio_licenses.txt?download=1"
SDD_CSV_URL = "https://zenodo.org/records/10072001/files/song_describer.csv?download=1"

OK_FSD_LICENCES = {"http://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
                   "http://creativecommons.org/licenses/by/3.0/": "CC-BY-3.0"}

SOURCES_INFO = {
    "fsd50k": {"licence": "per clip CC0-1.0 or CC-BY-3.0 (FSD50K dataset CC-BY-4.0); CC-BY-NC and Sampling+ dropped",
               "origin": f"hf:{FSD_AUDIO_REPO}@{FSD_AUDIO_REV} (16 kHz FLAC repack) + labels/licences "
                         f"hf:{FSD_META_REPO}@{FSD_META_REV}; FSD50K DEV split only (Zenodo 4060432)"},
    "fma": {"licence": "per track CC-BY (1.0-4.0) / CC-BY-SA (2.0-4.0, flagged: share-alike) / CC0 / public domain, "
                       "verified against the official FMA metadata licence URL (raw_tracks.csv) and the repack's licence "
                       "field; NC / ND / Sampling+ / FMA-limited / Sound-Recording-Common-Law / FMP / Art Libre dropped. "
                       "FMA metadata (genres, album type) CC-BY-4.0",
            "origin": f"audio hf:{FMA_REPO}@{FMA_REV} (30 s clips of FMA 'large'); metadata {FMA_META_ZIP} "
                      "(github.com/mdeff/fma)"},
    "fma_signal": {"licence": "as fma", "origin": "fma excerpts, gain applied by construction (level family)"},
    "openmic": {"licence": "OpenMIC-2018 labels CC-BY-4.0 (Spotify / MARL@NYU, Zenodo 1432913 v1.0.0); audio = FMA "
                           "tracks, kept only if the official FMA licence of the track is CC-BY / CC-BY-SA / CC0 / PD",
                "origin": f"{OPENMIC_URL} (md5 {OPENMIC_MD5})"},
    "audioset_ontology": {"licence": "CC-BY-SA-4.0 (class hierarchy only, not training content)", "origin": ONTOLOGY_URL},
}

PHONE_FRAC = 0.10
CAL_FRAC = 0.03
MAX_ITEMS_PER_CLIP = 3
EXCERPT_S = 12.0
FMA_ARTIST_CAP = 50  # tracks per artist (diversity)

# targets (full build)
N_FSD_CLIPS = 10000      # each gets one presence item; PAIR_FRAC of them get a second one with the opposite answer
PAIR_FRAC = 0.5
N_FSD_WHICH = 3000
NEAR_FRAC = 0.8
Q_FMA = {"genre_top": 4500, "genre_sub": 2500, "genre_noul": 3000, "tempo": 2400, "level": 1200, "live": 400}
Q_OM = {"instr_noul": 3500, "instr_which": 800, "vocals": 1600, "use_bed": 600}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class NoShuffle(random.Random):
    def shuffle(self, x, *args, **kwargs):  # keeps score levels in natural order inside make_item
        return None


def _download(url: str, dst: Path, timeout: int = 1800) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    tmp = dst.with_suffix(dst.suffix + ".part")
    subprocess.run(["curl", "-fsSL", "--retry", "5", "--max-time", str(timeout), "-o", str(tmp), url], check=True)
    tmp.rename(dst)
    return dst


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests."""

    def __init__(self, url: str, block: int = 1 << 22):
        import requests

        self.s = requests.Session()
        r = self.s.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.url, self.size, self.pos, self.block = r.url, int(r.headers["Content-Length"]), 0, block
        self.buf_start, self.buf = 0, b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _get(self, start, end):
        for attempt in range(8):
            r = self.s.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=300)
            if r.status_code == 206:
                return r.content
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"range request failed: {self.url}")

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size)
        if not (self.buf_start <= self.pos and end <= self.buf_start + len(self.buf)):
            fetch_end = min(self.size, max(end, self.pos + self.block)) - 1
            self.buf_start, self.buf = self.pos, self._get(self.pos, fetch_end)
        data = self.buf[self.pos - self.buf_start: end - self.buf_start]
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


# 1. Metadata
def fetch_meta(raw: Path) -> dict:
    import zipfile

    from huggingface_hub import hf_hub_download

    raw.mkdir(parents=True, exist_ok=True)
    for f in ["labels/dev.csv", "labels/eval.csv", "labels/vocabulary.csv", "metadata/dev_clips_info_FSD50K.json"]:
        hf_hub_download(FSD_META_REPO, f, repo_type="dataset", revision=FSD_META_REV, local_dir=raw / "fsd_meta")
    _download(ONTOLOGY_URL, raw / "ontology.json")
    z = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(FMA_META_ZIP), buffer_size=1 << 22))
    for n in ["raw_tracks.csv", "tracks.csv", "genres.csv", "README.txt"]:
        (raw / "fma_meta").mkdir(exist_ok=True)
        (raw / "fma_meta" / n).write_bytes(z.read("fma_metadata/" + n))
    _download(SDD_LIC_URL, raw / "sdd_audio_licenses.txt")
    _download(SDD_CSV_URL, raw / "sdd_song_describer.csv")
    log("meta fetched")
    return {"fma_meta_zip_size": z.fp.raw.size if hasattr(z.fp, "raw") else None}


def _s(x):
    """str or None (pandas gives NaN for missing strings, which must not reach the JSON manifests)."""
    return x if isinstance(x, str) else None


def load_fma_meta(raw: Path) -> tuple[dict, L2.GenreTree, dict]:
    """-> ({track_id: info} for tracks with a commercially usable official licence and music genres, GenreTree, stats)."""
    import ast

    import pandas as pd

    m = raw / "fma_meta"
    tree = L2.GenreTree(list(csv.DictReader(open(m / "genres.csv"))))
    rt = pd.read_csv(m / "raw_tracks.csv", usecols=["track_id", "license_url", "license_title", "track_url", "artist_id",
                                                     "artist_name", "track_title", "album_title"])
    t = pd.read_csv(m / "tracks.csv", index_col=0, header=[0, 1])
    t.columns = [a + "." + b for a, b in t.columns]
    sdd_artists, sdd_ids = L2.parse_sdd_artists(open(raw / "sdd_audio_licenses.txt", encoding="utf-8").read())
    stats = collections.Counter()
    out = {}
    tt = t[["track.genres", "track.genres_all", "album.type", "track.tags", "artist.name", "track.title",
            "album.title"]].to_dict("index")
    for r in rt.itertuples(index=False):
        tid = int(r.track_id)
        stats["fma_tracks"] += 1
        lc = L2.licence_class(r.license_url if isinstance(r.license_url, str) else "")
        if lc is None:
            stats["drop_licence"] += 1
            continue
        if tid not in tt:
            stats["drop_no_tracks_row"] += 1
            continue
        row = tt[tid]
        if L2.norm_name(r.artist_name if isinstance(r.artist_name, str) else "") in sdd_artists:
            stats["drop_sdd_artist"] += 1
            continue
        genres = [int(g) for g in ast.literal_eval(row["track.genres"])]
        gall = [int(g) for g in ast.literal_eval(row["track.genres_all"])]
        titles = [tree.title[g] for g in gall if g in tree.title]
        out[tid] = {"track_id": tid, "licence_class": lc, "licence_url": r.license_url, "licence_title": _s(r.license_title),
                    "url": r.track_url if isinstance(r.track_url, str) else "", "artist_id": int(r.artist_id),
                    "artist": _s(r.artist_name), "title": _s(r.track_title), "album_title": _s(row["album.title"]),
                    "album_type": row["album.type"] if isinstance(row["album.type"], str) else None,
                    "tags": [str(x).lower() for x in ast.literal_eval(row["track.tags"])],
                    "genres": genres, "genres_all": gall, "music_ok": L2.track_genres_ok(titles)}
        stats["ok_licence_" + lc] += 1
    stats["sdd_artists"] = len(sdd_artists)
    stats["sdd_tracks"] = len(sdd_ids)
    return out, tree, dict(stats)


# 2. FMA shard processing (one container per parquet shard)
def choose_excerpt(x: np.ndarray, key: str, win_s: float = EXCERPT_S) -> np.ndarray | None:
    win = int(win_s * SR)
    if len(x) < int(8 * SR):
        return None
    if len(x) <= win:
        return x
    rng = random.Random(f"{common.SEED}:{key}")
    starts = list(range(0, len(x) - win + 1, SR // 2))
    rng.shuffle(starts)
    for s in starts:
        seg = x[s: s + win]
        db = FL.frame_rms_db(seg)
        if db.max() < -45:
            continue
        if np.mean(db < db.max() - 40) < 0.15:
            return seg
    return best_window(x, win)


def best_window(x: np.ndarray, win: int) -> np.ndarray:
    if len(x) <= win:
        return x
    hop = SR // 10
    e = np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64) ** 2)])
    starts = np.arange(0, len(x) - win + 1, hop)
    s = int(starts[np.argmax(e[starts + win] - e[starts])])
    return x[s: s + win]


def process_fma_shard(shard: int, work: Path) -> dict:
    """Download one parquet shard of the FMA repack, keep rows whose repack licence is commercially usable, decode,
    cut a 12 s excerpt, measure tempo + level, write work/fma/shard_XXX.{npz,json}."""
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    outj = work / "fma" / f"shard_{shard:03d}.json"
    if outj.exists():
        return json.loads(outj.read_text())["stats"]
    t0 = time.time()
    fn = f"data/train-{shard:05d}-of-{FMA_SHARDS:05d}.parquet"
    p = hf_hub_download(FMA_REPO, fn, repo_type="dataset", revision=FMA_REV, local_dir="/tmp/fma")
    pf = pq.ParquetFile(p)
    feats = json.loads(pf.schema_arrow.metadata[b"huggingface"])["info"]["features"]
    lic_names = feats["license"]["names"]
    genre_names = feats["genres"]["feature"]["names"] if "feature" in feats["genres"] else None
    stats = collections.Counter()
    metas, arrays = [], {}
    cols = ["audio", "url", "title", "artist", "license", "genres", "album_title", "instrumental"]
    for batch in pf.iter_batches(columns=cols, batch_size=64):
        d = batch.to_pydict()
        for i in range(len(d["url"])):
            stats["rows"] += 1
            lname = lic_names[d["license"][i]] if d["license"][i] is not None else None
            lc = L2.mirror_licence_class(lname)
            if lc is None:
                continue
            stats["licence_ok"] += 1
            a = d["audio"][i]
            try:
                x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=False)
            except Exception:
                stats["decode_fail"] += 1
                continue
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != SR:
                x = common.resample(x, sr, SR)
            key = f"s{shard:03d}_{stats['rows']:05d}"
            seg = choose_excerpt(x, d["url"][i] or key)
            if seg is None:
                stats["too_short"] += 1
                continue
            seg = np.ascontiguousarray(seg, dtype=np.float32)
            pk = float(np.max(np.abs(seg)))
            if pk > 0.99:
                seg = (seg * (0.99 / pk)).astype(np.float32)
            tempo = L2.estimate_tempo(seg)
            arrays[key] = seg
            metas.append({"key": key, "url": d["url"][i] or "", "title": d["title"][i], "artist": d["artist"][i],
                          "mirror_licence": lname, "mirror_licence_class": lc,
                          "mirror_genres": [genre_names[g] for g in (d["genres"][i] or [])] if genre_names else None,
                          "album_title": d["album_title"][i], "orig_seconds": round(len(x) / SR, 2),
                          "active_db": round(FL.active_rms_db(seg), 2), "tempo": tempo})
            stats["kept"] += 1
    (work / "fma").mkdir(parents=True, exist_ok=True)
    np.savez(work / "fma" / f"shard_{shard:03d}.npz", **arrays)
    stats = dict(stats)
    stats["seconds"] = round(time.time() - t0)
    outj.write_text(json.dumps({"stats": stats, "items": metas}))
    os.remove(p)
    return stats


# 3. OpenMIC-2018
def process_openmic(work: Path, fma_ok: dict[int, dict]) -> dict:
    """fma_ok: {track_id: {licence_class, artist_id, artist}} for tracks with a usable official FMA licence."""
    import pandas as pd
    import soundfile as sf

    outj = work / "openmic" / "openmic.json"
    if outj.exists():
        return json.loads(outj.read_text())["stats"]
    t0 = time.time()
    cached = work / "openmic" / "openmic-2018-v1.0.0.tgz"  # kept so a re-run does not hit Zenodo again
    if cached.exists() and cached.stat().st_size > 2_000_000_000:
        tgz = cached
    else:
        tgz = _download(OPENMIC_URL, Path("/tmp/openmic.tgz"), timeout=3600)
    md5 = subprocess.run(["md5sum", str(tgz)], capture_output=True, text=True).stdout.split()[0]
    subprocess.run(["tar", "xzf", str(tgz), "-C", "/tmp"], check=True)
    if tgz != cached:
        cached.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy(tgz, cached)
        tgz.unlink()
    root = Path(glob.glob("/tmp/openmic-2018*")[0]) if glob.glob("/tmp/openmic-2018*") else Path("/tmp")
    metaf = glob.glob("/tmp/**/openmic-2018-metadata.csv", recursive=True)[0]
    labf = glob.glob("/tmp/**/openmic-2018-aggregated-labels.csv", recursive=True)[0]
    meta = pd.read_csv(metaf)
    lab = pd.read_csv(labf)
    log("openmic metadata columns", list(meta.columns), "labels columns", list(lab.columns))
    lic_col = "license_url" if "license_url" in meta.columns else None  # not license_image_file
    stats = collections.Counter({"md5_ok": int(md5 == OPENMIC_MD5)})
    labels = collections.defaultdict(list)
    for r in lab.itertuples(index=False):
        labels[str(r.sample_key)].append((str(r.instrument), float(r.relevance)))
    keep = {}
    for r in meta.to_dict("records"):
        sk = str(r["sample_key"])
        stats["clips"] += 1
        tid = int(r["track_id"]) if "track_id" in r else int(sk.split("_")[0])
        info = fma_ok.get(tid)
        if info is None:
            stats["drop_official_fma_licence_or_sdd"] += 1
            continue
        if lic_col is not None and isinstance(r[lic_col], str) and r[lic_col].startswith("http") \
                and L2.licence_class(r[lic_col]) is None:
            stats["drop_openmic_licence_field"] += 1
            continue
        pos, neg = L2.instrument_labels(labels.get(sk, []))
        if not pos and not neg:
            stats["drop_no_confident_label"] += 1
            continue
        keep[sk] = {"sample_key": sk, "track_id": tid, "artist_id": info["artist_id"], "artist": info["artist"],
                    "licence_class": info["licence_class"], "licence_url": info["licence_url"],
                    "openmic_licence_field": _s(r.get(lic_col)) if lic_col else None,
                    "start_time": float(r["start_time"]) if isinstance(r.get("start_time"), (int, float)) else None, "present": sorted(pos), "absent": sorted(neg),
                    "relevance": {i: round(v, 3) for i, v in labels.get(sk, [])}}
    ogg = {}
    for dp, _, fns in os.walk("/tmp"):
        for fn in fns:
            if fn.endswith(".ogg"):
                ogg[fn[:-4]] = os.path.join(dp, fn)
    stats["ogg_files"] = len(ogg)
    arrays, metas = {}, []
    for sk, m in sorted(keep.items()):
        if sk not in ogg:
            stats["drop_no_audio"] += 1
            continue
        try:
            x, sr = sf.read(ogg[sk], dtype="float32", always_2d=False)
        except Exception:
            stats["decode_fail"] += 1
            continue
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            x = common.resample(x, sr, SR)
        x = x[: int(10.5 * SR)]
        if len(x) < 5 * SR or FL.active_rms_db(x) < -50:
            stats["drop_short_or_silent"] += 1
            continue
        pk = float(np.max(np.abs(x)))
        if pk > 0.99:
            x = x * (0.99 / pk)
        arrays[sk] = np.ascontiguousarray(x, dtype=np.float32)
        metas.append(m)
        stats["kept"] += 1
    (work / "openmic").mkdir(parents=True, exist_ok=True)
    np.savez(work / "openmic" / "openmic.npz", **arrays)
    stats = dict(stats)
    stats["seconds"] = round(time.time() - t0)
    stats["metadata_columns"] = list(meta.columns)
    outj.write_text(json.dumps({"stats": stats, "items": metas}))
    return stats


# 4. FSD50K (as in sb_sounds, preferring clips sb_sounds did not use)
def load_fsd_meta(raw: Path):
    m = raw / "fsd_meta"
    vocab = {row[2]: row[1] for row in csv.reader(open(m / "labels/vocabulary.csv"))}
    info = json.load(open(m / "metadata/dev_clips_info_FSD50K.json"))
    eval_ids = {r["fname"] for r in csv.DictReader(open(m / "labels/eval.csv"))}
    clips = {}
    for r in csv.DictReader(open(m / "labels/dev.csv")):
        fid = r["fname"]
        assert fid not in eval_ids
        clips[fid] = {"labels": r["labels"].split(","), "license": info[fid]["license"],
                      "uploader": info[fid]["uploader"], "title": info[fid].get("title", "")}
    ontology = json.load(open(raw / "ontology.json"))
    return vocab, ontology, clips, eval_ids


def fetch_fsd_audio(raw: Path, smoke: bool = False):
    from huggingface_hub import snapshot_download

    pats = ["data/train-00000-*", "data/validation-00000-*"] if smoke else ["data/train-*", "data/validation-*"]
    snapshot_download(FSD_AUDIO_REPO, repo_type="dataset", revision=FSD_AUDIO_REV, local_dir=raw / "fsd_audio",
                      allow_patterns=pats + ["README.md"], max_workers=16)


def read_parquet_index(files) -> dict[str, dict]:
    import pyarrow.parquet as pq

    out = {}
    for f in files:
        t = pq.read_table(f, columns=["freesound_id", "duration_seconds", "source_partition", "license"])
        for fid, dur, part, lic in zip(*(t.column(c).to_pylist() for c in t.column_names)):
            out[str(fid)] = {"dur": float(dur), "part": part, "license": lic, "file": f}
    return out


def sb_sounds_used_ids(path: Path = Path("/vol/stageb/sb_sounds")) -> set[str]:
    used = set()
    for f in ("items_train.jsonl", "items_cal.jsonl"):
        p = path / f
        if not p.exists():
            continue
        for line in open(p):
            m = json.loads(line)["meta"]
            if m.get("freesound_id") is not None:
                used.add(str(m["freesound_id"]))
            for c in m.get("components") or []:
                if str(c).startswith("fsd50k:"):
                    used.add(str(c).split(":", 1)[1])
    return used


def select_fsd(clips, pidx, tax, n, rng, used: set[str]):
    stats = collections.Counter()
    elig = []
    for fid in sorted(clips, key=int):
        c = clips[fid]
        if c["license"] not in OK_FSD_LICENCES:
            stats["drop_licence"] += 1
            continue
        if fid not in pidx or pidx[fid]["part"] != "dev":
            stats["drop_no_audio"] += 1
            continue
        dur = pidx[fid]["dur"]
        if dur < 0.5:
            stats["drop_short"] += 1
            continue
        leaves = tax.leaves(c["labels"])
        if dur > 12.0 and len(leaves) > 1:
            stats["drop_long_multilabel"] += 1
            continue
        c["leaves"] = leaves
        elig.append(fid)
    stats["eligible"] = len(elig)
    stats["eligible_not_in_sb_sounds"] = sum(1 for f in elig if f not in used)
    freq = collections.Counter(l for fid in elig for l in clips[fid]["leaves"])
    keys = []
    for fid in elig:
        w = max((1.0 / freq[l]) ** 0.5 for l in clips[fid]["leaves"])
        keys.append((0 if fid not in used else 1, -math.log(max(rng.random(), 1e-12)) / w, fid))
    keys.sort()
    per_up = collections.Counter()
    cap = max(20, int(n * 0.015))
    chosen = []
    for _, _, fid in keys:
        up = clips[fid]["uploader"]
        if per_up[up] >= cap:
            continue
        per_up[up] += 1
        chosen.append(fid)
        if len(chosen) >= n:
            break
    stats["selected"] = len(chosen)
    stats["selected_overlap_with_sb_sounds"] = sum(1 for f in chosen if f in used)
    return sorted(chosen, key=int), dict(stats)


def choose_cal_groups(group_of: dict[str, str], frac: float, rng: random.Random, max_group: int) -> set[str]:
    sizes = collections.Counter(group_of.values())
    groups = sorted(sizes)
    rng.shuffle(groups)
    target = frac * len(group_of)
    cal, n = set(), 0
    for g in groups:
        if n >= target:
            break
        if sizes[g] <= max_group:
            cal.add(g)
            n += sizes[g]
    return cal


def _decode_file(args):
    import pyarrow.parquet as pq
    import soundfile as sf

    f, wanted = args
    wanted = set(wanted)
    out = {}
    pf = pq.ParquetFile(f)
    for batch in pf.iter_batches(columns=["freesound_id", "audio"], batch_size=128):
        for fid, a in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            fid = str(fid)
            if fid not in wanted:
                continue
            x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=False)
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != SR:
                x = common.resample(x, sr, SR)
            if len(x) > 12 * SR:
                x = best_window(x, 10 * SR)
            pk = float(np.max(np.abs(x))) if len(x) else 0.0
            if pk > 0.99:
                x = x * (0.99 / pk)
            out[fid] = np.ascontiguousarray(x, dtype=np.float32)
    return out


def decode_fsd(pidx, ids, workers=8):
    by_file = collections.defaultdict(list)
    for fid in ids:
        by_file[pidx[fid]["file"]].append(fid)
    audio = {}
    with Pool(workers) as pool:
        for part in pool.imap_unordered(_decode_file, sorted(by_file.items())):
            audio.update(part)
    return audio


# 5. Items
class ItemMaker:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.items: list[dict] = []
        self.noshuf = NoShuffle()
        self.n_per_clip = collections.Counter()

    def add(self, *, split, source, family, qtype, question, options, gold, audio, meta, allow_phone=True):
        assert len(set(options)) == len(options), options
        target = [0.0] * len(options)
        target[gold] = 1.0
        it = common.make_item(item_id=f"{BUILDER}_{len(self.items):06d}", set_name=split, source=source,
                              question=question, options=list(options), target=target, gold=gold,
                              rng=self.noshuf if qtype == "score" else self.rng, audio=dict(audio),
                              channel="phone" if allow_phone and self.rng.random() < PHONE_FRAC else "clean",
                              state_text=None, meta={"qtype": qtype, "family": family, **meta})
        self.items.append(it)
        self.n_per_clip[audio["key"]] += 1
        return it

    def noul(self, *, answer: str, **kw):
        return self.add(qtype="noul", options=["yes", "no"], gold=0 if answer == "yes" else 1, **kw)

    def room(self, key) -> bool:
        return self.n_per_clip[key] < MAX_ITEMS_PER_CLIP


def fsd_items(mk: ItemMaker, sel, clips, split_of, tax, onto, cooc, rng, n_pair, n_which) -> dict:
    stats = collections.Counter()

    def meta(fid):
        c = clips[fid]
        return {"source_id": f"fsd50k:{fid}", "freesound_id": int(fid), "speaker": f"freesound_uploader:{c['uploader']}",
                "licence": OK_FSD_LICENCES[c["license"]], "fsd50k_labels": c["labels"], "fsd50k_split": "dev",
                "attribution": f'"{c["title"]}" by {c["uploader"]} (freesound.org/s/{fid}), {OK_FSD_LICENCES[c["license"]]}',
                "label_provenance": "FSD50K dev ground truth (human-verified, smeared to ontology ancestors)"}

    def question(cls):
        if rng.random() < 0.5:
            return rng.choice(L2.Q_PRES_NAME).format(name=onto.official[cls]), "official_name"
        return FL.presence_question(rng.choice(L2.Q_PRES_NP), cls), "noun_phrase"

    def yes_class(fid):
        c = clips[fid]
        leaves = [l for l in c["leaves"] if FL.CLASSES[l][3]]
        anyl = [l for l in c["labels"] if FL.CLASSES[l][3]]
        pool = leaves if (leaves and rng.random() < 0.8) else anyl
        return rng.choice(sorted(pool)) if pool else None

    def ask(fid, ans, near_flag):
        c = clips[fid]
        if ans == "yes":
            cls, kind = yes_class(fid), "positive"
        else:
            cls, kind = L2.pick_negative(c["labels"], c["leaves"], tax, onto, cooc, rng, want_near=near_flag)
        if cls is None:
            return False
        q, style = question(cls)
        mk.noul(split=split_of[fid], source="fsd50k", family="presence", answer=ans, question=q,
                audio={"bank": None, "key": f"fsd_{fid}"},
                meta={**meta(fid), "asked_class": cls, "asked_official_name": onto.official[cls],
                      "negative_kind": kind, "question_style": style})
        stats[f"{ans}_{kind}"] += 1
        return True

    order = list(sel)
    rng.shuffle(order)
    paired = set(order[:n_pair])
    bal = collections.Counter()

    def want_near():  # keep the near share of "no" answers at NEAR_FRAC
        near = sum(v for k, v in stats.items() if k in ("no_sibling", "no_confusable", "no_broad_sibling", "no_cousin"))
        return near < NEAR_FRAC * (near + stats["no_random"] + 1)

    for fid in order:
        if fid in paired:  # the same clip asked twice: one present class, one (mostly near) absent class
            if ask(fid, "no", want_near()):
                bal["no"] += 1
                if ask(fid, "yes", False):
                    bal["yes"] += 1
        else:
            ans = "yes" if bal["yes"] <= bal["no"] else "no"
            if ask(fid, ans, want_near()):
                bal[ans] += 1
    # which-of-these choices (all distractors near)
    n_w = 0
    cand = [f for f in order if mk.room(f"fsd_{f}")]
    for fid in cand:
        if n_w >= n_which:
            break
        c = clips[fid]
        golds = [l for l in c["leaves"] if FL.CLASSES[l][3] and FL.CLASSES[l][4]]
        if not golds:
            continue
        gold = rng.choice(sorted(golds))
        official = rng.random() < 0.5
        disp = (lambda x: onto.official[x]) if official else (lambda x: FL.CLASSES[x][0])
        k = rng.choice([3, 3, 4])
        dis = L2.near_distractors(c["labels"], c["leaves"], gold, k, tax, onto, cooc, rng, disp)
        if dis is None:
            stats["which_skip_no_near"] += 1
            continue
        mk.add(split=split_of[fid], source="fsd50k", family="which_near", qtype="choice",
               question=rng.choice(L2.Q_WHICH), options=[disp(gold)] + [disp(d) for d, _ in dis], gold=0,
               audio={"bank": None, "key": f"fsd_{fid}"},
               meta={**meta(fid), "answer_class": gold, "distractor_classes": [d for d, _ in dis],
                     "distractor_kinds": [k2 for _, k2 in dis], "option_style": "official" if official else "friendly"})
        n_w += 1
    stats["which"] = n_w
    return dict(stats)


def fma_items(mk: ItemMaker, tracks: list[dict], audio: dict, tree: L2.GenreTree, split_of_artist, rng, quota) -> dict:
    stats = collections.Counter()
    by_key = {t["key"]: t for t in tracks}

    def meta(t):
        return {"source_id": f"fma:{t['track_id']}", "fma_track_id": t["track_id"],
                "speaker": f"fma_artist:{t['artist_id']}", "artist": t["artist"], "title": t["title"],
                "licence": t["licence_class"], "licence_url": t["licence_url"],
                "share_alike": t["licence_class"] == "CC-BY-SA", "fma_url": t["url"],
                "fma_genres": [tree.title[g] for g in t["genres_all"] if g in tree.title],
                "attribution": f'"{t["title"]}" by {t["artist"]} (freemusicarchive.org, {t["licence_url"]})'}

    def ref(t):
        return {"bank": None, "key": t["key"]}

    def split(t):
        return split_of_artist[t["artist_id"]]

    def order(pred):
        c = [t for t in tracks if pred(t) and mk.room(t["key"])]
        rng.shuffle(c)
        c.sort(key=lambda t: mk.n_per_clip[t["key"]])
        return c

    gtitle = tree.title
    # top-level genre (choice over sibling top-level genres)
    n = 0
    for t in order(lambda t: t["music_ok"] and any(gtitle.get(tree.root(g)) in L2.TOP_GENRES for g in t["genres_all"])):
        if n >= quota["genre_top"]:
            break
        gall = set(t["genres_all"])
        roots = sorted({tree.root(g) for g in gall if gtitle.get(tree.root(g)) in L2.TOP_GENRES})
        gold = rng.choice(roots)
        dis = [tree.id_of[r] for r in L2.TOP_GENRES if L2.genre_valid_negative(tree.id_of[r], gall, tree)]
        if len(dis) < 2:
            continue
        rng.shuffle(dis)
        k = min(len(dis), rng.randint(3, 5))
        opts = [L2.GENRE_DISPLAY[gtitle[gold]]] + [L2.GENRE_DISPLAY[gtitle[d]] for d in dis[:k]]
        mk.add(split=split(t), source="fma", family="genre_top", qtype="choice", question=rng.choice(L2.Q_GENRE_TOP),
               options=opts, gold=0, audio=ref(t), meta={**meta(t), "answer_genre": gtitle[gold],
                                                          "distractor_genres": [gtitle[d] for d in dis[:k]]})
        n += 1
    stats["genre_top"] = n
    # sub-genre among siblings
    n = 0
    for t in order(lambda t: t["music_ok"]):
        if n >= quota["genre_sub"]:
            break
        gall = set(t["genres_all"])
        subs = [g for g in t["genres"] if tree.parent.get(g, 0) and gtitle[g] in L2.GENRE_DISPLAY]
        rng.shuffle(subs)
        made = False
        for g in subs:
            sib = [s for s in tree.siblings(g) if L2.genre_valid_negative(s, gall, tree)]
            if len(sib) < 2:  # fall back to the parent's siblings within the same top-level genre
                par = tree.parent[g]
                sib += [s for s in tree.siblings(par) if tree.parent.get(par, 0) and
                        L2.genre_valid_negative(s, gall, tree) and s not in sib]
            if len(sib) < 2:
                continue
            rng.shuffle(sib)
            k = min(len(sib), rng.randint(2, 4))
            par_t = gtitle[tree.root(g)]
            q = rng.choice(L2.Q_GENRE_SUB).format(parent=L2.GENRE_DISPLAY.get(par_t, par_t.lower()))
            mk.add(split=split(t), source="fma", family="genre_sub", qtype="choice", question=q,
                   options=[L2.GENRE_DISPLAY[gtitle[g]]] + [L2.GENRE_DISPLAY[gtitle[s]] for s in sib[:k]], gold=0,
                   audio=ref(t), meta={**meta(t), "answer_genre": gtitle[g], "distractor_genres": [gtitle[s] for s in sib[:k]]})
            made = True
            break
        n += made
    stats["genre_sub"] = n
    # genre yes/no (80% near negatives)
    bal = collections.Counter()
    for t in order(lambda t: t["music_ok"]):
        if bal["yes"] + bal["no"] >= quota["genre_noul"]:
            break
        gall = set(t["genres_all"])
        ans = "yes" if bal["yes"] <= bal["no"] else "no"
        if ans == "yes":
            pool = [g for g in gall if gtitle.get(g) in L2.GENRE_DISPLAY]
            if not pool:
                continue
            g, kind = rng.choice(sorted(pool)), "positive"
        else:
            near = L2.genre_near(gall, tree)
            if near and bal["no_near"] <= NEAR_FRAC * (bal["no"] + 1):
                g, kind = rng.choice(near), "sibling"
            else:
                rnd = [x for x in L2.genre_random(gall, tree) if x not in near]
                if not rnd:
                    continue
                g, kind = rng.choice(rnd), "random"
            bal["no_near"] += kind == "sibling"
        mk.noul(split=split(t), source="fma", family="genre_noul", answer=ans,
                question=rng.choice(L2.Q_GENRE_NOUL).format(g=L2.GENRE_DISPLAY[gtitle[g]]), audio=ref(t),
                meta={**meta(t), "asked_genre": gtitle[g], "negative_kind": kind})
        bal[ans] += 1
    stats.update({f"genre_noul_{k}": v for k, v in bal.items()})
    # tempo (measured)
    tc = collections.Counter()
    cands = [(t, L2.tempo_class(t["tempo"])) for t in order(lambda t: t["music_ok"])]
    cands = [(t, c) for t, c in cands if c]
    stats.update({f"tempo_available_{c}": n for c, n in collections.Counter(c for _, c in cands).items()})
    per = quota["tempo"] // 3
    for t, c in cands:
        if tc[c] >= per or not mk.room(t["key"]):
            continue
        mk.add(split=split(t), source="fma", family="tempo", qtype="score", question=rng.choice(L2.Q_TEMPO),
               options=list(L2.TEMPO_LEVELS), gold=L2.TEMPO_LEVELS.index(c), audio=ref(t),
               meta={**meta(t), "bpm": round(t["tempo"]["bpm"], 1), "tempo_estimate": t["tempo"],
                     "label_rule": "measured BPM (onset-envelope autocorrelation, 120 BPM log-normal prior, both halves "
                                   "agree within 5%, octave ambiguity <= 0.8): slow 40-82, medium 95-125, fast 140-200"})
        tc[c] += 1
    stats.update({f"tempo_{k}": v for k, v in tc.items()})
    # level (by construction, measured afterwards)
    lc = collections.Counter()
    for t in order(lambda t: t["music_ok"]):
        if sum(lc.values()) >= quota["level"]:
            break
        x = audio[t["key"]]
        fl = FL.feasible_levels(x)
        if not fl or FL.active_rms_db(x) < -60:
            continue
        lev = min(fl, key=lambda z: (lc[z], FL.LEVELS.index(z)))
        lo, hi = FL.LEVEL_RANGES_DB[lev]
        y = x.astype(np.float64) * 10 ** ((rng.uniform(lo, hi) - FL.active_rms_db(x)) / 20)
        pk = float(np.max(np.abs(y)))
        if pk > 0.99:
            y *= 0.99 / pk
        audio[t["key"]] = y.astype(np.float32)
        lc[lev] += 1
        mk.add(split=split(t), source="fma_signal", family="level", qtype="score", question=rng.choice(L2.Q_LEVEL),
               options=list(FL.LEVELS), gold=FL.LEVELS.index(lev), audio=ref(t),
               meta={**meta(t), "active_rms_dbfs": round(FL.active_rms_db(audio[t["key"]]), 2),
                     "label_rule": "excerpt scaled into level band " + str(FL.LEVEL_RANGES_DB[lev])})
    stats.update({f"level_{k}": v for k, v in lc.items()})
    # live vs studio
    def live_ans(t):
        txt = " ".join(str(x).lower() for x in [t["title"], t["album_title"]] + t["tags"])
        if t["album_type"] == "Live Performance":
            return "yes"
        if t["album_type"] == "Album" and "live" not in txt and "session" not in txt and "concert" not in txt:
            return "no"
        return None

    yes = [t for t in order(lambda t: t["music_ok"] and live_ans(t) == "yes")]
    no = [t for t in order(lambda t: t["music_ok"] and live_ans(t) == "no")]
    k = min(len(yes), len(no), quota["live"] // 2)
    for t, ans in [(t, "yes") for t in yes[:k]] + [(t, "no") for t in no[:k]]:
        mk.noul(split=split(t), source="fma", family="live", answer=ans, question=rng.choice(L2.Q_LIVE), audio=ref(t),
                meta={**meta(t), "album_type": t["album_type"],
                      "label_rule": "yes = FMA album type 'Live Performance'; no = album type 'Album' with no "
                                    "live/session/concert in title, album title or tags"})
    stats["live_each"] = k
    return dict(stats)


def openmic_items(mk: ItemMaker, clips: list[dict], split_of_artist, rng, quota) -> dict:
    stats = collections.Counter()

    def meta(c):
        return {"source_id": f"openmic:{c['sample_key']}", "openmic_sample_key": c["sample_key"],
                "fma_track_id": c["track_id"], "speaker": f"fma_artist:{c['artist_id']}", "artist": c["artist"],
                "licence": c["licence_class"], "licence_url": c["licence_url"],
                "share_alike": c["licence_class"] == "CC-BY-SA", "instruments_present": c["present"],
                "instruments_verified_absent": c["absent"],
                "label_provenance": f"OpenMIC-2018 aggregated crowd relevance (present >= {L2.POS_REL}, "
                                    f"verified absent <= {L2.NEG_REL})"}

    def ref(c):
        return {"bank": None, "key": f"om_{c['sample_key']}"}

    def split(c):
        return split_of_artist[c["artist_id"]]

    def order(pred):
        cc = [c for c in clips if pred(c) and mk.room(ref(c)["key"])]
        rng.shuffle(cc)
        cc.sort(key=lambda c: mk.n_per_clip[ref(c)["key"]])
        return cc

    # instruments yes/no
    bal = collections.Counter()
    for c in order(lambda c: (set(c["present"]) | set(c["absent"])) - {"voice"}):
        if bal["yes"] + bal["no"] >= quota["instr_noul"]:
            break
        pos = sorted(set(c["present"]) - {"voice"})
        neg = sorted(set(c["absent"]) - {"voice"})
        want = "yes" if bal["yes"] <= bal["no"] else "no"
        ans = want if (pos if want == "yes" else neg) else ("no" if want == "yes" and neg else "yes" if pos else None)
        if ans is None:
            continue
        if ans == "yes":
            ins, kind = rng.choice(pos), "positive"
        else:
            near = [i for i in neg if any(i in L2.instr_near(p) for p in pos)]
            if near and bal["no_near"] <= NEAR_FRAC * (bal["no"] + 1):
                ins, kind = rng.choice(near), "near"
            else:
                ins, kind = rng.choice(neg), "verified_absent"
            bal["no_near"] += kind == "near"
        mk.noul(split=split(c), source="openmic", family="instr_noul", answer=ans,
                question=rng.choice(L2.Q_INSTR).format(np=L2.INSTRUMENTS[ins][1]), audio=ref(c),
                meta={**meta(c), "asked_instrument": ins, "negative_kind": kind}, allow_phone=ins != "cymbals")
        bal[ans] += 1
    stats.update({f"instr_noul_{k}": v for k, v in bal.items()})
    # which instrument (gold present, distractors verified absent)
    n = 0
    for c in order(lambda c: len(set(c["present"]) - {"voice"}) >= 1 and len(set(c["absent"]) - {"voice"}) >= 2):
        if n >= quota["instr_which"]:
            break
        gold = rng.choice(sorted(set(c["present"]) - {"voice"}))
        neg = sorted(set(c["absent"]) - {"voice"})
        rng.shuffle(neg)
        neg.sort(key=lambda i: 0 if i in L2.instr_near(gold) else 1)
        k = min(len(neg), rng.choice([2, 3]))
        mk.add(split=split(c), source="openmic", family="instr_which", qtype="choice", question=rng.choice(L2.Q_INSTR_WHICH),
               options=[L2.INSTRUMENTS[gold][0]] + [L2.INSTRUMENTS[i][0] for i in neg[:k]], gold=0, audio=ref(c),
               meta={**meta(c), "answer_instrument": gold, "distractor_instruments": neg[:k]})
        n += 1
    stats["instr_which"] = n
    # vocals / instrumental (both polarities) and vocal-free use
    def voice(c):
        return "yes" if "voice" in c["present"] else "no" if "voice" in c["absent"] else None

    for fam, q_yes_if_voice, qs, qn in (("vocals", True, None, quota["vocals"]), ("use_bed", False, L2.Q_USE_BED, quota["use_bed"])):
        b = collections.Counter()
        for c in order(lambda c: voice(c) is not None):
            if b["yes"] + b["no"] >= qn:
                break
            v = voice(c)
            if fam == "vocals":
                inverted = rng.random() < 0.5
                q = rng.choice(L2.Q_INSTRUMENTAL if inverted else L2.Q_VOCALS)
                ans = ("no" if v == "yes" else "yes") if inverted else v
            else:
                q, ans = rng.choice(qs), ("no" if v == "yes" else "yes")
                inverted = None
            if b[ans] > b["no" if ans == "yes" else "yes"]:
                continue
            mk.noul(split=split(c), source="openmic", family=fam, answer=ans, question=q, audio=ref(c),
                    meta={**meta(c), "voice_present": v == "yes", "question_polarity": "instrumental" if inverted else "vocals",
                          "label_rule": "OpenMIC 'voice' label" + ("; suitable iff no singing voice" if fam == "use_bed" else "")})
            b[ans] += 1
        stats.update({f"{fam}_{k}": v for k, v in b.items()})
    return dict(stats)


# 6. Checks, leakage, summary
def validate(out: Path, items: list[dict]) -> dict:
    keys_needed = {"id", "set", "source", "question", "options", "target", "gold", "audio", "channel", "state_text", "meta"}
    errs = collections.Counter()
    banks, durs, ids = {}, [], set()
    for it in items:
        if set(it) != keys_needed:
            errs["schema"] += 1
        if it["id"] in ids:
            errs["dup_id"] += 1
        ids.add(it["id"])
        k = len(it["options"])
        if not 2 <= k <= 20 or len(it["target"]) != k:
            errs["options_len"] += 1
        if abs(sum(it["target"]) - 1) > 1e-5:
            errs["target_sum"] += 1
        if int(np.argmax(it["target"])) != it["gold"]:
            errs["gold_argmax"] += 1
        if len(set(it["options"])) != k:
            errs["dup_options"] += 1
        qt = it["meta"].get("qtype")
        if qt not in ("choice", "noul", "score"):
            errs["qtype"] += 1
        if qt == "noul" and sorted(it["options"]) != ["no", "yes"]:
            errs["noul_options"] += 1
        if qt == "score" and it["options"] not in (FL.LEVELS, L2.TEMPO_LEVELS):
            errs["score_order"] += 1
        if it["channel"] not in ("clean", "phone"):
            errs["channel"] += 1
        b = it["audio"]["bank"]
        if b not in banks:
            banks[b] = common.AudioBank(out / "audio", b)
        if it["audio"]["key"] not in banks[b].index:
            errs["missing_audio_key"] += 1
            continue
        durs.append(banks[b].index[it["audio"]["key"]][1] / SR)
    rng = random.Random(1)
    for it in rng.sample(items, min(400, len(items))):
        x = banks[it["audio"]["bank"]].get(it["audio"]["key"])
        if not np.isfinite(x).all() or np.max(np.abs(x)) > 1.0 or len(x) < 0.5 * SR or len(x) > 20 * SR:
            errs["bad_audio"] += 1
        if it["channel"] == "phone" and len(common.phone_channel(x)) != len(x):
            errs["phone_len"] += 1
    d = np.array(durs)
    bal = collections.defaultdict(collections.Counter)
    for it in items:
        bal[f"{it['source']}/{it['meta']['family']}"][it["options"][it["gold"]]] += 1
    per_clip = collections.Counter((it["audio"]["bank"], it["audio"]["key"]) for it in items)
    return {"ok": not errs, "errors": dict(errs), "n_items": len(items),
            "duration_s": {"min": float(d.min()), "p50": float(np.median(d)), "max": float(d.max()), "mean": float(d.mean())},
            "max_items_per_clip": max(per_clip.values()),
            "answer_balance": {k: dict(v) if len(v) <= 6 else {"n_answers": len(v), "top": dict(v.most_common(6))}
                               for k, v in sorted(bal.items())}}


def _meta_strings(obj, out: set):
    if isinstance(obj, dict):
        for v in obj.values():
            _meta_strings(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _meta_strings(v, out)
    elif obj is not None:
        out.add(str(obj))


def sdd_near_dup_scan(music_audio: dict, bank) -> dict:
    """Normalised cross-correlation of 20 ms level envelopes (dB, floored at -80) of every music excerpt against every
    track in `bank` (MuChoMusic / SDD audio of our evaluation set), at every offset (FFT). Flags max NCC >= 0.95."""
    env = lambda x: np.maximum(FL.frame_rms_db(x), -80.0)  # noqa: E731
    keys = sorted(music_audio)
    envs = [env(music_audio[k]) for k in keys]
    hits, worst, n_pairs = [], 0.0, 0
    by_len = collections.defaultdict(list)
    for i, e in enumerate(envs):
        by_len[len(e)].append(i)
    for rk in sorted(bank.index):
        r = env(bank.get(rk)).astype(np.float64)
        for n, idx in by_len.items():
            if len(r) < n or n < 50:
                continue
            E = np.stack([envs[i] for i in idx]).astype(np.float64)
            E = (E - E.mean(1, keepdims=True)) / (E.std(1, keepdims=True) + 1e-9)
            L = 1 << int(math.ceil(math.log2(len(r) + n)))
            corr = np.fft.irfft(np.fft.rfft(E[:, ::-1], L) * np.fft.rfft(r, L)[None, :], L)[:, n - 1: len(r)]
            cs = np.concatenate([[0.0], np.cumsum(r)])
            cs2 = np.concatenate([[0.0], np.cumsum(r * r)])
            mu = (cs[n:] - cs[:-n]) / n
            sd = np.sqrt(np.maximum((cs2[n:] - cs2[:-n]) / n - mu ** 2, 1e-9))
            ncc = corr / (n * sd[None, :])
            mx = ncc.max(1)
            n_pairs += len(idx)
            worst = max(worst, float(mx.max()))
            for j in np.nonzero(mx >= 0.95)[0]:
                hits.append((keys[idx[j]], rk, round(float(mx[j]), 3)))
    return {"n_sdd_tracks": len(bank.index), "n_excerpts": len(keys), "pairs": n_pairs, "hits": hits[:50],
            "n_hits": len(hits), "max_ncc": round(worst, 3), "clean": not hits}


def leakage_check(items: list[dict], raw: Path, music_audio: dict | None = None) -> dict:
    """Checks against our evaluation items: source ids, FSD numeric ids, uploader / artist groups, exact question texts,
    and an audio near-duplicate scan of our music excerpts against the evaluation set's MuChoMusic audio."""
    ours_ids, ours_groups, fsd_ids, questions = set(), set(), set(), set()
    for it in items:
        m = it["meta"]
        questions.add(it["question"].strip().lower())
        if m.get("source_id"):
            ours_ids.add(m["source_id"])
        if m.get("freesound_id") is not None:
            fsd_ids.add(str(m["freesound_id"]))
        sp = m.get("speaker")
        if sp:
            ours_groups.add(str(sp).split(":", 1)[-1])
    res = {}
    eval_ids = {r["fname"] for r in csv.DictReader(open(raw / "fsd_meta/labels/eval.csv"))}
    res["fsd50k_eval_clips_used"] = len(fsd_ids & eval_ids)
    for name, path in (("adi", "/vol/adi/items.jsonl"), ("exp0_eval", "/vol/data/eval_items.jsonl")):
        if not os.path.exists(path):
            res[name] = "not present"
            continue
        their, fsd_like, their_q, uploaders, sdd_tracks = set(), set(), set(), set(), set()
        for line in open(path):
            if not line.strip():
                continue
            it = json.loads(line)
            s = set()
            _meta_strings(it.get("meta", {}), s)
            if it.get("audio"):
                s.add(str(it["audio"].get("key")))
            their |= s
            their_q.add(str(it.get("question", "")).strip().lower())
            js = json.dumps(it.get("meta", {})).lower()
            if "fsd" in js or "fsd" in str(it.get("source", "")).lower() or "fsd" in str(it.get("set", "")).lower():
                fsd_like |= s
                up = (it.get("meta") or {}).get("uploader")
                if up:
                    uploaders.add(str(up))
            if (it.get("meta") or {}).get("sdd_track"):
                sdd_tracks.add(str(it["meta"]["sdd_track"]))
        id_hits = sorted(x for x in ours_ids if x in their)
        num_hits = sorted(x for x in fsd_ids if x in fsd_like)
        grp_hits = sorted(x for x in ours_groups if x in their and len(x) > 3)
        q_hits = sorted(q for q in questions if q in their_q)
        res[name] = {"n_their_meta_strings": len(their), "source_id_hits": id_hits[:50], "fsd_numeric_id_hits": num_hits[:50],
                     "group_hits": grp_hits[:50], "fsd_uploader_hits": sorted(
                         x for x in ours_groups if x in uploaders)[:50],
                     "exact_question_text_hits": q_hits[:50], "n_exact_question_text_hits": len(q_hits),
                     "adi_sdd_tracks": len(sdd_tracks),
                     "clean": not (id_hits or num_hits or grp_hits)}
    # audio near-duplicates vs the evaluation set's MuChoMusic (SDD) tracks
    sdd = Path("/vol/adi/audio/sdd.npy")
    if music_audio and sdd.exists():
        res["sdd_audio_near_duplicates"] = sdd_near_dup_scan(music_audio, common.AudioBank("/vol/adi/audio", "sdd"))
    else:
        res["sdd_audio_near_duplicates"] = "evaluation sdd bank not found"
    return res


def make_summary(train, cal, bank_seconds, checks, leakage, smoke, secs, **extra):
    allit = train + cal
    by = lambda f: dict(sorted(collections.Counter(f(it) for it in allit).items()))  # noqa: E731
    examples = {}
    for src in sorted({it["source"] for it in allit}):
        rng = random.Random(7)
        its = [it for it in allit if it["source"] == src]
        examples[src] = [{"question": it["question"], "options": it["options"], "gold": it["gold"],
                          "answer": it["options"][it["gold"]], "family": it["meta"]["family"], "audio": it["audio"]}
                         for it in rng.sample(its, min(3, len(its)))]
    return {
        "builder": BUILDER, "smoke": smoke, "build_seconds": round(secs),
        "counts": {"total": len(allit), "train": len(train), "CAL": len(cal),
                   "by_source": by(lambda it: it["source"]),
                   "by_source_split": by(lambda it: f"{it['source']}/{it['set']}"),
                   "by_qtype": by(lambda it: it["meta"]["qtype"]),
                   "by_family": by(lambda it: f"{it['source']}/{it['meta']['family']}"),
                   "by_channel": by(lambda it: it["channel"]),
                   "by_n_options": by(lambda it: len(it["options"])),
                   "negative_kinds": by(lambda it: f"{it['source']}/{it['meta']['family']}/{it['meta'].get('negative_kind')}"
                                        if it["options"][it["gold"]] == "no" else "positive_or_not_noul")},
        "audio_hours": {k: round(v / 3600, 3) for k, v in bank_seconds.items()},
        "audio_hours_total": round(sum(bank_seconds.values()) / 3600, 3),
        "sources": SOURCES_INFO,
        "revisions": {"fsd50k_meta": f"{FSD_META_REPO}@{FSD_META_REV}", "fsd50k_audio": f"{FSD_AUDIO_REPO}@{FSD_AUDIO_REV}",
                      "audioset_ontology": ONTOLOGY_URL, "fma_audio": f"{FMA_REPO}@{FMA_REV}", "fma_metadata": FMA_META_ZIP,
                      "openmic": f"zenodo 1432913 v1.0.0 md5 {OPENMIC_MD5}", "sdd_exclusion_list": SDD_LIC_URL},
        "extra": extra, "checks": checks, "leakage": leakage, "examples": examples,
    }


# 7. The build
def build(raw: Path, work: Path, out: Path, fma_ok: dict, tree: L2.GenreTree, fma_stats: dict, smoke: bool = False) -> dict:
    t0 = time.time()
    rng = random.Random(common.SEED)
    scale = 0.05 if smoke else 1.0

    # FSD50K
    vocab, ontology, clips, eval_ids = load_fsd_meta(raw)
    tax = FL.Taxonomy(vocab, ontology)
    FL.validate_class_table(sorted(vocab.values()))
    onto = L2.FullOntology(vocab, ontology)
    bad = sorted({c for g in L2.SEMANTIC_BLOCK for c in g} - set(vocab.values()))
    assert not bad, bad
    files = sorted(glob.glob(str(raw / "fsd_audio/data/*.parquet")))
    pidx = read_parquet_index(files)
    cooc = FL.cooccurrence([c["labels"] for c in clips.values()])
    used = sb_sounds_used_ids()
    n_clips = int(N_FSD_CLIPS * scale)
    sel, sel_stats = select_fsd(clips, pidx, tax, n_clips, rng, used)
    sel_stats["sb_sounds_used_clip_ids"] = len(used)
    cal_up = choose_cal_groups({f: clips[f]["uploader"] for f in sel}, CAL_FRAC, rng, max_group=max(5, n_clips // 300))
    split_fsd = {f: ("CAL" if clips[f]["uploader"] in cal_up else "train") for f in sel}
    fsd_audio = decode_fsd(pidx, sel)
    sel = [f for f in sel if f in fsd_audio and len(fsd_audio[f]) >= int(0.5 * SR)]
    log("fsd decoded", len(sel))
    import shutil

    shutil.rmtree(raw / "fsd_audio", ignore_errors=True)  # free container disk before the banks are written
    mk = ItemMaker(rng)
    fsd_stats = fsd_items(mk, sel, clips, split_fsd, tax, onto, cooc, rng, int(len(sel) * PAIR_FRAC),
                          int(N_FSD_WHICH * scale))
    log("fsd items", len(mk.items), fsd_stats)

    # FMA
    url2tid = {v["url"]: k for k, v in fma_ok.items() if v["url"]}
    an2tid = collections.defaultdict(list)
    for k, v in fma_ok.items():
        an2tid[(L2.norm_name(v["artist"]), L2.norm_name(v["title"]))].append(k)
    tracks, fma_audio = [], {}
    jstats = collections.Counter()
    per_artist = collections.Counter()
    shard_files = sorted(glob.glob(str(work / "fma" / "shard_*.json")))
    if smoke:
        shard_files = shard_files[:3]
    for jf in shard_files:
        d = json.loads(open(jf).read())
        npz = np.load(jf.replace(".json", ".npz"))
        for m in d["items"]:
            jstats["repack_rows_licence_ok"] += 1
            tid = url2tid.get(m["url"])
            if tid is None:
                c = an2tid.get((L2.norm_name(m["artist"]), L2.norm_name(m["title"])), [])
                tid = c[0] if len(c) == 1 else None
            if tid is None:
                jstats["drop_no_official_match_or_official_licence_not_ok"] += 1
                continue
            info = fma_ok[tid]
            if info["licence_class"] != m["mirror_licence_class"]:
                jstats["drop_licence_mismatch"] += 1
                continue
            if per_artist[info["artist_id"]] >= FMA_ARTIST_CAP:
                jstats["drop_artist_cap"] += 1
                continue
            if f"fma_{tid}" in fma_audio:
                jstats["drop_duplicate"] += 1
                continue
            per_artist[info["artist_id"]] += 1
            key = f"fma_{tid}"
            fma_audio[key] = np.asarray(npz[m["key"]], dtype=np.float32)
            tracks.append({**info, "key": key, "tempo": m["tempo"], "active_db": m["active_db"],
                           "mirror_licence": m["mirror_licence"]})
            jstats["kept"] += 1
    jstats["artists"] = len(per_artist)
    log("fma tracks", dict(jstats))

    # OpenMIC
    om = json.loads((work / "openmic" / "openmic.json").read_text())
    om_npz = np.load(work / "openmic" / "openmic.npz")
    om_clips = om["items"][: int(len(om["items"]) * scale) if smoke else None]
    om_audio = {f"om_{c['sample_key']}": np.asarray(om_npz[c["sample_key"]], dtype=np.float32) for c in om_clips}

    # music CAL split by artist (shared by FMA and OpenMIC)
    art = {}
    for t in tracks:
        art[t["key"]] = t["artist_id"]
    for c in om_clips:
        art[f"om_{c['sample_key']}"] = c["artist_id"]
    cal_art = choose_cal_groups({k: str(v) for k, v in art.items()}, CAL_FRAC, rng, max_group=60)
    split_of_artist = collections.defaultdict(lambda: "train")
    for a in {v for v in art.values()}:
        split_of_artist[a] = "CAL" if str(a) in cal_art else "train"

    quota_f = {k: max(2, int(v * scale)) for k, v in Q_FMA.items()}
    quota_o = {k: max(2, int(v * scale)) for k, v in Q_OM.items()}
    fma_s = fma_items(mk, tracks, fma_audio, tree, split_of_artist, rng, quota_f)
    log("fma items", fma_s)
    om_s = openmic_items(mk, om_clips, split_of_artist, rng, quota_o)
    log("openmic items", om_s)

    # banks
    out.mkdir(parents=True, exist_ok=True)
    adir = out / "audio"
    used_keys = {it["audio"]["key"] for it in mk.items}
    pools = [("fsd", {f"fsd_{f}": fsd_audio[f] for f in sel if f"fsd_{f}" in used_keys}),
             ("fma", {k: v for k, v in fma_audio.items() if k in used_keys}),
             ("openmic", {k: v for k, v in om_audio.items() if k in used_keys})]
    bank_of, bank_seconds = {}, {}
    for prefix, pool in pools:
        cur, cur_bytes, bi = {}, 0, 0
        chunks = []
        for k in sorted(pool):
            x = pool[k]
            if cur and cur_bytes + 4 * len(x) > 3.5e9:
                chunks.append(cur)
                cur, cur_bytes = {}, 0
            cur[k] = x
            cur_bytes += 4 * len(x)
        if cur:
            chunks.append(cur)
        for bi, ch in enumerate(chunks):
            name = f"{BUILDER}_{prefix}{bi}"
            common.write_audio_bank(adir, name, ch)
            for k in ch:
                bank_of[k] = name
            bank_seconds[name] = sum(len(v) for v in ch.values()) / SR
            log("wrote bank", name, len(ch), f"{bank_seconds[name] / 3600:.2f} h")
    for it in mk.items:
        it["audio"]["bank"] = bank_of[it["audio"]["key"]]
    train = [it for it in mk.items if it["set"] == "train"]
    cal = [it for it in mk.items if it["set"] == "CAL"]
    def clean(o):  # NaN / inf (e.g. from pandas) -> None, so the manifests are strict JSON
        if isinstance(o, float) and not math.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o

    n_nan = 0
    for it in mk.items:
        try:
            json.dumps(it, allow_nan=False)
        except ValueError:
            n_nan += 1
            it["meta"] = clean(it["meta"])
    log("items with NaN meta cleaned", n_nan)
    common.write_jsonl(out / "items_train.jsonl", train)
    common.write_jsonl(out / "items_cal.jsonl", cal)
    checks = validate(out, train + cal)
    # CAL / train group disjointness
    def groups(its):
        return {it["meta"].get("speaker") for it in its}
    checks["cal_train_group_overlap"] = sorted(groups(train) & groups(cal))[:20]
    checks["cal_frac"] = round(len(cal) / max(1, len(mk.items)), 4)
    music_keys = {k: v for k, v in list(fma_audio.items()) + list(om_audio.items()) if k in used_keys}
    leakage = leakage_check(train + cal, raw, music_keys if not smoke else dict(list(music_keys.items())[:50]))
    overlap = {"fsd_clips_used": len({it["meta"]["freesound_id"] for it in mk.items if it["source"] == "fsd50k"}),
               "fsd_clips_also_in_sb_sounds": len({str(it["meta"]["freesound_id"]) for it in mk.items
                                                    if it["source"] == "fsd50k"} & used)}
    summary = make_summary(train, cal, bank_seconds, checks, leakage, smoke, time.time() - t0,
                           fsd_selection=sel_stats, fsd_items=fsd_stats, sb_sounds_overlap=overlap,
                           fma_metadata=fma_stats, fma_join=dict(jstats), fma_items=fma_s, openmic_prep=om["stats"],
                           openmic_items=om_s, cal_uploaders=len(cal_up), cal_artists=len(cal_art),
                           fsd_clip_ids=sorted({int(it["meta"]["freesound_id"]) for it in mk.items if it["source"] == "fsd50k"}))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    log("done", len(train), "train", len(cal), "CAL", checks.get("ok"))
    return summary


# 8. Finalize: second-stage check of SDD near-duplicate flags, namespace-aware group check, conservative drop
def log_spec(x: np.ndarray) -> np.ndarray:
    n_fft, hop = 1024, 320
    if len(x) < n_fft:
        return np.zeros((1, 64))
    win = np.hanning(n_fft).astype(np.float32)
    n = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * win, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    edges = np.geomspace(50, 7600, 65)
    b = np.stack([spec[:, (freqs >= lo) & (freqs < hi)].sum(1) for lo, hi in zip(edges[:-1], edges[1:])], 1)
    return np.log10(b + 1e-10)


def pair_offset_and_spectral_corr(x: np.ndarray, r: np.ndarray) -> tuple[int, float, float]:
    """Best envelope-NCC offset of excerpt x inside reference r, that NCC, and the correlation of the two
    (band-mean-removed) log spectrograms at that offset. Same recording -> spectral corr ~1; different music -> low."""
    ex = np.maximum(FL.frame_rms_db(x), -80.0)
    er = np.maximum(FL.frame_rms_db(r), -80.0)
    n = len(ex)
    e = (ex - ex.mean()) / (ex.std() + 1e-9)
    c = np.correlate(er, e, mode="valid")
    cs = np.concatenate([[0.0], np.cumsum(er)])
    cs2 = np.concatenate([[0.0], np.cumsum(er * er)])
    mu = (cs[n:] - cs[:-n]) / n
    sd = np.sqrt(np.maximum((cs2[n:] - cs2[:-n]) / n - mu ** 2, 1e-9))
    ncc = c / (n * sd)
    off = int(np.argmax(ncc))
    seg = r[off * FL.FRAME: off * FL.FRAME + len(x)]
    a, b = log_spec(x), log_spec(seg)
    m = min(len(a), len(b))
    a, b = a[:m], b[:m]
    # spectral shape only: drop frames > 50 dB below either side's loudest, remove each frame's overall level (so
    # fades cannot drive the score), then each band's mean
    la, lb = a.mean(1), b.mean(1)
    act = (la > la.max() - 5) & (lb > lb.max() - 5)
    if act.sum() < 20:
        return off, float(ncc[off]), 0.0
    a, b = a[act] - la[act, None], b[act] - lb[act, None]
    a, b = a - a.mean(0), b - b.mean(0)
    sc = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))
    return off, float(ncc[off]), sc


def finalize(out: Path) -> dict:
    """Re-check the written dataset: give every SDD near-duplicate flag a spectral second opinion and drop its items
    anyway, check namespace-aware group overlap with our evaluation set, re-validate, rewrite manifests and summary."""
    summary = json.loads((out / "summary.json").read_text())
    train = common.read_jsonl(out / "items_train.jsonl")
    cal = common.read_jsonl(out / "items_cal.jsonl")
    items = train + cal
    flags = summary["leakage"].get("sdd_audio_near_duplicates", {})
    flags = flags.get("hits", []) if isinstance(flags, dict) else []
    ref_bank = common.AudioBank("/vol/adi/audio", "sdd")
    key_bank = {}
    for jf in sorted((out / "audio").glob("*.json")):  # every stored clip, including ones whose items were dropped
        for k in json.loads(jf.read_text()):
            key_bank[k] = jf.stem
    banks = {}
    second = []
    for key, rk, ncc in flags:
        b = key_bank.get(key)
        if b is None:
            continue
        if b not in banks:
            banks[b] = common.AudioBank(out / "audio", b)
        off, ncc2, sc = pair_offset_and_spectral_corr(banks[b].get(key), ref_bank.get(rk))
        second.append({"key": key, "sdd_track": rk, "envelope_ncc": round(ncc2, 3), "offset_s": round(off * 0.02, 2),
                       "spectral_corr": round(sc, 3), "same_recording": sc >= 0.9})
    drop_keys = {f[0] for f in flags}
    kept = [it for it in items if it["audio"]["key"] not in drop_keys]
    n_dropped = len(items) - len(kept) + int(summary["extra"].get("finalize", {}).get("dropped_items_sdd_flags", 0))
    # namespace-aware groups: our FSD uploaders vs the evaluation set's fsd uploader groups (FMA is not in the
    # evaluation set; SDD artists were excluded by name at build time)
    adi_groups, adi_fsd_ids = set(), set()
    for line in open("/vol/adi/items.jsonl"):
        it = json.loads(line)
        for g in it.get("groups") or []:
            adi_groups.add(str(g))
        m = it.get("meta") or {}
        if m.get("freesound_id") is not None:
            adi_fsd_ids.add(str(m["freesound_id"]))
    ours_up = {"fsd_uploader:" + str(it["meta"]["speaker"]).split(":", 1)[1] for it in kept if it["source"] == "fsd50k"}
    ours_fsd = {str(it["meta"]["freesound_id"]) for it in kept if it["source"] == "fsd50k"}
    lk = summary["leakage"]
    lk["namespaced"] = {
        "fsd_uploader_overlap_with_adi_groups": sorted(ours_up & adi_groups)[:50],
        "fsd_clip_id_overlap_with_adi": sorted(ours_fsd & adi_fsd_ids)[:50],
        "adi_group_namespaces": sorted({g.split(":")[0] for g in adi_groups}),
        "note": "leakage.adi.group_hits / exp0_eval.group_hits are cross-namespace numeric/string collisions "
                "(FMA artist ids vs CREMA-D actor / speechocean speaker / Freesound clip ids; a Freesound uploader "
                "named 'Vehicle' vs an FSD class name); the namespaced checks above are the real ones.",
    }
    lk["sdd_audio_second_stage"] = {"pairs": second, "n_same_recording": sum(s["same_recording"] for s in second),
                                    "dropped_audio_keys": sorted(drop_keys), "dropped_items": n_dropped}
    lk["clean"] = (not lk["namespaced"]["fsd_uploader_overlap_with_adi_groups"]
                   and not lk["namespaced"]["fsd_clip_id_overlap_with_adi"] and lk["fsd50k_eval_clips_used"] == 0
                   and not lk["adi"]["source_id_hits"] and not lk["adi"]["fsd_numeric_id_hits"]
                   and lk["adi"]["n_exact_question_text_hits"] == 0)
    train = [it for it in kept if it["set"] == "train"]
    cal = [it for it in kept if it["set"] == "CAL"]
    common.write_jsonl(out / "items_train.jsonl", train)
    common.write_jsonl(out / "items_cal.jsonl", cal)
    checks = validate(out, train + cal)
    g = lambda its: {it["meta"].get("speaker") for it in its}  # noqa: E731
    checks["cal_train_group_overlap"] = sorted(g(train) & g(cal))[:20]
    checks["cal_frac"] = round(len(cal) / max(1, len(kept)), 4)
    secs = {k: v * 3600 for k, v in summary["audio_hours"].items()}
    new = make_summary(train, cal, secs, checks, lk, summary["smoke"], summary["build_seconds"], **summary["extra"])
    new["extra"]["finalize"] = {"dropped_items_sdd_flags": n_dropped}
    (out / "summary.json").write_text(json.dumps(new, indent=1))
    return {"dropped": n_dropped, "second_stage": second, "namespaced": lk["namespaced"], "checks_ok": checks["ok"],
            "counts": new["counts"]["total"], "clean": lk["clean"]}
