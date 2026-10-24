"""sb_sounds build: sound events and categories, music, mixtures, signal-level facts and NSynth notes.

fetch_all() downloads the pinned sources to container disk; build() selects clips, makes items and writes the output.
"""

from __future__ import annotations

import collections
import csv
import glob
import io
import json
import math
import os
import random
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import labels as L  # noqa: E402

BUILDER = "sb_sounds"
SR = common.SAMPLE_RATE

FSD_META_REPO, FSD_META_REV = "Fhrozen/FSD50k", "ccf1acaff12f3f4a4c10052dddafbb6d22152c9b"
FSD_AUDIO_REPO, FSD_AUDIO_REV = "quinnlue/FSD50K-16k", "2a60d475f4e2f0db624a902881af2df79d656145"
ONTOLOGY_URL = "https://raw.githubusercontent.com/audioset/ontology/d417d32bf59c711abb5910fd2f76a0eb44697991/ontology.json"
NSYNTH_URLS = {"valid": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz",
               "test": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz"}
LIBRI_URLS = ["https://www.openslr.org/resources/12/dev-clean.tar.gz",
              "https://us.openslr.org/resources/12/dev-clean.tar.gz",
              "https://openslr.trmal.net/resources/12/dev-clean.tar.gz"]

OK_LICENCES = {"http://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
               "http://creativecommons.org/licenses/by/3.0/": "CC-BY-3.0"}

SOURCES_INFO = {
    "fsd50k": {"licence": "per clip CC0-1.0 or CC-BY-3.0 (FSD50K dataset CC-BY-4.0); CC-BY-NC and Sampling+ clips dropped",
               "origin": f"hf:{FSD_AUDIO_REPO}@{FSD_AUDIO_REV} (16 kHz FLAC repack) + labels/licences from "
                         f"hf:{FSD_META_REPO}@{FSD_META_REV}; FSD50K dev split only (Zenodo 4060432)"},
    "fsd50k_mix": {"licence": "components as fsd50k; speech from LibriSpeech dev-clean CC-BY-4.0",
                   "origin": "synthetic mixtures of fsd50k dev clips (+ LibriSpeech dev-clean, openslr.org/12)"},
    "fsd50k_signal": {"licence": "as fsd50k", "origin": "fsd50k dev clips, gain / silence applied by construction"},
    "nsynth": {"licence": "CC-BY-4.0", "origin": "NSynth valid+test jsonwav (download.magenta.tensorflow.org)"},
    "audioset_ontology": {"licence": "CC-BY-SA-4.0 (used only for the class hierarchy, not as training content)",
                          "origin": ONTOLOGY_URL},
}

# targets (full build); smoke build divides them
N_FSD = 16000
QUOTA = {"ident": 5000, "category": 4200, "music": 2600, "level": 2400}
N_MIX = 2200          # two-sound mixtures (-> count "two" + presence)
N_OVERLAY = 1500      # speech under a sound (-> "is someone talking?" yes)
N_TRAIL = 2000        # trailing-silence constructs (half yes, half no)
N_NSYNTH = 3000       # notes (2 items each)
PHONE_FRAC = 0.10
CAL_FRAC = 0.03
MAX_ITEMS_PER_CLIP = 3


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class NoShuffle(random.Random):
    """rng whose shuffle is a no-op: keeps score levels in their natural order inside make_item."""

    def shuffle(self, x, *args, **kwargs):  # noqa: D401
        return None


# 1. Downloads
def _download(url: str, dst: Path, timeout: int = 900) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    tmp = dst.with_suffix(dst.suffix + ".part")
    subprocess.run(["curl", "-fsSL", "--retry", "3", "--max-time", str(timeout), "-o", str(tmp), url], check=True)
    tmp.rename(dst)
    return dst


def fetch_all(raw: Path, smoke: bool = False) -> dict:
    from huggingface_hub import hf_hub_download, snapshot_download

    raw.mkdir(parents=True, exist_ok=True)
    status = {}
    meta_files = ["labels/dev.csv", "labels/eval.csv", "labels/vocabulary.csv", "metadata/dev_clips_info_FSD50K.json",
                  "LICENSE-DATASET.md"]
    for f in meta_files:
        hf_hub_download(FSD_META_REPO, f, repo_type="dataset", revision=FSD_META_REV, local_dir=raw / "fsd_meta")
    _download(ONTOLOGY_URL, raw / "ontology.json")
    log("fsd meta + ontology done")

    def get_audio():
        pats = ["data/train-00000-*", "data/validation-00000-*"] if smoke else ["data/train-*", "data/validation-*"]
        snapshot_download(FSD_AUDIO_REPO, repo_type="dataset", revision=FSD_AUDIO_REV, local_dir=raw / "fsd_audio",
                          allow_patterns=pats + ["README.md"], max_workers=16)
        return "ok"

    def get_nsynth(split):
        tgz = _download(NSYNTH_URLS[split], raw / f"nsynth-{split}.tgz")
        subprocess.run(["tar", "xzf", str(tgz), "-C", str(raw)], check=True)
        tgz.unlink()
        return "ok"

    def get_libri():
        for url in LIBRI_URLS:
            try:
                tgz = _download(url, raw / "dev-clean.tgz", timeout=1200)
                subprocess.run(["tar", "xzf", str(tgz), "-C", str(raw)], check=True)
                tgz.unlink()
                return f"ok:{url}"
            except Exception as e:  # try the next mirror
                log("librispeech mirror failed", url, repr(e))
                for p in [raw / "dev-clean.tgz", raw / "dev-clean.tgz.part"]:
                    if p.exists():
                        p.unlink()
        return "failed"

    jobs = {"fsd_audio": get_audio, "nsynth_test": lambda: get_nsynth("test"), "librispeech": get_libri}
    if not smoke:
        jobs["nsynth_valid"] = lambda: get_nsynth("valid")
    with ThreadPoolExecutor(len(jobs)) as ex:
        futs = {k: ex.submit(fn) for k, fn in jobs.items()}
        for k, f in futs.items():
            try:
                status[k] = f.result()
            except Exception as e:
                status[k] = f"failed: {e!r}"
            log("download", k, status[k])
    return status


# 2. FSD50K metadata + selection
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
                      "uploader": info[fid]["uploader"], "fsd_split": r["split"]}
    ontology = json.load(open(raw / "ontology.json"))
    return vocab, ontology, clips, eval_ids


def parquet_files(raw: Path) -> list[str]:
    return sorted(glob.glob(str(raw / "fsd_audio/data/*.parquet")))


def read_parquet_index(files) -> dict[str, dict]:
    import pyarrow.parquet as pq

    out = {}
    for f in files:
        t = pq.read_table(f, columns=["freesound_id", "duration_seconds", "source_partition", "license"])
        for fid, dur, part, lic in zip(*(t.column(c).to_pylist() for c in t.column_names)):
            out[str(fid)] = {"dur": float(dur), "part": part, "license": lic, "file": f}
    return out


def select_fsd(clips: dict, pidx: dict, tax: L.Taxonomy, n: int, rng: random.Random) -> tuple[list[str], dict]:
    """Licence filter, duration filter, then weighted sampling favouring rare classes, uploader cap."""
    stats = collections.Counter()
    elig = []
    for fid in sorted(clips, key=int):
        c = clips[fid]
        stats["dev_total"] += 1
        if c["license"] not in OK_LICENCES:
            stats["drop_licence_" + c["license"].rstrip("/").split("/")[-2]] += 1
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
    freq = collections.Counter(l for fid in elig for l in clips[fid]["leaves"])
    keys = []
    for fid in elig:
        w = max((1.0 / freq[l]) ** 0.5 for l in clips[fid]["leaves"])
        keys.append((math.log(max(rng.random(), 1e-12)) / w, fid))
    keys.sort(reverse=True)
    per_up = collections.Counter()
    cap = max(20, int(n * 0.015))
    chosen = []
    for _, fid in keys:
        up = clips[fid]["uploader"]
        if per_up[up] >= cap:
            continue
        per_up[up] += 1
        chosen.append(fid)
        if len(chosen) >= n:
            break
    stats["selected"] = len(chosen)
    return sorted(chosen, key=int), dict(stats)


def choose_cal_groups(group_of: dict[str, str], frac: float, rng: random.Random, max_group: int) -> set[str]:
    """Pick whole groups (uploaders / instruments / speakers) until ~frac of the members are CAL."""
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


# 3. Audio decoding (parallel over parquet files)
def best_window(x: np.ndarray, win: int) -> np.ndarray:
    if len(x) <= win:
        return x
    hop = SR // 10
    e = np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64) ** 2)])
    starts = np.arange(0, len(x) - win + 1, hop)
    s = int(starts[np.argmax(e[starts + win] - e[starts])])
    return x[s: s + win]


def _decode_file(args):
    import pyarrow.parquet as pq
    import soundfile as sf

    f, wanted = args
    wanted = set(wanted)
    out = {}
    pf = pq.ParquetFile(f)
    for batch in pf.iter_batches(columns=["freesound_id", "audio"], batch_size=128):
        ids = batch.column(0).to_pylist()
        auds = batch.column(1).to_pylist()
        for fid, a in zip(ids, auds):
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


def decode_fsd(pidx: dict, ids: list[str], workers: int = 8) -> dict[str, np.ndarray]:
    by_file = collections.defaultdict(list)
    for fid in ids:
        by_file[pidx[fid]["file"]].append(fid)
    audio = {}
    with Pool(workers) as pool:
        for part in pool.imap_unordered(_decode_file, sorted(by_file.items())):
            audio.update(part)
            log("decoded", len(audio), "/", len(ids))
    return audio


# 4. Item construction helpers
class ItemMaker:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.items: list[dict] = []
        self.noshuf = NoShuffle()
        self.n = 0

    def channel(self, allow_phone=True):
        return "phone" if allow_phone and self.rng.random() < PHONE_FRAC else "clean"

    def add(self, *, split, source, qtype, question, options, gold, audio, meta, family=None, allow_phone=True):
        k = len(options)
        target = [0.0] * k
        target[gold] = 1.0
        rng = self.noshuf if qtype == "score" else self.rng
        it = common.make_item(item_id=f"{BUILDER}_{self.n:06d}", set_name=split, source=source, question=question,
                              options=list(options), target=target, gold=gold, rng=rng, audio=audio,
                              channel=self.channel(allow_phone), state_text=None,
                              meta={"qtype": qtype, "family": family, **meta})
        self.n += 1
        self.items.append(it)
        return it

    def noul(self, *, answer: str, **kw):
        return self.add(qtype="noul", options=["yes", "no"], gold=0 if answer == "yes" else 1, **kw)


def fsd_meta(fid, c):
    return {"source_id": f"fsd50k:{fid}", "freesound_id": int(fid), "speaker": f"freesound_uploader:{c['uploader']}",
            "licence": OK_LICENCES[c["license"]], "fsd50k_labels": c["labels"], "fsd50k_split": "dev",
            "label_provenance": "FSD50K dev ground truth (human-verified, smeared to ontology ancestors)"}


def mix_two(a: np.ndarray, b: np.ndarray, snr_db: float, rng: random.Random) -> np.ndarray:
    """a + b with b's active level snr_db below a's; b cropped (best window) or placed at a random offset."""
    n = len(a)
    if len(b) > n:
        b = best_window(b, n)
    off = rng.randint(0, n - len(b)) if len(b) < n else 0
    gain = 10 ** ((L.active_rms_db(a) - snr_db - L.active_rms_db(b)) / 20)
    y = a.astype(np.float64).copy()
    y[off: off + len(b)] += gain * b
    pk = float(np.max(np.abs(y)))
    if pk > 0.99:
        y *= 0.99 / pk
    return y.astype(np.float32)


def floor_noise(n: int, db: float, rng: np.random.Generator) -> np.ndarray:
    return (rng.standard_normal(n) * 10 ** (db / 20)).astype(np.float32)


# 5. The build
def build(raw: Path, out: Path, smoke: bool = False, fetch_status: dict | None = None) -> dict:
    t0 = time.time()
    rng = random.Random(common.SEED)
    nrng = np.random.default_rng(common.SEED)
    scale = 0.05 if smoke else 1.0
    n_fsd = int(N_FSD * scale)
    quota = {k: int(v * scale) for k, v in QUOTA.items()}
    n_mix, n_ovl, n_trail, n_ns = (int(x * scale) for x in (N_MIX, N_OVERLAY, N_TRAIL, N_NSYNTH))

    vocab, ontology, clips, eval_ids = load_fsd_meta(raw)
    tax = L.Taxonomy(vocab, ontology)
    L.validate_class_table(sorted(vocab.values()))
    L.init_music_subtree(tax)
    files = parquet_files(raw)
    pidx = read_parquet_index(files)
    log("parquet index", len(pidx), "files", len(files))
    # licence cross-check between the official metadata and the repack
    lic_mismatch = sum(1 for fid, c in clips.items() if fid in pidx and pidx[fid]["license"] and
                       pidx[fid]["license"].rstrip("/") != c["license"].rstrip("/"))
    cooc = L.cooccurrence([c["labels"] for c in clips.values()])

    sel, sel_stats = select_fsd(clips, pidx, tax, n_fsd, rng)
    sel_stats["licence_mismatch_official_vs_repack"] = lic_mismatch
    log("selected", sel_stats)
    cal_up = choose_cal_groups({fid: clips[fid]["uploader"] for fid in sel}, CAL_FRAC, rng, max_group=max(5, n_fsd // 300))
    split_of = {fid: ("CAL" if clips[fid]["uploader"] in cal_up else "train") for fid in sel}
    audio = decode_fsd(pidx, sel)
    sel = [fid for fid in sel if fid in audio and len(audio[fid]) >= int(0.5 * SR)]
    log("decoded", len(sel), "clips", f"{sum(len(audio[f]) for f in sel) / SR / 3600:.1f} h")

    mk = ItemMaker(rng)
    n_items = collections.Counter()  # items per stored clip key
    used_fsd: set[str] = set()

    def fsd_audio_ref(fid):
        used_fsd.add(fid)
        n_items[fid] += 1
        return {"bank": None, "key": f"fsd_{fid}"}  # bank filled in when banks are written

    # family 1: presence (one per clip, 50/50 yes/no; negatives 50% siblings, 50% random)
    order = list(sel)
    rng.shuffle(order)
    pres_count = collections.Counter()
    for fid in order:
        c = clips[fid]
        want_yes = pres_count["yes"] <= pres_count["no"]
        ans, cls, kind = None, None, None
        if want_yes:
            leaves = [l for l in c["leaves"] if L.CLASSES[l][3]]
            anyl = [l for l in c["labels"] if L.CLASSES[l][3]]
            pool = leaves if (leaves and rng.random() < 0.65) else anyl
            if pool:
                ans, cls, kind = "yes", rng.choice(sorted(pool)), "label"
        if ans is None:
            hard = rng.random() < 0.5
            cls = L.pick_absent(c["labels"], tax, cooc, rng, hard=hard)
            if cls is None:
                continue
            ans, kind = "no", ("sibling" if hard and cls in {x for l in c["labels"] for x in tax.siblings(l)} else "random")
        q = L.presence_question(rng.choice(L.Q_PRESENCE), cls)
        mk.noul(split=split_of[fid], source="fsd50k", family="presence", question=q, answer=ans,
                audio=fsd_audio_ref(fid), meta={**fsd_meta(fid, c), "asked_class": cls, "negative_kind": kind})
        pres_count[ans] += 1

    # secondary families on single clips (deficit-based, <= MAX_ITEMS_PER_CLIP per clip)
    def pick(cands: list[tuple[str, str]], n: int, balanced: bool) -> list[tuple[str, str]]:
        """cands: (fid, answer). Prefer clips with fewer items; balance yes/no when asked."""
        rng.shuffle(cands)
        cands.sort(key=lambda t: n_items[t[0]])
        cands = [t for t in cands if n_items[t[0]] < MAX_ITEMS_PER_CLIP]
        if not balanced:
            return cands[:n]
        yes = [t for t in cands if t[1] == "yes"]
        no = [t for t in cands if t[1] == "no"]
        k = min(len(yes), len(no), n // 2)
        return yes[:k] + no[:k]

    # speech presence ("is someone talking?") natural clips; overlays add more "yes" later
    libri = sorted(glob.glob(str(raw / "LibriSpeech/dev-clean/*/*/*.flac")))
    if not libri:
        n_ovl = 0
    sp_yes = [(f, "yes") for f in sel if L.speech_answer(clips[f]["labels"]) == "yes"
              and "Speech_synthesizer" not in clips[f]["leaves"]]
    sp_no = [(f, "no") for f in sel if L.speech_answer(clips[f]["labels"]) == "no"]
    speech_pairs = pick(sp_yes, len(sp_yes), False)
    # the natural "no" side is added after the mixtures, to balance overlay "yes" and mixture "no" items
    for fid, ans in speech_pairs:
        mk.noul(split=split_of[fid], source="fsd50k", family="speech", question=rng.choice(L.Q_SPEECH), answer=ans,
                audio=fsd_audio_ref(fid), meta={**fsd_meta(fid, clips[fid]), "label_rule": "speech labels"})

    mus = [(f, L.music_answer(clips[f]["labels"])) for f in sel]
    for fid, ans in pick([t for t in mus if t[1]], quota["music"], True):
        mk.noul(split=split_of[fid], source="fsd50k", family="music", question=rng.choice(L.Q_MUSIC), answer=ans,
                audio=fsd_audio_ref(fid), meta={**fsd_meta(fid, clips[fid]), "label_rule": "music subtree labels"})

    # count "one" (single kind of sound): one leaf, not ambience, unambiguous category
    count_one = [(f, "one") for f in sel if len(clips[f]["leaves"]) == 1 and not (set(clips[f]["leaves"]) & L.AMBIENCE)
                 and L.clip_category(clips[f]["labels"], tax) is not None]
    for fid, _ in pick(count_one, n_mix, False):
        mk.add(split=split_of[fid], source="fsd50k", family="count", qtype="score", question=rng.choice(L.Q_COUNT),
               options=["one", "two"], gold=0, audio=fsd_audio_ref(fid),
               meta={**fsd_meta(fid, clips[fid]), "label_rule": "single labelled sound class"})

    # identification: single leaf, nameable
    idc = [(f, clips[f]["leaves"][0]) for f in sel if len(clips[f]["leaves"]) == 1 and L.CLASSES[clips[f]["leaves"][0]][4]]
    n_ident = 0
    for fid, gold_cls in pick(idc, quota["ident"], False):
        k = rng.randint(4, 8)
        dis = L.pick_distractors(clips[fid]["labels"], gold_cls, k - 1, tax, cooc, rng)
        if dis is None:
            continue
        opts = [L.CLASSES[gold_cls][0]] + [L.CLASSES[d][0] for d in dis]
        mk.add(split=split_of[fid], source="fsd50k", family="ident", qtype="choice", question=rng.choice(L.Q_IDENT),
               options=opts, gold=0, audio=fsd_audio_ref(fid),
               meta={**fsd_meta(fid, clips[fid]), "answer_class": gold_cls, "distractor_classes": dis})
        n_ident += 1

    catc = [(f, L.clip_category(clips[f]["labels"], tax)) for f in sel]
    for fid, cat in pick([t for t in catc if t[1]], quota["category"], False):
        k = rng.randint(4, 6)
        others = [c for c in L.CATEGORIES if c != cat]
        rng.shuffle(others)
        mk.add(split=split_of[fid], source="fsd50k", family="category", qtype="choice",
               question=rng.choice(L.Q_CATEGORY), options=[cat] + others[: k - 1], gold=0, audio=fsd_audio_ref(fid),
               meta={**fsd_meta(fid, clips[fid]), "label_rule": "broad category of the most specific labels"})

    # loudness (by construction: the stored clip is scaled to a level band)
    lvl_cands = []
    for fid in sel:
        fl = L.feasible_levels(audio[fid])
        if fl and L.active_rms_db(audio[fid]) > -60:
            lvl_cands.append((fid, "x"))
    lvl_count = collections.Counter()
    for fid, _ in pick(lvl_cands, quota["level"], False):
        fl = L.feasible_levels(audio[fid])
        lev = min(fl, key=lambda z: (lvl_count[z], L.LEVELS.index(z)))
        lo, hi = L.LEVEL_RANGES_DB[lev]
        tgt = rng.uniform(lo, hi)
        x = audio[fid].astype(np.float64) * 10 ** ((tgt - L.active_rms_db(audio[fid])) / 20)
        pk = float(np.max(np.abs(x)))
        if pk > 0.99:
            x *= 0.99 / pk
        audio[fid] = x.astype(np.float32)
        lvl_count[lev] += 1
        mk.add(split=split_of[fid], source="fsd50k_signal", family="level", qtype="score",
               question=rng.choice(L.Q_LEVEL), options=list(L.LEVELS), gold=L.LEVELS.index(lev),
               audio=fsd_audio_ref(fid), meta={**fsd_meta(fid, clips[fid]), "active_rms_dbfs": round(L.active_rms_db(audio[fid]), 2),
                                               "label_rule": "clip scaled into level band " + str(L.LEVEL_RANGES_DB[lev])})
    log("single-clip items", len(mk.items))

    # synthetic audio (mixtures, speech overlays, trailing silence)
    synth: dict[str, np.ndarray] = {}
    comp_ok = [f for f in sel if len(audio[f]) >= SR and L.clip_category(clips[f]["labels"], tax) is not None
               and not (set(clips[f]["leaves"]) & L.AMBIENCE)]
    used_comp: set[str] = set()

    def comp_pool(split):
        p = [f for f in comp_ok if split_of[f] == split and f not in used_comp]
        rng.shuffle(p)
        return p

    mix_i = 0
    mix_speech_no = 0
    for split, n_target in (("train", n_mix - max(1, int(n_mix * CAL_FRAC))), ("CAL", max(1, int(n_mix * CAL_FRAC)))):
        pool = comp_pool(split)
        made = 0
        while made < n_target and len(pool) >= 2:
            a = pool.pop()
            cat_a = L.clip_category(clips[a]["labels"], tax)
            j = next((k for k in range(len(pool) - 1, -1, -1)
                      if L.clip_category(clips[pool[k]]["labels"], tax) != cat_a
                      and not (set(clips[pool[k]]["labels"]) & L.group_blocked(clips[a]["labels"]))), None)
            if j is None:
                continue
            b = pool.pop(j)
            used_comp |= {a, b}
            snr = rng.uniform(0, 10)
            y = mix_two(audio[a], audio[b], snr, rng)
            key = f"mix_{mix_i:05d}"
            mix_i += 1
            synth[key] = y
            labs = sorted(set(clips[a]["labels"]) | set(clips[b]["labels"]))
            meta = {"source_id": f"fsd50k_mix:{key}", "components": [f"fsd50k:{a}", f"fsd50k:{b}"],
                    "speaker": [f"freesound_uploader:{clips[a]['uploader']}", f"freesound_uploader:{clips[b]['uploader']}"],
                    "licence": [OK_LICENCES[clips[a]["license"]], OK_LICENCES[clips[b]["license"]]],
                    "snr_db": round(snr, 2), "fsd50k_labels": labs, "label_provenance": "labels by construction"}
            ref = {"bank": "sb_sounds_mix", "key": key}
            mk.add(split=split, source="fsd50k_mix", family="count", qtype="score", question=rng.choice(L.Q_COUNT),
                   options=["one", "two"], gold=1, audio=ref, meta=meta)
            def usable(src):
                return ([l for l in clips[src]["leaves"] if L.CLASSES[l][3]]
                        or [l for l in clips[src]["labels"] if L.CLASSES[l][3]])

            srcs = [s for s in rng.sample([a, b], 2) if usable(s)]
            if made % 2 == 0 and srcs:
                src = srcs[0]
                cls, ans = rng.choice(sorted(usable(src))), "yes"
                meta = {**meta, "asked_component": "louder" if src == a else "quieter"}
            else:
                cls, ans = L.pick_absent(labs, tax, cooc, rng, hard=rng.random() < 0.5), "no"
            if cls:
                mk.noul(split=split, source="fsd50k_mix", family="presence", answer=ans,
                        question=L.presence_question(rng.choice(L.Q_PRESENCE), cls), audio=ref,
                        meta={**meta, "asked_class": cls})
            # "is someone talking?" -> no on voice-free mixtures, so that "mixture" is not a cue for "yes"
            if mix_speech_no < n_ovl * 0.8 and L.speech_answer(labs) == "no" and (made % 3 != 2):
                mk.noul(split=split, source="fsd50k_mix", family="speech", answer="no",
                        question=rng.choice(L.Q_SPEECH), audio=ref, meta={**meta, "label_rule": "no voice labels in either component"})
                mix_speech_no += 1
            made += 1
    log("mixtures", mix_i)

    # speech overlays: LibriSpeech dev-clean speech under a non-voice FSD sound
    ovl_made = 0
    if libri and n_ovl:
        import soundfile as sf

        spk_of = {p: Path(p).parts[-3] for p in libri}
        cal_spk = choose_cal_groups(spk_of, 0.06, rng, max_group=10_000)
        sound_ok = [f for f in sel if L.speech_answer(clips[f]["labels"]) == "no" and len(audio[f]) >= 2 * SR
                    and f not in used_comp]
        rng.shuffle(sound_ok)
        by_split = {"train": [p for p in libri if spk_of[p] not in cal_spk], "CAL": [p for p in libri if spk_of[p] in cal_spk]}
        for split, n_target in (("train", n_ovl - max(1, int(n_ovl * CAL_FRAC))), ("CAL", max(1, int(n_ovl * CAL_FRAC)))):
            sounds = [f for f in sound_ok if split_of[f] == split]
            made = 0
            for f in sounds:
                if made >= n_target:
                    break
                p = rng.choice(by_split[split])
                sp, sr = sf.read(p, dtype="float32")
                dur = min(len(audio[f]), int(rng.uniform(2.0, 6.0) * SR))
                sp = best_window(sp, dur) if len(sp) > dur else sp
                if len(sp) < SR:
                    continue
                y = mix_two(audio[f], sp, rng.uniform(-3.0, 6.0), rng)  # speech 3 dB above .. 6 dB below the sound
                key = f"ovl_{ovl_made:05d}"
                synth[key] = y
                used_comp.add(f)
                uid = Path(p).stem
                meta = {"source_id": f"fsd50k_mix:{key}", "components": [f"fsd50k:{f}", f"librispeech:{uid}"],
                        "speaker": [f"freesound_uploader:{clips[f]['uploader']}", f"librispeech_speaker:{spk_of[p]}"],
                        "licence": [OK_LICENCES[clips[f]["license"]], "CC-BY-4.0"], "fsd50k_labels": clips[f]["labels"],
                        "label_provenance": "speech overlaid by construction"}
                mk.noul(split=split, source="fsd50k_mix", family="speech", question=rng.choice(L.Q_SPEECH), answer="yes",
                        audio={"bank": "sb_sounds_mix", "key": key}, meta=meta)
                made += 1
                ovl_made += 1
    log("speech overlays", ovl_made, "mixture speech-no", mix_speech_no)
    n_sp_no = max(0, len(speech_pairs) + ovl_made - mix_speech_no)
    for fid, ans in pick(sp_no, n_sp_no, False):
        mk.noul(split=split_of[fid], source="fsd50k", family="speech", question=rng.choice(L.Q_SPEECH), answer=ans,
                audio=fsd_audio_ref(fid), meta={**fsd_meta(fid, clips[fid]), "label_rule": "no human-voice labels"})

    # trailing silence by construction
    # constructed clips are new audio, so components may be reused; prefer clips not used elsewhere
    tr_ok = [f for f in sel if SR <= len(audio[f]) <= 10 * SR]
    rng.shuffle(tr_ok)
    tr_ok.sort(key=lambda f: f in used_comp)
    tr_i = 0
    want = {"yes": n_trail // 2, "no": n_trail // 2}
    got = collections.Counter()
    for f in tr_ok:
        if got["yes"] >= want["yes"] and got["no"] >= want["no"]:
            break
        x = audio[f]
        ans = "yes" if got["yes"] <= got["no"] else "no"
        if ans == "yes" and got["yes"] >= want["yes"]:
            ans = "no"
        if ans == "no" and got["no"] >= want["no"]:
            ans = "yes"
        if ans == "no" and not L.active_until_end(x):
            if got["yes"] >= want["yes"]:
                continue
            ans = "yes"
        lead = rng.uniform(0.0, 1.0) if ans == "yes" else rng.uniform(0.5, 4.0)
        tail = rng.uniform(1.5, 4.0) if ans == "yes" else 0.0
        y = np.concatenate([np.zeros(int(lead * SR), np.float32), x, np.zeros(int(tail * SR), np.float32)])
        y = y + floor_noise(len(y), rng.uniform(-75.0, -65.0), nrng)
        pk = float(np.max(np.abs(y)))
        if pk > 0.99:
            y *= 0.99 / pk
        key = f"trail_{tr_i:05d}"
        tr_i += 1
        synth[key] = y.astype(np.float32)
        used_comp.add(f)
        got[ans] += 1
        mk.noul(split=split_of[f], source="fsd50k_signal", family="trailing_silence", question=rng.choice(L.Q_TRAIL),
                answer=ans, audio={"bank": "sb_sounds_mix", "key": key},
                meta={**fsd_meta(f, clips[f]), "source_id": f"fsd50k_signal:{key}", "component": f"fsd50k:{f}",
                      "lead_s": round(lead, 2), "tail_silence_s": round(tail, 2),
                      "label_provenance": "silence appended by construction (no: sound active in the last 0.4 s)"})
    log("trailing", dict(got))

    # NSynth
    ns_audio: dict[str, np.ndarray] = {}
    ns_stats = {}
    ns_meta = {}
    for split_dir in sorted(glob.glob(str(raw / "nsynth-*"))):
        ej = Path(split_dir) / "examples.json"
        if ej.exists():
            for k, v in json.load(open(ej)).items():
                v["_wav"] = str(Path(split_dir) / "audio" / f"{k}.wav")
                v["_split"] = Path(split_dir).name
                ns_meta[k] = v
    if ns_meta and n_ns:
        import soundfile as sf

        notes = sorted(ns_meta)
        rng.shuffle(notes)
        fam_count = collections.Counter(ns_meta[k]["instrument_family_str"] for k in notes)
        per_fam = max(1, n_ns // max(1, len(fam_count)))
        inst_cap = max(5, n_ns // 40)
        picked, fc, ic = [], collections.Counter(), collections.Counter()
        for k in notes:  # stratified by family, capped per instrument
            v = ns_meta[k]
            fam, ins = v["instrument_family_str"], v["instrument_str"]
            if fc[fam] >= per_fam or ic[ins] >= inst_cap:
                continue
            fc[fam] += 1
            ic[ins] += 1
            picked.append(k)
        if len(picked) < n_ns:  # fill up with the remaining notes
            rest = [k for k in notes if k not in set(picked)]
            for k in rest:
                if len(picked) >= n_ns:
                    break
                if ic[ns_meta[k]["instrument_str"]] < inst_cap * 2:
                    ic[ns_meta[k]["instrument_str"]] += 1
                    picked.append(k)
        cal_ins = choose_cal_groups({k: ns_meta[k]["instrument_str"] for k in picked}, CAL_FRAC, rng, max_group=10_000)
        fams_present = sorted({ns_meta[k]["instrument_family_str"] for k in picked})
        qual_count = collections.Counter()
        sec_count = collections.Counter()
        for i, k in enumerate(sorted(picked)):
            v = ns_meta[k]
            x, sr = sf.read(v["_wav"], dtype="float32")
            if L.active_rms_db(x) < -45:
                continue
            ns_audio[k] = x
            split = "CAL" if v["instrument_str"] in cal_ins else "train"
            ref = {"bank": "sb_sounds_nsynth", "key": k}
            meta = {"source_id": f"nsynth:{k}", "speaker": f"nsynth_instrument:{v['instrument_str']}", "licence": "CC-BY-4.0",
                    "nsynth_split": v["_split"], "pitch": v["pitch"], "velocity": v["velocity"],
                    "family": None, "label_provenance": "NSynth metadata"}
            fam = v["instrument_family_str"]
            others = [L.NSYNTH_FAMILY[f] for f in fams_present if f != fam]
            rng.shuffle(others)
            kopt = min(rng.randint(4, 6), len(others) + 1)
            mk.add(split=split, source="nsynth", qtype="choice", question=rng.choice(L.Q_NS_FAMILY),
                   options=[L.NSYNTH_FAMILY[fam]] + others[: kopt - 1], gold=0, audio=ref,
                   meta={**meta, "family": "nsynth_family", "instrument_family": fam})
            # second item: the least-used of source / register / quality that applies
            choices = ["source"]
            reg = L.pitch_register(int(v["pitch"]))
            if reg:
                choices.append("register")
            choices.append("quality")
            fam2 = min(choices, key=lambda z: (sec_count[z], rng.random()))
            if fam2 == "source":
                src = v["instrument_source_str"]
                mk.add(split=split, source="nsynth", qtype="choice", question=rng.choice(L.Q_NS_SOURCE),
                       options=[L.NSYNTH_SOURCE[src]] + [L.NSYNTH_SOURCE[s] for s in L.NSYNTH_SOURCE if s != src], gold=0,
                       audio=ref, meta={**meta, "family": "nsynth_source", "instrument_source": src})
            elif fam2 == "register":
                mk.add(split=split, source="nsynth", qtype="score", question=rng.choice(L.Q_NS_REGISTER),
                       options=list(L.REGISTER_LEVELS), gold=L.REGISTER_LEVELS.index(reg), audio=ref,
                       meta={**meta, "family": "nsynth_register", "midi_pitch": v["pitch"],
                             "label_rule": "low<=45, middle 52..67, high>=76 (MIDI)"}, allow_phone=False)
            else:
                qs = v["qualities_str"]
                # pick the quality / polarity with the largest deficit
                opts = []
                for qn in L.Q_NS_QUALITY:
                    ans = "yes" if qn in qs else "no"
                    opts.append((qual_count[(qn, ans)] - qual_count[(qn, "no" if ans == "yes" else "yes")], qn, ans))
                opts.sort(key=lambda t: (t[0], rng.random()))
                _, qn, ans = opts[0]
                if qual_count[(qn, ans)] > qual_count[(qn, "no" if ans == "yes" else "yes")]:
                    fam2 = "source"
                    src = v["instrument_source_str"]
                    mk.add(split=split, source="nsynth", qtype="choice", question=rng.choice(L.Q_NS_SOURCE),
                           options=[L.NSYNTH_SOURCE[src]] + [L.NSYNTH_SOURCE[s] for s in L.NSYNTH_SOURCE if s != src],
                           gold=0, audio=ref, meta={**meta, "family": "nsynth_source", "instrument_source": src})
                else:
                    qual_count[(qn, ans)] += 1
                    mk.noul(split=split, source="nsynth", question=rng.choice(L.Q_NS_QUALITY[qn]), answer=ans, audio=ref,
                            meta={**meta, "family": "nsynth_quality", "quality": qn, "qualities": qs}, allow_phone=False)
            sec_count[fam2] += 1
        ns_stats = {"notes_available": len(ns_meta), "notes_used": len(ns_audio), "families": dict(fc),
                    "cal_instruments": sorted(cal_ins), "secondary": dict(sec_count),
                    "quality_balance": {f"{a}:{b}": n for (a, b), n in sorted(qual_count.items())}}
    log("nsynth", ns_stats.get("notes_used"))

    # write banks
    out.mkdir(parents=True, exist_ok=True)
    adir = out / "audio"
    bank_of: dict[str, str] = {}
    fsd_keys = sorted(used_fsd, key=int)
    max_bytes = 3.5e9
    banks: list[tuple[str, dict]] = []
    cur, cur_bytes, bi = {}, 0, 0
    for fid in fsd_keys:
        x = audio[fid]
        if cur and cur_bytes + 4 * len(x) > max_bytes:
            banks.append((f"sb_sounds_fsd{bi}", cur))
            cur, cur_bytes, bi = {}, 0, bi + 1
        cur[f"fsd_{fid}"] = x
        cur_bytes += 4 * len(x)
    if cur:
        banks.append((f"sb_sounds_fsd{bi}", cur))
    banks.append(("sb_sounds_mix", synth))
    banks.append(("sb_sounds_nsynth", ns_audio))
    bank_seconds = {}
    for name, clipsd in banks:
        common.write_audio_bank(adir, name, clipsd)
        for k in clipsd:
            bank_of[k] = name
        bank_seconds[name] = sum(len(v) for v in clipsd.values()) / SR
        log("wrote bank", name, len(clipsd), f"{bank_seconds[name] / 3600:.2f} h")
    for it in mk.items:
        it["audio"]["bank"] = bank_of[it["audio"]["key"]]

    train = [it for it in mk.items if it["set"] == "train"]
    cal = [it for it in mk.items if it["set"] == "CAL"]
    common.write_jsonl(out / "items_train.jsonl", train)
    common.write_jsonl(out / "items_cal.jsonl", cal)

    checks = validate(out, train + cal)
    leakage = leakage_check(train + cal)
    summary = make_summary(train, cal, bank_seconds, sel_stats, ns_stats, checks, leakage, fetch_status, smoke,
                           time.time() - t0, n_ident=n_ident, overlays=ovl_made, mixtures=mix_i, trailing=dict(got),
                           levels=dict(lvl_count), cal_uploaders=len(cal_up))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    log("done", len(train), "train", len(cal), "CAL", checks.get("ok"))
    return summary


# 6. Checks + summary
def validate(out: Path, items: list[dict]) -> dict:
    keys_needed = {"id", "set", "source", "question", "options", "target", "gold", "audio", "channel", "state_text", "meta"}
    errs = collections.Counter()
    banks = {}
    durs = []
    ids = set()
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
        if it["channel"] not in ("clean", "phone"):
            errs["channel"] += 1
        b = it["audio"]["bank"]
        if b not in banks:
            banks[b] = common.AudioBank(out / "audio", b)
        if it["audio"]["key"] not in banks[b].index:
            errs["missing_audio_key"] += 1
            continue
        durs.append(banks[b].index[it["audio"]["key"]][1] / SR)
    # spot-check actual audio for a sample
    rng = random.Random(1)
    for it in rng.sample(items, min(300, len(items))):
        x = banks[it["audio"]["bank"]].get(it["audio"]["key"])
        if not np.isfinite(x).all() or np.max(np.abs(x)) > 1.0 or len(x) < 0.5 * SR:
            errs["bad_audio"] += 1
        if it["channel"] == "phone" and len(common.phone_channel(x)) != len(x):
            errs["phone_len"] += 1
    d = np.array(durs)
    bal = collections.defaultdict(collections.Counter)
    for it in items:
        bal[f"{it['source']}/{it['meta']['family']}"][it["options"][it["gold"]]] += 1
    return {"ok": not errs, "errors": dict(errs), "n_items": len(items),
            "duration_s": {"min": float(d.min()), "p50": float(np.median(d)), "max": float(d.max()), "mean": float(d.mean())},
            "answer_balance": {k: dict(v) for k, v in sorted(bal.items())}}


def _meta_strings(obj, out: set):
    if isinstance(obj, dict):
        for v in obj.values():
            _meta_strings(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _meta_strings(v, out)
    elif obj is not None:
        out.add(str(obj))


def leakage_check(items: list[dict]) -> dict:
    """Compare our source ids / groups with the evaluation items (if present on the volume)."""
    ours_ids = set()
    ours_groups = set()
    for it in items:
        m = it["meta"]
        for key in ("source_id", "components", "component"):
            v = m.get(key)
            for s in (v if isinstance(v, list) else [v]):
                if s:
                    ours_ids.add(str(s))
                    ours_ids.add(str(s).split(":", 1)[-1])
        for s in (m["speaker"] if isinstance(m.get("speaker"), list) else [m.get("speaker")]):
            if s:
                ours_groups.add(str(s).split(":", 1)[-1])
    res = {}
    for name, path in (("adi", "/vol/adi/items.jsonl"), ("exp0_eval", "/vol/data/eval_items.jsonl")):
        if not os.path.exists(path):
            res[name] = "not present at build time"
            continue
        their = set()
        fsd_like = set()
        for line in open(path):
            if not line.strip():
                continue
            it = json.loads(line)
            s = set()
            _meta_strings(it.get("meta", {}), s)
            if it.get("audio"):
                s.add(str(it["audio"].get("key")))
            their |= s
            if "fsd" in json.dumps(it.get("meta", {})).lower() or "fsd" in str(it.get("source", "")).lower():
                fsd_like |= s
        id_hits = sorted(x for x in ours_ids if x in their and not x.isdigit())
        num_hits = sorted(x for x in ours_ids if x.isdigit() and x in fsd_like)
        grp_hits = sorted(x for x in ours_groups if x in their and len(x) > 3)
        res[name] = {"n_their_meta_strings": len(their), "source_id_hits": id_hits[:50],
                     "fsd_numeric_id_hits": num_hits[:50], "group_hits": grp_hits[:50],
                     "clean": not (id_hits or num_hits or grp_hits)}
    return res


def make_summary(train, cal, bank_seconds, sel_stats, ns_stats, checks, leakage, fetch_status, smoke, secs, **extra):
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
                   "by_n_options": by(lambda it: len(it["options"]))},
        "audio_hours": {k: round(v / 3600, 3) for k, v in bank_seconds.items()},
        "audio_hours_total": round(sum(bank_seconds.values()) / 3600, 3),
        "sources": SOURCES_INFO,
        "revisions": {"fsd50k_meta": f"{FSD_META_REPO}@{FSD_META_REV}", "fsd50k_audio": f"{FSD_AUDIO_REPO}@{FSD_AUDIO_REV}",
                      "audioset_ontology": ONTOLOGY_URL, "nsynth": NSYNTH_URLS, "librispeech": LIBRI_URLS[0]},
        "fsd_selection": sel_stats, "nsynth": ns_stats, "fetch_status": fetch_status, "extra": extra,
        "checks": checks, "leakage": leakage, "examples": examples,
    }
