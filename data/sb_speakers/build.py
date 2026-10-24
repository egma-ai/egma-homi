"""sb_speakers clip building (runs in Modal, see modal_build.py): LibriSpeech pools, then per shard timelines whose
duration is independent of N, cut from real utterances and rendered per speaker. N is the number of distinct real
speakers placed, re-checked with a per-track VAD.
"""
from __future__ import annotations

import io
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import common as C
import labels as L
import questions as Q

BUILDER = "sb_speakers"
SR = C.SAMPLE_RATE
VOL = Path("/vol")
WORK = VOL / "stageb_work" / BUILDER
POOL_LS = WORK / "pool_ls"
POOL_LTR = VOL / "stageb_work" / "sb_count" / "pool"            # read-only (sb_count's LibriTTS-R pools)
SHARDS = WORK / "shards"
RAW_HVB = VOL / "stageb" / "sb_dialog" / "work" / "raw" / "hvb"   # read-only
RAW_AMI = VOL / "stageb" / "sb_dialog" / "work" / "raw" / "ami"   # read-only
RAW_PHONE = VOL / "stageb" / "sb_phone" / "work" / "raw"          # read-only (MUSAN noise excerpts + licence texts)
HERE = Path(__file__).resolve().parent

LIBRI_REPO, LIBRI_REV = "mythicinfinity/libritts_r", "0d1718db351207d5a493135a62d5ed8cc4733830"
LS_REPO, LS_REV = "openslr/librispeech_asr", "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
HVB_SHA = "0bd721e877c4a85d8c13ff837e68661ea6200a98"
SLUE_REPO, SLUE_REV = "asapp/slue-phase-2", "e2989c55a53593a8e39b8f8ebdb47ccaccbe484a"
SOURCES = {
    "libritts_r": {"name": "LibriTTS-R utterances (train.clean.100 + subsets of train.clean.360 / train.other.500 -> train; "
                           "dev.clean -> CAL); pools re-used read-only from /vol/stageb_work/sb_count/pool",
                   "repo": LIBRI_REPO, "rev": LIBRI_REV,
                   "licence": "CC BY 4.0 (HF cardData license: cc-by-4.0). Gender from LibriSpeech SPEAKERS.TXT (CC BY 4.0)."},
    "librispeech": {"name": "LibriSpeech ORIGINAL audio (clean/train.100 + other/train.500 shards -> train; clean/validation -> CAL)",
                    "repo": LS_REPO, "rev": LS_REV, "licence": "CC BY 4.0 (HF cardData license: cc-by-4.0; openslr 12)"},
    "hvb": {"name": "HarperValleyBank per-speaker channels + word timings (raw copy of sb_dialog, read-only)",
            "rev": f"github.com/cricketclub/gridspace-stanford-harper-valley@{HVB_SHA} (raw copy at {RAW_HVB})",
            "licence": "CC BY 4.0 (repo LICENSE)",
            "exclusion": f"every SLUE-HVB test conversation ({SLUE_REPO}@{SLUE_REV}) and every HVB call in our "
                         "evaluation set"},
    "ami": {"name": "AMI Meeting Corpus Mix-Headset wavs + manual word alignments; only stretches where exactly one "
                    "participant is active (others silent +-0.25 s)",
            "rev": f"official mirror groups.inf.ed.ac.uk/ami (raw copy at {RAW_AMI}, read-only); HF mirror "
                   "edinburghcstr/ami@46f28f2503e2ec48f8867a84eef356c70476beab cardData.license=cc-by-4.0",
            "licence": "CC BY 4.0"},
    "musan_noise": {"name": "MUSAN noise excerpts (sb_phone raw copy, read-only); only files whose per-file licence is "
                            "public domain or CC BY 3.0 (no share-alike / NC / ND / Sampling+); ambient 'background' files "
                            "and sound-bible titles that suggest human voices excluded (they could add a speaker)",
                    "rev": "https://www.openslr.org/17/ musan.tar.gz (raw copy at /vol/stageb/sb_phone/work/raw)",
                    "licence": "MUSAN CC BY 4.0 corpus; per-file: Public Domain (free-sound) / Attribution 3.0 or Public Domain (sound-bible)"},
    "phone_fx": {"name": "data/sb_phone/fx.py: 300-3400 Hz band-limit + real codec through ffmpeg (mulaw/alaw/gsm/opus)",
                 "rev": "repo code", "licence": "own code; ffmpeg codecs (LGPL tool, not redistributed)"},
}
LICENCE = {"libritts_r": "CC BY 4.0", "librispeech": "CC BY 4.0", "hvb": "CC BY 4.0", "ami": "CC BY 4.0"}


def _gender_map() -> dict:
    return json.loads((HERE / "librispeech_speaker_gender.json").read_text())["gender"]


def fade(x: np.ndarray, ms: float = 10.0) -> np.ndarray:
    x = np.array(x, dtype=np.float32)
    f = min(len(x) // 2, int(ms / 1000 * SR))
    if f > 1:
        x[:f] *= np.linspace(0, 1, f, dtype=np.float32)
        x[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
    return x


# LibriSpeech pool

def _hf_files(repo: str, rev: str, prefix: str) -> list[str]:
    from huggingface_hub import HfApi
    return sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=rev)
                  if f.startswith(prefix) and f.endswith(".parquet"))


def decode_bytes(b: bytes) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    return C.resample(x.mean(axis=1), sr, SR)


def prep_librispeech(prefix: str, shard: int, tag: str, keep: float, cap: int, max_s: float = 16.0) -> dict:
    """One LibriSpeech parquet shard -> pool (utterances trimmed, cut to <= max_s at a low-energy point)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    out_npy = POOL_LS / f"{tag}_{shard:02d}.npy"
    if out_npy.exists() and out_npy.with_suffix(".jsonl").exists():
        return {"tag": tag, "shard": shard, "skipped": True}
    files = _hf_files(LS_REPO, LS_REV, prefix)
    path = hf_hub_download(LS_REPO, files[shard], repo_type="dataset", revision=LS_REV, cache_dir="/tmp/hf")
    gender = _gender_map()
    per_spk = Counter()
    xs, recs, off = [], [], 0
    pf = pq.ParquetFile(path)
    cols = [c for c in ("audio", "speaker_id", "chapter_id", "id") if c in pf.schema_arrow.names]
    rng = random.Random(f"{C.SEED}:{BUILDER}:ls:{tag}:{shard}")
    for batch in pf.iter_batches(batch_size=32, columns=cols):
        for r in batch.to_pylist():
            spk = str(r["speaker_id"])
            if spk not in gender or per_spk[spk] >= cap or L.stable_frac(str(r["id"]), BUILDER) >= keep:
                continue
            x = decode_bytes(r["audio"]["bytes"])
            a, b = L.trim_bounds(x, SR)
            x = x[a:b]
            if len(x) / SR > max_s:
                pts = [p for p in L.cut_points(x, SR) if 6.0 <= p <= max_s]
                if not pts:
                    continue
                x = x[: int(rng.choice(pts) * SR)]
            dur = len(x) / SR
            if dur < 1.0 or not np.isfinite(x).all() or np.abs(x).max() < 1e-3:
                continue
            per_spk[spk] += 1
            xs.append(fade(x, 8))
            recs.append({"id": f"ls:{r['id']}", "spk": f"libri:{spk}", "gender": gender[spk], "chapter": str(r["chapter_id"]),
                         "dur": round(dur, 3), "off": off, "n": len(x), "split": prefix})
            off += len(x)
    POOL_LS.mkdir(parents=True, exist_ok=True)
    tmp = out_npy.with_name(out_npy.stem + ".tmp.npy")
    np.save(tmp, np.concatenate(xs) if xs else np.zeros(0, np.float32))
    with open(out_npy.with_suffix(".jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    tmp.rename(out_npy)
    return {"tag": tag, "shard": shard, "utts": len(recs), "speakers": len(per_spk), "file": files[shard]}


# Utterance pools

class UttPool:
    """Read-speech utterances by speaker (LibriTTS-R or LibriSpeech pools; memory-mapped)."""

    def __init__(self, corpus: str, files: list[Path]):
        self.corpus = corpus
        self.data, self.by_spk, self.gender = {}, defaultdict(list), {}
        for p in files:
            if not p.exists() or not p.with_suffix(".jsonl").exists():
                continue
            t = p.stem
            self.data[t] = np.load(p, mmap_mode="r")
            for line in open(p.with_suffix(".jsonl")):
                r = json.loads(line)
                r["tag"] = t
                if corpus == "libritts_r":
                    r["spk"] = "libri:" + r["spk"].split(":")[-1]
                    r["id"] = "ltr:" + r["id"]
                self.by_spk[r["spk"]].append(r)
                self.gender[r["spk"]] = r["gender"]
        self.spks = sorted(s for s, v in self.by_spk.items() if len(v) >= 3)
        self._f0 = {}

    def audio(self, r) -> np.ndarray:
        return np.array(self.data[r["tag"]][r["off"]: r["off"] + r["n"]], dtype=np.float32)

    def gender_of(self, s):
        return self.gender.get(s)

    def f0(self, s):
        if s not in self._f0:
            rs = sorted(self.by_spk[s], key=lambda r: -r["dur"])[:2]
            v = [L.median_f0(self.audio(r), SR) for r in rs]
            v = [x for x in v if x]
            self._f0[s] = float(np.median(v)) if v else None
        return self._f0[s]

    def piece(self, s, target: float, rng, avoid: set, chapter=None):
        """-> (audio, source id, chapter) close to target seconds, cut at low-energy points."""
        rs = [r for r in self.by_spk[s] if r["id"] not in avoid]
        if chapter is not None:
            same = [r for r in rs if r["chapter"] == chapter]
            rs = same or rs
        if not rs:
            return None
        if target <= L.SHORT_HI:
            c = sorted(rs, key=lambda r: r["dur"])[: max(3, len(rs) // 3)]
        else:
            c = [r for r in rs if target - 0.1 <= r["dur"] <= target + 2.5] or [r for r in rs if r["dur"] >= target - 0.1]
            c = c or sorted(rs, key=lambda r: -r["dur"])[:3]
        r = rng.choice(c)
        x = self.audio(r)
        if len(x) / SR > target + 0.2:
            w = L.best_window(L.cut_points(x, SR), target, rng)
            if w is None or abs((w[1] - w[0]) - target) > max(0.3, 0.25 * target):
                w = L.best_window(L.cut_points(x, SR, rel_db=15.0), target, rng) or w
            if w is not None:
                x = x[int(w[0] * SR): int(w[1] * SR)]
        return fade(x), r["id"], r["chapter"], r["id"]


# HarperValleyBank

def hvb_parse(segs: list[dict]) -> list[dict]:
    """HVB transcript json -> turns {role, words[(s, e)]} (consecutive same-role speech segments; sb_dialog's rules)."""
    noise = re.compile(r"^(\s*\[[^\]]*\]\s*)*$")
    out = []
    for g in segs:
        role = g.get("speaker_role")
        if role not in ("agent", "caller"):
            continue
        s = g["start_ms"] / 1000.0
        text = (g.get("human_transcript") or g.get("transcript") or "").strip()
        if not text or noise.match(text):
            continue
        words = [(s + o / 1000.0, s + (o + d) / 1000.0)
                 for o, d in zip(g.get("word_offsets_ms") or [], g.get("word_durations_ms") or []) if d > 0]
        if words:
            out.append({"role": role, "s": s, "words": sorted(words)})
    out.sort(key=lambda g: g["s"])
    return out


def hvb_speaker_split(conv_spk: dict, frac: float):
    """sb_dialog's rule: CAL speaker set S grown pairwise; conversations with both speakers in S -> CAL, neither ->
    train, exactly one -> dropped."""
    import itertools
    spk_all = sorted({s for v in conv_spk.values() for s in v})
    n = len(conv_spk)

    def stats(S):
        return (sum(1 for v in conv_spk.values() if v <= S), sum(1 for v in conv_spk.values() if (v & S) and not v <= S))

    S = set()
    while stats(S)[0] < frac * n and len(S) < max(2, len(spk_all) // 3):
        c0, d0 = stats(S)
        best = None
        for a, b in itertools.combinations(spk_all, 2):
            T = S | {a, b}
            if T == S:
                continue
            c, d = stats(T)
            sc = (c - c0) / (1 + d - d0)
            if best is None or sc > best[0]:
                best = (sc, a, b)
        if best is None or best[0] <= 0:
            break
        S |= {best[1], best[2]}
    return {c: ("CAL" if v <= S else ("train" if not (v & S) else None)) for c, v in conv_spk.items()}, S


def adi_ids() -> dict:
    out = {"hvb_calls": set(), "ids": set(), "speakers": set(), "speakers_by_source": defaultdict(set)}
    p = VOL / "adi" / "items.jsonl"
    if not p.exists():
        return out
    for line in open(p):
        it = json.loads(line)
        m = it.get("meta") or {}
        if it.get("source") == "hvb" and m.get("call") is not None:
            out["hvb_calls"].add(str(m["call"]))
        for k in ("call", "recording", "file", "source_id", "meeting", "clip", "utt_id", "id"):
            if m.get(k) is not None:
                out["ids"].add(str(m[k]))
        for k in ("speaker", "speaker_id", "actor"):
            if m.get(k) is not None:
                out["speakers"].add(str(m[k]))
                out["speakers_by_source"][str(it.get("source"))].add(str(m[k]))
    return out


def word_spans(words, lo: float = 0.4, hi: float = 6.3, min_gap: float = 0.06, max_per_run: int = 6, rng=None):
    """Spans [start of word i, end of word j] whose outer boundaries are separated from neighbouring words by
    >= min_gap (clean cuts), lengths in [lo, hi]."""
    words = sorted(words)
    n = len(words)
    ok_start = [i == 0 or words[i][0] - words[i - 1][1] >= min_gap for i in range(n)]
    ok_end = [j == n - 1 or words[j + 1][0] - words[j][1] >= min_gap for j in range(n)]
    out = []
    for i in range(n):
        if not ok_start[i]:
            continue
        for j in range(i, n):
            d = words[j][1] - words[i][0]
            if d > hi:
                break
            if ok_end[j] and d >= lo:
                out.append((words[i][0], words[j][1]))
    if rng is not None and len(out) > max_per_run:
        short = [s for s in out if s[1] - s[0] <= L.SHORT_HI]
        rest = [s for s in out if s[1] - s[0] > L.SHORT_HI]
        out = rng.sample(short, min(len(short), max_per_run // 2)) + rng.sample(rest, min(len(rest), max_per_run - min(len(short), max_per_run // 2)))
    return out


class HVB:
    corpus = "hvb"

    def __init__(self, adi_calls: set, split: str, rng, n_convs: int):
        slue = set(json.loads((RAW_HVB / "slue_ids.json").read_text())["test"])
        conv_spk, self.role_spk = {}, {}
        self.n_excluded = 0
        for p in sorted((RAW_HVB / "transcript").glob("*.json")):
            sid = p.stem
            if sid in slue or sid in adi_calls:
                self.n_excluded += 1
                continue
            md = json.loads((RAW_HVB / "metadata" / f"{sid}.json").read_text())
            self.role_spk[sid] = {r: f"hvb_spk:{md[r]['speaker_id']}" for r in ("agent", "caller")}
            conv_spk[sid] = set(self.role_spk[sid].values())
        self.split_of, self.cal_spk = hvb_speaker_split(conv_spk, 0.03)
        sids = sorted(c for c, v in self.split_of.items() if v == split)
        sids = rng.sample(sids, min(n_convs, len(sids)))
        self.idx = defaultdict(list)  # speaker -> [(sid, role, a, b)]
        for sid in sids:
            segs = json.loads((RAW_HVB / "transcript" / f"{sid}.json").read_text())
            for t in hvb_parse(segs):
                for a, b in word_spans(t["words"], rng=rng):
                    self.idx[self.role_spk[sid][t["role"]]].append((sid, t["role"], a, b))
        self.spks = sorted(s for s, v in self.idx.items() if len(v) >= 3)
        self._audio, self._f0 = {}, {}

    def chan(self, sid, role):
        import soundfile as sf
        k = (sid, role)
        if k not in self._audio:
            if len(self._audio) > 48:
                self._audio.clear()
            x, sr = sf.read(str(RAW_HVB / "audio" / role / f"{sid}.wav"), dtype="float32", always_2d=True)
            self._audio[k] = C.resample(x.mean(1), sr, SR)
        return self._audio[k]

    def load(self, e, pad=0.04):
        sid, role, a, b = e
        ch = self.chan(sid, role)
        return fade(ch[max(0, int((a - pad) * SR)): int((b + pad) * SR)])

    def gender_of(self, s):
        f = self.f0(s)
        return None if f is None else ("male" if f < 165 else "female")

    def f0(self, s):
        if s not in self._f0:
            es = sorted(self.idx[s], key=lambda e: -(e[3] - e[2]))[:2]
            v = [L.median_f0(self.load(e), SR) for e in es]
            v = [x for x in v if x]
            self._f0[s] = float(np.median(v)) if v else None
        return self._f0[s]

    def piece(self, s, target, rng, avoid, chapter=None):
        es = [e for e in self.idx[s] if (e[0], e[2]) not in avoid]
        if chapter is not None:
            same = [e for e in es if e[0] == chapter]
            es = same or es
        if not es:
            return None
        best = min(abs((e[3] - e[2]) - target) for e in es)
        c = [e for e in es if abs((e[3] - e[2]) - target) <= best + max(0.15, 0.15 * target)]
        e = rng.choice(c)
        return self.load(e), f"hvb:{e[0]}", e[0], (e[0], e[2])


# AMI

def components(groups: dict) -> list[set]:
    parent = {g: g for g in groups}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner = {}
    for g, spk in groups.items():
        for s in spk:
            if s in owner:
                parent[find(g)] = find(owner[s])
            else:
                owner[s] = g
    comp = {}
    for g in groups:
        comp.setdefault(find(g), set()).add(g)
    return sorted(comp.values(), key=lambda c: sorted(c)[0])


def cal_groups(groups: dict, frac: float, rng: random.Random) -> set:
    comps = components(groups)
    rng.shuffle(comps)
    cal, n = set(), len(groups)
    for c in comps:
        if len(cal) >= frac * n:
            break
        if len(c) <= max(2, 0.5 * frac * n) + 4:
            cal |= c
    return cal


class AMI:
    corpus = "ami"

    def __init__(self, split: str, rng, max_meetings: int = 70):
        import soundfile as sf
        ann = json.loads((RAW_AMI / "ann.json").read_text())
        self.ann, self.dur = {}, {}
        for m, spk in sorted(ann.items()):
            p = RAW_AMI / f"{m}.wav"
            if not p.exists():
                continue
            try:
                self.dur[m] = sf.info(str(p)).duration
            except Exception:  # noqa: BLE001
                continue
            self.ann[m] = {g: {"sp": [tuple(x) for x in v["sp"]], "inc": [tuple(x) for x in v["inc"]]} for g, v in spk.items()}
        groups = {m: set(v) for m, v in self.ann.items()}
        # same CAL rule and seed as sb_count (whole speaker-closed groups of meetings)
        cal = cal_groups(groups, 0.06, random.Random(f"{C.SEED}:sb_count:ami_cal"))
        split_of = {m: ("CAL" if m in cal else "train") for m in self.ann}
        cal_spk = {g for m in cal for g in self.ann[m]}
        for m in list(split_of):
            if split_of[m] == "train" and set(self.ann[m]) & cal_spk:
                split_of[m] = None
        self.split_of = split_of
        ms = sorted(m for m, v in split_of.items() if v == split)
        if len(ms) > max_meetings:
            ms = rng.sample(ms, max_meetings)
        self.idx = defaultdict(list)  # "ami:<gid>" -> [(meeting, a, b)]
        for m in ms:
            n = int(self.dur[m] * 100) + 10
            masks = {}
            for g, v in self.ann[m].items():
                a_ = np.zeros(n, bool)
                for s, e in v["inc"]:
                    a_[max(0, int((s - 0.25) * 100)): min(n, int(math.ceil((e + 0.25) * 100)))] = True
                masks[g] = a_
            for g, v in self.ann[m].items():
                others = np.zeros(n, bool)
                for h, a_ in masks.items():
                    if h != g:
                        others |= a_
                own = np.zeros(n, bool)
                for s, e in v["inc"]:
                    own[max(0, int(s * 100)): min(n, int(math.ceil(e * 100)))] = True
                runs, cur = [], []
                for w in sorted(v["sp"]):
                    if cur and w[0] - cur[-1][1] > 0.35:
                        runs.append(cur)
                        cur = []
                    cur.append(w)
                if cur:
                    runs.append(cur)
                for wds in runs:
                    ra, rb = wds[0][0], wds[-1][1]
                    if others[max(0, int(ra * 100)): int(rb * 100) + 1].any():
                        continue
                    if own[max(0, int((ra - 0.2) * 100)): max(0, int((ra - 0.03) * 100))].any() or \
                            own[int((rb + 0.03) * 100): int((rb + 0.2) * 100)].any():
                        continue  # a vocal sound of the same person right outside the run would be cut
                    for a, b in word_spans(wds, min_gap=0.08, max_per_run=4, rng=rng):
                        self.idx[f"ami:{g}"].append((m, a, b))
        self.spks = sorted(s for s, v in self.idx.items() if len(v) >= 3)
        self._f0 = {}

    def read(self, m, t0, t1):
        import soundfile as sf
        x, sr = sf.read(str(RAW_AMI / f"{m}.wav"), start=max(0, int(t0 * 16000)), stop=int(t1 * 16000), dtype="float32",
                        always_2d=True)
        x = x.mean(1)
        return C.resample(x, sr, SR) if sr != SR else x

    def load(self, e, pad=0.05):
        m, a, b = e
        return fade(self.read(m, a - pad, b + pad))

    def gender_of(self, s):
        g = s.split(":")[-1]
        return "female" if g.startswith("F") else ("male" if g.startswith("M") else None)

    def f0(self, s):
        if s not in self._f0:
            es = sorted(self.idx[s], key=lambda e: -(e[2] - e[1]))[:2]
            v = [L.median_f0(self.load(e), SR) for e in es]
            v = [x for x in v if x]
            self._f0[s] = float(np.median(v)) if v else None
        return self._f0[s]

    def meeting_of(self, s):
        return Counter(e[0] for e in self.idx[s]).most_common(1)[0][0]

    def piece(self, s, target, rng, avoid, chapter=None):
        es = [e for e in self.idx[s] if (e[0], e[1]) not in avoid]
        if chapter is not None:
            same = [e for e in es if e[0] == chapter]
            es = same or es
        if not es:
            return None
        best = min(abs((e[2] - e[1]) - target) for e in es)
        c = [e for e in es if abs((e[2] - e[1]) - target) <= best + max(0.15, 0.15 * target)]
        e = rng.choice(c)
        return self.load(e), f"ami:{e[0]}", e[0], (e[0], e[1])


# MUSAN noise

_VOICE_WORDS = re.compile(r"crowd|people|talk|speech|voice|laugh|scream|yell|shout|cheer|applause|audience|baby|child|"
                          r"kid|girl|boy|man\b|men\b|woman|women|cough|sneez|sing|choir|party|restaurant|cafe|chatter|"
                          r"babble|human|burp|kiss|cry|whistl|hello|count|say|said|announce|radio|tv|phone call", re.I)


def musan_noise_keys() -> tuple[dict, dict]:
    """{key: licence} of allowed MUSAN noise excerpts + a report."""
    texts = json.loads((RAW_PHONE / "musan_texts.json").read_text())
    meta = json.loads((RAW_PHONE / "musan_noise_meta.json").read_text())
    rep = Counter()
    fs_lic = " ".join(texts.get("musan/noise/free-sound/LICENSE", "").split())
    background = set(re.findall(r"noise-free-sound-\d{4}", texts.get("musan/noise/free-sound/ANNOTATIONS", "")))
    sb = {}
    for block in re.split(r"\n\s*={10,}\s*\n?", texts.get("musan/noise/sound-bible/LICENSE", "")):
        ids = re.findall(r"noise-sound-bible-\d{4}", block)
        lic = re.findall(r"License:\s*(.*)", block)
        title = re.findall(r"Title:\s*(.*)", block)
        for i in ids:
            sb[i] = ((lic[0].strip() if lic else ""), (title[0].strip() if title else ""))
    allowed = {}
    for k, m in meta.items():
        if m.get("subdir") == "free-sound":
            if "public domain" not in fs_lic.lower():
                rep["rejected:free-sound licence"] += 1
                continue
            if k in background:
                rep["rejected:ambient background (may contain voices)"] += 1
                continue
            allowed[k] = "Public Domain (MUSAN free-sound LICENSE)"
            rep["ok:free-sound public domain"] += 1
        elif m.get("subdir") == "sound-bible":
            lic, title = sb.get(k, ("", ""))
            low = lic.lower()
            if not (low.startswith("public domain") or low.startswith("attribution 3.0")) or re.search(r"share|non|deriv|sampling|personal", low):
                rep[f"rejected:sound-bible licence {lic[:30]}"] += 1
                continue
            if _VOICE_WORDS.search(title):
                rep["rejected:sound-bible title suggests a voice"] += 1
                continue
            allowed[k] = f"{lic} (soundbible.com, '{title}')"
            rep[f"ok:sound-bible {lic[:20]}"] += 1
        else:
            rep[f"rejected:subdir {m.get('subdir')}"] += 1
    return allowed, dict(rep)


# Rendering

def active_rms(x: np.ndarray) -> float:
    db = L.frame_db(x, SR)
    if len(db) == 0:
        return 1e-4
    thr = np.percentile(db, 95) - 25
    act = db[db > thr]
    return float(10 ** (np.mean(act) / 20)) if len(act) else 1e-4


def level(x: np.ndarray, target_db: float) -> np.ndarray:
    return (x * (10 ** (target_db / 20) / (active_rms(x) + 1e-9))).astype(np.float32)


def split_audio(x: np.ndarray, frac: float):
    """Split at the low-energy cut point nearest to frac of the length (None if no clean point)."""
    pts = [p for p in L.cut_points(x, SR, rel_db=30.0)[1:-1] if 0.25 <= p <= len(x) / SR - 0.25]
    if not pts:
        return None
    t = min(pts, key=lambda p: abs(p - frac * len(x) / SR))
    i = int(t * SR)
    return fade(x[:i], 6), fade(x[i:], 6)


def finish(mix: np.ndarray, rng, noise: tuple | None, phone_p: float):
    import fx
    facts = {}
    y = mix.astype(np.float32)
    if noise is not None:
        key, bg, snr = noise
        y = fx.mix_at_snr(y, bg, snr, rng)
        facts["noise"] = key
        facts["noise_snr_db"] = round(snr, 1)
    if rng.random() < phone_p:
        codec = rng.choice(["mulaw", "alaw", "gsm", "opus"])
        y = fx.bandlimit(y / (np.abs(y).max() + 1e-9) * 0.7)
        try:
            y = fx.ffmpeg_codec(y, codec, rng.choice([8, 10, 12]) if codec == "opus" else None)
        except Exception:  # noqa: BLE001
            codec = "bandlimit_only"
        facts["phone_fx"] = codec
    pk = float(np.abs(y).max() + 1e-9)
    y = y * (rng.uniform(0.5, 0.95) / pk)
    return np.clip(y, -0.99, 0.99).astype(np.float32), facts


def pick_speakers(src, rng, N: int, same_gender: bool, similar_pitch: bool, prefer_group=None):
    """N distinct speakers; same_gender -> all of one gender (when known); similar_pitch -> speaker 2 within 12 % F0
    of speaker 1 (measured)."""
    pool = list(src.spks)
    if len(pool) < N:
        return None, {}
    facts = {}
    s0 = rng.choice(pool)
    rest = [s for s in pool if s != s0]
    if prefer_group is not None and rng.random() < 0.5:
        g = [s for s in rest if prefer_group(s) == prefer_group(s0)]
        if len(g) >= N - 1:
            rest = g
            facts["same_group"] = True
    if same_gender and N >= 2:
        g0 = src.gender_of(s0)
        if g0 is not None:
            g = [s for s in rest if src.gender_of(s) == g0] if len(rest) <= 120 else \
                [s for s in rng.sample(rest, min(len(rest), 60 + 12 * N)) if src.gender_of(s) == g0]
            if len(g) >= N - 1:
                rest = g
    chosen = [s0]
    if similar_pitch and N >= 2 and src.f0(s0):
        f0 = src.f0(s0)
        for s in rng.sample(rest, min(len(rest), 30)):
            fs = src.f0(s)
            if fs and max(fs, f0) / min(fs, f0) <= 1.12:
                chosen.append(s)
                facts["similar_pitch_target"] = True
                break
    rest = [s for s in rest if s not in chosen]
    if len(rest) < N - len(chosen):
        return None, {}
    chosen += rng.sample(rest, N - len(chosen))
    rng.shuffle(chosen)
    return chosen, facts


def build_clip(rng, src, N: int, split: str) -> dict | None:
    """Design + cut + place one clip. Returns dict with pieces [(spk_idx, start_s, audio)], T, facts."""
    T = L.draw_T(rng, N)
    same_gender = N >= 2 and rng.random() < 0.55
    similar = N >= 2 and rng.random() < 0.3
    prefer = src.meeting_of if getattr(src, "corpus", "") == "ami" and N <= 4 else None
    spks, pf = pick_speakers(src, rng, N, same_gender, similar, prefer)
    if spks is None:
        return None
    d = L.design_timeline(rng, N, T, short_phrase=rng.random() < 0.45, overlap=rng.random() < 0.11,
                          seg_mode="free" if rng.random() < 0.7 else "natural")
    gains = {j: rng.uniform(-4, 4) for j in range(N)}
    home = {}
    multi_src = rng.random() < 0.3  # the same person's turns may come from different chapters / calls / meetings
    used, srcs = set(), []
    pieces, t = [], d["lead"]
    plan_iv = []
    for p, j in enumerate(d["seq"]):
        tgt = d["dur"][p]
        got = src.piece(spks[j], tgt, rng, used, None if (multi_src or j not in home) else home[j])
        if got is None:
            return None
        x, sid, ch, uid = got
        home.setdefault(j, ch)
        used.add(uid)
        srcs.append(sid)
        x = level(x, -22 + gains[j] + rng.uniform(-1.5, 1.5))
        parts = [x]
        if p in d["splits"]:
            sp = split_audio(x, d["splits"][p][0])
            if sp is not None:
                parts = [sp[0], None, sp[1]]
        for q in parts:
            if q is None:
                t += d["splits"][p][1]
                continue
            pieces.append((j, t, q))
            plan_iv.append((t, t + len(q) / SR, j))
            t += len(q) / SR
        if p < len(d["gaps"]):
            t += d["gaps"][p]
            t = max(t, 0.0)
    T_end = max(b for a, b, _ in plan_iv) + d["tail"]
    return {"pieces": pieces, "T": T_end, "speakers": spks, "design": d, "source_ids": sorted(set(srcs)),
            "pick_facts": pf, "same_gender_target": same_gender, "plan_iv": plan_iv}


def render(pieces, T: float):
    n = int(math.ceil(T * SR))
    tracks = {}
    for tr, st, x in pieces:
        a = int(round(st * SR))
        if tr not in tracks:
            tracks[tr] = np.zeros(n, np.float32)
        b = min(n, a + len(x))
        tracks[tr][a:b] += x[: b - a]
    mix = np.zeros(n, np.float32)
    for v in tracks.values():
        mix += v
    return mix, tracks


# Items

class ItemMaker:
    def __init__(self):
        self.items = []
        self.yes = Counter()

    def add(self, clip, qtype, q, opts, gold, view, rng, extra=None) -> bool:
        if len(clip["_n"]) >= 4:
            return False
        tgt = [1.0 if i == gold else 0.0 for i in range(len(opts))]
        meta = {"qtype": qtype, "view": view, "family": "speakers", **clip["meta"], **(extra or {})}
        r = L.NoShuffle() if qtype == "score" else rng
        it = C.make_item(item_id=f"{BUILDER}_{clip['key']}_{len(clip['_n'])}", set_name=clip["set"],
                         source=f"{BUILDER}_{clip['meta']['corpus']}", question=q, options=opts, target=tgt, gold=gold,
                         rng=r, audio={"bank": clip["bank"], "key": clip["key"]}, channel="clean", state_text=None, meta=meta)
        clip["_n"].append(it["id"])
        self.items.append(it)
        return True

    def want_yes(self, kind, split, rng):
        y, n = self.yes[(kind, split, True)], self.yes[(kind, split, False)]
        return (rng.random() < 0.5) if y == n else (y < n)


def clip_items(mk: ItemMaker, clip, rng):
    N = clip["meta"]["n_speakers"]
    views = []
    if rng.random() < 0.85:
        views.append("choice")
    if rng.random() < 0.75:
        views.append("noul")
    if rng.random() < (0.5 if N <= 3 else 0.17):
        views.append("score")
    if not views:
        views.append("choice")
    if rng.random() < 0.15:
        views.append("noul")
    kinds_used = set()
    for v in views:
        if v == "choice":
            vals = L.count_window(N, 1, 10, rng)
            fm = rng.choice(Q.COUNT_FMT)
            opts = [Q.fmt_count(x, fm) for x in vals]
            mk.add(clip, "choice", rng.choice(Q.CHOICE), opts, vals.index(N), "count", rng)
        elif v == "score":
            lv = rng.choice(Q.SCORE_LEVELS)
            mk.add(clip, "score", rng.choice(Q.SCORE), list(lv), L.score_index(N), "score4", rng)
        else:
            kinds = [k for k in ("exactly", "more", "fewer", "atleast") if k not in kinds_used]
            kind = rng.choices(kinds, weights=[{"exactly": 0.45, "more": 0.3, "fewer": 0.15, "atleast": 0.1}[k] for k in kinds])[0]
            kinds_used.add(kind)
            want = mk.want_yes(kind, clip["set"], rng)
            n_cmp = L.noul_count(N, kind, want, rng)
            if n_cmp is None:
                continue
            truth = L.noul_truth(N, kind, n_cmp)
            q = Q.noul_question(rng.choice(Q.NOUL[kind]), n_cmp, rng.random() < 0.4)
            if mk.add(clip, "noul", q, ["yes", "no"], 0 if truth else 1, f"noul_{kind}", rng, {"N_cmp": n_cmp}):
                mk.yes[(kind, clip["set"], truth)] += 1


# One shard

CORPUS_W = {"libritts_r": 0.46, "librispeech": 0.16, "hvb": 0.17, "ami": 0.21}


def render_shard(shard: int, n_clips: int, out_dir: str, ls_train: list[str], ls_cal: list[str]) -> dict:
    import fx
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bank = f"{BUILDER}_b{shard:02d}"
    if (out / f"items_{shard:02d}.jsonl").exists() and (out / f"{bank}.npy").exists():
        return {"shard": shard, "skipped": True}
    rng = random.Random(f"{C.SEED}:{BUILDER}:{shard}")
    adi = adi_ids()
    ltr_files = sorted(POOL_LTR.glob("*.npy"))
    srcs = {
        ("libritts_r", "train"): UttPool("libritts_r", [p for p in ltr_files if not p.stem.startswith("dev")]),
        ("libritts_r", "CAL"): UttPool("libritts_r", [p for p in ltr_files if p.stem.startswith("dev")]),
        ("librispeech", "train"): UttPool("librispeech", [POOL_LS / f"{t}.npy" for t in ls_train]),
        ("librispeech", "CAL"): UttPool("librispeech", [POOL_LS / f"{t}.npy" for t in ls_cal]),
        ("hvb", "train"): HVB(adi["hvb_calls"], "train", random.Random(f"{C.SEED}:hvb:{shard}"), 260),
        ("hvb", "CAL"): HVB(adi["hvb_calls"], "CAL", random.Random(f"{C.SEED}:hvbc:{shard}"), 60),
        ("ami", "train"): AMI("train", random.Random(f"{C.SEED}:ami:{shard}"), 70),
        ("ami", "CAL"): AMI("CAL", random.Random(f"{C.SEED}:amic:{shard}"), 70),
    }
    t_setup = time.time() - t0
    noise_ok, noise_rep = musan_noise_keys()
    nidx = json.loads((RAW_PHONE / "musan_noise.json").read_text())
    ndata = np.load(RAW_PHONE / "musan_noise.npy", mmap_mode="r")
    nkeys = {"train": sorted(k for k in noise_ok if k in nidx and nidx[k][1] >= 2 * SR and L.stable_frac(k, "nz") >= 0.15),
             "CAL": sorted(k for k in noise_ok if k in nidx and nidx[k][1] >= 2 * SR and L.stable_frac(k, "nz") < 0.15)}
    mk = ItemMaker()
    clips_audio, stats, drops = {}, Counter(), Counter()
    for ci in range(n_clips):
        crng = random.Random(f"{C.SEED}:{BUILDER}:{shard}:{ci}")
        split = "CAL" if crng.random() < 0.035 else "train"
        corpus = crng.choices(list(CORPUS_W), weights=list(CORPUS_W.values()))[0]
        N = crng.randint(1, 8)
        src = srcs[(corpus, split)]
        if len(src.spks) < N + 2:
            corpus = "libritts_r"
            src = srcs[(corpus, split)]
        try:
            clip = build_clip(crng, src, N, split)
        except Exception as e:  # noqa: BLE001
            drops[f"error:{corpus}:{type(e).__name__}:{str(e)[:60]}"] += 1
            continue
        if clip is None:
            drops[f"none:{corpus}"] += 1
            continue
        mix, tracks = render(clip["pieces"], clip["T"])
        # every placed speaker must be audible on its own track (energy VAD)
        vad = {k: L.energy_vad(v, SR) for k, v in tracks.items()}
        talk = {k: L.total(v) for k, v in vad.items()}
        if len(tracks) != N or any(talk[k] < 0.2 for k in tracks):
            drops[f"vad_speaker:{corpus}"] += 1
            continue
        all_iv = [iv for v in vad.values() for iv in v]
        n_seg = L.count_segments(all_iv, L.SEG_GAP_S)
        d = clip["design"]
        short_spk = [j for j in d["short"] if talk[j] <= L.SHORT_HI + 0.15]
        dur_by_spk = defaultdict(float)
        for j, st, x in clip["pieces"]:
            dur_by_spk[j] += len(x) / SR
        f0s = {}
        for j in range(N):
            xs = np.concatenate([x for jj, st, x in clip["pieces"] if jj == j])
            f0s[j] = L.median_f0(xs, SR)
        fv = [v for v in f0s.values() if v]
        min_ratio = None
        if len(fv) >= 2:
            fv = sorted(fv)
            min_ratio = min(fv[i + 1] / fv[i] for i in range(len(fv) - 1))
        genders = [src.gender_of(s) for s in clip["speakers"]]
        known = [g for g in genders if g is not None]
        single_gender = (len(set(known)) == 1 and len(known) == N) if N >= 2 else None
        # overlap actually rendered
        ivs = sorted(clip["plan_iv"])
        ov = 0.0
        for i in range(len(ivs) - 1):
            a, b, j = ivs[i]
            c, e, k = ivs[i + 1]
            if k != j:
                ov = max(ov, min(b, e) - c)
        noise = None
        if crng.random() < 0.15 and nkeys[split]:
            k = crng.choice(nkeys[split])
            s0, ln = nidx[k]
            noise = (k, np.array(ndata[s0: s0 + ln], dtype=np.float32), crng.uniform(10, 30))
        phone_p = 0.1 if corpus == "hvb" else 0.2
        y, facts = finish(mix, crng, noise, phone_p)
        dur = len(y) / SR
        if not (1.0 <= dur <= 30.0):
            drops[f"len:{corpus}"] += 1
            continue
        key = f"{corpus}_{shard:02d}_{ci:05d}"
        meta = {"corpus": corpus, "n_speakers": N, "speakers": clip["speakers"], "source_ids": clip["source_ids"][:40],
                "licence": LICENCE[corpus], "clip_kind": "stitched",
                "label_rule": "N = number of distinct real speaker ids placed by construction; every speaker re-checked "
                              "audible on its own track (energy VAD >= 0.2 s)",
                "dur_s": round(dur, 3), "T_target": round(d["T_plan"], 2), "n_segments": n_seg,
                "n_segments_02": L.count_segments(all_iv, 0.2), "n_segments_05": L.count_segments(all_iv, 0.5),
                "n_pieces": len(d["seq"]), "turns_per_speaker": d["counts"], "seg_target": d["s_target"],
                "short_phrase_speakers": len(short_spk), "talk_s": [round(talk[j], 2) for j in range(N)],
                "genders": genders, "single_gender": single_gender, "f0_hz": [None if f0s[j] is None else round(f0s[j], 1) for j in range(N)],
                "min_f0_ratio": None if min_ratio is None else round(min_ratio, 3),
                "similar_pitch_pair": (min_ratio is not None and min_ratio <= 1.15),
                "overlap_s": round(max(0.0, ov), 3), **facts}
        clip_rec = {"set": split, "key": key, "bank": bank, "_n": [], "meta": meta}
        n0 = len(mk.items)
        clip_items(mk, clip_rec, crng)
        if len(mk.items) > n0:
            clips_audio[key] = y
            stats[f"clips:{corpus}:{split}"] += 1
    C.write_audio_bank(out, bank, clips_audio)
    C.write_jsonl(out / f"items_{shard:02d}.jsonl", mk.items)
    hours = sum(len(v) for v in clips_audio.values()) / SR / 3600
    return {"shard": shard, "items": len(mk.items), "clips": len(clips_audio), "hours": round(hours, 3),
            "stats": dict(stats), "drops": dict(drops), "setup_s": round(t_setup, 1), "elapsed_s": round(time.time() - t0, 1),
            "pool_speakers": {f"{a}|{b}": len(s.spks) for (a, b), s in srcs.items()}, "noise_keys": {k: len(v) for k, v in nkeys.items()},
            "noise_licence_report": noise_rep, "hvb_excluded_convs": srcs[("hvb", "train")].n_excluded}
