"""sb_count build steps (turn counting, speaker counting, speaker identity), run inside Modal containers.

Labels are exact by construction (we place every utterance) and re-checked with an energy VAD per track.
"""
from __future__ import annotations

import io
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import common as C
import labels as L
import questions as Q

BUILDER = "sb_count"
SR = C.SAMPLE_RATE
VOL = Path("/vol")
WORK = VOL / "stageb_work" / BUILDER
POOL = WORK / "pool"
SHARDS = WORK / "shards"
RAW_HVB = VOL / "stageb" / "sb_dialog" / "work" / "raw" / "hvb"   # read-only
RAW_AMI = VOL / "stageb" / "sb_dialog" / "work" / "raw" / "ami"   # read-only
HERE = Path(__file__).resolve().parent

LIBRI_REPO, LIBRI_REV = "mythicinfinity/libritts_r", "0d1718db351207d5a493135a62d5ed8cc4733830"
HVB_SHA = "0bd721e877c4a85d8c13ff837e68661ea6200a98"
SLUE_REPO, SLUE_REV = "asapp/slue-phase-2", "e2989c55a53593a8e39b8f8ebdb47ccaccbe484a"
KOKORO_REPO, KOKORO_REV = "hexgrad/Kokoro-82M", "f3ff3571791e39611d31c381e3a41a3af07b4987"
SOURCES = {
    "libritts_r": {"name": "LibriTTS-R (train.clean.100, train.clean.360 / train.other.500 subsets -> train; dev.clean -> CAL)",
                   "repo": LIBRI_REPO, "rev": LIBRI_REV,
                   "licence": "CC BY 4.0 (HF cardData license: cc-by-4.0; derived from LibriSpeech, CC BY 4.0). Speaker "
                              "gender from LibriSpeech SPEAKERS.TXT (CC BY 4.0, openslr 12 raw-metadata)."},
    "hvb": {"name": "HarperValleyBank (Gridspace-Stanford) per-speaker channels + transcripts",
            "rev": f"github.com/cricketclub/gridspace-stanford-harper-valley@{HVB_SHA} (raw copy at {RAW_HVB}, read-only)",
            "licence": "CC BY 4.0 (repo LICENSE)",
            "exclusion": f"every SLUE-HVB test conversation ({SLUE_REPO}@{SLUE_REV}) and every HVB call in our evaluation set"},
    "ami": {"name": "AMI Meeting Corpus Mix-Headset wavs + manual word alignments (ami_public_manual_1.6.2)",
            "rev": f"official mirror groups.inf.ed.ac.uk/ami (raw copy at {RAW_AMI}, read-only); HF mirror "
                   "edinburghcstr/ami@46f28f2503e2ec48f8867a84eef356c70476beab cardData.license=cc-by-4.0",
            "licence": "CC BY 4.0"},
    "kokoro": {"name": "Kokoro-82M TTS (only the announcer phrases 'Clip three.' etc., never a counted speaker)",
               "rev": f"{KOKORO_REPO}@{KOKORO_REV}", "licence": "Apache-2.0"},
    "tones": {"name": "beep separators + pink noise generated with numpy", "rev": "build.py", "licence": "CC0 (generated)"},
}
LICENCE = {"libritts_r": "CC BY 4.0", "hvb": "CC BY 4.0", "ami": "CC BY 4.0"}
ANN_VOICES_TRAIN = ["af_heart", "af_bella", "af_sarah", "am_michael", "am_adam", "am_liam", "bf_emma", "bm_daniel"]
ANN_VOICES_CAL = ["af_nova", "am_eric"]
ANN_NOUNS = Q.NOUNS


# Utterance pools

def _hf_list(prefix: str) -> list[str]:
    from huggingface_hub import HfApi
    return sorted(f for f in HfApi().list_repo_files(LIBRI_REPO, repo_type="dataset", revision=LIBRI_REV)
                  if f.startswith(prefix) and f.endswith(".parquet"))


def decode_bytes(b: bytes) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    return C.resample(x.mean(axis=1), sr, SR)


def prep_libri(split: str, shard: int, tag: str, keep: float, cap: int, min_s: float = 1.0, max_s: float = 6.5) -> dict:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    out_npy = POOL / f"{tag}_{shard:02d}.npy"
    if out_npy.exists() and out_npy.with_suffix(".jsonl").exists():
        return {"tag": tag, "shard": shard, "skipped": True}
    files = _hf_list(f"data/{split}/")
    path = hf_hub_download(LIBRI_REPO, files[shard], repo_type="dataset", revision=LIBRI_REV, cache_dir="/tmp/hf")
    gender = json.loads((HERE / "librispeech_speaker_gender.json").read_text())["gender"]
    per_spk = Counter()
    xs, recs, off = [], [], 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["audio", "text_normalized", "speaker_id", "id", "chapter_id"]):
        for r in batch.to_pylist():
            spk = str(r["speaker_id"])
            if spk not in gender or per_spk[spk] >= cap or L.stable_frac(r["id"], "sb_count") >= keep:
                continue
            x = decode_bytes(r["audio"]["bytes"])
            a, b = L.trim_bounds(x, SR)
            x = x[a:b]
            dur = len(x) / SR
            if not (min_s <= dur <= max_s) or not np.isfinite(x).all() or np.abs(x).max() < 1e-3:
                continue
            per_spk[spk] += 1
            xs.append(x.astype(np.float32))
            recs.append({"id": r["id"], "spk": f"ltr:{spk}", "gender": gender[spk], "chapter": str(r["chapter_id"]),
                         "dur": round(dur, 3), "off": off, "n": len(x), "split": split, "text": r["text_normalized"]})
            off += len(x)
    POOL.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, np.concatenate(xs) if xs else np.zeros(0, np.float32))
    with open(out_npy.with_suffix(".jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return {"tag": tag, "shard": shard, "utts": len(recs), "speakers": len(per_spk), "file": files[shard]}


class LibriPool:
    """Utterances by speaker from a list of pool shards (memory-mapped)."""

    def __init__(self, tags: list[str]):
        self.data, self.by_spk, self.gender = {}, defaultdict(list), {}
        for t in tags:
            p = POOL / f"{t}.npy"
            if not p.exists():
                continue
            self.data[t] = np.load(p, mmap_mode="r")
            for line in open(p.with_suffix(".jsonl")):
                r = json.loads(line)
                r["tag"] = t
                self.by_spk[r["spk"]].append(r)
                self.gender[r["spk"]] = r["gender"]
        self.spks = sorted(s for s, v in self.by_spk.items() if len(v) >= 3)

    def audio(self, r) -> np.ndarray:
        return np.array(self.data[r["tag"]][r["off"]: r["off"] + r["n"]], dtype=np.float32)

    def pick_speakers(self, rng, k: int, same_gender: bool, exclude=()):
        pool = [s for s in self.spks if s not in exclude]
        if same_gender:
            g = rng.choice(["female", "male"])
            pool = [s for s in pool if self.gender[s] == g] or pool
        return rng.sample(pool, k) if len(pool) >= k else None

    def utt(self, rng, spk, max_s: float = 6.5, min_s: float = 1.0, avoid=()):
        c = [r for r in self.by_spk[spk] if min_s <= r["dur"] <= max_s and r["id"] not in avoid]
        if not c:
            return None
        return rng.choice(c)


# HVB / AMI

def hvb_parse(segs: list[dict]) -> list[dict]:
    """HVB transcript json -> turns (consecutive same-role speech segments) {role, s, e, words}. (sb_dialog's rules.)"""
    import re
    noise = re.compile(r"^(\s*\[[^\]]*\]\s*)*$")
    out = []
    for g in segs:
        role = g.get("speaker_role")
        if role not in ("agent", "caller"):
            continue
        s = g["start_ms"] / 1000.0
        e = s + g["duration_ms"] / 1000.0
        text = (g.get("human_transcript") or g.get("transcript") or "").strip()
        speech = bool(text) and not noise.match(text)
        words = [(s + o / 1000.0, s + (o + d) / 1000.0)
                 for o, d in zip(g.get("word_offsets_ms") or [], g.get("word_durations_ms") or []) if d > 0]
        if speech and not words:
            words = [(s, e)]
        out.append({"role": role, "s": s, "e": e, "words": words if speech else [], "speech": speech})
    out.sort(key=lambda g: (g["s"], g["e"]))
    turns = []
    for g in out:
        if turns and turns[-1]["role"] == g["role"]:
            t = turns[-1]
        elif g["speech"]:
            t = {"role": g["role"], "s": g["s"], "e": g["e"], "words": []}
            turns.append(t)
        else:
            continue
        t["e"] = max(t["e"], g["e"])
        t["words"] += g["words"]
    for t in turns:
        if t["words"]:
            t["ws"], t["we"] = min(a for a, b in t["words"]), max(b for a, b in t["words"])
            t["speech_s"] = L.total(L.merge(t["words"]))
        else:
            t["ws"], t["we"], t["speech_s"] = t["s"], t["e"], 0.0
    return turns


def hvb_speaker_split(conv_spk: dict, frac: float):
    """sb_dialog's rule: CAL speaker set S grown pairwise; conversations with both speakers in S -> CAL, neither -> train,
    exactly one -> dropped."""
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


class HVB:
    def __init__(self, adi_calls: set):
        slue = set(json.loads((RAW_HVB / "slue_ids.json").read_text())["test"])
        self.excluded = Counter()
        conv_spk, self.role_spk = {}, {}
        for p in sorted((RAW_HVB / "transcript").glob("*.json")):
            sid = p.stem
            if sid in slue or sid in adi_calls:
                self.excluded["slue_test_or_adi"] += 1
                continue
            md = json.loads((RAW_HVB / "metadata" / f"{sid}.json").read_text())
            self.role_spk[sid] = {r: f"hvb_spk:{md[r]['speaker_id']}" for r in ("agent", "caller")}
            conv_spk[sid] = set(self.role_spk[sid].values())
        self.split, self.cal_spk = hvb_speaker_split(conv_spk, 0.03)
        self.sids = {s: sorted(c for c, v in self.split.items() if v == s) for s in ("train", "CAL")}
        self._turns, self._audio = {}, {}

    def turns(self, sid):
        if sid not in self._turns:
            self._turns[sid] = hvb_parse(json.loads((RAW_HVB / "transcript" / f"{sid}.json").read_text()))
        return self._turns[sid]

    def chan(self, sid, role):
        import soundfile as sf
        k = (sid, role)
        if k not in self._audio:
            if len(self._audio) > 64:
                self._audio.clear()
            x, sr = sf.read(str(RAW_HVB / "audio" / role / f"{sid}.wav"), dtype="float32", always_2d=True)
            self._audio[k] = C.resample(x.mean(1), sr, SR)
        return self._audio[k]

    def excerpt_index(self, split: str, rng, n_convs: int):
        """{speaker: [(sid, role, s, e)]} single-speaker turns with >= 1.5 s speech, <= 6 s long."""
        idx = defaultdict(list)
        for sid in rng.sample(self.sids[split], min(n_convs, len(self.sids[split]))):
            for t in self.turns(sid):
                d = t["we"] - t["ws"]
                if t["speech_s"] >= 1.5 and 1.6 <= d <= 6.0:
                    idx[self.role_spk[sid][t["role"]]].append((sid, t["role"], t["ws"] - 0.05, t["we"] + 0.05))
        return idx


class AMI:
    def __init__(self):
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
        cal = L.cal_groups(groups, 0.06, random.Random(f"{C.SEED}:sb_count:ami_cal"))
        self.split = {m: ("CAL" if m in cal else "train") for m in self.ann}
        cal_spk = {g for m in cal for g in self.ann[m]}
        for m in list(self.split):  # safety: train meetings never share a CAL speaker
            if self.split[m] == "train" and set(self.ann[m]) & cal_spk:
                self.split[m] = None
        self.meetings = {s: sorted(m for m, v in self.split.items() if v == s) for s in ("train", "CAL")}

    def read(self, m, t0, t1):
        import soundfile as sf
        x, sr = sf.read(str(RAW_AMI / f"{m}.wav"), start=int(t0 * 16000), stop=int(t1 * 16000), dtype="float32", always_2d=True)
        x = x.mean(1)
        return C.resample(x, sr, SR) if sr != SR else x

    @staticmethod
    def gender(g):
        return "female" if g.startswith("F") else ("male" if g.startswith("M") else None)

    def masks(self, m):
        """10 ms activity masks per speaker (inc = words + vocal sounds) for meeting m (cached)."""
        if getattr(self, "_mk", None) is None:
            self._mk = {}
        if m not in self._mk:
            if len(self._mk) > 24:
                self._mk.clear()
            n = int(self.dur[m] * 100) + 10
            mk = {}
            for g, v in self.ann[m].items():
                a = np.zeros(n, bool)
                for s, e in v["inc"]:
                    a[max(0, int(s * 100)): min(n, int(math.ceil(e * 100)))] = True
                mk[g] = a
            self._mk[m] = mk
        return self._mk[m]

    def window(self, m, rng, L_s):
        """A natural window -> dict(t0, t1, runs, spk, turns or None) or None (ambiguous)."""
        D = self.dur[m]
        if D < L_s + 2:
            return None
        mk = self.masks(m)
        busy = np.zeros_like(next(iter(mk.values())))
        for a in mk.values():
            busy |= a
        t0 = rng.uniform(1.0, D - L_s - 1.0)
        t1 = t0 + L_s

        def snap(t):
            c0 = int(t * 100)
            for d in range(0, 200, 5):
                for c in (c0 - d, c0 + d):
                    if 15 <= c < len(busy) - 15 and not busy[c - 15: c + 15].any():
                        return c / 100.0
            return None

        t0, t1 = snap(t0), snap(t1)
        if t0 is None or t1 is None or not (3.0 <= t1 - t0 <= 24.0):
            return None
        counted, runs = [], []
        for g, v in self.ann[m].items():
            sp = L.clip_iv(v["sp"], t0, t1)
            inc = L.clip_iv(v["inc"], t0, t1)
            st = L.total(sp)
            if st >= 0.6:
                counted.append(g)
            elif L.total(inc) > 0:
                return None  # a barely-audible speaker (a word or a laugh): count ambiguous
            for a, b in L.merge(inc, 0.3):
                runs.append((a, b, g, L.total(L.clip_iv(sp, a, b))))
        if not counted:
            return None
        ok_turns = all(b - a >= 0.5 and s >= 0.3 for a, b, g, s in runs)
        r3 = [(a, b, g) for a, b, g, s in runs]
        if ok_turns and (L.contained(r3, 0.6) or L.max_overlap(r3) > 1.0):
            ok_turns = False
        return {"t0": t0, "t1": t1, "spk": sorted(counted), "runs": r3,
                "turns": L.count_turns(r3) if ok_turns else None}

    def excerpt_index(self, split: str, max_meetings: int = 60, rng=None):
        idx = defaultdict(list)
        ms = list(self.meetings[split])
        if rng is not None and len(ms) > max_meetings:
            ms = rng.sample(ms, max_meetings)
        for m in ms:
            mk = self.masks(m)
            for g, v in self.ann[m].items():
                others = np.zeros_like(mk[g])
                for h, a in mk.items():
                    if h != g:
                        others |= a
                for a, b in L.merge(v["sp"], 0.35):
                    if not (1.6 <= b - a <= 5.5):
                        continue
                    if others[max(0, int((a - 0.3) * 100)): int((b + 0.3) * 100) + 1].any():
                        continue
                    idx[g].append((m, a - 0.05, b + 0.05))
        return idx


# Rendering

def beep(rng) -> np.ndarray:
    f = rng.choice([600, 750, 880, 1000, 1200])
    d = rng.uniform(0.2, 0.4)
    t = np.arange(int(d * SR)) / SR
    x = 0.4 * np.sin(2 * np.pi * f * t)
    r = int(0.01 * SR)
    x[:r] *= np.linspace(0, 1, r)
    x[-r:] *= np.linspace(1, 0, r)
    return x.astype(np.float32)


def pink(n: int, rng) -> np.ndarray:
    g = np.random.default_rng(rng.randrange(1 << 30))
    X = np.fft.rfft(g.standard_normal(n))
    f = np.arange(len(X))
    f[0] = 1
    y = np.fft.irfft(X / np.sqrt(f), n)
    return (y / (np.std(y) + 1e-9)).astype(np.float32)


def active_rms(x: np.ndarray) -> float:
    db = L.frame_db(x, SR)
    if len(db) == 0:
        return 1e-4
    thr = np.percentile(db, 95) - 25
    act = db[db > thr]
    return float(10 ** (np.mean(act) / 20)) if len(act) else 1e-4


def level(x: np.ndarray, target_db: float) -> np.ndarray:
    return (x * (10 ** (target_db / 20) / (active_rms(x) + 1e-9))).astype(np.float32)


def render(pieces, T: float):
    """pieces [(track, start_s, audio)] -> (mix, {track: track audio})"""
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


def finish(mix: np.ndarray, rng, phone_p: float, noise_p: float = 0.3):
    """Optional pink-noise bed, optional phone fx (8 kHz band + codec, baked in), peak U(0.5, 0.95)."""
    import fx
    facts = {}
    y = mix.astype(np.float32)
    if rng.random() < noise_p:
        snr = rng.uniform(22, 40)
        y = y + pink(len(y), rng) * (active_rms(y) / 10 ** (snr / 20))
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


def vad_runs(tracks: dict):
    runs = []
    for k, x in tracks.items():
        for a, b in L.energy_vad(x, SR):
            runs.append((a, b, k))
    return runs


# Clip builders

def gaps_rng(rng):
    return rng.uniform(0.15, 1.1) if rng.random() < 0.85 else rng.uniform(1.1, 2.2)


def dialog_libri(rng, pool: LibriPool):
    """2-12 turns, 2-4 speakers (consecutive turns by different people), 25 % of turns = 2 sentences with a pause,
    ~12 % of speaker changes overlap by 0.2-0.7 s."""
    n = rng.randint(2, 12)
    k = rng.randint(2, min(4, n))
    spks = pool.pick_speakers(rng, k, same_gender=rng.random() < 0.4)
    if not spks:
        return None
    for _ in range(50):
        seq = [rng.randrange(k)]
        for i in range(1, n):
            seq.append(rng.choice([j for j in range(k) if j != seq[-1]]))
        if len(set(seq)) == k:
            break
    else:
        return None
    max_turn = max(1.4, min(6.0, 26.0 / n))
    gains = {j: rng.uniform(-3, 3) for j in range(k)}
    pieces, runs, t, used = [], [], rng.uniform(0.2, 0.8), set()
    prev_last = None
    for i, j in enumerate(seq):
        parts = 2 if (rng.random() < 0.25 and max_turn >= 3.0) else 1
        utts = []
        for _p in range(parts):
            r = pool.utt(rng, spks[j], max_s=max_turn / parts, avoid=used)
            if r is None:
                r = pool.utt(rng, spks[j], max_s=6.5, avoid=used)
            if r is None:
                return None
            used.add(r["id"])
            utts.append(r)
        if i > 0:
            if prev_last is not None and prev_last >= 1.6 and utts[0]["dur"] >= 1.4 and rng.random() < 0.12:
                t = t - rng.uniform(0.2, min(0.7, prev_last - 0.8))
            else:
                t = t + gaps_rng(rng)
        for pi, r in enumerate(utts):
            if pi > 0:
                t += rng.uniform(0.25, 0.8)
            x = level(pool.audio(r), -22 + gains[j] + rng.uniform(-1.5, 1.5))
            pieces.append((j, t, x))
            runs.append((t, t + len(x) / SR, j))
            t = t + len(x) / SR
        prev_last = utts[-1]["dur"]
        t = max(t, max(b for a, b, _ in runs))
    T = max(b for a, b, _ in runs) + rng.uniform(0.2, 0.8)
    return {"kind": "dialog", "corpus": "libritts_r", "pieces": pieces, "runs": runs, "T": T, "n_turns": n, "k": k,
            "speakers": spks, "source_ids": sorted(used), "genders": [pool.gender[s] for s in spks]}


def dialog_hvb(rng, hvb: HVB, split: str):
    n = rng.randint(2, 12)
    for _ in range(30):
        sid = rng.choice(hvb.sids[split])
        turns = hvb.turns(sid)
        if len(turns) < n:
            continue
        a = rng.randrange(0, len(turns) - n + 1)
        sel = turns[a:a + n]
        if any(t["speech_s"] < 0.4 or t["we"] - t["ws"] > 9.0 for t in sel):
            continue
        if sum(t["we"] - t["ws"] for t in sel) > 40.0:
            continue
        break
    else:
        return None
    pieces, runs, t = [], [], rng.uniform(0.2, 0.8)
    prev = None
    for i, tr in enumerate(sel):
        d = tr["we"] - tr["ws"]
        if i > 0:
            if prev is not None and prev >= 1.6 and d >= 1.4 and rng.random() < 0.1:
                t -= rng.uniform(0.2, min(0.6, prev - 0.8))
            else:
                t += gaps_rng(rng)
        ch = hvb.chan(sid, tr["role"])
        x = ch[int((tr["ws"] - 0.04) * SR): int((tr["we"] + 0.04) * SR)].copy()
        if len(x) < 0.2 * SR:
            return None
        f = int(0.01 * SR)
        x[:f] *= np.linspace(0, 1, f)
        x[-f:] *= np.linspace(1, 0, f)
        pieces.append((tr["role"], t, x))
        # word-level speech of this turn, shifted to clip time
        for wa, wb in L.merge(tr["words"], 0.0):
            runs.append((t + 0.04 + wa - tr["ws"], t + 0.04 + wb - tr["ws"], tr["role"]))
        t = t + len(x) / SR
        prev = d
        t = max(t, max(b for a, b, _ in runs))
    T = max(b for a, b, _ in runs) + rng.uniform(0.2, 0.8)
    spk = hvb.role_spk[sid]
    order = L.speaker_order(runs)
    return {"kind": "dialog", "corpus": "hvb", "pieces": pieces, "runs": runs, "T": T, "n_turns": n, "k": len(set(r[2] for r in runs)),
            "speakers": [spk[r] for r in order], "source_ids": [f"hvb:{sid}"], "genders": [None, None]}


def multi_libri(rng, pool: LibriPool, k: int):
    """k distinct speakers (1-8), each 1-3 sentences, order shuffled (same person may talk twice in a row)."""
    spks = pool.pick_speakers(rng, k, same_gender=rng.random() < 0.35)
    if not spks:
        return None
    m = rng.randint(k, min(12, k + 3)) if k > 1 else rng.randint(1, 4)
    seq = list(range(k)) + [rng.randrange(k) for _ in range(m - k)]
    rng.shuffle(seq)
    max_u = max(1.3, min(5.0, 22.0 / m))
    pieces, runs, used, t = [], [], set(), rng.uniform(0.2, 0.8)
    gains = {j: rng.uniform(-3, 3) for j in range(k)}
    for i, j in enumerate(seq):
        r = pool.utt(rng, spks[j], max_s=max_u, avoid=used) or pool.utt(rng, spks[j], avoid=used)
        if r is None:
            return None
        used.add(r["id"])
        if i > 0:
            t += rng.uniform(0.2, 1.0)
        x = level(pool.audio(r), -22 + gains[j] + rng.uniform(-1.5, 1.5))
        pieces.append((j, t, x))
        runs.append((t, t + len(x) / SR, j))
        t += len(x) / SR
    T = t + rng.uniform(0.2, 0.8)
    return {"kind": "multi", "corpus": "libritts_r", "pieces": pieces, "runs": runs, "T": T, "k": k,
            "n_turns": L.count_turns(runs), "speakers": spks, "source_ids": sorted(used),
            "genders": [pool.gender[s] for s in spks]}


def multi_excerpts(rng, idx: dict, k: int, loader, corpus: str):
    """k distinct speakers from an excerpt index (HVB / AMI single-speaker turns)."""
    spks = [s for s, v in idx.items() if v]
    if len(spks) < k:
        return None
    chosen = rng.sample(spks, k)
    m = rng.randint(k, min(10, k + 2)) if k > 1 else rng.randint(1, 3)
    seq = list(range(k)) + [rng.randrange(k) for _ in range(m - k)]
    rng.shuffle(seq)
    pieces, runs, src, t = [], [], set(), rng.uniform(0.2, 0.8)
    used = set()
    for i, j in enumerate(seq):
        c = [e for e in idx[chosen[j]] if e not in used]
        if not c:
            return None
        e = rng.choice(c)
        used.add(e)
        x = loader(e)
        if i > 0:
            t += rng.uniform(0.2, 1.0)
        x = level(x, -22 + rng.uniform(-3, 3))
        pieces.append((j, t, x))
        runs.append((t, t + len(x) / SR, j))
        src.add(f"{corpus}:{e[0]}")
        t += len(x) / SR
        if t > 40:
            return None
    T = t + rng.uniform(0.2, 0.8)
    return {"kind": "multi", "corpus": corpus, "pieces": pieces, "runs": runs, "T": T, "k": k,
            "n_turns": L.count_turns(runs), "speakers": chosen, "source_ids": sorted(src), "genders": [None] * k}


def numbered_set(rng, segs, sep: str, noun: str, ann: dict | None):
    """segs [(speaker index, audio)] -> pieces with separators; returns pieces, spans [(s, e)], T."""
    pieces, spans = [], []
    t = rng.uniform(0.2, 0.6)
    for i, (j, x) in enumerate(segs):
        if sep == "beep":
            if i > 0:
                t += rng.uniform(0.25, 0.5)
                b = beep(rng)
                pieces.append(("sep", t, b))
                t += len(b) / SR + rng.uniform(0.25, 0.5)
        elif sep == "spoken":
            a = ann[f"{noun}|{i + 1}"]
            t += rng.uniform(0.3, 0.6) if i > 0 else 0.0
            pieces.append(("sep", t, level(a, -22 + rng.uniform(-2, 2))))
            t += len(a) / SR + rng.uniform(0.3, 0.6)
        else:
            if i > 0:
                t += rng.uniform(1.0, 1.8)
        pieces.append((j, t, x))
        spans.append((t, t + len(x) / SR))
        t += len(x) / SR
    return pieces, spans, t + rng.uniform(0.2, 0.6)


def spkid_clip(rng, getter, K: int, no_match: bool, sep: str, noun: str, ann):
    """K numbered clips; clip N's speaker appears in exactly one other clip M (unless no_match).
    getter(rng, n_distinct, same_gender) -> list of speaker groups: [[audio, ...]] or None."""
    N = rng.randint(1, K)
    M = rng.choice([i for i in range(1, K + 1) if i != N])
    n_distinct = K if no_match else K - 1
    got = getter(rng, n_distinct, rng.random() < 0.6, 2 if not no_match else 1)
    if got is None:
        return None
    spk_ids, groups, srcs = got
    assign = {}
    others = list(range(1, n_distinct))
    rng.shuffle(others)
    for c in range(1, K + 1):
        if c == N:
            assign[c] = (0, 0)
        elif c == M and not no_match:
            assign[c] = (0, 1)
        else:
            assign[c] = (others.pop(), 0)
    segs = [(f"s{assign[c][0]}", groups[assign[c][0]][assign[c][1]]) for c in range(1, K + 1)]
    pieces, spans, T = numbered_set(rng, segs, sep, noun, ann)
    clip_spk = [spk_ids[assign[c][0]] for c in range(1, K + 1)]
    return {"kind": "spkid", "pieces": pieces, "T": T, "K": K, "N": N, "M": None if no_match else M, "sep": sep,
            "noun": noun, "clip_speakers": clip_spk, "speakers": sorted(set(spk_ids)), "source_ids": srcs,
            "spans": spans, "n_distinct": n_distinct}


def pair_clip(rng, getter, same: bool, sep: str, noun: str, ann):
    got = getter(rng, 1 if same else 2, rng.random() < 0.6, 2 if same else 1)
    if got is None:
        return None
    spk_ids, groups, srcs = got
    segs = [("s0", groups[0][0]), ("s0" if same else "s1", groups[0][1] if same else groups[1][0])]
    pieces, spans, T = numbered_set(rng, segs, sep, noun, ann)
    return {"kind": "pair", "pieces": pieces, "T": T, "same": same, "sep": sep, "noun": noun,
            "clip_speakers": [spk_ids[0], spk_ids[0] if same else spk_ids[1]], "speakers": sorted(set(spk_ids)),
            "source_ids": srcs, "spans": spans}


def libri_getter(pool: LibriPool):
    def get(rng, n, same_gender, n_first):
        spks = pool.pick_speakers(rng, n, same_gender)
        if not spks:
            return None
        groups, srcs = [], []
        for i, s in enumerate(spks):
            need = n_first if i == 0 else 1
            g, used_ch = [], set()
            for _ in range(need):
                cands = [r for r in pool.by_spk[s] if 1.4 <= r["dur"] <= 4.5 and r["id"] not in srcs]
                if i == 0 and g:
                    diff = [r for r in cands if r["chapter"] not in used_ch]
                    cands = diff or cands
                if not cands:
                    return None
                r = rng.choice(cands)
                used_ch.add(r["chapter"])
                srcs.append(r["id"])
                g.append(level(pool.audio(r), -22 + rng.uniform(-3, 3)))
            groups.append(g)
        return spks, groups, srcs
    return get


def excerpt_getter(idx: dict, loader, corpus: str, gender_of=None, prefer_same_group=None):
    spks = sorted(s for s, v in idx.items() if len(v) >= 1)
    multi = sorted(s for s, v in idx.items() if len(v) >= 2)

    def get(rng, n, same_gender, n_first):
        first_pool = multi if n_first >= 2 else spks
        if not first_pool or len(spks) < n:
            return None
        s0 = rng.choice(first_pool)
        rest = [s for s in spks if s != s0]
        if same_gender and gender_of and gender_of(s0):
            rg = [s for s in rest if gender_of(s) == gender_of(s0)]
            rest = rg if len(rg) >= n - 1 else rest
        if prefer_same_group and rng.random() < 0.5:
            grp = [s for s in rest if prefer_same_group(s, s0, idx)]
            if len(grp) >= n - 1:
                rest = grp
        if len(rest) < n - 1:
            return None
        chosen = [s0] + rng.sample(rest, n - 1)
        groups, srcs = [], []
        for i, s in enumerate(chosen):
            need = n_first if i == 0 else 1
            es = rng.sample(idx[s], need) if len(idx[s]) >= need else None
            if es is None:
                return None
            if need == 2 and es[0][0] == es[1][0] and len({e[0] for e in idx[s]}) > 1 and rng.random() < 0.7:
                other = [e for e in idx[s] if e[0] != es[0][0]]
                es[1] = rng.choice(other)
            g = [level(loader(e), -22 + rng.uniform(-3, 3)) for e in es]
            groups.append(g)
            srcs += [f"{corpus}:{e[0]}" for e in es]
        return chosen, groups, sorted(set(srcs))
    return get


# Items

class ItemMaker:
    def __init__(self, shard: int):
        self.shard = shard
        self.items = []
        self.yes = Counter()

    def _id(self, fam, key, j):
        return f"{BUILDER}_{fam}_{key}_{j}"

    def add(self, clip, fam, qtype, q, opts, gold, view, extra=None, target=None, rng=None) -> bool:
        if len(clip.setdefault("_n", [])) >= 4:  # <= 4 items per clip
            return False
        q = singular(q)
        opts = [singular(o) for o in opts]
        tgt = target if target is not None else [1.0 if i == gold else 0.0 for i in range(len(opts))]
        meta = {"qtype": qtype, "family": fam, "view": view, "corpus": clip["corpus"], "speakers": clip["speakers"],
                "source_ids": clip["source_ids"][:40], "licence": LICENCE.get(clip["corpus"], "CC BY 4.0"),
                "clip_kind": clip["kind"], "label_rule": clip.get("label_rule", "by construction + per-track energy VAD"),
                **(clip.get("facts") or {}), **(extra or {})}
        r = L.NoShuffle() if qtype == "score" else rng
        it = C.make_item(item_id=self._id(fam, clip["key"], len(clip.setdefault("_n", []))), set_name=clip["set"],
                         source=f"{BUILDER}_{fam}", question=q, options=opts, target=tgt, gold=gold, rng=r,
                         audio={"bank": clip["bank"], "key": clip["key"]}, channel="clean", state_text=None, meta=meta)
        clip["_n"].append(it["id"])
        self.items.append(it)
        return True

    def noul(self, clip, fam, q, truth: bool, view, rng, extra=None):
        if self.add(clip, fam, "noul", q, ["yes", "no"], 0 if truth else 1, view, extra, rng=rng):
            self.yes[(fam, clip["set"], truth)] += 1

    def want_yes(self, fam, split, rng):
        y, n = self.yes[(fam, split, True)], self.yes[(fam, split, False)]
        return (rng.random() < 0.5) if y == n else (y < n)


def singular(s: str) -> str:
    import re
    s = re.sub(r"\b(1|one) (turn|speaker|voice|talker|time|change|person|clip|recording|sample)s\b", r"\1 \2", s)
    s = re.sub(r"\b(1|one) different (speaker|people|person)s?\b", r"\1 person", s)
    s = re.sub(r"\b(1|one) (distinct|separate|unique) (speaker|voice|talker|people|person)s?\b", r"\1 \2 \3", s)
    s = s.replace("1 distinct people", "1 distinct person").replace("1 separate people", "1 separate person")
    return s


def num_opts(vals, fmt, rng):
    fm = rng.choice(fmt)
    out = []
    for v in vals:
        w = L.NUM_WORDS[v] if v < len(L.NUM_WORDS) else str(v)
        s = fm.format(n=v, w=w)
        if v == 1:
            s = s.replace("1 turns", "1 turn").replace("one turns", "one turn").replace("1 speakers", "1 speaker") \
                 .replace("one speakers", "one speaker").replace("one voices", "one voice").replace("1 times", "1 time") \
                 .replace("1 different people", "1 person").replace("one changes", "one change")
        if v == 0:
            s = s.replace("0 times", "no times").replace("zero changes", "no changes")
        out.append(s)
    return out


def count_items(mk: ItemMaker, clip, fam, n, rng, lo, hi, q_choice, fmt, q_noul, bins, q_score, views, extra=None):
    """Up to len(views) items about a count n (choice / noul / score)."""
    for view in views:
        if view == "choice":
            vals = L.count_window(n, lo, hi, rng)
            mk.add(clip, fam, "choice", rng.choice(q_choice), num_opts(vals, fmt, rng), vals.index(n), "count", extra, rng=rng)
        elif view == "noul":
            want = mk.want_yes(fam, clip["set"], rng)
            r = L.noul_count(n, want, rng, lo=max(1, lo), hi=hi)
            if r is None:
                continue
            kind, N = r
            truth = {"more": n > N, "atleast": n >= N, "exactly": n == N, "fewer": n < N}[kind]
            mk.noul(clip, fam, rng.choice(q_noul[kind]).format(n=N), truth, f"noul_{kind}", rng, {**(extra or {}), "N": N})
        elif view == "score":
            b = rng.choice(bins)
            gi = L.bin_index(n, b)
            if gi < 0:
                continue
            mk.add(clip, fam, "score", rng.choice(q_score), [x[0] for x in b], gi, "bins", extra, rng=rng)


def spk_ref(i: int, fmt: str) -> str:
    return fmt.format(o=L.ORDINALS[i + 1], n=i + 1)


def dialog_items(mk, clip, rng):
    runs = clip["runs"]
    n = L.count_turns(runs)
    k = len(set(r[2] for r in runs))
    ex = {"n_turns": n, "n_speakers": k}
    v1 = rng.choice(["choice", "choice", "noul", "score"])
    views = [v1]
    if rng.random() < 0.45:
        views.append("noul" if v1 != "noul" else "choice")
    count_items(mk, clip, "turns", n, rng, 1, 14, Q.TURN_COUNT, Q.COUNT_FMT_TURN, Q.TURN_NOUL, Q.TURN_BINS, Q.TURN_SCORE, views, ex)
    if rng.random() < 0.2 and n >= 2:
        vals = L.count_window(n - 1, 0, 13, rng)
        mk.add(clip, "turns", "choice", rng.choice(Q.TURN_CHANGES), num_opts(vals, Q.COUNT_FMT_CHANGE, rng), vals.index(n - 1),
               "changes", ex, rng=rng)
    if rng.random() < 0.3:
        count_items(mk, clip, "speakers", k, rng, 1, 9, Q.SPK_COUNT, Q.COUNT_FMT_SPK, Q.SPK_NOUL, Q.SPK_BINS, Q.SPK_SCORE,
                    [rng.choice(["choice", "noul"])], ex)
    # who spoke most
    order = L.speaker_order(runs)
    if k >= 2 and len(clip["_n"]) < 4:
        tt = L.talk_time(runs)
        tp = L.turns_per_speaker(runs)
        fmt = rng.choice(Q.SPK_REF)
        opts = [spk_ref(i, fmt) for i in range(k)]
        if rng.random() < 0.5:
            w = L.argmax_margin(tt, 1.3, 1.5)
            vt = clip.get("vad_talk")
            if w is not None and vt is not None and L.argmax_margin(vt, 1.2, 1.0) != w:
                w = None  # the energy VAD does not confirm the longest talker -> do not ask
            if w is not None:
                mk.add(clip, "most", "choice", rng.choice(Q.MOST_TIME), opts, order.index(w), "longest",
                       {**ex, "winner": order.index(w) + 1, "talk_s": {str(order.index(a) + 1): round(b, 2) for a, b in tt.items()}}, rng=rng)
                return
        w = L.argmax_margin(tp, 1.0, 1.0)
        if w is not None:
            mk.add(clip, "most", "choice", rng.choice(Q.MOST_TURNS), opts, order.index(w), "most_turns",
                   {**ex, "winner": order.index(w) + 1, "turns_by_speaker": {str(order.index(a) + 1): b for a, b in tp.items()}}, rng=rng)


def multi_items(mk, clip, rng):
    k = clip["k"]
    ex = {"n_speakers": k}
    views = [rng.choice(["choice", "choice", "noul", "score"])]
    if rng.random() < 0.4:
        views.append("noul" if views[0] != "noul" else "choice")
    count_items(mk, clip, "speakers", k, rng, 1, 9, Q.SPK_COUNT, Q.COUNT_FMT_SPK, Q.SPK_NOUL, Q.SPK_BINS, Q.SPK_SCORE, views, ex)
    if k <= 2 and rng.random() < 0.5 and len(clip["_n"]) < 4:
        mk.noul(clip, "speakers", rng.choice(Q.SPK_SINGLE), k == 1, "single", rng, ex)
    if clip.get("turns_ok") and rng.random() < 0.5 and len(clip["_n"]) < 4:
        n = L.count_turns(clip["runs"])
        count_items(mk, clip, "turns", n, rng, 1, 14, Q.TURN_COUNT, Q.COUNT_FMT_TURN, Q.TURN_NOUL, Q.TURN_BINS, Q.TURN_SCORE,
                    [rng.choice(["choice", "noul"])], {"n_turns": n, "n_speakers": k})


def clip_label(noun, i, fmt):
    return fmt.format(noun=noun, Noun=noun.capitalize(), n=i, o=L.ORDINALS[i])


def intro(rng, clip):
    return rng.choice(Q.SEP_INTRO[clip["sep"]]).format(noun=clip["noun"])


def spkid_items(mk, clip, rng):
    K, N, M, noun = clip["K"], clip["N"], clip["M"], clip["noun"]
    ex = {"K": K, "N": N, "M": M, "sep": clip["sep"], "clip_speakers": clip["clip_speakers"]}
    fmt = rng.choice(Q.CLIP_OPT)
    fill = dict(noun=noun, Noun=noun.capitalize())
    u = rng.random()
    if M is None or u < 0.55:
        cands = [c for c in range(1, K + 1) if c != N]
        opts = [clip_label(noun, c, fmt) for c in cands]
        with_none = M is None or rng.random() < 0.25
        if with_none:
            opts.append(rng.choice(Q.SPKID_NONE).format(**fill))
        gold = len(opts) - 1 if M is None else cands.index(M)
        q = intro(rng, clip) + rng.choice(Q.SPKID_MATCH).format(n=N, **fill)
        mk.add(clip, "spkid", "choice", q, opts, gold, "match", ex, rng=rng)
    elif u < 0.75:
        pairs = [(a, b) for a in range(1, K + 1) for b in range(a + 1, K + 1)]
        gp = tuple(sorted((N, M)))
        others = [p for p in pairs if p != gp]
        rng.shuffle(others)
        sel = [gp] + others[: min(len(others), rng.randint(3, 7))]
        sel.sort()
        opts = [f"{noun}s {a} and {b}" if fmt.startswith("{noun}") or fmt.startswith("{Noun}") else L.pair_label(a, b) for a, b in sel]
        q = intro(rng, clip) + rng.choice(Q.SPKID_PAIR).format(**fill)
        mk.add(clip, "spkid", "choice", q, opts, sel.index(gp), "pair", ex, rng=rng)
    else:
        vals = L.count_window(clip["n_distinct"], 1, 8, rng, width=rng.randint(3, 5))
        q = intro(rng, clip) + rng.choice(Q.SPKID_COUNT).format(**fill)
        mk.add(clip, "spkid", "choice", q, num_opts(vals, Q.COUNT_FMT_SPK, rng), vals.index(clip["n_distinct"]), "count", ex, rng=rng)
    if rng.random() < 0.45:
        want = mk.want_yes("spkid", clip["set"], rng)
        cs = clip["clip_speakers"]
        pairs = [(a, b) for a in range(1, K + 1) for b in range(1, K + 1) if a != b and (cs[a - 1] == cs[b - 1]) == want]
        if pairs:
            a, b = rng.choice(pairs)
            q = intro(rng, clip) + rng.choice(Q.SPKID_NOUL).format(a=a, b=b, **fill)
            mk.noul(clip, "spkid", q, want, "pair_noul", rng, {**ex, "a": a, "b": b})


def pair_items(mk, clip, rng):
    noun = clip["noun"]
    ex = {"same": clip["same"], "sep": clip["sep"]}
    if rng.random() < 0.75:
        mk.noul(clip, "same", intro(rng, clip) + rng.choice(Q.SAME_NOUL).format(noun=noun), clip["same"], "noul", rng, ex)
    else:
        o = rng.choice(Q.SAME_OPTS)
        mk.add(clip, "same", "choice", intro(rng, clip) + rng.choice(Q.SAME_CHOICE).format(noun=noun), list(o),
               0 if clip["same"] else 1, "choice", ex, rng=rng)


# One render shard

def plan_for(scale: float) -> dict:
    """Clips per shard by type (x scale)."""
    base = {"dialog_libri": 100, "dialog_hvb": 34, "multi_libri": 92, "multi_hvb": 10, "multi_ami_nat": 30,
            "multi_ami_exc": 8, "spkid_libri": 70, "spkid_hvb": 16, "spkid_ami": 20, "pair_libri": 66, "pair_hvb": 16,
            "pair_ami": 22}
    return {k: max(1, int(round(v * scale))) for k, v in base.items()}


def render_shard(shard: int, scale: float, ann_path: str, train_tags: list[str], cal_tags: list[str]) -> dict:
    import time
    t_start = time.time()
    out_dir = SHARDS
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = f"{BUILDER}_b{shard:02d}"
    if (out_dir / f"items_{shard:02d}.jsonl").exists() and (out_dir / f"{bank}.npy").exists():
        return {"shard": shard, "skipped": True}
    rng = random.Random(f"{C.SEED}:{BUILDER}:{shard}")
    adi = adi_ids()
    pools = {"train": LibriPool(train_tags), "CAL": LibriPool(cal_tags)}
    hvb = HVB(adi["hvb_calls"])
    ami = AMI()
    ann = {}
    if Path(ann_path).exists():
        annz = np.load(ann_path)
        for k in annz.files:
            v, noun, num = k.split("|")
            ann.setdefault(v, {})[f"{noun}|{num}"] = annz[k].astype(np.float32)
    exc = {}

    def hvb_idx(split):
        if ("hvb", split) not in exc:
            exc[("hvb", split)] = hvb.excerpt_index(split, random.Random(f"{C.SEED}:hvbx:{shard}:{split}"), 220 if split == "train" else 60)
        return exc[("hvb", split)]

    def ami_idx(split):
        if ("ami", split) not in exc:
            exc[("ami", split)] = ami.excerpt_index(split, 60, random.Random(f"{C.SEED}:amix:{shard}:{split}"))
        return exc[("ami", split)]

    def hvb_load(e):
        sid, role, a, b = e
        x = hvb.chan(sid, role)[int(a * SR): int(b * SR)].copy()
        f = int(0.01 * SR)
        x[:f] *= np.linspace(0, 1, f)
        x[-f:] *= np.linspace(1, 0, f)
        return x

    def ami_load(e):
        m, a, b = e
        x = ami.read(m, a, b)
        f = int(0.01 * SR)
        x[:f] *= np.linspace(0, 1, f)
        x[-f:] *= np.linspace(1, 0, f)
        return x

    def same_meeting(s, s0, idx):
        m0 = {e[0] for e in idx[s0]}
        return any(e[0] in m0 for e in idx[s])

    t_setup = round(time.time() - t_start, 1)
    plan = plan_for(scale)
    mk = ItemMaker(shard)
    tsec = Counter()
    clips_audio, stats, drops = {}, Counter(), Counter()
    ci = 0
    for ctype, n_clips in plan.items():
        for _ in range(n_clips):
            t_clip = time.time()
            split = "CAL" if rng.random() < 0.035 else "train"
            if split == "CAL" and ctype in ("spkid_hvb", "pair_hvb"):
                split = "train"  # HVB CAL speakers are reserved for CAL dialogs only (decided before the announcer voice)
            crng = random.Random(f"{C.SEED}:{BUILDER}:{shard}:{ci}")
            ci += 1
            clip = None
            try:
                noun = crng.choice(ANN_NOUNS)
                av = crng.choice(ANN_VOICES_CAL if split == "CAL" else ANN_VOICES_TRAIN)
                sep = crng.choice(["beep", "spoken", "pause"] if av in ann else ["beep", "pause"])
                if ctype == "dialog_libri":
                    clip = dialog_libri(crng, pools[split])
                elif ctype == "dialog_hvb":
                    clip = dialog_hvb(crng, hvb, split)
                elif ctype == "multi_libri":
                    clip = multi_libri(crng, pools[split], crng.randint(1, 8))
                elif ctype == "multi_hvb":
                    clip = multi_excerpts(crng, hvb_idx(split), crng.randint(1, 4), hvb_load, "hvb")
                elif ctype == "multi_ami_exc":
                    clip = multi_excerpts(crng, ami_idx(split), crng.randint(1, 4), ami_load, "ami")
                elif ctype == "multi_ami_nat":
                    clip = ami_natural(crng, ami, split)
                elif ctype.startswith("spkid") or ctype.startswith("pair"):
                    corpus = ctype.split("_")[1]
                    if corpus == "libri":
                        getter = libri_getter(pools[split])
                        cname = "libritts_r"
                    elif corpus == "hvb":
                        if split == "CAL":
                            split = "train"  # HVB CAL speakers are reserved for CAL dialogs only
                        getter = excerpt_getter(hvb_idx(split), hvb_load, "hvb")
                        cname = "hvb"
                    else:
                        getter = excerpt_getter(ami_idx(split), ami_load, "ami", AMI.gender, same_meeting)
                        cname = "ami"
                    if ctype.startswith("spkid"):
                        K = crng.randint(3, 6)
                        clip = spkid_clip(crng, getter, K, crng.random() < 0.1, sep, noun, ann.get(av))
                    else:
                        clip = pair_clip(crng, getter, crng.random() < 0.5, sep if sep != "spoken" or crng.random() < 0.5 else "pause",
                                         noun, ann.get(av))
                    if clip is not None:
                        clip["corpus"] = cname
                        clip["announcer"] = av if clip["sep"] == "spoken" else None
            except Exception as e:  # noqa: BLE001
                drops[f"error:{ctype}:{type(e).__name__}"] += 1
                clip = None
            if clip is None:
                drops[f"none:{ctype}"] += 1
                continue
            clip["set"] = split
            clip["_n"] = []
            clip["bank"] = bank
            clip["key"] = f"{ctype}_{shard:02d}_{ci:05d}"
            # render, then re-check the labels with the per-track VAD
            if clip["kind"] == "nat":
                mix = clip.pop("audio")
                tracks = None
            else:
                mix, tracks = render(clip["pieces"], clip["T"])
            if clip["kind"] in ("dialog", "multi") and tracks is not None:
                vr = vad_runs({k: v for k, v in tracks.items() if k != "sep"})
                if clip["kind"] == "dialog" and L.count_turns(vr) != L.count_turns(clip["runs"]):
                    drops[f"vad_turns:{ctype}"] += 1
                    continue
                spk_vad = {k for a, b, k in vr if b - a > 0}
                if len(spk_vad) != len(set(r[2] for r in clip["runs"])):
                    drops[f"vad_spk:{ctype}"] += 1
                    continue
                if clip["kind"] == "dialog":
                    tt_v, tt_p = L.talk_time(vr), L.talk_time(clip["runs"])
                    clip["vad_talk"] = tt_v
                    clip["plan_talk"] = tt_p
                if clip["kind"] == "multi":
                    clip["turns_ok"] = (L.count_turns(vr) == L.count_turns(clip["runs"])) and not L.contained(clip["runs"], 0.6)
            phone_p = 0.0 if clip["corpus"] == "hvb" else 0.2
            y, facts = finish(mix, crng, phone_p, noise_p=0.0 if clip["kind"] == "nat" else 0.3)
            clip["facts"] = {**facts, "dur_s": round(len(y) / SR, 2)}
            if clip.get("announcer"):
                clip["facts"]["announcer_voice"] = clip["announcer"]
            if not (2.0 <= len(y) / SR <= 60.0):
                drops[f"len:{ctype}"] += 1
                continue
            n0 = len(mk.items)
            if clip["kind"] == "dialog":
                dialog_items(mk, clip, crng)
            elif clip["kind"] in ("multi", "nat"):
                multi_items(mk, clip, crng)
            elif clip["kind"] == "spkid":
                spkid_items(mk, clip, crng)
            else:
                pair_items(mk, clip, crng)
            if len(mk.items) > n0:
                clips_audio[clip["key"]] = y
                stats[f"clips:{ctype}:{split}"] += 1
            tsec[ctype] += time.time() - t_clip
    C.write_audio_bank(out_dir, bank, clips_audio)
    C.write_jsonl(out_dir / f"items_{shard:02d}.jsonl", mk.items)
    hours = sum(len(v) for v in clips_audio.values()) / SR / 3600
    return {"shard": shard, "items": len(mk.items), "clips": len(clips_audio), "hours": round(hours, 3),
            "stats": dict(stats), "drops": dict(drops), "setup_s": t_setup, "sec_by_type": {k: round(v, 1) for k, v in tsec.items()},
            "elapsed_s": round(time.time() - t_start, 1)}


def ami_natural(rng, ami: AMI, split: str):
    """A natural AMI Mix-Headset window with a target speaker count (1-4) drawn uniformly."""
    want = rng.randint(1, 4)
    for _ in range(60):
        m = rng.choice(ami.meetings[split])
        w = ami.window(m, rng, rng.uniform(4.0, 20.0) if want > 1 else rng.uniform(3.0, 10.0))
        if w is None or len(w["spk"]) != want:
            continue
        x = ami.read(m, w["t0"], w["t1"])
        if len(x) < 2 * SR:
            continue
        clip = {"kind": "nat", "corpus": "ami", "audio": x, "T": len(x) / SR, "k": len(w["spk"]), "runs": w["runs"],
                "speakers": w["spk"], "source_ids": [f"ami:{m}"], "turns_ok": w["turns"] is not None,
                "label_rule": "AMI manual word alignments (speaker counted if >= 0.6 s of words; windows with a "
                              "barely-audible speaker skipped)", "window": [round(w["t0"], 2), round(w["t1"], 2)]}
        return clip
    return None
