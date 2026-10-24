"""sb_compare build steps, run inside Modal containers (see modal_build.py).

plan() -> render_shard() (LibriTTS-R clips) and render_pause() (Kokoro sentences) -> assemble() (banks, items, summary).
"""
from __future__ import annotations

import collections
import json
import math
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

import cmp_labels as L
import common as C

BUILDER = "sb_compare"
SR = 16_000
SBV_WORK = Path("/vol/stageb/sb_voice/work2")
SBV = Path("/vol/stageb/sb_voice")
FINAL_OUT = Path("/vol/stageb/sb_compare")
WORK_ROOT = Path("/vol/stageb_work/sb_compare")
ADI_ITEMS = Path("/vol/adi/items.jsonl")
EXP0_EVAL = Path("/vol/data/eval_items.jsonl")
MMSU_TRUTH = Path("/vol/bench_mmsu/truth.jsonl")  # question/option text only, for the exact-match check

KOKORO_REF = "hexgrad/Kokoro-82M@f3ff3571791e39611d31c381e3a41a3af07b4987 (Apache-2.0)"
EN_VOICES = ["af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river",
             "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck",
             "am_santa", "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]
CAL_VOICES = ["af_sky", "bm_george", "am_puck"]
TRAIN_VOICES = [v for v in EN_VOICES if v not in CAL_VOICES]

SOURCES = {
    "libritts_r": {"name": "LibriTTS-R dev-clean + train-clean-100 (clips + features from sb_voice stage 1, /vol/stageb/sb_voice/work2; "
                           "sb_voice CAL speakers excluded)",
                   "repo": "mythicinfinity/libritts_r", "rev": "0d1718db351207d5a493135a62d5ed8cc4733830", "licence": "CC BY 4.0"},
    "kokoro": {"name": "Kokoro-82M TTS (sentences: LibriTTS-R transcripts with punctuation removed + templates written for this builder)",
               "repo": "hexgrad/Kokoro-82M", "rev": "f3ff3571791e39611d31c381e3a41a3af07b4987", "licence": "Apache-2.0"},
    "generated": {"name": "numpy tones / silence / noise beds; Rubber Band (rubberband-cli) pitch-shift + time-stretch; ffmpeg codecs",
                  "rev": "build.py", "licence": "CC0 (generated); tools only"},
}

# Clips per family (over-generated because measurement drops some)
CLIPS = {"pitch": 1700, "speed": 1450, "volume": 1250, "range": 1500, "duration": 1150}
PAUSE_SENTS = 2300          # Kokoro sentences for the pause family (each -> 1 clip)
MEANING_BASES = 640         # Kokoro renderings of ambiguous sentences (each -> 2 clips, one per reading)
PHONE_P = 0.20
CAL_SPK_N = 10
N_RENDER = 80
N_PAUSE = 12
N_TTS = 2


def paths(smoke: bool):
    work = WORK_ROOT / ("smoke" if smoke else "full")
    out = (WORK_ROOT / "smoke_out") if smoke else FINAL_OUT
    return work, out


# Records

def load_records() -> list[dict]:
    recs = []
    for p in sorted(SBV_WORK.glob("libritts_*.jsonl")):
        bank = p.stem
        for line in p.read_text().splitlines():
            r = json.loads(line)
            r["bank"] = bank
            recs.append(r)
    return recs


def sbv_cal_speakers() -> set:
    f = SBV / "cal_speakers.json"
    return set(json.loads(f.read_text())) if f.exists() else set()


def fnum(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


# Plan

def _pick_segments(pool_by_spk, spk_list, rng, k, kind, dmin, dmax, need, max_total=19.0):
    """k records: same_rep (1 record repeated), same_spk (k records of one speaker), diff_spk (k speakers)."""
    for _ in range(50):
        if kind == "diff_spk":
            spks = rng.sample(spk_list, k)
            rs = [rng.choice(pool_by_spk[s]) for s in spks]
        else:
            s = rng.choice(spk_list)
            cands = pool_by_spk[s]
            if kind == "same_rep":
                rs = [rng.choice(cands)] * k
            else:
                if len(cands) < k:
                    continue
                rs = rng.sample(cands, k)
        if not all(dmin <= r["dur"] <= dmax and need(r) for r in rs):
            continue
        if sum(r["dur"] for r in rs) + 1.2 * k > max_total:
            continue
        return rs
    return None


def _levels(k: int, spacing: float, rng: random.Random, offset: float) -> tuple[list[int], list[float]]:
    ranks = list(range(k))
    rng.shuffle(ranks)
    lv = [(r - (k - 1) / 2) * spacing + offset + rng.uniform(-0.12, 0.12) * spacing for r in ranks]
    return ranks, lv


EXTRA_CLIPS = {"pitch": 330, "speed": 300, "volume": 280, "range": 200, "duration": 330}


def plan(smoke: bool, extra: bool = False) -> dict:
    """extra=True: a top-up plan (plan_extra.json) with new clip keys '<family>x_NNNNN', same CAL speakers, no TTS."""
    work, _ = paths(smoke)
    work.mkdir(parents=True, exist_ok=True)
    rng = random.Random(f"{C.SEED}:{BUILDER}:plan")
    recs = load_records()
    sbv_cal = sbv_cal_speakers()
    recs = [r for r in recs if r["speaker"] not in sbv_cal and r.get("feat")]
    by_spk = collections.defaultdict(list)
    for r in recs:
        by_spk[r["speaker"]].append(r)
    spks = sorted(s for s in by_spk if len(by_spk[s]) >= 8)
    # CAL speakers: gender-balanced, seeded
    by_g = collections.defaultdict(list)
    for s in spks:
        by_g[by_spk[s][0].get("gender")].append(s)
    cal = set()
    for g in sorted(by_g, key=str):
        cal.update(rng.sample(sorted(by_g[g]), min(CAL_SPK_N // 2, len(by_g[g]) // 6)))
    split_spk = {"train": [s for s in spks if s not in cal], "CAL": sorted(cal)}
    scale = 0.03 if smoke else 1.0
    sizes, tag = CLIPS, ""
    if extra:
        rng = random.Random(f"{C.SEED}:{BUILDER}:plan_extra")
        sizes, tag = EXTRA_CLIPS, "x"

    def need_for(fam):
        if fam in ("pitch",):
            return lambda r: fnum(r["feat"].get("f0_med")) and (r["feat"].get("n_voiced") or 0) >= 40
        if fam == "range":
            return lambda r: fnum(r["feat"].get("f0_spread_st")) and (r["feat"].get("n_voiced") or 0) >= 60 and r["feat"]["f0_spread_st"] > 0.4
        if fam == "speed":
            return lambda r: fnum(r["feat"].get("rate")) and (r["feat"].get("syll") or 0) >= 6
        return lambda r: fnum(r["feat"].get("active_db"))

    clips = []
    for fam in ("pitch", "speed", "volume", "range"):
        n_total = max(8, int(sizes[fam] * scale))
        for i in range(n_total):
            split = "CAL" if L.stable_frac(f"{fam}{tag}:{i}", "cal") < 0.035 else "train"
            pool = split_spk[split]
            k = rng.choices([2, 3, 4], weights=[0.28, 0.52, 0.20])[0]
            if fam == "range":
                kind = rng.choices(["same_spk", "diff_spk"], weights=[0.5, 0.5])[0]
                dmin, dmax = 2.0, 6.0
            else:
                kind = rng.choices(["same_rep", "same_spk", "diff_spk"], weights=[0.45, 0.30, 0.25])[0]
                dmin, dmax = (1.6, 5.0) if k <= 3 else (1.6, 4.0)
            same_pair = k == 2 and rng.random() < 0.34
            natural_pitch = fam == "pitch" and kind == "diff_spk" and not same_pair and rng.random() < 0.5
            rs = None
            for _try in range(40 if (fam == "range" or natural_pitch) else 1):
                rs = _pick_segments(by_spk, pool, rng, k, kind, dmin, dmax, need_for(fam))
                if rs is None or not (fam == "range" or natural_pitch):
                    break
                # natural-only clips: preselect with the stored features (1.2x margin); labels come from re-measurement
                key_ = "f0_spread_st" if fam == "range" else "f0_med"
                u = [L.to_unit(fam, r["feat"][key_]) for r in rs]
                if k == 2 and fam == "range":
                    d = u[1] - u[0]
                    ok = abs(d) <= 0.5 * L.SAME["range"] if same_pair else abs(d) >= 1.2 * L.SEP["range"]
                else:
                    srt = sorted(u)
                    ok = all(b - a >= 1.2 * L.SEP[fam] for a, b in zip(srt, srt[1:]))
                if ok:
                    break
                rs = None
            if rs is None:
                continue
            ranks_plan, amounts = None, [0.0] * k
            if fam == "pitch":
                nat = [12 * math.log2(r["feat"]["f0_med"]) for r in rs]
                sp = rng.uniform(2.3, 3.6)
                if natural_pitch:
                    amounts = [0.0] * k  # natural voices only
                else:
                    ranks_plan, lv = _levels(k, 0.0 if same_pair else sp, rng, rng.uniform(-1.0, 1.0))
                    m = sum(nat) / k
                    amounts = [max(-6.0, min(6.0, lv[j] - (nat[j] - m))) for j in range(k)]
                    if same_pair:
                        amounts = [a + rng.uniform(-0.15, 0.15) for a in amounts]
            elif fam == "speed":
                nat = [math.log(r["feat"]["rate"]) for r in rs]
                sp = rng.uniform(0.27, 0.40)
                ranks_plan, lv = _levels(k, 0.0 if same_pair else sp, rng, rng.uniform(-0.08, 0.08))
                m = sum(nat) / k
                amounts = [max(math.log(1 / 1.7), min(math.log(1.7), lv[j] - (nat[j] - m))) for j in range(k)]
                if same_pair:
                    amounts = [a + rng.uniform(-0.01, 0.01) for a in amounts]
            elif fam == "volume":
                nat = [r["feat"]["active_db"] for r in rs]
                sp = rng.uniform(5.0, 8.0)
                ranks_plan, lv = _levels(k, 0.0 if same_pair else sp, rng, rng.uniform(-2.0, 1.0))
                m = sum(nat) / k
                amounts = [max(-15.0, min(12.0, lv[j] - (nat[j] - m))) for j in range(k)]
                if same_pair:
                    amounts = [a + rng.uniform(-0.3, 0.3) for a in amounts]
            # range: natural only (amounts 0); ranks come from measurement
            sep = rng.choices(["silence", "tone", "long"], weights=[0.62, 0.28, 0.10])[0]
            clips.append({"key": f"{fam}{tag}_{i:05d}", "family": fam, "split": split, "kind": kind, "k": k,
                          "segs": [{"bank": r["bank"], "key": r["key"], "speaker": r["speaker"], "source_id": r["source_id"],
                                    "syll": r["feat"].get("syll"), "text": r["text"]} for r in rs],
                          "amounts": [round(a, 4) for a in amounts], "ranks_plan": ranks_plan, "same_pair": same_pair,
                          "sep": sep, "phone": rng.random() < PHONE_P, "seed": rng.randrange(1 << 30)})
    # duration family
    n_total = max(8, int(sizes["duration"] * scale))
    for i in range(n_total):
        split = "CAL" if L.stable_frac(f"duration{tag}:{i}", "cal") < 0.035 else "train"
        pool = split_spk[split]
        n_utt = rng.choices([1, 2, 3, 4], weights=[0.15, 0.35, 0.30, 0.20])[0]
        same = rng.random() < 0.5
        rs = []
        s0 = rng.choice(pool)
        for _ in range(60):
            if len(rs) == n_utt:
                break
            s = s0 if same else rng.choice(pool)
            r = rng.choice(by_spk[s])
            if 0.8 <= r["dur"] <= 9.0 and r not in rs:
                rs.append(r)
        if len(rs) < n_utt:
            continue
        gaps = [rng.choice([rng.uniform(0.35, 0.65), rng.uniform(0.95, 1.3), rng.uniform(1.75, 2.75), rng.uniform(3.3, 5.0)])
                for _ in range(n_utt - 1)]
        lead, trail = rng.uniform(0.1, 1.5), rng.uniform(0.1, 1.5)
        if sum(r["dur"] for r in rs) + sum(gaps) + lead + trail > 30.0:
            continue
        clips.append({"key": f"duration{tag}_{i:05d}", "family": "duration", "split": split, "kind": "same_spk" if same else "diff_spk",
                      "k": n_utt, "segs": [{"bank": r["bank"], "key": r["key"], "speaker": r["speaker"], "source_id": r["source_id"],
                                            "syll": r["feat"].get("syll"), "text": r["text"]} for r in rs],
                      "gaps": [round(g, 3) for g in gaps], "lead": round(lead, 3), "trail": round(trail, 3),
                      "gains": [round(rng.uniform(-4, 3), 2) for _ in rs], "noise": rng.random() < 0.25,
                      "phone": rng.random() < PHONE_P, "seed": rng.randrange(1 << 30)})

    # TTS jobs: pause sentences from LibriTTS-R transcripts (text only), ambiguous-meaning templates
    texts = sorted({" ".join(L.clean_words(r["text"])) for r in recs})
    texts = [t for t in texts if 7 <= len(t.split()) <= 15 and all(len(w) <= 12 for w in t.split()) and "'" not in t]
    rng.shuffle(texts)
    tts = []
    n_ps = max(16, int(PAUSE_SENTS * scale))
    for i, t in enumerate(texts[:n_ps]):
        split = "CAL" if L.stable_frac(f"pause:{i}", "cal") < 0.035 else "train"
        voice = rng.choice(CAL_VOICES if split == "CAL" else TRAIN_VOICES)
        mode = rng.choices(["none", "word", "count", "dur", "where", "longer"], weights=[0.24, 0.24, 0.16, 0.14, 0.12, 0.10])[0]
        tts.append({"key": f"pause_{i:05d}", "family": "pause", "split": split, "voice": voice, "speed": round(rng.uniform(0.9, 1.1), 3),
                    "words": t.split(), "mode": mode, "phone": rng.random() < PHONE_P, "seed": rng.randrange(1 << 30)})
    n_mb = max(8, int(MEANING_BASES * scale))
    for i in range(n_mb):
        split = "CAL" if L.stable_frac(f"meaning:{i}", "cal") < 0.035 else "train"
        t = L.PAUSE_MEANING[i % len(L.PAUSE_MEANING)]
        filled = L.fill_meaning_template(t, rng)
        voice = rng.choice(CAL_VOICES if split == "CAL" else TRAIN_VOICES)
        tts.append({"key": f"meaning_{i:05d}", "family": "pause_meaning", "split": split, "voice": voice,
                    "speed": round(rng.uniform(0.9, 1.05), 3), "words": filled["words"], "template": i % len(L.PAUSE_MEANING),
                    "readings": filled["readings"], "distractors": filled["distractors"], "phone": rng.random() < PHONE_P,
                    "seed": rng.randrange(1 << 30)})
    if extra:
        tts = []
    p = {"clips": clips, "tts": tts, "cal_speakers": sorted(cal), "sbv_cal_excluded": len(sbv_cal)}
    (work / ("plan_extra.json" if extra else "plan.json")).write_text(json.dumps(p))
    return {"clips": len(clips), "by_family": dict(collections.Counter(c["family"] for c in clips)),
            "tts": len(tts), "cal_speakers": len(cal), "speakers": len(spks), "records": len(recs)}


# Audio helpers
_BANKS: dict = {}


def load_clip(bank: str, key: str) -> np.ndarray:
    if bank not in _BANKS:
        _BANKS[bank] = C.AudioBank(SBV_WORK, bank)
    return _BANKS[bank].get(key)


def rubberband(x: np.ndarray, kind: str, amount: float) -> np.ndarray:
    """kind 'tempo_log': tempo x exp(amount) (pitch unchanged); 'shift_st': pitch shift with formant preservation."""
    import tempfile

    import soundfile as sf

    with tempfile.TemporaryDirectory() as d:
        sf.write(f"{d}/in.wav", np.asarray(x, dtype=np.float32), SR, subtype="FLOAT")
        extra = ["-T", f"{math.exp(amount):.6f}"] if kind == "tempo_log" else ["-p", f"{amount:.5f}", "-F"]
        for engine in (["--fine"], []):
            r = subprocess.run(["rubberband", "-q", *engine, *extra, f"{d}/in.wav", f"{d}/out.wav"], capture_output=True, timeout=40)
            if r.returncode == 0:
                break
        else:
            raise RuntimeError(f"rubberband failed: {r.stderr[-300:]!r}")
        y, sr = sf.read(f"{d}/out.wav", dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    return C.resample(y, sr, SR) if sr != SR else y.astype(np.float32)


def fade(x: np.ndarray, n: int = 160) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64).copy()
    f = min(n, len(y) // 4)
    if f > 0:
        r = np.linspace(0, 1, f)
        y[:f] *= r
        y[-f:] *= r[::-1]
    return y


def tone() -> np.ndarray:
    n = int(0.15 * SR)
    t = np.arange(n) / SR
    y = 0.1 * np.sin(2 * np.pi * 1000.0 * t)
    f = int(0.01 * SR)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, f))
    y[:f] *= ramp
    y[-f:] *= ramp[::-1]
    return y


CODECS = {"mulaw": (["-c:a", "pcm_mulaw"], "wav"), "alaw": (["-c:a", "pcm_alaw"], "wav"), "gsm": (["-c:a", "libgsm"], "gsm"),
          "opus": (["-c:a", "libopus", "-application", "voip"], "ogg")}


def ffmpeg_codec(x: np.ndarray, codec: str, bitrate_k: int | None = None) -> np.ndarray:
    """Copied from data/sb_phone/fx.py: encode with a telephone codec at 8 kHz and decode back to 16 kHz."""
    enc, fmt = CODECS[codec]
    args = list(enc) + (["-b:a", f"{bitrate_k}k"] if bitrate_k else [])
    pcm = np.clip(np.asarray(x, np.float32), -1, 1).tobytes()
    p1 = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0", "-ar", "8000", "-ac", "1",
                         *args, "-f", fmt, "pipe:1"], input=pcm, capture_output=True, timeout=30)
    if p1.returncode != 0 or not p1.stdout:
        raise RuntimeError(f"ffmpeg encode {codec}: {p1.stderr.decode()[-300:]}")
    in_args = ["-f", "gsm", "-ar", "8000"] if fmt == "gsm" else []
    p2 = subprocess.run(["ffmpeg", "-v", "error", *in_args, "-i", "pipe:0", "-f", "f32le", "-ar", str(SR), "-ac", "1", "pipe:1"],
                        input=p1.stdout, capture_output=True, timeout=30)
    if p2.returncode != 0:
        raise RuntimeError(f"ffmpeg decode {codec}: {p2.stderr.decode()[-300:]}")
    y = np.frombuffer(p2.stdout, dtype=np.float32).copy()
    return y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


def bandlimit(x: np.ndarray, lo: float = 300.0, hi: float = 3400.0) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def phone_fx(x: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    """Whole-clip phone effect (relative levels between segments preserved): band 300-3400 Hz + one codec."""
    codec = rng.choice(["mulaw", "alaw", "gsm", "opus"])
    br = rng.choice([8, 10, 12]) if codec == "opus" else None
    peak = float(np.max(np.abs(x)) + 1e-9)
    y = bandlimit(x) * (0.7 / peak)
    y = ffmpeg_codec(y, codec, br)
    return y.astype(np.float32), {"codec": codec, "bitrate_k": br}


def finalize_level(x: np.ndarray, rng: random.Random) -> np.ndarray:
    peak = float(np.max(np.abs(x)) + 1e-9)
    return (x * (rng.uniform(0.35, 0.95) / peak)).astype(np.float32)


def pyin_stats(x: np.ndarray) -> dict:
    import librosa

    x = np.asarray(x, dtype=np.float32)
    f0, vflag, _ = librosa.pyin(x, fmin=65.0, fmax=500.0, sr=SR, frame_length=1024, hop_length=320)
    rms = librosa.feature.rms(y=x, frame_length=1024, hop_length=320, center=True)[0]
    n = min(len(rms), len(f0))
    db = 20.0 * np.log10(rms[:n] + 1e-10)
    voiced = np.asarray(vflag[:n], bool) & np.isfinite(f0[:n]) & (db > db.max() - 25.0)
    return L.f0_stats(f0[:n], voiced)


# Comparison clips: render + measure + items

def render_compare(c: dict) -> tuple[np.ndarray, dict]:
    rng = random.Random(c["seed"])
    fam = c["family"]
    parts = []
    for s, a in zip(c["segs"], c["amounts"]):
        x = load_clip(s["bank"], s["key"]).astype(np.float64)
        if fam == "pitch" and abs(a) > 1e-4:
            x = rubberband(x, "shift_st", a)
        elif fam == "speed" and abs(a) > 1e-4:
            x = rubberband(x, "tempo_log", a)
        elif fam == "volume":
            x = x * 10 ** (a / 20.0)
        parts.append(fade(np.asarray(x, dtype=np.float64)))
    gap = lambda: np.zeros(int(round(rng.uniform(0.55, 1.0) * SR)))  # noqa: E731
    tone_amp = float(np.median([np.max(np.abs(p)) for p in parts])) / 0.3  # constant tone level within a clip (no label cue)
    seq, bounds = [np.zeros(int(round(rng.uniform(0.15, 0.6) * SR)))], []
    pos = len(seq[0])
    for j, p in enumerate(parts):
        if j > 0:
            if c["sep"] == "tone":
                sep = np.concatenate([np.zeros(int(0.35 * SR)), tone() * tone_amp, np.zeros(int(0.35 * SR))])
            elif c["sep"] == "long":
                sep = np.zeros(int(round(rng.uniform(1.3, 2.0) * SR)))
            else:
                sep = gap()
            seq.append(sep)
            pos += len(sep)
        seq.append(p)
        bounds.append((pos, pos + len(p)))
        pos += len(p)
    seq.append(np.zeros(int(round(rng.uniform(0.15, 0.6) * SR))))
    y = np.concatenate(seq)
    y = y * (0.7 / (np.max(np.abs(y)) + 1e-9))
    fx = None
    if c["phone"]:
        y, fx = phone_fx(y.astype(np.float32), rng)
    y = finalize_level(np.asarray(y), rng)
    return y, {"bounds": bounds, "fx": fx}


def measure_compare(c: dict, y: np.ndarray, bounds) -> list:
    fam = c["family"]
    vals = []
    for (s0, s1), seg in zip(bounds, c["segs"]):
        x = y[s0:s1]
        if fam in ("pitch", "range"):
            st = pyin_stats(x)
            v = st["f0_med"] if fam == "pitch" else st["f0_spread_st"]
            if fam == "range" and (st["n_voiced"] < 40 or v is None or v <= 0.2):
                v = None
            vals.append(None if v is None else float(v))
        elif fam == "speed":
            sp = L.speech_span(x)
            vals.append(seg["syll"] / (sp[1] - sp[0]) if sp and seg.get("syll") and sp[1] - sp[0] > 0.5 else None)
        else:
            vals.append(L.active_db(x))
    return vals


def compare_items(c: dict, vals: list, rng: random.Random) -> list[dict]:
    """Question specs (not yet make_item) for a measured comparison clip."""
    fam, k = c["family"], c["k"]
    if any(v is None for v in vals):
        return []
    units = [L.to_unit(fam, v) for v in vals]
    nw = L.NUMW[k]
    intro = rng.choice(L.INTRO[c["kind"]]).format(n=nw, N=nw.capitalize()) if rng.random() < 0.7 else ""
    feat = rng.choice(L.FEAT_NOUN[fam])
    out = []

    def q(text):
        return (intro + " " + text).strip()

    if k == 2:
        cls = L.pair_class(units[0], units[1], fam)
        if cls is None:
            return []
        qs, optsets = L.PAIR_SCORE[fam]
        opts = rng.choice(optsets)
        out.append({"qtype": "score", "qkind": "pair_score", "question": q(rng.choice(qs)), "options": opts, "gold": cls,
                    "answer": opts[cls]})
        if cls == 1 or rng.random() < 0.3:
            yes = cls == 1
            out.append({"qtype": "noul", "qkind": "same_noul", "question": q(rng.choice(L.SAME_NOUL[fam])), "yes": yes})
        else:
            # direction question; pick wording so yes/no is balanced: ask "is second MORE than first" (yes iff cls==2)
            tmpl = rng.choice(L.PAIR_NOUL[fam])
            out.append({"qtype": "noul", "qkind": "pair_noul", "question": q(tmpl.format(i="first", j="second", I="1", J="2")),
                        "yes": cls == 2})
        return out
    ranks = L.rank_levels(units, fam)
    if ranks is None:
        return []
    if k == 3:
        words = rng.choice(L.LEVEL_WORDS[fam])
        style = rng.choices([0, 1, 2], weights=[0.6, 0.25, 0.15])[0]
        n_opts = rng.choice([4, 4, 6])
        perms, gi = L.pattern_options(ranks, n_opts, rng)
        opts = [L.pattern_string(list(p), words, style) for p in perms]
        out.append({"qtype": "choice", "qkind": "pattern", "question": q(rng.choice(L.PATTERN_Q).format(feat=feat, n=nw)),
                    "options": opts, "gold": gi, "answer": "-".join(str(r) for r in ranks)})
    if k >= 3 and (k == 4 or rng.random() < 0.5):
        qt, end = rng.choice(L.EXTREME_Q[fam])
        ol = rng.choice(L.EXTREME_OPTS)[:k]
        g = ranks.index(k - 1 if end == "max" else 0)
        out.append({"qtype": "choice", "qkind": f"extreme_{end}", "question": q(qt), "options": ol, "gold": g, "answer": str(g)})
    n_noul = 2 if k == 4 else (1 if len(out) < 2 or rng.random() < 0.5 else 0)
    pairs = [(i, j) for i in range(k) for j in range(k) if i < j]
    rng.shuffle(pairs)
    for i, j in pairs[:n_noul]:
        if rng.random() < 0.5:
            i, j = j, i  # ask about j vs i in either order
        tmpl = rng.choice(L.PAIR_NOUL[fam])
        text = tmpl.format(i=L.ORD[i], j=L.ORD[j], I=str(i + 1), J=str(j + 1))
        out.append({"qtype": "noul", "qkind": "pair_noul", "question": q(text), "yes": ranks[j] > ranks[i]})
    return out[:3]


# Duration clips

def render_duration(c: dict) -> tuple[np.ndarray, dict]:
    rng = random.Random(c["seed"])
    parts = [fade(load_clip(s["bank"], s["key"]).astype(np.float64) * 10 ** (g / 20.0)) for s, g in zip(c["segs"], c["gains"])]
    seq = [np.zeros(int(round(c["lead"] * SR)))]
    pos = len(seq[0])
    bounds = []
    for j, p in enumerate(parts):
        if j > 0:
            g = np.zeros(int(round(c["gaps"][j - 1] * SR)))
            seq.append(g)
            pos += len(g)
        seq.append(p)
        bounds.append((pos, pos + len(p)))
        pos += len(p)
    seq.append(np.zeros(int(round(c["trail"] * SR))))
    y = np.concatenate(seq)
    y = y * (0.7 / (np.max(np.abs(y)) + 1e-9))
    if c["noise"]:
        act = L.active_db(y)
        if act is not None:
            nz = np.random.default_rng(c["seed"]).standard_normal(len(y))
            nz = nz / np.sqrt(np.mean(nz ** 2)) * 10 ** ((act - rng.uniform(42, 50)) / 20.0)
            y = y + nz
    fx = None
    if c["phone"]:
        y, fx = phone_fx(y.astype(np.float32), rng)
    y = finalize_level(np.asarray(y), rng)
    return y, {"bounds": bounds, "fx": fx}


def measure_duration(y: np.ndarray, bounds) -> dict | None:
    ref = float(L.frame_db(y).max())
    spans = []
    n = len(y)
    for j, (s0, s1) in enumerate(bounds):
        lo = max(0, s0 - int(0.25 * SR)) if j == 0 else (bounds[j - 1][1] + s0) // 2
        hi = min(n, s1 + int(0.25 * SR)) if j == len(bounds) - 1 else (s1 + bounds[j + 1][0]) // 2
        sp = L.speech_span(y[lo:hi], ref_db=ref)
        if sp is None:
            return None
        spans.append((lo / SR + sp[0], lo / SR + sp[1]))
    total = sum(e - s for s, e in spans)
    gaps = [spans[j + 1][0] - spans[j][1] for j in range(len(spans) - 1)]
    # independent check: whole-clip silences >= 0.3 s must include every planned gap
    sil = L.silences(y, 0.3, ref_db=ref)
    return {"spans": [(round(s, 3), round(e, 3)) for s, e in spans], "total": round(total, 3),
            "gaps": [round(g, 3) for g in gaps], "n_sil_ge_0.3": len(sil)}


def duration_items(c: dict, m: dict, rng: random.Random) -> list[dict]:
    out = []
    total = m["total"]
    if total < 0.8:
        return []
    res = L.numeric_options(total, rng)
    if res is not None:
        nums, g = res
        style = rng.randrange(3)
        opts = [L.fmt_seconds(v, style) for v in nums]
        out.append({"qtype": "choice", "qkind": "total_numeric", "question": rng.choice(L.DUR_TOTAL_Q), "options": opts, "gold": g,
                    "answer": f"pos{g}"})
    if rng.random() < 0.5:
        cuts, labels = rng.choice(L.DUR_BINNINGS)
        b = L.bin_index(total, cuts, L.DUR_MARGIN)
        if b is not None:
            out.append({"qtype": "score", "qkind": "total_bins", "question": rng.choice(L.DUR_BIN_Q), "options": labels, "gold": b,
                        "answer": f"{len(labels)}:{b}"})
    else:
        want_yes = rng.random() < 0.5
        for _ in range(30):
            N = rng.choice([2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
            mg = max(0.8, 0.12 * N)
            if (want_yes and total >= N + mg) or (not want_yes and total <= N - mg):
                out.append({"qtype": "noul", "qkind": "total_noul", "question": rng.choice(L.DUR_NOUL_Q).format(N=N), "yes": want_yes})
                break
    if len(m["gaps"]) >= 1 and rng.random() < 0.6:
        cuts, labels = L.GAP_BINNING
        b = L.bin_index(m["gaps"][0], cuts, L.GAP_MARGIN)
        if b is not None:
            out.append({"qtype": "score", "qkind": "gap_bins", "question": rng.choice(L.GAP_Q), "options": labels, "gold": b,
                        "answer": str(b)})
    elif len(m["spans"]) >= 2:
        d0 = m["spans"][0][1] - m["spans"][0][0]
        d1 = m["spans"][1][1] - m["spans"][1][0]
        if max(d0, d1) / max(1e-3, min(d0, d1)) >= 1.35:
            opts = rng.choice([["the first", "the second"], ["the first sentence", "the second sentence"], ["first", "second"]])
            g = 0 if d0 > d1 else 1
            out.append({"qtype": "choice", "qkind": "longer_utt", "question": rng.choice(L.LONGER_Q), "options": opts, "gold": g,
                        "answer": str(g)})
    return out[:3]


# Pause clips (Kokoro + word timings)

def insert_pauses(x: np.ndarray, words: list, pauses: dict) -> tuple[np.ndarray, list]:
    """x: TTS audio (16 kHz), words: [(w, start_s, end_s)], pauses: {word index: seconds} (pause after that word).
    Cut in the middle of the natural gap between word i and i+1; return audio and new word end times."""
    cuts = []
    for i in sorted(pauses):
        a = words[i][2]
        b = words[i + 1][1]
        cuts.append((int(round(0.5 * (a + b) * SR)), int(round(pauses[i] * SR)), i))
    pieces, last = [], 0
    for cut, n, _ in cuts:
        seg = np.asarray(x[last:cut], dtype=np.float64).copy()
        f = min(80, len(seg) // 4)
        if f > 0:
            seg[-f:] *= np.linspace(1, 0, f)
        pieces.append(seg)
        pieces.append(np.zeros(n))
        last = cut
    tail = np.asarray(x[last:], dtype=np.float64).copy()
    f = min(80, len(tail) // 4)
    if f > 0 and cuts:
        tail[:f] *= np.linspace(0, 1, f)
    pieces.append(tail)
    y = np.concatenate(pieces)
    ends = []
    for i, (w, s, e) in enumerate(words):
        shift = sum(n for cut, n, j in cuts if j < i)
        ends.append(e + shift / SR)
    # inserted silence intervals in output time: {word index: (start_s, end_s)}
    ins, acc = {}, 0
    for cut, n, j in cuts:
        ins[j] = ((cut + acc) / SR, (cut + acc + n) / SR)
        acc += n
    return y, ends, ins


def pause_layout(t: dict, words: list, rng: random.Random) -> dict | None:
    """Planned pauses {word index: seconds} for a pause-family sentence."""
    n = len(words)
    inner = list(range(1, n - 2))  # never after the first word or before the last word
    if len(inner) < 3:
        return None
    mode = t["mode"]
    if mode == "none":
        return {}
    if mode in ("word", "where"):
        i = rng.choice(inner)
        if mode == "where":
            pos = (i + 1) / n
            if min(abs(pos - 1 / 3), abs(pos - 2 / 3)) < 0.06:
                return None
        return {i: rng.uniform(0.5, 1.3)}
    if mode == "dur":
        b = rng.randrange(4)
        return {rng.choice(inner): rng.uniform(*L.PAUSE_DUR_DRAW[b])}
    if mode in ("count", "longer"):
        k = 2 if mode == "longer" else rng.choice([0, 1, 2, 3])
        if k == 0:
            return {}
        for _ in range(40):
            idx = sorted(rng.sample(inner, k))
            if all(b - a >= 2 for a, b in zip(idx, idx[1:])):
                if mode == "longer":
                    short, long_ = rng.uniform(0.45, 0.65), rng.uniform(1.3, 2.0)
                    d = [short, long_] if rng.random() < 0.5 else [long_, short]
                    return dict(zip(idx, d))
                return {i: rng.uniform(0.5, 1.2) for i in idx}
        return None
    return None


def measure_pauses(y: np.ndarray, ins: dict) -> dict:
    """Measured silences (> NO_PAUSE_MAX_S, energy VAD on the final audio); each is attributed to the planned pause whose
    inserted interval it contains (+-50 ms), else after_word = None (an unplanned pause; the clip is dropped)."""
    sil = L.silences(y, L.NO_PAUSE_MAX_S)
    found = []
    for s, e in sil:
        w = None
        for j, (a, b) in ins.items():
            if s <= a + 0.05 and e >= b - 0.05:
                w = j
        found.append({"start": s, "end": e, "dur": round(e - s, 3), "after_word": w})
    return {"silences": found}


def pause_items(t: dict, plan_p: dict, meas: dict, words: list, rng: random.Random) -> list[dict]:
    """Validate measured pauses against the plan and build questions; every label comes from the measured silences."""
    found = meas["silences"]
    n = len(words)
    planned = sorted(plan_p)
    if len(found) != len(planned):
        return []
    for f, i in zip(found, planned):
        if f["after_word"] != i or f["dur"] < L.PAUSE_MIN_S - 0.05:
            return []
    wl = [w for w, _, _ in words]
    out = []
    mode = t["mode"]
    if mode in ("none", "word"):
        no_opt = rng.choice(L.NO_PAUSE_OPTS)
        if planned:
            i = planned[0]
            pool = [j for j in range(n) if wl[j] != wl[i] and j not in planned]
        else:
            i = None
            pool = list(range(n))
        nd = rng.choice([2, 3, 3, 4])
        if len({wl[j] for j in pool}) < nd:
            return []
        cand = []
        for j in rng.sample(pool, len(pool)):
            if wl[j] not in [wl[c] for c in cand] and (i is None or wl[j] != wl[i]):
                cand.append(j)
            if len(cand) == (nd if i is not None else nd + 1):
                break
        idxs = sorted(cand + ([i] if i is not None else []))
        opts = [wl[j] for j in idxs] + [no_opt]
        g = idxs.index(i) if i is not None else len(opts) - 1
        out.append({"qtype": "choice", "qkind": "pause_word", "question": rng.choice(L.PAUSE_WORD_Q), "options": opts, "gold": g,
                    "answer": "none" if i is None else "word"})
        qi = rng.randrange(len(L.PAUSE_YN_Q))
        has = bool(planned)
        out.append({"qtype": "noul", "qkind": "pause_yn", "question": L.PAUSE_YN_Q[qi],
                    "yes": (not has) if qi in L.PAUSE_YN_INVERTED else has})
    elif mode == "count":
        k = len(planned)
        opts = rng.choice([["none", "one", "two", "three"], ["0", "1", "2", "3"], ["no pauses", "one pause", "two pauses", "three pauses"]])
        out.append({"qtype": "score", "qkind": "pause_count", "question": rng.choice(L.PAUSE_COUNT_Q), "options": opts, "gold": k,
                    "answer": str(k)})
    elif mode == "dur":
        d = found[0]["dur"]
        cuts, labels = L.PAUSE_DUR_BINS
        b = L.bin_index(d, cuts, L.PAUSE_DUR_MARGIN)
        if b is None:
            return []
        out.append({"qtype": "score", "qkind": "pause_dur", "question": rng.choice(L.PAUSE_DUR_Q), "options": labels, "gold": b,
                    "answer": str(b)})
    elif mode == "where":
        th = L.third_of(planned[0], n)
        opts = rng.choice(L.WHERE_OPTS)
        out.append({"qtype": "score", "qkind": "pause_where", "question": rng.choice(L.PAUSE_WHERE_Q), "options": opts, "gold": th,
                    "answer": str(th)})
    elif mode == "longer":
        d0, d1 = found[0]["dur"], found[1]["dur"]
        if max(d0, d1) / min(d0, d1) < 1.8:
            return []
        opts = rng.choice([["the first pause", "the second pause"], ["the first one", "the second one"], ["the earlier pause", "the later pause"]])
        g = 0 if d0 > d1 else 1
        out.append({"qtype": "choice", "qkind": "pause_longer", "question": rng.choice(L.PAUSE_LONGER_Q), "options": opts, "gold": g,
                    "answer": str(g)})
    return out


def meaning_items(t: dict, reading: int, meas: dict, words: list, rng: random.Random) -> list[dict]:
    planned = sorted(t["readings"][reading][0])
    found = meas["silences"]
    if len(found) != len(planned) or any(f["after_word"] != i or f["dur"] < L.PAUSE_MIN_S - 0.05 for f, i in zip(found, planned)):
        return []
    opts = [t["readings"][0][1], t["readings"][1][1]] + list(t["distractors"])
    out = [{"qtype": "choice", "qkind": "meaning", "question": rng.choice(L.MEANING_Q), "options": opts, "gold": reading,
            "answer": f"t{t['template']}r{reading}"}]
    return out


# Item construction (shared)

def to_item(spec: dict, clip_key: str, bank: str, split: str, fam: str, meta: dict, n: int) -> dict:
    rng = random.Random(f"{C.SEED}:{clip_key}:{n}")
    if spec["qtype"] == "noul":
        opts, gold = ["yes", "no"], (0 if spec["yes"] else 1)
        q_rng = rng
    else:
        opts, gold = spec["options"], spec["gold"]
        q_rng = L.NoShuffle() if spec["qtype"] == "score" else rng
    target = [0.0] * len(opts)
    target[gold] = 1.0
    m = dict(meta)
    m.update({"qtype": spec["qtype"], "qkind": spec["qkind"], "answer": ("yes" if spec.get("yes") else "no") if spec["qtype"] == "noul" else spec["answer"]})
    return C.make_item(item_id=f"{BUILDER}_{clip_key}_{n}", set_name=split, source=f"{BUILDER}_{fam}", question=spec["question"],
                       options=list(opts), target=target, gold=gold, rng=q_rng, audio={"bank": bank, "key": clip_key},
                       channel="clean", state_text=None, meta=m)


# Shard workers

def _compare_job(c):
    try:
        if c["family"] == "duration":
            y, info = render_duration(c)
            m = measure_duration(y, info["bounds"])
            if m is None:
                return c["key"], None, None, "vad_fail"
            rng = random.Random(c["seed"] + 7)
            specs = duration_items(c, m, rng)
            facts = {"measured": m, "fx": info["fx"], "planned": {"gaps": c["gaps"], "lead": c["lead"], "trail": c["trail"]}}
            # planned-vs-measured check: every measured gap within 0.25 s of the planned gap (else labels are untrustworthy)
            if any(abs(a - b) > 0.25 for a, b in zip(m["gaps"], c["gaps"])):
                return c["key"], None, facts, "gap_mismatch"
        else:
            y, info = render_compare(c)
            vals = measure_compare(c, y, info["bounds"])
            rng = random.Random(c["seed"] + 7)
            specs = compare_items(c, vals, rng)
            facts = {"measured": vals, "bounds_s": [(round(a / SR, 3), round(b / SR, 3)) for a, b in info["bounds"]], "fx": info["fx"]}
            if c.get("ranks_plan") is not None and all(v is not None for v in vals):
                u = [L.to_unit(c["family"], v) for v in vals]
                r = L.rank_levels(u, c["family"])
                facts["plan_rank_match"] = (r == c["ranks_plan"]) if r is not None and not c["same_pair"] else None
                if c["same_pair"]:
                    facts["plan_same_match"] = L.pair_class(u[0], u[1], c["family"]) == 1
                elif c["k"] == 2:
                    facts["plan_pair_match"] = L.pair_class(u[0], u[1], c["family"]) == (2 if c["ranks_plan"][1] > c["ranks_plan"][0] else 0)
                if r is not None and c["k"] >= 3 and r != c["ranks_plan"]:
                    specs = []  # measured order disagrees with the planned manipulation -> drop (should be rare)
                    facts["dropped"] = "plan_rank_mismatch"
        if not specs:
            return c["key"], None, facts, "no_items"
        return c["key"], (y, specs), facts, None
    except Exception as e:  # noqa: BLE001
        return c["key"], None, None, repr(e)[:300]


def render_shard(shard: int, n_shards: int, n_proc: int, smoke: bool, extra: bool = False) -> dict:
    work, _ = paths(smoke)
    p = json.loads((work / ("plan_extra.json" if extra else "plan.json")).read_text())
    clips = [c for i, c in enumerate(p["clips"]) if i % n_shards == shard]
    out_dir = work / "shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    rtag = "x" if extra else "r"
    if (out_dir / f"{rtag}_{shard:02d}.items.jsonl").exists():
        return {"shard": shard, "skipped": True}
    t0 = time.time()
    audio, items, facts_all, errs = {}, [], {}, collections.Counter()
    bank = f"{BUILDER}_{rtag}{shard:02d}"
    import signal

    def _alarm(signum, frame):
        raise TimeoutError("clip timeout")

    signal.signal(signal.SIGALRM, _alarm)

    def _guarded(c):
        signal.alarm(90)  # watchdog: a single clip never blocks the shard
        try:
            return _compare_job(c)
        except TimeoutError as e:
            return c["key"], None, None, repr(e)[:100]
        finally:
            signal.alarm(0)

    for c in clips:
        key, res, facts, err = _guarded(c)
        if True:
            facts_all[key] = {"facts": facts, "err": err, "family": c["family"], "split": c["split"]}
            if err:
                errs[err[:40]] += 1
                continue
            y, specs = res
            audio[key] = y
            meta = {"family": c["family"], "kind": c["kind"], "n_segments": c["k"], "speakers": sorted({s["speaker"] for s in c["segs"]}),
                    "speaker": c["segs"][0]["speaker"], "source_ids": [s["source_id"] for s in c["segs"]],
                    "manipulation": ({"pitch": "pitch_shift_st", "speed": "tempo_log", "volume": "gain_db"}.get(c["family"]) if c["family"] != "duration" else "layout"),
                    "amounts": c.get("amounts"), "measured": facts.get("measured"), "fx": facts.get("fx"),
                    "licence": "LibriTTS-R CC BY 4.0 (mythicinfinity/libritts_r@0d1718db); manipulations generated (CC0)"}
            for n, s in enumerate(specs):
                items.append(to_item(s, key, bank, c["split"], c["family"], meta, n))
    C.write_audio_bank(out_dir, bank, audio)
    (out_dir / f"{rtag}_{shard:02d}.facts.json").write_text(json.dumps(facts_all, default=str))
    C.write_jsonl(out_dir / f"{rtag}_{shard:02d}.items.jsonl", items)
    return {"shard": shard, "clips": len(clips), "kept": len(audio), "items": len(items), "errs": dict(errs),
            "sec": round(time.time() - t0)}


def render_pause(shard: int, n_shards: int, smoke: bool) -> dict:
    work, _ = paths(smoke)
    p = json.loads((work / "plan.json").read_text())
    tts_dir = work / "tts"
    tts = {}
    for f in sorted(tts_dir.glob("tts_*.npz")):
        meta = json.loads(f.with_suffix(".json").read_text())
        with np.load(f) as z:
            for k in z.files:
                if int(k.split("_")[1]) % n_shards == shard:
                    tts[k] = (z[k], meta[k])
    out_dir = work / "shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [t for t in p["tts"] if int(t["key"].split("_")[1]) % n_shards == shard]
    bank = f"{BUILDER}_p{shard:02d}"
    audio, items, facts_all, errs = {}, [], {}, collections.Counter()
    t0 = time.time()
    for t in jobs:
        if t["key"] not in tts:
            errs["tts_missing"] += 1
            continue
        x, words = tts[t["key"]]
        words = [tuple(w) for w in words]
        if [w for w, _, _ in words] != t["words"]:
            errs["word_mismatch"] += 1
            continue
        rng = random.Random(t["seed"])
        # base check: the un-paused TTS sentence must have no internal silence > NO_PAUSE_MAX_S
        base_sil = L.silences(x, L.NO_PAUSE_MAX_S)
        if base_sil:
            errs["base_has_pause"] += 1
            continue
        variants = []
        if t["family"] == "pause":
            pl = pause_layout(t, words, rng)
            if pl is None:
                errs["layout"] += 1
                continue
            variants.append((t["key"], pl, None))
        else:
            for ri, (pidx, _) in enumerate(t["readings"]):
                variants.append((f"{t['key']}r{ri}", {i: rng.uniform(0.55, 1.0) for i in pidx}, ri))
        for vkey, pl, ri in variants:
            try:
                y, ends, ins = insert_pauses(x, words, pl)
                n_lead = int(rng.uniform(0.15, 0.5) * SR)
                y = np.concatenate([np.zeros(n_lead), y, np.zeros(int(rng.uniform(0.15, 0.5) * SR))])
                ins = {j: (a + n_lead / SR, b + n_lead / SR) for j, (a, b) in ins.items()}
                fx = None
                y = y * (0.7 / (np.max(np.abs(y)) + 1e-9))
                if t["phone"]:
                    y, fx = phone_fx(y.astype(np.float32), rng)
                y = finalize_level(np.asarray(y), rng)
                meas = measure_pauses(y, ins)
                if t["family"] == "pause":
                    specs = pause_items(t, pl, meas, words, rng)
                else:
                    specs = meaning_items(t, ri, meas, words, rng)
                facts_all[vkey] = {"planned": {str(k): round(v, 3) for k, v in pl.items()}, "measured": meas, "fx": fx,
                                   "family": t["family"], "split": t["split"], "kept": bool(specs)}
                if not specs:
                    errs["measure_mismatch"] += 1
                    continue
                audio[vkey] = y
                meta = {"family": t["family"], "voice": t["voice"], "speaker": f"kokoro:{t['voice']}", "tts_speed": t["speed"],
                        "text": " ".join(t["words"]), "planned_pauses": {str(k): round(v, 3) for k, v in pl.items()},
                        "measured_pauses": meas["silences"], "fx": fx, "mode": t.get("mode", "meaning"),
                        "licence": f"Kokoro-82M Apache-2.0 ({KOKORO_REF}); text: LibriTTS-R transcript (CC BY 4.0) or builder template"}
                for n, s in enumerate(specs):
                    items.append(to_item(s, vkey, bank, t["split"], t["family"], meta, n))
            except Exception as e:  # noqa: BLE001
                errs[repr(e)[:40]] += 1
    C.write_audio_bank(out_dir, bank, audio)
    (out_dir / f"p_{shard:02d}.facts.json").write_text(json.dumps(facts_all, default=str))
    C.write_jsonl(out_dir / f"p_{shard:02d}.items.jsonl", items)
    return {"shard": shard, "jobs": len(jobs), "kept": len(audio), "items": len(items), "errs": dict(errs), "sec": round(time.time() - t0)}


# Assembly, balancing, validation, summary

def balance(items: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
    """noul exactly 50/50 per (source, qkind, split); choice/score answers capped at 2x the rarest per (source, qkind, split)."""
    groups = collections.defaultdict(list)
    for it in items:
        groups[(it["source"], it["meta"]["qkind"], it["set"])].append(it)
    out, stats = [], {}
    for g, its in sorted(groups.items()):
        by = collections.defaultdict(list)
        for it in its:
            by[it["meta"]["answer"]].append(it)
        if its[0]["meta"]["qtype"] == "noul":
            m = min(len(by.get("yes", [])), len(by.get("no", [])))
            keep = []
            for a in ("yes", "no"):
                lst = by.get(a, [])
                rng.shuffle(lst)
                keep += lst[:m]
        else:
            m = min(len(v) for v in by.values())
            keep = []
            for a, lst in by.items():
                rng.shuffle(lst)
                keep += lst[: 2 * m]
        out += keep
        stats["|".join(g)] = {"in": len(its), "kept": len(keep)}
    return out, stats


def cap_per_clip(items: list[dict], cap: int = 4) -> list[dict]:
    seen = collections.Counter()
    out = []
    for it in items:
        k = it["audio"]["key"]
        if seen[k] < cap:
            out.append(it)
            seen[k] += 1
    return out


def norm_q(s: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", " ", s.lower().replace("’", "'")).strip()


def assemble(smoke: bool) -> dict:
    work, out = paths(smoke)
    t0 = time.time()
    shards = work / "shards"
    items = []
    for f in sorted(shards.glob("*.items.jsonl")):
        items += C.read_jsonl(f)
    facts = {}
    for f in sorted(shards.glob("*.facts.json")):
        facts.update(json.loads(f.read_text()))
    n_raw = len(items)
    # drop items on clips shorter than 2 s
    blen = {}
    for f in sorted(shards.glob(f"{BUILDER}_*.json")):
        for k, (st, ln) in json.loads(f.read_text()).items():
            blen[(f.stem, k)] = ln / SR
    n_before = len(items)
    items = [it for it in items if blen.get((it["audio"]["bank"], it["audio"]["key"]), 0.0) >= 2.0]
    n_short_dropped = n_before - len(items)
    rng = random.Random(f"{C.SEED}:{BUILDER}:balance")
    items.sort(key=lambda it: it["id"])
    items, bal_stats = balance(items, rng)
    items = cap_per_clip(sorted(items, key=lambda it: it["id"]))
    used = collections.defaultdict(set)
    for it in items:
        used[it["audio"]["bank"]].add(it["audio"]["key"])

    # validation
    errors = []
    bank_index = {}
    durs = {}
    for b in sorted(used):
        idx = json.loads((shards / f"{b}.json").read_text())
        bank_index[b] = idx
        for k in used[b]:
            if k not in idx:
                errors.append(f"missing audio {b}/{k}")
            else:
                durs[(b, k)] = idx[k][1] / SR
    keys_needed = {"id", "set", "source", "question", "options", "target", "gold", "audio", "channel", "state_text", "meta"}
    ids = set()
    for it in items:
        if set(it) != keys_needed:
            errors.append(f"keys {it['id']}")
        if it["id"] in ids:
            errors.append(f"dup id {it['id']}")
        ids.add(it["id"])
        if not 2 <= len(it["options"]) <= 20 or len(set(it["options"])) != len(it["options"]):
            errors.append(f"options {it['id']}")
        if abs(sum(it["target"]) - 1) > 1e-6 or int(np.argmax(it["target"])) != it["gold"]:
            errors.append(f"target {it['id']}")
        if it["meta"].get("qtype") not in ("choice", "noul", "score"):
            errors.append(f"qtype {it['id']}")
        if it["meta"]["qtype"] == "noul" and sorted(it["options"]) != ["no", "yes"]:
            errors.append(f"noul opts {it['id']}")
        if it["channel"] != "clean":
            errors.append(f"channel {it['id']}")
    dd = list(durs.values())
    if dd and (min(dd) < 2.0 or max(dd) > 60.0):
        errors.append(f"durations out of range {min(dd):.2f}-{max(dd):.2f}")

    # leakage and text checks
    leak = {}
    mmsu_q, mmsu_opts = set(), set()
    if MMSU_TRUTH.exists():
        for line in MMSU_TRUTH.read_text().splitlines():
            r = json.loads(line)
            mmsu_q.add(norm_q(r["question"]))
            for o in r.get("options") or []:
                if isinstance(o, str) and len(o) >= 12:
                    mmsu_opts.add(norm_q(o))
    leak["mmsu_questions_loaded"] = len(mmsu_q)
    leak["items_question_equal_mmsu"] = sum(1 for it in items if norm_q(it["question"]) in mmsu_q)
    leak["templates_equal_mmsu"] = sum(1 for q in L.all_question_templates() if norm_q(q) in mmsu_q)
    leak["items_option_equal_mmsu_long_option"] = sum(1 for it in items for o in it["options"]
                                                      if len(o) >= 12 and norm_q(o) in mmsu_opts and not L.is_generic_pattern(o))
    leak["generic_pattern_label_options_shared_with_mmsu (allowed: generic answer format)"] = len(
        {o for it in items for o in it["options"] if norm_q(o) in mmsu_opts and L.is_generic_pattern(o)})
    if leak["items_question_equal_mmsu"] or leak["templates_equal_mmsu"] or leak["items_option_equal_mmsu_long_option"]:
        errors.append("MMSU text overlap")
    ours_src = set()
    ours_spk = set()
    for it in items:
        for s in it["meta"].get("source_ids") or []:
            ours_src.add(s)
        for s in it["meta"].get("speakers") or [it["meta"].get("speaker")]:
            ours_spk.add(s)
    for name, path in (("adi", ADI_ITEMS), ("exp0_eval", EXP0_EVAL)):
        hits_src = hits_spk = 0
        vals = set()
        if path.exists():
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    a = r.get("audio") or {}
                    if isinstance(a, dict) and a.get("key"):
                        vals.add(str(a["key"]))
                    for v in (r.get("meta") or {}).values():
                        if isinstance(v, (str, int)):
                            vals.add(str(v))
                    q = r.get("question")
            for s in ours_src:
                if s in vals or f"libritts_r/{s}" in vals:
                    hits_src += 1
            for s in ours_spk:
                if s in vals:
                    hits_spk += 1
        leak[f"{name}_exists"] = path.exists()
        leak[f"{name}_source_id_hits"] = hits_src
        leak[f"{name}_speaker_hits"] = hits_spk
        if hits_src or hits_spk:
            errors.append(f"leak {name}")
    sbv_cal = sbv_cal_speakers()
    leak["sb_voice_cal_speaker_hits"] = len(ours_spk & sbv_cal)
    tr_spk = {s for it in items if it["set"] == "train" for s in (it["meta"].get("speakers") or [it["meta"]["speaker"]])}
    cal_spk = {s for it in items if it["set"] == "CAL" for s in (it["meta"].get("speakers") or [it["meta"]["speaker"]])}
    leak["train_cal_speaker_overlap"] = sorted(tr_spk & cal_spk)
    if leak["train_cal_speaker_overlap"] or leak["sb_voice_cal_speaker_hits"]:
        errors.append("cal overlap")

    # measured vs planned manipulation checks
    mcheck = collections.defaultdict(collections.Counter)
    for k, f in facts.items():
        fam = f.get("family")
        ff = f.get("facts") or {}
        for key in ("plan_rank_match", "plan_same_match", "plan_pair_match"):
            if key in ff and ff[key] is not None:
                mcheck[f"{fam}:{key}"][str(ff[key])] += 1
        if f.get("err"):
            mcheck[f"{fam}:drop_reason"][f["err"][:40]] += 1
        if fam in ("pause", "pause_meaning"):
            mcheck[f"{fam}:measured_matches_plan"][str(f.get("kept"))] += 1
    # re-measure a sample from the final banks (pause, duration, volume)
    spot = collections.Counter()
    srng = random.Random(1)
    sample = srng.sample(items, min(300, len(items)))
    banks = {}
    for it in sample:
        b = it["audio"]["bank"]
        if b not in banks:
            banks[b] = C.AudioBank(shards, b)
        y = banks[b].get(it["audio"]["key"])
        fam = it["meta"]["family"]
        if fam in ("pause", "pause_meaning"):
            sil = L.silences(y, L.NO_PAUSE_MAX_S)
            spot[f"{fam}:n_pauses_same"] += int(len(sil) == len(it["meta"]["measured_pauses"]))
            spot[f"{fam}:n"] += 1
        elif fam == "volume" and it["meta"].get("measured"):
            f = facts.get(it["audio"]["key"], {}).get("facts") or {}
            bs = f.get("bounds_s")
            if bs:
                v = [L.active_db(y[int(a * SR): int(b2 * SR)]) for a, b2 in bs]
                ok = all(x is not None and abs(x - m) < 0.3 for x, m in zip(v, it["meta"]["measured"]))
                spot["volume:remeasure_same"] += int(ok)
                spot["volume:n"] += 1
        elif fam == "duration":
            bs = (facts.get(it["audio"]["key"], {}).get("facts") or {}).get("measured")
            spot["duration:n"] += 1
            spot["duration:ok"] += int(bs is not None)

    # write OUT once: banks with only the used clips, items, then summary.json last
    if out.exists() and smoke:
        shutil.rmtree(out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    total_s = 0.0
    bank_hours = {}
    remap = {}  # (shard bank, key) -> output bank
    cur, cur_n, out_i = {}, 0, 0

    def flush():
        nonlocal cur, cur_n, out_i, total_s
        if cur:
            name = f"{BUILDER}_{out_i:02d}"
            C.write_audio_bank(out / "audio", name, cur)
            s_ = cur_n / SR
            bank_hours[name] = round(s_ / 3600, 3)
            total_s += s_
            out_i += 1
        cur, cur_n = {}, 0

    for b in sorted(used):
        src = C.AudioBank(shards, b)
        for k in sorted(used[b]):
            x = src.get(k)
            if not np.isfinite(x).all() or np.abs(x).max() > 0.99 + 1e-6:
                errors.append(f"audio range {b}/{k}")
            if k in cur:
                errors.append(f"duplicate clip key {k}")
            cur[k] = x
            cur_n += len(x)
            remap[(b, k)] = f"{BUILDER}_{out_i:02d}"
            if cur_n >= 450_000_000:  # ~1.8 GB per bank file
                flush()
    flush()
    for it in items:
        it["audio"] = {"bank": remap[(it["audio"]["bank"], it["audio"]["key"])], "key": it["audio"]["key"]}
    train = [it for it in items if it["set"] == "train"]
    cal = [it for it in items if it["set"] == "CAL"]
    C.write_jsonl(out / "items_train.jsonl", train)
    C.write_jsonl(out / "items_cal.jsonl", cal)
    # reload check
    rt = C.read_jsonl(out / "items_train.jsonl")
    rc = C.read_jsonl(out / "items_cal.jsonl")
    if len(rt) != len(train) or len(rc) != len(cal):
        errors.append("reload count mismatch")
    out_idx = {b: json.loads((out / "audio" / f"{b}.json").read_text()) for b in bank_hours}
    for it in rt + rc:
        if it["audio"]["key"] not in out_idx.get(it["audio"]["bank"], {}):
            errors.append(f"reload missing {it['id']}")
            break

    counts = collections.defaultdict(lambda: collections.Counter())
    for it in items:
        counts[it["source"]][f"{it['set']}:{it['meta']['qtype']}"] += 1
    label_balance = collections.defaultdict(collections.Counter)
    for it in items:
        label_balance[f"{it['source']}|{it['meta']['qkind']}"][str(it["meta"]["answer"])] += 1
    examples = {}
    for src in sorted(counts):
        ex = [it for it in items if it["source"] == src][:: max(1, len([1 for it in items if it["source"] == src]) // 3)][:3]
        examples[src] = [{"question": it["question"], "options": it["options"], "gold": it["gold"], "qtype": it["meta"]["qtype"]} for it in ex]
    clips_used = sum(len(v) for v in used.values())
    phone_clips = sum(1 for b in used for k in used[b] if (facts.get(k, {}).get("facts") or facts.get(k, {})).get("fx"))
    summary = {
        "builder": BUILDER, "smoke": smoke, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_train": len(train), "n_cal": len(cal), "cal_frac": round(len(cal) / max(1, len(items)), 4),
        "n_raw_items": n_raw, "n_items_dropped_clip_under_2s": n_short_dropped, "clips": clips_used, "audio_hours": round(total_s / 3600, 3), "bank_hours": bank_hours,
        "clip_seconds": {"min": round(min(dd), 2) if dd else None, "median": round(float(np.median(dd)), 2) if dd else None,
                         "max": round(max(dd), 2) if dd else None},
        "phone_fx_clip_share": round(phone_clips / max(1, clips_used), 3),
        "counts": {k: dict(v) for k, v in counts.items()},
        "by_qtype": dict(collections.Counter(it["meta"]["qtype"] for it in items)),
        "by_split_source": dict(collections.Counter(f"{it['set']}:{it['source']}" for it in items)),
        "label_balance": {k: dict(v) for k, v in label_balance.items()},
        "balance_stats": bal_stats,
        "sources": SOURCES,
        "manipulation_checks": {k: dict(v) for k, v in mcheck.items()},
        "final_bank_spot_check": dict(spot),
        "examples": examples,
        "validation": {"ok": not errors, "errors": errors[:50], "leakage": leak},
        "seconds": round(time.time() - t0),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return {"n_train": len(train), "n_cal": len(cal), "hours": summary["audio_hours"], "ok": not errors, "errors": errors[:10],
            "leak": leak, "by_split_source": summary["by_split_source"], "mcheck": summary["manipulation_checks"], "spot": dict(spot)}
