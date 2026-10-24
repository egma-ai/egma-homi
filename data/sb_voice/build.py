"""sb_voice: items about how something is said and who is speaking. Stage 1 (`process_<source>`) decodes one shard
and computes features; stage 2 (`assemble`) writes banks, items and summary.json; `validate` re-checks the output."""

from __future__ import annotations

import io
import json
import math
import random
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import common as C
import labels as L

BUILDER = "sb_voice"
OUT = Path("/vol/stageb/sb_voice")
WORK = OUT / "work2"  # per-shard clips + features
SR = C.SAMPLE_RATE
MIN_S, MAX_S = 1.0, 15.0
BANK_MAX_SAMPLES = 700_000_000  # ~2.8 GB float32 per bank file

SOURCES = {
    "vctk": {
        "name": "VCTK 0.92 (mic1)", "repo": "sanchit-gandhi/vctk", "rev": "73ef4ee7d49a6fed4ea1efd65f82b4c95faeb9de",
        "licence": "CC BY 4.0 — VCTK corpus licence (CSTR-Edinburgh/vctk card @31539806e8a6ee3b0c0fef88659ed518542bc564, "
                   "https://datashare.ed.ac.uk/handle/10283/3443). Audio + speaker-info (age/gender/accent) from the parquet "
                   "mirror sanchit-gandhi/vctk, a plain conversion of that corpus (mirror carries no licence tag of its own).",
    },
    "libritts_r": {
        "name": "LibriTTS-R dev-clean + train-clean-100 (shards 0-5)", "repo": "mythicinfinity/libritts_r",
        "rev": "0d1718db351207d5a493135a62d5ed8cc4733830",
        "licence": "CC BY 4.0 (HF cardData). Speaker gender: LibriSpeech SPEAKERS.TXT (CC BY 4.0, "
                   "https://www.openslr.org/resources/12/raw-metadata.tar.gz).",
    },
    "crema_d": {
        "name": "CREMA-D (training actors only)", "repo": "cfahlgren1/crema-d (via /vol/data/audio/crema_d)",
        "rev": "b7d5beded69285eb5eb6a1c6894a5b0ad17cab92; demographics github CheyneyComputerScience/CREMA-D@1658cd342dff90010aa843eaeebd53610a08b1dc",
        "licence": "ODbL 1.0 + Database Contents License (https://github.com/CheyneyComputerScience/CREMA-D)",
    },
    "vocalsound": {
        "name": "VocalSound (official 44.1 kHz release zip, resampled to 16 kHz), official train split speakers (datafiles/tr_rev.json in the zip)",
        "repo": "YuxinXie/VocalSound", "file": "vs_release_44k.zip", "rev": "dad0fcfa060d5291948f23fad33117068b28e121",
        "official": "https://github.com/YuanGongND/vocalsound@337ac5526f392987ab6e9214bf986bc923639338 (Dropbox links "
                    "'temporarily disabled' on 2026-10-07, so the release zip vs_release_44k.zip as mirrored on HF is used; the official split lists datafiles/{tr,val,te}_rev.json are read from inside the zip)",
        "licence": "CC BY-SA 4.0 (share-alike) — https://github.com/YuanGongND/vocalsound",
    },
    "so762": {
        "name": "speechocean762 train split", "repo": "mispeech/speechocean762",
        "rev": "06385584fad212b26134c656fdd3ccf9f093f33e", "licence": "Apache-2.0 (HF cardData)",
    },
    "synthetic_noise": {"name": "generated white/pink/brown/hum/rumble noise", "rev": "labels.py", "licence": "CC0 (generated)"},
}

VCTK_KEEP = 0.35  # share of each speaker's mic1 utterances kept (deterministic hash)
VCTK_N_SHARDS = 27
LIBRI_PLAN = [("dev.clean", i, 0.45) for i in range(4)] + [("train.clean.100", i, 0.30) for i in range(6)]
VOCALSOUND_KEEP = 0.70
CREMA_SENTENCES = {
    "IEO": "It's eleven o'clock", "TIE": "That is exactly what happened", "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about", "TAI": "The airplane is almost full", "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock", "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket", "ITS": "I think I've seen this before", "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}
HERE = Path(__file__).resolve().parent


# Stage 1 helpers

def _hf_file(repo: str, rev: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset", revision=rev, cache_dir="/tmp/hf")


def _hf_list(repo: str, rev: str, prefix: str) -> list[str]:
    from huggingface_hub import HfApi

    return sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=rev)
                  if f.startswith(prefix) and f.endswith(".parquet"))


def decode_bytes(b: bytes) -> np.ndarray:
    import soundfile as sf

    x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    return C.resample(x.mean(axis=1), sr, SR)


_CMU = None


def _init_worker():
    global _CMU
    try:
        import cmudict

        _CMU = cmudict.dict()
    except Exception as e:  # pragma: no cover - fallback to the heuristic counter only
        print(f"[sb_voice] cmudict unavailable ({e}); heuristic syllables only", flush=True)
        _CMU = None


def clip_features(x: np.ndarray, text: str | None) -> dict | None:
    """Acoustic features of one clip (see README 'Measured prosody')."""
    import librosa

    act = L.activity(x)
    if act is None:
        return None
    f0, vflag, _ = librosa.pyin(np.asarray(x, dtype=np.float32), fmin=65.0, fmax=500.0, sr=SR,
                                frame_length=1024, hop_length=320)
    rms = librosa.feature.rms(y=np.asarray(x, dtype=np.float32), frame_length=1024, hop_length=320, center=True)[0]
    n = min(len(rms), len(f0))
    # pYIN voiced and within 25 dB of the loudest frame; without the gate, VCTK room hum and breaths are tracked at fmin
    voiced = L.energy_gated_voicing(np.asarray(vflag[:n], bool) & np.isfinite(f0[:n]), rms[:n])
    fs = L.f0_stats(f0[:n], voiced)
    v = voiced
    loud = float(10 * np.log10(np.mean(rms[:n][v] ** 2) + 1e-20)) if v.sum() >= 10 else act["active_db"]
    syl = L.count_syllables(text, _CMU) if text else 0
    rate = syl / act["span_s"] if syl > 0 and act["span_s"] >= 0.5 else float("nan")
    out = {"loud_db": loud, **fs, "voiced_frac": float(v.mean()) if n else 0.0, "span_s": act["span_s"],
           "pause_frac": act["pause_frac"], "n_pauses": act["n_pauses"], "active_db": act["active_db"],
           "syll": syl, "rate": rate}
    return {k: (None if isinstance(v_, float) and not math.isfinite(v_) else (round(v_, 5) if isinstance(v_, float) else v_))
            for k, v_ in out.items()}


def _feat_star(args):
    x, text = args
    try:
        return clip_features(x, text)
    except Exception as e:  # never kill a shard for one bad clip
        return {"error": repr(e)[:200]}


def compute_features(xs: list, texts: list, n_proc: int) -> list:
    from multiprocessing import get_context

    with get_context("fork").Pool(n_proc, initializer=_init_worker) as pool:
        return pool.map(_feat_star, list(zip(xs, texts)), chunksize=4)


def save_work(name: str, clips: dict, records: list) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    C.write_audio_bank(WORK, name, clips)
    C.write_jsonl(WORK / f"{name}.jsonl", records)
    hours = sum(len(x) for x in clips.values()) / SR / 3600
    print(f"[sb_voice] wrote work/{name}: {len(records)} clips, {hours:.2f} h", flush=True)


# Stage 1: one function per source

def process_vctk(shard: int, n_proc: int) -> dict:
    import pyarrow.parquet as pq

    src = SOURCES["vctk"]
    files = _hf_list(src["repo"], src["rev"], "data/train-")
    assert len(files) == VCTK_N_SHARDS, files
    path = _hf_file(src["repo"], src["rev"], files[shard])
    xs, recs = [], []
    cols = ["speaker_id", "audio", "file", "text", "age", "gender", "accent", "region"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=cols):
        for r in batch.to_pylist():
            fname = Path(r["file"]).name
            if not fname.endswith("_mic1.flac"):
                continue
            uid = fname.removesuffix("_mic1.flac")
            if L.stable_frac(uid, "vctk") >= VCTK_KEEP:
                continue
            x = decode_bytes(r["audio"]["bytes"])
            dur = len(x) / SR
            if not MIN_S <= dur <= MAX_S:
                continue
            age = str(r.get("age") or "").strip()
            xs.append(x)
            recs.append({"key": f"vctk/{uid}", "source": "vctk", "source_id": f"{uid}_mic1",
                         "speaker": f"vctk:{r['speaker_id']}", "dur": round(dur, 3), "text": r.get("text") or None,
                         "gender": {"F": "female", "M": "male"}.get(str(r.get("gender")).strip()),
                         "age": int(age) if age.isdigit() else None, "accent": (r.get("accent") or "").strip() or None,
                         "region": (r.get("region") or "").strip() or None})
    feats = compute_features(xs, [r["text"] for r in recs], n_proc)
    for r, f in zip(recs, feats):
        r["feat"] = f
    save_work(f"vctk_{shard:02d}", {r["key"]: x for r, x in zip(recs, xs)}, recs)
    return {"shard": shard, "clips": len(recs)}


def process_libritts(split: str, shard: int, keep: float, n_proc: int) -> dict:
    import pyarrow.parquet as pq

    src = SOURCES["libritts_r"]
    files = _hf_list(src["repo"], src["rev"], f"data/{split}/")
    path = _hf_file(src["repo"], src["rev"], files[shard])
    gender = json.loads((HERE / "librispeech_speaker_gender.json").read_text())["gender"]
    xs, recs = [], []
    cols = ["audio", "text_normalized", "speaker_id", "id", "chapter_id"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=cols):
        for r in batch.to_pylist():
            if L.stable_frac(r["id"], "libritts_r") >= keep or str(r["speaker_id"]) not in gender:
                continue
            x = decode_bytes(r["audio"]["bytes"])
            dur = len(x) / SR
            if not MIN_S <= dur <= MAX_S:
                continue
            xs.append(x)
            recs.append({"key": f"libritts_r/{r['id']}", "source": "libritts_r", "source_id": r["id"],
                         "speaker": f"libritts_r:{r['speaker_id']}", "dur": round(dur, 3),
                         "text": r["text_normalized"], "gender": gender[str(r["speaker_id"])], "subset": split})
    feats = compute_features(xs, [r["text"] for r in recs], n_proc)
    for r, f in zip(recs, feats):
        r["feat"] = f
    tag = "dev" if split == "dev.clean" else "tc100"
    save_work(f"libritts_{tag}_{shard:02d}", {r["key"]: x for r, x in zip(recs, xs)}, recs)
    return {"split": split, "shard": shard, "clips": len(recs)}


def crema_train_clips() -> dict:
    """CREMA-D clips of the training actors (actors in /vol/data/train.jsonl), minus every clip and actor that
    appears in /vol/data/eval_items.jsonl."""
    train = [it for it in C.read_jsonl("/vol/data/train.jsonl") if it.get("source") == "crema_d"]
    eval_keys, eval_actors = set(), set()
    for it in C.read_jsonl("/vol/data/eval_items.jsonl"):
        if it.get("audio") and it["audio"]["bank"] == "crema_d":
            eval_keys.add(it["audio"]["key"])
            eval_actors.add(str(it["meta"].get("actor")))
    clips = {}
    for it in train:
        key, actor = it["audio"]["key"], str(it["meta"]["actor"])
        if key in eval_keys or actor in eval_actors:
            continue
        clips[key] = {"actor": actor, "emotion": it["meta"]["emotion"], "votes": it["meta"]["votes"]}
    return clips


def process_crema(chunk: int, n_chunks: int, n_proc: int) -> dict:
    demo = json.loads((HERE / "crema_demographics.json").read_text())["actors"]
    meta = crema_train_clips()
    keys = sorted(meta)[chunk::n_chunks]
    bank = C.AudioBank("/vol/data/audio", "crema_d")
    xs, recs = [], []
    for key in keys:
        x = bank.get(key)
        dur = len(x) / SR
        if not MIN_S <= dur <= MAX_S:
            continue
        parts = key.split("_")  # 1001_DFA_ANG_XX
        m = meta[key]
        d = demo[m["actor"]]
        xs.append(x)
        recs.append({"key": f"crema_d/{key}", "source": "crema_d", "source_id": key, "speaker": f"crema_d:{m['actor']}",
                     "dur": round(dur, 3), "text": CREMA_SENTENCES[parts[1]], "gender": d["gender"], "age": d["age"],
                     "emotion": m["emotion"], "votes": m["votes"], "intensity": parts[3]})
    feats = compute_features(xs, [r["text"] for r in recs], n_proc)
    for r, f in zip(recs, feats):
        r["feat"] = f
    save_work(f"crema_{chunk:02d}", {r["key"]: x for r, x in zip(recs, xs)}, recs)
    return {"chunk": chunk, "clips": len(recs)}


def _vs_split_speakers(zf: zipfile.ZipFile) -> dict:
    """Official VocalSound splits from the release zip's datafiles/{tr,val,te}_rev.json ({"data": [{"spk_id", "wav",
    "labels"}]}). Returns {"train": set, "val": set, "test": set} of speaker ids; train must be disjoint from the others."""
    out = {}
    for split, stem in (("train", "tr"), ("val", "val"), ("test", "te")):
        names = [n for n in zf.namelist() if not n.startswith("__MACOSX") and n.endswith(f"datafiles/{stem}_rev.json")]
        if not names:
            raise RuntimeError(f"VocalSound zip has no datafiles/{stem}_rev.json")
        out[split] = {str(d["spk_id"]) for d in json.loads(zf.read(names[0]))["data"]}
    if out["train"] & (out["val"] | out["test"]):
        raise RuntimeError("VocalSound train speakers overlap val/test")
    print(f"[sb_voice] VocalSound official split speakers: { {k: len(v) for k, v in out.items()} }", flush=True)
    return out


def process_vocalsound(n_proc: int) -> dict:
    import soundfile as sf

    src = SOURCES["vocalsound"]
    t0 = time.time()
    zpath = Path(_hf_file(src["repo"], src["rev"], src["file"]))
    print(f"[sb_voice] VocalSound zip {zpath.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f}s", flush=True)
    zf = zipfile.ZipFile(zpath)
    train_spk = _vs_split_speakers(zf)["train"]
    wavs = sorted(n for n in zf.namelist() if n.endswith(".wav") and not n.startswith("__MACOSX") and "/._" not in n)
    print(f"[sb_voice] VocalSound: {len(wavs)} wavs, {len(train_spk)} train speakers, e.g. {wavs[:2]} {sorted(train_spk)[:3]}",
          flush=True)
    xs, recs, skipped = [], [], Counter()
    for n in wavs:
        stem = Path(n).stem  # f0003_0_cough
        parts = stem.split("_")
        spk, label = parts[0], parts[-1]
        if spk not in train_spk:
            skipped["not_train"] += 1
            continue
        if L.stable_frac(stem, "vocalsound") >= VOCALSOUND_KEEP:
            skipped["subsample"] += 1
            continue
        x, sr = sf.read(io.BytesIO(zf.read(n)), dtype="float32", always_2d=True)
        x = C.resample(x.mean(axis=1), sr, SR)
        dur = len(x) / SR
        if dur < MIN_S:
            skipped["too_short"] += 1
            continue
        if dur > MAX_S:
            x, dur = x[: int(MAX_S * SR)], MAX_S
        xs.append(x)
        recs.append({"key": f"vocalsound/{stem}", "source": "vocalsound", "source_id": stem, "speaker": f"vocalsound:{spk}",
                     "dur": round(dur, 3), "label": label, "gender": {"f": "female", "m": "male"}.get(spk[:1])})
    print(f"[sb_voice] VocalSound kept {len(recs)}; skipped {dict(skipped)}; labels {Counter(r['label'] for r in recs)}", flush=True)
    save_work("vocalsound_00", {r["key"]: x for r, x in zip(recs, xs)}, recs)
    return {"clips": len(recs), "skipped": dict(skipped)}


def process_so762(n_proc: int) -> dict:
    import pyarrow.parquet as pq

    src = SOURCES["so762"]
    path = _hf_file(src["repo"], src["rev"], "data/train-00000-of-00001.parquet")
    xs, recs = [], []
    cols = ["accuracy", "fluency", "prosodic", "total", "completeness", "text", "speaker", "gender", "age", "audio"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=cols):
        for r in batch.to_pylist():
            stem = Path(r["audio"].get("path") or "").stem
            x = decode_bytes(r["audio"]["bytes"])
            dur = len(x) / SR
            if not MIN_S <= dur <= MAX_S:
                continue
            xs.append(x)
            recs.append({"key": f"so762/{stem}", "source": "so762", "source_id": stem, "speaker": f"so762:{r['speaker']}",
                         "dur": round(dur, 3), "text": r["text"], "gender": {"f": "female", "m": "male"}.get(str(r["gender"]).lower()),
                         "age": int(r["age"]), "scores": {k: r[k] for k in ("accuracy", "fluency", "prosodic", "total", "completeness")}})
    save_work("so762_00", {r["key"]: x for r, x in zip(recs, xs)}, recs)
    return {"clips": len(recs)}


# Stage 2: question wording
YES_NO = ["yes", "no"]


class NoShuffle(random.Random):
    """rng for make_item whose shuffle is a no-op (score items keep their natural order)."""

    def shuffle(self, x, *a, **k):  # noqa: D401
        return None


NO_SHUFFLE = NoShuffle(0)

PROSODY = {
    "loudness": {
        "levels": ["quiet", "normal", "loud"],
        "score_q": ["How loud is the speaker?", "How loudly is the person speaking?", "Rate the speaker's volume.",
                    "How would you describe the speaker's loudness?", "How strong is the speaker's voice volume in this clip?",
                    "What is the volume of the speaker's voice like?"],
        "hi_q": ["Is the speaker talking loudly?", "Is the person raising their voice?", "Is the speaker speaking louder than usual?",
                 "Does the speaker sound loud?", "Is the voice loud in this recording?"],
        "lo_q": ["Is the speaker talking quietly?", "Is the person speaking softly?", "Does the speaker sound quiet?",
                 "Is the speaker keeping their voice down?", "Is the voice soft or quiet here?"],
    },
    "rate": {
        "levels": ["slow", "normal", "fast"],
        "score_q": ["How fast is the speaker talking?", "What is the speaker's speaking rate?", "How quickly does the person speak?",
                    "Describe the pace of the speech.", "At what speed is the speaker talking?", "How would you rate the speed of the speech?"],
        "hi_q": ["Is the speaker talking fast?", "Is the person speaking quickly?", "Does the speaker talk at a fast pace?",
                 "Is the speech rushed or fast?", "Is the speaker speaking faster than usual?"],
        "lo_q": ["Is the speaker talking slowly?", "Is the person speaking at a slow pace?", "Does the speaker take their time when talking?",
                 "Is the speech slow?", "Is the speaker speaking more slowly than usual?"],
    },
    "pitch": {
        "levels": ["lower than usual", "their usual pitch", "higher than usual"],
        "score_q": ["Compared with this speaker's normal voice, how high is their pitch here?",
                    "Is the speaker's pitch lower or higher than their usual speaking voice?",
                    "How does the speaker's pitch compare with how they normally sound?",
                    "Relative to their typical voice, where is the speaker's pitch in this clip?",
                    "Rate the speaker's pitch relative to their normal voice."],
        "hi_q": ["Is the speaker's voice pitched higher than usual?", "Is the speaker talking in a raised pitch?",
                 "Does the speaker's voice go up in pitch compared with their normal voice?",
                 "Is the speaker using a higher pitch than they normally do?", "Does the speaker sound higher-pitched than usual?"],
        "lo_q": ["Is the speaker's voice pitched lower than usual?", "Is the speaker talking in a lowered pitch?",
                 "Is the speaker using a deeper pitch than they normally do?", "Does the speaker sound lower-pitched than usual?",
                 "Has the speaker dropped their pitch compared with their normal voice?"],
    },
    "pitch_variation": {
        "levels": ["monotone", "somewhat varied", "very expressive"],
        "score_q": ["How much does the speaker's pitch vary?", "How expressive is the speaker's intonation?",
                    "How lively is the speaker's tone of voice?", "Describe the speaker's intonation.",
                    "How much melody is there in the speaker's voice?"],
        "hi_q": ["Is the speaker's intonation expressive?", "Does the speaker's pitch vary a lot?", "Does the speaker sound animated?",
                 "Is the speaker talking in a lively, varied tone?", "Does the speaker use a lot of intonation?"],
        "lo_q": ["Does the speaker sound monotone?", "Is the speaker talking in a flat, monotone voice?", "Is the speaker's intonation flat?",
                 "Does the speaker speak without much pitch variation?", "Is the delivery monotonous?"],
    },
    "pauses": {
        "levels": ["no long pauses", "a few pauses", "many pauses"],
        "score_q": ["How much does the speaker pause while talking?", "How often does the speaker pause?",
                    "How choppy is the speech?", "How many long pauses are there in the speech?",
                    "Describe the pauses in the speaker's speech."],
        "hi_q": ["Does the speaker pause a lot?", "Are there long pauses in the speech?", "Does the speaker stop and pause often while talking?",
                 "Is the speech broken up by pauses?", "Does the speaker hesitate or pause between words?"],
        "lo_q": ["Does the speaker talk without pausing?", "Is the speech continuous, with no long pauses?",
                 "Does the speaker speak in one smooth stretch?", "Is the speech free of long pauses?",
                 "Does the speaker keep talking without stopping?"],
    },
}
PROSODY_FEATURE = {"loudness": "loud_db", "rate": "rate", "pitch": "f0_med", "pitch_variation": "f0_spread_st",
                   "pauses": "pause_frac"}

GENDER_OPTS = ["male voice", "female voice"]
GENDER_CHOICE_Q = ["Is this a male or a female voice?", "What kind of voice is this?", "Which best describes the speaker's voice?",
                   "Does this voice sound male or female?", "How would you classify the speaker's voice?"]
GENDER_NOUL_Q = {"male": ["Is the speaker a man?", "Is this a male voice?", "Is the speaker male?", "Is this a man's voice?",
                          "Does the voice belong to a man?"],
                 "female": ["Is the speaker a woman?", "Is this a female voice?", "Is the speaker female?", "Is this a woman's voice?",
                            "Does the voice belong to a woman?"]}
GENDER_VS_CHOICE_Q = ["Was this sound made by a male or a female voice?", "Whose voice makes this sound?",
                      "Does this vocal sound come from a male or a female voice?", "Which best describes the voice making this sound?",
                      "Is the person making this sound male or female?"]
GENDER_VS_NOUL_Q = {"male": ["Is this sound made by a man?", "Does a male voice make this sound?", "Is the person making this sound male?",
                             "Does this sound come from a man?", "Is it a man making this sound?"],
                    "female": ["Is this sound made by a woman?", "Does a female voice make this sound?", "Is the person making this sound female?",
                               "Does this sound come from a woman?", "Is it a woman making this sound?"]}

ACCENT_CHOICE_Q = ["Which accent does the speaker have?", "Where is the speaker's accent from?", "What kind of English accent does the speaker have?",
                   "Which best describes the speaker's accent?", "Judging by their accent, where is the speaker most likely from?"]
ACCENT_NOUL_T = ["Does the speaker have {a} accent?", "Is this speaker's accent {adj}?", "Would you say the speaker has {a} accent?",
                 "Does the speaker speak with {a} accent?", "Is the accent you hear {adj}?"]
ACCENT_ADJ = {"American": "an American", "Canadian": "a Canadian", "English (England)": "an English (England)",
              "Scottish": "a Scottish", "Irish": "an Irish", "Indian": "an Indian",
              "Australian or New Zealand": "an Australian or New Zealand", "South African": "a South African"}

CREMA_AGE_LEVELS = ["under 30", "30 to 49", "50 or older"]
CREMA_AGE_SCORE_Q = ["How old does the speaker sound?", "Which age group is the speaker in?", "Estimate the speaker's age.",
                     "What is the speaker's likely age range?", "How old is the speaker?"]
CREMA_AGE_NOUL = {"old": ["Is the speaker 50 or older?", "Does the speaker sound like an older adult (50+)?", "Is the speaker over fifty?",
                          "Is this an older speaker, aged 50 or more?", "Would you say the speaker is at least 50 years old?"],
                  "young": ["Is the speaker under 30?", "Does the speaker sound like a young adult (under 30)?", "Is the speaker younger than thirty?",
                            "Is this a young speaker, below 30?", "Would you say the speaker is in their twenties or younger?"]}
VCTK_AGE_OPTS = ["under 25", "25 or older"]
VCTK_AGE_CHOICE_Q = ["How old is the speaker?", "Which age group does the speaker belong to?", "Estimate the speaker's age.",
                     "Is the speaker under 25, or 25 and older?", "What is the speaker's likely age group?"]
VCTK_AGE_NOUL_Q = ["Is the speaker younger than 25?", "Is the speaker under 25 years old?", "Does the speaker sound younger than 25?",
                   "Is this speaker below the age of 25?", "Would you say the speaker is under 25?"]
CHILD_OPTS = ["a child", "an adult"]
CHILD_CHOICE_Q = ["Is the speaker a child or an adult?", "Who is speaking: a child or an adult?", "Does this voice belong to a child or an adult?",
                  "What kind of speaker is this?", "Which best describes the speaker?"]
CHILD_NOUL_Q = ["Is the speaker a child?", "Is this a child's voice?", "Is a child speaking?", "Does the speaker sound like a kid?",
                "Is the person talking a young child?"]

SPK_COUNT_OPTS = ["one", "two"]
SPK_COUNT_Q = ["How many different people speak in this recording?", "How many speakers are in this clip?",
               "How many distinct voices can you hear?", "Count the number of different speakers.", "How many people are talking in this recording?"]
SAME_SPK_Q = ["Are both sentences spoken by the same person?", "Is it the same voice throughout the recording?",
              "Do the two parts of the recording come from the same speaker?", "Is only one person speaking in this recording?",
              "Is the second sentence said by the same speaker as the first?"]
MULTI_SPK_Q = ["Is there more than one speaker?", "Do two different people speak in this recording?", "Can you hear two different voices?",
               "Does a second person speak after the first one?", "Are the two sentences spoken by different people?"]

VS_LABELS = ["laughter", "sigh", "cough", "throatclearing", "sneeze", "sniff"]
VS_OPTS = {"laughter": "laughing", "sigh": "sighing", "cough": "coughing", "throatclearing": "clearing their throat",
           "sneeze": "sneezing", "sniff": "sniffing"}
VS_CHOICE_Q = ["Which sound does the person make?", "What vocal sound is in this recording?", "What is the person doing in this clip?",
               "Which of these sounds do you hear?", "What kind of non-speech sound does the person make?"]
VS_FORMS = {  # verb, gerund, noun
    "laughter": ("laugh", "laughing", "laughter"), "sigh": ("sigh", "sighing", "a sigh"), "cough": ("cough", "coughing", "coughing"),
    "throatclearing": ("clear their throat", "clearing their throat", "throat clearing"), "sneeze": ("sneeze", "sneezing", "a sneeze"),
    "sniff": ("sniff", "sniffing", "sniffing"),
}
VS_NOUL_T = ["Does the person {verb}?", "Is someone {ger}?", "Can you hear {noun}?", "Is the person {ger} in this clip?",
             "Does this recording contain {noun}?"]

SO_SPECS = {
    "accuracy": (["poor", "fair", "good"], ["How accurate is the speaker's pronunciation?", "Rate the speaker's pronunciation.",
                                            "How well does the speaker pronounce the words?", "How correct is the speaker's English pronunciation?",
                                            "How good is the speaker's pronunciation?"]),
    "fluency": (["not fluent", "somewhat fluent", "fluent"], ["How fluent is the speaker?", "How smoothly does the speaker read the sentence?",
                                                              "Rate the fluency of the speech.", "How fluently does the person speak?",
                                                              "How smooth and natural is the speaker's delivery?"]),
    "prosodic": (["flat or unnatural", "somewhat natural", "natural"], ["How natural are the speaker's intonation and rhythm?",
                                                                       "Rate the speaker's prosody (rhythm, stress and intonation).",
                                                                       "How natural does the speaker's sentence melody sound?",
                                                                       "How native-like is the rhythm of the speech?",
                                                                       "How well does the speaker use stress and intonation?"]),
}
SO_CLEAR_Q = ["Is the pronunciation clear?", "Does the speaker pronounce the words correctly?", "Is the speaker's pronunciation good?",
              "Are the words pronounced clearly and accurately?", "Is the speaker easy to understand?"]
SO_FLUENT_Q = ["Does the speaker read fluently?", "Is the speech fluent?", "Does the speaker speak smoothly, without stumbling?",
               "Is the speaker a fluent reader?", "Does the speaker read the sentence without hesitating?"]

INTENSITY_LEVELS = ["low", "medium", "high"]
INTENSITY_SCORE_Q = ["How intense is the speaker's delivery?", "How strongly is the emotion expressed?", "Rate the intensity of the speaker's voice.",
                     "How forceful is the speaker's delivery?", "What is the emotional intensity of the speech?"]
INTENSITY_HI_Q = ["Is the speaker speaking with high intensity?", "Is the emotion expressed strongly?", "Does the speaker sound intense?",
                  "Is the delivery forceful?", "Is the speaker's emotion very strong here?"]
INTENSITY_LO_Q = ["Is the emotion expressed only mildly?", "Is the speaker's delivery subdued?", "Does the speaker sound restrained?",
                  "Is the speaker speaking with low intensity?", "Is the emotion in the voice weak or understated?"]

EMO_WORDS = {"anger": ["angry", "annoyed", "irritated"], "happy": ["happy", "cheerful", "pleased"], "sad": ["sad", "unhappy", "down"],
             "fear": ["scared", "afraid", "anxious"], "disgust": ["disgusted", "repulsed", "grossed out"],
             "neutral": ["neutral", "emotionally flat", "matter-of-fact"]}
EMO_NOUL_T = ["Does the speaker sound {w}?", "Is the speaker {w}?", "Would you say the speaker is {w}?", "Does the voice sound {w}?",
              "Is the person {w}?"]
UPSET_Q = ["Is the speaker upset?", "Does the speaker sound upset?", "Does the person sound distressed or unhappy?",
           "Is the speaker in a bad mood?", "Is something bothering the speaker?"]
VALENCE_OPTS = ["positive", "neutral", "negative"]
VALENCE_Q = ["Is the speaker's tone positive, neutral, or negative?", "What is the overall emotional tone of the speaker?",
             "How would you describe the speaker's mood?", "Which best describes the feeling in the speaker's voice?",
             "What is the emotional valence of the speaker's voice?"]
NEGATIVE = {"anger", "sad", "fear", "disgust"}

NOISE_OPTS = ["quiet background", "noisy background"]
NOISE_CHOICE_Q = ["What is the background like in this recording?", "How would you describe the recording background?",
                  "Which best describes the background of the audio?", "Is the background quiet or noisy?", "What are the recording conditions like?"]
NOISE_YES_Q = ["Is there background noise?", "Can you hear noise behind the speaker?", "Is the recording noisy?",
               "Is there a hiss, hum or other noise in the background?", "Is the speech mixed with background noise?"]
NOISE_NO_Q = ["Is the recording clean, without background noise?", "Is the background silent apart from the speaker?",
              "Is this a clean recording?", "Is the audio free of background noise?", "Is the speaker recorded in a quiet place?"]

# Quotas (train items; CAL gets CAL_FRAC of each, from CAL speakers only)
CAL_FRAC = 0.032
Q_PROSODY = {"noul": 1600, "score": 1300, "choice": 500}  # per feature
Q_GENDER = {"vctk": 1600, "libritts_r": 1500, "crema_d": 1000, "so762": 500, "vocalsound": 1200}
Q_ACCENT = 3400
Q_AGE = {"crema_d": 1500, "vctk": 700, "so762": 1400}
Q_PAIRS = 4200
Q_NOISE = 4000
Q_VS = {"choice": 4200, "noul": 3600}
Q_INTENSITY = 1400
Q_EMOTION = {"emotion_noul": 2900, "upset": 1400, "valence": 1450}  # <= ~6k incl. CAL
NOISY_SHARE = 0.20
PHONE_P = 0.19
PHONE_OK = {"loudness", "rate", "pitch", "pitch_variation", "pauses", "gender", "age", "accent", "speaker_count",
            "vocal_sound", "intensity", "emotion"}
MAX_ITEMS_PER_CLIP = 4


# Stage 2: assemble

class Builder:
    def __init__(self, recs: list[dict]):
        self.rng = random.Random(f"{C.SEED}:{BUILDER}")
        self.recs = {r["key"]: r for r in recs}
        self.n_items = Counter()  # clip key -> number of items
        self.items = {"train": [], "CAL": []}
        self.counter = 0
        self.cap_skips = 0
        self.cal_speakers = self.pick_cal_speakers()
        for r in self.recs.values():
            r["split"] = "CAL" if r["speaker"] in self.cal_speakers else "train"
        self.pairs = []  # synthetic 2-speaker clip specs
        self.speaker_stats = {}

    # Helpers
    def pick_cal_speakers(self) -> set:
        rng = random.Random(f"{C.SEED}:{BUILDER}:cal")
        by = defaultdict(lambda: defaultdict(set))  # source -> group -> speakers
        for r in self.recs.values():
            s = r["source"]
            if s == "vctk":
                g = (L.accent_group(r.get("accent")) or "none", r.get("gender"))
            elif s == "so762":
                g = ("child" if (r.get("age") or 99) <= 12 else "adult", r.get("gender"))
            elif s == "libritts_r":
                g = (r.get("subset"), r.get("gender"))
            else:
                g = ("all", r.get("gender"))
            by[s][g].add(r["speaker"])
        cal = set()
        want = {"vctk": [("English (England)", "female"), ("American", "male"), ("Scottish", "female"), ("Irish", "male"),
                         ("Canadian", "female")],
                "libritts_r": [("dev.clean", "female"), ("dev.clean", "male"), ("train.clean.100", "female"), ("train.clean.100", "male")],
                "crema_d": [("all", "female"), ("all", "male")],
                "so762": [("child", "female"), ("child", "male"), ("adult", "female"), ("adult", "male")]}
        for s, groups in want.items():
            for g in groups:
                pool = sorted(by[s].get(g, set()))
                if pool:
                    cal.add(rng.choice(pool))
        vs = sorted({r["speaker"] for r in self.recs.values() if r["source"] == "vocalsound"})
        cal |= {s for s in vs if L.stable_frac(s, "vs_cal") < 0.03}
        return cal

    def clips(self, source=None, split=None, pred=None) -> list[dict]:
        out = [r for r in self.recs.values() if (source is None or r["source"] == source or (isinstance(source, (set, tuple, list))
               and r["source"] in source)) and (split is None or r["split"] == split) and (pred is None or pred(r))]
        out.sort(key=lambda r: r["key"])
        return out

    def free(self, r) -> bool:
        return self.n_items[r["key"]] < MAX_ITEMS_PER_CLIP

    def channel(self, family: str) -> str:
        return "phone" if family in PHONE_OK and self.rng.random() < PHONE_P else "clean"

    def add(self, r, family, view, question, options, target, gold, meta=None, audio_key=None, bank_source=None,
            shuffle=True, channel=None, split=None):
        split = split or r["split"]
        if audio_key is None and not self.free(r):
            self.cap_skips += 1
            return None
        if audio_key is not None:
            if self.n_items[audio_key] >= MAX_ITEMS_PER_CLIP:
                return None
            self.n_items[audio_key] += 1
        self.counter += 1
        qtype = {"score": "score", "noul": "noul"}.get(view, "choice")
        m = {"qtype": qtype, "family": family, "speaker": r.get("speaker"), "source_id": r.get("source_id"),
             "corpus": r.get("source"), "licence": SOURCES[r["source"]]["licence"] if r.get("source") in SOURCES else None}
        m.update(meta or {})
        item = C.make_item(
            item_id=f"{BUILDER}/{split}/{family}/{self.counter:06d}", set_name=split, source=f"{BUILDER}_{family}",
            question=question, options=options, target=target, gold=gold,
            rng=self.rng if shuffle and qtype != "score" else NO_SHUFFLE,
            audio={"bank": bank_source or r["source"], "key": audio_key or r["key"]},
            channel=channel or self.channel(family), state_text=None, meta=m)
        self.items[split].append(item)
        if audio_key is None:
            self.n_items[r["key"]] += 1
        return item

    def add_noul(self, r, family, question, yes: bool, p_yes=None, **kw):
        p = (1.0 if yes else 0.0) if p_yes is None else p_yes
        return self.add(r, family, "noul", question, YES_NO, [p, 1 - p], 0 if p >= 0.5 else 1, **kw)

    def quota(self, q: int, split: str) -> int:
        return q if split == "train" else max(4, int(round(q * CAL_FRAC)))

    def take(self, pool, n, key=None):
        """First n free clips of a seeded shuffle of pool."""
        pool = list(pool)
        self.rng.shuffle(pool)
        out = []
        for r in pool:
            if len(out) >= n:
                break
            if self.free(r):
                out.append(r)
        return out

    def balanced(self, groups: dict, n_total: int) -> dict:
        """Equal numbers per group (as many as the smallest group allows, up to n_total / k each)."""
        k = len(groups)
        per = n_total // k
        shuffled = {}
        for g, pool in groups.items():
            pool = [r for r in pool if self.free(r)]
            self.rng.shuffle(pool)
            shuffled[g] = pool
        per = min([per] + [len(p) for p in shuffled.values()])
        return {g: p[:per] for g, p in shuffled.items()}

    # Families
    def build_noise_plan(self):
        """20 % of the VCTK + LibriTTS-R clips become noisy (stored mixed with synthetic noise)."""
        rng = random.Random(f"{C.SEED}:{BUILDER}:noise")
        for r in self.clips({"vctk", "libritts_r"}):
            if r.get("feat") and "error" not in r["feat"] and rng.random() < NOISY_SHARE:
                r["noisy"] = {"type": rng.choice(L.NOISE_TYPES), "snr_db": round(rng.uniform(0.0, 20.0), 1)}

    def prosody(self):
        srcs = {"vctk", "libritts_r", "crema_d"}
        base = [r for r in self.clips(srcs) if r.get("feat") and "error" not in r["feat"] and not r.get("noisy")]
        spk = [r["speaker"] for r in base]
        for fam, feat_name in PROSODY_FEATURE.items():
            vals = [r["feat"].get(feat_name) for r in base]
            vals = [np.nan if v is None else float(v) for v in vals]
            pr = L.speaker_relative_levels(spk, vals)
            # speaker stats (traceability)
            stats = {}
            arr_spk = np.array(spk)
            arr_val = np.array(vals, dtype=float)
            for s in sorted(set(spk)):
                v = arr_val[(arr_spk == s) & np.isfinite(arr_val)]
                if len(v) >= L.MIN_CLIPS_PER_SPEAKER:
                    q33, q50, q67 = np.percentile(v, [100 / 3, 50, 200 / 3])
                    stats[s] = {"n": int(len(v)), "q33": round(float(q33), 4), "median": round(float(q50), 4), "q67": round(float(q67), 4)}
            self.speaker_stats[fam] = stats
            labelled = []
            for r, p, v in zip(base, pr, vals):
                if not np.isfinite(p) or r["speaker"] not in stats:
                    continue
                st = stats[r["speaker"]]
                if fam == "pauses":
                    lvl = L.pause_level(r["feat"]["pause_frac"], r["feat"]["n_pauses"], st["q67"])
                    tgt = [0.0, 0.0, 0.0]
                    tgt[lvl] = 1.0
                    clear = lvl in (0, 2)
                else:
                    lvl = L.tertile_level(p)
                    tgt = L.tertile_target(p)
                    clear = p < 1 / 3 - L.SOFT_BAND or p > 2 / 3 + L.SOFT_BAND
                meta = {"feature": feat_name, "value": round(v, 4), "pct_rank_in_speaker": round(float(p), 4),
                        "level": lvl, "speaker_stats": st}
                labelled.append((r, lvl, tgt, clear, meta))
            spec = PROSODY[fam]
            for split in ("train", "CAL"):
                rows = [x for x in labelled if x[0]["split"] == split]
                # noul: clear extremes only, equal numbers top / bottom, each asked with both polarities
                n_noul = self.quota(Q_PROSODY["noul"], split)
                ext = self.balanced({0: [x[0] for x in rows if x[3] and x[1] == 0], 2: [x[0] for x in rows if x[3] and x[1] == 2]}, n_noul)
                by_key = {x[0]["key"]: x for x in rows}
                for lvl, lst in ext.items():
                    for i, r in enumerate(lst):
                        _, lv, tgt, _, meta = by_key[r["key"]]
                        ask_hi = i % 2 == 0
                        q = self.rng.choice(spec["hi_q"] if ask_hi else spec["lo_q"])
                        yes = (lv == 2) == ask_hi
                        self.add_noul(r, fam, q, yes, meta={**meta, "asked": "high" if ask_hi else "low"})
                # score (all levels) and choice (same options, shuffled). Tertile levels are balanced by construction;
                # pause levels are not (most read sentences have no long pause), so sample them level-balanced
                for view in ("score", "choice"):
                    n = self.quota(Q_PROSODY[view], split)
                    if fam == "pauses":
                        pool = []
                        for lv_ in range(3):
                            pool += self.take([x[0] for x in rows if x[1] == lv_], n // 3)
                        pool = self.take(pool, n)
                    else:
                        pool = self.take([x[0] for x in rows], n)
                    for r in pool:
                        _, lv, tgt, _, meta = by_key[r["key"]]
                        self.add(r, fam, view, self.rng.choice(spec["score_q"]), spec["levels"], tgt, int(np.argmax(tgt)), meta=meta)

    def gender(self):
        for src, q in Q_GENDER.items():
            vs = src == "vocalsound"
            pred = (lambda r: r.get("gender") in ("male", "female") and (r["source"] != "so762" or (r.get("age") or 0) >= 16))
            for split in ("train", "CAL"):
                pools = {g: self.clips(src, split, lambda r, g=g: pred(r) and r["gender"] == g) for g in ("male", "female")}
                sel = self.balanced(pools, self.quota(q, split))
                for g, lst in sel.items():
                    for i, r in enumerate(lst):
                        meta = {"label": g, "label_source": "corpus metadata"}
                        if i % 2 == 0:  # choice view
                            qs = GENDER_VS_CHOICE_Q if vs else GENDER_CHOICE_Q
                            gold = 0 if g == "male" else 1
                            self.add(r, "gender", "choice", self.rng.choice(qs), GENDER_OPTS, [1.0 - gold, float(gold)], gold, meta=meta)
                        else:  # noul, asked about male or female (alternating so yes/no is 50/50)
                            asked = "male" if (i // 2) % 2 == 0 else "female"
                            qs = (GENDER_VS_NOUL_Q if vs else GENDER_NOUL_Q)[asked]
                            self.add_noul(r, "gender", self.rng.choice(qs), asked == g, meta={**meta, "asked": asked})

    def age(self):
        # CREMA-D: 3 age brackets (score) + noul old/young on the extremes
        for split in ("train", "CAL"):
            pools = {b: self.clips("crema_d", split, lambda r, b=b: L.crema_age_bracket(r["age"]) == b) for b in range(3)}
            n = self.quota(Q_AGE["crema_d"], split)
            sel = self.balanced(pools, n)
            for b, lst in sel.items():
                for i, r in enumerate(lst):
                    meta = {"label": CREMA_AGE_LEVELS[b], "age": r["age"]}
                    if b == 1 or i % 2 == 0:
                        t = [0.0, 0.0, 0.0]
                        t[b] = 1.0
                        self.add(r, "age", "score", self.rng.choice(CREMA_AGE_SCORE_Q), CREMA_AGE_LEVELS, t, b, meta=meta)
                    else:
                        asked = "old" if (i // 2) % 2 == 0 else "young"
                        yes = (asked == "old" and b == 2) or (asked == "young" and b == 0)
                        self.add_noul(r, "age", self.rng.choice(CREMA_AGE_NOUL[asked]), yes, meta={**meta, "asked": asked})
            # VCTK: under 25 vs 25+
            pools = {g: self.clips("vctk", split, lambda r, g=g: r.get("age") is not None and (r["age"] < 25) == g) for g in (True, False)}
            sel = self.balanced(pools, self.quota(Q_AGE["vctk"], split))
            for young, lst in sel.items():
                for i, r in enumerate(lst):
                    meta = {"label": "under 25" if young else "25 or older", "age": r["age"]}
                    if i % 2 == 0:
                        gold = 0 if young else 1
                        self.add(r, "age", "choice", self.rng.choice(VCTK_AGE_CHOICE_Q), VCTK_AGE_OPTS, [1.0 - gold, float(gold)], gold, meta=meta)
                    else:
                        self.add_noul(r, "age", self.rng.choice(VCTK_AGE_NOUL_Q), young, meta=meta)
            # speechocean762: child (<= 12) vs adult (>= 18)
            pools = {c: self.clips("so762", split, lambda r, c=c: L.child_label(r["age"]) is c) for c in (True, False)}
            sel = self.balanced(pools, self.quota(Q_AGE["so762"], split))
            for child, lst in sel.items():
                for i, r in enumerate(lst):
                    meta = {"label": "child" if child else "adult", "age": r["age"], "child_rule": "child <= 12, adult >= 18"}
                    if i % 2 == 0:
                        gold = 0 if child else 1
                        self.add(r, "child", "choice", self.rng.choice(CHILD_CHOICE_Q), CHILD_OPTS, [1.0 - gold, float(gold)], gold, meta=meta)
                    else:
                        self.add_noul(r, "child", self.rng.choice(CHILD_NOUL_Q), child, meta=meta)

    def accent(self):
        groups_all = sorted({L.accent_group(r.get("accent")) for r in self.clips("vctk")} - {None})
        for split in ("train", "CAL"):
            n = self.quota(Q_ACCENT, split)
            pools = defaultdict(list)
            for r in self.clips("vctk", split, lambda r: L.accent_group(r.get("accent")) is not None):
                pools[L.accent_group(r["accent"])].append(r)
            # roughly balanced over groups: each group gets up to n/k, leftovers go to the big groups
            per = max(1, n // max(1, len(pools)))
            chosen = []
            for g in sorted(pools):
                chosen += [(g, r) for r in self.take(pools[g], per)]
            rest = n - len(chosen)
            if rest > 0:
                used = {r["key"] for _, r in chosen}
                extra = self.take([r for g in pools for r in pools[g] if r["key"] not in used], rest)
                chosen += [(L.accent_group(r["accent"]), r) for r in extra]
            for i, (g, r) in enumerate(chosen):
                meta = {"label": g, "accent_raw": r.get("accent"), "region": r.get("region")}
                if i % 2 == 0:
                    opts = groups_all
                    gold = opts.index(g)
                    t = [0.0] * len(opts)
                    t[gold] = 1.0
                    self.add(r, "accent", "choice", self.rng.choice(ACCENT_CHOICE_Q), opts, t, gold, meta=meta)
                else:
                    yes = (i // 2) % 2 == 0
                    if not yes and len(groups_all) < 2:
                        continue
                    asked = g if yes else self.rng.choice([x for x in groups_all if x != g])
                    q = self.rng.choice(ACCENT_NOUL_T).format(a=ACCENT_ADJ[asked], adj=asked if asked != "English (England)" else "English (from England)")
                    self.add_noul(r, "accent", q, yes, meta={**meta, "asked": asked})

    def noise(self):
        for split in ("train", "CAL"):
            n = self.quota(Q_NOISE, split)
            noisy = self.take(self.clips({"vctk", "libritts_r"}, split, lambda r: bool(r.get("noisy"))), n // 2)
            clean = self.take(self.clips({"vctk", "libritts_r"}, split,
                                         lambda r: not r.get("noisy") and bool(r.get("feat")) and "error" not in r["feat"]), n // 2)
            m = min(len(noisy), len(clean))
            for is_noisy, lst in ((True, noisy[:m]), (False, clean[:m])):
                for i, r in enumerate(lst):
                    meta = {"label": "noisy" if is_noisy else "clean", "noise": r.get("noisy"),
                            "label_rule": "label = whether we mixed synthetic noise into the clip"}
                    view = i % 3
                    if view == 0:
                        gold = 1 if is_noisy else 0
                        self.add(r, "background_noise", "choice", self.rng.choice(NOISE_CHOICE_Q), NOISE_OPTS,
                                 [1.0 - gold, float(gold)], gold, meta=meta, channel="clean")
                    else:
                        ask_noise = (i // 3) % 2 == 0
                        q = self.rng.choice(NOISE_YES_Q if ask_noise else NOISE_NO_Q)
                        self.add_noul(r, "background_noise", q, is_noisy == ask_noise, meta={**meta, "asked": "noisy" if ask_noise else "clean"},
                                      channel="clean")
            # the noisy clips not used for noise questions still serve gender / accent questions (added before)

    def vocal_sound(self):
        for split in ("train", "CAL"):
            pools = {lab: self.clips("vocalsound", split, lambda r, lab=lab: r["label"] == lab) for lab in VS_LABELS}
            sel = self.balanced(pools, self.quota(Q_VS["choice"] + Q_VS["noul"], split))
            n_choice = self.quota(Q_VS["choice"], split)
            flat = [(lab, r) for lab, lst in sel.items() for r in lst]
            self.rng.shuffle(flat)
            opts = [VS_OPTS[l] for l in VS_LABELS]
            noul_i = 0
            for i, (lab, r) in enumerate(flat):
                meta = {"label": lab, "label_source": "VocalSound class (filename)"}
                if i < n_choice:
                    gold = VS_LABELS.index(lab)
                    t = [0.0] * 6
                    t[gold] = 1.0
                    self.add(r, "vocal_sound", "choice", self.rng.choice(VS_CHOICE_Q), opts, t, gold, meta=meta)
                else:
                    yes = noul_i % 2 == 0
                    noul_i += 1
                    asked = lab if yes else self.rng.choice([x for x in VS_LABELS if x != lab])
                    verb, ger, noun = VS_FORMS[asked]
                    q = self.rng.choice(VS_NOUL_T).format(verb=verb, ger=ger, noun=noun)
                    self.add_noul(r, "vocal_sound", q, yes, meta={**meta, "asked": asked})

    def pronunciation(self):
        for split in ("train", "CAL"):
            clips = self.clips("so762", split)
            for dim, (levels, qs) in SO_SPECS.items():
                pools = {lv: [r for r in clips if L.so762_level(r["scores"][dim]) == lv] for lv in range(3)}
                # keep all low / mid, and as many 'good' as low + mid together / 2 (the corpus is ~70 % 'good')
                n_good = (len(pools[0]) + len(pools[1])) // 2
                for lv in range(3):
                    lst = self.take(pools[lv], len(pools[lv]) if lv < 2 else n_good)
                    for r in lst:
                        t = [0.0, 0.0, 0.0]
                        t[lv] = 1.0
                        self.add(r, "pronunciation", "score", self.rng.choice(qs), levels, t, lv, channel="clean",
                                 meta={"dimension": dim, "score_0_10": r["scores"][dim], "level_rule": "<=5 / 6-7 / >=8",
                                       "age": r.get("age")})
            for dim, qs in (("accuracy", SO_CLEAR_Q), ("fluency", SO_FLUENT_Q)):
                pools = {True: [r for r in clips if r["scores"][dim] >= 9], False: [r for r in clips if r["scores"][dim] <= 5]}
                sel = self.balanced(pools, 10**6)
                for good, lst in sel.items():
                    for r in lst:
                        self.add_noul(r, "pronunciation", self.rng.choice(qs), good, channel="clean",
                                      meta={"dimension": dim, "score_0_10": r["scores"][dim], "label_rule": "yes >= 9, no <= 5"})

    def intensity(self):
        code = {"LO": 0, "MD": 1, "HI": 2}
        for split in ("train", "CAL"):
            clips = self.clips("crema_d", split, lambda r: r.get("intensity") in code)
            n = self.quota(Q_INTENSITY, split)
            ext = self.balanced({0: [r for r in clips if r["intensity"] == "LO"], 2: [r for r in clips if r["intensity"] == "HI"]}, n // 2)
            for lv, lst in ext.items():
                for i, r in enumerate(lst):
                    ask_hi = i % 2 == 0
                    q = self.rng.choice(INTENSITY_HI_Q if ask_hi else INTENSITY_LO_Q)
                    self.add_noul(r, "intensity", q, (lv == 2) == ask_hi, meta={"label": r["intensity"], "emotion": r["emotion"],
                                                                               "asked": "high" if ask_hi else "low"})
            for r in self.take(clips, n - n // 2):
                lv = code[r["intensity"]]
                t = [0.0, 0.0, 0.0]
                t[lv] = 1.0
                self.add(r, "intensity", "score", self.rng.choice(INTENSITY_SCORE_Q), INTENSITY_LEVELS, t, lv,
                         meta={"label": r["intensity"], "emotion": r["emotion"]})

    def emotion(self):
        """New phrasings of CREMA-D emotion questions (train actors). Target = 0.4 x listener-vote share + 0.6 x actor intent."""
        for split in ("train", "CAL"):
            clips = self.clips("crema_d", split, lambda r: r.get("votes") and sum(r["votes"]) > 0)
            # specific emotion yes/no, 50/50
            n = self.quota(Q_EMOTION["emotion_noul"], split)
            for i, r in enumerate(self.take(clips, n)):
                intended = C.EMOTION_CODES[r["emotion"]]
                share = np.asarray(r["votes"], float) / sum(r["votes"])
                asked = intended if i % 2 == 0 else self.rng.choice([e for e in C.EMOTIONS if e != intended])
                p = 0.4 * float(share[C.EMOTIONS.index(asked)]) + 0.6 * float(asked == intended)
                q = self.rng.choice(EMO_NOUL_T).format(w=self.rng.choice(EMO_WORDS[asked]))
                self.add_noul(r, "emotion", q, None, p_yes=round(p, 4),
                              meta={"intended": intended, "asked": asked, "votes": r["votes"], "target_rule": "0.4*vote share + 0.6*intended"})
            # upset (negative emotions) yes/no, 50/50 by intended group
            n = self.quota(Q_EMOTION["upset"], split)
            sel = self.balanced({True: [r for r in clips if C.EMOTION_CODES[r["emotion"]] in NEGATIVE],
                                 False: [r for r in clips if C.EMOTION_CODES[r["emotion"]] not in NEGATIVE]}, n)
            for neg, lst in sel.items():
                for r in lst:
                    share = np.asarray(r["votes"], float) / sum(r["votes"])
                    p = 0.4 * float(sum(share[C.EMOTIONS.index(e)] for e in NEGATIVE)) + 0.6 * float(neg)
                    self.add_noul(r, "emotion", self.rng.choice(UPSET_Q), None, p_yes=round(p, 4),
                                  meta={"intended": C.EMOTION_CODES[r["emotion"]], "asked": "upset", "votes": r["votes"],
                                        "target_rule": "0.4*negative vote share + 0.6*intended negative"})
            # valence choice (positive / neutral / negative)
            n = self.quota(Q_EMOTION["valence"], split)
            grp = {"happy": 0, "neutral": 1}
            for r in self.take(clips, n):
                share = np.asarray(r["votes"], float) / sum(r["votes"])
                g_share = [share[3], share[4], sum(share[C.EMOTIONS.index(e)] for e in NEGATIVE)]
                g = grp.get(C.EMOTION_CODES[r["emotion"]], 2)
                t = [0.4 * float(s) for s in g_share]
                t[g] += 0.6
                self.add(r, "emotion", "choice", self.rng.choice(VALENCE_Q), VALENCE_OPTS, t, g,
                         meta={"intended": C.EMOTION_CODES[r["emotion"]], "votes": r["votes"], "target_rule": "0.4*grouped votes + 0.6*intended group"})

    def speaker_count(self):
        """Synthetic 2-utterance clips: same speaker vs two different speakers (half of them same gender)."""
        rng = random.Random(f"{C.SEED}:{BUILDER}:pairs")
        seen_pairs = set()
        for split in ("train", "CAL"):
            n = self.quota(Q_PAIRS, split)
            for src, share in (("vctk", 0.6), ("libritts_r", 0.4)):
                pool = self.clips(src, split, lambda r: not r.get("noisy") and r.get("feat") and "error" not in r["feat"]
                                  and r["dur"] <= 6.5 and r["feat"].get("active_db") is not None)
                by_spk = defaultdict(list)
                for r in pool:
                    by_spk[r["speaker"]].append(r)
                spks = sorted(s for s in by_spk if len(by_spk[s]) >= 2)
                if len(spks) < 2:
                    continue
                by_gender = defaultdict(list)
                for s in spks:
                    by_gender[by_spk[s][0]["gender"]].append(s)
                m = int(round(n * share))
                for i in range(m):
                    same = i % 2 == 0
                    s1 = rng.choice(spks)
                    if same:
                        a, b = rng.sample(by_spk[s1], 2)
                    else:
                        cands = by_gender[by_spk[s1][0]["gender"]] if (i // 2) % 2 == 0 else spks
                        others = [s for s in cands if s != s1] or [s for s in spks if s != s1]
                        s2 = rng.choice(others)
                        a, b = rng.choice(by_spk[s1]), rng.choice(by_spk[s2])
                    gap = round(rng.uniform(0.3, 0.8), 3)
                    if a["dur"] + b["dur"] + gap > MAX_S:
                        continue
                    key = f"pair/{a['key'].split('/', 1)[1]}+{b['key'].split('/', 1)[1]}"
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    spec = {"key": key, "a": a["key"], "b": b["key"], "gap_s": gap, "same": same,
                            "jit_a": round(rng.uniform(-3, 3), 2), "jit_b": round(rng.uniform(-3, 3), 2),
                            "target_db": -23.0, "dur": round(a["dur"] + b["dur"] + gap, 3), "source": src}
                    self.pairs.append(spec)
                    r = {"key": key, "source": src, "speaker": f"{a['speaker']}+{b['speaker']}", "source_id": f"{a['source_id']}+{b['source_id']}",
                         "split": split}
                    meta = {"label": "same speaker" if same else "two speakers", "speakers": [a["speaker"], b["speaker"]],
                            "same_gender": a["gender"] == b["gender"], "gap_s": gap, "parts": [a["source_id"], b["source_id"]]}
                    view = i % 3
                    if view == 0:
                        gold = 0 if same else 1
                        self.add(r, "speaker_count", "choice", self.rng.choice(SPK_COUNT_Q), SPK_COUNT_OPTS, [1.0 - gold, float(gold)], gold,
                                 meta=meta, audio_key=key, bank_source="pairs")
                    elif view == 1:
                        self.add_noul(r, "speaker_count", self.rng.choice(SAME_SPK_Q), same, meta=meta, audio_key=key, bank_source="pairs")
                    else:
                        self.add_noul(r, "speaker_count", self.rng.choice(MULTI_SPK_Q), not same, meta=meta, audio_key=key, bank_source="pairs")

    def build(self):
        self.build_noise_plan()
        for step in (self.intensity, self.pronunciation, self.emotion, self.age, self.accent, self.noise, self.gender,
                     self.prosody, self.vocal_sound, self.speaker_count):
            n0 = sum(len(v) for v in self.items.values())
            step()
            print(f"[sb_voice] {step.__name__}: +{sum(len(v) for v in self.items.values()) - n0} items", flush=True)
        return self


# Stage 2: banks + outputs
BANK_NAMES = {"vctk": "sb_voice_vctk", "libritts_r": "sb_voice_libritts", "crema_d": "sb_voice_crema",
              "vocalsound": "sb_voice_vocalsound", "so762": "sb_voice_so762", "pairs": "sb_voice_pairs"}


def load_records() -> tuple[list[dict], dict]:
    recs, shard_of = [], {}
    for p in sorted(WORK.glob("*.jsonl")):
        for r in C.read_jsonl(p):
            shard_of[r["key"]] = p.stem
            recs.append(r)
    return recs, shard_of


def noisy_audio(x: np.ndarray, r: dict) -> np.ndarray:
    seed = int(L.hashlib.md5(r["key"].encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    noise = L.colored_noise(len(x), r["noisy"]["type"], rng)
    ps = 10 ** (r["feat"]["active_db"] / 10.0) if r["feat"].get("active_db") is not None else None
    return L.mix_at_snr(x, noise, r["noisy"]["snr_db"], speech_power=ps)


def assemble() -> dict:
    t0 = time.time()
    recs, shard_of = load_records()
    print(f"[sb_voice] {len(recs)} clip records from {len(set(shard_of.values()))} work shards", flush=True)
    b = Builder(recs).build()
    items = b.items

    # audio needed per bank
    need = defaultdict(set)
    for split in items:
        for it in items[split]:
            need[it["audio"]["bank"]].add(it["audio"]["key"])
    pair_by_key = {p["key"]: p for p in b.pairs}
    work_banks = {}

    def get_clip(key):
        sh = shard_of[key]
        if sh not in work_banks:
            work_banks[sh] = C.AudioBank(WORK, sh)
        return work_banks[sh].get(key)

    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    bank_of = {}  # (logical bank, key) -> physical bank name
    bank_stats = {}
    for logical in sorted(need):
        keys = sorted(need[logical])
        clips, part, size, parts = {}, 0, 0, []

        def flush():
            nonlocal clips, part, size
            if clips:
                name = BANK_NAMES[logical] + (f"_{part}" if part or size > 0 else "")
                parts.append((name, clips))
            clips, size = {}, 0
            part += 1

        for key in keys:
            if logical == "pairs":
                p = pair_by_key[key]
                ra, rb = b.recs[p["a"]], b.recs[p["b"]]
                x = L.concat_pair(get_clip(p["a"]), get_clip(p["b"]), p["gap_s"], ra["feat"]["active_db"], rb["feat"]["active_db"],
                                  p["target_db"], p["jit_a"], p["jit_b"])
            else:
                x = get_clip(key)
                r = b.recs[key]
                if r.get("noisy"):
                    x = noisy_audio(x, r)
                x = L.peak_limit(x)
            if size + len(x) > BANK_MAX_SAMPLES:
                flush()
            clips[key] = x
            size += len(x)
        flush()
        if len(parts) == 1:  # single part: plain name without suffix
            parts = [(BANK_NAMES[logical], parts[0][1])]
        for name, cl in parts:
            C.write_audio_bank(OUT / "audio", name, cl)
            for k in cl:
                bank_of[(logical, k)] = name
            bank_stats[name] = {"clips": len(cl), "hours": round(sum(len(v) for v in cl.values()) / SR / 3600, 3)}
            print(f"[sb_voice] bank {name}: {bank_stats[name]}", flush=True)
        work_banks.clear()

    for split in items:
        for it in items[split]:
            it["audio"]["bank"] = bank_of[(it["audio"]["bank"], it["audio"]["key"])]
        random.Random(f"{C.SEED}:{BUILDER}:order:{split}").shuffle(items[split])
    C.write_jsonl(OUT / "items_train.jsonl", items["train"])
    C.write_jsonl(OUT / "items_cal.jsonl", items["CAL"])
    (OUT / "speaker_stats.json").write_text(json.dumps(b.speaker_stats))
    (OUT / "cal_speakers.json").write_text(json.dumps(sorted(b.cal_speakers)))
    summary = make_summary(items, bank_stats, b)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[sb_voice] assemble done in {time.time() - t0:.0f}s: train {len(items['train'])}, CAL {len(items['CAL'])}", flush=True)
    return summary


def make_summary(items, bank_stats, b) -> dict:
    allit = items["train"] + items["CAL"]
    by_src = Counter((it["set"], it["source"]) for it in allit)
    by_type = Counter((it["source"], it["meta"]["qtype"]) for it in allit)
    corpus = Counter((it["meta"].get("corpus"), it["set"]) for it in allit)
    ex = {}
    for it in items["train"]:
        lst = ex.setdefault(it["source"], [])
        if len(lst) < 3:
            lst.append({"question": it["question"], "options": it["options"], "gold": it["gold"],
                        "target": it["target"], "channel": it["channel"], "audio": it["audio"]})
    noul_bal = {}
    for it in allit:
        if it["meta"]["qtype"] == "noul":
            d = noul_bal.setdefault(it["source"], Counter())
            d[it["options"][it["gold"]]] += 1
    return {
        "builder": BUILDER, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_train": len(items["train"]), "n_cal": len(items["CAL"]),
        "counts_by_split_source": {f"{s}|{src}": n for (s, src), n in sorted(by_src.items())},
        "counts_by_source_qtype": {f"{src}|{q}": n for (src, q), n in sorted(by_type.items())},
        "counts_by_corpus_split": {f"{c}|{s}": n for (c, s), n in sorted(corpus.items(), key=str)},
        "phone_share": round(sum(it["channel"] == "phone" for it in allit) / max(1, len(allit)), 4),
        "noul_yes_no": {k: dict(v) for k, v in noul_bal.items()},
        "banks": bank_stats, "total_audio_hours": round(sum(v["hours"] for v in bank_stats.values()), 2),
        "sources": SOURCES, "cal_speakers": sorted(b.cal_speakers),
        "n_pairs": len(b.pairs), "cap_skips": b.cap_skips, "n_noisy_clips_used": sum(1 for k, r in b.recs.items() if r.get("noisy") and b.n_items[k] > 0),
        "examples": ex,
    }


# Validation + leakage
REQUIRED = ["id", "set", "source", "question", "options", "target", "gold", "audio", "channel", "state_text", "meta"]


def validate(sample: int | None = None) -> dict:
    report = {"errors": [], "checks": {}}
    tr = C.read_jsonl(OUT / "items_train.jsonl")
    cal = C.read_jsonl(OUT / "items_cal.jsonl")
    items = tr + cal
    ids = [it["id"] for it in items]
    if len(set(ids)) != len(ids):
        report["errors"].append("duplicate ids")
    banks = {}
    durs = []
    per_clip = Counter()
    for it in items:
        for k in REQUIRED:
            if k not in it:
                report["errors"].append(f"{it.get('id')}: missing {k}")
        n = len(it["options"])
        if not 2 <= n <= 20 or len(it["target"]) != n:
            report["errors"].append(f"{it['id']}: options/target length")
        if abs(sum(it["target"]) - 1) > 1e-4:
            report["errors"].append(f"{it['id']}: target sum {sum(it['target'])}")
        if it["gold"] != int(np.argmax(it["target"])):
            report["errors"].append(f"{it['id']}: gold != argmax")
        if it["meta"].get("qtype") not in ("choice", "noul", "score"):
            report["errors"].append(f"{it['id']}: qtype")
        if it["meta"]["qtype"] == "noul" and sorted(it["options"]) != ["no", "yes"]:
            report["errors"].append(f"{it['id']}: noul options")
        if it["channel"] not in ("clean", "phone"):
            report["errors"].append(f"{it['id']}: channel")
        if not it["id"].startswith(BUILDER + "/") or not it["source"].startswith(BUILDER):
            report["errors"].append(f"{it['id']}: prefix")
        a = it["audio"]
        if a["bank"] not in banks:
            banks[a["bank"]] = C.AudioBank(OUT / "audio", a["bank"])
        if a["key"] not in banks[a["bank"]].index:
            report["errors"].append(f"{it['id']}: audio key missing")
        else:
            durs.append(banks[a["bank"]].index[a["key"]][1] / SR)
        per_clip[(a["bank"], a["key"])] += 1
    # a sample of real waveforms: finite, peak <= 1, phone channel runs
    rng = random.Random(1)
    for it in rng.sample(items, min(sample or 300, len(items))):
        raw = banks[it["audio"]["bank"]].get(it["audio"]["key"])
        x = C.item_audio(it, banks)  # phone channel applied for phone items
        if not np.all(np.isfinite(raw)) or np.max(np.abs(raw)) > 0.9901 or len(raw) < MIN_S * SR - 1 or not np.all(np.isfinite(x)):
            report["errors"].append(f"{it['id']}: bad waveform (peak {float(np.max(np.abs(raw))):.3f}, len {len(raw)})")
    durs = np.array(durs)
    report["checks"] = {
        "n_items": len(items), "n_train": len(tr), "n_cal": len(cal), "cal_share": round(len(cal) / len(items), 4),
        "dur_min_s": round(float(durs.min()), 2), "dur_max_s": round(float(durs.max()), 2), "dur_mean_s": round(float(durs.mean()), 2),
        "max_items_per_clip": max(per_clip.values()), "clips_over_cap": [list(k) for k, v in per_clip.items() if v > MAX_ITEMS_PER_CLIP][:5], "phone_share": round(float(np.mean([it["channel"] == "phone" for it in items])), 4),
        "soft_target_items": sum(1 for it in items if max(it["target"]) < 0.999),
        "noul_yes_share_by_source": {s: round(float(np.mean([it["options"][it["gold"]] == "yes" for it in items
                                                             if it["source"] == s and it["meta"]["qtype"] == "noul"])), 3)
                                     for s in sorted({it["source"] for it in items if it["meta"]["qtype"] == "noul"})},
        "gold_position_hist": dict(Counter(it["gold"] for it in items if it["meta"]["qtype"] == "choice")),
        "question_paraphrases_by_source": {s: len({it["question"] for it in items if it["source"] == s}) for s in sorted({it["source"] for it in items})},
        "train_cal_speaker_overlap": sorted({sp for it in tr for sp in str(it["meta"].get("speaker")).split("+")} &
                                            {sp for it in cal for sp in str(it["meta"].get("speaker")).split("+")})[:20],
    }
    if report["checks"]["train_cal_speaker_overlap"]:
        report["errors"].append("train/CAL speaker overlap")
    report["leakage"] = leakage_check(items)
    report["errors"] = report["errors"][:50]
    report["ok"] = not report["errors"] and report["leakage"].get("ok", True)
    s = json.loads((OUT / "summary.json").read_text())
    s["validation"] = report
    (OUT / "summary.json").write_text(json.dumps(s, indent=2))
    return report


def leakage_check(items: list[dict]) -> dict:
    """No CREMA-D clip or actor from /vol/data/eval_items.jsonl; no speaker or recording id shared with the
    evaluation set (/vol/adi/items.jsonl)."""
    out = {}
    eval_actors, eval_keys = set(), set()
    for it in C.read_jsonl("/vol/data/eval_items.jsonl"):
        if it.get("audio") and it["audio"]["bank"] == "crema_d":
            eval_keys.add(it["audio"]["key"])
            eval_actors.add(str(it["meta"].get("actor")))
    mine_crema = {it["meta"]["source_id"] for it in items if it["meta"].get("corpus") == "crema_d"}
    mine_actors = {it["meta"]["speaker"].split(":")[1] for it in items if it["meta"].get("corpus") == "crema_d"}
    out["exp0_eval_crema_clip_overlap"] = len(mine_crema & eval_keys)
    out["exp0_eval_crema_actor_overlap"] = sorted(mine_actors & eval_actors)
    adi = Path("/vol/adi/items.jsonl")
    if adi.exists():
        adi_items = C.read_jsonl(adi)
        # compare ids within the same corpus only (CREMA-D actor "1012" is not speechocean762 speaker "1012")
        adi_src = {"crema_d": "crema_d", "vocalsound": "vocalsound", "so762": "speechocean762"}
        adi_spk, adi_ids = defaultdict(set), defaultdict(set)
        for it in adi_items:
            m = it.get("meta") or {}
            src = it.get("source")
            sp = m.get("actor") or m.get("speaker")
            if sp is not None:
                adi_spk[src].add(str(sp))
            for k in (it.get("item_key"), (it.get("audio") or {}).get("key")):
                if k:
                    adi_ids[src].add(str(k))
        mine_spk, mine_ids = defaultdict(set), defaultdict(set)
        for it in items:
            m = it["meta"]
            c = m.get("corpus")
            for sp in str(m.get("speaker")).split("+"):
                if ":" in sp:
                    mine_spk[c].add(sp.split(":", 1)[1])
            for sid in str(m.get("source_id")).split("+"):
                mine_ids[c].add(sid)
        res = {}
        for c, a_src in adi_src.items():
            res[c] = {"adi_source": a_src, "adi_speakers": len(adi_spk[a_src]), "our_speakers": len(mine_spk[c]),
                      "speaker_overlap": sorted(mine_spk[c] & adi_spk[a_src])[:20],
                      "recording_overlap": sorted(mine_ids[c] & adi_ids[a_src])[:20]}
        out["adi_items"] = len(adi_items)
        out["adi_sources"] = dict(Counter(str(it.get("source")) for it in adi_items))
        out["per_corpus"] = res
        out["not_in_adi"] = "VCTK, LibriTTS-R and synthetic noise are not evaluation-set sources; its MMSU clips carry no source ids " \
                            "(content overlap with VCTK/LibriTTS-R cannot be checked by id)"
        out["ok"] = not any(v["speaker_overlap"] or v["recording_overlap"] for v in res.values()) \
            and not out["exp0_eval_crema_clip_overlap"] and not out["exp0_eval_crema_actor_overlap"]
    else:
        out["adi_items"] = None
        out["note"] = "/vol/adi/items.jsonl not found at validation time; re-run validate once it exists"
        out["ok"] = not out["exp0_eval_crema_clip_overlap"] and not out["exp0_eval_crema_actor_overlap"]
    return out


def _fingerprint(x: np.ndarray) -> np.ndarray:
    """Channel-level fingerprint: mean + std of 40-band log-mel over speech-level frames, centred, unit norm."""
    import librosa

    S = np.log(librosa.feature.melspectrogram(y=np.asarray(x, np.float32), sr=SR, n_fft=512, hop_length=160, n_mels=40) + 1e-8)
    e = S.mean(axis=0)
    S = S[:, e > e.max() - 6.0]
    v = np.concatenate([S.mean(axis=1), S.std(axis=1)])
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-9)


def mmsu_content_check(threshold: float = 0.995, max_dur_diff: float = 0.3) -> dict:
    """The evaluation set's MMSU clips carry no source ids, so look for our VCTK / LibriTTS-R recordings among them by
    content: duration-matched fingerprint candidates -> frame alignment -> waveform cross-correlation."""
    adi_items = [it for it in C.read_jsonl("/vol/adi/items.jsonl") if it.get("source") == "mmsu" and it.get("audio")]
    adi_bank = C.AudioBank("/vol/adi/audio", adi_items[0]["audio"]["bank"])
    task_of = {it["audio"]["key"]: (it.get("meta") or {}).get("task") for it in adi_items}
    akeys = sorted(task_of)
    A = np.stack([_fingerprint(adi_bank.get(k)) for k in akeys])
    adur = np.array([adi_bank.index[k][1] / SR for k in akeys])
    bkeys, B, bdur = [], [], []
    for p in sorted((OUT / "audio").glob("sb_voice_*.json")):
        name = p.stem
        if not name.startswith(("sb_voice_vctk", "sb_voice_libritts")):
            continue
        bank = C.AudioBank(OUT / "audio", name)
        for k in bank.index:
            bkeys.append(k)
            B.append(_fingerprint(bank.get(k)))
            bdur.append(bank.index[k][1] / SR)
    B, bdur = np.stack(B), np.array(bdur)
    sims = A @ B.T
    close = np.abs(adur[:, None] - bdur[None, :]) <= max_dur_diff
    best = np.where(close, sims, -1).max(axis=1)
    arg = np.where(close, sims, -1).argmax(axis=1)
    # stage 2: for every MMSU clip, its 3 best duration-matched candidates get a frame-level alignment score
    # (max over +-0.5 s offsets of the mean cosine between band-centred log-mel frames): ~1 for the same recording,
    # far lower for different recordings
    banks = {}

    def clip(key):
        name = next(n for n in bank_names if key in banks.setdefault(n, C.AudioBank(OUT / "audio", n)).index)
        return banks[name].get(key)

    bank_names = sorted(p.stem for p in (OUT / "audio").glob("sb_voice_*.json") if p.stem.startswith(("sb_voice_vctk", "sb_voice_libritts")))
    masked = np.where(close, sims, -1)
    results = []
    for i in range(len(akeys)):
        cands = np.argsort(-masked[i])[:3]
        sa = _logmel_frames(adi_bank.get(akeys[i]))
        best_al, best_k = -1.0, None
        for j in cands:
            if masked[i, j] < 0:
                continue
            al = _align_score(sa, _logmel_frames(clip(bkeys[j])))
            if al > best_al:
                best_al, best_k = al, bkeys[j]
        results.append((best_al, i, best_k))
    results.sort(reverse=True)
    # stage 3, for the 40 best matches: peak normalised waveform cross-correlation. The same recording (even
    # resampled / re-encoded) gives ~0.9+, a different recording of the same sentence ~0.1
    hits = []
    for a, i, k in results[:40]:
        xc = _wave_xcorr(adi_bank.get(akeys[i]), clip(k))
        hits.append({"mmsu_key": akeys[i], "task": task_of[akeys[i]], "our_key": k, "align": round(float(a), 4),
                     "fp_cos": round(float(best[i]), 5), "wave_xcorr": round(xc, 4)})
    hits.sort(key=lambda h: -h["wave_xcorr"])
    n_dup = sum(1 for h in hits if h["wave_xcorr"] >= 0.6)
    return {"n_mmsu_clips": len(akeys), "n_our_clips": len(bkeys),
            "rule": "candidates = 3 best fingerprint matches (cos of mean/std log-mel, duration within 0.3 s) -> frame "
                    "alignment score -> top 40 checked by peak normalised waveform cross-correlation; duplicate if >= 0.6",
            "n_duplicates": n_dup, "top_matches": hits[:12], "ok": n_dup == 0}


def _wave_xcorr(a: np.ndarray, b: np.ndarray) -> float:
    """max_k |sum a[t] b[t+k]| / (||a|| ||b||) via FFT (both clips 16 kHz)."""
    a = np.asarray(a, np.float64) - np.mean(a)
    b = np.asarray(b, np.float64) - np.mean(b)
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    c = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    return float(np.max(np.abs(c)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _logmel_frames(x: np.ndarray) -> np.ndarray:
    import librosa

    S = np.log(librosa.feature.melspectrogram(y=np.asarray(x, np.float32), sr=SR, n_fft=512, hop_length=160, n_mels=40) + 1e-8)
    S = S - S.mean(axis=1, keepdims=True)
    return S / (np.linalg.norm(S, axis=0, keepdims=True) + 1e-9)


def _align_score(a: np.ndarray, b: np.ndarray, max_shift: int = 50) -> float:
    best = -1.0
    for sh in range(-max_shift, max_shift + 1):
        if sh >= 0:
            x, y = a[:, sh:], b
        else:
            x, y = a, b[:, -sh:]
        t = min(x.shape[1], y.shape[1])
        if t < 50:
            continue
        best = max(best, float(np.mean(np.sum(x[:, :t] * y[:, :t], axis=0))))
    return best
