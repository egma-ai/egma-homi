"""sb_phone_pitch: phone-line skills and pitch / prosody questions without VCTK or Vystadial. Part A rebuilds the
sb_phone question families on HarperValleyBank, LibriTTS-R, Apollo, MInDS-14, robocalls and sb_calls clips; part B adds
relative, absolute and cross-speaker prosody questions on LibriTTS-R and CREMA-D. Runs inside Modal (modal_build.py)."""
from __future__ import annotations

import glob
import hashlib
import json
import math
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import common as C
import phone_labels as L
import rel_labels as R

BUILDER = "sb_phone_pitch"
SR = 16000
WORK = Path("/vol/stageb_work/sb_phone_pitch")
FINAL = Path("/vol/stageb/sb_phone_pitch")
PHONE_RAW = Path("/vol/stageb/sb_phone/work/raw")  # read-only: apollo, minds, robocall, musan_*, rirs (sb_phone stage 0)
SBV = Path("/vol/stageb/sb_voice")  # read-only: work2 clips + features (LibriTTS-R, CREMA-D)
SBR = Path("/vol/stageb/sb_relative/work")  # read-only: plan.json + aug_XX (augmented targets, re-measured)
HVB = Path("/vol/stageb/sb_dialog/work/raw/hvb")  # read-only HarperValleyBank copy
CALLS = Path("/vol/stageb/sb_calls")  # read-only Kokoro call-outcome clips + items
ADI_ITEMS = Path("/vol/adi/items.jsonl")
EXP0_EVAL = Path("/vol/data/eval_items.jsonl")
EXP0_TRAIN = Path("/vol/data/train.jsonl")
MMSU_TRUTH = Path("/vol/bench_mmsu/truth.jsonl")  # only the "question" field, for the non-equality check

HVB_SHA = "0bd721e877c4a85d8c13ff837e68661ea6200a98"
LIC = {
    "hvb": f"CC BY 4.0 — HarperValleyBank, github.com/cricketclub/gridspace-stanford-harper-valley@{HVB_SHA} (repo LICENSE); read-only copy /vol/stageb/sb_dialog/work/raw/hvb",
    "apollo": "CC BY 4.0 — BrunoHays/apollo-telephony-en@b9d3a68a9faa550ae1f2069eb1eb7634cdc8966c (via /vol/stageb/sb_phone/work/raw)",
    "minds": "CC BY 4.0 — PolyAI/minds14@40ce77cb32a384e4d50a568e1ec39ac804019d33 (11 non-English locales only; via sb_phone raw)",
    "robocall": "Public domain audio (FTC Project Point of No Entry) — github wspr-ncsu/robocall-audio-dataset@5aa6f3bfa8563ce8c1c75ebf8a2271e6ff6b4272 (README: 'The data itself is in the public domain')",
    "libritts_r": "CC BY 4.0 — mythicinfinity/libritts_r@0d1718db351207d5a493135a62d5ed8cc4733830 (via sb_voice stage-1 /vol/stageb/sb_voice/work2)",
    "crema_d": "ODbL 1.0 + DbCL — CREMA-D, cfahlgren1/crema-d@b7d5beded69285eb5eb6a1c6894a5b0ad17cab92, github CheyneyComputerScience/CREMA-D@1658cd342dff90010aa843eaeebd53610a08b1dc; training actors only",
    "sb_calls": "Apache-2.0 — Kokoro-82M TTS output (hexgrad/Kokoro-82M, code + weights Apache-2.0) + synthetic tones, /vol/stageb/sb_calls (read-only)",
    "musan": "CC BY 4.0 corpus (OpenSLR 17); per-file licence filter: only CC BY / CC0 / public domain files (NC, ND, ShareAlike, Sampling+, unlicensed rejected)",
    "rirs": "Apache-2.0 — OpenSLR 28 RIRS_NOISES simulated_rirs only",
    "tone": "CC0 — generated 1 kHz separator tone",
}
SOURCE_REVS = {
    "hvb": {"repo": "github.com/cricketclub/gridspace-stanford-harper-valley", "revision": HVB_SHA, "licence": "CC BY 4.0",
            "excluded": "SLUE-HVB test conversations (asapp/slue-phase-2@e2989c55a53593a8e39b8f8ebdb47ccaccbe484a) + every HVB call in our evaluation set"},
    "apollo": {"repo": "BrunoHays/apollo-telephony-en", "revision": "b9d3a68a9faa550ae1f2069eb1eb7634cdc8966c", "licence": "cc-by-4.0"},
    "minds14": {"repo": "PolyAI/minds14", "revision": "40ce77cb32a384e4d50a568e1ec39ac804019d33", "licence": "cc-by-4.0",
                "locales": "cs-CZ de-DE es-ES fr-FR it-IT ko-KR nl-NL pl-PL pt-PT ru-RU zh-CN (no English: our evaluation set uses en-*)"},
    "robocall": {"repo": "github.com/wspr-ncsu/robocall-audio-dataset", "revision": "5aa6f3bfa8563ce8c1c75ebf8a2271e6ff6b4272", "licence": "audio public domain"},
    "libritts_r": {"repo": "mythicinfinity/libritts_r", "revision": "0d1718db351207d5a493135a62d5ed8cc4733830", "licence": "cc-by-4.0"},
    "crema_d": {"repo": "cfahlgren1/crema-d", "revision": "b7d5beded69285eb5eb6a1c6894a5b0ad17cab92", "github": "CheyneyComputerScience/CREMA-D@1658cd342dff90010aa843eaeebd53610a08b1dc", "licence": "ODbL-1.0 + DbCL-1.0"},
    "kokoro": {"repo": "hexgrad/Kokoro-82M", "revision": "f3ff3571791e39611d31c381e3a41a3af07b4987", "licence": "Apache-2.0 (code + weights)", "used_via": "/vol/stageb/sb_calls banks"},
    "musan": {"url": "https://www.openslr.org/17/ (musan.tar.gz)", "licence": "CC BY 4.0 + per-file filter"},
    "rirs": {"url": "https://www.openslr.org/28/ (rirs_noises.zip, simulated_rirs)", "licence": "Apache-2.0"},
    "tools": {"ffmpeg": "Debian ffmpeg 5.1 (G.711, GSM, Opus, G.726, Speex)", "rubberband": "used upstream by sb_relative for the augmented targets"},
}


def h01(s: str) -> float:
    return int(hashlib.md5(s.encode()).hexdigest()[:12], 16) / float(16 ** 12)


class NoShuffle(random.Random):
    def shuffle(self, x, *a, **k):  # keep ordered levels in natural order
        return None


NO_SHUFFLE = NoShuffle(0)


def read_jsonl(p) -> list[dict]:
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def bank_index(d: Path, name: str) -> dict:
    return json.loads((Path(d) / f"{name}.json").read_text())


# MUSAN licences
# copied from data/sb_phone/build.py, but ShareAlike (CC BY-SA) files are rejected too
_BAD_LIC = re.compile(r"non-?commercial|noncommercial|\bnc\b|no-?deriv|noderiv|\bnd\b|all rights reserved|sampling|share-?alike|by-sa|\bsa\b", re.I)
_GOOD_LIC = re.compile(r"cc by|cc0|attribution|public domain", re.I)
_LIC_LINE = re.compile(r"cc by|cc0|attribution|public domain|licen[cs]e|sampling|non-?commercial|creative commons", re.I)
_MUSAN_ID = re.compile(r"^\s*((?:music|noise|speech)-[a-z-]+-\d{4})\s*$")


def musan_licences(texts: dict) -> dict:
    out = {}
    for path, txt in texts.items():
        if not path.endswith("/LICENSE"):
            continue
        d = path.rsplit("/", 1)[0]
        blocks = re.split(r"\n\s*={10,}\s*\n?", txt)
        header, files, parsed = None, {}, []
        for b in blocks:
            lines = [l.strip() for l in b.splitlines() if l.strip()]
            ids = [m.group(1) for l in lines for m in [_MUSAN_ID.match(l)] if m]
            lic = " / ".join(l for l in lines if _LIC_LINE.search(l) and not _MUSAN_ID.match(l) and not l.startswith("http"))
            parsed.append((ids, lic))
            if lic and not ids and header is None:
                header = lic
        for ids, lic in parsed:
            for i in ids:
                files[i] = lic or header or ""
        any_ids = any(ids for ids, _ in parsed)
        out[d] = {"dir_licence": None if any_ids else " ".join(txt.split())[:300], "files": files}
    return out


def licence_ok(lic: str) -> bool:
    """Permissive only: CC BY / CC0 / public domain; NC, ND, ShareAlike, Sampling+ and missing licences rejected."""
    return bool(lic) and bool(_GOOD_LIC.search(lic)) and not _BAD_LIC.search(lic)


def musan_allowed(texts: dict, metas: dict) -> tuple[dict, dict]:
    lic = musan_licences(texts)
    allowed, rep, stems = {}, Counter(), {}
    for bank_meta in metas.values():
        for k, m in bank_meta.items():
            stems[k] = "musan/" + m["path"].split("musan/", 1)[-1].rsplit("/", 1)[0]
    for d, info in lic.items():
        if info["dir_licence"] is not None:
            ok = licence_ok(info["dir_licence"])
            for k, dd in stems.items():
                if dd == d:
                    rep[(d, "ok" if ok else "rejected", info["dir_licence"][:50])] += 1
                    if ok:
                        allowed[k] = info["dir_licence"][:120]
        for k, l in info["files"].items():
            ok = licence_ok(l)
            rep[(d, "ok" if ok else "rejected", l[:50])] += 1
            if ok:
                allowed[k] = l[:120]
    return allowed, {" | ".join(k): v for k, v in rep.most_common()}


# Evaluation-set ids
def adi_ids() -> dict:
    """Ids in our evaluation items, for exclusion and leakage checks: HVB calls and speakers, CREMA-D actors, meta values."""
    out = {"hvb_calls": set(), "hvb_speakers": set(), "crema_actors": set(), "meta_values": set(), "n": 0, "banks": set()}
    if not ADI_ITEMS.exists():
        return out
    for line in open(ADI_ITEMS):
        if not line.strip():
            continue
        it = json.loads(line)
        out["n"] += 1
        m = it.get("meta") or {}
        a = it.get("audio") or {}
        if a.get("bank"):
            out["banks"].add(a["bank"])
        src = str(it.get("source") or it.get("benchmark") or "").lower()
        blob = json.dumps(m)
        if "hvb" in src or "harper" in src or "hvb" in str(a.get("bank", "")).lower():
            if m.get("call") is not None:
                out["hvb_calls"].add(str(m["call"]))
            if m.get("speaker") is not None:
                out["hvb_speakers"].add(str(m["speaker"]))
            for k in ("conversation", "conv", "sid", "issue_id"):
                if m.get(k) is not None:
                    out["hvb_calls"].add(str(m[k]))
        if "crema" in src or "crema" in str(a.get("bank", "")).lower():
            for k in ("actor", "speaker"):
                if m.get(k) is not None:
                    out["crema_actors"].add(str(m[k]).split(":")[-1])
            mm = re.search(r"\b(1\d{3})_[A-Z]{3}_[A-Z]{3}_[A-Z]{2}\b", blob + " " + str(a.get("key", "")))
            if mm:
                out["crema_actors"].add(mm.group(1))
        for v in m.values():
            if isinstance(v, (str, int)) and len(str(v)) >= 4:
                out["meta_values"].add(str(v))
    return out


def exp0_crema_actors() -> tuple[set, set]:
    ev, tr = set(), set()
    for it in read_jsonl(EXP0_EVAL):
        if it.get("audio") and it["audio"].get("bank") == "crema_d":
            ev.add(str(it["meta"].get("actor")))
    for it in read_jsonl(EXP0_TRAIN):
        if it.get("source") == "crema_d":
            tr.add(str(it["meta"]["actor"]))
    return ev, tr


# Stage 0: HVB segments
def hvb_speaker_split(conv_spk: dict, frac: float):
    """Copied from data/sb_dialog/build.py: CAL speaker set grown by speaker pairs; conversations with both speakers
    in CAL -> CAL, neither -> train, exactly one -> dropped."""
    import itertools

    spk_all = sorted({s for v in conv_spk.values() for s in v})
    n = len(conv_spk)

    def stats(S):
        cal = sum(1 for v in conv_spk.values() if v <= S)
        drop = sum(1 for v in conv_spk.values() if (v & S) and not v <= S)
        return cal, drop

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
    split = {c: ("CAL" if v <= S else ("train" if not (v & S) else None)) for c, v in conv_spk.items()}
    return split, S


_NOISE_ONLY = re.compile(r"^(\s*\[[^\]]*\]\s*)*$")


def prep_hvb(max_groups_per_role: int = 1) -> dict:
    """HVB per-role speech runs (1-3 consecutive same-role segments, 2-15 s of speech, transcript kept) cut from the
    8 kHz channel wavs -> 16 kHz bank WORK/raw/hvb + hvb_meta.json. Excludes SLUE-HVB test and evaluation-set calls."""
    import soundfile as sf

    t0 = time.time()
    adi = adi_ids()
    slue_test = set(json.loads((HVB / "slue_ids.json").read_text())["test"])
    sids = sorted(p.stem for p in (HVB / "transcript").glob("*.json"))
    conv_spk, excl = {}, Counter()
    for sid in sids:
        if sid in slue_test or sid in adi["hvb_calls"]:
            excl["slue_test_or_adi_call"] += 1
            continue
        md = json.loads((HVB / "metadata" / f"{sid}.json").read_text())
        conv_spk[sid] = {f"hvb_spk:{md['agent']['speaker_id']}", f"hvb_spk:{md['caller']['speaker_id']}"}
    split, cal_spk = hvb_speaker_split(conv_spk, 0.03)
    clips, meta = {}, {}
    rng = random.Random(f"{C.SEED}:{BUILDER}:hvb_prep")
    for sid in sorted(conv_spk):
        sp = split[sid]
        if sp is None:
            excl["split_dropped"] += 1
            continue
        md = json.loads((HVB / "metadata" / f"{sid}.json").read_text())
        segs = json.loads((HVB / "transcript" / f"{sid}.json").read_text())
        by_role = defaultdict(list)
        for g in segs:
            role = g.get("speaker_role")
            if role not in ("agent", "caller"):
                continue
            text = (g.get("human_transcript") or g.get("transcript") or "").strip()
            if not text or _NOISE_ONLY.match(text) or "[" in text:
                continue
            s = g["start_ms"] / 1000.0
            e = s + g["duration_ms"] / 1000.0
            if e - s < 0.5:
                continue
            by_role[role].append((s, e, text))
        for role in ("caller", "agent"):
            lst = sorted(by_role[role])
            if not lst:
                continue
            wav = HVB / "audio" / role / f"{sid}.wav"
            if not wav.exists():
                continue
            groups, i = [], 0
            while i < len(lst):
                m = rng.choice([1, 2, 2, 3])
                g = lst[i: i + m]
                i += m
                tot = sum(e - s for s, e, _ in g)
                if 2.0 <= tot <= 15.0:
                    groups.append(g)
            rng.shuffle(groups)
            if not groups:
                continue
            info = sf.info(str(wav))
            wsr, wlen = int(info.samplerate), int(info.frames)
            for gi, g in enumerate(groups[:max_groups_per_role]):
                keys = []
                for si, (s, e, text) in enumerate(g):
                    a, b = max(0, int((s - 0.08) * wsr)), min(wlen, int((e + 0.08) * wsr))
                    if b - a < int(0.4 * wsr):
                        continue
                    xs, _ = sf.read(str(wav), start=a, stop=b, dtype="float32", always_2d=True)
                    xs = xs.mean(axis=1)
                    seg = C.resample(xs, wsr, SR) if wsr != SR else xs.copy()
                    k = f"hvb/{sid}/{role}/{gi}/{si}"
                    clips[k] = seg.astype(np.float32)
                    keys.append(k)
                if not keys:
                    continue
                spk = f"hvb_spk:{md[role]['speaker_id']}"
                meta[f"hvb/{sid}/{role}/{gi}"] = {"keys": keys, "conv": sid, "role": role, "speaker": spk, "split": sp,
                                                  "text": " ".join(t for _, _, t in g), "seg_times": [[round(s, 3), round(e, 3)] for s, e, _ in g]}
    (WORK / "raw").mkdir(parents=True, exist_ok=True)
    C.write_audio_bank(WORK / "raw", "hvb", clips)
    (WORK / "raw" / "hvb_meta.json").write_text(json.dumps({"groups": meta, "cal_speakers": sorted(cal_spk), "excluded": dict(excl),
                                                            "n_conv_usable": len(conv_spk), "adi_hvb_calls": len(adi["hvb_calls"]),
                                                            "adi_hvb_speakers": sorted(adi["hvb_speakers"])[:50]}))
    return {"groups": len(meta), "segments": len(clips), "hours": round(sum(len(v) for v in clips.values()) / SR / 3600, 2),
            "excluded": dict(excl), "cal_speakers": sorted(cal_spk), "split": dict(Counter(m["split"] for m in meta.values())),
            "adi_hvb_calls": len(adi["hvb_calls"]), "adi_hvb_speakers_n": len(adi["hvb_speakers"]), "seconds": round(time.time() - t0)}


# Prosody records
def load_records() -> tuple[list[dict], dict]:
    """sb_voice stage-1 LibriTTS-R + CREMA-D records with features. CREMA-D: training actors only, never evaluation
    actors; sb_voice CAL speakers are skipped. CREMA-D rate uses hand syllable counts."""
    sbv_cal = set(json.loads((SBV / "cal_speakers.json").read_text()))
    ev, tr = exp0_crema_actors()
    adi = adi_ids()
    recs, skipped = [], Counter()
    for p in sorted(glob.glob(str(SBV / "work2" / "*.jsonl"))):
        shard = Path(p).stem
        if not shard.startswith(("libritts", "crema")):
            continue
        for r in read_jsonl(p):
            spk = r["speaker"]
            if spk in sbv_cal:
                skipped["sb_voice_cal_speaker"] += 1
                continue
            if r["source"] == "crema_d":
                actor = spk.split(":")[1]
                if actor in ev or actor not in tr or actor in adi["crema_actors"]:
                    skipped["crema_not_allowed_actor"] += 1
                    continue
                f = r.get("feat") or {}
                if f and "error" not in f:
                    syl = R.crema_syllables(r["source_id"])
                    f["syll"] = syl
                    f["rate"] = syl / f["span_s"] if f.get("span_s") and f["span_s"] >= 0.5 else None
            f = r.get("feat")
            if not f or "error" in f:
                skipped["no_features"] += 1
                continue
            r["shard"] = shard
            recs.append(r)
    info = {"skipped": dict(skipped), "kept": dict(Counter(r["source"] for r in recs)),
            "n_speakers": dict(Counter(s.split(":")[0] for s in {r["speaker"] for r in recs})),
            "exp0_eval_crema_actors": len(ev), "adi_crema_actors": sorted(adi["crema_actors"])}
    return recs, info


def cal_speakers() -> set:
    """CAL speakers (LibriTTS-R + CREMA-D): sb_relative's CAL set (gender-balanced, from sb_voice train speakers)."""
    pl = json.loads((SBR / "plan.json").read_text())
    return {s for s in pl["cal_speakers"] if s.startswith(("libritts_r:", "crema_d:"))}


# Part A: phone plan
BANK_DIRS = {"raw": str(PHONE_RAW), "work_raw": str(WORK / "raw"), "sbv_work2": str(SBV / "work2"),
             "sb_calls": str(CALLS / "audio"), "sbr_work": str(SBR)}

LIVE_CLASSES = {"live_person"}
AUTO_CLASSES = {"personal_voicemail", "carrier_voicemail", "ivr", "call_screening", "business_ivr", "screening",
                "personal_vm", "carrier_vm"}


def live_views(rng: random.Random, is_live: bool, n: int) -> list:
    views = rng.sample(["live_noul", "auto_noul", "live_choice"], n)
    return [(v, is_live) for v in views]


PHONE_SIZES = {"hvb": 1500, "minds": 880, "apollo": 950, "robocall_calls": 360, "studio": 1750, "calls": 480}


class PhonePlanner:
    """Adapted from data/sb_phone/build.py Planner: Vystadial -> HVB, VCTK/LibriTTS-R(sb_voice banks) -> LibriTTS-R
    (sb_voice work2 clean stage-1 clips)."""

    ROBUST_BY_CORPUS = {"hvb": ("breakup", "music", "talker"), "minds14": ("breakup", "music", "talker"),
                        "apollo": ("music",), "robocall": ("breakup",)}

    def __init__(self, scale: float, recs: list[dict], cal_spk: set):
        self.scale = scale
        self.clips: list[dict] = []
        self.cal_groups = {}
        S = {}
        for name in ("apollo", "minds", "robocall", "musan_noise", "musan_music", "musan_speech", "rirs"):
            S[name] = {"meta": json.loads((PHONE_RAW / f"{name}_meta.json").read_text()), "index": bank_index(PHONE_RAW, name)}
        S["musan_texts"] = json.loads((PHONE_RAW / "musan_texts.json").read_text())
        S["hvb"] = json.loads((WORK / "raw" / "hvb_meta.json").read_text())
        S["hvb_index"] = bank_index(WORK / "raw", "hvb")
        S["calls_items"] = read_jsonl(CALLS / "items_train.jsonl") + read_jsonl(CALLS / "items_cal.jsonl")
        S["calls_index"] = {b: bank_index(CALLS / "audio", b) for b in [f"sb_calls_train_{i}" for i in range(4)] + ["sb_calls_cal"]}
        self.S = S
        self.recs = [r for r in recs if r["source"] == "libritts_r"]
        self.cal_spk = cal_spk
        allowed, self.musan_report = musan_allowed(S["musan_texts"], {b: S[b]["meta"] for b in ("musan_noise", "musan_music", "musan_speech")})
        self.bg = {}
        for kind, bank in (("noise", "musan_noise"), ("music", "musan_music"), ("talker", "musan_speech")):
            keys = sorted(k for k, (s, n) in S[bank]["index"].items() if n >= 3 * SR and k in allowed)
            self.bg[kind] = {"train": [k for k in keys if h01("bg:" + k) >= 0.15], "CAL": [k for k in keys if h01("bg:" + k) < 0.15], "bank": bank}
        self.musan_used_licences = {kind: dict(Counter(allowed[k][:40] for k in v["train"] + v["CAL"])) for kind, v in self.bg.items()}
        rm = S["rirs"]["meta"]
        far = sorted(k for k, m in rm.items() if m["drr_db"] is not None and m["drr_db"] <= -2.0 and (m.get("rt60_s") or 0) >= 0.3)
        self.rir = {"train": [k for k in far if h01("room:" + rm[k]["room"]) >= 0.15], "CAL": [k for k in far if h01("room:" + rm[k]["room"]) < 0.15]}

    def n(self, x: int) -> int:
        return max(4, int(round(x * self.scale)))

    def recipe(self, split: str, base_clean: bool, rng: random.Random, scenario: str | None = None):
        w = L.SCENARIO_W_CLEAN if base_clean else L.SCENARIO_W_REAL
        if scenario is None:
            scenario = rng.choices(list(w), weights=list(w.values()))[0]
        r = L.sample_recipe("clean" if scenario == "orig" else scenario, rng, p_codec_none=0.3 if base_clean else 0.5)
        r["scenario"] = scenario
        bg = {}
        for kind in ("noise", "music", "talker"):
            if r.get(kind):
                bg[kind] = [self.bg[kind]["bank"], rng.choice(self.bg[kind][split])]
        if r.get("rir"):
            k = rng.choice(self.rir[split])
            bg["rir"] = ["rirs", k]
            m = self.S["rirs"]["meta"][k]
            r["rir_drr_db"], r["rir_rt60_s"], r["rir_room"] = m["drr_db"], m["rt60_s"], m["room"]
        return r, bg

    def add(self, *, key, split, corpus, segs, recipe, bg, base_clean, questions, meta, crop=None):
        self.clips.append({"key": key, "split": split, "corpus": corpus, "segs": segs, "recipe": recipe, "bg": bg,
                           "base_clean": base_clean, "questions": questions, "meta": meta, "crop": crop})

    def robust_q(self, recipe: dict, rng: random.Random, n: int = 1, corpus: str | None = None) -> list:
        app = L.applicable(recipe, base_clean=False)
        if corpus in self.ROBUST_BY_CORPUS:
            app = {k: v for k, v in app.items() if k in self.ROBUST_BY_CORPUS[corpus]}
        rel = {"drop_mod": "breakup", "drop_sev": "breakup", "music": "music", "talker": "talker"}.get(recipe["scenario"])
        out = []
        if rel and rel in app and rng.random() < 0.7:
            out.append((rel, app[rel]))
        others = [k for k in ("breakup", "music", "talker") if k in app and k not in [q for q, _ in out]]
        rng.shuffle(others)
        for k in others:
            if len(out) >= n:
                break
            out.append((k, app[k]))
        return out[:n]

    # HVB (replaces Vystadial): real 8 kHz call audio
    def plan_hvb(self, n_clips: int):
        rng = random.Random(f"{C.SEED}:{BUILDER}:plan:hvb")
        groups = self.S["hvb"]["groups"]
        self.cal_groups["hvb_cal_speakers"] = self.S["hvb"]["cal_speakers"]
        keys = sorted(groups, key=lambda k: h01("hvbp:" + k))
        cal = [k for k in keys if groups[k]["split"] == "CAL"]
        tr = [k for k in keys if groups[k]["split"] == "train"]
        n_cal = max(2, int(0.03 * n_clips))
        for split, lst in (("CAL", cal[:n_cal]), ("train", tr[: n_clips - min(n_cal, len(cal))])):
            for gk in lst:
                g = groups[gk]
                r, bg = self.recipe(split, False, rng)
                qs = []
                if rng.random() < 0.55:
                    qs += live_views(rng, True, 1)
                num = L.number_label(g["text"])
                if num is not None and (num or rng.random() < 0.5):
                    qs.append(("number", num))
                if rng.random() < 0.15:
                    qs.append(("lang", "en"))
                qs += self.robust_q(r, rng, corpus="hvb")
                self.add(key=f"hvb_{split}_{gk[4:].replace('/', '_')}", split=split, corpus="hvb",
                         segs=[["work_raw", "hvb", k] for k in g["keys"]], recipe=r, bg=bg, base_clean=False, questions=qs[:4],
                         meta={"speaker": g["speaker"], "source_id": f"hvb:{g['conv']}:{g['role']}", "hvb_call": g["conv"],
                               "role": g["role"], "seg_times": g["seg_times"], "transcript": g["text"], "licence": LIC["hvb"]})

    def plan_minds(self, n_clips: int):
        rng = random.Random(f"{C.SEED}:{BUILDER}:plan:minds")
        meta = self.S["minds"]["meta"]
        by_loc = defaultdict(list)
        for k, m in meta.items():
            by_loc[m["locale"]].append(k)
        assert not any(l.startswith("en") for l in by_loc), "MInDS-14 English locales must never be used"
        per = n_clips // len(by_loc)
        for loc in sorted(by_loc):
            ks = sorted(by_loc[loc], key=lambda k: h01("v21minds:" + k))[:per]
            for k in ks:
                split = "CAL" if h01("minds_cal:" + meta[k]["path"]) < 0.03 else "train"
                r, bg = self.recipe(split, False, rng)
                qs = [("lang", loc)]
                if rng.random() < 0.3:
                    qs += live_views(rng, True, 1)
                qs += self.robust_q(r, rng, n=2, corpus="minds14")
                self.add(key=f"minds_{split}_{k[6:]}", split=split, corpus="minds14", segs=[["raw", "minds", k]], recipe=r,
                         bg=bg, base_clean=False, questions=qs,
                         meta={"speaker": f"minds14:{meta[k]['path']}", "source_id": meta[k]["path"], "locale": loc,
                               "transcript": meta[k]["text"], "licence": LIC["minds"]})

    def plan_apollo(self, n_clips: int):
        rng = random.Random(f"{C.SEED}:{BUILDER}:plan:apollo")
        meta = self.S["apollo"]["meta"]
        tr = sorted([k for k, m in meta.items() if m["split"] == "train"], key=lambda k: h01("v21ap:" + k))
        dv = sorted([k for k, m in meta.items() if m["split"] == "dev"], key=lambda k: h01("v21ap:" + k))
        n_cal = max(2, int(0.03 * n_clips))
        for split, ks in (("CAL", dv[:n_cal]), ("train", tr[: n_clips - n_cal])):
            for k in ks:
                text = meta[k]["text"]
                r, bg = self.recipe(split, False, rng)
                qs = []
                num = L.number_label(text)
                if num is not None:
                    qs.append(("number", num))
                if rng.random() < 0.10:
                    qs.append(("lang", "en"))
                if rng.random() < 0.25:
                    qs += live_views(rng, True, 1)
                qs += self.robust_q(r, rng, corpus="apollo")
                self.add(key=f"apollo_{split}_{k}", split=split, corpus="apollo", segs=[["raw", "apollo", k]], recipe=r,
                         bg=bg, base_clean=False, questions=qs[:4],
                         meta={"speaker": f"apollo:{meta[k]['audio_id']}", "source_id": meta[k]["audio_id"],
                               "apollo_split": meta[k]["split"], "transcript": text, "licence": LIC["apollo"]})

    def plan_robocall(self, n_calls: int, max_windows: int = 2):
        rng = random.Random(f"{C.SEED}:{BUILDER}:plan:robo")
        meta = self.S["robocall"]["meta"]
        idx = self.S["robocall"]["index"]
        cases = Counter(m["case"] for m in meta.values())
        order = sorted(cases, key=lambda c: h01("robo_case:" + c))  # same CAL cases as sb_phone
        cal_cases, tot = set(), 0
        for c in order:
            if tot + cases[c] <= 0.05 * len(meta) and cases[c] < 0.04 * len(meta):
                cal_cases.add(c)
                tot += cases[c]
            if tot >= 0.025 * len(meta):
                break
        self.cal_groups["robocall_cases"] = sorted(cal_cases)
        keys = sorted(meta, key=lambda k: h01("v21robo:" + k))
        cal_keys = [k for k in keys if meta[k]["case"] in cal_cases][: max(2, int(0.04 * n_calls))]
        tr_keys = [k for k in keys if meta[k]["case"] not in cal_cases][:n_calls]
        for k in cal_keys + tr_keys:
            m = meta[k]
            split = "CAL" if m["case"] in cal_cases else "train"
            dur = idx[k][1] / SR
            w1 = min(dur, rng.uniform(8.0, 25.0))
            wins = [(0.0, w1)]
            rest = dur - w1 - 0.5
            if max_windows > 1 and rest >= 6.0:
                w2 = min(rest, rng.uniform(8.0, 25.0))
                wins.append((round(w1 + 0.5 + rng.uniform(0, max(0.0, rest - w2)), 2), w2))
            for wi, (st, wl) in enumerate(wins):
                r, bg = self.recipe(split, False, rng)
                qs = live_views(rng, False, 2) + self.robust_q(r, rng, n=2, corpus="robocall")
                self.add(key=f"robo_{split}_{k[5:]}_w{wi}", split=split, corpus="robocall", segs=[["raw", "robocall", k]],
                         crop=[st, wl], recipe=r, bg=bg, base_clean=False, questions=qs,
                         meta={"speaker": f"robocall_case:{m['case']}", "source_id": m["file"], "case": m["case"],
                               "window_s": [round(st, 2), round(wl, 2)], "licence": LIC["robocall"]})

    # Clean studio speech (LibriTTS-R), labels by construction
    def plan_studio(self, n_clips: int):
        rng = random.Random(f"{C.SEED}:{BUILDER}:plan:studio")
        by_spk = defaultdict(list)
        for r in sorted(self.recs, key=lambda r: r["key"]):
            if 0.8 <= r["dur"] <= 15.0:
                by_spk[r["speaker"]].append(r)
        groups = {"CAL": [], "train": []}
        for spk in sorted(by_spk):
            split = "CAL" if spk in self.cal_spk else "train"
            rs = sorted(by_spk[spk], key=lambda r: h01("v21st:" + r["key"]))
            i = 0
            while i < len(rs):
                m = rng.choice([1, 2, 2, 3])
                g = rs[i: i + m]
                i += m
                while i < len(rs) and sum(x["dur"] for x in g) < 3.0:
                    g.append(rs[i])
                    i += 1
                tot = sum(x["dur"] for x in g)
                if 2.0 <= tot <= 20.0:
                    groups[split].append((spk, g))
            groups[split].sort(key=lambda g: h01("v21stg:" + g[1][0]["key"]))
        n_cal = max(4, int(0.03 * n_clips))
        sel = {"CAL": groups["CAL"][:n_cal], "train": groups["train"][: n_clips - n_cal]}
        for split, lst in sel.items():
            for spk, g in lst:
                r, bg = self.recipe(split, True, rng)
                qs = self.studio_questions(r, rng)
                self.add(key=f"studio_{split}_{g[0]['key'].replace('/', '_')}", split=split, corpus="libritts_r",
                         segs=[["sbv_work2", x["shard"], x["key"]] for x in g], recipe=r, bg=bg, base_clean=True, questions=qs,
                         meta={"speaker": spk, "source_id": [x["source_id"] for x in g], "licence": LIC["libritts_r"]})

    def studio_questions(self, r: dict, rng: random.Random, n: int = 3) -> list:
        app = L.applicable(r, base_clean=True)
        sc = r["scenario"]
        rel = {"noise_mod": "noise", "noise_sev": "noise", "drop_mod": "breakup", "drop_sev": "breakup", "music": "music",
               "talker": "talker", "far": "far", "clip_sev": "poor", "combo_sev": "poor", "lowvol": "quality",
               "clip_mod": "quality", "clean": None}.get(sc)
        out = []
        if "quality" in app and rng.random() < 0.55:
            out.append(("quality", app["quality"]))
        if rel and rel in app and rel not in [q for q, _ in out]:
            out.append((rel, app[rel]))
        if "poor" in app and not any(q in ("poor", "voice_clear") for q, _ in out) and rng.random() < 0.8:
            k = rng.choice(["poor", "voice_clear"])
            out.append((k, app[k]))
        others = [k for k in ("poor", "voice_clear", "noise", "breakup", "music", "talker", "far") if k in app and k not in [q for q, _ in out]]
        rng.shuffle(others)
        for k in others:
            if len(out) >= n:
                break
            out.append((k, app[k]))
        return out[:n]

    # sb_calls Kokoro clips, degraded, same call semantics
    def plan_calls(self, n_clips: int):
        rng = random.Random(f"{C.SEED}:{BUILDER}:plan:calls")
        by_clip = defaultdict(list)
        for it in self.S["calls_items"]:
            by_clip[(it["audio"]["bank"], it["audio"]["key"])].append(it)
        cidx = self.S["calls_index"]
        keys = sorted((bk for bk in by_clip if cidx[bk[0]][bk[1]][1] >= 2 * SR), key=lambda bk: h01("v21calls:" + bk[1]))
        cal = [bk for bk in keys if by_clip[bk][0]["set"] == "CAL"]
        tr = [bk for bk in keys if by_clip[bk][0]["set"] == "train"]
        n_cal = max(4, int(0.03 * n_clips))
        for split, lst in (("CAL", cal[:n_cal]), ("train", tr[: n_clips - n_cal])):
            for bk in lst:
                its = by_clip[bk]
                m0 = its[0]["meta"]
                base_clean = m0.get("noise") in (None, "none")
                r, bg = self.recipe(split, base_clean, rng)
                its_sorted = sorted(its, key=lambda i: h01("ci:" + i["id"]))
                qs = [("calls_item", i["id"]) for i in its_sorted[: (2 if rng.random() < 0.4 else 1)]]
                cls = m0.get("class")
                if cls in LIVE_CLASSES and rng.random() < 0.3:
                    qs += live_views(rng, True, 1)
                elif cls in AUTO_CLASSES and rng.random() < 0.5:
                    qs += live_views(rng, False, 1)
                if base_clean:
                    qs += self.studio_questions(r, rng, n=4 - len(qs))
                else:
                    qs += self.robust_q(r, rng, n=1)
                self.add(key=f"calls_{split}_{bk[1]}", split=split, corpus="sb_calls", segs=[["sb_calls", bk[0], bk[1]]],
                         recipe=r, bg=bg, base_clean=base_clean, questions=qs[:4],
                         meta={"speaker": f"kokoro:{m0.get('voice')}", "source_id": f"{bk[0]}/{bk[1]}", "call_class": cls,
                               "licence": LIC["sb_calls"]})

    def build(self) -> list[dict]:
        s = PHONE_SIZES
        self.plan_hvb(self.n(s["hvb"]))
        self.plan_minds(self.n(s["minds"]))
        self.plan_apollo(self.n(s["apollo"]))
        self.plan_robocall(self.n(s["robocall_calls"]))
        self.plan_studio(self.n(s["studio"]))
        self.plan_calls(self.n(s["calls"]))
        return self.clips


# Rendering (workers)
_BANKS: dict = {}


def _bank(where: str, name: str):
    k = (where, name)
    if k not in _BANKS:
        _BANKS[k] = C.AudioBank(BANK_DIRS[where], name)
    return _BANKS[k]


def phone_base_audio(spec: dict, rng: random.Random) -> np.ndarray:
    xs = []
    for where, bank, key in spec["segs"]:
        x = _bank(where, bank).get(key)
        if spec.get("crop"):
            st, wl = spec["crop"]
            x = x[int(st * SR): int((st + wl) * SR)]
        if xs:
            xs.append(np.zeros(int(rng.uniform(0.3, 0.8) * SR), np.float32))
        xs.append(x)
    return np.concatenate(xs).astype(np.float32)[: 30 * SR]


PROS_CODECS = [("mulaw", None), ("alaw", None), ("gsm", None), ("opus", 10), ("opus", 12), ("g726", 32)]


def render_one(spec: dict):
    import fx

    rng = random.Random(f"{C.SEED}:{BUILDER}:render:{spec['key']}")
    try:
        if spec["kind"] == "phone":
            x = phone_base_audio(spec, rng)
            bgs = {kind: _bank("raw", bank).get(key) for kind, (bank, key) in spec["bg"].items()}
            y, facts = fx.render(x, spec["recipe"], bgs, rng)
        else:
            parts = [_bank(w, b).get(k) for w, b, k in spec["parts"]]
            if spec["kind"] == "two_part":
                y, facts = R.two_part(parts[0], parts[1])
            else:
                y = np.asarray(parts[0], np.float32)
                pk = float(np.max(np.abs(y)) + 1e-9)
                if pk > 0.99:
                    y = (y * 0.99 / pk).astype(np.float32)
                facts = {"scale": round(min(1.0, 0.99 / pk), 5)}
            if spec.get("fx"):
                # baked-in phone line on the whole recording (relative levels / pitch / rate between parts survive)
                pk0 = float(np.max(np.abs(y)) + 1e-9)
                z = fx.bandlimit(y)
                z = fx.ffmpeg_codec(z, spec["fx"]["codec"], spec["fx"].get("bitrate_k"))
                pk1 = float(np.max(np.abs(z)) + 1e-9)
                y = (z * min(pk0 / pk1, 0.99 / pk1)).astype(np.float32)
                facts["fx"] = spec["fx"]
        if not np.isfinite(y).all():
            raise ValueError("non-finite audio")
        pk = float(np.max(np.abs(y)) + 1e-9)
        if pk > 0.99:
            y = (y * (0.99 / pk)).astype(np.float32)
        facts["duration_s"] = round(len(y) / SR, 3)
        return spec["key"], y.astype(np.float32), facts
    except Exception as e:  # recorded; the clip is dropped
        return spec["key"], None, {"error": repr(e)[:300]}


def render_shard(bank_name: str, specs: list[dict], out_audio: str, n_proc: int = 4) -> dict:
    from multiprocessing import get_context

    t0 = time.time()
    fdir = Path(out_audio).parent / "facts"
    if (Path(out_audio) / f"{bank_name}.npy").exists() and (fdir / f"{bank_name}.json").exists():
        return {"bank": bank_name, "skipped": "exists"}
    clips, facts = {}, {}
    pool = get_context("spawn").Pool(n_proc)
    try:
        pending = [(s["key"], pool.apply_async(render_one, (s,))) for s in specs]
        deadline = time.time() + 1800
        for key, ar in pending:
            try:
                k, y, f = ar.get(timeout=max(5.0, deadline - time.time()))
            except Exception as e:
                k, y, f = key, None, {"error": f"timeout/failed: {type(e).__name__}"}
            facts[k] = f
            if y is not None:
                clips[k] = y
    finally:
        pool.terminate()
    C.write_audio_bank(out_audio, bank_name, clips)
    fdir.mkdir(parents=True, exist_ok=True)
    (fdir / f"{bank_name}.json").write_text(json.dumps(facts))
    return {"bank": bank_name, "n": len(clips), "errors": sum(1 for f in facts.values() if "error" in f),
            "seconds": round(time.time() - t0, 1), "audio_s": round(sum(len(v) for v in clips.values()) / SR, 1)}


# Part A: phone items
def _noul(items, rng, *, iid, split, source, question, yes, audio, channel, meta):
    items.append(C.make_item(item_id=iid, set_name=split, source=source, question=question, options=["yes", "no"],
                             target=[1.0, 0.0] if yes else [0.0, 1.0], gold=0 if yes else 1, rng=rng, audio=audio, channel=channel,
                             meta={**meta, "qtype": "noul", "answer": "yes" if yes else "no", "answer_class": "yes" if yes else "no"}))


PHONE_RULES = {
    "breakup": "yes = we zeroed 20-60 ms chunks covering >= 4 % of the speech time (packet loss; realised >= 3 %); no = no dropouts added",
    "music": "yes = MUSAN music (per-file CC BY / CC0 / PD) mixed in at 0-12 dB speech-to-music ratio; no = no music added",
    "talker": "yes = MUSAN speech (another person) mixed in at 6-14 dB below the main speaker; no = none added",
    "noise": "yes = MUSAN noise mixed in (SNR -3..20 dB) into clean studio/TTS speech; no = clean base and no noise/music/talker added",
    "far": "yes = convolved with a simulated room impulse response (OpenSLR 28) with DRR <= -2 dB and RT60 >= 0.3 s; no = no RIR (close-talk)",
    "poor": "yes = quality level 'very poor' by construction; no = 'clear' (no added impairment; normal phone codecs allowed)",
    "voice_clear": "yes = quality level 'clear' by construction; no = 'very poor'",
}


def make_phone_items(clips: list[dict], facts: dict, bank_of: dict, calls_by_id: dict) -> list[dict]:
    items = []
    lang_all = list(L.LANG_NAMES)
    P = f"{BUILDER}_phone"
    for c in clips:
        f = facts.get(c["key"], {})
        if "error" in f or c["key"] not in bank_of:
            continue
        rng = random.Random(f"{C.SEED}:{BUILDER}:items:{c['key']}")
        audio = {"bank": bank_of[c["key"]], "key": c["key"]}
        r = c["recipe"]
        base_meta = {"clip": c["key"], "corpus": c["corpus"], **c["meta"], "recipe": dict(r), "render": f,
                     "background": c["bg"], "label_rule": None, "part": "phone"}
        used = 0
        for qi, (qname, ans) in enumerate(c["questions"]):
            if used >= 4:
                break
            channel = "phone" if rng.random() < 0.6 else "clean"
            iid = f"{BUILDER}/{c['split']}/{c['key']}/{qi}_{qname}"
            meta = {**base_meta, "qname": qname}
            split = c["split"]
            if qname == "breakup":
                if ans and f.get("lost_frac", 0.0) < 0.03:
                    continue
                meta["label_rule"] = PHONE_RULES["breakup"]
                _noul(items, rng, iid=iid, split=split, source=f"{P}_breakup", question=L.qtext("breakup_noul", rng),
                      yes=ans, audio=audio, channel=channel, meta=meta)
            elif qname in ("music", "talker", "noise", "far", "poor", "voice_clear"):
                meta["label_rule"] = PHONE_RULES[qname]
                _noul(items, rng, iid=iid, split=split, source=f"{P}_{qname}", question=L.qtext(f"{qname}_noul", rng),
                      yes=ans, audio=audio, channel=channel, meta=meta)
            elif qname == "quality":
                meta["label_rule"] = ("by construction: clear = no added impairment; some problems = exactly one moderate impairment "
                                      "(noise SNR 12-20 dB / 4-9 % dropouts / 2-5 % clipping / quiet -34..-42 dBFS / far-field echo); "
                                      "very poor = a severe one (SNR <= 4 dB / 15-30 % dropouts / 10-25 % clipping)")
                opt_map = L.QUALITY_OPTIONS if rng.random() < 0.6 else L.QUALITY_OPTIONS_ALT
                opts = [opt_map[x] for x in L.LEVELS]
                gold = L.LEVELS.index(ans)
                tgt = [1.0 if i == gold else 0.0 for i in range(3)]
                is_score = rng.random() < 0.6
                items.append(C.make_item(item_id=iid, set_name=split, source=f"{P}_quality",
                                         question=L.qtext("quality_score" if is_score else "quality_choice", rng), options=opts,
                                         target=tgt, gold=gold, rng=NO_SHUFFLE if is_score else rng, audio=audio, channel=channel,
                                         meta={**meta, "qtype": "score" if is_score else "choice", "answer": ans, "answer_class": ans,
                                               "levels": opts if is_score else None}))
            elif qname in ("live_noul", "auto_noul", "live_choice"):
                meta["label_rule"] = ("live = real human speech on a phone line (HarperValleyBank callers/agents, MInDS-14 callers, "
                                      "Apollo mission-control loops) or a TTS 'live person' clip from sb_calls; automated/recorded = "
                                      "FTC robocall recordings, sb_calls voicemail/IVR/screening greetings")
                is_live = bool(ans)
                if qname == "live_choice":
                    o = rng.choice(L.LIVE_OPTIONS)
                    gold = 0 if is_live else 1
                    items.append(C.make_item(item_id=iid, set_name=split, source=f"{P}_live", question=L.qtext("live_choice", rng),
                                             options=list(o), target=[1.0, 0.0] if is_live else [0.0, 1.0], gold=gold, rng=rng,
                                             audio=audio, channel=channel,
                                             meta={**meta, "qtype": "choice", "answer": o[gold], "answer_class": "live" if is_live else "automated"}))
                elif qname == "live_noul":
                    _noul(items, rng, iid=iid, split=split, source=f"{P}_live", question=L.qtext("live_noul", rng),
                          yes=is_live, audio=audio, channel=channel, meta=meta)
                else:
                    _noul(items, rng, iid=iid, split=split, source=f"{P}_live", question=L.qtext("auto_noul", rng),
                          yes=not is_live, audio=audio, channel=channel, meta=meta)
            elif qname == "lang":
                meta["label_rule"] = "language = MInDS-14 locale of the recording (English = HarperValleyBank / Apollo English corpora)"
                gold_lang = ans
                if rng.random() < 0.6:
                    k = len(lang_all) if rng.random() < 0.4 else rng.randint(4, 6)
                    others = [x for x in lang_all if x != gold_lang]
                    rng.shuffle(others)
                    opts = [L.LANG_NAMES[x] for x in [gold_lang] + others[: k - 1]]
                    items.append(C.make_item(item_id=iid, set_name=split, source=f"{P}_language", question=L.qtext("lang_choice", rng),
                                             options=opts, target=[1.0] + [0.0] * (len(opts) - 1), gold=0, rng=rng, audio=audio,
                                             channel=channel, meta={**meta, "qtype": "choice", "answer": L.LANG_NAMES[gold_lang],
                                                                    "answer_class": gold_lang}))
                else:
                    yes = rng.random() < 0.5
                    asked = gold_lang if yes else rng.choice([x for x in lang_all if x != gold_lang])
                    _noul(items, rng, iid=iid, split=split, source=f"{P}_language",
                          question=L.qtext("lang_noul", rng, lang=L.LANG_NAMES[asked]), yes=yes, audio=audio, channel=channel,
                          meta={**meta, "asked_language": L.LANG_NAMES[asked]})
            elif qname == "number":
                meta["label_rule"] = "from the human transcript: digits or number words = yes; no number words = no (a lone 'one'/'oh', ordinals, 'number' = ambiguous, never asked)"
                _noul(items, rng, iid=iid, split=split, source=f"{P}_number", question=L.qtext("number_noul", rng),
                      yes=ans, audio=audio, channel=channel, meta=meta)
            elif qname == "calls_item":
                src = calls_by_id.get(ans)
                if src is None:
                    continue
                it_meta = {**base_meta, "qname": "calls_" + str(src["meta"].get("qname")), "qtype": src["meta"].get("qtype", "noul"),
                           "sb_calls_item": src["id"], "call_class": src["meta"].get("class"), "answer": src["meta"].get("answer"),
                           "answer_class": None, "label_rule": "label copied from the sb_calls item (by construction); audio re-degraded here"}
                if it_meta["qtype"] == "score":
                    it_meta["levels"] = list(src["options"])
                items.append({"id": iid, "set": split, "source": f"{P}_calls", "question": src["question"],
                              "options": list(src["options"]), "target": list(src["target"]), "gold": src["gold"],
                              "audio": audio, "channel": channel, "state_text": None, "meta": it_meta})
            else:
                continue
            used += 1
    return items


# Part B: prosody plan
PROS_REL = {"loudness": 1100, "rate": 1000, "pitch": 1250, "pitch_variation": 650}
PROS_ABS = {"pitch": 800, "pitch_variation": 350, "rate": 350}
PROS_CMP = {"pitch": 800, "rate": 400, "pitch_variation": 300}
SRC_SHARE = {"crema_d": 0.55, "libritts_r": 0.45}
AUG_SHARE = {"crema_d": 0.25, "libritts_r": 0.40}
CAL_FRAC = 0.032
PROS_FX_P = 0.20


def _fx_spec(rng: random.Random):
    if rng.random() >= PROS_FX_P:
        return None
    c, br = rng.choice(PROS_CODECS)
    return {"codec": c, "bitrate_k": br, "bandpass": [300, 3400]}


class ProsodyPlanner:
    def __init__(self, scale: float, recs: list[dict], cal_spk: set):
        self.scale = scale
        self.rec = {r["key"]: r for r in recs}
        self.cal_spk = cal_spk
        self.rng = random.Random(f"{C.SEED}:{BUILDER}:prosody")
        self.recordings: dict = {}  # key -> spec (kind, parts, fx, split, meta)
        self.items: list[dict] = []
        self.n_items = Counter()
        self.stats = Counter()
        self.shortfall = {}

    def split_of(self, spk: str) -> str:
        return "CAL" if spk in self.cal_spk else "train"

    def quota(self, n: float, split: str) -> int:
        return int(round(n * self.scale * (1.0 if split == "train" else CAL_FRAC)))

    def add_item(self, rec_key, *, source, question, options, target, gold, qtype, meta, shuffle=True):
        spec = self.recordings[rec_key]
        split = spec["split"]
        rng = self.rng
        iid = f"{BUILDER}/{split}/{source[len(BUILDER) + 1:]}/{len(self.items):06d}"
        m = {**spec["meta"], **meta, "qtype": qtype, "part": "prosody", "recording": rec_key, "fx": spec.get("fx")}
        if qtype == "score":
            m["levels"] = list(options)
        self.items.append(C.make_item(item_id=iid, set_name=split, source=source, question=question, options=list(options),
                                      target=target, gold=gold, rng=NO_SHUFFLE if (qtype == "score" or not shuffle) else rng,
                                      audio={"bank": None, "key": rec_key}, channel="clean", state_text=None, meta=m))
        self.n_items[rec_key] += 1

    # Relative two-part (sb_relative plan, no VCTK)
    def plan_relative(self):
        pl = json.loads((SBR / "plan.json").read_text())
        aug_feat, aug_bank = {}, {}
        for p in sorted(glob.glob(str(SBR / "aug_[0-9][0-9].jsonl"))):
            for row in read_jsonl(p):
                aug_feat[row["id"]] = row
                aug_bank[row["id"]] = Path(p).stem
        cands, verify = [], Counter()

        def labels_of(ref_f, tgt_f):
            out = {}
            for f in R.FEATURES:
                d = R.rel_diff(f, ref_f, tgt_f)
                if f == "loudness" and min(ref_f.get("n_voiced") or 0, tgt_f.get("n_voiced") or 0) < 10:
                    d = None
                out[f] = {"d": None if d is None else round(d, 4), "cls": R.classify(f, d)}
            return out

        for n in pl["natural"]:
            if n["source"] not in SRC_SHARE or n["ref"] not in self.rec or n["tgt"] not in self.rec:
                verify["natural_dropped_source_or_speaker"] += 1
                continue
            ref, tgt = self.rec[n["ref"]], self.rec[n["tgt"]]
            if R.total_seconds(ref["dur"], tgt["dur"]) > R.MAX_TOTAL_S:
                continue
            cands.append({"id": n["id"], "kind": "natural", "source": n["source"], "speaker": n["speaker"],
                          "ref": n["ref"], "tgt_part": ["sbv_work2", tgt["shard"], tgt["key"]], "tgt_key": tgt["key"],
                          "labels": labels_of(ref["feat"], tgt["feat"]), "tgt_s": tgt["dur"]})
        for a in pl["aug"]:
            if a["source"] not in SRC_SHARE or a["ref"] not in self.rec or a["tgt"] not in self.rec:
                continue
            row = aug_feat.get(a["id"])
            if row is None:
                verify["aug_not_rendered"] += 1
                continue
            ref = self.rec[a["ref"]]
            if R.total_seconds(ref["dur"], row["dur"]) > R.MAX_TOTAL_S:
                continue
            tf = dict(row["feat"])
            if a["source"] == "crema_d" and tf.get("span_s"):
                syl = R.crema_syllables(self.rec[a["tgt"]]["source_id"])
                tf["syll"], tf["rate"] = syl, (syl / tf["span_s"] if tf["span_s"] >= 0.5 else None)
            labels = labels_of(ref["feat"], tf)
            f, exp = a["aug"]["feature"], a["aug"]["expected_cls"]
            if labels[f]["cls"] != exp:
                verify[f"aug_mismatch_{f}"] += 1
                labels[f] = {"d": labels[f]["d"], "cls": None}
            else:
                verify[f"aug_ok_{f}"] += 1
            cands.append({"id": a["id"], "kind": "augmented", "source": a["source"], "speaker": a["speaker"], "ref": a["ref"],
                          "tgt_part": ["sbr_work", aug_bank[a["id"]], a["id"]], "tgt_key": a["tgt"], "aug": a["aug"],
                          "labels": labels, "tgt_s": row["dur"]})
        self.stats.update({f"rel_{k}": v for k, v in verify.items()})
        by = defaultdict(list)
        for c in cands:
            c["split"] = self.split_of(c["speaker"])
            for f, lab in c["labels"].items():
                if lab["cls"] is not None:
                    by[(c["split"], c["source"], f, lab["cls"])].append(c)
        used_feat = defaultdict(set)
        rng = self.rng

        def rec_key_of(c):
            k = f"rel/{c['id']}"
            if k not in self.recordings:
                ref = self.rec[c["ref"]]
                self.recordings[k] = {"kind": "two_part", "split": c["split"], "fx": _fx_spec(rng),
                                      "parts": [["sbv_work2", ref["shard"], ref["key"]], c["tgt_part"]],
                                      "meta": {"corpus": c["source"], "speaker": c["speaker"], "source_id": self.rec[c["tgt_key"]]["source_id"],
                                               "ref_source_id": ref["source_id"], "pair_kind": c["kind"], "augmentation": c.get("aug"),
                                               "ref_emotion": ref.get("emotion"), "tgt_emotion": self.rec[c["tgt_key"]].get("emotion"),
                                               "licence": LIC[c["source"]], "sb_relative_pair": c["id"]}}
            return k

        def pool(split, src, f, cls):
            lst = [c for c in by[(split, src, f, cls)] if f not in used_feat[c["id"]] and self.n_items[f"rel/{c['id']}"] < 4]
            nat = [c for c in lst if c["kind"] == "natural"]
            aug = [c for c in lst if c["kind"] == "augmented"]
            rng.shuffle(nat)
            rng.shuffle(aug)
            nat.sort(key=lambda c: -self.n_items[f"rel/{c['id']}"])
            aug.sort(key=lambda c: -self.n_items[f"rel/{c['id']}"])
            share = AUG_SHARE[src] if f != "pitch_variation" else 0.0
            out, i, j = [], 0, 0
            while i < len(nat) or j < len(aug):
                if j < len(aug) and (i >= len(nat) or rng.random() < share):
                    out.append(aug[j]); j += 1
                else:
                    out.append(nat[i]); i += 1
            return out

        def emit(c, f, qtype, question, options, target, gold, extra):
            k = rec_key_of(c)
            lab = c["labels"][f]
            th = R.THRESHOLDS[f]
            self.add_item(k, source=f"{BUILDER}_rel_{f}", question=question, options=options, target=target, gold=gold, qtype=qtype,
                          meta={"family": f"relative_{f}", "feature": f, "diff": lab["d"], "diff_unit": th[2],
                                "class": R.CLASS_NAMES[lab["cls"]], "answer_class": R.CLASS_NAMES[lab["cls"]] if qtype != "noul" else extra.get("answer"),
                                "label_rule": f"same speaker; second part minus first part (reference = the speaker's normal voice); clear >= {th[0]:.4g}, "
                                              f"about the same <= {th[1]:.4g} ({th[2]}); in-between pairs dropped", **extra})
            used_feat[c["id"]].add(f)

        for split in ("train", "CAL"):
            for f in ("pitch_variation", "pitch", "rate", "loudness"):
                for src in SRC_SHARE:
                    n_total = self.quota(PROS_REL[f] * SRC_SHARE[src], split)
                    n_score = int(round(n_total * 0.55 / 3.0))
                    n_noul = int(round(n_total * 0.45 / 6.0))
                    pools = {cls: pool(split, src, f, cls) for cls in (0, 1, 2)}
                    k = min([n_score] + [len(p) for p in pools.values()])
                    self.shortfall[f"rel/{split}/{src}/{f}/score"] = n_score - k
                    for cls in (0, 1, 2):
                        for c in pools[cls][:k]:
                            q, opts, oset = R.score_question(f, rng)
                            t = [0.0, 0.0, 0.0]
                            t[cls] = 1.0
                            emit(c, f, "score", q, opts, t, cls, {"option_set": oset, "qname": "score"})
                    for direction in ("up", "down", "same"):
                        yes_cls = {"up": 2, "down": 0, "same": 1}[direction]
                        no_cls = [x for x in (0, 1, 2) if x != yes_cls]
                        py = pool(split, src, f, yes_cls)
                        pn = [pool(split, src, f, x) for x in no_cls]
                        half = n_noul // 2
                        no_list = pn[0][:half] + pn[1][: n_noul - min(half, len(pn[0]))]
                        m = min(n_noul, len(py), len(no_list))
                        self.shortfall[f"rel/{split}/{src}/{f}/noul_{direction}"] = n_noul - m
                        rng.shuffle(no_list)
                        for is_yes, lst in ((True, py[:m]), (False, no_list[:m])):
                            for c in lst:
                                if f in used_feat[c["id"]]:
                                    continue
                                emit(c, f, "noul", R.noul_question(f, direction, rng), ["yes", "no"],
                                     [1.0, 0.0] if is_yes else [0.0, 1.0], 0 if is_yes else 1,
                                     {"direction": direction, "answer": "yes" if is_yes else "no", "qname": f"noul_{direction}"})

    # Absolute single-clip levels
    def plan_absolute(self):
        rng = self.rng
        by = defaultdict(list)
        for r in sorted(self.rec.values(), key=lambda r: r["key"]):
            if not 1.5 <= r["dur"] <= 15.0:
                continue
            for f in PROS_ABS:
                lv = R.abs_level(f, R.abs_value(f, r["feat"]))
                if lv is not None:
                    by[(self.split_of(r["speaker"]), f, lv)].append(r)
        self.stats.update({f"abs_avail/{k[0]}/{k[1]}/{k[2]}": len(v) for k, v in by.items()})
        used = defaultdict(set)

        def rec_key_of(r):
            k = f"abs/{r['key']}"
            if k not in self.recordings:
                self.recordings[k] = {"kind": "single", "split": self.split_of(r["speaker"]), "fx": _fx_spec(rng),
                                      "parts": [["sbv_work2", r["shard"], r["key"]]],
                                      "meta": {"corpus": r["source"], "speaker": r["speaker"], "source_id": r["source_id"],
                                               "emotion": r.get("emotion"), "licence": LIC[r["source"]]}}
            return k

        def pool(split, f, lv):
            lst = [r for r in by[(split, f, lv)] if f not in used[r["key"]] and self.n_items[f"abs/{r['key']}"] < 2]
            rng.shuffle(lst)
            # spread over speakers / corpora: interleave by corpus share
            cr = [r for r in lst if r["source"] == "crema_d"]
            lb = [r for r in lst if r["source"] != "crema_d"]
            out, i, j = [], 0, 0
            while i < len(cr) or j < len(lb):
                if i < len(cr) and (j >= len(lb) or rng.random() < 0.5):
                    out.append(cr[i]); i += 1
                else:
                    out.append(lb[j]); j += 1
            return out

        def emit(r, f, qtype, q, opts, target, gold, extra):
            k = rec_key_of(r)
            v = R.abs_value(f, r["feat"])
            lv = R.abs_level(f, v)
            self.add_item(k, source=f"{BUILDER}_abs_{f}", question=q, options=opts, target=target, gold=gold, qtype=qtype,
                          meta={"family": f"absolute_{f}", "feature": f, "value": round(v, 4), "unit": R.ABS_UNIT[f], "level": lv,
                                "answer_class": str(lv) if qtype != "noul" else extra.get("answer"),
                                "label_rule": f"measured on the clean clip; bins {R.ABS_BINS[f]} ({R.ABS_UNIT[f]}); values in the gaps not asked",
                                **extra})
            used[r["key"]].add(f)

        for split in ("train", "CAL"):
            for f, n in PROS_ABS.items():
                n_total = self.quota(n, split)
                n_score = int(round(n_total * 0.5 / 3.0))
                pools = {lv: pool(split, f, lv) for lv in (0, 1, 2)}
                k = min([n_score] + [len(p) for p in pools.values()])
                self.shortfall[f"abs/{split}/{f}/score"] = n_score - k
                for lv in (0, 1, 2):
                    for r in pools[lv][:k]:
                        q = rng.choice(R.ABS_SCORE_Q[f])
                        oi = rng.randrange(len(R.ABS_OPTIONS[f]))
                        t = [0.0, 0.0, 0.0]
                        t[lv] = 1.0
                        emit(r, f, "score", q, R.ABS_OPTIONS[f][oi], t, lv, {"option_set": oi, "qname": "score"})
                # noul: "is it high/low?"; yes = that extreme, no = the other levels half / half
                n_noul = int(round(n_total * 0.5 / 4.0))
                for ext in (2, 0):
                    others = [x for x in (0, 1, 2) if x != ext]
                    py = pool(split, f, ext)
                    pn = [pool(split, f, x) for x in others]
                    half = n_noul // 2
                    no_list = pn[0][:half] + pn[1][: n_noul - min(half, len(pn[0]))]
                    m = min(n_noul, len(py), len(no_list))
                    self.shortfall[f"abs/{split}/{f}/noul_{ext}"] = n_noul - m
                    for is_yes, lst in ((True, py[:m]), (False, no_list[:m])):
                        for r in lst:
                            if f in used[r["key"]]:
                                continue
                            emit(r, f, "noul", rng.choice(R.ABS_NOUL_Q[(f, ext)]), ["yes", "no"], [1.0, 0.0] if is_yes else [0.0, 1.0],
                                 0 if is_yes else 1, {"direction": "high" if ext == 2 else "low", "answer": "yes" if is_yes else "no",
                                                      "qname": f"noul_{ext}"})

    # Cross-speaker comparisons
    def plan_compare(self):
        rng = self.rng
        pool_by_split = defaultdict(list)
        for r in sorted(self.rec.values(), key=lambda r: r["key"]):
            if 1.5 <= r["dur"] <= 8.0:
                pool_by_split[self.split_of(r["speaker"])].append(r)
        n_rec_used = Counter()
        for split in ("train", "CAL"):
            P = pool_by_split[split]
            for f, n in PROS_CMP.items():
                n_total = self.quota(n, split)
                vals = [(R.abs_value(f, r["feat"]), r) for r in P]
                vals = [(v, r) for v, r in vals if v is not None and v > 0]
                if len(vals) < 20:
                    continue
                vals.sort(key=lambda t: t[0])
                by_cls = defaultdict(list)
                tries = 0
                want = int(n_total * 0.6) + 10
                # random pairs (mostly clear differences) + neighbour pairs (about the same)
                while tries < want * 40 and (len(by_cls[0]) < want or len(by_cls[2]) < want or len(by_cls[1]) < want):
                    tries += 1
                    if rng.random() < 0.45:
                        i = rng.randrange(len(vals) - 1)
                        j = min(len(vals) - 1, i + rng.randint(1, 6))
                        a, b = (vals[i][1], vals[j][1]) if rng.random() < 0.5 else (vals[j][1], vals[i][1])
                    else:
                        a, b = rng.choice(vals)[1], rng.choice(vals)[1]
                    if a["speaker"] == b["speaker"] or R.total_seconds(a["dur"], b["dur"]) > R.MAX_TOTAL_S:
                        continue
                    if n_rec_used[a["key"]] >= 3 or n_rec_used[b["key"]] >= 3:
                        continue
                    d = R.cmp_diff(f, a["feat"], b["feat"])
                    cls = R.cmp_class(f, d)
                    if cls is None or len(by_cls[cls]) >= want:
                        continue
                    by_cls[cls].append((a, b, d))
                    n_rec_used[a["key"]] += 1
                    n_rec_used[b["key"]] += 1
                self.stats.update({f"cmp_avail/{split}/{f}/{c}": len(v) for c, v in by_cls.items()})
                n_choice = int(round(n_total * 0.4 / 2.0))
                n_score = int(round(n_total * 0.3 / 3.0))
                n_noul = int(round(n_total * 0.3 / 4.0))
                ptr = {c: 0 for c in (0, 1, 2)}

                def take(c, k):
                    out = by_cls[c][ptr[c]: ptr[c] + k]
                    ptr[c] += len(out)
                    return out

                def emit(pair, qtype, q, opts, target, gold, extra):
                    a, b, d = pair
                    key = f"cmp/{f}/{a['key']}__{b['key']}"
                    if key not in self.recordings:
                        self.recordings[key] = {"kind": "two_part", "split": split, "fx": _fx_spec(rng),
                                                "parts": [["sbv_work2", a["shard"], a["key"]], ["sbv_work2", b["shard"], b["key"]]],
                                                "meta": {"corpus": a["source"] if a["source"] == b["source"] else f"{a['source']}+{b['source']}",
                                                         "speaker": [a["speaker"], b["speaker"]], "source_id": [a["source_id"], b["source_id"]],
                                                         "licence": "; ".join(sorted({LIC[a["source"]], LIC[b["source"]]}))}}
                    cls = R.cmp_class(f, d)
                    th = R.CMP_THRESHOLDS[f]
                    self.add_item(key, source=f"{BUILDER}_cmp_{f}", question=q, options=opts, target=target, gold=gold, qtype=qtype,
                                  meta={"family": f"compare_{f}", "feature": f, "diff": round(d, 4), "diff_unit": th[2], "class": R.CLASS_NAMES[cls],
                                        "values": [round(R.abs_value(f, a["feat"]), 4), round(R.abs_value(f, b["feat"]), 4)],
                                        "answer_class": extra.pop("answer_class", None),
                                        "label_rule": f"two different speakers; second minus first; clear >= {th[0]:.4g}, about the same <= {th[1]:.4g} ({th[2]}); in-between dropped",
                                        **extra})

                def intro(q):
                    return f"{rng.choice(R.CMP_INTROS)} {q}" if rng.random() < R.CMP_INTRO_P else q

                # choice: which one is higher / lower (answer first vs second balanced: classes 0 and 2 equally)
                k = min(n_choice, len(by_cls[0]), len(by_cls[2]))
                self.shortfall[f"cmp/{split}/{f}/choice"] = n_choice - k
                for c in (0, 2):
                    for pair in take(c, k):
                        ask = rng.choice(["hi", "lo"])
                        second_is_hi = c == 2
                        gold = 1 if (second_is_hi == (ask == "hi")) else 0
                        opts = rng.choice(R.CMP_CHOICE_OPTS)
                        emit(pair, "choice", intro(rng.choice(R.CMP_CHOICE_Q[(f, ask)])), list(opts),
                             [1.0, 0.0] if gold == 0 else [0.0, 1.0], gold, {"ask": ask, "answer": opts[gold], "answer_class": ["first", "second"][gold],
                                                                             "qname": "choice"})
                # score: lower / same / higher, balanced
                k = min([n_score] + [len(by_cls[c]) - ptr[c] for c in (0, 1, 2)])
                self.shortfall[f"cmp/{split}/{f}/score"] = n_score - k
                for c in (0, 1, 2):
                    for pair in take(c, k):
                        oi = rng.randrange(len(R.CMP_SCORE_OPTS[f]))
                        t = [0.0, 0.0, 0.0]
                        t[c] = 1.0
                        emit(pair, "score", intro(rng.choice(R.CMP_SCORE_Q[f])), R.CMP_SCORE_OPTS[f][oi], t, c,
                             {"option_set": oi, "answer_class": R.CLASS_NAMES[c], "qname": "score"})
                # noul: "is the second higher?" (yes = 2; no = 0 / 1) and "about the same?" (yes = 1; no = 0 / 2)
                for direction, yes_c, no_cs in (("up", 2, (0, 1)), ("same", 1, (0, 2))):
                    half = n_noul // 2
                    py = take(yes_c, n_noul)
                    pn = take(no_cs[0], half) + take(no_cs[1], n_noul - half)
                    m = min(len(py), len(pn))
                    self.shortfall[f"cmp/{split}/{f}/noul_{direction}"] = n_noul - m
                    rng.shuffle(pn)
                    for is_yes, lst in ((True, py[:m]), (False, pn[:m])):
                        for pair in lst:
                            emit(pair, "noul", intro(rng.choice(R.CMP_NOUL_Q[(f, direction)])), ["yes", "no"],
                                 [1.0, 0.0] if is_yes else [0.0, 1.0], 0 if is_yes else 1,
                                 {"direction": direction, "answer": "yes" if is_yes else "no", "answer_class": "yes" if is_yes else "no",
                                  "qname": f"noul_{direction}"})

    def build(self):
        self.plan_relative()
        self.plan_absolute()
        self.plan_compare()
        return self.recordings, self.items


# Balance
def balance_noul(items: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
    """Every yes/no family (source + qname) exactly 50/50 per split (drop random majority items)."""
    groups, keep, rep = defaultdict(list), [], {}
    for it in items:
        if it["meta"]["qtype"] == "noul":
            groups[(it["set"], it["source"], str(it["meta"].get("qname")))].append(it)
        else:
            keep.append(it)
    for g, lst in sorted(groups.items()):
        yes = [i for i in lst if i["options"][i["gold"]] == "yes"]
        no = [i for i in lst if i["options"][i["gold"]] == "no"]
        n = min(len(yes), len(no))
        rng.shuffle(yes)
        rng.shuffle(no)
        keep += yes[:n] + no[:n]
        rep["/".join(g)] = {"yes": len(yes), "no": len(no), "kept_each": n}
    return keep, rep


def cap_choice(items: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
    """choice / score answers capped at 2x the rarest answer class per (split, source, qtype). sb_calls copies keep
    sb_calls' own balance (answer_class None)."""
    groups, keep, rep = defaultdict(list), [], {}
    for it in items:
        ac = it["meta"].get("answer_class")
        if it["meta"]["qtype"] in ("choice", "score") and ac is not None:
            groups[(it["set"], it["source"], it["meta"]["qtype"])].append(it)
        else:
            keep.append(it)
    for g, lst in sorted(groups.items()):
        by = defaultdict(list)
        for it in lst:
            by[str(it["meta"]["answer_class"])].append(it)
        lo = min(len(v) for v in by.values())
        cap = max(2 * lo, 1)
        kept = {}
        for a, v in sorted(by.items()):
            rng.shuffle(v)
            keep += v[:cap]
            kept[a] = [len(v), min(len(v), cap)]
        rep["/".join(g)] = kept
    return keep, rep


# Checks
def mmsu_questions() -> set:
    """The normalised 'question' field of /vol/bench_mmsu/truth.jsonl, used only for the non-equality test."""
    out = set()
    if not MMSU_TRUTH.exists():
        return out
    for line in open(MMSU_TRUTH):
        if line.strip():
            q = json.loads(line).get("question")
            if q:
                out.add(norm_q(q))
    return out


def norm_q(q: str) -> str:
    return " ".join(str(q).lower().split()).strip(" ?.!")


def mmsu_equal_count(questions, mmsu: set) -> int:
    return sum(1 for q in questions if norm_q(q) in mmsu)


def validate(out: Path, items: list[dict]) -> dict:
    idx = {}
    for p in (out / "audio").glob("*.json"):
        idx[p.stem] = json.loads(p.read_text())
    errs, durs, per_clip = Counter(), [], Counter()
    ids = Counter(i["id"] for i in items)
    for it in items:
        for k in ("id", "set", "source", "question", "options", "target", "gold", "audio", "channel", "state_text", "meta"):
            if k not in it:
                errs[f"missing_{k}"] += 1
        if not (2 <= len(it["options"]) <= 20):
            errs["n_options"] += 1
        if len(it["target"]) != len(it["options"]):
            errs["target_len"] += 1
        if abs(sum(it["target"]) - 1) > 1e-4:
            errs["target_sum"] += 1
        if max(it["target"]) >= 0.99 and int(np.argmax(it["target"])) != it["gold"]:
            errs["gold_argmax"] += 1
        q = it["meta"].get("qtype")
        if q not in ("choice", "noul", "score"):
            errs["qtype"] += 1
        if q == "noul" and sorted(it["options"]) != ["no", "yes"]:
            errs["noul_options"] += 1
        if q == "score" and it["meta"].get("levels") and list(it["options"]) != list(it["meta"]["levels"]):
            errs["score_order"] += 1
        if len(set(it["options"])) != len(it["options"]):
            errs["dup_options"] += 1
        if it["set"] not in ("train", "CAL"):
            errs["set"] += 1
        if it["channel"] not in ("clean", "phone"):
            errs["channel"] += 1
        if not it["source"].startswith(BUILDER + "_") or any(it["source"].startswith(p) for p in ("sb_phone_", "sb_count_", "sb_compare_", "sb_phon_", "fsd50k", "ts:", "jade")
                                                             if not BUILDER.startswith(p)):  # "sb_phone_pitch" starts with "sb_phone_"
            errs["source_prefix"] += 1
        if not it["id"].startswith(BUILDER + "/"):
            errs["id_prefix"] += 1
        a = it["audio"]
        if not a or a["bank"] not in idx or a["key"] not in idx[a["bank"]]:
            errs["audio_key_missing"] += 1
        else:
            durs.append(idx[a["bank"]][a["key"]][1] / SR)
            if not a["bank"].startswith(BUILDER + "_"):
                errs["bank_prefix"] += 1
            per_clip[(a["bank"], a["key"])] += 1
    errs["dup_ids"] = sum(1 for v in ids.values() if v > 1)
    errs["clips_over_4_items"] = sum(1 for v in per_clip.values() if v > 4)
    spk = defaultdict(set)
    for it in items:
        s = it["meta"].get("speaker")
        for x in (s if isinstance(s, list) else [s]):
            if x and not str(x).startswith("kokoro:None"):
                spk[it["set"]].add(str(x))
    overlap = spk["train"] & spk["CAL"]
    errs["train_cal_speaker_overlap"] = len(overlap)
    d = np.array(durs) if durs else np.zeros(1)
    errs["clips_shorter_1s"] = int((d < 1.0).sum())
    errs["clips_longer_30s"] = int((d > 30.05).sum())
    samp = random.Random(1).sample(items, min(400, len(items)))
    bad = 0
    banks = {}
    for it in samp:
        a = it["audio"]
        try:
            if a["bank"] not in banks:
                banks[a["bank"]] = C.AudioBank(out / "audio", a["bank"])
            x = banks[a["bank"]].get(a["key"])
            if not np.isfinite(x).all() or np.max(np.abs(x)) > 1.0 or len(x) < 1.0 * SR or np.max(np.abs(x)) < 1e-4:
                bad += 1
        except Exception:
            bad += 1
    errs["bad_audio_sample"] = bad
    bal = defaultdict(Counter)
    for it in items:
        if it["meta"]["qtype"] == "noul":
            bal[f"{it['set']}/{it['source']}/{it['meta'].get('qname')}"][it["options"][it["gold"]]] += 1
    errs["noul_unbalanced_groups"] = sum(1 for v in bal.values() if v["yes"] != v["no"])
    errs = {k: v for k, v in errs.items() if v}
    return {"ok": not errs, "errors": dict(errs), "n_items": len(items),
            "duration_s": {"min": round(float(d.min()), 2), "mean": round(float(d.mean()), 2), "median": round(float(np.median(d)), 2),
                           "max": round(float(d.max()), 2)},
            "max_items_per_clip": max(per_clip.values()) if per_clip else 0, "train_cal_overlap_examples": sorted(overlap)[:5]}


def leakage_check(items: list[dict], phone_clips: list[dict], recs: dict) -> dict:
    adi = adi_ids()
    rep = {"adi_items": adi["n"]}
    our_spk, our_src = set(), set()
    for it in items:
        m = it["meta"]
        for k, dst in (("speaker", our_spk), ("source_id", our_src), ("ref_source_id", our_src)):
            v = m.get(k)
            for x in (v if isinstance(v, list) else [v]):
                if isinstance(x, list):
                    dst.update(map(str, x))
                elif x:
                    dst.add(str(x))
    our_crema = {s.split(":")[1] for s in our_spk if s.startswith("crema_d:")}
    our_hvb_calls = {str(it["meta"].get("hvb_call")) for it in items if it["meta"].get("hvb_call")}
    slue_test = set(json.loads((HVB / "slue_ids.json").read_text())["test"])
    ev, _ = exp0_crema_actors()
    rep["crema_actors_used"] = len(our_crema)
    rep["crema_actor_overlap_adi"] = sorted(our_crema & adi["crema_actors"])
    rep["crema_actor_overlap_exp0_eval"] = sorted(our_crema & ev)
    rep["hvb_calls_used"] = len(our_hvb_calls)
    rep["hvb_calls_in_slue_test"] = len(our_hvb_calls & slue_test)
    rep["hvb_calls_in_adi"] = len(our_hvb_calls & adi["hvb_calls"])
    our_hvb_spk = {s for s in our_spk if s.startswith("hvb_spk:")}
    adi_hvb_spk = {s if s.startswith("hvb_spk:") else f"hvb_spk:{s}" for s in adi["hvb_speakers"]}
    rep["hvb_speakers_also_in_adi_hvb_items"] = len(our_hvb_spk & adi_hvb_spk)
    rep["adi_bank_name_hits"] = sorted({it["audio"]["bank"] for it in items} & adi["banks"])
    minds_paths = {c["meta"]["source_id"] for c in phone_clips if c["corpus"] == "minds14"}
    rep["minds_locales_used"] = sorted({c["meta"].get("locale") for c in phone_clips if c["corpus"] == "minds14"})
    rep["minds_path_hits_adi"] = len(minds_paths & adi["meta_values"])
    src_long = {s for s in our_src if len(s) > 8 and "response_" not in s}
    rep["source_id_hits_adi_meta"] = sorted(src_long & adi["meta_values"])[:10]
    # earlier eval items (EXP0_EVAL): no shared CREMA-D clip / MInDS path / source id
    ev_vals = set()
    for it in read_jsonl(EXP0_EVAL):
        m = it.get("meta") or {}
        for v in list(m.values()) + [(it.get("audio") or {}).get("key")]:
            if isinstance(v, (str, int)) and len(str(v)) >= 4:
                ev_vals.add(str(v))
                ev_vals.add(Path(str(v)).stem)
    rep["source_id_hits_exp0_eval"] = sorted(src_long & ev_vals)[:10]
    rep["minds_path_hits_exp0_eval"] = len(minds_paths & ev_vals)
    sbv_cal = set(json.loads((SBV / "cal_speakers.json").read_text()))
    rep["sb_voice_cal_speakers_used"] = sorted(our_spk & sbv_cal)
    rep["banned_corpora_present"] = sorted({str(it["meta"].get("corpus")) for it in items
                                            if re.search(r"vctk|vystadial|common.?voice|gigaspeech|emilia|meld|ravdess|iemocap|vocalsound",
                                                         str(it["meta"].get("corpus")), re.I)})
    rep["ok"] = not (rep["crema_actor_overlap_adi"] or rep["crema_actor_overlap_exp0_eval"] or rep["hvb_calls_in_slue_test"]
                     or rep["hvb_calls_in_adi"] or rep["adi_bank_name_hits"] or rep["minds_path_hits_adi"] or rep["source_id_hits_adi_meta"]
                     or rep["source_id_hits_exp0_eval"] or rep["minds_path_hits_exp0_eval"] or rep["sb_voice_cal_speakers_used"]
                     or rep["banned_corpora_present"] or any(l.startswith("en") for l in rep["minds_locales_used"]))
    return rep


def text_checks(items: list[dict]) -> dict:
    mm = mmsu_questions()
    qs = [it["question"] for it in items]
    pools = set(R.all_question_texts())
    for v in L.Q.values():
        pools |= set(v)
    adi_eval_rel = R.eval_questions()
    return {"mmsu_questions_loaded": len(mm), "item_questions_equal_mmsu": mmsu_equal_count(qs, mm),
            "pool_templates_equal_mmsu": mmsu_equal_count(pools, mm),
            "item_questions_equal_adi_extra_relative": sum(1 for q in qs if q in adi_eval_rel),
            "n_distinct_questions": len(set(qs))}


def spot_check(out: Path, items: list[dict], n: int = 120) -> dict:
    """Re-measure median F0 of both parts of a sample of final pitch recordings (incl. phone-fx ones): class agreement."""
    import librosa

    cand = [it for it in items if it["source"] in (f"{BUILDER}_rel_pitch", f"{BUILDER}_cmp_pitch") and it["meta"]["qtype"] == "score"]
    samp = random.Random(7).sample(cand, min(n, len(cand)))
    banks, agree, tot, fx_agree, fx_tot, errs = {}, 0, 0, 0, 0, []
    for it in samp:
        a = it["audio"]
        if a["bank"] not in banks:
            banks[a["bank"]] = C.AudioBank(out / "audio", a["bank"])
        y = banks[a["bank"]].get(a["key"])
        lay = it["meta"].get("layout") or {}
        if not lay:
            continue
        p1 = y[: int(lay["ref_s"] * SR)]
        p2 = y[int(lay["tgt_start_s"] * SR):]
        meds = []
        for x in (p1, p2):
            f0, vf, _ = librosa.pyin(x, fmin=65.0, fmax=500.0, sr=SR, frame_length=1024, hop_length=320)
            rms = librosa.feature.rms(y=x, frame_length=1024, hop_length=320, center=True)[0]
            k = min(len(rms), len(f0))
            v = R.energy_gated_voicing(np.asarray(vf[:k], bool) & np.isfinite(f0[:k]), rms[:k])
            meds.append(R.f0_stats(f0[:k], v)["f0_med"])
        if not all(np.isfinite(meds)):
            continue
        d = 12 * math.log2(meds[1] / meds[0])
        cls = R.classify("pitch", d) if it["source"].endswith("rel_pitch") else R.cmp_class("pitch", d)
        want = R.CLASS_NAMES.index(it["meta"]["class"])
        tot += 1
        ok = cls == want
        agree += ok
        if it["meta"].get("fx"):
            fx_tot += 1
            fx_agree += ok
        if not ok and len(errs) < 8:
            errs.append({"id": it["id"], "want": want, "got": cls, "d_measured": round(d, 2), "d_label": it["meta"]["diff"]})
    return {"pitch_recordings_checked": tot, "same_class": agree, "phone_fx_checked": fx_tot, "phone_fx_same_class": fx_agree,
            "clean_checked": tot - fx_tot, "clean_same_class": agree - fx_agree, "examples_disagree": errs,
            "note": "labels come from the clean per-part measurement; re-measuring the final recording moves some 'about the same' "
                    "pairs into the dropped 0.5-1.5 st zone, and on phone-fx recordings (300-3400 Hz band-pass) pYIN loses the "
                    "fundamental of low voices (octave errors), so fx agreement is a measurement limit, not a label change"}


# Main
def main(smoke: bool, render_map, commit, reload) -> dict:
    t0 = time.time()
    out = WORK / ("smoke_out" if smoke else "full_out")
    scale = 0.03 if smoke else 1.0
    if out.exists():
        shutil.rmtree(out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    recs, rec_info = load_records()
    cal = cal_speakers()
    # Plans
    PP = PhonePlanner(scale, recs, cal)
    phone_clips = PP.build()
    print(f"[plan] phone clips {len(phone_clips)} {Counter((c['corpus'], c['split']) for c in phone_clips)}", flush=True)
    QP = ProsodyPlanner(scale, recs, cal)
    pros_recs, pros_items = QP.build()
    print(f"[plan] prosody recordings {len(pros_recs)} items {len(pros_items)}", flush=True)
    # Shards
    shards = []
    for part, specs_all in (("phone", [{"key": c["key"], "kind": "phone", "segs": c["segs"], "crop": c["crop"], "recipe": c["recipe"],
                                         "bg": c["bg"], "split": c["split"]} for c in phone_clips]),
                            ("pros", [{"key": k, "kind": v["kind"], "parts": v["parts"], "fx": v["fx"], "split": v["split"]}
                                      for k, v in sorted(pros_recs.items())])):
        for split in ("train", "CAL"):
            lst = [s for s in specs_all if s["split"] == split]
            random.Random(f"{C.SEED}:{BUILDER}:shard:{part}:{split}").shuffle(lst)
            per = 650 if part == "phone" else 1100
            nb = max(1, math.ceil(len(lst) / per))
            for b in range(nb):
                name = f"{BUILDER}_{part}_{'cal' if split == 'CAL' else 'train'}_{b:02d}"
                shards.append((name, lst[b::nb]))
    bank_of = {s["key"]: name for name, specs in shards for s in specs}
    (out / "plan.json").write_text(json.dumps({"phone_clips": phone_clips, "pros_recordings": pros_recs, "cal_groups": PP.cal_groups}))
    commit()
    print(f"[render] {len(shards)} shards", flush=True)
    res = list(render_map([(name, specs, str(out / "audio")) for name, specs in shards]))
    print(f"[render] {res}", flush=True)
    reload()
    facts = {}
    for name, _ in shards:
        p = out / "facts" / f"{name}.json"
        if p.exists():
            facts.update(json.loads(p.read_text()))
    # Items
    calls_by_id = {i["id"]: i for i in PP.S["calls_items"]}
    items = make_phone_items(phone_clips, facts, bank_of, calls_by_id)
    for it in pros_items:
        k = it["audio"]["key"]
        f = facts.get(k, {})
        if "error" in f or k not in bank_of or not f:
            continue
        it["audio"]["bank"] = bank_of[k]
        if pros_recs[k]["kind"] == "two_part":
            it["meta"]["layout"] = {kk: f.get(kk) for kk in ("ref_s", "tgt_start_s", "tgt_s", "scale")}
        items.append(it)
    items, cap_rep = cap_choice(items, random.Random(f"{C.SEED}:{BUILDER}:cap"))
    items, bal_rep = balance_noul(items, random.Random(f"{C.SEED}:{BUILDER}:balance"))
    items.sort(key=lambda i: h01("order:" + i["id"]))
    tr = [i for i in items if i["set"] == "train"]
    calit = [i for i in items if i["set"] == "CAL"]
    # unused clips stay in the banks (small)
    C.write_jsonl(out / "items_train.jsonl", tr)
    C.write_jsonl(out / "items_cal.jsonl", calit)
    commit()
    val = validate(out, tr + calit)
    leak = leakage_check(tr + calit, phone_clips, QP.rec)
    txt = text_checks(tr + calit)
    try:
        spot = spot_check(out, tr + calit, n=30 if smoke else 120)
    except Exception as e:
        spot = {"error": repr(e)[:300]}
    summ = make_summary(out, tr, calit, res, cap_rep, bal_rep, val, leak, txt, spot, PP, QP, rec_info, facts, time.time() - t0, smoke)
    ok = val["ok"] and leak["ok"] and txt["item_questions_equal_mmsu"] == 0 and txt["item_questions_equal_adi_extra_relative"] == 0
    summ["all_checks_ok"] = ok
    (out / "summary_draft.json").write_text(json.dumps(summ, indent=1, default=str))
    commit()
    published = None
    if not smoke and ok:
        published = publish(out, summ)
        commit()
    return {"n_train": len(tr), "n_cal": len(calit), "validation_ok": val["ok"], "validation_errors": val["errors"], "leak_ok": leak["ok"],
            "text": txt, "spot": spot, "published": published, "seconds": round(time.time() - t0),
            "by_source": summ["by_source_split"], "hours": summ["audio_hours_used"]}


def publish(out: Path, summ: dict) -> str:
    """Copy the validated build to /vol/stageb/sb_phone_pitch/ once: banks, items, then summary.json last."""
    assert not (FINAL / "summary.json").exists(), "final summary.json already exists"
    (FINAL / "audio").mkdir(parents=True, exist_ok=True)
    for p in sorted((out / "audio").glob("*")):
        shutil.copyfile(p, FINAL / "audio" / p.name)
    shutil.copyfile(out / "items_train.jsonl", FINAL / "items_train.jsonl")
    shutil.copyfile(out / "items_cal.jsonl", FINAL / "items_cal.jsonl")
    # re-validate the published copy (audio keys resolve from /vol/stageb/sb_phone_pitch/audio)
    items = read_jsonl(FINAL / "items_train.jsonl") + read_jsonl(FINAL / "items_cal.jsonl")
    v2 = validate(FINAL, items)
    summ["validation_published_copy"] = v2
    summ["paths"] = {"items_train": str(FINAL / "items_train.jsonl"), "items_cal": str(FINAL / "items_cal.jsonl"),
                     "audio": str(FINAL / "audio"), "summary": str(FINAL / "summary.json")}
    (FINAL / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    return str(FINAL / "summary.json")


def make_summary(out, tr, cal, res, cap_rep, bal_rep, val, leak, txt, spot, PP, QP, rec_info, facts, secs, smoke) -> dict:
    allit = tr + cal
    by = lambda f: dict(Counter(f(i) for i in allit).most_common())  # noqa: E731
    used_keys = {(i["audio"]["bank"], i["audio"]["key"]) for i in allit}
    idx_cache, hours, hours_part, hours_corpus = {}, 0.0, Counter(), Counter()
    corpus_of = {}
    for i in allit:
        corpus_of[(i["audio"]["bank"], i["audio"]["key"])] = (i["meta"].get("part"), str(i["meta"].get("corpus")))
    for b, k in used_keys:
        if b not in idx_cache:
            idx_cache[b] = json.loads((out / "audio" / f"{b}.json").read_text())
        h = idx_cache[b][k][1] / SR / 3600
        hours += h
        part, corp = corpus_of[(b, k)]
        hours_part[part] += h
        hours_corpus[corp] += h
    bank_hours = {b: round(sum(v[1] for v in ix.values()) / SR / 3600, 2) for b, ix in idx_cache.items()}
    n_part = Counter(i["meta"].get("part") for i in allit)
    speech_items = Counter()
    for i in allit:
        c = str(i["meta"].get("corpus"))
        for x in c.split("+"):
            speech_items[x] += 1.0 / len(c.split("+"))
    tot = sum(speech_items.values())
    examples = {}
    for src in sorted({i["source"] for i in allit}):
        ex = [i for i in tr if i["source"] == src][:3]
        examples[src] = [{"question": e["question"], "options": e["options"], "gold": e["gold"], "answer": e["options"][e["gold"]],
                          "corpus": e["meta"].get("corpus"), "audio": e["audio"]} for e in ex]
    return {
        "builder": BUILDER, "smoke": smoke, "built_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "build_seconds": round(secs),
        "counts": {"train": len(tr), "CAL": len(cal), "total": len(allit), "cal_frac": round(len(cal) / max(1, len(allit)), 4),
                   "by_part": dict(n_part)},
        "by_source_split": by(lambda i: f"{i['source']}/{i['set']}"),
        "by_source": by(lambda i: i["source"]),
        "by_source_qtype": by(lambda i: f"{i['source']}/{i['meta']['qtype']}"),
        "by_qtype": by(lambda i: i["meta"]["qtype"]),
        "by_corpus_items": by(lambda i: str(i["meta"].get("corpus"))),
        "speech_source_item_share": {k: round(v / tot, 4) for k, v in sorted(speech_items.items(), key=lambda t: -t[1])},
        "by_channel": by(lambda i: i["channel"]),
        "prosody_phone_fx_items": sum(1 for i in allit if i["meta"].get("part") == "prosody" and i["meta"].get("fx")),
        "audio_hours_used": round(hours, 2), "audio_hours_by_part": {k: round(v, 2) for k, v in hours_part.items()},
        "audio_hours_by_corpus": {k: round(v, 2) for k, v in sorted(hours_corpus.items(), key=lambda t: -t[1])},
        "bank_hours": bank_hours, "clips_used": len(used_keys), "render": res,
        "render_errors": sum(1 for f in facts.values() if "error" in f),
        "render_error_examples": [f["error"] for f in facts.values() if "error" in f][:5],
        "licences": LIC, "sources": SOURCE_REVS,
        "musan_licence_report": PP.musan_report, "musan_files_allowed": {k: (len(v["train"]), len(v["CAL"])) for k, v in PP.bg.items()},
        "musan_licences_used": PP.musan_used_licences, "rirs_far_rooms": {k: len(v) for k, v in PP.rir.items()},
        "cal_groups": {**PP.cal_groups, "prosody_and_studio_cal_speakers": sorted(QP.cal_spk)},
        "prosody_records": rec_info, "prosody_stats": dict(QP.stats),
        "prosody_shortfall": {k: v for k, v in QP.shortfall.items() if v > 0},
        "choice_cap": cap_rep, "noul_balance": bal_rep, "validation": val, "leakage": leak, "text_checks": txt, "spot_check": spot,
        "examples": examples,
        "skipped": {
            "VCTK, Vystadial": "not used -> replaced by LibriTTS-R (clean studio base) and HarperValleyBank (real call audio)",
            "AMI": "allowed but not used: headset channels contain crosstalk from other meeting participants, so 'background talker = no' and 'clean base' labels would not be safe",
            "LibriSpeech original audio": "allowed but not on the volume; LibriTTS-R (same readers) already provides clean read speech",
            "number on robocalls / MInDS-14": "no human transcript for robocalls (Whisper metadata never used as labels); MInDS-14 transcripts are non-English (rule is English-only)",
            "loudness absolute (single clip)": "corpora are level-normalised; absolute loudness of a single clip is not a property of the voice -> only relative loudness",
            "cross-speaker loudness": "not asked for the same reason",
            "AMR-NB": "no encoder in the Debian ffmpeg build (as sb_phone)",
            "MInDS-14 English locales": "used by our evaluation set",
            "MUSAN ShareAlike / NC / ND / unlicensed files": "rejected by the per-file licence filter",
        },
    }
