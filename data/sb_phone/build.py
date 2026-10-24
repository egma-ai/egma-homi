"""sb_phone build: seeded plan -> render shards in parallel -> items -> balance -> validation -> leakage check ->
summary.json."""
from __future__ import annotations

import hashlib
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

OUT = Path("/vol/stageb/sb_phone")
RAW = OUT / "work" / "raw"
SR = 16000
BUILDER = "sb_phone"
SHARD_CLIPS = 1100

LIC = {
    "apollo": "CC BY 4.0 — BrunoHays/apollo-telephony-en@b9d3a68a9faa550ae1f2069eb1eb7634cdc8966c (Apollo FSC P4 ASR track 2, 8 kHz telephony)",
    "vystadial": "CC BY-SA 3.0 (ShareAlike, flagged) — BrunoHays/vystadial-telephony-en@b7454512a3547c829be201d9afceb64af8c28e61 (Vystadial 2013 English VoIP, OpenSLR 6)",
    "minds": "CC BY 4.0 — PolyAI/minds14@40ce77cb32a384e4d50a568e1ec39ac804019d33 (non-English locales only)",
    "robocall": "Public domain audio (FTC Project Point of No Entry) — github wspr-ncsu/robocall-audio-dataset@5aa6f3bfa8563ce8c1c75ebf8a2271e6ff6b4272 (repo docs CC BY-ND; README: 'The data itself is in the public domain')",
    "vctk": "CC BY 4.0 — VCTK 0.92 via /vol/stageb/sb_voice bank sb_voice_vctk (read-only reuse)",
    "libritts": "CC BY 4.0 — LibriTTS-R via /vol/stageb/sb_voice bank sb_voice_libritts (read-only reuse)",
    "sb_calls": "Apache-2.0 — Kokoro-82M TTS output + synthetic tones, /vol/stageb/sb_calls banks (read-only reuse)",
    "musan": "CC BY 4.0 — MUSAN (OpenSLR 17); per-file licences filtered to commercially usable ones",
    "rirs": "Apache-2.0 — OpenSLR 28 RIRS_NOISES simulated_rirs",
}
SOURCE_REVS = {
    "apollo": {"repo": "BrunoHays/apollo-telephony-en", "revision": "b9d3a68a9faa550ae1f2069eb1eb7634cdc8966c", "licence": "cc-by-4.0"},
    "vystadial": {"repo": "BrunoHays/vystadial-telephony-en", "revision": "b7454512a3547c829be201d9afceb64af8c28e61", "licence": "cc-by-sa-3.0 (ShareAlike)"},
    "minds14": {"repo": "PolyAI/minds14", "revision": "40ce77cb32a384e4d50a568e1ec39ac804019d33", "licence": "cc-by-4.0", "locales": "cs-CZ de-DE es-ES fr-FR it-IT ko-KR nl-NL pl-PL pt-PT ru-RU zh-CN (no English)"},
    "robocall": {"repo": "github.com/wspr-ncsu/robocall-audio-dataset", "revision": "5aa6f3bfa8563ce8c1c75ebf8a2271e6ff6b4272", "licence": "audio public domain (README); repo text CC BY-ND 4.0"},
    "musan": {"url": "https://www.openslr.org/17/ (musan.tar.gz)", "licence": "CC BY 4.0 (corpus); per-file licence filter, see musan_licence"},
    "rirs": {"url": "https://www.openslr.org/28/ (rirs_noises.zip, simulated_rirs only)", "licence": "Apache-2.0"},
    "sb_voice": {"path": "/vol/stageb/sb_voice/audio (vctk, libritts banks; read-only)", "licence": "CC BY 4.0"},
    "sb_calls": {"path": "/vol/stageb/sb_calls (banks + items; read-only)", "licence": "Apache-2.0 (Kokoro TTS output)"},
}


def h01(s: str) -> float:
    return int(hashlib.md5(s.encode()).hexdigest()[:12], 16) / float(16 ** 12)


class NoShuffle(random.Random):
    def shuffle(self, x, *a, **k):  # keep ordered levels in natural order
        return None


def read_jsonl(p) -> list[dict]:
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def bank_index(d: Path, name: str) -> dict:
    return json.loads((d / f"{name}.json").read_text())


# MUSAN licences
_BAD_LIC = re.compile(r"non-?commercial|noncommercial|\bnc\b|no-?deriv|noderiv|\bnd\b|all rights reserved|sampling", re.I)
_GOOD_LIC = re.compile(r"cc by|cc0|attribution|public domain", re.I)
_LIC_LINE = re.compile(r"cc by|cc0|attribution|public domain|licen[cs]e|sampling|non-?commercial|creative commons", re.I)
_MUSAN_ID = re.compile(r"^\s*((?:music|noise|speech)-[a-z-]+-\d{4})\s*$")


def musan_licences(texts: dict) -> dict:
    """Parse MUSAN's per-directory LICENSE files -> {dir: {"dir_licence": str|None, "files": {stem: licence}}}.
    Blocks are separated by '=====' lines; a block's ids get the licence line(s) of that block, or the directory
    header licence (a block with a licence but no ids, e.g. music/rfm) when the block has none. A LICENSE file with
    no ids at all (librivox, us-gov, free-sound) is a whole-directory licence."""
    out = {}
    for path, txt in texts.items():
        if not path.endswith("/LICENSE"):
            continue
        d = path.rsplit("/", 1)[0]  # musan/music/fma
        blocks = re.split(r"\n\s*={10,}\s*\n?", txt)
        header, files = None, {}
        parsed = []
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
    return bool(lic) and bool(_GOOD_LIC.search(lic)) and not _BAD_LIC.search(lic)


def musan_allowed(texts: dict, metas: dict | None = None) -> tuple[dict, dict]:
    """{stem: licence} for every MUSAN file whose licence is commercially usable (CC BY / CC BY-SA / CC0 / public
    domain; NC, ND and Sampling+ rejected; files without a licence line rejected) + a report."""
    lic = musan_licences(texts)
    allowed, rep = {}, Counter()
    stems = {}
    for bank_meta in (metas or {}).values():
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


# Plan
def load_sources(smoke: bool) -> dict:
    S = {}
    for name in ("apollo", "vystadial", "minds", "robocall", "musan_noise", "musan_music", "musan_speech", "rirs"):
        S[name] = {"meta": json.loads((RAW / f"{name}_meta.json").read_text()), "index": bank_index(RAW, name)}
    S["musan_texts"] = json.loads((RAW / "musan_texts.json").read_text())
    # sb_voice clean speech: keys used only by measured-prosody families (clean clips), never by noise items
    vdir = Path("/vol/stageb/sb_voice")
    vit = read_jsonl(vdir / "items_train.jsonl") + read_jsonl(vdir / "items_cal.jsonl")
    noisy = {(i["audio"]["bank"], i["audio"]["key"]) for i in vit if i.get("audio") and i["source"] == "sb_voice_background_noise"}
    clean = {}
    for i in vit:
        a = i.get("audio")
        if not a or a["bank"] not in ("sb_voice_vctk", "sb_voice_libritts"):
            continue
        if i["source"] not in ("sb_voice_rate", "sb_voice_pitch", "sb_voice_loudness", "sb_voice_pauses",
                               "sb_voice_pitch_variation", "sb_voice_gender"):
            continue
        k = (a["bank"], a["key"])
        if k in noisy or i.get("channel") != "clean":
            continue
        clean[k] = {"speaker": i["meta"].get("speaker"), "split": i["set"], "corpus": i["meta"].get("corpus"),
                    "source_id": i["meta"].get("source_id")}
    S["voice_clean"] = clean
    S["voice_index"] = {b: bank_index(vdir / "audio", b) for b in ("sb_voice_vctk", "sb_voice_libritts")}
    cdir = Path("/vol/stageb/sb_calls")
    S["calls_items"] = read_jsonl(cdir / "items_train.jsonl") + read_jsonl(cdir / "items_cal.jsonl")
    S["calls_index"] = {b: bank_index(cdir / "audio", b) for b in
                        [f"sb_calls_train_{i}" for i in range(4)] + ["sb_calls_cal"]}
    return S


BANK_DIRS = {
    "raw": str(RAW), "sb_voice": "/vol/stageb/sb_voice/audio", "sb_calls": "/vol/stageb/sb_calls/audio",
}


class Planner:
    def __init__(self, S: dict, smoke: bool):
        self.S = S
        self.smoke = smoke
        self.rng = random.Random(f"{C.SEED}:sb_phone:plan")
        self.clips: list[dict] = []
        self.scale = 0.04 if smoke else 1.0
        allowed, self.musan_report = musan_allowed(S["musan_texts"], {b: S[b]["meta"] for b in
                                                                       ("musan_noise", "musan_music", "musan_speech")})
        self.musan_allowed = allowed
        self.bg = {}
        for kind, bank in (("noise", "musan_noise"), ("music", "musan_music"), ("talker", "musan_speech")):
            keys = sorted(k for k, (s, n) in S[bank]["index"].items() if n >= 3 * SR and k in allowed)
            self.bg[kind] = {"train": [k for k in keys if h01("bg:" + k) >= 0.15],
                             "CAL": [k for k in keys if h01("bg:" + k) < 0.15], "bank": bank}
        rm = S["rirs"]["meta"]
        # far-field / speakerphone: direct-to-reverberant ratio <= -2 dB and RT60 >= 0.3 s (clearly distant + echoey)
        far = sorted(k for k, m in rm.items() if m["drr_db"] is not None and m["drr_db"] <= -2.0
                     and (m.get("rt60_s") or 0) >= 0.3)
        self.rir = {"train": [k for k in far if h01("room:" + rm[k]["room"]) >= 0.15],
                    "CAL": [k for k in far if h01("room:" + rm[k]["room"]) < 0.15]}

    def n(self, x: int) -> int:
        return max(4, int(round(x * self.scale)))

    # Shared helpers
    def recipe(self, split: str, base_clean: bool, rng: random.Random, scenario: str | None = None) -> dict:
        w = L.SCENARIO_W_CLEAN if base_clean else L.SCENARIO_W_REAL
        if scenario is None:
            scenario = rng.choices(list(w), weights=list(w.values()))[0]
        r = L.sample_recipe("clean" if scenario == "orig" else scenario, rng, p_codec_none=0.3 if base_clean else 0.5)
        r["scenario"] = scenario
        bg = {}
        for kind in ("noise", "music", "talker"):
            if r.get(kind):
                pool = self.bg[kind][split]
                bg[kind] = [self.bg[kind]["bank"], rng.choice(pool)]
        if r.get("rir"):
            k = rng.choice(self.rir[split])
            bg["rir"] = ["rirs", k]
            r["rir_drr_db"] = self.S["rirs"]["meta"][k]["drr_db"]
            r["rir_rt60_s"] = self.S["rirs"]["meta"][k]["rt60_s"]
            r["rir_room"] = self.S["rirs"]["meta"][k]["room"]
        return r, bg

    def add(self, *, key, split, corpus, segs, recipe, bg, base_clean, questions, meta, crop=None):
        self.clips.append({"key": key, "split": split, "corpus": corpus, "segs": segs, "recipe": recipe, "bg": bg,
                           "base_clean": base_clean, "questions": questions, "meta": meta, "crop": crop})

    # Which added impairments may be asked on a real recording. A "no" answer is only safe when the original
    # recording cannot already contain that thing: robocalls often have music beds / several voices, and Apollo
    # mission-control loops have their own dropouts and background voices -> those questions are not asked there.
    ROBUST_BY_CORPUS = {"vystadial": ("breakup", "music", "talker"), "minds14": ("breakup", "music", "talker"),
                        "apollo": ("music",), "robocall": ("breakup",)}

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

    # Real phone recordings
    def plan_vystadial(self, n_clips: int):
        rng = random.Random(f"{C.SEED}:plan:vys")
        meta = self.S["vystadial"]["meta"]
        by_spk = defaultdict(list)
        for k, m in meta.items():
            by_spk[m["spk"]].append(k)
        spks = sorted(by_spk)
        # CAL speakers: hash order, each < 6 % of the clips, until ~3-4 % of all clips are CAL-only
        tot_n = sum(len(v) for v in by_spk.values())
        cal_spk, acc = set(), 0
        for s in sorted(spks, key=lambda s: h01("vys_spk:" + s)):
            if len(by_spk[s]) < 0.06 * tot_n and acc + len(by_spk[s]) <= 0.045 * tot_n:
                cal_spk.add(s)
                acc += len(by_spk[s])
            if acc >= 0.03 * tot_n:
                break
        self.cal_groups["vystadial_spk"] = sorted(cal_spk)
        pool = []
        for spk in spks:
            ks = sorted(by_spk[spk], key=lambda k: meta[k]["session"])
            i = 0
            while i < len(ks):
                m = rng.choice([1, 1, 2, 2, 3])
                pool.append((spk, ks[i: i + m]))
                i += m
        pool.sort(key=lambda p: h01("vys_pool:" + p[1][0]))
        n_cal = max(2, int(0.03 * n_clips))
        cal = [p for p in pool if p[0] in cal_spk][:n_cal]
        train = [p for p in pool if p[0] not in cal_spk][: n_clips - len(cal)]
        for split, lst in (("CAL", cal), ("train", train)):
            for spk, ks in lst:
                text = " ".join(meta[k]["text"] for k in ks)
                r, bg = self.recipe(split, False, rng)
                qs = []
                if rng.random() < 0.5:
                    qs += live_views(rng, True, 1)
                num = L.number_label(text)
                if num is not None and (num or rng.random() < 0.08):
                    qs.append(("number", num))
                if rng.random() < 0.12:
                    qs.append(("lang", "en"))
                qs += self.robust_q(r, rng, corpus="vystadial")
                segs = [["raw", "vystadial", k] for k in ks]
                self.add(key=f"vys_{split}_{ks[0][4:]}", split=split, corpus="vystadial", segs=segs, recipe=r, bg=bg,
                         base_clean=False, questions=qs[:4],
                         meta={"speaker": f"vystadial:{spk}", "source_id": [meta[k]["audio_ids"] for k in ks],
                               "transcript": text, "licence": LIC["vystadial"]})

    def plan_minds(self, n_clips: int):
        rng = random.Random(f"{C.SEED}:plan:minds")
        meta = self.S["minds"]["meta"]
        by_loc = defaultdict(list)
        for k, m in meta.items():
            by_loc[m["locale"]].append(k)
        per = n_clips // len(by_loc)
        for loc in sorted(by_loc):
            ks = sorted(by_loc[loc], key=lambda k: h01("minds:" + k))[:per]
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
        rng = random.Random(f"{C.SEED}:plan:apollo")
        meta = self.S["apollo"]["meta"]
        tr = sorted([k for k, m in meta.items() if m["split"] == "train"], key=lambda k: h01("ap:" + k))
        dv = sorted([k for k, m in meta.items() if m["split"] == "dev"], key=lambda k: h01("ap:" + k))
        n_cal = max(2, int(0.03 * n_clips))
        for split, ks in (("CAL", dv[:n_cal]), ("train", tr[: n_clips - n_cal])):
            for k in ks:
                text = meta[k]["text"]
                r, bg = self.recipe(split, False, rng)
                qs = []
                num = L.number_label(text)
                if num is not None:
                    qs.append(("number", num))
                if rng.random() < 0.08:
                    qs.append(("lang", "en"))
                if rng.random() < 0.15:
                    qs += live_views(rng, True, 1)
                qs += self.robust_q(r, rng, corpus="apollo")
                self.add(key=f"apollo_{split}_{k}", split=split, corpus="apollo", segs=[["raw", "apollo", k]], recipe=r,
                         bg=bg, base_clean=False, questions=qs[:4],
                         meta={"speaker": f"apollo:{meta[k]['audio_id']}", "source_id": meta[k]["audio_id"],
                               "apollo_split": meta[k]["split"], "transcript": text, "licence": LIC["apollo"]})

    def plan_robocall(self, max_windows: int = 2):
        rng = random.Random(f"{C.SEED}:plan:robo")
        meta = self.S["robocall"]["meta"]
        idx = self.S["robocall"]["index"]
        cases = Counter(m["case"] for m in meta.values())
        # CAL = whole cease-and-desist cases (campaign / carrier), ~3-5 % of the calls
        order = sorted(cases, key=lambda c: h01("robo_case:" + c))
        cal_cases, tot = set(), 0
        for c in order:
            if tot + cases[c] <= 0.05 * len(meta) and cases[c] < 0.04 * len(meta):
                cal_cases.add(c)
                tot += cases[c]
            if tot >= 0.025 * len(meta):
                break
        self.cal_groups["robocall_cases"] = sorted(cal_cases)
        keys = sorted(meta, key=lambda k: h01("robo:" + k))
        if self.smoke:
            keys = keys[: self.n(len(keys))]
        for k in keys:
            m = meta[k]
            split = "CAL" if m["case"] in cal_cases else "train"
            dur = idx[k][1] / SR
            wins = []
            w1 = min(dur, rng.uniform(8.0, 25.0))
            wins.append((0.0, w1))
            rest = dur - w1 - 0.5
            if max_windows > 1 and rest >= 6.0:
                w2 = min(rest, rng.uniform(8.0, 25.0))
                st = w1 + 0.5 + rng.uniform(0, max(0.0, rest - w2))
                wins.append((round(st, 2), w2))
            for wi, (st, wl) in enumerate(wins):
                r, bg = self.recipe(split, False, rng)
                qs = live_views(rng, False, 2)
                qs += self.robust_q(r, rng, n=2, corpus="robocall")
                self.add(key=f"robo_{split}_{k[5:]}_w{wi}", split=split, corpus="robocall", segs=[["raw", "robocall", k]],
                         crop=[st, wl], recipe=r, bg=bg, base_clean=False, questions=qs,
                         meta={"speaker": f"robocall_case:{m['case']}", "source_id": m["file"], "case": m["case"],
                               "window_s": [round(st, 2), round(wl, 2)], "licence": LIC["robocall"]})

    # Clean speech, labels by construction
    def plan_studio(self, n_clips: int):
        rng = random.Random(f"{C.SEED}:plan:studio")
        by_spk = defaultdict(list)
        for (bank, key), m in self.S["voice_clean"].items():
            by_spk[(m["speaker"], m["split"])].append((bank, key))
        n_cal = max(4, int(0.03 * n_clips))
        cal_spk = sorted(s for s in by_spk if s[1] == "CAL")
        tr_spk = sorted(s for s in by_spk if s[1] == "train")
        groups = {"CAL": [], "train": []}
        for split, spks in (("CAL", cal_spk), ("train", tr_spk)):
            for s in spks:
                ks = sorted(by_spk[s], key=lambda bk: h01("st:" + bk[1]))
                vidx = self.S["voice_index"]
                i = 0
                while i < len(ks):
                    m = rng.choice([1, 2, 2, 3])
                    g = ks[i: i + m]
                    i += m
                    # keep adding utterances of the same speaker until the clip is >= 3 s (max 20 s)
                    while i < len(ks) and sum(vidx[b][k][1] for b, k in g) < 3 * SR:
                        g.append(ks[i])
                        i += 1
                    tot = sum(vidx[b][k][1] for b, k in g)
                    if 2 * SR <= tot <= 20 * SR:
                        groups[split].append((s[0], g))
            groups[split].sort(key=lambda g: h01("stg:" + g[1][0][1]))
        sel = {"CAL": groups["CAL"][:n_cal], "train": groups["train"][: n_clips - n_cal]}
        for split, lst in sel.items():
            for spk, bks in lst:
                r, bg = self.recipe(split, True, rng)
                qs = self.studio_questions(r, rng)
                corpus = self.S["voice_clean"][bks[0]]["corpus"]
                segs = [["sb_voice", b, k] for b, k in bks]
                self.add(key=f"studio_{split}_{bks[0][1].replace('/', '_')}", split=split, corpus=corpus, segs=segs,
                         recipe=r, bg=bg, base_clean=True, questions=qs,
                         meta={"speaker": spk, "source_id": [self.S["voice_clean"][bk]["source_id"] for bk in bks],
                               "licence": LIC.get(corpus, LIC["vctk"])})

    def studio_questions(self, r: dict, rng: random.Random, n: int = 3) -> list:
        app = L.applicable(r, base_clean=True)
        sc = r["scenario"]
        rel = {"noise_mod": "noise", "noise_sev": "noise", "drop_mod": "breakup", "drop_sev": "breakup",
               "music": "music", "talker": "talker", "far": "far", "clip_sev": "poor", "combo_sev": "poor",
               "lowvol": "quality", "clip_mod": "quality", "clean": None}.get(sc)
        out = []
        if "quality" in app and rng.random() < 0.55:
            out.append(("quality", app["quality"]))
        if rel and rel in app and rel not in [q for q, _ in out]:
            out.append((rel, app[rel]))
        if "poor" in app and not any(q in ("poor", "voice_clear") for q, _ in out) and rng.random() < 0.8:
            k = rng.choice(["poor", "voice_clear"])
            out.append((k, app[k]))
        others = [k for k in ("poor", "voice_clear", "noise", "breakup", "music", "talker", "far") if k in app
                  and k not in [q for q, _ in out]]
        rng.shuffle(others)
        for k in others:
            if len(out) >= n:
                break
            out.append((k, app[k]))
        return out[:n]

    # sb_calls clips, re-degraded, same call labels
    def plan_calls(self, n_clips: int):
        rng = random.Random(f"{C.SEED}:plan:calls")
        by_clip = defaultdict(list)
        for it in self.S["calls_items"]:
            by_clip[(it["audio"]["bank"], it["audio"]["key"])].append(it)
        cidx = self.S["calls_index"]
        keys = sorted((bk for bk in by_clip if cidx[bk[0]][bk[1]][1] >= 2 * SR), key=lambda bk: h01("calls:" + bk[1]))
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
                n_orig = 2 if rng.random() < 0.4 else 1
                qs = [("calls_item", i["id"]) for i in its_sorted[:n_orig]]
                cls = m0.get("class")
                if cls in LIVE_CLASSES and rng.random() < 0.3:
                    qs += live_views(rng, True, 1)
                elif cls in AUTO_CLASSES and rng.random() < 0.4:
                    qs += live_views(rng, False, 1)
                if base_clean:
                    qs += self.studio_questions(r, rng, n=4 - len(qs))
                else:
                    qs += self.robust_q(r, rng, n=1)
                self.add(key=f"calls_{split}_{bk[1]}", split=split, corpus="sb_calls", segs=[["sb_calls", bk[0], bk[1]]],
                         recipe=r, bg=bg, base_clean=base_clean, questions=qs[:4],
                         meta={"speaker": f"kokoro:{m0.get('voice')}", "source_id": f"{bk[0]}/{bk[1]}",
                               "call_class": cls, "licence": LIC["sb_calls"]})

    def build(self, sizes: dict) -> list[dict]:
        self.cal_groups = {}
        self.plan_vystadial(self.n(sizes["vystadial"]))
        self.plan_minds(self.n(sizes["minds"]))
        self.plan_apollo(self.n(sizes["apollo"]))
        self.plan_robocall()
        self.plan_studio(self.n(sizes["studio"]))
        self.plan_calls(self.n(sizes["calls"]))
        return self.clips


LIVE_CLASSES = {"live_person"}
AUTO_CLASSES = {"personal_voicemail", "carrier_voicemail", "ivr", "call_screening", "business_ivr", "screening",
                "personal_vm", "carrier_vm"}


def live_views(rng: random.Random, is_live: bool, n: int) -> list:
    """n distinct views of the live-person-vs-recording label: yes/no 'live?', yes/no 'automated?', or a choice."""
    views = rng.sample(["live_noul", "auto_noul", "live_choice"], n)
    return [(v, is_live) for v in views]


SIZES = {"vystadial": 3300, "minds": 3100, "apollo": 4000, "studio": 6000, "calls": 3500}


# Render (workers)
_BANKS: dict = {}


def _bank(where: str, name: str):
    k = (where, name)
    if k not in _BANKS:
        _BANKS[k] = C.AudioBank(BANK_DIRS[where], name)
    return _BANKS[k]


def base_audio(spec: dict, rng: random.Random) -> np.ndarray:
    xs = []
    for where, bank, key in spec["segs"]:
        x = _bank(where, bank).get(key)
        if spec.get("crop"):
            st, wl = spec["crop"]
            x = x[int(st * SR): int((st + wl) * SR)]
        if xs:
            xs.append(np.zeros(int(rng.uniform(0.3, 0.8) * SR), np.float32))
        xs.append(x)
    y = np.concatenate(xs).astype(np.float32)
    return y[: 30 * SR]


def render_one(spec: dict):
    import fx

    rng = random.Random(f"{C.SEED}:render:{spec['key']}")
    try:
        x = base_audio(spec, rng)
        bgs = {}
        for kind, (bank, key) in spec["bg"].items():
            bgs[kind] = _bank("raw", bank).get(key)
        y, facts = fx.render(x, spec["recipe"], bgs, rng)
        if not np.isfinite(y).all():
            raise ValueError("non-finite audio")
        facts["duration_s"] = round(len(y) / SR, 3)
        return spec["key"], y, facts
    except Exception as e:  # recorded; the clip is dropped
        return spec["key"], None, {"error": repr(e)[:300]}


def render_shard(bank_name: str, specs: list[dict], out_audio: str, n_proc: int = 4) -> dict:
    from multiprocessing import Pool

    t0 = time.time()
    clips, facts = {}, {}
    with Pool(n_proc) as pool:
        for key, y, f in pool.imap(render_one, specs, chunksize=8):
            facts[key] = f
            if y is not None:
                clips[key] = y
    C.write_audio_bank(out_audio, bank_name, clips)
    fdir = Path(out_audio).parent / "work" / "facts"
    fdir.mkdir(parents=True, exist_ok=True)
    (fdir / f"{bank_name}.json").write_text(json.dumps(facts))
    return {"bank": bank_name, "n": len(clips), "errors": sum(1 for f in facts.values() if "error" in f),
            "seconds": round(time.time() - t0, 1), "audio_s": round(sum(len(v) for v in clips.values()) / SR, 1)}


# Items
def _noul(items, rng, *, iid, split, source, question, yes, audio, channel, meta):
    target = [1.0, 0.0] if yes else [0.0, 1.0]
    items.append(C.make_item(item_id=iid, set_name=split, source=source, question=question, options=["yes", "no"],
                             target=target, gold=0 if yes else 1, rng=rng, audio=audio, channel=channel,
                             meta={**meta, "qtype": "noul", "answer": "yes" if yes else "no"}))


def make_items(clips: list[dict], facts: dict, bank_of: dict, calls_by_id: dict) -> list[dict]:
    items = []
    lang_all = list(L.LANG_NAMES)
    for c in clips:
        f = facts.get(c["key"], {})
        if "error" in f or c["key"] not in bank_of:
            continue
        rng = random.Random(f"{C.SEED}:items:{c['key']}")
        audio = {"bank": bank_of[c["key"]], "key": c["key"]}
        r = c["recipe"]
        base_meta = {"clip": c["key"], "corpus": c["corpus"], **c["meta"],
                     "recipe": {k: v for k, v in r.items()}, "render": f, "background": c["bg"],
                     "label_rule": None}
        used = 0
        for qi, (qname, ans) in enumerate(c["questions"]):
            if used >= 4:
                break
            channel = "phone" if rng.random() < 0.6 else "clean"
            iid = f"{BUILDER}/{c['split']}/{c['key']}/{qi}_{qname}"
            meta = {**base_meta, "qname": qname}
            split = c["split"]
            if qname == "breakup":
                lf = f.get("lost_frac", 0.0)
                if ans and lf < 0.03:  # realised loss too small to be sure -> skip
                    continue
                meta["label_rule"] = "yes = we zeroed 20-60 ms chunks covering >= 4 % of the speech time (packet loss); no = no dropouts added"
                _noul(items, rng, iid=iid, split=split, source="sb_phone_breakup", question=L.qtext("breakup_noul", rng),
                      yes=ans, audio=audio, channel=channel, meta=meta)
            elif qname in ("music", "talker", "noise", "far", "poor", "voice_clear"):
                rules = {
                    "music": "yes = MUSAN music mixed in at 0-12 dB speech-to-music ratio; no = no music added",
                    "talker": "yes = MUSAN speech (another person) mixed in at 6-14 dB below the main speaker; no = none added",
                    "noise": "yes = MUSAN noise mixed in (SNR -3..20 dB) into clean studio/TTS speech; no = clean base and no noise/music/talker added",
                    "far": "yes = convolved with a simulated room impulse response (OpenSLR 28) with direct-to-reverberant ratio <= -2 dB and RT60 >= 0.3 s; no = no RIR (close-talk)",
                    "poor": "yes = quality level 'very poor' by construction; no = 'clear' (no added impairment; normal phone codecs allowed)",
                    "voice_clear": "yes = quality level 'clear' by construction; no = 'very poor'",
                }
                meta["label_rule"] = rules[qname]
                _noul(items, rng, iid=iid, split=split, source=f"sb_phone_{qname}", question=L.qtext(f"{qname}_noul", rng),
                      yes=ans, audio=audio, channel=channel, meta=meta)
            elif qname == "quality":
                meta["label_rule"] = ("by construction: clear = no added impairment; some problems = exactly one moderate "
                                      "impairment (noise SNR 12-20 dB / 4-9 % dropouts / 2-5 % clipping / quiet -34..-42 dBFS "
                                      "/ far-field echo); very poor = a severe one (SNR <= 4 dB / 15-30 % dropouts / 10-25 % clipping)")
                lv = L.LEVELS
                opt_map = L.QUALITY_OPTIONS if rng.random() < 0.6 else L.QUALITY_OPTIONS_ALT
                opts = [opt_map[x] for x in lv]
                gold = lv.index(ans)
                tgt = [1.0 if i == gold else 0.0 for i in range(3)]
                if rng.random() < 0.6:
                    items.append(C.make_item(item_id=iid, set_name=split, source="sb_phone_quality",
                                             question=L.qtext("quality_score", rng), options=opts, target=tgt, gold=gold,
                                             rng=NoShuffle(0), audio=audio, channel=channel,
                                             meta={**meta, "qtype": "score", "answer": ans}))
                else:
                    items.append(C.make_item(item_id=iid, set_name=split, source="sb_phone_quality",
                                             question=L.qtext("quality_choice", rng), options=opts, target=tgt, gold=gold,
                                             rng=rng, audio=audio, channel=channel,
                                             meta={**meta, "qtype": "choice", "answer": ans}))
            elif qname in ("live_noul", "auto_noul", "live_choice"):
                meta["label_rule"] = ("live = real human speech recorded on a phone line (Vystadial VoIP callers, MInDS-14 "
                                      "callers) or a TTS 'live person' clip from sb_calls; automated/recorded = FTC robocall "
                                      "recordings, sb_calls voicemail/IVR/screening greetings")
                is_live = bool(ans)
                if qname == "live_choice":
                    o = rng.choice(L.LIVE_OPTIONS)
                    gold = 0 if is_live else 1
                    items.append(C.make_item(item_id=iid, set_name=split, source="sb_phone_live",
                                             question=L.qtext("live_choice", rng), options=list(o),
                                             target=[1.0, 0.0] if is_live else [0.0, 1.0], gold=gold, rng=rng,
                                             audio=audio, channel=channel,
                                             meta={**meta, "qtype": "choice", "answer": o[gold]}))
                elif qname == "live_noul":
                    _noul(items, rng, iid=iid, split=split, source="sb_phone_live", question=L.qtext("live_noul", rng),
                          yes=is_live, audio=audio, channel=channel, meta=meta)
                else:
                    _noul(items, rng, iid=iid, split=split, source="sb_phone_live", question=L.qtext("auto_noul", rng),
                          yes=not is_live, audio=audio, channel=channel, meta=meta)
            elif qname == "lang":
                meta["label_rule"] = "language = MInDS-14 locale of the recording (English = Vystadial / Apollo English corpora)"
                gold_lang = ans
                if rng.random() < 0.6:
                    k = len(lang_all) if rng.random() < 0.4 else rng.randint(4, 6)
                    others = [x for x in lang_all if x != gold_lang]
                    rng.shuffle(others)
                    opts_codes = [gold_lang] + others[: k - 1]
                    opts = [L.LANG_NAMES[x] for x in opts_codes]
                    items.append(C.make_item(item_id=iid, set_name=split, source="sb_phone_language",
                                             question=L.qtext("lang_choice", rng), options=opts,
                                             target=[1.0] + [0.0] * (len(opts) - 1), gold=0, rng=rng, audio=audio,
                                             channel=channel, meta={**meta, "qtype": "choice", "answer": L.LANG_NAMES[gold_lang]}))
                else:
                    yes = rng.random() < 0.5
                    asked = gold_lang if yes else rng.choice([x for x in lang_all if x != gold_lang])
                    _noul(items, rng, iid=iid, split=split, source="sb_phone_language",
                          question=L.qtext("lang_noul", rng, lang=L.LANG_NAMES[asked]), yes=yes, audio=audio,
                          channel=channel, meta={**meta, "asked_language": L.LANG_NAMES[asked]})
            elif qname in ("number", "date"):
                meta["label_rule"] = {"number": "from the human transcript: digits or number words (a lone 'one'/'oh' is ambiguous and never asked)",
                                      "date": "from the human transcript: a month name or weekday = yes; no date/time words at all = no"}[qname]
                _noul(items, rng, iid=iid, split=split, source=f"sb_phone_{qname}", question=L.qtext(f"{qname}_noul", rng),
                      yes=ans, audio=audio, channel=channel, meta=meta)
            elif qname == "calls_item":
                src = calls_by_id.get(ans)
                if src is None:
                    continue
                it = dict(src)
                it_meta = {**base_meta, "qname": "calls_" + str(src["meta"].get("qname")), "qtype": src["meta"].get("qtype", "noul"),
                           "sb_calls_item": src["id"], "call_class": src["meta"].get("class"),
                           "answer": src["meta"].get("answer"),
                           "label_rule": "label copied from the sb_calls item (by construction); audio re-degraded here"}
                items.append({"id": iid, "set": split, "source": "sb_phone_calls", "question": it["question"],
                              "options": list(it["options"]), "target": list(it["target"]), "gold": it["gold"],
                              "audio": audio, "channel": channel, "state_text": None, "meta": it_meta})
            else:
                continue
            used += 1
    return items


def balance(items: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
    """Every noul question family (source + qname) exactly 50/50 yes/no per split: drop random majority items."""
    groups = defaultdict(list)
    keep = []
    for it in items:
        if it["meta"]["qtype"] == "noul":
            groups[(it["set"], it["source"], it["meta"]["qname"])].append(it)
        else:
            keep.append(it)
    rep = {}
    for g, lst in sorted(groups.items()):
        yes = [i for i in lst if i["options"][i["gold"]] == "yes"]
        no = [i for i in lst if i["options"][i["gold"]] == "no"]
        n = min(len(yes), len(no))
        rng.shuffle(yes)
        rng.shuffle(no)
        keep += yes[:n] + no[:n]
        rep["/".join(g)] = {"yes": len(yes), "no": len(no), "kept_each": n}
    return keep, rep


# Main
def main(smoke: bool, render_map, commit, reload) -> dict:
    t0 = time.time()
    out = OUT / "work" / "smoke" if smoke else OUT
    S = load_sources(smoke)
    P = Planner(S, smoke)
    clips = P.build(SIZES)
    print(f"[plan] {len(clips)} clips {Counter((c['corpus'], c['split']) for c in clips)}", flush=True)
    # shards -> banks: CAL clips in their own bank(s)
    shards = []
    for split in ("train", "CAL"):
        lst = [c for c in clips if c["split"] == split]
        random.Random(f"{C.SEED}:shard:{split}").shuffle(lst)
        nb = max(1, math.ceil(len(lst) / SHARD_CLIPS))
        for b in range(nb):
            name = f"{BUILDER}_{'cal' if split == 'CAL' else 'train'}_{b:02d}"
            sub = lst[b::nb]
            shards.append((name, [{k: c[k] for k in ("key", "segs", "crop", "recipe", "bg")} for c in sub]))
    bank_of = {s["key"]: name for name, specs in shards for s in specs}
    (out / "work").mkdir(parents=True, exist_ok=True)
    (out / "work" / "plan.json").write_text(json.dumps({"clips": clips, "cal_groups": P.cal_groups}))
    commit()
    res = list(render_map([(name, specs, str(out / "audio")) for name, specs in shards]))
    print(f"[render] {res}", flush=True)
    reload()
    facts = {}
    for name, _ in shards:
        p = out / "work" / "facts" / f"{name}.json"
        if p.exists():
            facts.update(json.loads(p.read_text()))
    calls_by_id = {i["id"]: i for i in S["calls_items"]}
    items = make_items(clips, facts, bank_of, calls_by_id)
    items, bal = balance(items, random.Random(f"{C.SEED}:balance"))
    items.sort(key=lambda i: h01("order:" + i["id"]))
    tr = [i for i in items if i["set"] == "train"]
    cal = [i for i in items if i["set"] == "CAL"]
    C.write_jsonl(out / "items_train.jsonl", tr)
    C.write_jsonl(out / "items_cal.jsonl", cal)
    commit()
    val = validate(out, tr + cal, [n for n, _ in shards])
    leak = leakage_check(tr + cal, clips)
    summ = make_summary(out, tr, cal, res, bal, val, leak, P, clips, facts, time.time() - t0)
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    commit()
    return {"n_train": len(tr), "n_cal": len(cal), "validation_ok": val["ok"], "leak": leak.get("ok"),
            "seconds": round(time.time() - t0)}


# Checks
def validate(out: Path, items: list[dict], banks: list[str]) -> dict:
    idx = {}
    for b in banks:
        p = out / "audio" / f"{b}.json"
        if p.exists():
            idx[b] = json.loads(p.read_text())
    errs = Counter()
    durs = []
    per_clip = Counter()
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
        if int(np.argmax(it["target"])) != it["gold"]:
            errs["gold_argmax"] += 1
        if it["meta"].get("qtype") not in ("choice", "noul", "score"):
            errs["qtype"] += 1
        if it["meta"]["qtype"] == "noul" and sorted(it["options"]) != ["no", "yes"]:
            errs["noul_options"] += 1
        if it["meta"]["qtype"] == "score" and it["options"] not in ([L.QUALITY_OPTIONS[x] for x in L.LEVELS],
                                                                    [L.QUALITY_OPTIONS_ALT[x] for x in L.LEVELS]):
            errs["score_order"] += 1
        if len(set(it["options"])) != len(it["options"]):
            errs["dup_options"] += 1
        if it["channel"] not in ("clean", "phone"):
            errs["channel"] += 1
        a = it["audio"]
        if a["bank"] not in idx or a["key"] not in idx[a["bank"]]:
            errs["audio_key_missing"] += 1
        else:
            durs.append(idx[a["bank"]][a["key"]][1] / SR)
        per_clip[a["key"]] += 1
        if not it["id"].startswith(BUILDER):
            errs["id_prefix"] += 1
    errs["dup_ids"] = sum(1 for v in ids.values() if v > 1)
    errs["clips_over_4_items"] = sum(1 for v in per_clip.values() if v > 4)
    # train/CAL group disjointness
    grp = defaultdict(set)
    for it in items:
        grp[it["set"]].add(it["meta"].get("speaker"))
    overlap = grp["train"] & grp["CAL"]
    errs["train_cal_speaker_overlap"] = len(overlap)
    # actual audio sanity on a sample
    samp = random.Random(1).sample(items, min(300, len(items)))
    bad_audio = 0
    for it in samp:
        a = it["audio"]
        try:
            x = C.AudioBank(out / "audio", a["bank"]).get(a["key"])
            if not np.isfinite(x).all() or np.max(np.abs(x)) > 1.0 or len(x) < 1.5 * SR:
                bad_audio += 1
        except Exception:
            bad_audio += 1
    errs["bad_audio_sample"] = bad_audio
    bal = defaultdict(Counter)
    for it in items:
        if it["meta"]["qtype"] == "noul":
            bal[f"{it['set']}/{it['source']}"][it["options"][it["gold"]]] += 1
    d = np.array(durs) if durs else np.zeros(1)
    errs = {k: v for k, v in errs.items() if v}
    return {"ok": not errs, "errors": dict(errs), "n_items": len(items),
            "duration_s": {"min": round(float(d.min()), 2), "mean": round(float(d.mean()), 2), "max": round(float(d.max()), 2)},
            "noul_balance": {k: dict(v) for k, v in sorted(bal.items())},
            "max_items_per_clip": max(per_clip.values()) if per_clip else 0,
            "train_cal_overlap_examples": sorted(overlap)[:5]}


_MINDS_PATH = re.compile(r"[a-z]{2}-[A-Z]{2}~[A-Z_]+/[A-Za-z0-9_]+\.wav")


def leakage_check(items: list[dict], clips: list[dict]) -> dict:
    """Overlap check against the evaluation items (/vol/adi/items.jsonl, /vol/data/eval_items.jsonl): no shared bank,
    MInDS-14 path, speaker or source id."""
    rep = {}
    our_banks = {i["audio"]["bank"] for i in items}
    our_minds_paths = {c["meta"]["source_id"] for c in clips if c["corpus"] == "minds14"}
    our_ids = set()
    for c in clips:
        sid = c["meta"].get("source_id")
        for s in (sid if isinstance(sid, list) else [sid]):
            if isinstance(s, list):
                our_ids.update(map(str, s))
            elif s:
                our_ids.add(str(s))
    our_ids |= {Path(s).stem for s in list(our_ids)}
    our_ids -= {Path(p).stem for p in our_minds_paths}  # 'response_N' stems are not unique across locales
    our_spk = {c["meta"].get("speaker") for c in clips}
    ok = True
    for name, path in (("adi", "/vol/adi/items.jsonl"), ("exp0_eval", "/vol/data/eval_items.jsonl")):
        p = Path(path)
        if not p.exists():
            rep[name] = "missing"
            continue
        n, bank_hits, key_hits, minds_hits, id_hits, spk_hits = 0, 0, 0, 0, 0, 0
        ex = []
        for line in open(p):
            if not line.strip():
                continue
            it = json.loads(line)
            n += 1
            a = it.get("audio") or {}
            if a.get("bank") in our_banks:
                bank_hits += 1
            blob = json.dumps(it.get("meta", {})) + " " + str(a.get("key", "")) + " " + str(it.get("id", ""))
            for mp in set(_MINDS_PATH.findall(blob)):
                if mp in our_minds_paths:
                    minds_hits += 1
                    ex.append(mp)
            for tok in set(re.findall(r"[A-Za-z0-9_\-~./]+", blob)):
                stem = Path(tok).stem
                if tok in our_ids or (len(stem) > 8 and stem in our_ids and "response_" not in stem):
                    id_hits += 1
                    ex.append(tok)
            sp = (it.get("meta") or {}).get("speaker")
            if sp and sp in our_spk:
                spk_hits += 1
        rep[name] = {"items": n, "bank_hits": bank_hits, "minds_path_hits": minds_hits, "source_id_hits": id_hits,
                     "speaker_hits": spk_hits, "examples": ex[:10]}
        if bank_hits or minds_hits or id_hits or spk_hits:
            ok = False
    # the evaluation set's MInDS-14 clips are English (en-US/en-GB/en-AU); none of those locales is used
    rep["minds_locales_used"] = sorted({c["meta"].get("locale") for c in clips if c["corpus"] == "minds14"})
    rep["ok"] = ok
    return rep


def make_summary(out, tr, cal, res, bal, val, leak, P, clips, facts, secs) -> dict:
    allit = tr + cal
    by = lambda f: dict(Counter(f(i) for i in allit).most_common())  # noqa: E731
    used_keys = {(i["audio"]["bank"], i["audio"]["key"]) for i in allit}
    hours = 0.0
    idx_cache = {}
    for b, k in used_keys:
        if b not in idx_cache:
            idx_cache[b] = json.loads((out / "audio" / f"{b}.json").read_text())
        hours += idx_cache[b][k][1] / SR / 3600
    bank_hours = {}
    for b, ix in idx_cache.items():
        bank_hours[b] = round(sum(v[1] for v in ix.values()) / SR / 3600, 2)
    examples = {}
    for src in sorted({i["source"] for i in allit}):
        ex = [i for i in tr if i["source"] == src][:3]
        examples[src] = [{"question": e["question"], "options": e["options"], "gold": e["gold"],
                          "answer": e["options"][e["gold"]], "corpus": e["meta"].get("corpus"),
                          "audio": e["audio"], "channel": e["channel"]} for e in ex]
    return {
        "builder": BUILDER, "built_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "build_seconds": round(secs),
        "counts": {"train": len(tr), "CAL": len(cal), "cal_frac": round(len(cal) / max(1, len(allit)), 4)},
        "by_source_split": by(lambda i: f"{i['source']}/{i['set']}"),
        "by_source_qtype": by(lambda i: f"{i['source']}/{i['meta']['qtype']}"),
        "by_qtype": by(lambda i: i["meta"]["qtype"]),
        "by_corpus": by(lambda i: i["meta"].get("corpus")),
        "by_channel": by(lambda i: i["channel"]),
        "by_scenario": by(lambda i: i["meta"]["recipe"].get("scenario")),
        "by_codec": by(lambda i: i["meta"]["recipe"].get("codec") or "none"),
        "audio_hours_used": round(hours, 2), "bank_hours": bank_hours,
        "clips_used": len(used_keys), "render": res,
        "render_errors": sum(1 for f in facts.values() if "error" in f),
        "render_error_examples": [f["error"] for f in facts.values() if "error" in f][:5],
        "licences": LIC, "sources": SOURCE_REVS,
        "musan_licence": P.musan_report, "musan_files_allowed": {k: (len(v["train"]), len(v["CAL"])) for k, v in P.bg.items()},
        "rirs_far_rooms": {k: len(v) for k, v in P.rir.items()},
        "cal_groups": P.cal_groups,
        "balance": bal, "validation": val, "leakage": leak, "examples": examples,
        "skipped": {
            "AMR-NB": "ffmpeg in the Debian image has only an AMR-NB decoder (no libopencore encoder); used G.726 and Speex instead",
            "MInDS-14 English locales": "en-US/en-GB/en-AU are evaluation data; only the 11 non-English locales are used",
            "name mentions": "no reliable rule from transcripts (Apollo call signs / phonetic alphabet 'MIKE', 'CHARLIE'; Vystadial venue names) -> not asked",
            "robocall transcripts/language": "metadata transcripts + language are Whisper outputs -> never used as labels",
            "real RIRs / point-source noises (OpenSLR 28)": "third-party licences; only the Apache-2.0 simulated RIRs are used",
        },
    }
