"""Build sb_relative inside Modal containers: plan pairs, augment targets, assemble items, plus an extra evaluation set.

Steps: plan() -> render_aug(i, n) -> assemble(); build_eval() writes the crema_relative_<feature> evaluation items.
"""

from __future__ import annotations

import glob
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import common as C
import relabels as R

BUILDER = "sb_relative"
OUT = Path("/vol/stageb/sb_relative")
WORK = OUT / "work"
SBV = Path("/vol/stageb/sb_voice")
ADI_ITEMS = Path("/vol/adi/items.jsonl")
ADI_EXTRA = Path("/vol/adi/extra")
SR = C.SAMPLE_RATE
PART_MIN_S, PART_MAX_S = 1.0, 8.0
BANK_MAX_SAMPLES = 650_000_000  # ~2.6 GB float32 per bank file

SOURCES = {
    "vctk": {
        "name": "VCTK 0.92 (mic1) — clips + features from sb_voice stage 1 (/vol/stageb/sb_voice/work2), sb_voice train speakers only",
        "repo": "sanchit-gandhi/vctk", "rev": "73ef4ee7d49a6fed4ea1efd65f82b4c95faeb9de",
        "licence": "CC BY 4.0 — VCTK corpus licence (CSTR-Edinburgh/vctk card @31539806e8a6ee3b0c0fef88659ed518542bc564, "
                   "https://datashare.ed.ac.uk/handle/10283/3443); parquet mirror sanchit-gandhi/vctk (plain conversion).",
    },
    "libritts_r": {
        "name": "LibriTTS-R dev-clean + train-clean-100 shards 0-5 — via sb_voice stage 1, sb_voice train speakers only",
        "repo": "mythicinfinity/libritts_r", "rev": "0d1718db351207d5a493135a62d5ed8cc4733830",
        "licence": "CC BY 4.0 (HF cardData)",
    },
    "crema_d": {
        "name": "CREMA-D, training actors only (minus sb_voice's 2 CAL actors) — via sb_voice stage 1 (/vol/data/audio/crema_d)",
        "repo": "cfahlgren1/crema-d", "rev": "b7d5beded69285eb5eb6a1c6894a5b0ad17cab92; github CheyneyComputerScience/CREMA-D@1658cd342dff90010aa843eaeebd53610a08b1dc",
        "licence": "ODbL 1.0 + Database Contents License (https://github.com/CheyneyComputerScience/CREMA-D)",
    },
    "tone": {"name": "generated 1 kHz separator tone + silence", "rev": "relabels.py", "licence": "CC0 (generated)"},
}

# Quotas
TRAIN_ITEMS = {"loudness": 5600, "rate": 5200, "pitch": 5600, "pitch_variation": 3600}
VIEW_SHARE = {"score": 0.55, "noul": 0.45}
SRC_SHARE = {"crema_d": 0.40, "vctk": 0.35, "libritts_r": 0.25}
AUG_SHARE = {"crema_d": 0.25, "vctk": 0.40, "libritts_r": 0.40}  # wanted share of augmented recordings (loud/rate/pitch)
AUG_OVERGEN = 1.35
CAL_FRAC = 0.032
PHONE_P = 0.15
MAX_ITEMS_PER_REC = 4
CAL_SPEAKERS_N = {"vctk": 4, "libritts_r": 6, "crema_d": 2}


# Inputs

def sbv_train_speakers() -> tuple[set, set]:
    """(speakers in sb_voice items_train.jsonl, sb_voice CAL speakers)."""
    spk = set()
    with open(SBV / "items_train.jsonl") as f:
        for line in f:
            m = json.loads(line)["meta"]
            s = m.get("speaker")
            if isinstance(s, str):
                spk.add(s)
    cal = set(json.loads((SBV / "cal_speakers.json").read_text()))
    return spk, cal


def exp0_eval_crema_actors() -> set:
    out = set()
    for it in C.read_jsonl("/vol/data/eval_items.jsonl"):
        if it.get("audio") and it["audio"]["bank"] == "crema_d":
            out.add(str(it["meta"].get("actor")))
    return out


def exp0_train_crema_actors() -> set:
    return {str(it["meta"]["actor"]) for it in C.read_jsonl("/vol/data/train.jsonl") if it.get("source") == "crema_d"}


def load_records() -> tuple[list[dict], dict]:
    """sb_voice stage-1 records (VCTK, LibriTTS-R, CREMA-D) restricted to allowed speakers, with features."""
    train_spk, sbv_cal = sbv_train_speakers()
    ev_actors, tr_actors = exp0_eval_crema_actors(), exp0_train_crema_actors()
    recs, skipped = [], Counter()
    for p in sorted(glob.glob(str(SBV / "work2" / "*.jsonl"))):
        shard = Path(p).stem
        if not shard.startswith(("vctk", "libritts", "crema")):
            continue
        for r in C.read_jsonl(p):
            spk = r["speaker"]
            if spk in sbv_cal:
                skipped["sb_voice_cal_speaker"] += 1
                continue
            if spk not in train_spk:
                skipped["not_sb_voice_train_speaker"] += 1
                continue
            if r["source"] == "crema_d":
                actor = spk.split(":")[1]
                if actor in ev_actors or actor not in tr_actors:
                    skipped["crema_not_exp0_train_actor"] += 1
                    continue
            f = r.get("feat")
            if not f or "error" in f:
                skipped["no_features"] += 1
                continue
            if not PART_MIN_S <= r["dur"] <= PART_MAX_S:
                skipped["duration"] += 1
                continue
            if r["source"] == "crema_d":  # hand syllable counts per sentence instead of CMUdict
                syl = R.crema_syllables(r["source_id"])
                f["syll"] = syl
                f["rate"] = syl / f["span_s"] if f.get("span_s") and f["span_s"] >= 0.5 else None
            r["shard"] = shard
            recs.append(r)
    info = {"skipped": dict(skipped), "kept": dict(Counter(r["source"] for r in recs)),
            "sb_voice_cal_speakers": sorted(sbv_cal), "exp0_eval_crema_actors": sorted(ev_actors)}
    return recs, info


class Clips:
    """Lazy access to sb_voice work2 audio banks (memory-mapped)."""

    def __init__(self):
        self.banks = {}

    def get(self, shard: str, key: str, root: Path = SBV / "work2") -> np.ndarray:
        if (root, shard) not in self.banks:
            self.banks[(root, shard)] = C.AudioBank(root, shard)
        return self.banks[(root, shard)].get(key)


# Features + augmentation (inside containers)

def clip_features(x: np.ndarray, syll: int) -> dict | None:
    """Copy of sb_voice build.clip_features (same pYIN / RMS / VAD settings); syllables passed in."""
    import librosa

    x = np.asarray(x, dtype=np.float32)
    act = R.activity(x)
    if act is None:
        return None
    f0, vflag, _ = librosa.pyin(x, fmin=65.0, fmax=500.0, sr=SR, frame_length=1024, hop_length=320)
    rms = librosa.feature.rms(y=x, frame_length=1024, hop_length=320, center=True)[0]
    n = min(len(rms), len(f0))
    voiced = R.energy_gated_voicing(np.asarray(vflag[:n], bool) & np.isfinite(f0[:n]), rms[:n])
    fs = R.f0_stats(f0[:n], voiced)
    loud = float(10 * np.log10(np.mean(rms[:n][voiced] ** 2) + 1e-20)) if voiced.sum() >= 10 else act["active_db"]
    rate = syll / act["span_s"] if syll > 0 and act["span_s"] >= 0.5 else float("nan")
    out = {"loud_db": loud, **fs, "voiced_frac": float(voiced.mean()) if n else 0.0, "span_s": act["span_s"],
           "pause_frac": act["pause_frac"], "n_pauses": act["n_pauses"], "active_db": act["active_db"], "syll": syll, "rate": rate}
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else (round(v, 5) if isinstance(v, float) else v))
            for k, v in out.items()}


def augment(x: np.ndarray, kind: str, amount: float) -> np.ndarray:
    """gain_db: multiply; tempo_log: rubberband tempo x exp(amount) (R3 engine); shift_st: rubberband pitch shift with
    formant preservation."""
    if kind == "gain_db":
        return (np.asarray(x, dtype=np.float64) * 10 ** (amount / 20.0)).astype(np.float32)
    import subprocess
    import tempfile

    import soundfile as sf

    with tempfile.TemporaryDirectory() as d:
        sf.write(f"{d}/in.wav", np.asarray(x, dtype=np.float32), SR, subtype="FLOAT")
        extra = ["-T", f"{math.exp(amount):.6f}"] if kind == "tempo_log" else ["-p", f"{amount:.5f}", "-F"]
        for engine in (["--fine"], []):
            r = subprocess.run(["rubberband", "-q", *engine, *extra, f"{d}/in.wav", f"{d}/out.wav"], capture_output=True, timeout=120)
            if r.returncode == 0:
                break
        else:
            raise RuntimeError(f"rubberband failed: {r.stderr[-300:]!r}")
        y, sr = sf.read(f"{d}/out.wav", dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    return C.resample(y, sr, SR) if sr != SR else y


def _aug_job(args):
    spec, x = args
    try:
        a = spec["aug"]
        y = augment(x, a["kind"], a["amount"])
        if a["kind"] == "gain_db":  # gain shifts only the level: loudness moves by exactly the gain (no re-measure needed)
            f = dict(spec["tgt_feat"])
            for k in ("loud_db", "active_db"):
                if f.get(k) is not None:
                    f[k] = round(f[k] + a["amount"], 5)
        else:
            f = clip_features(y, spec["tgt_feat"]["syll"])
        return spec["id"], y, f, None
    except Exception as e:  # never kill a shard for one clip
        return spec["id"], None, None, repr(e)[:300]


# Plan

def pick_cal_speakers(by_spk: dict) -> set:
    rng = random.Random(f"{C.SEED}:{BUILDER}:cal")
    cal = set()
    for src, n in CAL_SPEAKERS_N.items():
        pool = sorted(s for s, rs in by_spk.items() if rs[0]["source"] == src and len(rs) >= 20)
        by_g = defaultdict(list)
        for s in pool:
            by_g[by_spk[s][0].get("gender")].append(s)
        gs = sorted(by_g, key=str)
        i = 0
        while sum(1 for c in cal if c.startswith(src)) < n and any(by_g.values()):
            g = gs[i % len(gs)]
            if by_g[g]:
                s = rng.choice(by_g[g])
                by_g[g].remove(s)
                cal.add(s)
            i += 1
    return cal


def reference_pool(rs: list[dict]) -> list[dict]:
    """The speaker's normal voice: CREMA-D neutral clips; for read speech, clips whose loudness, rate and median F0
    all lie within the speaker's 20th-80th percentile."""
    if rs[0]["source"] == "crema_d":
        return [r for r in rs if r.get("emotion") == "NEU" and R._finite(r["feat"].get("loud_db")) and R._finite(r["feat"].get("rate"))]
    ranks = {}
    for k in ("loud_db", "rate", "f0_med"):
        ranks[k] = R.percentile_ranks([r["feat"].get(k) if R._finite(r["feat"].get(k)) else np.nan for r in rs])
    out = []
    for i, r in enumerate(rs):
        ok = all(np.isfinite(ranks[k][i]) and 0.2 <= ranks[k][i] <= 0.8 for k in ("loud_db", "rate"))
        if ok and (not np.isfinite(ranks["f0_med"][i]) or 0.2 <= ranks["f0_med"][i] <= 0.8):
            out.append(r)
    return out


def _ref_for(target: dict, refs: list[dict], rng: random.Random) -> dict | None:
    cands = [r for r in refs if r["key"] != target["key"] and r.get("text") != target.get("text")
             and R.total_seconds(r["dur"], target["dur"]) <= R.MAX_TOTAL_S]
    return rng.choice(cands) if cands else None


def labels_of(ref_f: dict, tgt_f: dict) -> dict:
    out = {}
    for f in R.FEATURES:
        d = R.rel_diff(f, ref_f, tgt_f)
        if f == "loudness" and min(ref_f.get("n_voiced") or 0, tgt_f.get("n_voiced") or 0) < 10:
            d = None  # loudness = RMS on voiced frames; with < 10 voiced frames sb_voice falls back to all active frames
        out[f] = {"d": None if d is None else round(d, 4), "cls": R.classify(f, d)}
    return out


def plan(smoke: bool = False, replan: bool = False) -> dict:
    t0 = time.time()
    p = WORK / ("plan_smoke.json" if smoke else "plan.json")
    if p.exists() and not replan:
        pl = json.loads(p.read_text())
        return {"reused": str(p), "n_natural": len(pl["natural"]), "n_aug": len(pl["aug"])}
    recs, info = load_records()
    by_spk = defaultdict(list)
    for r in sorted(recs, key=lambda r: r["key"]):
        by_spk[r["speaker"]].append(r)
    by_spk = {s: rs for s, rs in by_spk.items() if len(rs) >= 8}
    cal = pick_cal_speakers(by_spk)
    rng = random.Random(f"{C.SEED}:{BUILDER}:plan")
    natural, refs_of = [], {}
    for spk in sorted(by_spk):
        rs = by_spk[spk]
        refs = reference_pool(rs)
        if len(refs) < 2:
            continue
        refs_of[spk] = refs
        split = "CAL" if spk in cal else "train"
        n_per = 2 if rs[0]["source"] == "crema_d" else 1
        for t in rs:
            used = set()
            for j in range(n_per):
                ref = _ref_for(t, [r for r in refs if r["key"] not in used], rng)
                if ref is None:
                    break
                used.add(ref["key"])
                natural.append({"id": f"nat/{t['key']}/{j}", "kind": "natural", "source": t["source"], "speaker": spk,
                                "split": split, "ref": ref["key"], "tgt": t["key"], "labels": labels_of(ref["feat"], t["feat"])})
    if smoke:
        natural = [n for n in natural if R.stable_frac(n["id"], "smoke") < 0.03]
    # availability of natural labels, per split / source / feature / class
    avail = Counter()
    for n in natural:
        for f, lab in n["labels"].items():
            if lab["cls"] is not None:
                avail[(n["split"], n["source"], f, lab["cls"])] += 1
    # augmentation specs to reach AUG_SHARE (and to fill classes the natural pairs cannot)
    recs_by_key = {r["key"]: r for r in recs}
    aug, aug_need = [], {}
    scale = 0.03 if smoke else 1.0
    for split in ("train", "CAL"):
        spks_by_src = defaultdict(list)
        for s in refs_of:
            if (split == "CAL") == (s in cal):
                spks_by_src[by_spk[s][0]["source"]].append(s)
        for f in ("loudness", "rate", "pitch"):
            for src in SRC_SHARE:
                n_items = TRAIN_ITEMS[f] * SRC_SHARE[src] * (1.0 if split == "train" else CAL_FRAC) * scale
                need_cls = n_items / 3.0
                for cls in (0, 1, 2):
                    nat = avail[(split, src, f, cls)]
                    want = max(need_cls * AUG_SHARE[src], need_cls - 0.85 * nat)
                    n_aug = int(math.ceil(want * AUG_OVERGEN))
                    aug_need[f"{split}/{src}/{f}/{cls}"] = n_aug
                    spks = sorted(spks_by_src[src])
                    tries, made = 0, 0
                    while made < n_aug and tries < n_aug * 30 and spks:
                        tries += 1
                        spk = rng.choice(spks)
                        rs = by_spk[spk]
                        t = rng.choice(rs)
                        ref = _ref_for(t, refs_of[spk], rng)
                        if ref is None:
                            continue
                        d0 = R.rel_diff(f, ref["feat"], t["feat"])
                        if d0 is None:
                            continue
                        amt = R.plan_amount(f, d0, cls, rng)
                        if amt is None:
                            continue
                        kind = R.AUG[f][0]
                        if kind == "tempo_log" and (t["dur"] / math.exp(amt) > PART_MAX_S + 2 or
                                                    R.total_seconds(ref["dur"], t["dur"] / math.exp(amt)) > R.MAX_TOTAL_S):
                            continue
                        made += 1
                        aug.append({"id": f"aug/{split}/{f}/{cls}/{len(aug):06d}", "kind": "augmented", "source": src, "speaker": spk,
                                    "split": split, "ref": ref["key"], "tgt": t["key"], "tgt_shard": t["shard"],
                                    "aug": {"feature": f, "kind": kind, "amount": round(amt, 5), "natural_diff": round(d0, 4),
                                            "expected_cls": cls, "expected_diff": round(d0 + amt, 4)},
                                    "tgt_feat": t["feat"]})
    out = {"natural": natural, "aug": aug, "cal_speakers": sorted(cal), "records": info,
           "avail_natural": {"/".join(map(str, k)): v for k, v in sorted(avail.items())}, "aug_need": aug_need,
           "n_speakers": dict(Counter(by_spk[s][0]["source"] for s in refs_of)),
           "n_refs": dict(Counter(by_spk[s][0]["source"] for s in refs_of for _ in refs_of[s])), "smoke": smoke,
           "seconds": round(time.time() - t0)}
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / ("plan_smoke.json" if smoke else "plan.json")).write_text(json.dumps(out))
    print(f"[plan] natural {len(natural)}, aug {len(aug)}, cal speakers {sorted(cal)} ({out['seconds']}s)", flush=True)
    return {k: v for k, v in out.items() if k not in ("natural", "aug")} | {"n_natural": len(natural), "n_aug": len(aug)}


# Render augmented targets

def render_aug(shard: int, n_shards: int, n_proc: int, smoke: bool = False) -> dict:
    from multiprocessing import get_context

    name = f"aug{'_smoke' if smoke else ''}_{shard:02d}"
    if (WORK / f"{name}.jsonl").exists() and (WORK / f"{name}.npy").exists():
        return {"shard": shard, "skipped": "exists"}
    pl = json.loads((WORK / ("plan_smoke.json" if smoke else "plan.json")).read_text())
    specs = [s for i, s in enumerate(pl["aug"]) if i % n_shards == shard]
    clips = Clips()
    jobs = [(s, clips.get(s["tgt_shard"], s["tgt"])) for s in specs]
    t0 = time.time()
    # one async task per clip with a per-result timeout: a hung / killed worker costs one clip, not the shard
    res = []
    pool = get_context("spawn").Pool(n_proc)
    try:
        pending = [(j[0]["id"], pool.apply_async(_aug_job, (j,))) for j in jobs]
        deadline = time.time() + 1500
        for sid, ar in pending:
            try:
                res.append(ar.get(timeout=max(5.0, deadline - time.time())))
            except Exception as e:
                res.append((sid, None, None, f"timeout/failed: {type(e).__name__}"))
    finally:
        pool.terminate()
    audio, rows, errors = {}, [], Counter()
    for sid, y, f, err in res:
        if err or y is None or f is None:
            errors[(err or "no features")[:80]] += 1
            continue
        audio[sid] = y
        rows.append({"id": sid, "feat": f, "dur": round(len(y) / SR, 3)})
    C.write_audio_bank(WORK, name, audio)
    C.write_jsonl(WORK / f"{name}.jsonl", rows)
    print(f"[render_aug] shard {shard}: {len(rows)}/{len(specs)} ok, errors {dict(errors)} ({time.time() - t0:.0f}s)", flush=True)
    return {"shard": shard, "ok": len(rows), "n": len(specs), "errors": dict(errors), "seconds": round(time.time() - t0)}


# Assemble

class NoShuffle(random.Random):
    def shuffle(self, x, *a, **k):  # noqa: D401
        return None


NO_SHUFFLE = NoShuffle(0)


def _interleave(nat: list, aug: list, aug_share: float, rng: random.Random) -> list:
    """Merge two (shuffled) lists so that about aug_share of the first elements come from aug."""
    out, i, j = [], 0, 0
    while i < len(nat) or j < len(aug):
        take_aug = j < len(aug) and (i >= len(nat) or rng.random() < aug_share)
        if take_aug:
            out.append(aug[j])
            j += 1
        else:
            out.append(nat[i])
            i += 1
    return out


def assemble(smoke: bool = False) -> dict:
    t0 = time.time()
    tag = "_smoke" if smoke else ""
    pl = json.loads((WORK / f"plan{tag}.json").read_text())
    recs, _ = load_records()
    rec = {r["key"]: r for r in recs}
    # rendered augmented targets
    aug_feat, aug_bank = {}, {}
    for p in sorted(glob.glob(str(WORK / f"aug{tag}_[0-9][0-9].jsonl"))):
        name = Path(p).stem
        for row in C.read_jsonl(p):
            aug_feat[row["id"]] = row
            aug_bank[row["id"]] = name
    # candidate recordings with final labels
    cands, verify = [], Counter()
    for n in pl["natural"]:
        cands.append({**n, "ref_s": rec[n["ref"]]["dur"], "tgt_s": rec[n["tgt"]]["dur"]})
    for a in pl["aug"]:
        row = aug_feat.get(a["id"])
        if row is None:
            verify["not_rendered"] += 1
            continue
        if R.total_seconds(rec[a["ref"]]["dur"], row["dur"]) > R.MAX_TOTAL_S:
            verify["too_long"] += 1
            continue
        labels = labels_of(rec[a["ref"]]["feat"], row["feat"])
        f, exp = a["aug"]["feature"], a["aug"]["expected_cls"]
        got = labels[f]["cls"]
        if got != exp:  # the manipulation did not land where planned (measurement disagreed): do not use that label
            verify[f"mismatch_{f}"] += 1
            labels[f] = {"d": labels[f]["d"], "cls": None, "dropped": f"expected class {exp}, measured {got}"}
        else:
            verify[f"ok_{f}"] += 1
        cands.append({**{k: v for k, v in a.items() if k != "tgt_feat"}, "labels": labels, "tgt_feat": row["feat"],
                      "ref_s": rec[a["ref"]]["dur"], "tgt_s": row["dur"]})
    print(f"[assemble] candidates {len(cands)}; augmentation verification {dict(verify)}", flush=True)

    rng = random.Random(f"{C.SEED}:{BUILDER}:assemble")
    used_feat = defaultdict(set)  # cand id -> features already asked
    n_items = Counter()
    items = {"train": [], "CAL": []}
    shortfall = {}
    counter = 0
    scale = 0.03 if smoke else 1.0
    by = defaultdict(list)  # (split, source, feature, cls) -> candidate list
    for c in cands:
        for f, lab in c["labels"].items():
            if lab["cls"] is not None:
                by[(c["split"], c["source"], f, lab["cls"])].append(c)

    def add_item(c, f, view, question, options, target, gold, extra):
        nonlocal counter
        counter += 1
        split = c["split"]
        lab = c["labels"][f]
        ref_r = rec[c["ref"]]
        tgt_r = rec[c["tgt"]]
        qtype = view
        meta = {"qtype": qtype, "family": f"relative_{f}", "feature": f, "view": view, "corpus": c["source"],
                "speaker": c["speaker"], "source_id": tgt_r["source_id"], "ref_source_id": ref_r["source_id"],
                "recording": c["id"], "pair_kind": c["kind"], "diff": lab["d"], "diff_unit": R.THRESHOLDS[f][2],
                "class": R.CLASS_NAMES[lab["cls"]],
                "label_rule": f"second part minus first part; clear >= {R.THRESHOLDS[f][0]:.4g}, about the same <= {R.THRESHOLDS[f][1]:.4g} "
                              f"({R.THRESHOLDS[f][2]}), in-between pairs dropped",
                "licence": SOURCES[c["source"]]["licence"], "ref_emotion": ref_r.get("emotion"), "tgt_emotion": tgt_r.get("emotion"),
                "layout": {"ref_s": c["ref_s"], "gap_s": R.GAP_S, "tone_s": R.TONE_S, "tgt_s": c["tgt_s"]}}
        if c["kind"] == "augmented":
            meta["augmentation"] = c["aug"]
        meta.update(extra)
        it = C.make_item(item_id=f"{BUILDER}/{split}/{f}/{counter:06d}", set_name=split, source=f"{BUILDER}_{f}", question=question,
                         options=options, target=target, gold=gold, rng=rng if view != "score" else NO_SHUFFLE,
                         audio={"bank": None, "key": f"{BUILDER}/{c['id']}"}, channel="phone" if rng.random() < PHONE_P else "clean",
                         state_text=None, meta=meta)
        items[split].append(it)
        used_feat[c["id"]].add(f)
        n_items[c["id"]] += 1

    def pool(split, src, f, cls):
        lst = [c for c in by[(split, src, f, cls)] if f not in used_feat[c["id"]] and n_items[c["id"]] < MAX_ITEMS_PER_REC]
        nat = [c for c in lst if c["kind"] == "natural"]
        aug = [c for c in lst if c["kind"] == "augmented"]
        rng.shuffle(nat)
        rng.shuffle(aug)
        # reuse recordings that already carry items first (less audio), keeping the natural / augmented mix
        nat.sort(key=lambda c: -n_items[c["id"]] if n_items[c["id"]] else 0)
        aug.sort(key=lambda c: -n_items[c["id"]] if n_items[c["id"]] else 0)
        share = AUG_SHARE[src] if f != "pitch_variation" else 0.0
        return _interleave(nat, aug, share, rng)

    for split in ("train", "CAL"):
        for f in ("pitch_variation", "pitch", "rate", "loudness"):
            for src in SRC_SHARE:
                n_total = TRAIN_ITEMS[f] * SRC_SHARE[src] * (1.0 if split == "train" else CAL_FRAC) * scale
                n_score = int(round(n_total * VIEW_SHARE["score"] / 3.0))
                n_noul = int(round(n_total * VIEW_SHARE["noul"] / 6.0))  # per direction and yes/no side
                # score: equal numbers per class (as far as available)
                pools = {cls: pool(split, src, f, cls) for cls in (0, 1, 2)}
                k = min([n_score] + [len(p) for p in pools.values()])
                shortfall[f"{split}/{src}/{f}/score"] = n_score - k
                for cls in (0, 1, 2):
                    for c in pools[cls][:k]:
                        q, opts, oset = R.score_question(f, rng)
                        t = [0.0, 0.0, 0.0]
                        t[cls] = 1.0
                        add_item(c, f, "score", q, opts, t, cls, {"option_set": oset})
                # noul: per direction, yes = matching class, no = the other two classes half / half; exact 50/50
                for direction in ("up", "down", "same"):
                    yes_cls = {"up": 2, "down": 0, "same": 1}[direction]
                    no_cls = [c_ for c_ in (0, 1, 2) if c_ != yes_cls]
                    py = pool(split, src, f, yes_cls)
                    pn = [pool(split, src, f, c_) for c_ in no_cls]
                    half = n_noul // 2
                    no_take = [pn[0][:half], pn[1][: n_noul - min(half, len(pn[0]))]]
                    no_list = no_take[0] + no_take[1]
                    m = min(n_noul, len(py), len(no_list))
                    shortfall[f"{split}/{src}/{f}/noul_{direction}"] = n_noul - m
                    yes_list = py[:m]
                    rng.shuffle(no_list)
                    no_list = no_list[:m]
                    for is_yes, lst in ((True, yes_list), (False, no_list)):
                        for c in lst:
                            if f in used_feat[c["id"]]:
                                continue
                            q = R.noul_question(f, direction, rng)
                            add_item(c, f, "noul", q, ["yes", "no"], [1.0, 0.0] if is_yes else [0.0, 1.0], 0 if is_yes else 1,
                                     {"direction": direction, "answer": "yes" if is_yes else "no"})
    used = {c["id"]: c for c in cands if n_items[c["id"]] > 0}
    print(f"[assemble] items train {len(items['train'])}, CAL {len(items['CAL'])}; recordings {len(used)} ({time.time() - t0:.0f}s)", flush=True)

    # Render final two-part recordings into banks
    out_audio = OUT / ("smoke_audio" if smoke else "audio")
    out_audio.mkdir(parents=True, exist_ok=True)
    clips = Clips()
    bank_of, cur, cur_n, bank_i, hours, durs = {}, {}, 0, 0, 0.0, []

    def flush():
        nonlocal cur, cur_n, bank_i
        if cur:
            name = f"{BUILDER}_{bank_i:02d}"
            C.write_audio_bank(out_audio, name, cur)
            print(f"[assemble] wrote bank {name}: {len(cur)} recordings, {cur_n / SR / 3600:.2f} h", flush=True)
            bank_i += 1
            cur, cur_n = {}, 0

    layout = {}
    for cid in sorted(used):
        c = used[cid]
        ref = clips.get(rec[c["ref"]]["shard"], c["ref"])
        if c["kind"] == "augmented":
            tgt = clips.get(aug_bank[cid], cid, root=WORK)
        else:
            tgt = clips.get(rec[c["tgt"]]["shard"], c["tgt"])
        y, info = R.two_part(ref, tgt)
        key = f"{BUILDER}/{cid}"
        if cur_n + len(y) > BANK_MAX_SAMPLES:
            flush()
        cur[key] = y
        cur_n += len(y)
        bank_of[key] = f"{BUILDER}_{bank_i:02d}"
        layout[key] = info
        hours += len(y) / SR / 3600
        durs.append(len(y) / SR)
    flush()
    for split in items:
        for it in items[split]:
            it["audio"]["bank"] = bank_of[it["audio"]["key"]]
            it["meta"]["layout"].update(layout[it["audio"]["key"]])
    out_dir = OUT / "smoke" if smoke else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    C.write_jsonl(out_dir / "items_train.jsonl", items["train"])
    C.write_jsonl(out_dir / "items_cal.jsonl", items["CAL"])
    stats = {"n_train": len(items["train"]), "n_cal": len(items["CAL"]), "recordings": len(used),
             "recordings_by_kind": dict(Counter(c["kind"] for c in used.values())), "audio_hours": round(hours, 3),
             "recording_seconds_pct": [round(float(x), 2) for x in np.percentile(durs, [0, 5, 50, 95, 100])] if durs else [],
             "augmentation_verification": dict(verify), "shortfall": {k: v for k, v in shortfall.items() if v > 0},
             "banks": sorted(set(bank_of.values())), "cal_speakers": pl["cal_speakers"], "plan": {k: pl[k] for k in ("records", "n_speakers", "n_refs")},
             "seconds": round(time.time() - t0)}
    (out_dir / "assemble_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


# Validation + summary (summary.json is written last)
SCHEMA = ["id", "set", "source", "question", "options", "target", "gold", "audio", "channel", "state_text", "meta"]


def check_items(items: list[dict], audio_dir: Path, errors: list, max_s: float = R.MAX_TOTAL_S) -> dict:
    banks = {}
    durs, per_rec = [], Counter()
    for it in items:
        miss = [k for k in SCHEMA if k not in it]
        if miss:
            errors.append(f"{it.get('id')}: missing {miss}")
        n = len(it["options"])
        if not 2 <= n <= 20:
            errors.append(f"{it['id']}: {n} options")
        t = np.asarray(it["target"], float)
        if len(t) != n or abs(t.sum() - 1) > 1e-6:
            errors.append(f"{it['id']}: bad target")
        if t.max() == 1.0 and int(np.argmax(t)) != it["gold"]:
            errors.append(f"{it['id']}: gold != argmax")
        q = it["meta"].get("qtype")
        if q not in ("choice", "noul", "score"):
            errors.append(f"{it['id']}: qtype {q}")
        if q == "noul" and sorted(it["options"]) != ["no", "yes"]:
            errors.append(f"{it['id']}: noul options {it['options']}")
        if q == "score":
            f = it["meta"]["feature"]
            if it["options"] not in R.OPTION_SETS[f] and list(it["options"]) != R.EVAL_SCORE[f][1]:
                errors.append(f"{it['id']}: score options not low->high {it['options']}")
        if it["channel"] not in ("clean", "phone"):
            errors.append(f"{it['id']}: channel")
        a = it["audio"]
        if a["bank"] not in banks:
            banks[a["bank"]] = json.loads((audio_dir / f"{a['bank']}.json").read_text())
        if a["key"] not in banks[a["bank"]]:
            errors.append(f"{it['id']}: audio key missing")
            continue
        ln = banks[a["bank"]][a["key"]][1]
        durs.append(ln / SR)
        if ln / SR > max_s + 1e-6:
            errors.append(f"{it['id']}: {ln / SR:.2f} s > {max_s}")
        per_rec[a["key"]] += 1
    if per_rec and max(per_rec.values()) > MAX_ITEMS_PER_REC:
        errors.append(f"max items per recording {max(per_rec.values())}")
    return {"n": len(items), "dur_s_pct": [round(float(x), 2) for x in np.percentile(durs, [0, 5, 50, 95, 100])] if durs else [],
            "max_items_per_recording": max(per_rec.values()) if per_rec else 0, "banks": sorted(banks)}


def adi_index() -> dict:
    adi = C.read_jsonl(ADI_ITEMS)
    out = {"n": len(adi), "questions": set(), "crema_actors": set(), "crema_keys": set(), "audio_keys": set(), "speaker_like": set(),
           "sources": set()}
    for it in adi:
        out["questions"].add(it["question"])
        out["sources"].add(f"{it.get('source')}|{it.get('benchmark')}")
        if it.get("audio"):
            out["audio_keys"].add(str(it["audio"]["key"]))
        m = it.get("meta") or {}
        for k in ("speaker", "speaker_id", "source_id", "file", "path", "recording_id", "track_id", "spk_id"):
            if m.get(k) is not None:
                out["speaker_like"].add(str(m[k]))
        if str(it.get("benchmark", "")).startswith("crema"):
            out["crema_actors"].add(str(m.get("actor")))
            if it.get("audio"):
                out["crema_keys"].add(it["audio"]["key"])
    return out


def leakage(items: list[dict], adi: dict) -> dict:
    my_actors = {it["meta"]["speaker"].split(":")[1] for it in items if it["meta"]["corpus"] == "crema_d"}
    my_crema_keys = set()
    my_ids = set()
    for it in items:
        m = it["meta"]
        for k in ("source_id", "ref_source_id"):
            my_ids.add(str(m[k]))
            if m["corpus"] == "crema_d":
                my_crema_keys.add(str(m[k]))
    my_spk = {it["meta"]["speaker"] for it in items}
    train_q = {it["question"] for it in items}
    ev_actors = exp0_eval_crema_actors()
    eval_q = R.eval_questions()
    sbv_cal = set(json.loads((SBV / "cal_speakers.json").read_text()))
    eval_keys = set()
    for it in C.read_jsonl("/vol/data/eval_items.jsonl"):
        if it.get("audio"):
            eval_keys.add(str(it["audio"]["key"]))
    return {
        "adi_items": adi["n"],
        "crema_actor_overlap_with_adi": sorted(my_actors & adi["crema_actors"]),
        "crema_actor_overlap_with_exp0_eval": sorted(my_actors & ev_actors),
        "crema_clip_overlap_with_adi": len(my_crema_keys & adi["crema_keys"]),
        "source_id_overlap_with_adi_audio_keys": len(my_ids & adi["audio_keys"]),
        "speaker_overlap_with_adi_meta_ids": sorted(my_spk & adi["speaker_like"])[:20],
        "adi_sources_from_our_corpora": sorted(s for s in adi["sources"] if any(x in s.lower() for x in ("vctk", "libri"))),
        "crema_clip_overlap_with_exp0_eval": len(my_crema_keys & eval_keys),
        "question_text_overlap_with_adi": len(train_q & adi["questions"]),
        "question_text_overlap_with_adi_extra_relative": len(train_q & eval_q),
        "training_pool_vs_adi_questions": len(R.all_training_question_texts() & adi["questions"]),
        "sb_voice_cal_speakers_used": sorted(my_spk & sbv_cal),
        "note": "VCTK / LibriTTS-R are not evaluation-set sources; every clip used here is one of sb_voice's clips, for which "
                "sb_voice's MMSU content check (fingerprint + waveform cross-correlation vs the 957 MMSU clips) found 0 duplicates.",
    }


def validate_and_summarize(smoke: bool = False) -> dict:
    out_dir = OUT / "smoke" if smoke else OUT
    audio_dir = OUT / ("smoke_audio" if smoke else "audio")
    tr = C.read_jsonl(out_dir / "items_train.jsonl")
    cal = C.read_jsonl(out_dir / "items_cal.jsonl")
    errors = []
    checks = {"train": check_items(tr, audio_dir, errors), "CAL": check_items(cal, audio_dir, errors)}
    ids = [it["id"] for it in tr + cal]
    if len(set(ids)) != len(ids):
        errors.append("duplicate ids")
    spk_tr = {it["meta"]["speaker"] for it in tr}
    spk_cal = {it["meta"]["speaker"] for it in cal}
    if spk_tr & spk_cal:
        errors.append(f"train/CAL speaker overlap {sorted(spk_tr & spk_cal)[:5]}")
    # balance
    bal = {}
    for split, its in (("train", tr), ("CAL", cal)):
        cnt = Counter()
        for it in its:
            m = it["meta"]
            if m["qtype"] == "score":
                cnt[(m["feature"], "score", m["class"])] += 1
            else:
                cnt[(m["feature"], "noul_" + m["direction"], m["answer"])] += 1
        bal[split] = {"/".join(k): v for k, v in sorted(cnt.items())}
    for split, its in (("train", tr), ("CAL", cal)):
        yn = defaultdict(Counter)
        for it in its:
            m = it["meta"]
            if m["qtype"] == "noul":
                yn[(m["feature"], m["direction"])][m["answer"]] += 1
        for k, v in yn.items():
            if v["yes"] != v["no"]:
                errors.append(f"{split} noul {k} not 50/50: {dict(v)}")
    leak = leakage(tr + cal, adi_index())
    for k in ("crema_actor_overlap_with_adi", "crema_actor_overlap_with_exp0_eval", "sb_voice_cal_speakers_used", "speaker_overlap_with_adi_meta_ids",
              "adi_sources_from_our_corpora"):
        if leak[k]:
            errors.append(f"leakage {k}: {leak[k]}")
    for k in ("crema_clip_overlap_with_adi", "source_id_overlap_with_adi_audio_keys", "crema_clip_overlap_with_exp0_eval",
              "question_text_overlap_with_adi", "question_text_overlap_with_adi_extra_relative", "training_pool_vs_adi_questions"):
        if leak[k]:
            errors.append(f"leakage {k}: {leak[k]}")
    stats = json.loads((out_dir / "assemble_stats.json").read_text())
    spot = json.loads((out_dir / "spot_check.json").read_text()) if (out_dir / "spot_check.json").exists() else None
    extra = json.loads((ADI_EXTRA / "relative_summary.json").read_text()) if (ADI_EXTRA / "relative_summary.json").exists() and not smoke else None
    allit = tr + cal
    by_src = defaultdict(list)
    for it in allit:
        by_src[it["source"]].append(it)
    rng = random.Random(1)
    examples = {s: [{"question": e["question"], "options": e["options"], "gold": e["gold"], "channel": e["channel"],
                     "audio": e["audio"], "diff": e["meta"]["diff"], "unit": e["meta"]["diff_unit"], "pair_kind": e["meta"]["pair_kind"]}
                    for e in rng.sample(v, min(3, len(v)))] for s, v in sorted(by_src.items())}
    summary = {
        "builder": BUILDER, "version": "v1.1", "smoke": smoke,
        "paths": {"items_train": str(out_dir / "items_train.jsonl"), "items_cal": str(out_dir / "items_cal.jsonl"),
                  "audio": str(audio_dir), "banks": stats["banks"]},
        "counts": {
            "train": len(tr), "CAL": len(cal), "cal_share": round(len(cal) / max(1, len(tr) + len(cal)), 4),
            "by_source": dict(Counter(it["source"] for it in allit)),
            "by_qtype": dict(Counter(it["meta"]["qtype"] for it in allit)),
            "by_split_source_qtype": {f"{s}/{src}/{q}": n for (s, src, q), n in sorted(Counter((it["set"], it["source"], it["meta"]["qtype"]) for it in allit).items())},
            "by_corpus": dict(Counter(it["meta"]["corpus"] for it in allit)),
            "by_pair_kind": dict(Counter(it["meta"]["pair_kind"] for it in allit)),
            "by_feature_pair_kind": {f"{a}/{b}": n for (a, b), n in sorted(Counter((it["meta"]["feature"], it["meta"]["pair_kind"]) for it in allit).items())},
            "by_channel": dict(Counter(it["channel"] for it in allit)),
            "balance": bal,
        },
        "audio": {"hours": stats["audio_hours"], "recordings": stats["recordings"], "recordings_by_kind": stats["recordings_by_kind"],
                  "recording_seconds_pct_0_5_50_95_100": stats["recording_seconds_pct"]},
        "label_rules": {f: {"clear": R.THRESHOLDS[f][0], "same_band": R.THRESHOLDS[f][1], "unit": R.THRESHOLDS[f][2]} for f in R.FEATURES},
        "augmentation_verification": stats["augmentation_verification"], "shortfall": stats["shortfall"],
        "sources": SOURCES, "cal_speakers": stats["cal_speakers"], "plan": stats["plan"],
        "examples": examples,
        "validation": {"ok": not errors, "errors": errors[:50], "n_errors": len(errors), "checks": checks, "leakage": leak,
                       "spot_check_final_recordings": {k: v for k, v in (spot or {}).items() if k != "examples"} or None},
        "adi_extra_benchmark": None if extra is None else {
            "items": str(ADI_EXTRA / "relative_items.jsonl"), "audio": str(ADI_EXTRA / "audio" / "crema_relative.npy"),
            "n_items": extra["n_items"], "by_benchmark_split_qtype": extra["by_benchmark_split_qtype"], "chance": extra["chance"],
            "validation_ok": extra["validation"]["ok"]},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


# Part B: extra evaluation set crema_relative_<feature> (the 21 evaluation-set CREMA-D actors, natural clips)
EVAL_SCORE_CAP = 250  # per benchmark (dev + test)
EVAL_NOUL_CAP = 80  # per benchmark and direction (dev + test), exactly 50 % yes


def _eval_feat_job(args):
    key, x = args
    try:
        return key, clip_features(x, R.crema_syllables(key))
    except Exception as e:
        return key, {"error": repr(e)[:200]}


def build_eval(n_proc: int = 16) -> dict:
    from multiprocessing import get_context

    t0 = time.time()
    adi = C.read_jsonl(ADI_ITEMS)
    actor_of, split_of = {}, {}
    for it in adi:
        if str(it.get("benchmark", "")).startswith("crema") and it.get("audio"):
            k = it["audio"]["key"]
            a = str(it["meta"]["actor"])
            actor_of[k] = a
            s = it["split"]
            if split_of.setdefault(a, s) != s:
                raise RuntimeError(f"actor {a} in both evaluation splits")
    bank = C.AudioBank("/vol/adi/audio", "crema_d")
    keys = sorted(actor_of)
    feats_path = ADI_EXTRA / "relative_work" / "features.json"
    if feats_path.exists():
        feats = json.loads(feats_path.read_text())
    else:
        with get_context("fork").Pool(n_proc) as pool:
            res = pool.map(_eval_feat_job, [(k, bank.get(k)) for k in keys], chunksize=4)
        feats = {k: f for k, f in res}
        feats_path.parent.mkdir(parents=True, exist_ok=True)
        feats_path.write_text(json.dumps(feats))
    print(f"[eval] features for {len(feats)} clips ({time.time() - t0:.0f}s)", flush=True)
    rng = random.Random(f"{C.SEED}:{BUILDER}:eval")
    by_actor = defaultdict(list)
    for k in keys:
        f = feats.get(k)
        if f and "error" not in f:
            by_actor[actor_of[k]].append(k)
    recs = []
    for a in sorted(by_actor):
        ks = by_actor[a]
        refs = [k for k in ks if k.split("_")[2] == "NEU" and R._finite(feats[k].get("loud_db")) and R._finite(feats[k].get("rate"))]
        for t in ks:
            cands = [r for r in refs if r != t and r.split("_")[1] != t.split("_")[1]
                     and R.total_seconds(bank.index[r][1] / SR, bank.index[t][1] / SR) <= R.MAX_TOTAL_S]
            if not cands:
                continue
            r = rng.choice(cands)
            recs.append({"key": f"{r}__{t}", "ref": r, "tgt": t, "actor": a, "split": split_of[a], "labels": labels_of(feats[r], feats[t])})
    items, used = [], set()
    stats = {"recordings_candidates": len(recs), "actors": {s: sorted(a for a, p in split_of.items() if p == s) for s in ("dev", "test")}}
    for f in R.FEATURES:
        bench = f"crema_relative_{f}"
        lab = [c for c in recs if c["labels"][f]["cls"] is not None]
        stats[f"{bench}_label_counts"] = dict(Counter(R.CLASS_NAMES[c["labels"][f]["cls"]] for c in lab))
        taken = set()
        # score: class-balanced round robin within each split (`adi_take`), capped
        cands = [{"key": c["key"], "split": c["split"], "c": c} for c in lab]
        sel = R.adi_select(cands, f"{bench}_score", cap=EVAL_SCORE_CAP, label_of=lambda x: x["c"]["labels"][f]["cls"])
        q, opts = R.EVAL_SCORE[f]
        for x in sel:
            c = x["c"]
            taken.add(c["key"])
            items.append(_eval_item(bench, "score", c, f, f"{R.EVAL_INTRO} {q}", opts, c["labels"][f]["cls"], rng, None))
        for direction in ("up", "same"):
            yes_cls = {"up": 2, "same": 1}[direction]
            cands = [{"key": c["key"], "split": c["split"], "c": c, "pos": c["labels"][f]["cls"] == yes_cls}
                     for c in lab if c["key"] not in taken]
            sel = R.adi_select(cands, f"{bench}_noul_{direction}", cap=EVAL_NOUL_CAP * 2, label_of=lambda x: x["pos"], strict=True)
            # adi_take's strict mode cannot balance a split in which one answer is absent: enforce 50/50 per split here
            bal = []
            for sp in ("dev", "test"):
                ys = [x for x in sel if x["split"] == sp and x["pos"]]
                ns = [x for x in sel if x["split"] == sp and not x["pos"]]
                m = min(len(ys), len(ns))
                bal += ys[:m] + ns[:m]
            sel = bal
            for x in sel:
                c = x["c"]
                taken.add(c["key"])
                items.append(_eval_item(bench, "noul", c, f, f"{R.EVAL_INTRO} {R.EVAL_NOUL[f][direction]}", ["yes", "no"],
                                        0 if x["pos"] else 1, rng, direction))
        used |= taken
    # audio bank of the used two-part recordings
    audio = {}
    layout = {}
    rec_by = {c["key"]: c for c in recs}
    for k in sorted(used):
        c = rec_by[k]
        y, info = R.two_part(bank.get(c["ref"]), bank.get(c["tgt"]))
        audio[k] = y
        layout[k] = info
    (ADI_EXTRA / "audio").mkdir(parents=True, exist_ok=True)
    C.write_audio_bank(ADI_EXTRA / "audio", "crema_relative", audio)
    for it in items:
        it["meta"]["layout"] = layout[it["audio"]["key"]]
        it["meta"]["duration_s"] = round(len(audio[it["audio"]["key"]]) / SR, 3)
    C.write_jsonl(ADI_EXTRA / "relative_items.jsonl", items)
    # validation
    errors = []
    chk = check_items(items, ADI_EXTRA / "audio", errors)
    for it in items:
        for k in ("benchmark", "area", "split", "qtype", "groups", "item_key"):
            if k not in it:
                errors.append(f"{it['id']}: missing {k}")
        if it["meta"]["actor"] not in split_of or split_of[it["meta"]["actor"]] != it["split"]:
            errors.append(f"{it['id']}: split/actor mismatch")
    yn = defaultdict(Counter)
    for it in items:
        if it["qtype"] == "noul":
            yn[(it["benchmark"], it["meta"]["direction"], it["split"])][it["meta"]["answer"]] += 1
    for k, v in yn.items():
        if v["yes"] != v["no"]:
            errors.append(f"noul {k} not 50/50 {dict(v)}")
    adi_ids = {it["id"] for it in adi}
    dup = [it["id"] for it in items if it["id"] in adi_ids]
    if dup:
        errors.append(f"ids collide with evaluation-set items: {dup[:3]}")
    summ = {
        "benchmark_family": "crema_relative_<feature> (extra evaluation set; not part of /vol/adi/items.jsonl; reported separately)",
        "n_items": len(items), "recordings": len(used), "audio_hours": round(sum(len(v) for v in audio.values()) / SR / 3600, 3),
        "by_benchmark_split_qtype": {"/".join(k): v for k, v in sorted(Counter((it["benchmark"], it["split"], it["qtype"]) for it in items).items())},
        "score_class_counts": {"/".join(k): v for k, v in sorted(Counter((it["benchmark"], it["split"], it["meta"]["class"]) for it in items if it["qtype"] == "score").items())},
        "noul_yes_no": {"/".join(k): dict(v) for k, v in sorted(yn.items())},
        "chance": {b: round(float(np.mean([1 / len(it["options"]) for it in items if it["benchmark"] == b])), 4)
                   for b in sorted({it["benchmark"] for it in items})},
        "label_rules": {f: {"clear": R.THRESHOLDS[f][0], "same_band": R.THRESHOLDS[f][1], "unit": R.THRESHOLDS[f][2]} for f in R.FEATURES},
        "construction": "per evaluation-set CREMA-D actor (same 21 actors and dev/test split): every clip is a target, the reference is one of "
                        "the same actor's neutral clips of a different sentence (seeded random); two-part recording = ref + 0.6 s silence + "
                        "0.15 s 1 kHz tone + 0.6 s silence + target; natural clips only (no augmentation); features = sb_voice definitions "
                        "(copied), syllables = hand counts per sentence; ambiguous in-between differences are not used.",
        "stats": stats, "validation": {"ok": not errors, "errors": errors[:30], "checks": chk},
        "examples": [{"id": it["id"], "question": it["question"], "options": it["options"], "gold": it["gold"]} for it in items[:: max(1, len(items) // 6)][:6]],
        "seconds": round(time.time() - t0),
    }
    (ADI_EXTRA / "relative_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    return summ


def _eval_item(bench, qtype, c, f, question, options, gold, rng, direction):
    t = [0.0] * len(options)
    t[gold] = 1.0
    key = c["key"] if qtype == "score" else f"{direction}__{c['key']}"
    lab = c["labels"][f]
    item = C.make_item(item_id=f"ADI/B/{bench}/{c['split']}/{key}", set_name=f"ADI/B/{bench}/{c['split']}", source="crema_d",
                       question=question, options=list(options), target=t, gold=gold, rng=rng if qtype == "noul" else R.NO_SHUFFLE,
                       audio={"bank": "crema_relative", "key": c["key"]},
                       meta={"actor": c["actor"], "ref_key": c["ref"], "tgt_key": c["tgt"], "ref_emotion": c["ref"].split("_")[2],
                             "emotion": c["tgt"].split("_")[2], "sentence": c["tgt"].split("_")[1], "ref_sentence": c["ref"].split("_")[1],
                             "level_code": c["tgt"].split("_")[3], "feature": f, "diff": lab["d"], "diff_unit": R.THRESHOLDS[f][2],
                             "class": R.CLASS_NAMES[lab["cls"]], "direction": direction, "qtype": qtype,
                             "answer": None if qtype == "score" else ("yes" if gold == 0 else "no")})
    item.update(benchmark=bench, area="B", split=c["split"], qtype=qtype, groups=[f"crema_actor:{c['actor']}"], item_key=key)
    return item


# Spot check: re-measure a sample of final two-part recordings (cut at the stored layout) and compare with the labels

def spot_check(n: int = 160, smoke: bool = False) -> dict:
    out_dir = OUT / "smoke" if smoke else OUT
    audio_dir = OUT / ("smoke_audio" if smoke else "audio")
    items = [it for it in C.read_jsonl(out_dir / "items_train.jsonl") if it["meta"]["feature"] in ("loudness", "pitch")]
    rng = random.Random(7)
    sample = rng.sample(items, min(n, len(items)))
    banks, rows = {}, []
    for it in sample:
        b = it["audio"]["bank"]
        if b not in banks:
            banks[b] = C.AudioBank(audio_dir, b)
        y = banks[b].get(it["audio"]["key"])
        lay = it["meta"]["layout"]
        ref = y[: int(round(lay["ref_s"] * SR))]
        tgt = y[int(round(lay["tgt_start_s"] * SR)):]
        fr, ft = clip_features(ref, 1), clip_features(tgt, 1)
        f = it["meta"]["feature"]
        d = R.rel_diff(f, fr or {}, ft or {})
        rows.append({"feature": f, "label_diff": it["meta"]["diff"], "remeasured": None if d is None else round(d, 3),
                     "label_class": it["meta"]["class"], "remeasured_class": None if d is None else (R.CLASS_NAMES[R.classify(f, d)] if R.classify(f, d) is not None else "ambiguous"),
                     "pair_kind": it["meta"]["pair_kind"]})
    out = {}
    for f in ("loudness", "pitch"):
        rs = [r for r in rows if r["feature"] == f and r["remeasured"] is not None]
        err = [abs(r["remeasured"] - r["label_diff"]) for r in rs]
        out[f] = {"n": len(rs), "median_abs_err": round(float(np.median(err)), 3) if err else None,
                  "p95_abs_err": round(float(np.percentile(err, 95)), 3) if err else None,
                  "same_class": sum(r["remeasured_class"] == r["label_class"] for r in rs),
                  "opposite_class": sum({r["remeasured_class"], r["label_class"]} == {"lower", "higher"} for r in rs)}
    out["examples"] = rows[:8]
    (out_dir / "spot_check.json").write_text(json.dumps(out, indent=1))
    return out
