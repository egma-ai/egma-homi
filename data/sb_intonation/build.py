"""sb_intonation pipeline: plan, per-clip processing (WORLD contour reshaping checked with Harvest; pause insertion at
Kokoro word boundaries checked with a VAD; phone fx), items, validation. WORLD calls live in edit.py."""
from __future__ import annotations

import collections
import json
import math
import random
import re

import numpy as np

import labels as L
import sentences as S

SR = 16000
B = L.BUILDER
EN_VOICES = ["af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river",
             "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck",
             "am_santa", "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]
CAL_VOICES = ["af_nova", "bm_lewis", "am_echo"]
TRAIN_VOICES = [v for v in EN_VOICES if v not in CAL_VOICES]
CAL_FRAC = 0.035
PHONE_FRAC = 0.2
# planned clips (over-generated: verification drops some; balancing trims the rest)
N_INTON = {"real": 1500, "tts": 1500}   # 4-class contour clips per origin
N_FINAL = {"real": 380, "tts": 380}     # final-rise / final-fall clips per origin
CLASS_OVERSAMPLE = {"real": {"rising": 1.15, "rise-fall": 1.1, "final_rise": 1.2},
                    "tts": {"rising": 1.45, "fall-rise": 1.55, "rise-fall": 1.25, "falling": 1.05, "final_rise": 1.6}}
PAUSE_SAME_ROUNDS = 3                   # unpunctuated renders per sentence (all placements cut from the same audio)
PAUSE_PUNCT_ROUNDS = 3                  # punctuated renders per (sentence, placement)
N_CAL_SPK = 8


def voice_lang(v: str) -> str:
    return "b" if v[0] == "b" else "a"


def _voice(rng, split):
    return rng.choice(TRAIN_VOICES if split == "train" else CAL_VOICES)


# Planning

def ltr_pool(recs, exclude_speakers=()):
    out = []
    for r in recs:
        f = r.get("feat") or {}
        nw = len(r["text"].split())
        if not (1.6 <= r["dur"] <= 6.5 and 3 <= nw <= 18):
            continue
        if (f.get("n_pauses") or 0) != 0 or (f.get("voiced_frac") or 0) < 0.45:
            continue
        if r["speaker"] in exclude_speakers:
            continue
        out.append(r)
    out.sort(key=lambda r: r["key"])
    return out


def clean_text(t: str) -> str | None:
    """LibriTTS-R transcript -> Kokoro text for intonation clips: 4-14 words, no internal punctuation."""
    t = t.strip().strip('"').strip()
    if re.search(r"[,;:\"()\-—–]", t[:-1]) or re.search(r"[0-9]", t):
        return None
    t = re.sub(r"[.!?]+$", "", t).strip()
    nw = len(t.split())
    if not 4 <= nw <= 14 or not re.fullmatch(r"[A-Za-z' ]+", t):
        return None
    return t[0].upper() + t[1:] + "."


def plan(seed: int, scale: float, recs: list[dict], exclude_speakers=()):
    rng = random.Random(f"{seed}:plan")
    pool = ltr_pool(recs, exclude_speakers)
    by_g = collections.defaultdict(set)
    for r in pool:
        by_g[r.get("gender") or "unk"].add(r["speaker"])
    cal_spk = set()
    for g in ("female", "male"):
        cal_spk |= set(rng.sample(sorted(by_g[g]), min(N_CAL_SPK // 2, len(by_g[g]))))
    pool_tr = [r for r in pool if r["speaker"] not in cal_spk]
    pool_cal = [r for r in pool if r["speaker"] in cal_spk]
    rng.shuffle(pool_tr)
    rng.shuffle(pool_cal)
    texts = sorted({t for t in (clean_text(r["text"]) for r in recs) if t})
    rng.shuffle(texts)
    clips, tts = [], []
    import itertools

    it_tr, it_cal = itertools.cycle(pool_tr), itertools.cycle(pool_cal)
    it_txt = itertools.cycle(texts)

    def add_clip(fam, origin, info, split, tts_key=None, rec=None, extra=None):
        key = f"{B}_{fam}_{len(clips):06d}"
        c = {"key": key, "fam": fam, "origin": origin, "split": split, "info": info, "tts_key": tts_key,
             "phone": rng.random() < PHONE_FRAC, "seed": rng.randrange(1 << 30)}
        if rec is not None:
            c["rec"] = {k: rec[k] for k in ("key", "bank", "speaker", "source_id", "text", "gender", "dur")}
        c.update(extra or {})
        clips.append(c)
        return c

    def add_tts(text, split, voice=None):
        v = voice or _voice(rng, split)
        k = f"tts_{len(tts):06d}"
        tts.append({"key": k, "voice": v, "lang": voice_lang(v), "speed": round(rng.uniform(0.9, 1.1), 3), "text": text,
                    "split": split})
        return k, v

    def contour_info(pattern):
        if pattern in L.CLASSES:
            p = rng.uniform(1.0, 1.5) if pattern == "falling" else rng.uniform(1.2, 2.0)
            return {"kind": "contour", "pattern": pattern, "size": round(rng.uniform(6.0, 10.0), 2),
                    "resid": round(rng.uniform(0.3, 0.6), 2), "c": round(rng.uniform(0.42, 0.58), 3), "p": round(p, 3)}
        return {"kind": "final", "pattern": pattern, "size": round(rng.uniform(5.0, 9.0), 2),
                "resid": round(rng.uniform(0.4, 0.8), 2), "u0": round(rng.uniform(0.66, 0.78), 3)}

    for fam, counts, pats in (("inton", N_INTON, L.CLASSES), ("final", N_FINAL, L.FINALS)):
        for origin in ("real", "tts"):
            n_base = max(4, int(round(counts[origin] * scale / len(pats))))
            seq = []
            for pat in pats:  # oversample the classes whose F0 verification drops more (measured in the smoke runs)
                seq += [pat] * int(round(n_base * CLASS_OVERSAMPLE[origin].get(pat, 1.0)))
            rng.shuffle(seq)
            for pat in seq:
                split = "CAL" if rng.random() < CAL_FRAC else "train"
                info = contour_info(pat)
                if origin == "real":
                    rec = next(it_cal if split == "CAL" else it_tr)
                    add_clip(fam, "real", info, split, rec=rec)
                else:
                    k, v = add_tts(next(it_txt), split)
                    add_clip(fam, "tts", info, split, tts_key=k, extra={"voice": v})

    ents = [L.parse_entry(e) for e in S.SENTENCES]
    n_sent = len(ents) if scale >= 0.5 else max(3, int(round(len(ents) * scale * 4)))
    for si, e in enumerate(ents[:n_sent]):
        sid = f"s{si:03d}"
        for r in range(PAUSE_SAME_ROUNDS):
            split = "CAL" if rng.random() < CAL_FRAC else "train"
            k, v = add_tts(L.tts_text(e["words"], None, e["question"]), split)
            for pi in range(len(e["placements"])):
                add_clip("pause", "tts", {"sid": sid, "pi": pi, "mode": "same"}, split, tts_key=k, extra={"voice": v})
        for pi, pl in enumerate(e["placements"]):
            for r in range(PAUSE_PUNCT_ROUNDS):
                split = "CAL" if rng.random() < CAL_FRAC else "train"
                k, v = add_tts(L.tts_text(e["words"], pl["pauses"], e["question"]), split)
                add_clip("pause", "tts", {"sid": sid, "pi": pi, "mode": "punct"}, split, tts_key=k, extra={"voice": v})
    stats = {"ltr_pool": len(pool), "ltr_pool_speakers": len({r["speaker"] for r in pool}), "cal_speakers": sorted(cal_spk),
             "kokoro_texts": len(texts), "sentences": len(ents), "sentences_used": n_sent,
             "placements": sum(len(e["placements"]) for e in ents[:n_sent])}
    return clips, tts, stats


def shard_plan(clips, tts, n_shards):
    """TTS job i -> shard i % n; a clip goes with its TTS job; real clips round-robin."""
    tts_shard = {t["key"]: i % n_shards for i, t in enumerate(tts)}
    shards_c = [[] for _ in range(n_shards)]
    shards_t = [[] for _ in range(n_shards)]
    for i, t in enumerate(tts):
        shards_t[i % n_shards].append(t)
    j = 0
    for c in clips:
        if c["tts_key"]:
            shards_c[tts_shard[c["tts_key"]]].append(c)
        else:
            shards_c[j % n_shards].append(c)
            j += 1
    return shards_c, shards_t


# Processing

def alpha_tokens(tokens):
    return [t for t in tokens if re.search(r"[A-Za-z]", t[0] or "") and t[2] is not None and t[3] is not None]


def finish(x, rng, phone, fx=None):
    """Pad with a low noise floor (>= 1.6 s total), optional baked-in phone channel, peak-normalise."""
    x = np.asarray(x, np.float32)
    lead, tail = rng.uniform(0.15, 0.5), rng.uniform(0.2, 0.6)
    tot = len(x) / SR + lead + tail
    if tot < 1.6:
        lead += (1.6 - tot) / 2
        tail += (1.6 - tot) / 2
    a, b = int(lead * SR), int(tail * SR)
    floor = 10 ** (rng.uniform(-78, -64) / 20)
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    y = np.concatenate([nrng.standard_normal(a).astype(np.float32) * floor, x,
                        nrng.standard_normal(b).astype(np.float32) * floor])
    facts = {"lead_s": round(a / SR, 3)}
    if phone and fx is not None:
        codec = rng.choice(["mulaw", "alaw", "gsm", "opus"])
        y = fx.bandlimit(y)
        try:
            y = fx.ffmpeg_codec(y, codec, rng.choice([8, 10, 12]) if codec == "opus" else None)
        except Exception:  # noqa: BLE001
            codec = "mulaw"
            y = fx.ffmpeg_codec(y, codec)
        facts["codec"] = codec
    peak = float(np.max(np.abs(y)) + 1e-9)
    y = (y * (rng.uniform(0.45, 0.95) / peak)).astype(np.float32)
    return y, facts


def process_contour(spec, x, fx=None):
    import edit

    info = spec["info"]
    rng = random.Random(spec["seed"])
    try:
        y, f0med = edit.reshape(x, info)
    except ValueError:
        return None, {"drop": "unvoiced"}
    cls, st = edit.measure(y, f0med)
    pat = info["pattern"]
    if spec["fam"] == "inton":
        ok = cls == pat
    else:
        fin = st.get("final")
        ok = fin is not None and (fin >= L.FINAL_MIN if pat == "final_rise" else fin <= -L.FINAL_MIN)
    if not ok:
        return None, {"drop": "contour_mismatch", "plan": pat, "measured": cls, "stats": st}
    y2, ff = finish(y, rng, spec["phone"], fx)
    facts = {"measured": cls, "stats": st, "f0_med": round(f0med, 1), **ff}
    if not spec["phone"]:
        cls2, st2 = edit.measure(y2, f0med)
        fin2 = st2.get("final")
        if spec["fam"] == "inton" and cls2 != pat:
            return None, {"drop": "final_audio_mismatch", "plan": pat, "measured": cls2}
        if spec["fam"] == "final" and (fin2 is None or not (fin2 >= L.FINAL_MIN if pat == "final_rise" else fin2 <= -L.FINAL_MIN)):
            return None, {"drop": "final_audio_mismatch", "plan": pat}
    else:
        cls2, st2 = edit.measure(y2, f0med)
        facts["measured_after_fx"], facts["stats_after_fx"] = cls2, st2
        if spec["fam"] == "inton" and cls2 not in (pat, None):
            return None, {"drop": "fx_changed_contour", "plan": pat, "measured": cls2}
        fin2 = st2.get("final")
        if spec["fam"] == "final" and fin2 is not None and (fin2 < -1.0 if pat == "final_rise" else fin2 > 1.0):
            return None, {"drop": "fx_changed_final"}
    if spec["fam"] == "inton":
        fin = st["final"]
        facts["final_view_ok"] = bool(fin >= L.FINAL_VIEW_MIN if L.ends_rising(pat) else fin <= -L.FINAL_VIEW_MIN)
    return y2, facts


def _low_run(x, t, rel_db=35.0):
    """Duration (s) of the low-energy run of 10-ms frames around time t (0 if t is inside speech)."""
    db = L.frame_db(x)
    thr = max(float(db.max()) - rel_db, L.VAD_FLOOR_DB)
    k = int(t * SR / L.VAD_HOP)
    k = min(max(k, 0), len(db) - 1)
    if db[k] > thr:
        return 0.0
    a = k
    while a > 0 and db[a - 1] <= thr:
        a -= 1
    b = k
    while b < len(db) - 1 and db[b + 1] <= thr:
        b += 1
    return (b - a + 1) * L.VAD_HOP / SR


def insert_silences(x, words_t, plan_ins):
    """plan_ins {boundary b: (seconds, is_key)}: cut at the middle of the gap between word b and b+1 and insert zeros.
    Returns (y, {b: (start_s, end_s, is_key)} in output time)."""
    cuts = []
    for b in sorted(plan_ins):
        a, z = words_t[b][1], words_t[b + 1][0]
        cuts.append((int(round(0.5 * (a + z) * SR)), int(round(plan_ins[b][0] * SR)), b, plan_ins[b][1]))
    pieces, last = [], 0
    for cut, n, _, _ in cuts:
        seg = np.asarray(x[last:cut], np.float64).copy()
        f = min(80, len(seg) // 4)
        if f > 0:
            seg[-f:] *= np.linspace(1, 0, f)
        if last > 0:
            g = min(80, len(seg) // 4)
            seg[:g] *= np.linspace(0, 1, g)
        pieces += [seg, np.zeros(n)]
        last = cut
    tail = np.asarray(x[last:], np.float64).copy()
    if cuts:
        f = min(80, len(tail) // 4)
        tail[:f] *= np.linspace(0, 1, f)
    pieces.append(tail)
    ins, acc = {}, 0
    for cut, n, b, key in cuts:
        ins[b] = ((cut + acc) / SR, (cut + acc + n) / SR, key)
        acc += n
    return np.concatenate(pieces).astype(np.float32), ins


def process_pause(spec, x, tokens, ent, fx=None):
    info = spec["info"]
    rng = random.Random(spec["seed"])
    words = ent["words"]
    al = alpha_tokens(tokens)
    if [L.norm_word(a[0]) for a in al] != [L.norm_word(w) for w in words]:
        return None, {"drop": "token_alignment", "got": [a[0] for a in al][:20]}
    wt = [(a[2], a[3]) for a in al]
    pl = ent["placements"][info["pi"]]
    keys = set(pl["pauses"])
    n = len(words)
    if info["mode"] == "same":
        if L.silences(x, 0.22):
            return None, {"drop": "base_has_pause"}
    plan_ins = {}
    for b in sorted(keys):
        target = rng.uniform(L.KEY_MIN, L.KEY_MAX)
        nat = _low_run(x, 0.5 * (wt[b][1] + wt[b + 1][0]))
        plan_ins[b] = (max(0.04, target - nat), True)
    other = [b for b in range(n - 1) if b not in keys]
    rival = sorted({b for p in ent["placements"] for b in p["pauses"]} - keys)
    minors = []
    if other and rng.random() < 0.75:
        k = 1 if rng.random() < 0.6 or len(other) < 2 else 2
        if rival and rng.random() < 0.5:
            minors.append(rng.choice(rival))
        while len(minors) < k:
            b = rng.choice(other)
            if b not in minors:
                minors.append(b)
    for b in minors:
        plan_ins[b] = (rng.uniform(L.MINOR_MIN, L.MINOR_MAX), False)
    y, ins = insert_silences(x, wt, plan_ins)
    ok, found = L.match_pauses(L.silences(y, L.PAUSE_SIL_MIN), ins)
    if not ok:
        return None, {"drop": "pause_mismatch", "found": found}
    if min(found.values()) < L.KEY_MIN - 0.03 or max(found.values()) > L.KEY_MAX + 0.2:
        return None, {"drop": "key_duration", "found": found}
    # longest silence at every non-key boundary (incl. minor pauses + natural gaps), measured with a low threshold
    small = L.silences(y, 0.03)
    nonkey_max = 0.0
    for s, e in small:
        if not any(s <= a + 0.05 and e >= z - 0.05 for a, z, kk in ins.values() if kk):
            nonkey_max = max(nonkey_max, e - s)
    if nonkey_max > 0 and min(found.values()) < 1.6 * nonkey_max:
        return None, {"drop": "minor_too_close", "key_min": min(found.values()), "nonkey_max": nonkey_max}
    y2, ff = finish(y, rng, spec["phone"], fx)
    lead = ff["lead_s"]
    ins2 = {b: (a + lead, z + lead, kk) for b, (a, z, kk) in ins.items()}
    ok2, found2 = L.match_pauses(L.silences(y2, L.PAUSE_SIL_MIN), ins2)
    if not ok2:
        return None, {"drop": "pause_mismatch_after_fx"}
    facts = {"key_pauses": {str(b): d for b, d in sorted(found2.items())}, "minor_pauses": sorted(minors),
             "nonkey_max_s": round(nonkey_max, 3), **ff}
    return y2, facts


# Items

def _q(rng, name):
    return rng.choice(L.Q[name])


def items_for(spec, facts, ents_by_sid, make_item, bank):
    """Items (canonical option order) for one kept clip. <= 3 items per clip."""
    rng = random.Random(f"items:{spec['key']}")
    out = []
    fam, info, split = spec["fam"], spec["info"], spec["split"]
    meta0 = {"clip_kind": fam, "origin": spec["origin"], "phone_fx": bool(spec["phone"]), "codec": facts.get("codec")}
    if spec["origin"] == "real":
        r = spec["rec"]
        meta0.update({"speaker": r["speaker"], "source_ids": [r["source_id"]], "corpus": "libritts_r", "text": r["text"],
                      "licence": "LibriTTS-R CC BY 4.0"})
    else:
        meta0.update({"speaker": f"kokoro:{spec['voice']}", "voice": spec["voice"], "corpus": "kokoro",
                      "licence": "Kokoro-82M Apache-2.0 (TTS output); text: LibriTTS-R transcript (CC BY 4.0) or own"})

    def make(qkind, question, options, gold, qtype, answer_class, source, extra=None, ordered=False):
        n = len(out)
        r2 = L.NoShuffle() if ordered else random.Random(f"{spec['key']}:{qkind}:{n}")
        tgt = [0.0] * len(options)
        tgt[gold] = 1.0
        m = {**meta0, "qtype": qtype, "qkind": qkind, "answer_class": answer_class, **(extra or {})}
        out.append(make_item(item_id=f"{spec['key']}_{n}", set_name=split, source=source, question=question,
                             options=list(options), target=tgt, gold=gold, rng=r2, audio={"bank": bank, "key": spec["key"]},
                             channel="clean", state_text=None, meta=m))

    def noul(qkind, question, yes, source, extra=None):
        make(qkind, question, ["yes", "no"], 0 if yes else 1, "noul", "yes" if yes else "no", source, extra)

    if fam in ("inton", "final"):
        pat = info["pattern"]
        ex = {"contour": pat, "measured": facts["stats"], "plan": {k: v for k, v in info.items() if k != "kind"},
              "f0_verified": True}
        if fam == "inton":
            src = f"{B}_inton_{spec['origin']}"
            sty = rng.randrange(len(L.CLASS_STYLES))
            make("inton_choice", _q(rng, "inton_choice"), L.CLASS_STYLES[sty], L.CLASSES.index(pat), "choice", pat, src, ex)
            r = rng.random()
            if r < 0.5 or not facts.get("final_view_ok"):
                t = pat if rng.random() < 0.5 else rng.choice([c for c in L.CLASSES if c != pat])
                d = rng.choice(L.CLASS_NOUL_DESC[t])
                noul("inton_noul", _q(rng, "inton_noul").format(d=d), t == pat, src, {**ex, "asked": t})
            else:
                rise = L.ends_rising(pat)
                if rng.random() < 0.5:
                    make("final_choice", _q(rng, "final_choice"), rng.choice(L.FINAL_STYLES), 0 if rise else 1, "choice",
                         "rise" if rise else "fall", src, ex)
                else:
                    v = rng.choice(["rise", "fall"])
                    noul("final_noul", _q(rng, "final_noul").format(v=v), (v == "rise") == rise, src, ex)
        else:
            src = f"{B}_final_{spec['origin']}"
            rise = pat == "final_rise"
            make("final_choice", _q(rng, "final_choice"), rng.choice(L.FINAL_STYLES), 0 if rise else 1, "choice",
                 "rise" if rise else "fall", src, ex)
            v = rng.choice(["rise", "fall"])
            noul("final_noul", _q(rng, "final_noul").format(v=v), (v == "rise") == rise, src, ex)
        return out

    # Pause meaning
    ent = ents_by_sid[info["sid"]]
    words, pls, pi = ent["words"], ent["placements"], info["pi"]
    pl = pls[pi]
    bal = info["sid"] if split == "train" else ""
    ex = {"sentence_id": info["sid"], "sentence_type": ent["type"], "words": " ".join(words), "placement": pi,
          "pauses_after": [words[b] for b in pl["pauses"]], "pause_boundaries": pl["pauses"], "render_mode": info["mode"],
          "key_pauses_s": facts["key_pauses"], "minor_pause_boundaries": facts["minor_pauses"],
          "nonkey_max_s": facts["nonkey_max_s"], "bal_group": bal}
    src_m, src_w = f"{B}_pause_meaning", f"{B}_pause_where"
    opts = [rng.choice(p["meanings"]) for p in pls]
    if ent["foil"] and rng.random() < 0.6:
        opts.append(ent["foil"])
    make("pm_meaning", _q(rng, "pm_meaning"), opts, pi, "choice", str(pi) if bal else "any", src_m, ex)
    kinds = ["pm_word", "pm_meaning_noul", "pm_word_noul"]
    rng.shuffle(kinds)
    want = 2 if rng.random() < 0.35 else 1
    weights = {"pm_word": 0.4, "pm_meaning_noul": 0.35, "pm_word_noul": 0.25}
    first = rng.choices(kinds, [weights[k] for k in kinds])[0]
    order = [first] + [k for k in kinds if k != first]
    done = 0
    for kind in order:
        if done >= want:
            break
        if kind == "pm_word":
            sty = rng.randrange(len(L.WORD_OPT_STYLES))
            o = [L.placement_label(words, p["pauses"], sty) for p in pls]
            used = {tuple(p["pauses"]) for p in pls} | {(b,) for p in pls for b in p["pauses"]}
            cand = [b for b in range(len(words) - 1) if (b,) not in used]
            rng.shuffle(cand)
            for b in cand[: rng.choice([1, 1, 2])]:
                o.append(L.boundary_label(words, b, sty))
            if len(set(o)) != len(o):
                continue
            make("pm_word", _q(rng, "pm_word"), o, pi, "choice", str(pi) if bal else "any", src_w, ex)
        elif kind == "pm_meaning_noul":
            yes = rng.random() < 0.5
            if yes:
                m = rng.choice(pl["meanings"])
            else:
                pool = [mm for j, p in enumerate(pls) if j != pi for mm in p["meanings"]]
                m = ent["foil"] if ent["foil"] and rng.random() < 0.25 else rng.choice(pool)
            noul("pm_meaning_noul", _q(rng, "pm_meaning_noul").format(m=m), yes, src_m, {**ex, "asked_meaning": m})
        else:
            uniq = lambda b: sum(1 for w in words if L.norm_word(w) == L.norm_word(words[b])) == 1  # noqa: E731
            yes_c = [b for b in pl["pauses"] if uniq(b)]
            minor = set(facts["minor_pauses"])
            rivals = [b for p in pls for b in p["pauses"] if b not in pl["pauses"] and b not in minor and uniq(b)]
            others = [b for b in range(len(words) - 1) if b not in pl["pauses"] and b not in minor and uniq(b)]
            yes = rng.random() < 0.5
            if yes and yes_c:
                b = rng.choice(yes_c)
            elif not yes and (rivals or others):
                b = rng.choice(rivals) if rivals and rng.random() < 0.6 else rng.choice(others or rivals)
            else:
                continue
            w = words[b].strip(",.?")
            noul("pm_word_noul", _q(rng, "pm_word_noul").format(w=w), yes, src_w, {**ex, "asked_word": w})
        done += 1
    return out


# Validation

def validate(items, bank_index, durations, mmsu_questions, sentence_texts):
    errs = collections.Counter()
    keys_needed = {"id", "set", "source", "question", "options", "target", "gold", "audio", "channel", "state_text", "meta"}
    ids = set()
    for it in items:
        if set(it) < keys_needed:
            errs["schema"] += 1
        if it["id"] in ids:
            errs["dup_id"] += 1
        ids.add(it["id"])
        o, t = it["options"], it["target"]
        if not 2 <= len(o) <= 20 or len(set(o)) != len(o) or len(t) != len(o):
            errs["options"] += 1
        if abs(sum(t) - 1) > 1e-6 or int(np.argmax(t)) != it["gold"]:
            errs["target"] += 1
        m = it["meta"]
        if m.get("qtype") not in ("choice", "noul", "score"):
            errs["qtype"] += 1
        if m.get("qtype") == "noul" and sorted(o) != ["no", "yes"]:
            errs["noul_options"] += 1
        a = it["audio"]
        if a is None or a["key"] not in bank_index.get(a["bank"], {}):
            errs["audio_missing"] += 1
        else:
            d = durations.get(a["key"], 0)
            if not 1.0 <= d <= 30.0:
                errs["duration"] += 1
        if it["channel"] != "clean":
            errs["channel"] += 1
        if not it["source"].startswith(B + "_") or it["source"].startswith(("sb_count_", "sb_compare_", "sb_phon_")):
            errs["source_name"] += 1
        if it["set"] not in ("train", "CAL"):
            errs["set"] += 1
    n_eq = L.mmsu_equal_count([it["question"] for it in items], mmsu_questions)
    n_tpl = L.mmsu_equal_count(L.all_templates(), mmsu_questions)
    n_sent = L.mmsu_equal_count(sentence_texts, mmsu_questions)
    if n_eq or n_tpl or n_sent:
        errs["mmsu_question_equal"] += n_eq + n_tpl + n_sent
    noul_bal = collections.defaultdict(collections.Counter)
    for it in items:
        if it["meta"]["qtype"] == "noul":
            noul_bal[(it["source"], it["meta"]["qkind"], it["set"])][it["meta"]["answer_class"]] += 1
    for k, c in noul_bal.items():
        if c["yes"] != c["no"]:
            errs["noul_balance"] += 1
    per_clip = collections.Counter((it["audio"]["bank"], it["audio"]["key"]) for it in items)
    if per_clip and max(per_clip.values()) > 4:
        errs["items_per_clip"] += 1
    tr_spk = {it["meta"].get("speaker") for it in items if it["set"] == "train"}
    cal_spk = {it["meta"].get("speaker") for it in items if it["set"] == "CAL"}
    if tr_spk & cal_spk:
        errs["cal_speaker_overlap"] += 1
    return dict(errs), {"mmsu_questions_loaded": len(mmsu_questions), "items_question_equal_mmsu": n_eq,
                        "templates_equal_mmsu": n_tpl, "pause_sentences_equal_mmsu_question": n_sent,
                        "train_cal_speaker_overlap": sorted(x for x in tr_spk & cal_spk if x)}
