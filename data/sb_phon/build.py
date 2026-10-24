"""sb_phon build logic: clip plan, per-clip checks of the spoken phonemes and measured edits, items, balancing and
validation. Signal edits live in edit.py."""
from __future__ import annotations

import collections
import math
import random
import re

import numpy as np

import lexicon as L

SR = 16000
FAMS = ["syll", "stress", "homograph", "emph", "reason", "inton", "listrise", "disfl", "minpair", "combo",
        "sound", "release"]
SOURCE = {"syll": "sb_phon_syll", "stress": "sb_phon_stress", "homograph": "sb_phon_stress", "emph": "sb_phon_emph",
          "reason": "sb_phon_emph", "inton": "sb_phon_inton", "listrise": "sb_phon_inton", "disfl": "sb_phon_disfl",
          "minpair": "sb_phon_minpair", "combo": "sb_phon_minpair", "sound": "sb_phon_sound",
          "release": "sb_phon_release"}
N_CLIPS = {"syll": 2300, "stress": 1500, "homograph": 600, "emph": 1400, "reason": 1100, "inton": 1500,
           "listrise": 450, "disfl": 1900, "minpair": 1100, "combo": 500, "sound": 1500, "release": 800}
CAL_FRAC = 0.035
PHONE_FRAC = 0.2
CARRIERS = ["The next word is {w}.", "Please repeat after me: {w}.", "Say the word {w}.", "Here is the word {w}.",
            "Our word today is {w}.", "Listen to this word: {w}.", "Now I will say {w}.", "{w}."]


class NoShuffle(random.Random):
    def shuffle(self, x, *a, **k):  # ordered "score" levels keep their natural order
        return None


# Planning

def vocab(cmu, us_gold, gb_gold):
    """Words usable for syllable / stress items: lowercase alpha, in CMUdict and in both misaki gold lexicons
    (as plain strings), all CMUdict pronunciations agree on the syllable count."""
    out = {}
    for w, prons in cmu.items():
        if not re.fullmatch(r"[a-z]{3,18}", w):
            continue
        if not isinstance(us_gold.get(w), str) or not isinstance(gb_gold.get(w), str):
            continue
        n = L.consistent(prons, L.syllables)
        if n is None or not 1 <= n <= 6:
            continue
        out[w] = {"n": n, "stress": L.consistent(prons, L.primary_index)}
    return out


def heteronyms(us_gold, gb_gold):
    """Noun/verb stress-shift homographs from misaki's gold lexicon (DEFAULT = noun reading, VERB = verb reading)."""
    out = []
    for w, v in us_gold.items():
        if not (isinstance(v, dict) and "VERB" in v and isinstance(v.get("DEFAULT"), str) and re.fullmatch(r"[a-z]+", w)):
            continue
        g = gb_gold.get(w)
        if not (isinstance(g, dict) and "VERB" in g and isinstance(g.get("DEFAULT"), str)):
            continue
        ok = True
        rec = {"word": w}
        for lang, d, gb in (("a", v, False), ("b", g, True)):
            n_p, v_p = L.misaki_primary(d["DEFAULT"], gb), L.misaki_primary(d["VERB"], gb)
            if n_p is None or v_p is None or n_p == v_p or L.misaki_syllables(d["DEFAULT"], gb) != 2 or \
                    L.misaki_syllables(d["VERB"], gb) != 2:
                ok = False
                break
            rec[lang] = {"noun": d["DEFAULT"], "verb": d["VERB"], "noun_pos": n_p, "verb_pos": v_p}
        if ok:
            out.append(rec)
    return sorted(out, key=lambda r: r["word"])


def common_words(cmu):
    out = {}
    for w in dict.fromkeys(L.COMMON):
        prons = cmu.get(w)
        if not prons:
            continue
        ph = L.consistent(prons, L.unstressed)
        if ph is None or L.syllables(prons[0]) != 1:
            continue
        out[w] = ph
    return out


def minimal_groups(common):
    """Groups of common words whose phones differ in exactly one position (same length)."""
    groups = collections.defaultdict(set)
    for w, ph in common.items():
        for i in range(len(ph)):
            groups[(len(ph), i, ph[:i] + ("*",) + ph[i + 1:])].add(w)
    out = []
    for (n, i, pat), ws in groups.items():
        ws = sorted(ws)
        phs = {common[w] for w in ws}
        if len(ws) >= 2 and len(phs) == len(ws):
            out.append({"pos": i, "len": n, "words": ws, "is_vowel": common[ws[0]][i] in L.ARPA_VOWELS})
    return sorted(out, key=lambda g: (g["len"], g["pos"], g["words"]))


def _voice(rng, split):
    return rng.choice(L.TRAIN_VOICES if split == "train" else L.CAL_VOICES)


def _speed(rng):
    return round(rng.uniform(0.85, 1.12), 3)


def plan(seed: int, scale: float, cmu, us_gold, gb_gold):
    rng = random.Random(f"{seed}:plan")
    voc = vocab(cmu, us_gold, gb_gold)
    het = heteronyms(us_gold, gb_gold)
    com = common_words(cmu)
    groups = minimal_groups(com)
    by_n = collections.defaultdict(list)
    for w, d in voc.items():
        by_n[d["n"]].append(w)
    by_stress = collections.defaultdict(list)
    for w, d in voc.items():
        if 2 <= d["n"] <= 4 and d["stress"] is not None:
            by_stress[(d["n"], d["stress"])].append(w)
    for k in by_n:
        by_n[k].sort()
    for k in by_stress:
        by_stress[k].sort()
    clips = []

    def add(fam, text, info, split=None, phone_ok=True):
        split = split or ("CAL" if rng.random() < CAL_FRAC else "train")
        v = _voice(rng, split)
        if text and text[0].isalpha():
            text = text[0].upper() + text[1:]
        clips.append({"key": f"sb_phon_{fam}_{len(clips):06d}", "fam": fam, "split": split, "voice": v,
                      "lang": "b" if v.startswith("b") else "a", "speed": _speed(rng), "text": text,
                      "phone": phone_ok and rng.random() < PHONE_FRAC, "info": info,
                      "seed": rng.randrange(1 << 30)})

    def carrier(w):
        return rng.choice(CARRIERS).format(w=w)

    n = lambda f: max(2, int(round(N_CLIPS[f] * scale)))  # noqa: E731
    # syllables: single words balanced over 1..6 syllables (+ short phrases for the total count)
    n_words = int(n("syll") * 0.82)
    for j in range(n_words):
        k = j % 6 + 1
        w = rng.choice(by_n[k])
        add("syll", carrier(w), {"words": [w], "labels": [k], "phrase": False})
    for j in range(n("syll") - n_words):
        m = rng.choice([2, 3, 3, 4])
        ws = [rng.choice(by_n[rng.choice([1, 1, 2, 2, 3])]) for _ in range(m)]
        add("syll", " ".join(ws).capitalize() + ".", {"words": ws, "labels": [voc[w]["n"] for w in ws], "phrase": True})
    # lexical stress: balanced over (n syllables, stressed position) cells that have >= 15 words
    cells = [c for c, ws in by_stress.items() if len(ws) >= 15]
    for j in range(n("stress")):
        c = cells[j % len(cells)]
        w = rng.choice(by_stress[c])
        add("stress", carrier(w), {"words": [w], "n": c[0], "pos": c[1]})
    for j in range(n("homograph")):
        h = het[j % len(het)]
        form = "noun" if j % 2 == 0 else "verb"
        clips_lang = None
        add("homograph", None, {"word": h["word"], "form": form, "het": h})
        clips_lang = clips[-1]["lang"]
        ph = h[clips_lang][form]
        frame = rng.choice(["The word is {w}.", "Listen: {w}.", "{w}.", "Next one: {w}.", "Say {w}.", "Repeat: {w}."])
        clips[-1]["text"] = frame.format(w=f"[{h['word']}](/{ph}/)")
        clips[-1]["info"].update({"pos": h[clips_lang][f"{form}_pos"], "ph": ph})
    # emphasis on one word of a sentence
    for j in range(n("emph")):
        s = re.sub(r"\[([^|\]]+)\|[^\]]+\]", r"\1", rng.choice(L.SENTENCES))
        words = s.split()
        cand = [i for i, w in enumerate(words) if w.lower() not in L.STOPWORDS]
        t = rng.choice(cand)
        add("emph", s + ".", {"words": words, "target": t, **_emph_params(rng)})
    for j in range(n("reason")):
        tpl = L.REASON[j % len(L.REASON)]
        fill = {"N": rng.choice(L.NAMES)}
        for k, vals in tpl["slots"].items():
            fill[k] = rng.choice(vals)
        s = tpl["text"].format(**fill)
        words = s.split()
        t = rng.choice(sorted(tpl["emph"]))
        meanings = {i: m.format(**fill) for i, m in tpl["emph"].items()}
        meanings = {i: m[0].upper() + m[1:] for i, m in meanings.items()}
        add("reason", s + ".", {"words": words, "target": t, "meanings": {str(i): m for i, m in meanings.items()},
                                **_emph_params(rng)})
    pats = ["falling", "rising", "rise-fall", "fall-rise"]
    for j in range(n("inton")):
        s = rng.choice(L.INTON_SENTENCES)
        add("inton", s + ".", {"pattern": pats[j % 4], "size": round(rng.uniform(7.0, 11.0), 2)})
    for j in range(n("listrise")):
        nouns = rng.sample(L.LIST_NOUNS, 4)
        s = rng.choice(L.LIST_FRAMES).format(*nouns)
        for nn in nouns:
            if nn[0] in "aeiou":
                s = s.replace(f" a {nn}", f" an {nn}")
        add("listrise", s + ".", {"nouns": nouns, "target": j % 4, "size": round(rng.uniform(6.0, 9.0), 2),
                                  "words": s.split()})
    for j in range(n("disfl")):
        text, info = make_disfluent(rng)
        add("disfl", text, info)
    vowel_groups = [g for g in groups if g["is_vowel"]]
    cons_groups = [g for g in groups if not g["is_vowel"]]
    for j in range(n("minpair")):
        g = rng.choice(vowel_groups if j % 2 else cons_groups)
        k = min(len(g["words"]), rng.choice([2, 2, 3, 4]))
        opts = rng.sample(g["words"], k)
        t = opts[0]
        phone_ok = not _fricative_contrast(g, com)
        add("minpair", carrier(t) if rng.random() < 0.6 else t.capitalize() + ".",
            {"target": t, "options": opts, "pos": g["pos"], "is_vowel": g["is_vowel"]}, phone_ok=phone_ok)
        gold_lex = gb_gold if clips[-1]["lang"] == "b" else us_gold
        clips[-1]["info"]["lex"] = {o: gold_lex.get(o) for o in opts if isinstance(gold_lex.get(o), str)}
    for j in range(n("combo")):
        g1, g2 = rng.sample(groups, 2)
        a = rng.sample(g1["words"], 2)
        b = rng.sample(g2["words"], 2)
        if set(a) & set(b):
            continue
        frame = rng.choice(["I heard {0} and then {1}.", "Write down {0}, then {1}.", "The two words are {0} and {1}.",
                            "First {0}, and then {1}.", "She said {0} and {1}."])
        add("combo", frame.format(a[0], b[0]), {"target": [a[0], b[0]], "alts": [a[1], b[1]], "frame": frame},
            phone_ok=not (_fricative_contrast(g1, com) or _fricative_contrast(g2, com)))
    kinds = ["final", "vowel", "onset", "plosive", "voicing"]
    com_words = sorted(com)
    stop_onset = [w for w in com_words if com[w][0] in L.STOPS]
    for j in range(n("sound")):
        kind = kinds[j % 5]
        if kind == "voicing":
            w = rng.choice(stop_onset)
        elif kind == "plosive":
            w = rng.choice(stop_onset) if j % 10 < 5 else rng.choice([x for x in com_words if com[x][0] not in L.STOPS])
        else:
            w = rng.choice(com_words)
        opts = sound_options(kind, w, com, rng) if kind in ("final", "vowel", "onset") else None
        if kind in ("final", "vowel", "onset") and opts is None:
            continue
        add("sound", carrier(w) if rng.random() < 0.5 else w.capitalize() + ".", {"kind": kind, "target": w,
                                                                                    "options": opts},
            phone_ok=kind in ("vowel", "plosive", "voicing"))
    stop_final = [w for w in com_words if com[w][-1] in L.STOPS and com[w][-2] in L.ARPA_VOWELS]
    for j in range(n("release")):
        w = rng.choice(stop_final)
        frame = rng.choice(["{w}.", "The last word is {w}.", "Say {w}.", "And the word is {w}.", "Now {w}."])
        add("release", frame.format(w=w), {"target": w, "released": j % 2 == 0}, phone_ok=False)
    return clips, {"vocab": len(voc), "heteronyms": [h["word"] for h in het], "common": len(com),
                   "minimal_groups": len(groups), "stress_cells": {f"{a}-{b}": len(by_stress[(a, b)]) for a, b in cells},
                   "syll_words": {k: len(v) for k, v in sorted(by_n.items())}}


def _emph_params(rng):
    mode = rng.choice(["all", "all", "pitch_energy", "pitch_duration"])
    st = round(rng.uniform(3.0, 6.0), 2)
    db = round(rng.uniform(4.0, 8.0), 2) if mode != "pitch_duration" else round(rng.uniform(1.0, 3.0), 2)
    stretch = round(rng.uniform(1.2, 1.45), 3) if mode != "pitch_energy" else 1.0
    return {"mode": mode, "st": st, "db": db, "stretch": stretch}


def _fricative_contrast(g, com):
    if g["is_vowel"]:
        return False
    return any(com[w][g["pos"]] in L.FRICATIVE_PLACE for w in g["words"])


def sound_options(kind, w, com, rng):
    f = {"final": lambda p: p[[i for i, x in enumerate(p) if x in L.ARPA_VOWELS][-1] + 1:],
         "vowel": lambda p: tuple(x for x in p if x in L.ARPA_VOWELS),
         "onset": lambda p: p[:[i for i, x in enumerate(p) if x in L.ARPA_VOWELS][0]]}[kind]
    key = f(com[w])
    if not key:
        return None
    same = [x for x in com if x != w and f(com[x]) == key and com[x] != com[w]]
    diff = [x for x in com if f(com[x]) != key and f(com[x]) and (kind == "vowel" or f(com[x])[-1 if kind == "final" else 0] != key[-1 if kind == "final" else 0])]
    if not same or len(diff) < 3:
        return None
    return [rng.choice(sorted(same))] + rng.sample(sorted(diff), 3)


DISFL_TYPES = ["filled", "marker", "restart", "editing"]
DISFL_NAME = {"filled": "filled pause (um / uh)", "marker": "discourse marker (well / you know)",
              "restart": "restart or repeated start", "editing": "editing term with a correction (I mean / sorry)"}


def make_disfluent(rng):
    raw = rng.choice(L.SENTENCES)
    m = re.search(r"\[([^|\]]+)\|([^\]]+)\]", raw)
    right, wrong = m.group(1), m.group(2)
    words = re.sub(r"\[([^|\]]+)\|[^\]]+\]", "\u0001", raw).split()
    si = words.index("\u0001")
    words[si] = right
    r = rng.random()
    k = 0 if r < 0.2 else 1 if r < 0.5 else 2 if r < 0.8 else 3
    types = sorted(rng.sample(DISFL_TYPES, k), key=DISFL_TYPES.index)
    toks = list(words)
    toks[0] = toks[0][0].upper() + toks[0][1:]
    inserted = []
    n_fill = 0
    # editing term + repair before the swappable word
    if "editing" in types:
        term = rng.choice(["I mean", "sorry", "I mean,", "no, sorry"])
        term = term.rstrip(",")
        prev = toks[si - 1] if si > 0 and toks[si - 1].lower() in ("the", "a", "on", "at", "in", "to", "of") else None
        rep = f"{wrong}, {term}, {prev.lower() + ' ' if prev else ''}{right}"
        toks[si] = rep
        inserted += term.replace(",", "").split()
    if "filled" in types:
        n_fill = rng.choice([1, 1, 2, 2, 3])
        spots = sorted(rng.sample(range(1, len(toks)), min(n_fill, len(toks) - 1)))
        n_fill = len(spots)
        for off, sp in enumerate(spots):
            f = rng.choice(["um", "uh"])
            toks.insert(sp + off, f"{f},")
            inserted.append(f)
    if "marker" in types:
        mk = rng.choice(["well", "you know", "so, like", "well, you know"])
        if rng.random() < 0.5 or mk.startswith("well"):
            toks[0] = toks[0][0].lower() + toks[0][1:] if toks[0] not in ("I",) else toks[0]
            toks.insert(0, mk[0].upper() + mk[1:] + ",")
        else:
            toks.insert(min(3, len(toks) - 1), f"{mk},")
        inserted += mk.replace(",", "").split()
    if "restart" in types:
        n = rng.choice([1, 2])
        start = " ".join(toks[:n])
        toks.insert(0, start.rstrip(",") + " —")
        if toks[1][0].isupper() and toks[1] != "I" and not toks[1].startswith(("Um", "Uh", "Well", "So")):
            toks[1] = toks[1][0].lower() + toks[1][1:]
    text = " ".join(toks).replace(" ,", ",") + "."
    text = re.sub(r",\.$", ".", text)
    return text, {"types": types, "n_fill": n_fill, "inserted": inserted, "words": words}


# Processing

def alpha_tokens(tokens):
    """Spoken word tokens [(text, phonemes, start, end)] with timestamps (TTS tokens: [text, ph, start, end])."""
    return [t for t in tokens if re.search(r"[A-Za-z]", t[0] or "") and t[1] and t[2] is not None and t[3] is not None]


def find_word(tokens, w, last=True):
    c = [t for t in alpha_tokens(tokens) if t[0].lower().strip(".,:;!?") == w.lower()]
    return (c[-1] if last else c[0]) if c else None


def char_times(ps: str, pred_dur, n_samples: int):
    """Per-phoneme-character (start, end) in seconds from Kokoro's predicted durations (index 0 = BOS)."""
    d = np.asarray(pred_dur, np.float64)
    unit = n_samples / SR / d.sum()
    cs = np.cumsum(d)
    return [((cs[k]) * unit, (cs[k + 1]) * unit) for k in range(len(ps))]


def finish(x, rng, phone, fx=None):
    """Pad to >= 2 s with a low noise floor, optional phone channel (baked in), peak-normalise."""
    x = np.asarray(x, np.float32)
    lead = rng.uniform(0.15, 0.6)
    tail = rng.uniform(0.2, 0.7)
    tot = len(x) / SR + lead + tail
    if tot < 2.0:
        extra = 2.0 - tot + rng.uniform(0.0, 0.6)
        lead += extra * 0.5
        tail += extra * 0.5
    a, b = int(lead * SR), int(tail * SR)
    floor = 10 ** (rng.uniform(-78, -62) / 20)
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


def process(spec, tts, fx=None):
    """-> (audio | None, facts). facts['drop'] names the reason when the clip is dropped."""
    import edit

    fam, info = spec["fam"], spec["info"]
    gb = spec["lang"] == "b"
    x = np.asarray(tts["audio"], np.float32)
    toks = tts["tokens"]
    rng = random.Random(spec["seed"])
    facts = {}

    def drop(r):
        return None, {"drop": r}

    if fam in ("syll",):
        got = []
        for w in info["words"]:
            t = find_word(toks, w)
            if t is None:
                return drop("word_missing")
            got.append(L.misaki_syllables(t[1], gb))
        if got != info["labels"]:
            return drop("syllables_mismatch")
        facts["n"] = sum(info["labels"])
    elif fam == "stress":
        t = find_word(toks, info["words"][0])
        if t is None:
            return drop("word_missing")
        if L.misaki_syllables(t[1], gb) != info["n"] or L.misaki_primary(t[1], gb) != info["pos"]:
            return drop("stress_mismatch")
    elif fam == "homograph":
        t = find_word(toks, info["word"])
        if t is None or L.strip_stress(t[1]) != L.strip_stress(info["ph"]) or L.misaki_primary(t[1], gb) != info["pos"]:
            return drop("homograph_mismatch")
    elif fam in ("emph", "reason"):
        al = alpha_tokens(toks)
        if [a[0].lower().strip(".,") for a in al] != [w.lower() for w in info["words"]]:
            return drop("token_alignment")
        wt = [(a[2], a[3]) for a in al]
        ti = info["target"]
        ctrl = edit.resynth(x, flat=0.5)
        y, tmap = edit.emphasis(x, wt[ti][0], wt[ti][1], info["st"], info["db"], info["stretch"], flat=0.5)
        wt2 = [(tmap(s), tmap(e)) for s, e in wt]
        s0, s1 = edit.word_stats(ctrl, wt), edit.word_stats(y, wt2)
        d_st = s1[ti]["st50"] - s0[ti]["st50"]
        d_db = s1[ti]["db"] - s0[ti]["db"]
        prom = []
        med90 = float(np.nanmedian([s["st90"] for s in s1]))
        meddb = float(np.nanmedian([s["db"] for s in s1]))
        for s in s1:
            prom.append((s["st90"] - med90) + (s["db"] - meddb) / 3.0 if s["nv"] >= 3 else -99.0)
        others = [p for i, p in enumerate(prom) if i != ti]
        margin = prom[ti] - max(others)
        facts.update({"d_st": round(d_st, 2), "d_db": round(d_db, 2), "prom_margin": round(margin, 2),
                      "words_t": [[round(a, 3), round(b, 3)] for a, b in wt2]})
        if not (d_st >= max(1.5, 0.5 * info["st"]) and margin >= 1.0 and (info["mode"] == "pitch_duration" or d_db >= 2.0)):
            return None, {"drop": "emphasis_too_weak", "d_st": round(d_st, 2), "d_db": round(d_db, 2),
                          "margin": round(margin, 2), "plan": [info["st"], info["db"], info["stretch"]],
                          "prom": [round(p, 2) for p in prom], "target": ti}
        x = y
    elif fam == "inton":
        try:
            y = edit.contour(x, info["pattern"], info["size"])
        except ValueError:
            return drop("unvoiced")
        cls, st = edit.contour_class(y)
        facts.update({"measured": cls, "contour": st})
        if cls != info["pattern"]:
            return None, {"drop": "contour_mismatch", "plan": info["pattern"], "measured": cls, "contour": st}
        x = y
    elif fam == "listrise":
        al = alpha_tokens(toks)
        if [a[0].lower().strip(".,") for a in al] != [w.lower().strip(",") for w in info["words"]]:
            return drop("token_alignment")
        wt = [(a[2], a[3]) for a in al]
        widx = [i for i, a in enumerate(al) if a[0].lower().strip(".,") in info["nouns"]]
        if len(widx) != 4:
            return drop("token_alignment")
        try:
            y = edit.word_glide(x, wt, widx[info["target"]], info["size"])
        except ValueError:
            return drop("unvoiced")
        ws = edit.word_stats(y, [wt[i] for i in widx])
        gl = [s["glide"] for s in ws]
        facts["glides"] = [round(g, 2) for g in gl]
        tg = gl[info["target"]]
        oth = [g for i, g in enumerate(gl) if i != info["target"]]
        if not (tg >= 3.0 and all((not math.isnan(g)) and g <= 1.5 for g in oth)):
            return None, {"drop": "glide_too_weak", "glides": facts["glides"], "target": info["target"]}
        x = y
    elif fam == "disfl":
        al = alpha_tokens(toks)
        spoken = [a[0].lower().strip(".,—") for a in al]
        need = collections.Counter(w.lower() for w in info["inserted"])
        have = collections.Counter(spoken)
        if any(have[w] < c for w, c in need.items()):
            return drop("insert_missing")
        fill_t = [a for a in al if a[0].lower().strip(".,") in ("um", "uh")]
        if len(fill_t) != info["n_fill"] or any(a[3] - a[2] < 0.1 for a in fill_t):
            return drop("filler_count_or_short")
    elif fam in ("minpair", "sound", "release"):
        w = info["target"]
        t = find_word(toks, w)
        if t is None:
            return drop("word_missing")
        lex = info.get("lex", {})
        if fam == "minpair":
            mine = L.strip_stress(t[1])
            for o in info["options"][1:]:
                if o in lex and L.strip_stress(lex[o]) == mine:
                    return drop("homophone_in_tts")
        if fam == "release":
            ps, pd = tts.get("ps"), tts.get("pred_dur")
            if not ps or pd is None or len(pd) != len(ps) + 2:
                return drop("no_durations")
            ct = char_times(ps, pd, len(x))
            k = max(i for i, c in enumerate(ps) if c.isalpha() or c in "ʔɾ")
            if ps[k] not in "ptkbdɡ":
                return drop("final_not_stop")
            st0, st1 = ct[k]
            v0 = t[2]
            if info["released"]:
                b = edit.burst_db(x, v0, st0, min(len(x) / SR, st1 + 0.15))
                facts["burst_db"] = round(b, 1)
                if b < -32.0:
                    return drop("release_not_audible")
            else:
                x = edit.unrelease(x, st0)
                b = edit.burst_db(x, v0, st0, len(x) / SR, gap=0.035)
                facts["burst_db"] = round(b, 1)
                if b > -45.0:
                    return drop("unrelease_failed")
                x = x[: int((st0 + 0.05) * SR)]
    elif fam == "combo":
        for w in info["target"]:
            if find_word(toks, w) is None:
                return drop("word_missing")
    y, f2 = finish(x, rng, spec["phone"], fx)
    facts.update(f2)
    facts["dur"] = round(len(y) / SR, 3)
    return y, facts


# Questions

Q = {
    "syll_choice": ["How many syllables does the word in this recording have?", "Count the syllables in the spoken word. How many are there?",
                    "The speaker presents one word. How many syllables does it contain?", "How many syllables can you hear in the target word?",
                    "Listen to the word. What is its syllable count?", "How many syllables make up the word that is said?",
                    "What is the number of syllables in the word being spoken?", "Break the spoken word into syllables. How many do you get?"],
    "syll_noul": ["Does the spoken word have exactly {k} syllables?", "Is the word you hear made up of {k} syllables?",
                  "Would you say the target word has {k} syllables?", "Does the word in the clip contain {k} syllables?",
                  "Is {k} the correct syllable count for the spoken word?", "Listen: does this word have {k} syllables?"],
    "syll_more": ["Does the spoken word have more than {k} syllables?", "Is the word longer than {k} syllables?",
                  "Does the word in the clip contain more than {k} syllables?", "Are there over {k} syllables in the word you hear?"],
    "phrase_choice": ["How many syllables does the speaker say in total?", "Count every syllable in this short phrase. What is the total?",
                      "What is the total number of syllables across all the words spoken?", "Add up the syllables of all the words you hear. How many?",
                      "How many syllables are spoken altogether in this clip?", "In total, how many syllables does this phrase contain?"],
    "stress_choice": ["Which syllable of the word carries the main stress?", "Where does the primary stress fall in the spoken word?",
                      "Listen to the word. Which syllable is pronounced with the strongest stress?", "Which syllable is accented in the word you hear?",
                      "On which syllable does the speaker put the word stress?", "Identify the stressed syllable of the target word.",
                      "The word is spoken once. Which of its syllables is stressed?", "Which syllable sounds strongest in this word?"],
    "stress_noul": ["Is the word stressed on its first syllable?", "Does the main stress of the spoken word fall on the first syllable?",
                    "Is the first syllable the stressed one in this word?", "Does this word start with a stressed syllable?",
                    "Is the opening syllable of the word the one with the main stress?", "Would you say the stress is on the first syllable?"],
    "het_form": ["Judging only by how it is stressed, is '{w}' said here as a noun or as a verb?",
                 "The speaker says '{w}'. From its stress pattern, is this the noun or the verb?",
                 "'{w}' can be a noun or a verb depending on stress. Which one do you hear?",
                 "Based on where the stress falls, which part of speech is the spoken '{w}'?",
                 "Listen to the stress in '{w}'. Is it the noun form or the verb form?",
                 "Is the word '{w}' in this clip pronounced as a noun or as a verb?"],
    "het_pos": ["Which syllable of '{w}' is stressed in this recording?", "In the spoken '{w}', is the stress on the first or second syllable?",
                "Where does the speaker put the stress in '{w}'?", "Which syllable of '{w}' gets the emphasis here?",
                "Listen to '{w}'. Which of its two syllables is stressed?", "Is '{w}' stressed on the first syllable or the second?"],
    "emph_choice": ["Which word does the speaker emphasise most?", "Which word in the sentence is spoken with extra emphasis?",
                    "One word is stressed more strongly than the rest. Which one?", "Which word stands out as the most emphasised?",
                    "The speaker highlights one word with their voice. Which word is it?", "Which of these words receives the strongest emphasis?",
                    "Which word is said louder, longer or higher than the others?", "Which word carries the sentence focus?"],
    "emph_noul": ["Is the word '{w}' the one the speaker emphasises?", "Does the speaker put the main emphasis on '{w}'?",
                  "Is '{w}' the most strongly stressed word in the sentence?", "Is '{w}' spoken with the most emphasis?",
                  "Does '{w}' stand out as the emphasised word?", "Is the focus of the sentence on the word '{w}'?"],
    "reason_choice": ["What is the speaker most likely highlighting by stressing one word?", "Given the word the speaker stresses, what is being emphasised?",
                      "Which interpretation best matches where the speaker puts the emphasis?", "What does the speaker's emphasis suggest?",
                      "Considering which word is stressed, what point is the speaker making?", "What meaning does the emphasis in this sentence convey?",
                      "Based on the stressed word, which reading is intended?", "The speaker stresses one word. What are they implying?"],
    "inton_choice": ["Which pitch pattern best describes the whole sentence?", "How does the speaker's pitch move over the sentence?",
                     "What melody does the sentence follow?", "Describe the overall intonation contour of the utterance.",
                     "Which intonation pattern does the speaker use?", "How does the voice pitch change from the start to the end of the sentence?",
                     "Which shape best matches the pitch of this sentence?", "What is the overall pitch movement in this utterance?"],
    "inton_rise": ["Does the speaker's pitch go up at the end of the sentence?", "Does the sentence end with rising intonation?",
                   "Is the final pitch movement upward?", "Does the voice rise at the end of the utterance?",
                   "Does the speaker finish on a rising pitch?", "Is there an upward pitch movement at the end?"],
    "inton_qs": ["Judging only by the intonation, does this sound like a question or a statement?",
                 "From the pitch at the end, is the speaker asking or telling?",
                 "The words could be a statement or a question. Which does the intonation suggest?",
                 "Does the melody of this sentence make it sound like a question or a statement?",
                 "Based on how the pitch ends, is this a question or a statement?",
                 "Is the speaker asking something or stating something, judging by the intonation?"],
    "listrise": ["Which word in the list is said with a rising pitch?", "On which of these words does the speaker's pitch go up?",
                 "Which listed item has an upward pitch movement?", "Which word is spoken with a rising tone?",
                 "Which of the listed words ends higher than it starts?", "Where in the list does the pitch rise?"],
    "disfl_types": ["Which kinds of disfluency does the speaker produce?", "What types of hesitation or speech errors occur in this clip?",
                    "Which disfluencies can you hear in the sentence?", "How would you classify the disfluencies in this utterance?",
                    "Which of these describes the disfluencies present?", "What sort of non-fluent speech does the speaker show?",
                    "Which disfluency types appear in the recording?", "Identify the disfluencies in this speech."],
    "disfl_noul": {"filled": ["Does the speaker use a filled pause such as 'um' or 'uh'?", "Is there an 'um' or 'uh' in the sentence?",
                              "Do you hear a filler sound like 'um' or 'uh'?", "Does the speaker hesitate with 'um' or 'uh'?"],
                   "marker": ["Does the speaker use a discourse marker such as 'well' or 'you know'?", "Is there a filler phrase like 'well' or 'you know'?",
                              "Do you hear a discourse marker like 'well', 'so' or 'you know'?", "Does the sentence contain 'well' or 'you know' used as a filler?"],
                   "restart": ["Does the speaker restart the sentence after breaking off?", "Is there a false start or a repeated sentence start?",
                               "Does the speaker begin, stop, and start over?", "Do you hear the speaker restart what they were saying?"],
                   "editing": ["Does the speaker correct themselves with a phrase like 'I mean' or 'sorry'?", "Is there a self-correction such as 'I mean' or 'sorry'?",
                               "Does the speaker fix a wrong word using an editing term?", "Do you hear the speaker say a wrong word and then correct it?"],
                   "fluent": ["Is this sentence spoken fluently, without any hesitation or correction?", "Is the speech completely fluent?",
                              "Does the speaker say the sentence without any disfluency?", "Is the utterance free of fillers, restarts and corrections?"]},
    "disfl_count": ["How many times does the speaker say 'um' or 'uh'?", "Count the filled pauses ('um', 'uh') in the clip.",
                    "How many 'um' or 'uh' fillers do you hear?", "What is the number of filler sounds like 'um' or 'uh'?"],
    "minpair": ["Which word does the speaker say?", "Which of these words is spoken in the clip?", "Which word did you hear?",
                "The speaker says one of these similar-sounding words. Which one?", "Pick the word that matches what is spoken.",
                "Which of these words is pronounced in the recording?", "Listen carefully: which word is it?", "Which word is actually said?"],
    "combo": ["Which pair of words does the speaker say?", "Which two words are spoken in the clip?",
              "Pick the word pair that matches the recording.", "Which combination of words do you hear?",
              "The speaker says two words. Which pair is it?", "Which option lists the two spoken words correctly?"],
    "snd_final": ["Which of these words ends with the same consonant sound as the spoken word?", "Which option shares its final consonant sound with the word you hear?",
                  "Pick the word whose ending consonant sound matches the target word.", "Which word has the same last consonant sound as the one spoken?"],
    "snd_vowel": ["Which of these words has the same vowel sound as the spoken word?", "Which option rhymes in its vowel with the word you hear?",
                  "Pick the word that shares the vowel sound of the target word.", "Which word contains the vowel heard in the spoken word?"],
    "snd_onset": ["Which of these words begins with the same consonant sound as the spoken word?", "Which option starts with the same sound as the word you hear?",
                  "Pick the word whose first consonant sound matches the target word.", "Which word has the same opening sound as the spoken one?"],
    "snd_plosive": ["Does the spoken word begin with a stop consonant (p, b, t, d, k or g)?", "Is the first sound of the word a plosive?",
                    "Does the word start with a plosive sound like p, t, k, b, d or g?", "Is the word's initial consonant a stop?"],
    "snd_voicing": ["Is the first consonant of the spoken word voiced or voiceless?", "Does the word begin with a voiced or an unvoiced stop?",
                    "Is the initial stop consonant voiced (b, d, g) or voiceless (p, t, k)?", "Listen to the first sound: voiced or voiceless?"],
    "release": ["Is the final stop consonant of the last word released or unreleased?", "Does the word end with an audible release burst or not?",
                "At the end of the last word, is the stop released audibly or held without release?", "How is the word-final stop produced?",
                "Is the last consonant fully released or cut off?", "Is the closing stop of the final word released or unreleased?"],
    "listrise_noul": ["Does the pitch rise on the word '{w}'?", "Is '{w}' the listed word spoken with a rising tone?",
                      "Does the speaker's voice go up on '{w}'?", "Is there an upward pitch glide on '{w}'?",
                      "Is '{w}' said with rising intonation?", "Does '{w}' end on a higher pitch than it starts?"],
    "minpair_noul": ["Does the speaker say the word '{w}'?", "Is the spoken word '{w}'?", "Did you hear the word '{w}'?",
                     "Is '{w}' the word in the recording?", "Is the word that is said '{w}'?", "Does the clip contain the word '{w}'?"],
    "release_noul": ["Can you hear the release burst of the final consonant?", "Is the final stop of the last word audibly released?",
                     "Do you hear a burst of air when the final consonant is released?",
                     "Does the last consonant end with an audible burst?", "Is the word-final stop released?"],
}

NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
             "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen"]
ORD = ["first", "second", "third", "fourth", "fifth", "sixth"]


def _noul(make, rng, qname, qtext, yes, meta, **kw):
    return make(qname, qtext, ["yes", "no"], [1.0, 0.0] if yes else [0.0, 1.0], 0 if yes else 1, "noul",
                {**meta, "answer_class": "yes" if yes else "no"}, rng, **kw)


def items_for(rec, make_item, seed):
    """Items (<= 4) for one processed clip. rec = spec + facts."""
    spec, facts = rec, rec["facts"]
    fam, info = spec["fam"], spec["info"]
    rng = random.Random(f"{seed}:{spec['key']}:items")
    out = []
    base_meta = {"family": fam, "voice": spec["voice"], "speed": spec["speed"], "text": spec["text"],
                 "phone_fx": facts.get("codec"), "licence": "synthetic (Kokoro-82M Apache-2.0; labels CMUdict BSD-2 / by construction)"}

    def make(qname, question, options, target, gold, qtype, meta, r, ordered=False):
        it = make_item(item_id=f"{spec['key']}_{len(out)}", set_name=spec["split"], source=SOURCE[fam], question=question,
                       options=options, target=target, gold=gold, rng=NoShuffle() if ordered else r,
                       audio={"bank": rec["bank"], "key": spec["key"]}, channel="clean", state_text=None,
                       meta={**base_meta, **meta, "qname": qname, "qtype": qtype})
        out.append(it)

    def choice(qname, question, options, gold, meta, ordered=False, qtype="choice"):
        t = [0.0] * len(options)
        t[gold] = 1.0
        make(qname, question, options, t, gold, qtype, meta, rng, ordered)

    q = lambda k: rng.choice(Q[k])  # noqa: E731
    if fam == "syll" and not info["phrase"]:
        k = info["labels"][0]
        style = rng.choice(["num", "word", "suffix"])
        fmt = {"num": lambda i: str(i), "word": lambda i: NUM_WORDS[i],
               "suffix": lambda i: f"{NUM_WORDS[i]} syllable{'s' if i > 1 else ''}"}[style]
        r = rng.random()
        if r < 0.65:
            lo = max(1, min(k - rng.randint(0, 3), 7 - 4))
            opts = list(range(lo, lo + 4))
            choice("syll_choice", q("syll_choice"), [fmt(i) for i in opts], opts.index(k), {"answer_class": str(k)})
        else:
            levels = list(range(1, 7))
            choice("syll_score", q("syll_choice"), [str(i) for i in levels], levels.index(k), {"answer_class": str(k)},
                   ordered=True, qtype="score")
        if rng.random() < 0.7:
            yes = rng.random() < 0.5
            kk = k if yes else rng.choice([j for j in range(max(1, k - 2), min(6, k + 2) + 1) if j != k])
            _noul(make, rng, "syll_noul", q("syll_noul").format(k=NUM_WORDS[kk]), yes, {"answer_class": None})
        if rng.random() < 0.35:
            thr = rng.choice([j for j in range(1, 6)])
            if abs(k - thr) >= 1 and not (k == thr):
                _noul(make, rng, "syll_more", q("syll_more").format(k=NUM_WORDS[thr]), k > thr, {})
    elif fam == "syll":
        tot = facts["n"]
        lo = max(2, tot - rng.randint(0, 3))
        opts = list(range(lo, lo + 4))
        choice("phrase_choice", q("phrase_choice"), [str(i) for i in opts], opts.index(tot), {"answer_class": str(tot)})
    elif fam == "stress":
        n, pos = info["n"], info["pos"]
        long_fmt = rng.random() < 0.5
        opts = [f"the {ORD[i]} syllable" if long_fmt else ORD[i] for i in range(n)]
        choice("stress_choice", q("stress_choice"), opts, pos, {"answer_class": f"{n}-{pos}", "n_syll": n})
        if rng.random() < 0.8:
            _noul(make, rng, "stress_noul", q("stress_noul"), pos == 0, {"n_syll": n})
    elif fam == "homograph":
        w = info["word"]
        form = info["form"]
        choice("het_form", q("het_form").format(w=w), ["noun", "verb"], 0 if form == "noun" else 1, {"answer_class": form})
        if rng.random() < 0.95:
            choice("het_pos", q("het_pos").format(w=w), ["first syllable", "second syllable"], info["pos"],
                   {"answer_class": str(info["pos"])})
    elif fam in ("emph", "reason"):
        words = [w.strip(",") for w in info["words"]]
        ti = info["target"]
        meta = {"d_st": facts["d_st"], "d_db": facts["d_db"], "prom_margin": facts["prom_margin"], "edit": info["mode"]}
        if fam == "emph" or rng.random() < 0.6:
            pool = [i for i in range(len(words)) if i != ti and words[i].lower() != words[ti].lower()]
            pool_c = [i for i in pool if words[i].lower() not in L.STOPWORDS]
            pool = pool_c if len(pool_c) >= 3 else pool
            k = min(len(pool), rng.choice([3, 3, 4]))
            sel = rng.sample(pool, k)
            opts = [words[ti]] + [words[i] for i in sel]
            if len({o.lower() for o in opts}) == len(opts):
                pos_class = ["early", "middle", "late"][min(2, int(3 * ti / len(words)))]
                choice("emph_choice", q("emph_choice"), opts, 0, {**meta, "answer_class": pos_class})
            if rng.random() < (0.8 if fam == "emph" else 0.5):
                yes = rng.random() < 0.5
                w = words[ti] if yes else words[rng.choice(pool)]
                _noul(make, rng, "emph_noul", q("emph_noul").format(w=w), yes, meta)
        if fam == "reason":
            mean = info["meanings"]
            keys = [k for k in mean if int(k) != ti]
            sel = rng.sample(keys, min(3, len(keys)))
            opts = [mean[str(ti)]] + [mean[k] for k in sel]
            choice("reason_choice", q("reason_choice"), opts, 0, {**meta, "answer_class": None})
    elif fam == "inton":
        pat = info["pattern"]
        names = {"falling": ["falling", "falling intonation", "pitch falls"], "rising": ["rising", "rising intonation", "pitch rises"],
                 "rise-fall": ["rise-fall", "rise then fall", "rises then falls"], "fall-rise": ["fall-rise", "fall then rise", "falls then rises"]}
        sty = rng.randrange(3)
        pats = ["falling", "rising", "rise-fall", "fall-rise"]
        choice("inton_choice", q("inton_choice"), [names[p][sty] for p in pats], pats.index(pat),
               {"answer_class": pat, "contour": facts["contour"]})
        if pat in ("rising", "falling") and rng.random() < 0.7:
            choice("inton_qs", q("inton_qs"), ["question", "statement"], 0 if pat == "rising" else 1,
                   {"answer_class": pat, "contour": facts["contour"]})
        fin = facts["contour"].get("final", 0.0)
        if abs(fin) >= 2.0 and rng.random() < 0.7:
            _noul(make, rng, "inton_rise", q("inton_rise"), fin > 0, {"contour": facts["contour"]})
    elif fam == "listrise":
        nouns = info["nouns"]
        choice("listrise", q("listrise"), list(nouns), info["target"], {"answer_class": str(info["target"]),
                                                                         "glides": facts["glides"]})
        if rng.random() < 0.7:
            yes = rng.random() < 0.5
            w = nouns[info["target"]] if yes else rng.choice([x for i, x in enumerate(nouns) if i != info["target"]])
            _noul(make, rng, "listrise_noul", q("listrise_noul").format(w=w), yes, {"glides": facts["glides"]})
    elif fam == "disfl":
        types = info["types"]
        nm = lambda ts: " + ".join(DISFL_NAME[t] for t in ts) if ts else "none (fluent speech)"  # noqa: E731
        subsets = [()] + [tuple(c) for k in (1, 2, 3) for c in _combos(DISFL_TYPES, k)]
        wrong = [s for s in subsets if s != tuple(types)]
        sel = rng.sample(wrong, 3)
        opts = [nm(types)] + [nm(s) for s in sel]
        choice("disfl_types", q("disfl_types"), opts, 0, {"answer_class": str(len(types)), "types": types})
        kind = rng.choice(DISFL_TYPES + ["fluent"])
        yes = (not types) if kind == "fluent" else (kind in types)
        _noul(make, rng, f"disfl_noul_{kind}", rng.choice(Q["disfl_noul"][kind]), yes, {"types": types})
        if rng.random() < 0.4:
            nf = info["n_fill"]
            choice("disfl_count", q("disfl_count"), ["none", "one", "two", "three"], nf, {"answer_class": str(nf)},
                   ordered=True, qtype="score")
    elif fam == "minpair":
        choice("minpair", q("minpair"), list(info["options"]), 0, {"answer_class": None, "is_vowel": info["is_vowel"]})
        if rng.random() < 0.7:
            yes = rng.random() < 0.5
            w = info["target"] if yes else rng.choice(info["options"][1:])
            _noul(make, rng, "minpair_noul", q("minpair_noul").format(w=w), yes, {"is_vowel": info["is_vowel"]})
    elif fam == "combo":
        t, a = info["target"], info["alts"]
        opts = [f"{t[0]}, {t[1]}", f"{a[0]}, {t[1]}", f"{t[0]}, {a[1]}", f"{a[0]}, {a[1]}"]
        choice("combo", q("combo"), opts, 0, {"answer_class": None})
    elif fam == "sound":
        kind, w = info["kind"], info["target"]
        com = rec["com"]
        if kind in ("final", "vowel", "onset"):
            show_ipa = rng.random() < 0.5
            opts = info["options"]
            fmt = (lambda x: f"{x} (/{L.ipa(com[x])}/)") if show_ipa else (lambda x: x)
            choice(f"snd_{kind}", q(f"snd_{kind}"), [fmt(o) for o in opts], 0, {"answer_class": None, "target_word": w})
        if kind == "plosive" or (kind != "voicing" and rng.random() < 0.5):
            _noul(make, rng, "snd_plosive", q("snd_plosive"), com[w][0] in L.STOPS, {"target_word": w})
        if com[w][0] in L.STOPS and (kind == "voicing" or rng.random() < 0.4):
            v = com[w][0] in L.VOICED_STOPS
            choice("snd_voicing", q("snd_voicing"), ["voiced", "voiceless"], 0 if v else 1,
                   {"answer_class": "voiced" if v else "voiceless", "target_word": w})
        if rng.random() < 0.6:
            k2 = rng.choice([x for x in ("final", "vowel", "onset") if x != kind])
            o2 = sound_options(k2, w, com, rng)
            if o2 is not None:
                choice(f"snd_{k2}", q(f"snd_{k2}"), list(o2), 0, {"answer_class": None, "target_word": w})
    elif fam == "release":
        rel = info["released"]
        meta = {"burst_db": facts.get("burst_db"), "target_word": info["target"]}
        choice("release", q("release"), ["released", "unreleased"], 0 if rel else 1, {**meta, "answer_class": str(rel)})
        if rng.random() < 0.8:
            _noul(make, rng, "release_noul", q("release_noul"), rel, meta)
    return out[:4]


def _combos(xs, k):
    import itertools

    return list(itertools.combinations(xs, k))


def balance(items, seed):
    """noul: exactly 50/50 per (qname, split); choice/score: each answer_class at most 2x the rarest per (qname, split)."""
    rng = random.Random(f"{seed}:balance")
    groups = collections.defaultdict(list)
    for it in items:
        groups[(it["meta"]["qname"], it["set"])].append(it)
    keep = []
    for (qn, s), rows in sorted(groups.items()):
        by = collections.defaultdict(list)
        for it in rows:
            by[it["meta"].get("answer_class")].append(it)
        if None in by and len(by) == 1:
            keep += rows
            continue
        if rows[0]["meta"]["qtype"] == "noul":
            m = min(len(by["yes"]), len(by["no"]))
            for k in ("yes", "no"):
                keep += rng.sample(by[k], m)
            continue
        m = min(len(v) for v in by.values())
        for k, v in by.items():
            keep += rng.sample(v, min(len(v), 2 * max(m, 1)))
    keep.sort(key=lambda it: it["id"])
    return keep


def validate(items, bank_index, mmsu_questions):
    errs = collections.Counter()
    keys = {"id", "set", "source", "question", "options", "target", "gold", "audio", "channel", "state_text", "meta"}
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()  # noqa: E731
    mm = set(mmsu_questions)
    mmn = {norm(x) for x in mmsu_questions}
    ids = set()
    for it in items:
        if set(it) - {"majority"} != keys:
            errs["keys"] += 1
        if not 2 <= len(it["options"]) <= 20 or len(set(it["options"])) != len(it["options"]):
            errs["options"] += 1
        if abs(sum(it["target"]) - 1) > 1e-5 or int(np.argmax(it["target"])) != it["gold"]:
            errs["target"] += 1
        if it["meta"]["qtype"] not in ("choice", "noul", "score"):
            errs["qtype"] += 1
        if it["meta"]["qtype"] == "noul" and sorted(it["options"]) != ["no", "yes"]:
            errs["noul_options"] += 1
        a = it["audio"]
        if a["key"] not in bank_index.get(a["bank"], {}):
            errs["audio_key"] += 1
        if it["question"] in mm or norm(it["question"]) in mmn:
            errs["mmsu_question_equal"] += 1
        if it["id"] in ids:
            errs["dup_id"] += 1
        ids.add(it["id"])
        if it["channel"] != "clean":
            errs["channel"] += 1
    return dict(errs)
