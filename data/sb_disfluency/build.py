"""sb_disfluency: disfluency combinations and verse rhythm (planning, per-clip processing, items, validation).
Labels are exact by construction: scripted text plus CMUdict / misaki lexicon checks on the TTS output."""
from __future__ import annotations

import collections
import itertools
import random
import re

import numpy as np

import lexbank as B
import lexicon as L  # data/sb_phon/lexicon.py (CMUdict parsing + misaki helpers)

SR = 16000
FAMS = ["disfl", "verse"]
SOURCE = {"disfl": "sb_disfluency_disfl", "verse": "sb_disfluency_verse"}
N_CLIPS = {"disfl": 2350, "verse": 1950}
CAL_FRAC = 0.035
PHONE_FRAC = 0.2
TYPES = ["filled", "marker", "restart", "editing"]
COMBOS = [tuple(t for i, t in enumerate(TYPES) if m >> i & 1) for m in range(16)]
COMBOS.sort(key=lambda c: (len(c), [TYPES.index(t) for t in c]))


class NoShuffle(random.Random):
    def shuffle(self, x, *a, **k):  # ordered "score" levels keep their natural order
        return None


def combo_key(types) -> str:
    return "+".join(types) if types else "none"


def _voice(rng, split):
    return rng.choice(L.TRAIN_VOICES if split == "train" else L.CAL_VOICES)


# Disfluency text

def check_base_sentence(raw: str) -> bool:
    words = re.sub(r"\[([^|\]]+)\|([^\]]+)\]", r"\1 \2", raw).lower().split()
    return raw.count("[") == 1 and not any(w in B.DISFL_VOCAB for w in words) and not re.search(r"[\d'\-]", raw)


def base_sentences():
    return [s for s in B.DISFL_SENTENCES if check_base_sentence(s)]


def _lower_first(w):
    return w if w in ("I", "Grandma") else w[0].lower() + w[1:]


def make_disfluent(rng, types, raw):
    """Script one sentence with exactly the disfluency `types`. Returns (text, info) where info['spoken'] is the
    expected spoken word sequence and info['kinds'] the role of each spoken word (w / filled / marker / rfrag /
    ewrong / eterm / eright)."""
    m = re.search(r"\[([^|\]]+)\|([^\]]+)\]", raw)
    a, b = m.group(1), m.group(2)
    right, wrong = (a, b) if rng.random() < 0.5 else (b, a)
    words = re.sub(r"\[([^|\]]+)\|[^\]]+\]", "\u0001", raw).split()
    si = words.index("\u0001")
    words[si] = right
    n = len(words)
    before = collections.defaultdict(list)  # boundary b (before word b) -> [(kind, text)]
    edit = None
    if "editing" in types:
        edit = rng.choice(B.EDIT_TERMS)
    restart = None
    if "restart" in types:
        if rng.random() < 0.4:
            restart = ("abandon", 0, rng.choice(B.ABANDONED).split())
        else:
            for _ in range(50):
                k = rng.choice([1, 1, 2])
                p = rng.choice([0, 0] + list(range(1, n - 2)))
                if p + k <= n - 1 and not (p <= si < p + k) and p != si:
                    restart = ("repeat", p, words[p:p + k])
                    break
            if restart is None:
                restart = ("abandon", 0, rng.choice(B.ABANDONED).split())
    used = {restart[1]} if restart else set()
    markers = []
    if "marker" in types:
        nm = 1 if rng.random() < 0.7 else 2
        opts = ["start", "mid"] if nm == 2 else [rng.choice(["start", "start", "mid"])]
        for where in opts:
            if where == "start":
                markers.append((0, rng.choice(B.MARKER_START)))
            else:
                cand = [x for x in range(2, n - 1) if x not in used and x != si + 1]
                if cand:
                    bnd = rng.choice(cand)
                    used.add(bnd)
                    markers.append((bnd, rng.choice(B.MARKER_MID)))
                else:
                    markers.append((0, rng.choice(B.MARKER_START)))
        if len({mk for _, mk in markers}) < len(markers) or len({p for p, _ in markers}) < len(markers):
            markers = markers[:1]
    fills = []
    if "filled" in types:
        nf = rng.choices([1, 2, 3], [0.5, 0.35, 0.15])[0]
        cand = [x for x in range(0, n) if x not in used and x not in {p for p, _ in markers if p > 0}]
        for bnd in sorted(rng.sample(cand, min(nf, len(cand)))):
            fills.append((bnd, rng.choices(B.FILLED, B.FILLED_W)[0]))
    # assemble: at each boundary: start marker, filler, restart fragment, then the word
    toks, kinds = [], []  # display tokens; kinds per display token

    def emit(text, kind):
        toks.append(text)
        kinds.append(kind)

    mk_at = dict(markers)
    fill_at = collections.defaultdict(list)
    for bnd, f in fills:
        fill_at[bnd].append(f)
    for i in range(n):
        if i in mk_at:
            if i > 0 and not toks[-1].endswith((",", "—")):
                toks[-1] += ","
            emit(mk_at[i] + ",", "marker")
        for f in fill_at.get(i, []):
            if i > 0 and toks and not toks[-1].endswith((",", "—")):
                toks[-1] += ","
            emit(f + ",", "filled")
        if restart and restart[1] == i:
            frag = list(restart[2])
            if toks:
                frag[0] = _lower_first(frag[0])
            frag[-1] = frag[-1] + " —"
            for w in frag:
                emit(w, "rfrag")
        if i == si and edit:
            emit(wrong + ",", "ewrong")
            emit(edit + ",", "eterm")
            emit(right, "eright")
        else:
            emit(words[i], "w")
    # capitalisation: first display token upper, the base sentence's first word lower if preceded by anything
    first_w = kinds.index("w") if kinds[0] != "w" else 0
    if first_w > 0 or kinds[0] != "w":
        toks[first_w] = _lower_first(toks[first_w])
    toks[0] = toks[0][0].upper() + toks[0][1:]
    text = " ".join(toks).rstrip(",") + "."
    text = re.sub(r",\.$", ".", text)
    spoken, sk = [], []
    for t, k in zip(toks, kinds):
        for w in t.split():
            w2 = w.strip(",.—").lower()
            if w2:
                spoken.append(w2)
                sk.append(k)
    info = {"types": list(types), "spoken": spoken, "kinds": sk, "n_fill": len(fills), "fillers": [f for _, f in fills],
            "markers": [mk for _, mk in markers], "edit_term": edit, "restart": None if not restart else
            {"kind": restart[0], "frag": restart[2]}, "wrong": wrong if edit else None, "right": right, "base": raw}
    return text, info


def disfl_labels(info) -> dict:
    """Recompute the type set from the scripted word roles (label rule, independent of the plan)."""
    k = set(info["kinds"])
    sp = info["spoken"]
    has_f = any(w in B.FILLED for w, kk in zip(sp, info["kinds"]) if kk == "filled")
    has_m = "marker" in k
    has_r = "rfrag" in k
    has_e = "eterm" in k and "ewrong" in k and "eright" in k
    return {"filled": has_f, "marker": has_m, "restart": has_r, "editing": has_e}


# Verse

SHAPE_DIG = {"U": "0", "S": "1"}


def _stress_digits(pron):
    return "".join(p[-1] for p in pron if p[-1].isdigit())


def cmu_rhyme(pron):
    """Phones from the last primary-stressed vowel to the end (stress digits removed)."""
    idx = [i for i, p in enumerate(pron) if p.endswith("1")]
    return tuple(L.base(p) for p in pron[idx[-1]:]) if idx else None


def cmu_pre(pron):
    """The phone right before the rhyme part ('' if none): equal pre-phones = identical rhyme (night / knight)."""
    idx = [i for i, p in enumerate(pron) if p.endswith("1")]
    return L.base(pron[idx[-1] - 1]) if idx and idx[-1] > 0 else ""


def misaki_rhyme(ps: str, gb: bool = False):
    if not ps or "ˈ" not in ps:
        return None
    k = ps.rfind("ˈ")
    tail = ps[k + 1:].replace("ˌ", "")
    nuc = L.misaki_nuclei(tail, gb)
    return tail[nuc[0]:] if nuc else None


def gold_vals(lex, w):
    """misaki gold-lexicon pronunciations of w (all POS variants), or [] if absent."""
    v = lex.get(w)
    if isinstance(v, str):
        return [v]
    if isinstance(v, dict):
        return [x for x in v.values() if isinstance(x, str)]
    return []


def verse_vocab(cmu, us_gold, gb_gold):
    """Category -> verified words: every CMUdict and misaki gold pronunciation must match the declared shape
    (syllable count and stress). Spoken lines are re-checked on the TTS phonemes in process()."""
    cats, dropped, shape = {}, collections.defaultdict(list), {}
    for cat, (shp, words) in B.BANK.items():
        keep = []
        for w in dict.fromkeys(words):
            prons, u, g = cmu.get(w), gold_vals(us_gold, w), gold_vals(gb_gold, w)
            ok = bool(prons)
            if ok and shp in ("U", "S"):
                ok = L.consistent(prons, L.syllables) == 1 and all(L.misaki_syllables(x) == 1 for x in u) and \
                    all(L.misaki_syllables(x, True) == 1 for x in g)
            elif ok:
                ok = L.consistent(prons, _stress_digits) == shp and \
                    all(L.misaki_syllables(x) == len(shp) and L.misaki_primary(x) == shp.index("1") for x in u) and \
                    all(L.misaki_syllables(x, True) == len(shp) and L.misaki_primary(x, True) == shp.index("1")
                        for x in g)
            (keep.append(w) if ok else dropped[cat].append(w))
        cats[cat] = keep
        shape[cat] = "".join(SHAPE_DIG.get(c, c) for c in shp) if shp in ("U", "S") else shp
    cats["V2i_ed"] = [w for w in cats["V2i"] if B.REGULAR_ED(w)]
    cats["V1_ed"] = [w for w in cats["V1"] if B.REGULAR_ED(w)]
    shape["V2i_ed"], shape["V1_ed"] = shape["V2i"], shape["V1"]
    return cats, shape, dict(dropped)


def rhyme_classes(cats, cmu, us_gold, gb_gold):
    """Rhyme key = (CMUdict rime, misaki US rime, misaki GB rime); words rhyme iff all three agree."""
    key_of, pre_of = {}, {}
    for cat in sorted(B.END_CATS):
        for w in cats[cat]:
            ck = L.consistent(cmu[w], cmu_rhyme)
            pre = L.consistent(cmu[w], cmu_pre)
            us_k = {misaki_rhyme(x) for x in gold_vals(us_gold, w)}
            gb_k = {misaki_rhyme(x, True) for x in gold_vals(gb_gold, w)}
            uk = us_k.pop() if len(us_k) == 1 else None
            gk = gb_k.pop() if len(gb_k) == 1 else None
            if ck and uk and gk and pre is not None:
                key_of[w] = (ck, uk, gk)
                pre_of[w] = pre
    classes = collections.defaultdict(list)
    for w, k in key_of.items():
        if w not in classes[k]:
            classes[k].append(w)
    return dict(classes), key_of, pre_of


def strict_nonrhyme(k1, k2) -> bool:
    return k1[0] != k2[0] and k1[1] != k2[1] and k1[2] != k2[2]


def templates(cats, shape):
    """Meter -> list of templates (frame, chunk category sequences) whose stress pattern equals the meter."""
    out = collections.defaultdict(list)
    pats = {p: m for m, (_, p) in B.METERS.items()}
    for frame in B.FRAMES:
        for combo in itertools.product(*[B.CHUNKS[r] for r in frame]):
            seq = [c for ch in combo for c in ch]
            if seq[-1] not in B.END_CATS or any(not cats.get(c) for c in seq):
                continue
            pat = "".join(shape[c] for c in seq)
            if pat in pats:
                out[pats[pat]].append({"frame": list(frame), "chunks": [list(c) for c in combo]})
    return dict(out)


def line_pattern(words_cats, shape):
    return "".join(shape[c] for _, c in words_cats)


def realize(tpl, cats, rng, end_word):
    seq = [c for ch in tpl["chunks"] for c in ch]
    used, wc = {end_word}, []
    for i, c in enumerate(seq):
        if i == len(seq) - 1:
            w = end_word
        else:
            pool = [x for x in cats[c] if x not in used]
            w = rng.choice(pool)
        used.add(w)
        wc.append((w, c))
    words = [w for w, _ in wc]
    if tpl["frame"][0] == "PP":
        k = len(tpl["chunks"][0])
        words = words[:k - 1] + [words[k - 1] + ","] + words[k:]
    text = " ".join(words)
    return {"text": text[0].upper() + text[1:] + ".", "words": [w for w, _ in wc], "cats": [c for _, c in wc]}


def make_line(meter, cls_words, avoid_pre, avoid_words, ctx, rng):
    cats, tpls, pre_of = ctx["cats"], ctx["templates"], ctx["pre_of"]
    order = list(range(len(tpls[meter])))
    rng.shuffle(order)
    for ti in order:
        tpl = tpls[meter][ti]
        last = tpl["chunks"][-1][-1]
        pool = [w for w in cls_words if w in cats[last] and w not in avoid_words and pre_of[w] not in avoid_pre]
        if pool:
            ln = realize(tpl, cats, rng, rng.choice(pool))
            ln.update({"meter": meter, "pattern": line_pattern(list(zip(ln["words"], ln["cats"])), ctx["shape"]),
                       "parts": [ctx["shape"][c] for c in ln["cats"]],
                       "end": ln["words"][-1], "rkey": ctx["key_of"][ln["words"][-1]]})
            ln["n_syll"] = len(ln["pattern"])
            return ln
    return None


def alt_meter(meter, rng):
    pat = B.METERS[meter][1]
    same = [m for m, (f, p) in B.METERS.items() if len(p) == len(pat) and p != pat]
    far = [m for m, (f, p) in B.METERS.items() if abs(len(p) - len(pat)) >= 2]
    if same and rng.random() < 0.45:
        return rng.choice(same), "stress"
    return rng.choice(far), "count"


def plan_verse(rng, ctx, split):
    """Spoken line (meter M, rhyme class R) + the 2x2 candidate set {M,M'} x {R,R'}; exactly one candidate (M,R)
    keeps both the meter and the rhyme. Returns info or None."""
    fam_meters = ctx["fam_meters"]
    meter = rng.choice(fam_meters[ctx["want_family"]])
    rich = ctx["rich"]
    for _ in range(40):
        rk = rng.choice(rich)
        R = ctx["classes"][rk]
        sp = make_line(meter, R, set(), set(), ctx, rng)
        if sp is None:
            continue
        pre = {ctx["pre_of"][sp["end"]]}
        corr = make_line(meter, R, pre, {sp["end"]}, ctx, rng)
        if corr is None:
            continue
        pre2 = pre | {ctx["pre_of"][corr["end"]]}
        m2, how = alt_meter(meter, rng)
        mr = make_line(m2, R, pre2, {sp["end"], corr["end"]}, ctx, rng)
        if mr is None:
            continue
        slant = rng.random() < 0.5
        k0 = sp["rkey"]
        alts = [k for k in rich if strict_nonrhyme(k, k0) and len(ctx["classes"][k]) >= 2]
        sl = [k for k in alts if k[0][0] == k0[0][0] or k[0][1:] == k0[0][1:]]  # same vowel or same coda
        pool = sl if (slant and sl) else alts
        rk2 = rng.choice(pool)
        R2 = ctx["classes"][rk2]
        a = make_line(meter, R2, set(), {sp["end"]}, ctx, rng)
        if a is None:
            continue
        b = make_line(m2, R2, {ctx["pre_of"][a["end"]]}, {sp["end"], a["end"]}, ctx, rng)
        if b is None:
            continue
        cands = {"MR": corr, "MR'": a, "M'R": mr, "M'R'": b}
        texts = [sp["text"]] + [c["text"] for c in cands.values()]
        if len(set(texts)) != len(texts):
            continue
        return {"spoken": sp, "meter": meter, "family": B.METERS[meter][0], "alt_meter": m2, "alt_how": how,
                "slant": bool(slant and sl), "cands": cands}
    return None


def verse_ok(spoken, cand) -> bool:
    """Label rule: a candidate continues the line iff same stress pattern (=> same syllable count) and its last word
    rhymes (all three rhyme keys equal; identical rhymes excluded at construction)."""
    return cand["pattern"] == spoken["pattern"] and tuple(map(tuple, cand["rkey"])) == tuple(map(tuple, spoken["rkey"]))


# Planning

def plan(seed: int, scale: float, cmu, us_gold, gb_gold):
    rng = random.Random(f"{seed}:plan")
    cats, shape, vdrop = verse_vocab(cmu, us_gold, gb_gold)
    classes, key_of, pre_of = rhyme_classes(cats, cmu, us_gold, gb_gold)
    tpls = templates(cats, shape)
    fam_meters = collections.defaultdict(list)
    for m, (f, _) in B.METERS.items():
        if tpls.get(m):
            fam_meters[f].append(m)

    def n_distinct_pre(ws):
        return len({pre_of[w] for w in ws})

    rich = sorted([k for k, ws in classes.items() if n_distinct_pre(ws) >= 3 and any(w in cats["N1"] for w in ws)])
    ctx = {"cats": cats, "shape": shape, "templates": tpls, "classes": classes, "key_of": key_of, "pre_of": pre_of,
           "rich": rich, "fam_meters": dict(fam_meters)}
    sents = base_sentences()
    clips = []

    def add(fam, text, info, split, speed):
        v = _voice(rng, split)
        clips.append({"key": f"sb_disfluency_{fam}_{len(clips):06d}", "fam": fam, "split": split, "voice": v,
                      "lang": "b" if v.startswith("b") else "a", "speed": speed, "text": text,
                      "phone": rng.random() < PHONE_FRAC, "info": info, "seed": rng.randrange(1 << 30)})

    n = lambda f: max(16 if f == "disfl" else 8, int(round(N_CLIPS[f] * scale)))  # noqa: E731
    for j in range(n("disfl")):
        types = COMBOS[j % 16]
        split = "CAL" if rng.random() < CAL_FRAC else "train"
        text, info = make_disfluent(rng, types, rng.choice(sents))
        lab = disfl_labels(info)
        assert sorted(t for t, v in lab.items() if v) == sorted(types), (text, types, lab)
        add("disfl", text, info, split, round(rng.uniform(0.85, 1.12), 3))
    n_fail = 0
    for j in range(n("verse")):
        split = "CAL" if rng.random() < CAL_FRAC else "train"
        ctx["want_family"] = B.FAMILIES[j % 4]
        info = plan_verse(rng, ctx, split)
        if info is None:
            n_fail += 1
            continue
        ok = [k for k, c in info["cands"].items() if verse_ok(info["spoken"], c)]
        assert ok == ["MR"], ok
        add("verse", info["spoken"]["text"], info, split, round(rng.uniform(0.85, 1.05), 3))
    pstats = {"base_sentences": len(sents), "base_sentences_rejected": len(B.DISFL_SENTENCES) - len(sents),
              "verse_vocab": {c: len(v) for c, v in cats.items()}, "verse_vocab_dropped": vdrop,
              "rhyme_classes": len(classes), "rich_rhyme_classes": len(rich),
              "templates_per_meter": {m: len(v) for m, v in sorted(tpls.items())}, "verse_plan_failures": n_fail}
    return clips, pstats, ctx


# Processing

def alpha_tokens(tokens):
    return [t for t in tokens if re.search(r"[A-Za-z]", t[0] or "") and t[2] is not None and t[3] is not None]


def finish(x, rng, phone, fx=None):
    """Pad to >= 2 s with a low noise floor, optional phone channel (baked in), peak-normalise (as sb_phon)."""
    x = np.asarray(x, np.float32)
    lead, tail = rng.uniform(0.15, 0.6), rng.uniform(0.2, 0.7)
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


def _quiet_point(x, t0, t1):
    """Lowest-energy 10 ms frame centre in [t0, t1] (seconds)."""
    a, b = max(0, int(t0 * SR)), min(len(x), int(t1 * SR))
    if b - a < 320:
        return (a + b) // 2
    w = 160
    e = np.convolve(x[a:b] ** 2, np.ones(w) / w, mode="valid")
    return a + int(np.argmin(e)) + w // 2


def apply_timing(x, edits, rng):
    """edits: list of ('pause', sample, dur_s) / ('cut', s0, s1); applied from the end backwards."""
    x = np.asarray(x, np.float32).copy()
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    fade = int(0.006 * SR)
    for e in sorted(edits, key=lambda e: -e[1]):
        if e[0] == "cut":
            s0, s1 = e[1], e[2]
            if s1 - s0 <= fade:
                continue
            x[s0:s0 + fade] *= np.linspace(1, 0, fade, dtype=np.float32)
            x[s0 + fade:s1] = nrng.standard_normal(s1 - s0 - fade).astype(np.float32) * 1e-4
        else:
            p, d = e[1], int(e[2] * SR)
            pa = x[:p].copy()
            pb = x[p:].copy()
            k = min(fade, len(pa), len(pb))
            if k:
                pa[-k:] *= np.linspace(1, 0, k, dtype=np.float32)
                pb[:k] *= np.linspace(0, 1, k, dtype=np.float32)
            x = np.concatenate([pa, nrng.standard_normal(d).astype(np.float32) * 1e-4, pb])
    return x


FILL_VOWEL = {"um": "ʌ", "uh": "ʌ", "er": "ɜ"}


def process(spec, tts, fx=None):
    """-> (audio | None, facts). facts['drop'] names the reason when the clip is dropped."""
    fam, info = spec["fam"], spec["info"]
    gb = spec["lang"] == "b"
    x = np.asarray(tts["audio"], np.float32)
    al = alpha_tokens(tts["tokens"])
    rng = random.Random(spec["seed"])
    facts = {}

    def drop(r, **kw):
        return None, {"drop": r, **kw}

    spoken = [a[0].lower().strip(".,—!?;:") for a in al]
    if any(not a[1] for a in al):
        return drop("oov_word", spoken=spoken)
    if fam == "disfl":
        if spoken != info["spoken"]:
            return drop("token_mismatch", spoken=spoken, want=info["spoken"])
        kinds = info["kinds"]
        for a, k in zip(al, kinds):
            if k == "filled":
                w = a[0].lower().strip(".,")
                if a[3] - a[2] < 0.1 or FILL_VOWEL[w] not in a[1]:
                    return drop("filler_short_or_wrong", ph=a[1], dur=round(a[3] - a[2], 3))
        edits, pauses = [], []
        nw = len(al)

        def gap_point(i):  # boundary after spoken word i
            return _quiet_point(x, al[i][3] - 0.02, al[i + 1][2] + 0.02)

        for i in range(nw - 1):
            k, k2 = kinds[i], kinds[i + 1]
            if k == "filled" and rng.random() < 0.5:
                pauses.append((i, rng.uniform(0.08, 0.35), "after_filler"))
            elif k2 == "filled" and rng.random() < 0.3:
                pauses.append((i, rng.uniform(0.05, 0.25), "before_filler"))
            elif k == "rfrag" and k2 != "rfrag":
                if rng.random() < 0.85:
                    pauses.append((i, rng.uniform(0.1, 0.4), "restart_break"))
                dur = al[i][3] - al[i][2]
                if rng.random() < 0.4 and dur >= 0.18:
                    frac = rng.uniform(0.45, 0.7)
                    s0 = int((al[i][2] + frac * dur) * SR)
                    s1 = int(min(al[i][3] + 0.01, al[i + 1][2]) * SR)
                    if s1 > s0:
                        edits.append(("cut", s0, s1))
                        facts["restart_cut"] = round(frac, 2)
            elif k == "eterm" and k2 == "eright" and rng.random() < 0.3:
                pauses.append((i, rng.uniform(0.05, 0.2), "after_editing_term"))
        if rng.random() < 0.35:  # a plain pause between two fluent words, so silence alone is not a cue
            cand = [i for i in range(nw - 1) if kinds[i] == "w" and kinds[i + 1] == "w"]
            if cand:
                pauses.append((rng.choice(cand), rng.uniform(0.15, 0.45), "fluent_boundary"))
        for i, d, why in pauses:
            edits.append(("pause", gap_point(i), d))
        x = apply_timing(x, edits, rng)
        facts["pauses"] = [[why, round(d, 3)] for _, d, why in pauses]
    elif fam == "verse":
        sp = info["spoken"]
        if spoken != [w.lower() for w in sp["words"]]:
            return drop("token_mismatch", spoken=spoken, want=sp["words"])
        for a, pp in zip(al, sp["parts"]):
            if L.misaki_syllables(a[1], gb) != len(pp):
                return drop("syllables_mismatch", word=a[0], ph=a[1], want=pp)
            if len(pp) > 1 and L.misaki_primary(a[1], gb) != pp.index("1"):
                return drop("stress_mismatch", word=a[0], ph=a[1], want=pp)
        want_r = sp["rkey"][2] if gb else sp["rkey"][1]
        if misaki_rhyme(al[-1][1], gb) != want_r:
            return drop("rhyme_mismatch", word=al[-1][0], ph=al[-1][1])
    y, f2 = finish(x, rng, spec["phone"], fx)
    facts.update(f2)
    facts["dur"] = round(len(y) / SR, 3)
    return y, facts


# Questions

Q = {
    "disfl_combo": [
        "Which combination of disfluency types is present in this speech?",
        "Which kinds of disfluency does the speaker produce in this clip?",
        "Listen to the sentence. Which set of disfluencies can you hear?",
        "Which option lists exactly the disfluency types that occur in the recording?",
        "What types of non-fluent speech appear in this utterance?",
        "Which of these best describes all the disfluencies the speaker makes?",
        "Identify every kind of disfluency in the clip. Which option matches?",
        "Which disfluency categories are found in what the speaker says?",
        "Considering filled pauses, discourse markers, restarts and editing terms, which ones does the speaker use?",
        "Which group of speech disfluencies is heard in this sentence?"],
    "disfl_noul_filled": [
        "Does the speaker use a filled pause such as 'uh', 'um' or 'er'?",
        "Is there a hesitation sound like 'um' or 'uh' in the clip?",
        "Do you hear the speaker fill a pause with 'uh', 'um' or 'er'?",
        "Does the utterance contain a filled pause?",
        "Is any 'um', 'uh' or 'er' spoken in this sentence?",
        "Does the speaker hesitate out loud with a sound such as 'uh'?",
        "Can you hear at least one filled pause in the recording?",
        "Are there vocal hesitations like 'um' or 'er' in this speech?"],
    "disfl_noul_marker": [
        "Does the speaker use a discourse marker such as 'well', 'so', 'you know' or 'like'?",
        "Is there a discourse marker like 'well' or 'you know' in the sentence?",
        "Do you hear filler words such as 'like', 'so' or 'you know' used as discourse markers?",
        "Does the utterance contain a discourse marker?",
        "Does the speaker drop in a word like 'well' or 'you know' that adds no content?",
        "Can you hear a discourse marker such as 'so' or 'like' in the clip?",
        "Is any discourse marker ('well', 'so', 'like', 'you know') used by the speaker?",
        "Are there discourse markers in what the speaker says?"],
    "disfl_noul_restart": [
        "Does the speaker restart, repeating or abandoning the beginning of a phrase?",
        "Is there a false start or a repeated start in the sentence?",
        "Does the speaker break off and begin a phrase again?",
        "Does the utterance contain a restart?",
        "Do you hear the speaker cut a phrase short and start over?",
        "Is any part of the sentence begun, dropped and then started again?",
        "Can you hear a restart, such as repeating the first words or abandoning a start?",
        "Does the speaker make a false start anywhere in the clip?"],
    "disfl_noul_editing": [
        "Does the speaker use an explicit editing term such as 'I mean', 'sorry', 'rather' or 'no wait' to correct a word?",
        "Is there a self-correction signalled by a phrase like 'I mean' or 'sorry'?",
        "Does the speaker say a wrong word and fix it with an editing phrase such as 'no wait'?",
        "Does the utterance contain an editing term followed by a correction?",
        "Do you hear the speaker correct themselves using 'I mean', 'sorry' or 'rather'?",
        "Is any word replaced after an explicit correction cue like 'no wait' or 'I mean'?",
        "Can you hear an editing expression that introduces a repair?",
        "Does the speaker repair a mistake with an explicit editing term?"],
    "disfl_fluent": [
        "Is this sentence spoken completely fluently, with no disfluency of any kind?",
        "Is the speech free of filled pauses, discourse markers, restarts and editing terms?",
        "Does the speaker say the sentence without any disfluency?",
        "Is the utterance perfectly fluent?",
        "Would you describe this speech as entirely free of disfluencies?",
        "Does the speaker get through the sentence with no hesitations, fillers, restarts or corrections?",
        "Is the clip an example of fully fluent speech?",
        "Are there zero disfluencies in this recording?"],
    "disfl_ntypes": [
        "How many different kinds of disfluency does the speaker produce?",
        "Counting filled pauses, discourse markers, restarts and editing terms as four kinds, how many kinds occur?",
        "How many distinct disfluency types can you hear in the sentence?",
        "Of the four disfluency types (filled pause, discourse marker, restart, editing term), how many are present?",
        "What is the number of different disfluency categories in this clip?",
        "How many separate types of disfluency appear in the utterance?",
        "Count the kinds of disfluency in the recording (not the number of times). How many?",
        "How many disfluency types does this speech contain?"],
    "verse_cont": [
        "Which written line best continues the spoken line, keeping its rhythm and rhyme?",
        "Which of these lines would make the best next line for the verse you hear, matching its meter and rhyme?",
        "Listen to the line of verse. Which written line could follow it with the same rhythm and a rhyme?",
        "Pick the line that fits after the spoken line: same syllable count, same stress pattern and a rhyming last word.",
        "The speaker recites one line of a poem. Which option continues it in the same meter and rhymes with it?",
        "Which candidate line matches the spoken line's beat and rhymes with its final word?",
        "To complete a rhyming couplet with the spoken line, which written line should come next?",
        "Which line keeps the rhythm of the recited line and ends with a rhyme for it?",
        "Choose the next line that best preserves the spoken line's meter and rhyme.",
        "Which of the following lines pairs with the line you hear as a rhyming couplet in the same rhythm?"],
    "verse_cont_noul": [
        "Would the line \"{L}\" be a good next line for the spoken verse, keeping its rhythm and rhyme?",
        "Does \"{L}\" continue the spoken line with the same meter and a rhyme?",
        "Could \"{L}\" follow the line you hear as a rhyming couplet in the same rhythm?",
        "Is \"{L}\" a fitting next line, matching both the beat and the rhyme of the recited line?",
        "Does the written line \"{L}\" keep the spoken line's syllable count, stress pattern and rhyme?",
        "Listen to the verse. Would \"{L}\" complete it as a couplet with matching rhythm and rhyme?",
        "Does \"{L}\" match the recited line in meter and rhyme with its last word?",
        "Would a poet accept \"{L}\" as the next line, given the rhythm and rhyme of the spoken line?"],
    "verse_rhyme_noul": [
        "Does the written line \"{L}\" rhyme with the line you hear?",
        "Does the last word of \"{L}\" rhyme with the last word of the spoken line?",
        "Is \"{L}\" a rhyme for the recited line, judging only by the final words?",
        "Listen to the spoken line. Does \"{L}\" end with a word that rhymes with it?",
        "Ignoring rhythm, does \"{L}\" rhyme with the spoken line?",
        "Do the spoken line and the written line \"{L}\" end in rhyming words?",
        "Would \"{L}\" make a rhyme with the line that is read aloud?",
        "Does \"{L}\" share the end rhyme of the line in the recording?"],
    "verse_count_noul": [
        "Does the written line \"{L}\" have the same number of syllables as the spoken line?",
        "Is the syllable count of \"{L}\" equal to that of the line you hear?",
        "Counting syllables, is \"{L}\" exactly as long as the recited line?",
        "Do the spoken line and \"{L}\" contain the same number of syllables?",
        "Ignoring rhyme, does \"{L}\" have as many syllables as the spoken line?",
        "Would \"{L}\" match the recited line syllable for syllable in length?",
        "Does \"{L}\" have neither more nor fewer syllables than the line in the clip?",
        "Is the line \"{L}\" the same length in syllables as the spoken verse line?"],
    "verse_rhymeword": [
        "Which of these words rhymes with the last word of the spoken line?",
        "Which word would rhyme with the end of the line you hear?",
        "Listen to the verse line. Which option rhymes with its final word?",
        "Pick the word that makes a rhyme with the spoken line's last word.",
        "Which of the following words rhymes with how the recited line ends?",
        "The speaker reads one line. Which word rhymes with the word it ends on?",
        "Which word shares the end rhyme of the spoken line?",
        "To rhyme with the line in the clip, which word should the next line end with?"],
    "verse_syll": [
        "How many syllables does the spoken line contain?",
        "Count the syllables in the line of verse you hear. How many are there?",
        "What is the syllable count of the recited line?",
        "How many syllables are spoken in this line of poetry?",
        "Listen to the line and count its syllables. What is the total?",
        "How long is the spoken line, measured in syllables?",
        "In total, how many syllables does the speaker say in this verse line?",
        "What number of syllables makes up the line in the recording?"],
    "verse_meter": [
        "Which rhythm does the spoken line follow?",
        "What is the meter of the line you hear?",
        "Listen to the stressed and unstressed syllables. Which metrical pattern does the line use?",
        "Which description fits the beat of the recited line?",
        "How are the stresses arranged in this line of verse?",
        "Which foot repeats through the spoken line?",
        "What kind of metrical foot does the recited line mostly use?",
        "Which stress pattern best describes the line in the clip?"],
}

NAMES = {
    "plain": {"filled": "filled pauses", "marker": "discourse markers", "restart": "restarts",
              "editing": "editing terms", "none": "no disfluency"},
    "examples": {"filled": "filled pauses (uh / um / er)", "marker": "discourse markers (well / so / you know / like)",
                 "restart": "restarts (repeated or abandoned starts)",
                 "editing": "editing terms (I mean / sorry / rather / no wait)", "none": "none (fluent speech)"},
    "lay": {"filled": "hesitation sounds like 'um'", "marker": "words like 'well' or 'you know'",
            "restart": "starting a phrase over", "editing": "a correction with 'I mean' or 'sorry'",
            "none": "no disfluencies at all"},
    "tech": {"filled": "filled pause", "marker": "discourse marker", "restart": "false start / restart",
             "editing": "explicit editing term", "none": "fluent, no disfluency"},
}
JOINS = {"plain": "and", "examples": "+", "lay": "and", "tech": "+"}


def combo_name(types, style) -> str:
    nm = NAMES[style]
    if not types:
        return nm["none"]
    parts = [nm[t] for t in types]
    if JOINS[style] == "+":
        return " + ".join(parts)
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def combo_option_set(correct, k, rng):
    """k options incl. `correct`. Half the time a Hamming-1 cluster around a random centre (rejection-sampled so
    that, given the option set, every member is equally likely to be the answer), else uniform random."""
    others = [c for c in COMBOS if c != correct]
    if rng.random() < 0.5:
        for _ in range(500):
            c = rng.choice(COMBOS)
            nb = [d for d in COMBOS if len(set(c) ^ set(d)) == 1]
            s = [c] + rng.sample(nb, min(k - 1, len(nb)))
            if correct in s:
                return s
    return [correct] + rng.sample(others, k - 1)


NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
       "thirteen", "fourteen", "fifteen", "sixteen"]
METER_NAMES = [
    {"iambic": "iambic (da-DUM)", "trochaic": "trochaic (DUM-da)", "anapestic": "anapestic (da-da-DUM)",
     "dactylic": "dactylic (DUM-da-da)"},
    {"iambic": "da-DUM da-DUM", "trochaic": "DUM-da DUM-da", "anapestic": "da-da-DUM da-da-DUM",
     "dactylic": "DUM-da-da DUM-da-da"},
    {"iambic": "iambic", "trochaic": "trochaic", "anapestic": "anapestic", "dactylic": "dactylic"},
    {"iambic": "unstressed-stressed pairs", "trochaic": "stressed-unstressed pairs",
     "anapestic": "two unstressed then a stressed syllable", "dactylic": "a stressed then two unstressed syllables"},
]


def _noul(make, rng, qname, qtext, yes, meta):
    return make(qname, qtext, ["yes", "no"], [1.0, 0.0] if yes else [0.0, 1.0], 0 if yes else 1, "noul",
                {**meta, "answer_class": "yes" if yes else "no"}, rng)


def items_for(rec, make_item, seed, ctx=None):
    """Items (<= 4) for one processed clip. rec = spec + facts + bank."""
    spec, facts = rec, rec["facts"]
    fam, info = spec["fam"], spec["info"]
    rng = random.Random(f"{seed}:{spec['key']}:items")
    out = []
    base_meta = {"family": fam, "voice": spec["voice"], "speed": spec["speed"], "text": spec["text"],
                 "phone_fx": facts.get("codec"), "speaker": f"kokoro:{spec['voice']}",
                 "licence": "synthetic (Kokoro-82M Apache-2.0; text written in code; labels by construction / "
                            "CMUdict BSD-2 + misaki Apache-2.0 lexicons)"}

    def make(qname, question, options, target, gold, qtype, meta, r, ordered=False):
        it = make_item(item_id=f"{spec['key']}_{len(out)}", set_name=spec["split"], source=SOURCE[fam],
                       question=question, options=options, target=target, gold=gold,
                       rng=NoShuffle() if ordered else r, audio={"bank": rec["bank"], "key": spec["key"]},
                       channel="clean", state_text=None, meta={**base_meta, **meta, "qname": qname, "qtype": qtype})
        out.append(it)

    def choice(qname, question, options, gold, meta, ordered=False, qtype="choice"):
        t = [0.0] * len(options)
        t[gold] = 1.0
        make(qname, question, options, t, gold, qtype, meta, rng, ordered)

    q = lambda k: rng.choice(Q[k])  # noqa: E731
    if fam == "disfl":
        types = tuple(info["types"])
        meta = {"types": list(types), "combo": combo_key(types), "fillers": info["fillers"], "markers": info["markers"],
                "edit_term": info["edit_term"], "restart": info["restart"], "pauses": facts.get("pauses"),
                "restart_cut": facts.get("restart_cut")}
        k = rng.choices([2, 3, 4, 5], [0.15, 0.3, 0.35, 0.2])[0]
        style = rng.choice(list(NAMES))
        opts = combo_option_set(types, k, rng)
        names = [combo_name(c, style) for c in opts]
        assert len(set(names)) == len(names) and opts.count(types) == 1
        choice("disfl_combo", q("disfl_combo"), names, opts.index(types),
               {**meta, "answer_class": combo_key(types), "option_combos": [combo_key(c) for c in opts],
                "name_style": style})
        asked = rng.sample(TYPES, 1 if rng.random() < 0.6 else 2)
        for t in asked:
            _noul(make, rng, f"disfl_noul_{t}", q(f"disfl_noul_{t}"), t in types, meta)
        nt = len(types)
        if rng.random() < 0.8 / [1, 4, 6, 4, 1][nt]:
            lv = rng.choice([["none", "one", "two", "three", "four"], ["0", "1", "2", "3", "4"],
                             ["no type", "one type", "two types", "three types", "all four types"]])
            choice("disfl_ntypes", q("disfl_ntypes"), lv, nt, {**meta, "answer_class": str(nt)}, ordered=True,
                   qtype="score")
        if rng.random() < (0.8 if nt == 0 else 0.8 / 15):
            _noul(make, rng, "disfl_fluent", q("disfl_fluent"), nt == 0, meta)
    elif fam == "verse":
        sp, cands = info["spoken"], info["cands"]
        meta = {"meter": info["meter"], "meter_family": info["family"], "pattern": sp["pattern"],
                "n_syll": sp["n_syll"], "end_word": sp["end"], "alt_meter": info["alt_meter"],
                "alt_how": info["alt_how"], "slant_distractor": info["slant"]}
        cells = ["MR", "MR'", "M'R", "M'R'"]
        if rng.random() < 0.6:
            use = cells
        else:
            use = ["MR"] + rng.sample(cells[1:], 2)
        dtype = {"MR'": "rhyme_broken", "M'R": f"meter_broken_{info['alt_how']}", "M'R'": "both_broken"}
        opts = [cands[c]["text"] for c in use]
        choice("verse_cont", q("verse_cont"), opts, 0,
               {**meta, "answer_class": info["family"], "cells": use, "distractors": [dtype[c] for c in use[1:]],
                "candidates": {c: {"text": cands[c]["text"], "pattern": cands[c]["pattern"], "end": cands[c]["end"],
                                   "meter": cands[c]["meter"]} for c in use}})
        views = []
        if rng.random() < 0.5:
            views.append("cont_noul")
        if rng.random() < 0.3:
            views.append("rhyme_noul")
        if rng.random() < 0.25:
            views.append("count_noul")
        if rng.random() < 0.3:
            views.append("rhymeword")
        if rng.random() < 0.25:
            views.append("syll")
        if rng.random() < 0.2:
            views.append("meter")
        rng.shuffle(views)
        for v in views[:3]:
            if v == "cont_noul":
                yes = rng.random() < 0.5
                c = "MR" if yes else rng.choice(cells[1:])
                _noul(make, rng, "verse_cont_noul", q("verse_cont_noul").format(L=cands[c]["text"]), yes,
                      {**meta, "cell": c})
            elif v == "rhyme_noul":
                c = rng.choice(cells)
                _noul(make, rng, "verse_rhyme_noul", q("verse_rhyme_noul").format(L=cands[c]["text"]), "R'" not in c,
                      {**meta, "cell": c})
            elif v == "count_noul":
                c = rng.choice(cells)
                yes = cands[c]["n_syll"] == sp["n_syll"]
                _noul(make, rng, "verse_count_noul", q("verse_count_noul").format(L=cands[c]["text"]), yes,
                      {**meta, "cell": c, "cand_n_syll": cands[c]["n_syll"]})
            elif v == "rhymeword" and ctx is not None:
                cls = ctx["classes"][_rk(sp["rkey"])]
                pre = ctx["pre_of"]
                good = [w for w in cls if w != sp["end"] and pre[w] != pre[sp["end"]]]
                k0 = _rk(sp["rkey"])
                bad_keys = [k for k in ctx["classes"] if strict_nonrhyme(k, k0)]
                slant = [k for k in bad_keys if k[0][0] == k0[0][0] or k[0][1:] == k0[0][1:]]
                if good and len(bad_keys) >= 3:
                    nb = rng.choice([2, 3, 3])
                    ks = rng.sample(slant, min(len(slant), rng.choice([1, 2]))) if slant else []
                    ks += rng.sample([k for k in bad_keys if k not in ks], nb - len(ks))
                    bad = [rng.choice(ctx["classes"][k]) for k in ks]
                    opts = [rng.choice(good)] + bad
                    if len(set(opts)) == len(opts):
                        choice("verse_rhymeword", q("verse_rhymeword"), opts, 0, {**meta, "answer_class": None})
            elif v == "syll":
                n = sp["n_syll"]
                lo = max(4, n - rng.randint(0, 3))
                vals = list(range(lo, lo + 4))
                fmt = rng.choice([str, lambda i: NUM[i], lambda i: f"{i} syllables"])
                choice("verse_syll", q("verse_syll"), [fmt(i) for i in vals], vals.index(n),
                       {**meta, "answer_class": str(n)})
            elif v == "meter":
                names = rng.choice(METER_NAMES)
                k = rng.choice([2, 3, 4, 4])
                fams = [info["family"]] + rng.sample([f for f in B.FAMILIES if f != info["family"]], k - 1)
                choice("verse_meter", q("verse_meter"), [names[f] for f in fams], 0,
                       {**meta, "answer_class": info["family"]})
    return out[:4]


def _rk(k):
    return tuple(tuple(x) if isinstance(x, list) else x for x in k)


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


def norm_q(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def validate(items, bank_index, mmsu_questions):
    errs = collections.Counter()
    keys = {"id", "set", "source", "question", "options", "target", "gold", "audio", "channel", "state_text", "meta"}
    mm = set(mmsu_questions)
    mmn = {norm_q(x) for x in mmsu_questions}
    ids = set()
    for it in items:
        if set(it) - {"majority"} != keys:
            errs["keys"] += 1
        if not 2 <= len(it["options"]) <= 20 or len(set(it["options"])) != len(it["options"]):
            errs["options"] += 1
        if abs(sum(it["target"]) - 1) > 1e-5 or int(np.argmax(it["target"])) != it["gold"] or max(it["target"]) != 1.0:
            errs["target"] += 1
        if it["meta"]["qtype"] not in ("choice", "noul", "score"):
            errs["qtype"] += 1
        if it["meta"]["qtype"] == "noul" and sorted(it["options"]) != ["no", "yes"]:
            errs["noul_options"] += 1
        a = it["audio"]
        if a["key"] not in bank_index.get(a["bank"], {}):
            errs["audio_key"] += 1
        if it["question"] in mm or norm_q(it["question"]) in mmn:
            errs["mmsu_question_equal"] += 1
        if it["id"] in ids:
            errs["dup_id"] += 1
        ids.add(it["id"])
        if it["channel"] != "clean":
            errs["channel"] += 1
        if not it["source"].startswith("sb_disfluency"):
            errs["source_prefix"] += 1
        m = it["meta"]
        qn = m["qname"]
        gold = it["options"][it["gold"]]
        if qn.startswith("disfl_noul_"):
            t = qn[len("disfl_noul_"):]
            if (gold == "yes") != (t in m["types"]):
                errs["disfl_noul_label"] += 1
        if qn == "disfl_combo":
            want = combo_name(tuple(m["types"]), m["name_style"])
            if gold != want:
                errs["disfl_combo_label"] += 1
        if qn == "verse_cont":
            cs = m["candidates"]
            ok = [c for c, v in cs.items() if v["pattern"] == m["pattern"] and "R'" not in c]
            if ok != ["MR"] or gold != cs["MR"]["text"]:
                errs["verse_cont_label"] += 1
    return dict(errs)
