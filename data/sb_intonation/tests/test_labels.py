"""sb_intonation label logic tests. The MMSU test reads only the "question" field of /vol/bench_mmsu/truth.jsonl and
is skipped where /vol is not mounted (the Modal build runs it before writing anything)."""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import labels as L  # noqa: E402
import sentences as S  # noqa: E402

MMSU = Path("/vol/bench_mmsu/truth.jsonl")


def _track(pattern, size, n=400, c=0.5, p=1.2, noise=0.0, seed=0):
    u = np.linspace(0, 1, n)
    st = size * L.contour_shape(pattern, u, c, p) + np.random.default_rng(seed).normal(0, noise, n)
    f0 = 150.0 * 2 ** (st / 12)
    f0[::7] = 0.0  # some unvoiced frames
    return f0


@pytest.mark.parametrize("pattern", L.CLASSES)
def test_shapes_classify_as_planned(pattern):
    for size in (6.0, 8.0, 10.0):
        for c in (0.42, 0.5, 0.58):
            for p in ((1.0, 1.5) if pattern == "falling" else (1.2, 2.0)):
                cls, st = L.classify_track(_track(pattern, size, c=c, p=p, noise=0.3))
                assert cls == pattern, (pattern, size, c, p, st)


def test_flat_and_small_contours_rejected():
    assert L.classify_track(_track("rise-fall", 1.0))[0] is None
    assert L.classify_track(150.0 * np.ones(300))[0] is None
    assert L.classify_track(np.zeros(300))[0] is None


def test_classes_mutually_exclusive():
    rng = np.random.default_rng(0)
    for _ in range(20000):
        m = rng.uniform(-8, 8, 4)
        hits = [c for c in L.CLASSES if L.classify(*m) == c]
        assert len(hits) <= 1


def test_final_direction():
    u = np.linspace(0, 1, 300)
    for d in (1, -1):
        st = L.final_glide(u, 0.75, 6.0, d)
        _, _, _, fin = L.thirds(st, u)
        assert fin * d >= L.FINAL_MIN
    assert L.ends_rising("fall-rise") and L.ends_rising("rising") and not L.ends_rising("rise-fall")


def test_all_sentences_parse_and_are_own():
    ents = [L.parse_entry(e) for e in S.SENTENCES]
    assert len(ents) >= 200, len(ents)
    texts = [" ".join(L.norm_word(w) for w in e["words"]) for e in ents]
    assert len(set(texts)) == len(texts), "duplicate sentence"
    for e in ents:
        for p in e["placements"]:
            assert len(set(p["meanings"])) == len(p["meanings"])
        all_m = [m for p in e["placements"] for m in p["meanings"]]
        assert len(set(all_m)) == len(all_m), e["words"]
        if e["foil"]:
            assert e["foil"] not in all_m
    assert sum(len(e["placements"]) for e in ents) >= 2 * len(ents)


def test_parse_placement_and_tts_text():
    w, p = L.parse_placement("I invited Nora | Lee | and Sam to dinner")
    assert w[2] == "Nora" and p == [2, 3]
    assert L.tts_text(w, p, False) == "I invited Nora, Lee, and Sam to dinner."
    assert L.tts_text(w, None, True).endswith("dinner?")


def test_match_pauses():
    ins = {2: (1.0, 1.5, True), 5: (2.0, 2.1, False)}
    assert L.match_pauses([(0.98, 1.52)], ins) == (True, {2: 0.54})
    assert not L.match_pauses([(0.98, 1.52), (1.95, 2.3)], ins)[0]   # minor pause measured as a pause
    assert not L.match_pauses([], ins)[0]


def test_silences_on_synthetic_audio():
    sr = L.SR
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 0.1, sr), np.zeros(int(0.5 * sr)), rng.normal(0, 0.1, sr)])
    sil = L.silences(x, 0.25)
    assert len(sil) == 1 and abs((sil[0][1] - sil[0][0]) - 0.5) < 0.05


def test_boundary_labels_unique_words():
    words = "the cat saw the dog".split()
    assert L.boundary_label(words, 0, 0) == 'between "the" and "cat"'
    assert L.boundary_label(words, 1, 0) == 'after "cat"'
    assert L.placement_label(words, [1, 2], 2) == 'after the word "cat" and after "saw"'


def test_balance_noul_exact_and_choice_cap():
    items = []
    for i in range(30):
        ans = "yes" if i < 20 else "no"
        items.append({"id": f"a{i}", "source": "s", "set": "train", "audio": {"bank": "b", "key": f"k{i}"},
                      "meta": {"qtype": "noul", "qkind": "q", "answer_class": ans}})
    for i in range(40):
        c = "falling" if i < 25 else ("rising" if i < 35 else "fall-rise")
        items.append({"id": f"c{i}", "source": "s", "set": "train", "audio": {"bank": "b", "key": f"m{i}"},
                      "meta": {"qtype": "choice", "qkind": "inton_choice", "answer_class": c}})
    out, _ = L.balance(items, 0)
    yn = [it["meta"]["answer_class"] for it in out if it["meta"]["qtype"] == "noul"]
    assert yn.count("yes") == yn.count("no") == 10
    ch = [it["meta"]["answer_class"] for it in out if it["meta"]["qtype"] == "choice"]
    assert max(ch.count(c) for c in set(ch)) <= 1.15 * min(ch.count(c) for c in set(ch))


def test_question_paraphrases_per_family():
    for k, v in L.Q.items():
        assert len(v) >= 8, k
        assert len(set(v)) == len(v), k


def test_mmsu_equal_count_logic():
    assert L.mmsu_equal_count(["Is it rising?"], ["is it rising"]) == 1
    assert L.mmsu_equal_count(["Is it rising now?"], ["is it rising"]) == 0


@pytest.mark.skipif(not MMSU.exists(), reason="MMSU question strings only available inside the Modal build")
def test_no_question_equals_mmsu():
    qs = []
    for line in MMSU.read_text().splitlines():
        q = json.loads(line).get("question")
        if isinstance(q, str):
            qs.append(q)
    assert qs
    assert L.mmsu_equal_count(L.all_templates(), qs) == 0
    # filled templates (all wording variants)
    filled = []
    for t in L.Q["inton_noul"]:
        filled += [t.format(d=d) for ds in L.CLASS_NOUL_DESC.values() for d in ds]
    for t in L.Q["final_noul"]:
        filled += [t.format(v=v) for v in ("rise", "fall")]
    ents = [L.parse_entry(e) for e in S.SENTENCES]
    for t in L.Q["pm_meaning_noul"]:
        filled += [t.format(m=m) for e in ents for p in e["placements"] for m in p["meanings"]]
    for t in L.Q["pm_word_noul"]:
        filled += [t.format(w=w) for e in ents for w in e["words"]]
    filled += [" ".join(e["words"]) for e in ents]
    assert L.mmsu_equal_count(filled, qs) == 0
