import math
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "inference"))

import build  # noqa: E402
import cmp_labels as L  # noqa: E402


def test_rank_levels_margin():
    assert L.rank_levels([0.0, 3.2, 1.6], "pitch") == [0, 2, 1]
    assert L.rank_levels([0.0, 3.0, 1.0], "pitch") is None  # 1.0 st apart < 1.5 margin
    assert L.rank_levels([-20.0, -26.0, -32.5], "volume") == [2, 1, 0]
    assert L.rank_levels([math.log(3), math.log(4), math.log(5)], "speed") == [0, 1, 2]
    assert L.rank_levels([math.log(3), math.log(3.3), math.log(5)], "speed") is None
    assert L.rank_levels([0.0, None, 3.0], "pitch") is None


def test_pair_class():
    assert L.pair_class(0.0, 0.3, "pitch") == 1
    assert L.pair_class(0.0, 2.0, "pitch") == 2
    assert L.pair_class(0.0, -2.0, "pitch") == 0
    assert L.pair_class(0.0, 1.0, "pitch") is None
    assert L.pair_class(-20, -24, "volume") == 0


def test_pattern_options_contains_gold():
    rng = random.Random(0)
    for _ in range(50):
        ranks = rng.sample([0, 1, 2], 3)
        perms, gi = L.pattern_options(ranks, 4, rng)
        assert len(perms) == 4 and len(set(perms)) == 4 and perms[gi] == tuple(ranks)
    assert L.pattern_string([2, 0, 1], ["low", "medium", "high"], 0) == "high-low-medium"
    assert L.pattern_string([2, 0, 1], ["slow", "medium", "fast"], 2) == "fast, then slow, then medium"


def test_numeric_options_separated():
    rng = random.Random(1)
    for v in (1.3, 4.0, 7.25, 12.9, 22.0):
        res = L.numeric_options(v, rng)
        assert res is not None
        opts, g = res
        assert opts[g] == round(v, 1) and opts == sorted(opts)
        for a in opts:
            for b in opts:
                if a != b:
                    assert abs(a - b) >= max(1.2, 0.2 * max(a, b)) - 1e-9


def test_bin_index_margin():
    assert L.bin_index(4.0, [5, 10, 15], 0.6) == 0
    assert L.bin_index(4.7, [5, 10, 15], 0.6) is None
    assert L.bin_index(16.0, [5, 10, 15], 0.6) == 3


def test_third_of():
    assert L.third_of(1, 12) == 0 and L.third_of(5, 12) == 1 and L.third_of(9, 12) == 2


def _tone(sec, f=200.0, amp=0.3):
    t = np.arange(int(sec * L.SR)) / L.SR
    return amp * np.sin(2 * np.pi * f * t)


def test_silences_and_duration_measure():
    x = np.concatenate([np.zeros(4000), _tone(1.0), np.zeros(int(0.8 * L.SR)), _tone(2.0), np.zeros(4000)])
    sil = L.silences(x, 0.3)
    assert len(sil) == 1 and abs((sil[0][1] - sil[0][0]) - 0.8) < 0.05
    b1 = (4000, 4000 + 16000)
    b2 = (b1[1] + int(0.8 * L.SR), b1[1] + int(0.8 * L.SR) + 32000)
    m = build.measure_duration(x.astype(np.float32), [b1, b2])
    assert abs(m["total"] - 3.0) < 0.1 and abs(m["gaps"][0] - 0.8) < 0.06


def test_insert_pauses_positions():
    words = [("a", 0.0, 0.5), ("b", 0.5, 1.0), ("c", 1.0, 1.5), ("d", 1.5, 2.0)]
    x = _tone(2.0)
    y, ends, ins = build.insert_pauses(x, words, {1: 0.7})
    assert abs(len(y) / L.SR - 2.7) < 1e-3
    assert abs(ins[1][0] - 1.0) < 1e-3 and abs(ins[1][1] - 1.7) < 1e-3
    meas = build.measure_pauses(y.astype(np.float32), ins)
    assert len(meas["silences"]) == 1 and meas["silences"][0]["after_word"] == 1


def test_pause_items_labels():
    words = [(w, i * 0.4, (i + 1) * 0.4) for i, w in enumerate("the cat sat on the warm mat today".split())]
    t = {"mode": "word"}
    meas = {"silences": [{"start": 1.6, "end": 2.4, "dur": 0.8, "after_word": 3}]}
    specs = build.pause_items(t, {3: 0.8}, meas, words, random.Random(0))
    pw = [s for s in specs if s["qkind"] == "pause_word"][0]
    assert pw["options"][pw["gold"]] == "on"
    yn = [s for s in specs if s["qkind"] == "pause_yn"][0]
    assert isinstance(yn["yes"], bool)
    # unplanned extra silence -> no items
    meas2 = {"silences": meas["silences"] + [{"start": 2.6, "end": 3.0, "dur": 0.4, "after_word": None}]}
    assert build.pause_items(t, {3: 0.8}, meas2, words, random.Random(0)) == []


def test_meaning_templates_valid():
    rng = random.Random(3)
    for t in L.PAUSE_MEANING:
        f = L.fill_meaning_template(t, rng)
        n = len(f["words"])
        assert len(f["readings"]) == 2 and f["readings"][0][1] != f["readings"][1][1]
        for pidx, _ in f["readings"]:
            assert all(0 <= i < n - 1 for i in pidx)
        assert len(set([r[1] for r in f["readings"]] + f["distractors"])) == 4


def test_compare_items_balanced_yes_no():
    rng = random.Random(5)
    yes = no = 0
    for i in range(400):
        ranks = rng.sample([0, 1, 2], 3)
        vals = [100 * 2 ** ((r * 3.0) / 12) for r in ranks]
        c = {"family": "pitch", "k": 3, "kind": "same_rep"}
        for s in build.compare_items(c, vals, random.Random(i)):
            if s["qtype"] == "noul":
                yes += s["yes"]
                no += not s["yes"]
            if s["qkind"] == "pattern":
                assert s["answer"] == "-".join(map(str, ranks))
    assert abs(yes - no) < 0.2 * (yes + no)


def test_no_template_has_mmsu_wording():
    banned = ["which pitch pattern best matches the audio", "what is the total speaking time in the audio",
              "which word is most likely followed by a pause"]
    for q in L.all_question_templates():
        assert not any(b in q.lower() for b in banned)


def test_numeric_gold_position_varies():
    rng = random.Random(2)
    pos = [L.numeric_options(12.0, rng)[1] for _ in range(200)]
    assert set(pos) == {0, 1, 2, 3}


def test_generic_pattern():
    assert L.is_generic_pattern("high-low-medium") and L.is_generic_pattern("fast, then slow, then medium")
    assert not L.is_generic_pattern("Call three people: Anna, Smith and Rose.")
