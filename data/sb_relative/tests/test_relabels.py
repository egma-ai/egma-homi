import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import relabels as R  # noqa: E402


def test_rel_diff_units():
    ref = {"loud_db": -20.0, "rate": 4.0, "f0_med": 100.0, "f0_spread_st": 2.0, "n_voiced": 50}
    tgt = {"loud_db": -16.0, "rate": 5.0, "f0_med": 200.0, "f0_spread_st": 1.0, "n_voiced": 50}
    assert R.rel_diff("loudness", ref, tgt) == 4.0
    assert abs(R.rel_diff("rate", ref, tgt) - math.log(1.25)) < 1e-12
    assert R.rel_diff("pitch", ref, tgt) is None  # +12 st > PITCH_MAX_ABS_ST: treated as an octave error
    tgt["f0_med"] = 100.0 * 2 ** (3 / 12)
    assert abs(R.rel_diff("pitch", ref, tgt) - 3.0) < 1e-9
    assert abs(R.rel_diff("pitch_variation", ref, tgt) - math.log(0.5)) < 1e-12
    tgt["n_voiced"] = 5
    assert R.rel_diff("pitch", ref, tgt) is None
    assert R.rel_diff("loudness", ref, {"loud_db": None}) is None


def test_classify_bands_and_gap():
    assert R.classify("loudness", 3.0) == 2
    assert R.classify("loudness", -3.2) == 0
    assert R.classify("loudness", 0.9) == 1
    assert R.classify("loudness", 2.0) is None  # ambiguous in-between pair is dropped
    assert R.classify("rate", math.log(1.13)) == 2
    assert R.classify("rate", math.log(1.08)) is None
    assert R.classify("rate", math.log(1.03)) == 1
    assert R.classify("pitch", -1.6) == 0
    assert R.classify("pitch", None) is None


def test_noul_answer():
    assert R.noul_answer("up", 2) and not R.noul_answer("up", 1) and not R.noul_answer("up", 0)
    assert R.noul_answer("same", 1) and not R.noul_answer("same", 2)
    assert R.noul_answer("down", 0)


def test_plan_amount_lands_in_class():
    rng = random.Random(0)
    for f in ("loudness", "rate", "pitch"):
        for cls in (0, 1, 2):
            for d0 in (-1.0, 0.0, 0.05, 1.2):
                if f == "rate":
                    d0 = d0 / 10
                a = R.plan_amount(f, d0, cls, rng)
                if a is None:
                    continue
                assert R.classify(f, d0 + a) == cls
                assert abs(a) <= R.AUG[f][1]
    assert R.plan_amount("loudness", 30.0, 0, rng) is None  # would need more than 12 dB


def test_two_part_layout_and_level_preserved():
    ref = 0.2 * np.ones(16000, np.float32)
    tgt = 1.5 * np.ones(8000, np.float32)  # clips: whole recording must be scaled together
    y, info = R.two_part(ref, tgt)
    assert len(y) == 16000 + 2 * 9600 + 2400 + 8000
    assert np.max(np.abs(y)) <= 0.99 + 1e-6
    mid_ref = y[8000]
    mid_tgt = y[16000 + 2 * 9600 + 2400 + 4000]
    assert abs(20 * np.log10(mid_tgt / mid_ref) - 20 * np.log10(1.5 / 0.2)) < 1e-3
    assert info["tgt_start_s"] == round(1.0 + 1.2 + 0.15, 3)
    tone = y[16000 + 9600: 16000 + 9600 + 2400]
    assert np.max(np.abs(tone)) > 0.05


def test_question_pools_do_not_contain_eval_questions():
    train = R.all_training_question_texts()
    assert not (train & R.eval_questions())
    for f in R.FEATURES:
        assert len(R.SCORE_Q[f]) >= 5
        for d in ("up", "down", "same"):
            assert len(R.NOUL_Q[f][d]) >= 5
        for opts in R.OPTION_SETS[f]:
            assert len(opts) == 3 and len(set(opts)) == 3


def test_crema_syllables_and_activity():
    assert R.crema_syllables("1001_DFA_ANG_XX") == 6
    x = np.zeros(32000, np.float32)
    x[8000:24000] = 0.3 * np.sin(2 * np.pi * 200 * np.arange(16000) / 16000)
    a = R.activity(x)
    assert abs(a["span_s"] - 1.0) < 0.05


def test_adi_select_strict_balance():
    cands = [{"key": f"k{i}", "split": "dev" if i % 3 == 0 else "test", "pos": i % 4 == 0} for i in range(200)]
    sel = R.adi_select(cands, "t", cap=60, label_of=lambda c: c["pos"], strict=True)
    for s in ("dev", "test"):
        part = [c for c in sel if c["split"] == s]
        assert sum(c["pos"] for c in part) * 2 == len(part)
