"""Unit tests for the sb_phone_pitch label logic (no /vol access)."""
import json
import math
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "inference"))

import numpy as np  # noqa: E402

import build as B  # noqa: E402
import phone_labels as L  # noqa: E402
import rel_labels as R  # noqa: E402


def test_paraphrase_counts_at_least_8():
    for k, v in L.Q.items():
        assert len(set(v)) >= 8, k
    for f in R.FEATURES:
        assert len(set(R.SCORE_Q[f])) >= 8
        for d in ("up", "down", "same"):
            assert len(set(R.NOUL_Q[f][d])) >= 8
    for d in (R.ABS_SCORE_Q, R.CMP_SCORE_Q, R.ABS_NOUL_Q, R.CMP_CHOICE_Q, R.CMP_NOUL_Q):
        for k, v in d.items():
            assert len(set(v)) >= 8, k


def test_abs_levels_have_gaps():
    assert R.abs_level("pitch", 100.0) == 0
    assert R.abs_level("pitch", 140.0) is None  # gap 125-150
    assert R.abs_level("pitch", 170.0) == 1
    assert R.abs_level("pitch", 200.0) is None  # gap 190-220
    assert R.abs_level("pitch", 260.0) == 2
    assert R.abs_level("rate", 3.0) == 0 and R.abs_level("rate", 4.5) == 1 and R.abs_level("rate", 6.5) == 2
    assert R.abs_level("rate", 3.8) is None
    assert R.abs_value("pitch", {"f0_med": 200.0, "n_voiced": 10}) is None  # too few voiced frames
    assert R.abs_value("rate", {"rate": 5.0, "span_s": 0.6}) is None


def test_cmp_classes_and_margins():
    a = {"f0_med": 100.0, "n_voiced": 50}
    assert R.cmp_class("pitch", R.cmp_diff("pitch", a, {"f0_med": 113.0, "n_voiced": 50})) == 2  # +2.1 st
    assert R.cmp_class("pitch", R.cmp_diff("pitch", a, {"f0_med": 110.0, "n_voiced": 50})) is None  # +1.65 st ambiguous
    assert R.cmp_class("pitch", R.cmp_diff("pitch", a, {"f0_med": 102.0, "n_voiced": 50})) == 1
    assert R.cmp_class("pitch", R.cmp_diff("pitch", a, {"f0_med": 88.0, "n_voiced": 50})) == 0
    assert R.cmp_diff("pitch", a, {"f0_med": 230.0, "n_voiced": 50}) is None  # > 13 st (octave error guard)
    r1, r2 = {"rate": 4.0, "span_s": 2.0}, {"rate": 5.0, "span_s": 2.0}
    assert R.cmp_class("rate", R.cmp_diff("rate", r1, r2)) == 2
    assert R.cmp_class("rate", R.cmp_diff("rate", r2, r1)) == 0


def test_relative_thresholds_unchanged():
    assert R.classify("loudness", 3.2) == 2 and R.classify("loudness", -3.2) == 0
    assert R.classify("loudness", 0.5) == 1 and R.classify("loudness", 2.0) is None
    assert R.classify("pitch", R.rel_diff("pitch", {"f0_med": 100, "n_voiced": 40}, {"f0_med": 110, "n_voiced": 40})) == 2


def test_two_part_layout():
    x = np.ones(16000, np.float32) * 0.1
    y, info = R.two_part(x, x)
    assert abs(info["tgt_start_s"] - (1.0 + 2 * R.GAP_S + R.TONE_S)) < 1e-3
    assert len(y) == 16000 * 2 + int(round(2 * R.GAP_S * 16000)) + int(round(R.TONE_S * 16000))


def test_musan_licence_rejects_sharealike_nc_nd():
    assert B.licence_ok("Attribution 4.0 International (CC BY 4.0)")
    assert B.licence_ok("CC0 1.0")
    assert B.licence_ok("Public Domain")
    assert not B.licence_ok("Attribution-ShareAlike 3.0 (CC BY-SA 3.0)")
    assert not B.licence_ok("CC BY-NC 3.0")
    assert not B.licence_ok("Attribution-NoDerivs 3.0 (CC BY-ND)")
    assert not B.licence_ok("Sampling Plus 1.0")
    assert not B.licence_ok("")


def test_phone_number_and_quality_rules_unchanged():
    assert L.number_label("my account number is 4 5 6") is True
    assert L.number_label("I would like to check my balance") is False
    assert L.number_label("this one please") is None
    r = L.sample_recipe("noise_sev", random.Random(0))
    assert L.quality_level(r) == "very poor"
    assert L.quality_level(L.sample_recipe("clean", random.Random(0))) == "clear"


def test_balance_noul_exact_and_cap_choice():
    rng = random.Random(0)
    items = []
    for i in range(30):
        yes = i < 20
        items.append({"id": str(i), "set": "train", "source": "sb_phone_pitch_x", "options": ["yes", "no"], "gold": 0 if yes else 1,
                      "meta": {"qtype": "noul", "qname": "q"}})
    for i in range(30):
        items.append({"id": f"c{i}", "set": "train", "source": "sb_phone_pitch_y", "options": ["a", "b"], "gold": 0,
                      "meta": {"qtype": "choice", "answer_class": "a" if i < 25 else "b"}})
    kept, _ = B.balance_noul(items, rng)
    noul = [i for i in kept if i["meta"]["qtype"] == "noul"]
    assert sum(1 for i in noul if i["gold"] == 0) == sum(1 for i in noul if i["gold"] == 1) == 10
    capped, _ = B.cap_choice(kept, rng)
    ch = [i for i in capped if i["meta"]["qtype"] == "choice"]
    assert sum(1 for i in ch if i["meta"]["answer_class"] == "a") == 10  # 2x the rarest (5)


def test_mmsu_non_equality_logic():
    mm = {B.norm_q("Which speaker has a higher pitch?")}
    assert B.mmsu_equal_count(["which speaker has a higher pitch", "Who talks faster?"], mm) == 1
    # our own pools never contain that (hypothetical) string
    assert B.mmsu_equal_count(R.all_question_texts(), mm) == 0
    # optional real check: MMSU_QUESTIONS_JSON = path to a JSON list of MMSU question strings (question field only)
    p = os.environ.get("MMSU_QUESTIONS_JSON")
    if p:
        real = {B.norm_q(q) for q in json.loads(Path(p).read_text())}
        pools = set(R.all_question_texts()) | {q for v in L.Q.values() for q in v}
        assert B.mmsu_equal_count(pools, real) == 0


def test_adi_extra_relative_questions_not_in_pools():
    assert not (R.eval_questions() & R.all_question_texts())
