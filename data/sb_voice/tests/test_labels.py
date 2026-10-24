import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent / "inference"))

import labels as L  # noqa: E402


def test_heuristic_syllables():
    cases = {"cat": 1, "table": 2, "wanted": 2, "jumped": 1, "beautiful": 3, "yellow": 2, "stella": 2,
             "please": 1, "computer": 3, "the": 1, "make": 1, "horses": 2}
    for w, n in cases.items():
        assert L.heuristic_syllables(w) == n, (w, L.heuristic_syllables(w), n)


def test_count_syllables_uses_cmudict_then_heuristic():
    cmu = {"please": [["P", "L", "IY1", "Z"]], "stella": [["S", "T", "EH1", "L", "AH0"]]}
    assert L.count_syllables("Please call Stella.", cmu) == 1 + 1 + 2
    assert L.count_syllables("I have 2 dogs") == 1 + 1 + 1 + 1
    assert L.count_syllables("") == 0


def _tone(sec, f=200.0, amp=0.3):
    t = np.arange(int(sec * L.SR)) / L.SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_activity_trims_and_counts_long_pauses():
    sil = np.zeros(int(0.5 * L.SR), np.float32)
    x = np.concatenate([sil, _tone(1.0), np.zeros(int(0.6 * L.SR), np.float32), _tone(1.0), sil])
    a = L.activity(x)
    assert a["n_pauses"] == 1
    assert 2.5 <= a["span_s"] <= 2.7  # 1.0 + 0.6 + 1.0, leading / trailing silence trimmed
    assert 0.18 < a["pause_frac"] < 0.26
    short_gap = np.concatenate([_tone(1.0), np.zeros(int(0.1 * L.SR), np.float32), _tone(1.0)])
    assert L.activity(short_gap)["n_pauses"] == 0  # 100 ms is not a long pause
    assert L.activity(np.zeros(L.SR, np.float32)) is None


def test_f0_stats_and_energy_gate():
    f0 = np.array([100.0] * 20 + [200.0] * 20)
    s = L.f0_stats(f0, np.ones(40, bool))
    assert s["n_voiced"] == 40 and 100 <= s["f0_med"] <= 200
    assert np.isnan(L.f0_stats(f0[:5], np.ones(5, bool))["f0_med"])
    rms = np.array([0.5] * 10 + [0.001] * 10)
    v = L.energy_gated_voicing(np.ones(20, bool), rms)
    assert v[:10].all() and not v[10:].any()


def test_percentile_ranks_and_tertiles():
    pr = L.percentile_ranks([3.0, 1.0, 2.0, np.nan, 2.0])
    assert pr[1] == 0.0 and pr[0] == 1.0 and pr[2] == pr[4] == 0.5 and np.isnan(pr[3])
    assert L.tertile_level(0.1) == 0 and L.tertile_level(0.5) == 1 and L.tertile_level(0.9) == 2
    assert L.tertile_target(0.1) == [1.0, 0.0, 0.0]
    t = L.tertile_target(0.35)  # just above the lower boundary -> middle, with 0.3 on 'low'
    assert t == [0.3, 0.7, 0.0] and abs(sum(t) - 1) < 1e-9
    t = L.tertile_target(0.65)
    assert t == [0.0, 0.7, 0.3]


def test_speaker_relative_levels_are_within_speaker():
    spk = ["a"] * 12 + ["b"] * 12 + ["c"] * 3
    vals = list(range(12)) + list(range(100, 112)) + [1, 2, 3]
    pr = L.speaker_relative_levels(spk, vals)
    assert pr[0] == 0.0 and pr[11] == 1.0 and pr[12] == 0.0 and pr[23] == 1.0  # speaker b's lowest is 'low'
    assert np.isnan(pr[24:]).all()  # too few clips


def test_pause_level():
    assert L.pause_level(0.0, 0, 0.1) == 0
    assert L.pause_level(0.2, 2, 0.1) == 2
    assert L.pause_level(0.05, 1, 0.0) == 1  # below the 8 % floor
    assert L.pause_level(0.09, 1, 0.12) == 1


def test_label_maps():
    assert L.accent_group("NorthernIrish") == "Irish" and L.accent_group("NewZealand") == "Australian or New Zealand"
    assert L.accent_group("Welsh") is None and L.accent_group("Unknown") is None
    assert [L.so762_level(s) for s in (3, 5, 6, 7, 8, 10)] == [0, 0, 1, 1, 2, 2]
    assert [L.crema_age_bracket(a) for a in (20, 29, 30, 49, 50, 74)] == [0, 0, 1, 1, 2, 2]
    assert L.child_label(8) is True and L.child_label(15) is None and L.child_label(25) is False
    assert 0 <= L.stable_frac("x") < 1 and L.stable_frac("x") == L.stable_frac("x")


@pytest.mark.parametrize("kind", L.NOISE_TYPES)
def test_noise_and_snr(kind):
    rng = np.random.default_rng(0)
    n = L.colored_noise(L.SR * 2, kind, rng)
    assert abs(np.sqrt(np.mean(n.astype(float) ** 2)) - 1) < 1e-3
    s = _tone(2.0, amp=0.1)
    y = L.mix_at_snr(s, n, 10.0)
    added = y.astype(float) - s
    snr = 10 * np.log10(np.mean(s.astype(float) ** 2) / np.mean(added ** 2))
    assert abs(snr - 10.0) < 0.1
    assert np.max(np.abs(y)) <= 0.99 + 1e-6


def test_concat_pair_levels_and_length():
    a, b = _tone(1.0, amp=0.5), _tone(2.0, amp=0.05)
    la = 10 * np.log10(np.mean(a.astype(float) ** 2))
    lb = 10 * np.log10(np.mean(b.astype(float) ** 2))
    y = L.concat_pair(a, b, 0.5, la, lb, target_db=-23.0)
    assert len(y) == len(a) + len(b) + int(0.5 * L.SR)
    pa = 10 * np.log10(np.mean(y[: len(a)].astype(float) ** 2))
    pb = 10 * np.log10(np.mean(y[-len(b):].astype(float) ** 2))
    assert abs(pa - pb) < 0.5  # both parts brought to the same level


def test_score_items_keep_order():
    import random

    import common as C

    class NoShuffle(random.Random):
        def shuffle(self, x, *a, **k):
            return None

    it = C.make_item(item_id="x", set_name="train", source="s", question="q", options=["slow", "normal", "fast"],
                     target=[0.0, 0.3, 0.7], gold=2, rng=NoShuffle(0), meta={"qtype": "score"})
    assert it["options"] == ["slow", "normal", "fast"] and it["gold"] == 2
