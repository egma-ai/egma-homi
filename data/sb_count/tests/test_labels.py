import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import labels as L  # noqa: E402


def test_turns_merge_same_speaker_pauses():
    runs = [(0, 1, "a"), (1.5, 2.5, "a"), (3, 4, "b"), (4.2, 5, "a")]
    assert L.count_turns(runs) == 3
    assert L.turns_per_speaker(runs) == {"a": 2, "b": 1}
    assert L.speaker_order(runs) == ["a", "b"]


def test_overlap_still_counts_as_change_and_containment_detected():
    runs = [(0, 2, "a"), (1.7, 3, "b")]
    assert L.count_turns(runs) == 2
    assert not L.contained(runs)
    assert abs(L.max_overlap(runs) - 0.3) < 1e-9
    assert L.contained([(0, 3, "a"), (1, 1.5, "b")])


def test_talk_time_and_margin():
    runs = [(0, 3, "a"), (3.5, 4, "b"), (4.5, 7, "a")]
    tt = L.talk_time(runs)
    assert abs(tt["a"] - 5.5) < 1e-9 and abs(tt["b"] - 0.5) < 1e-9
    assert L.argmax_margin(tt, 1.3, 1.5) == "a"
    assert L.argmax_margin({"a": 2.0, "b": 1.9}, 1.3, 1.5) is None


def test_count_window_contains_gold_and_respects_bounds():
    rng = random.Random(0)
    for n in range(1, 13):
        for _ in range(20):
            w = L.count_window(n, 1, 14, rng)
            assert n in w and min(w) >= 1 and max(w) <= 14 and w == list(range(w[0], w[0] + len(w)))


def test_noul_count_truth_value():
    rng = random.Random(1)
    for n in range(1, 13):
        for want in (True, False):
            kind, N = L.noul_count(n, want, rng)
            truth = {"more": n > N, "atleast": n >= N, "exactly": n == N, "fewer": n < N}[kind]
            assert truth == want
            assert N >= 1 and not (kind in ("atleast", "fewer") and N < 2)


def test_bins_and_cap_balance():
    bins = [("a", 1, 1), ("b", 2, 3), ("c", 4, 99)]
    assert [L.bin_index(n, bins) for n in (1, 2, 3, 4, 50)] == [0, 1, 1, 2, 2]
    keep = L.cap_balance(["x"] * 10 + ["y"] * 2)
    assert sum(1 for i in keep if i < 10) == 4 and len(keep) == 6


def test_components_and_cal_groups_are_speaker_closed():
    groups = {"m1": {"A", "B"}, "m2": {"B", "C"}, "m3": {"D"}, "m4": {"E", "F"}, "m5": {"F"}}
    comps = L.components(groups)
    assert {"m1", "m2"} in comps and {"m3"} in comps and {"m4", "m5"} in comps
    cal = L.cal_groups(groups, 0.3, random.Random(0))
    cal_spk = {s for g in cal for s in groups[g]}
    assert all(not (groups[g] & cal_spk) for g in groups if g not in cal)


def test_energy_vad_finds_bursts():
    sr = 16000
    x = np.zeros(3 * sr, np.float32)
    t = np.arange(int(0.5 * sr)) / sr
    x[sr // 2: sr // 2 + len(t)] = 0.3 * np.sin(2 * np.pi * 200 * t)
    x[2 * sr: 2 * sr + len(t)] = 0.3 * np.sin(2 * np.pi * 200 * t)
    iv = L.energy_vad(x, sr)
    assert len(iv) == 2
    assert abs(iv[0][0] - 0.5) < 0.05 and abs(iv[1][1] - 2.5) < 0.05


def test_trim_bounds():
    sr = 16000
    x = np.zeros(2 * sr, np.float32)
    x[sr // 2: sr] = 0.2
    a, b = L.trim_bounds(x, sr)
    assert abs(a / sr - 0.46) < 0.02 and abs(b / sr - 1.04) < 0.02
