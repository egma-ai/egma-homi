import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import labels as L  # noqa: E402
import questions as Q  # noqa: E402


def test_segments_bridge_short_silences_only():
    iv = [(0.0, 1.0), (1.2, 2.0), (2.5, 3.0), (3.05, 3.08)]
    assert L.count_segments(iv, 0.3) == 2       # 0.2 s gap bridged, 0.5 s gap splits, 30 ms blip dropped
    assert L.count_segments(iv, 0.1) == 3


def test_turn_sequence_never_repeats_speaker():
    rng = random.Random(0)
    for _ in range(300):
        n = rng.randint(2, 8)
        counts = [rng.randint(1, 3) for _ in range(n)]
        seq = L.turn_sequence(counts, rng)
        if max(counts) > sum(counts) - max(counts) + 1:
            assert seq is None
            continue
        assert seq is not None and sorted(seq) == sorted(i for i, c in enumerate(counts) for _ in range(c))
        assert all(a != b for a, b in zip(seq, seq[1:]))


def test_design_has_exactly_N_speakers_and_bounds():
    rng = random.Random(1)
    for _ in range(500):
        N = rng.randint(1, 8)
        d = L.design_timeline(rng, N, L.draw_T(rng, N), rng.random() < 0.5, rng.random() < 0.2, rng.choice(["free", "natural"]))
        assert sorted(set(d["seq"])) == list(range(N))
        assert all(c in (1, 2, 3, 4) for c in d["counts"]) and (N == 1 or max(d["counts"]) <= 3)
        assert all(L.SHORT_LO - 1e-9 <= d["dur"][p] <= L.SHORT_HI + 1e-9 for p, s in enumerate(d["seq"]) if s in d["short"])
        assert all(g >= -0.3 - 1e-9 and g <= 1.0 + 1e-9 for g in d["gaps"])
        assert all(g >= 0.1 - 1e-9 for i, g in enumerate(d["gaps"]) if i != d["overlap_join"])


def test_duration_drawn_independently_of_N():
    rng = random.Random(2)
    Ns, Ts = [], []
    for _ in range(6000):
        N = rng.randint(1, 8)
        Ns.append(N)
        Ts.append(L.draw_T(rng, N))
    assert abs(L.pearson(Ts, Ns)) < 0.06
    assert min(Ts) >= 2.5 and max(Ts) <= 24.0


def test_noul_truth_and_near_misses():
    rng = random.Random(3)
    for n in range(1, 9):
        for kind in ("exactly", "more", "fewer", "atleast"):
            for want in (True, False):
                N = L.noul_count(n, kind, want, rng)
                if N is None:
                    continue
                assert L.noul_truth(n, kind, N) == want and abs(N - n) <= 3


def test_count_window_and_score():
    rng = random.Random(4)
    for n in range(1, 9):
        for _ in range(30):
            w = L.count_window(n, 1, 10, rng)
            assert n in w and w == list(range(w[0], w[0] + len(w))) and w[0] >= 1 and w[-1] <= 10
    assert [L.score_index(n) for n in range(1, 9)] == [0, 1, 2, 3, 3, 3, 3, 3]


def test_option_wording_singular():
    assert Q.fmt_count(1, "{n} people") == "1 person"
    assert Q.fmt_count(1, "{w} different voices") == "one different voice"
    assert Q.fmt_count(3, "{w} speakers") == "three speakers"
    assert Q.noul_question("Do exactly {n} unique talkers appear in this clip?", 1, True) == "Do exactly one unique talker appear in this clip?"


def test_enough_paraphrases():
    assert len(Q.CHOICE) >= 8 and len(Q.SCORE) >= 8
    assert all(len(v) >= 8 for v in Q.NOUL.values())


def test_median_f0_on_tone():
    sr = 16000
    t = np.arange(sr) / sr
    x = 0.3 * np.sign(np.sin(2 * np.pi * 150 * t)) * (1 + 0.1 * np.sin(2 * np.pi * 3 * t))
    f = L.median_f0(x.astype(np.float32), sr)
    assert f is not None and abs(f - 150) < 6


def test_mmsu_equal_helper():
    assert L.mmsu_equal(["How many?", "Other"], ["  how   MANY? "]) == ["How many?"]
