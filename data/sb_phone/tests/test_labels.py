import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "inference"))

import fx  # noqa: E402
import labels as L  # noqa: E402


def test_number_label():
    assert L.number_label("HES GOT ONE O EIGHT OVER SEVEN") is True
    assert L.number_label("ROGER WE HAVE 3 MINUTES") is True
    assert L.number_label("I WANT THE CHEAP ONE") is None  # lone 'one' = ambiguous
    assert L.number_label("I'M LOOKING FOR A CHEAP RESTAURANT") is False
    assert L.number_label("NUMBER ONE") is True
    assert L.number_label("COPY THE SECOND PAGE") is None
    assert L.number_label("") is None


def test_date_label():
    assert L.date_label("CAN I BOOK FOR MONDAY") is True
    assert L.date_label("SOMETIME IN JANUARY") is True
    assert L.date_label("MAY I HAVE THE ADDRESS") is None  # 'may' ambiguous
    assert L.date_label("I NEED IT TOMORROW") is None
    assert L.date_label("I WANT A CHEAP PUB") is False


def test_quality_levels_follow_scenarios():
    rng = random.Random(0)
    for sc, lvl in L.SCENARIOS_CLEAN.items():
        for _ in range(50):
            r = L.sample_recipe(sc, rng)
            assert L.quality_level(r) == lvl, (sc, r)


def test_applicable_by_construction():
    rng = random.Random(1)
    r = L.sample_recipe("music", rng)
    a = L.applicable(r, base_clean=True)
    assert a["music"] is True and a["talker"] is False and "quality" not in a and "noise" not in a
    r = L.sample_recipe("drop_sev", rng)
    a = L.applicable(r, base_clean=True)
    assert a["breakup"] is True and a["quality"] == "very poor" and a["poor"] is True and a["voice_clear"] is False
    r = L.sample_recipe("clean", rng)
    a = L.applicable(r, base_clean=True)
    assert a == {"breakup": False, "music": False, "talker": False, "quality": "clear", "poor": False,
                 "voice_clear": True, "noise": False, "far": False}
    a = L.applicable(L.sample_recipe("noise_mod", rng), base_clean=False)
    assert set(a) == {"breakup", "music", "talker"}  # real phone base: no quality / noise questions


def _speech(n=16000 * 4, seed=0):
    rs = np.random.RandomState(seed)
    t = np.arange(n) / 16000
    env = (np.sin(2 * np.pi * 3 * t) > -0.3).astype(np.float32)
    return (0.3 * env * np.sin(2 * np.pi * 220 * t) + 0.05 * env * rs.randn(n)).astype(np.float32)


def test_dropouts_fraction():
    x = _speech()
    y, f = fx.dropouts(x, 0.2, random.Random(0))
    assert 0.17 <= f["lost_frac"] <= 0.26 and f["n_drops"] > 5
    assert len(y) == len(x)


def test_mix_at_snr():
    x = _speech()
    n = np.random.RandomState(1).randn(16000 * 2).astype(np.float32)
    y = fx.mix_at_snr(x, n, 10.0, random.Random(0))
    b = y - x
    snr = 10 * np.log10(fx.active_power(x) / np.mean(b ** 2))
    assert abs(snr - 10.0) < 0.5


def test_clip_fraction():
    x = _speech()
    y, f = fx.hard_clip(x, 0.15)
    assert 0.05 <= f["clipped_frac"] <= 0.2 and np.max(np.abs(y)) <= 1.0
