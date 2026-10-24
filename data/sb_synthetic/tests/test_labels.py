import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "inference"))

import labels as L  # noqa: E402


def test_paraphrase_counts():
    for fam in (L.Q_IS_REAL, L.Q_IS_SYNTH, L.Q_CHOICE, L.Q_PAIR_SYNTH, L.Q_PAIR_REAL):
        assert len(set(fam)) >= 8
    assert len(set(L.OPT_REAL)) >= 5 and len(set(L.OPT_SYNTH)) >= 5 and len(L.OPT_PAIR) >= 5
    assert not set(L.OPT_REAL) & set(L.OPT_SYNTH)


@pytest.mark.parametrize("is_synth", [True, False])
def test_noul_view_answers(is_synth):
    rng = random.Random(0)
    q, a = L.noul_view(is_synth, "is_real", rng)
    assert q in L.Q_IS_REAL and a == ("no" if is_synth else "yes")
    q, a = L.noul_view(is_synth, "is_synth", rng)
    assert q in L.Q_IS_SYNTH and a == ("yes" if is_synth else "no")


def test_choice_and_pair_views():
    rng = random.Random(1)
    q, opts, g = L.choice_view(True, rng)
    assert opts[g] in L.OPT_SYNTH and opts[1 - g] in L.OPT_REAL
    q, opts, g = L.choice_view(False, rng)
    assert opts[g] in L.OPT_REAL
    for pos in (0, 1):
        q, opts, g = L.pair_view(pos, "synth", rng)
        assert g == pos and q in L.Q_PAIR_SYNTH
        q, opts, g = L.pair_view(pos, "real", rng)
        assert g == 1 - pos and q in L.Q_PAIR_REAL


def test_tts_text():
    assert L.tts_text("HE HAD NEVER SEEN I THINK SUCH A THING") == "He had never seen I think such a thing."
    assert L.tts_text("[noise] yes i'm calling about my card <unk>") == "Yes I'm calling about my card."
    assert L.tts_text("[laughter]") is None
    assert L.tts_text("Already fine?") == "Already fine?"


def test_duration_match_equalises_bins():
    rng = random.Random(0)
    real = [(f"r{i}", 2 + (i % 10) * 0.5) for i in range(200)]
    syn = [(f"s{i}", 2 + (i % 6) * 0.5) for i in range(150)]
    kr, ks = L.duration_match(real, syn, rng)
    assert len(kr) == len(ks)
    dr = dict(real)
    ds = dict(syn)
    assert abs(sum(dr[k] for k in kr) / len(kr) - sum(ds[k] for k in ks) / len(ks)) < 0.26


def test_mmsu_nonequality(tmp_path):
    p = tmp_path / "truth.jsonl"
    p.write_text("\n".join(json.dumps({"question": q, "answer": "x"}) for q in ["Is the speaker happy?", "What is said?"]))
    r = L.check_mmsu_nonequality(L.ALL_QUESTIONS, p)
    assert r["equal"] == 0
    with pytest.raises(AssertionError):
        L.check_mmsu_nonequality(L.ALL_QUESTIONS + ["is the speaker  HAPPY?"], p)
