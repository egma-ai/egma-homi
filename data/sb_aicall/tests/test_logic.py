"""Unit tests for sb_aicall label, layout, question and selection logic.

    uv run --python 3.12 --with pytest --with numpy --with scipy pytest data/sb_aicall/tests
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
for p in (HERE.parents[1] / "inference", HERE.parent / "sb_phone", HERE.parent / "sb_calls", HERE.parent / "sb_dialog", HERE):
    sys.path.insert(0, str(p))

import aic_labels as AL  # noqa: E402
import aic_layout as LY  # noqa: E402
import aic_questions as Q  # noqa: E402
import aic_scripts as S  # noqa: E402
import build  # noqa: E402
import common  # noqa: E402


def test_wait_labels_with_margins():
    # other speaks 0-3, agent replies at 6.2 (3.2 s wait), other 8-10, agent 10.5-12
    a = [(6.2, 7.5), (10.5, 12.0)]
    o = [(0.0, 3.0), (8.0, 10.0)]
    c, lab = AL.waited_candidates(a, o, 13.0)
    d = dict((n, ans) for ans, n in c)
    assert d[1] == "yes" and d[2.5] == "yes"
    assert 3 not in d  # 3.2 is inside 3 +- 0.4 -> ambiguous, not asked
    assert d[4] == "no" and d[8] == "no"
    assert lab == "3 to 6 seconds" or lab is None
    assert AL.bin_value(3.2, AL.WAIT_BINS) is None and AL.bin_value(3.5, AL.WAIT_BINS) == "3 to 6 seconds"


def test_wait_other_side_pause_blocks_no():
    # the other side pauses 4 s and speaks again before the agent answers -> "no@3" must not be produced
    a = [(10.0, 12.0)]
    o = [(0.0, 2.0), (6.0, 9.5)]
    c, _ = AL.waited_candidates(a, o, 13.0)
    assert ("no", 3) not in c and ("no", 4) not in c
    assert ("no", 5) in c


def test_silence_and_energy_veto():
    sp = [(0.0, 2.0), (8.0, 10.0)]
    c, lab = AL.silence_candidates(sp, sp, 10.5, low_runs=[(2.0, 8.0)])
    assert ("yes", 5) in c and ("no", 8) in c and ("yes", 6) not in c
    assert lab == "5 to 10 seconds"
    # energy says the gap is not silent -> no "yes"
    c2, lab2 = AL.silence_candidates(sp, sp, 10.5, low_runs=[])
    assert not any(a == "yes" for a, _ in c2) and lab2 is None


def test_cutin_rules():
    agent = [(0.0, 5.0)]
    # strong barge-in, agent stops 0.6 s later
    assert AL.cutin_labels([(0.0, 3.6)], [(3.0, 6.0)]) == ("yes", "once", "yes")
    # agent keeps talking 2 s after the onset
    assert AL.cutin_labels(agent, [(3.0, 6.0)]) == ("yes", "once", "no")
    # a 0.4 s backchannel is not an interruption
    assert AL.cutin_labels(agent, [(2.0, 2.4)]) == ("no", "none", None)
    # caller starts 0.1 s before the agent ends -> negligible
    assert AL.cutin_labels(agent, [(4.9, 7.0)])[0] == "no"
    # 0.25 s overlap -> ambiguous, not asked
    assert AL.cutin_labels([(0.0, 3.25)], [(3.0, 6.0)]) == (None, None, None)


def test_share_bins_and_margins():
    assert AL.share_bin(0.10, *AL.SHARE_BINNINGS["fifths"]) == "less than 20%"
    assert AL.share_bin(0.21, *AL.SHARE_BINNINGS["fifths"]) is None
    assert AL.share_bin(0.45, *AL.SHARE_BINNINGS["fifths"]) == "more than 40%"
    assert AL.who_more(0.3) == "agent" and AL.who_more(0.5) == "equal" and AL.who_more(0.42) is None


def test_agree_drops_contradictions():
    truth = {"barge": [("yes", None)], "waited": [("yes", 2), ("no", 6)]}
    vad = {"barge": [("no", None)], "waited": [("yes", 2), ("yes", 6)]}
    assert AL.agree(truth, vad) == {"waited": [("yes", 2)]}


def _turn(role, dur, words):
    return {"role": role, "dur": dur, "words": words, "incl": words, "src": ("x",)}


def test_layout_labels_recomputed_and_deterministic():
    turns = []
    for i in range(14):
        turns.append(_turn("agent", 4.0, [(0.0, 1.8), (2.2, 3.9)]) if i % 2 == 0 else _turn("other", 3.0, [(0.0, 2.9)]))
    for k in range(30):
        rc = LY.draw_recipe(random.Random(k), "A")
        l1 = LY.make_layout(turns, rc, f"s{k}")
        l2 = LY.make_layout(turns, rc, f"s{k}")
        if l1 is None:
            continue
        assert l1["speech"] == l2["speech"] and l1["T"] == l2["T"]
        assert 20 <= l1["T"] <= 179.5
        labs, _ = AL.two_party(l1["speech"], l1["incl"], l1["T"])
        for f, cands in labs.items():
            for ans, par in cands:
                if f == "waited" and ans == "yes":
                    w, _ = AL.wait_gaps(l1["speech"]["agent"], l1["speech"]["other"], l1["T"])
                    assert max(w) >= par + AL.margin(par)


def test_questions_and_forbidden():
    rng = random.Random(0)
    fq = HERE.parent / "forbidden_questions.json"  # optional list of question strings items must not equal
    forb = {q.lower() for q in json.loads(fq.read_text())} if fq.exists() else set()
    for fam, ts in Q.T.items():
        assert len(ts) >= 6, fam
        for t in ts:
            for A, O in [("the AI receptionist", "the caller"), ("the agent", "the customer")]:
                q = Q.cap(t.format(A=A, O=O, N="3 seconds"))
                assert q.lower() not in forb, q
    q, opts, gold, qt = Q.build_question("wait_bin", "3 to 6 seconds", None, ("the AI agent", "the caller"), rng)
    assert opts[gold] == "3 to 6 seconds"
    assert S.n_agent_templates() >= 40


def test_selection_balanced_per_threshold():
    rng = random.Random(0)
    clips = []
    for k in range(400):
        w = rng.uniform(0.5, 9)
        labs = {"waited": [("yes" if w >= n + AL.margin(n) else "no", n) for n in AL.WAIT_NS
                           if abs(w - n) >= AL.margin(n)],
                "live_in": [("yes" if k % 5 else "no", None)]}
        clips.append({"key": f"c{k}", "split": "train", "labels": labs, "scenario": "A", "direction": "in",
                      "sub": "A_in", "bank": "b", "dur": 30.0, "licence": [], "source_id": f"s{k}"})
    picked = build.select_items(clips, random.Random(1))
    by = Counter((f, a, p) for _, f, a, p in picked)
    for n in AL.WAIT_NS:
        assert by[("waited", "yes", n)] == by[("waited", "no", n)]
    assert by[("live_in", "yes", None)] == by[("live_in", "no", None)] > 0
    items = build.make_items(picked, random.Random(2), common)
    assert all(abs(sum(it["target"]) - 1) < 1e-9 for it in items)


def test_tts_text_cleanup():
    assert S.tts_sentence("uhm what is your [noise] address") == "What is your address?"
    assert S.turn_text(["hello this is harper valley", "uh i can help"]) == "Hello this is harper valley. I can help."
