import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "inference"))

import build as B  # noqa: E402
import llm_label  # noqa: E402


def test_run_order_roundtrip():
    for n in (2, 3, 6, 8):
        for run in (0, 1):
            order = B.run_order(n, run)
            probs = [0.0] * n
            probs[0] = 1.0  # first displayed option
            canon = B.to_canonical(probs, order)
            assert canon[order[0]] == 1.0
    assert B.run_order(3, 1) == [2, 1, 0]


def test_agree_rule():
    ok, top, mean = B.agree([0.8, 0.1, 0.1], [0.75, 0.2, 0.05])
    assert ok and top == 0 and abs(mean.sum() - 1) < 1e-9
    assert not B.agree([0.8, 0.2, 0.0], [0.6, 0.4, 0.0])[0]  # second run below 0.7
    assert not B.agree([0.9, 0.1], [0.1, 0.9])[0]  # different top


def test_label_request_letters_follow_order():
    r = B.label_request("x", "hvb", "AGENT: hi", "Q?", ["a", "b", "c"], run=1)
    assert "A. c" in r["user"] and "C. a" in r["user"] and r["letters"] == ["A", "B", "C"]


def test_letter_probs_merges_variants():
    p, mass = llm_label.letter_probs({"A": np.log(0.5), " A": np.log(0.2), "B": np.log(0.2), "x": np.log(0.1)}, ["A", "B"])
    assert abs(p[0] - 0.7 / 0.9) < 1e-6 and abs(mass - 0.9) < 1e-6


def test_views_sentiment_and_resolved():
    v = B.views_for("hvb", "sentiment", np.array([0.85, 0.1, 0.05]))
    kinds = {x["view"]: x for x in v}
    assert kinds["sentiment_neg"]["label"] == "yes" and kinds["sentiment_pos"]["label"] == "no"
    assert kinds["sentiment_score"]["options"] == ["negative", "neutral", "positive"]
    for x in v:
        assert abs(sum(x["target"]) - 1) < 1e-9
        assert x["options"][int(np.argmax(x["target"]))] == x["label"]
    assert B.views_for("hvb", "resolved", np.array([0.1, 0.1, 0.8])) == []
    r = B.views_for("hvb", "resolved", np.array([0.1, 0.85, 0.05]))[0]
    assert r["label"] == "no" and len(r["target"]) == 2
    assert B.views_for("e22", "tone", np.array([0.05, 0.05, 0.9])) == []


def test_balance_noul():
    cands = [{"view": "frustrated", "qtype": "noul", "label": "no"} for _ in range(100)]
    cands += [{"view": "frustrated", "qtype": "noul", "label": "yes"} for _ in range(10)]
    cands += [{"view": "emotion", "qtype": "choice", "label": "neutral"} for _ in range(7)]
    out = B.balance_noul(cands, random.Random(0))
    c = Counter((x["view"], x["label"]) for x in out)
    assert c[("frustrated", "yes")] == 10 and c[("frustrated", "no")] == 11 and c[("emotion", "neutral")] == 7


def test_allocate_caps_items_per_clip():
    cands = {("hvb", "g"): [{"view": f"v{i}", "label_key": f"k{i // 2}"} for i in range(12)]}
    out = B.allocate(cands, {("hvb", "g"): ["a", "b"]}, random.Random(0))
    per_clip = Counter(x["clip"] for x in out)
    assert max(per_clip.values()) <= B.MAX_ITEMS_PER_CLIP and len(out) == 8


def test_hvb_turns_merge_and_shift():
    segs = [{"speaker_role": "agent", "start_ms": 0, "duration_ms": 1000, "offset_ms": 500, "human_transcript": "hello"},
            {"speaker_role": "agent", "start_ms": 1200, "duration_ms": 500, "offset_ms": 1700, "human_transcript": "bank"},
            {"speaker_role": "caller", "start_ms": 2000, "duration_ms": 900, "offset_ms": 2000, "human_transcript": "hi [noise]"}]
    t = B.hvb_turns(segs)
    assert [x["role"] for x in t] == ["agent", "caller"] and t[0]["text"] == "hello bank" and t[1]["text"] == "hi"
    assert B.channel_shift_ms(segs, "agent") == 500 and B.channel_shift_ms(segs, "caller") == 0
    assert B.hvb_transcript(t).startswith("AGENT: hello bank\nCALLER: hi")


def test_nlp_windows():
    lines = ["token|speaker|ts|endTs|punctuation|prepunctuation|case|tags|oldTs|oldEndTs|ali_comment"]
    t = 0.0
    for i in range(400):
        spk = "0" if (i // 40) % 2 == 0 else "1"
        punct = "." if i % 10 == 9 else ""
        lines.append(f"w{i}|{spk}|{t:.2f}|{t + 0.4:.2f}|{punct}||LC|[]|||")
        t += 0.5
    sents = B.nlp_sentences(B.parse_nlp("\n".join(lines)))
    assert len(sents) == 40 and sents[0]["start"] == 0.0
    wins = B.make_windows(sents, random.Random(0), lo=30, hi=60)
    assert wins and all(30 <= w["end"] - w["start"] <= 61 for w in wins)
    for a, b in zip(wins, wins[1:]):
        assert b["start"] >= a["end"] - 1.0  # non-overlapping
    tr = B.window_transcript(sents, wins[0]["i0"], wins[0]["i1"])
    assert tr.startswith("S1: ")


def test_validate_item_flags_bad_gold():
    import common

    it = common.make_item(item_id="x", set_name="train", source="s", question="q", options=["yes", "no"],
                          target=[0.9, 0.1], gold=0, rng=random.Random(0), audio={"bank": "b", "key": "k"},
                          meta={"qtype": "noul"})
    assert B.validate_item(it) == []
    it["gold"] = 1 - it["gold"]
    assert "gold != argmax" in B.validate_item(it)


def test_score_rng_keeps_order():
    import common

    it = common.make_item(item_id="x", set_name="train", source="s", question="q", options=B.SENTIMENT,
                          target=[0.1, 0.2, 0.7], gold=2, rng=B._NoShuffle(), meta={"qtype": "score"})
    assert it["options"] == B.SENTIMENT and it["gold"] == 2


def test_choose_cal_groups_covers_fraction():
    groups = [f"f{i // 30}" for i in range(3000)]  # 100 groups of 30 clips
    cal = B.choose_cal_groups(groups, frac=0.03)
    n = sum(g in cal for g in groups)
    assert 90 <= n <= 120 and len(cal) in (3, 4)
