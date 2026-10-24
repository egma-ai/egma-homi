import collections
import json
import os
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parent.parent / "inference"))
sys.path.insert(0, str(HERE.parent / "sb_phon"))
sys.path.insert(0, str(HERE))

import build  # noqa: E402
import lexbank as B  # noqa: E402
import lexicon as L  # noqa: E402

CMU = L.parse_cmudict("""night  N AY1 T
light  L AY1 T
delight  D IH0 L AY1 T
sight  S AY1 T
time  T AY1 M
river  R IH1 V ER0
alone  AH0 L OW1 N
stone  S T OW1 N
""")


def test_combos_complete_and_names_unique():
    assert len(build.COMBOS) == 16 and len(set(build.COMBOS)) == 16
    assert build.COMBOS[0] == () and build.COMBOS[-1] == tuple(build.TYPES)
    for style in build.NAMES:
        names = [build.combo_name(c, style) for c in build.COMBOS]
        assert len(set(names)) == 16, style
    assert build.combo_name(("filled", "restart"), "plain") == "filled pauses and restarts"
    assert build.combo_name((), "plain") == "no disfluency"


def test_combo_option_sets_symmetric():
    """Given an option set, the correct answer must not be identifiable by its position in the set (centre vs leaf)."""
    rng = random.Random(0)
    centre_hits = collections.Counter()
    for j in range(4000):
        correct = build.COMBOS[j % 16]
        s = build.combo_option_set(correct, 5, rng)
        assert correct in s and len(set(s)) == len(s) == 5
        star = all(len(set(s[0]) ^ set(d)) == 1 for d in s[1:])
        if star:
            centre_hits[s[0] == correct] += 1
    # in star-shaped sets the centre is the answer about 1/5 of the time, not always
    frac = centre_hits[True] / max(1, sum(centre_hits.values()))
    assert 0.12 < frac < 0.3, frac


def test_base_sentences_have_no_disfluency_vocabulary():
    ok = build.base_sentences()
    assert len(ok) >= 80
    assert not build.check_base_sentence("Well the [red|blue] car is here")
    assert not build.check_base_sentence("I mean the [red|blue] car")
    for s in ok:
        words = set(re.sub(r"[\[\]|]", " ", s).lower().split())
        assert not words & B.DISFL_VOCAB, s


def test_disfluent_scripts_match_their_labels():
    rng = random.Random(1)
    sents = build.base_sentences()
    for j in range(800):
        types = build.COMBOS[j % 16]
        text, info = build.make_disfluent(rng, types, rng.choice(sents))
        lab = build.disfl_labels(info)
        assert sorted(t for t, v in lab.items() if v) == sorted(types), (text, types)
        spoken = [w.strip(",.—").lower() for w in text.split() if w.strip(",.—")]
        assert spoken == info["spoken"]
        if "editing" not in types:
            assert "eterm" not in info["kinds"]
        if "filled" in types:
            assert sum(w in B.FILLED for w in spoken) == info["n_fill"] >= 1
        else:
            assert not any(w in B.FILLED for w in spoken)
        if "restart" in types:
            assert "—" in text
        else:
            assert "—" not in text


def test_rhyme_keys_and_identical_rhyme():
    assert build.cmu_rhyme(CMU["night"][0]) == ("AY", "T") == build.cmu_rhyme(CMU["delight"][0])
    assert build.cmu_rhyme(CMU["time"][0]) != build.cmu_rhyme(CMU["night"][0])
    assert build.cmu_pre(CMU["light"][0]) == build.cmu_pre(CMU["delight"][0]) == "L"  # identical rhyme, never paired
    assert build.cmu_pre(CMU["night"][0]) == "N"
    assert build.cmu_rhyme(CMU["alone"][0]) == build.cmu_rhyme(CMU["stone"][0])
    assert build.misaki_rhyme("nˈIt") == build.misaki_rhyme("dəlˈIt") == "It"
    assert build.misaki_rhyme("ɹˈɪvəɹ") == "ɪvəɹ"
    assert build._stress_digits(CMU["river"][0]) == "10"
    k1, k2 = (("AY", "T"), "It", "It"), (("AY", "M"), "Im", "Im")
    assert build.strict_nonrhyme(k1, k2) and not build.strict_nonrhyme(k1, (("AY", "T"), "x", "y"))


def test_meter_patterns_and_partners():
    for m, (fam, p) in B.METERS.items():
        assert p.endswith("1") and "11" not in p and fam in B.FAMILIES
    rng = random.Random(0)
    for m in B.METERS:
        for _ in range(50):
            m2, how = build.alt_meter(m, rng)
            p, p2 = B.METERS[m][1], B.METERS[m2][1]
            assert p != p2
            assert (how == "stress" and len(p) == len(p2)) or (how == "count" and abs(len(p) - len(p2)) >= 2)


def test_verse_ok_rule():
    sp = {"pattern": "0101", "rkey": [["AY", "T"], "It", "It"]}
    assert build.verse_ok(sp, {"pattern": "0101", "rkey": [["AY", "T"], "It", "It"]})
    assert not build.verse_ok(sp, {"pattern": "1001", "rkey": [["AY", "T"], "It", "It"]})
    assert not build.verse_ok(sp, {"pattern": "0101", "rkey": [["OW", "N"], "On", "Qn"]})


def test_paraphrase_counts():
    for k, v in build.Q.items():
        assert len(v) >= 8 and len(set(v)) == len(v), k


def test_mmsu_equality_is_detected():
    it = {"id": "a", "set": "train", "source": "sb_disfluency_disfl", "question": "Which word is it?", "options": ["yes", "no"],
          "target": [1.0, 0.0], "gold": 0, "audio": {"bank": "b", "key": "k"}, "channel": "clean", "state_text": None,
          "meta": {"qtype": "noul", "qname": "x"}}
    assert build.validate([it], {"b": {"k": [0, 1]}}, ["which word is it"]).get("mmsu_question_equal") == 1
    assert build.validate([it], {"b": {"k": [0, 1]}}, ["Something else?"]).get("mmsu_question_equal") is None


def test_no_question_template_equals_an_mmsu_question():
    """Checks against the MMSU 'question' strings when they are mounted (inside Modal); a no-op elsewhere."""
    p = Path(os.environ.get("MMSU_TRUTH", "/vol/bench_mmsu/truth.jsonl"))
    if not p.exists():
        return
    qs = [json.loads(line)["question"] for line in p.read_text().splitlines() if line.strip()]
    norm = {build.norm_q(q) for q in qs}
    for v in build.Q.values():
        for t in v:
            assert t not in qs and build.norm_q(t) not in norm
