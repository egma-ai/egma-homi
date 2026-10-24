"""Pure-logic tests for sb_spoken2 (labels of the generated puzzles are re-derived independently)."""
import itertools
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "inference"))

import build as B  # noqa: E402


def test_numbers_and_ordinals():
    assert B.num_words(0) == "zero"
    assert B.num_words(21) == "twenty-one"
    assert B.num_words(105) == "one hundred and five"
    assert B.ordinal_words(1) == "first"
    assert B.ordinal_words(22) == "twenty-second"
    assert B.ordinal_words(30) == "thirtieth"
    assert B.ordinal_words(13) == "thirteenth"
    assert B.clock_words(7, 5) == "seven oh five"
    assert B.clock_words(9, 0) == "nine o'clock"
    assert B.spoken_option("8:45") == "eight forty-five"
    assert B.spoken_option("May 3") == "May third"
    assert B.spoken_option("12") == "twelve"


def test_truth_chain():
    # A lies; B says A tells the truth -> B lies; C says B lies -> C tells the truth
    assert B.truth_values(False, [True, False]) == [False, False, True]
    assert B.truth_values(True, [True, True, False]) == [True, True, True, False]


def test_syllogism_validity():
    S, M, P = 0, 1, 2
    # Barbara: all M are P, all S are M => all S are P (valid)
    assert B.syllogism_valid([("A", M, P), ("A", S, M)], ("A", S, P), False)
    # undistributed middle: all P are M, all S are M => all S are P (invalid)
    assert not B.syllogism_valid([("A", P, M), ("A", S, M)], ("A", S, P), False)
    # Barbari (=> some S are P) is valid only with existential import -> the generator never uses it
    assert not B.syllogism_valid([("A", M, P), ("A", S, M)], ("I", S, P), False)
    assert B.syllogism_valid([("A", M, P), ("A", S, M)], ("I", S, P), True)
    for i in range(40):
        for want in (True, False):
            pz = B.gen_syllogism(B.rng_for("t", i, want), want)
            f = [tuple(x) for x in pz["extra"]["forms"]]
            assert B.syllogism_valid(f[:2], f[2], False) == want == B.syllogism_valid(f[:2], f[2], True)


def _order_constraints(text, rel):
    more = re.escape(rel["more"]).replace(r"\{a\}", r"(\w+)").replace(r"\{b\}", r"(\w+)")
    less = re.escape(rel["less"]).replace(r"\{a\}", r"(\w+)").replace(r"\{b\}", r"(\w+)")
    cons = []
    for sent in re.findall(r"[^.]+\.", text):
        sent = sent.strip()
        m = re.fullmatch(more, sent)
        if m:
            cons.append((m.group(1), m.group(2)))  # a above b
            continue
        m = re.fullmatch(less, sent)
        assert m, sent
        cons.append((m.group(2), m.group(1)))
    return cons


def test_order_puzzles_have_unique_exact_answer():
    for i in range(150):
        want = i % 2 == 0
        pz = B.gen_order(B.rng_for("ord", i), want)
        rel = B.ORDER_RELS[pz["extra"]["rel"]]
        cons = _order_constraints(pz["text"], rel)
        ppl = pz["choice"][0]["options"]
        sols = [p for p in itertools.permutations(ppl) if all(p.index(a) < p.index(b) for a, b in cons)]
        assert len(sols) == 1 and list(sols[0]) == pz["extra"]["rank"]
        cq = pz["choice"][0]
        top = cq["q"] == rel["top"]
        assert cq["options"][cq["gold"]] == (sols[0][0] if top else sols[0][-1])
        # yes/no label re-derived from the question text
        rank = sols[0]
        q = pz["yn"]["q"]
        for key, above in (("ynm", True), ("ynl", False)):
            pat = re.escape(rel[key]).replace(r"\{a\}", r"(\w+)").replace(r"\{b\}", r"(\w+)")
            m = re.fullmatch(pat, q)
            if m:
                a, b = m.group(1), m.group(2)
                truth = (rank.index(a) < rank.index(b)) if above else (rank.index(a) > rank.index(b))
                assert truth == want
                break
        else:
            raise AssertionError(q)


def test_facing_and_grid():
    for i in range(100):
        pz = B.gen_facing(B.rng_for("f", i), i % 2 == 0)
        d0 = B.DIRS.index(re.search(r"facing (\w+)\.", pz["text"]).group(1))
        d = d0
        for t in re.findall(r"turn (left|right|around)", pz["text"]):
            d = (d + {"left": -1, "right": 1, "around": 2}[t]) % 4
        assert pz["choice"][0]["gold"] == d
        pz = B.gen_grid(B.rng_for("g", i), i % 2 == 0)
        x = y = 0
        for k, dname in re.findall(r"(\w+(?:-\w+)?) blocks? (north|south|east|west)", pz["text"]):
            n = [w for w in range(1, 10) if B.num_words(w) == k][0]
            x += B.DVEC[dname][0] * n
            y += B.DVEC[dname][1] * n
        assert ((x, y) == (0, 0)) == pz["yn"]["yes"]


def test_dates_and_weekdays():
    assert B.shift_date("December", 30, 5) == ("January", 4)
    assert B.shift_date("March", 3, -5) == ("February", 26)
    assert B.shift_date("April", 28, 4) == ("May", 2)
    for i in range(100):
        pz = B.gen_weekday(B.rng_for("w", i), True)
        m = re.match(r"Today is (\w+)\.|It is (\w+) today\.", pz["text"])
        if m:
            d0 = B.WEEKDAYS.index(m.group(1) or m.group(2))
            q = pz["choice"][0]["q"]
            n = [w for w in range(2, 21) if f" {B.num_words(w)} days" in q][0]
            exp = (d0 + n) % 7 if "will it be" in q else (d0 - n) % 7
            assert pz["choice"][0]["options"][pz["choice"][0]["gold"]] == B.WEEKDAYS[exp]


def test_generated_specs_and_items_schema():
    specs, _ = B.generated_specs(0.02)
    assert specs and len({s["clip"] for s in specs}) == len(specs)
    B.assign_voices(specs, ["af_heart", "am_adam"], ["af_kore"])
    items = [it for s in specs for it in B.make_items(s, "bank", 5.0)]
    for it in items:
        assert set(it) == B.KEYS
        assert abs(sum(it["target"]) - 1) < 1e-9 and it["target"][it["gold"]] == 1.0
        assert it["meta"]["qtype"] in ("choice", "noul")
        assert 2 <= len(it["options"]) <= 20
        assert B.n_words(it["meta"]["transcript"]) <= B.MAX_TTS_WORDS
        if it["meta"]["variant"] == "read":  # options keep the spoken order: letter i in the audio = option i
            for i, o in enumerate(it["options"]):
                assert f"{B.LETTERS[i]}. {B.spoken_option(o)}." in it["meta"]["transcript"]
        if it["meta"]["qtype"] == "noul":
            assert sorted(it["options"]) == ["no", "yes"]
    assert all((it["set"] == "CAL") == (it["meta"]["voice"] == "af_kore") for it in items)


def test_dataset_specs_balance_and_exclusion():
    rows = [{"id": f"r{i}", "q": f"Where would you keep item number {i}?", "choices": ["a box", "a bag", "a shelf", "a car", "a room"],
             "gold": i % 5} for i in range(400)]
    specs, st = B.dataset_specs(rows, ds="t", source="spoken_t", licence="MIT", dataset="x", n_max=10 ** 6,
                                fmt_weights=[("read", 0.4), ("stem", 0.3), ("yn", 0.3)], read_k=4,
                                exclude={"r0", "r1"})
    assert st["excluded_v1_row"] == 2 and all(s["row_id"] not in ("r0", "r1") for s in specs)
    yn = [s["qas"][0]["gold"] for s in specs if s["variant"] == "yn"]
    assert abs(yn.count(0) - yn.count(1)) <= 1
    for s in specs:
        if s["variant"] == "read":
            qa = s["qas"][0]
            assert len(qa["options"]) == 4 and not qa["shuffle"]
            assert qa["options"][qa["gold"]] == rows[int(s["row_id"][1:])]["choices"][rows[int(s["row_id"][1:])]["gold"]]
