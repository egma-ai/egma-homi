import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import teacher_core as tc  # noqa: E402


def item(options, question="Q?", state="S", audio=None):
    return {"id": "x", "question": question, "options": options, "state_text": state, "audio": audio, "gold": 0}


def test_yes_no_maps_through_noul_in_our_order():
    for opts in (["yes", "no"], ["no", "yes"]):
        row, mapping = tc.to_teacher_row(item(opts))
        assert row["question"] == {"type": "noul", "instructions": "Q?"}
        teacher = [0.2, 0.8]  # [false, true]
        ours = tc.map_back(teacher, mapping)
        assert ours[opts.index("yes")] == pytest.approx(0.8)
        assert ours[opts.index("no")] == pytest.approx(0.2)


def test_yes_no_choice_mode_keeps_our_order():
    row, mapping = tc.to_teacher_row(item(["no", "yes"]), yes_no_mode="choice")
    assert row["question"]["type"] == "choice"
    assert list(row["question"]["criteria"]) == ["no", "yes"]
    assert mapping == [0, 1]


def test_choice_keeps_order_and_duplicates_use_score_list():
    row, mapping = tc.to_teacher_row(item(["c", "a", "b"]))
    assert list(row["question"]["criteria"]) == ["c", "a", "b"] and mapping == [0, 1, 2]
    row, mapping = tc.to_teacher_row(item(["same", "same", "x"]))
    assert row["question"] == {"type": "score", "instructions": "Q?", "criteria": ["same", "same", "x"]}
    assert mapping == [0, 1, 2]


def test_state_none_becomes_empty_and_skip_rules():
    row, _ = tc.to_teacher_row(item(["a", "b"], state=None))
    assert row["state"] == ""
    assert tc.skip_reason(item(["a", "b"], audio={"bank": "b", "key": "k"})) == "audio"
    assert tc.skip_reason(item([str(i) for i in range(41)])) == "too_many_options"
    assert tc.skip_reason(item([str(i) for i in range(40)])) is None


def test_map_back_sums_to_one_and_batches_respect_budget():
    p = tc.map_back([1, 1, 1], [0, 1, 2])
    assert abs(sum(p) - 1) < 1e-9
    lengths = [10, 500, 30, 499, 7, 1000]
    batches = tc.make_batches(lengths, token_budget=1000, max_rows=3)
    assert sorted(i for b in batches for i in b) == list(range(6))
    for b in batches:
        assert len(b) <= 3 and len(b) * max(lengths[i] for i in b) <= 1000
