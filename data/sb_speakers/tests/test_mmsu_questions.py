"""No question this builder can emit equals an MMSU question string.

Reads only the "question" field of $MMSU_TRUTH (default /vol/bench_mmsu/truth.jsonl); skipped when it is not mounted."""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import labels as L  # noqa: E402
import questions as Q  # noqa: E402

P = Path(os.environ.get("MMSU_TRUTH", "/vol/bench_mmsu/truth.jsonl"))


@pytest.mark.skipif(not P.exists(), reason="MMSU truth file not mounted (run in Modal)")
def test_no_question_equals_mmsu():
    mmsu = [json.loads(line)["question"] for line in open(P)]
    assert len(mmsu) > 100
    emitted = Q.expand_all(10)
    assert len(emitted) > 500
    assert L.mmsu_equal(emitted, mmsu) == []
