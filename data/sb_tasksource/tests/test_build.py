import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "inference"))

import build as B  # noqa: E402


def row(**kw):
    base = {"id": "x:train:1:decision", "kind": "choice", "options": ["red", "green", "blue"], "target": [0.0, 1.0, 0.0],
            "question": "Which colour?", "state": "The sky at noon in summer looks like this.", "variant": "direct",
            "group_id": "g1", "question_id": "decision", "example_id": "e1", "split": "train"}
    base.update(kw)
    return base


def test_licence_tokens():
    ok = ["cc-by-4.0", "CC BY 4.0 (DPI)", "mit", "MIT License (DPI)", "apache-2.0", "Apache License 2.0 (DPI)",
          "cc0-1.0", "CC0 1.0 (DPI)", "odbl", "odc-by", "bsd", "BSD 2-Clause License (DPI)", "cc-by-3.0"]
    sa = ["cc-by-sa-4.0", "CC BY-SA 4.0 (DPI)", "cc-by-sa-3.0"]
    bad = ["cc-by-nc-4.0", "CC BY-NC-SA 4.0 (DPI)", "cc-by-nd-4.0", "gpl", "gpl-3.0", "agpl-3.0", "mpl-2.0", "afl-3.0",
           "OANC (DPI)", "Custom (DPI)", "cc", "other", "unspecified", "openrail", "llama3"]
    assert all(B.licence_token_class(t) == "ok" for t in ok), [t for t in ok if B.licence_token_class(t) != "ok"]
    assert all(B.licence_token_class(t) == "share_alike" for t in sa)
    assert all(B.licence_token_class(t) == "bad" for t in bad), [t for t in bad if B.licence_token_class(t) != "bad"]


def test_licence_decision_most_restrictive():
    ts = {"license": "mit, other", "license_use": "commercial"}
    keep, why, _ = B.licence_decision("head_qa/en", ts, {"license": "mit, other", "license_use": "commercial"}, {"commercial"})
    assert not keep and "other" in why
    keep, _, rec = B.licence_decision("snli", {"license": "cc-by-sa-4.0", "license_use": "commercial"},
                                      {"license": "cc-by-sa-4.0", "license_use": "commercial"}, {"commercial"})
    assert keep and rec["share_alike"]
    keep, why, _ = B.licence_decision("x", {"license": "mit", "license_use": "commercial"},
                                      {"license": "mit", "license_use": "commercial"}, {"commercial", "unspecified"})
    assert not keep


def test_leaderboard_exclusion():
    for t, repos in [("super_glue/boolq", []), ("ai2_arc/ARC-Easy/challenge", []), ("x", ["Rowan/hellaswag"]),
                     ("contract-nli/contractnli_a/seg", []), ("clinc_oos/plus", []), ("anli/a1", []),
                     ("allenai/swag", []), ("tweet_eval/irony", [])]:
        assert B.leaderboard_match(t, repos), t
    for t in ["glue/rte", "snli", "commonsense_qa", "civil_comments/toxicity_share", "procedural-typed-decisions/arithmetic"]:
        assert B.leaderboard_match(t, []) is None, t
    assert B.leaderboard_match("robust_nli/IS_CS", ["pietrolesci/robust_nli"]) is None  # author names are not matched


def test_convert_noul_hard_and_soft():
    it = B.convert_row(row(kind="noul", options=[], target=[1.0]), set_name="train", source="ts:t", meta_extra={})
    assert sorted(it["options"]) == ["no", "yes"] and it["options"][it["gold"]] == "yes"
    assert abs(max(it["target"]) - 0.975) < 1e-9 and it["meta"]["qtype"] == "noul" and not it["meta"]["soft"]
    cal = B.convert_row(row(kind="noul", options=[], target=[0.0]), set_name="CAL", source="ts:t", meta_extra={})
    assert cal["options"][cal["gold"]] == "no" and max(cal["target"]) == 1.0
    soft = B.convert_row(row(kind="noul", options=[], target=[0.3]), set_name="train", source="ts:t", meta_extra={})
    assert soft["meta"]["soft"] and abs(soft["target"][soft["options"].index("yes")] - 0.3) < 1e-9


def test_convert_score_keeps_order_and_soft_choice():
    it = B.convert_row(row(kind="score", options=["1", "2", "3", "4", "5"], target=[0, 0, 0.6, 0.4, 0]),
                       set_name="train", source="ts:t", meta_extra={})
    assert it["options"] == ["1", "2", "3", "4", "5"] and it["gold"] == 2 and it["meta"]["soft"]
    assert abs(sum(it["target"]) - 1) < 1e-4
    hard = B.convert_row(row(), set_name="train", source="ts:t", meta_extra={})
    assert hard["options"][hard["gold"]] == "green" and hard["gold"] == int(np.argmax(hard["target"]))
    assert abs(sum(hard["target"]) - 1) < 1e-4


def test_positional_options_keep_order():
    opts = ["Only A", "B", "Both A and B", "all of the above"]
    assert B.positional_refs(opts)
    it = B.convert_row(row(options=opts, target=[0, 0, 1, 0]), set_name="train", source="ts:t", meta_extra={})
    assert it["options"] == opts and it["meta"]["positional_order_kept"]
    assert not B.positional_refs(["red", "green", "none of the other options"])


def test_row_problem():
    assert B.row_problem(row()) is None
    assert B.row_problem(row(state="x" * (B.MAX_STATE_CHARS + 1))) == "state too long"
    assert B.row_problem(row(options=[str(i) for i in range(41)], target=[1.0] + [0.0] * 40)) == "option count"
    assert B.row_problem(row(options=["a", "A", "b"])) == "duplicate options"


def test_quotas_caps():
    avail = {f"fam{i}": 5000 for i in range(10)} | {f"big/t{i}": 5000 for i in range(20)} | \
            {f"multilingual/x/{i}": 5000 for i in range(30)}
    q = B.task_quotas(avail, target=20000, task_cap=800, family_cap=2400, multilingual_share=0.05)
    assert max(q.values()) <= 800
    assert sum(v for k, v in q.items() if k.startswith("big/")) <= 2400
    assert sum(v for k, v in q.items() if k.startswith("multilingual/")) <= 0.05 * 20000 + 1
    assert abs(sum(q.values()) - min(20000, 10 * 800 + 2400 + 1000)) <= 60


def test_cal_train_disjoint():
    rows = []
    for g in range(50):
        s = f"Premise number {g} is a long enough sentence.\nHypothesis {g} also long enough here."
        rows.append(row(id=f"r{g}a", group_id=f"g{g}", example_id=f"e{g}", state=s))
        rows.append(row(id=f"r{g}b", group_id=f"g{g}", example_id=f"e{g}", state=s, variant="label_verification"))
    # a packed train candidate that contains the text of example 0
    rows.append(row(id="pack", group_id="gp", example_id="e99", variant="packed_derived",
                    state="Item A:\ntext_A: Premise number 0 is a long enough sentence.\nItem B:\ntext_A: other text here ok"))
    gids, groups = B.order_groups(rows, seed="s")
    cal, cal_g = B.choose_cal(gids, groups, 10)
    ex = {r["example_id"] for r in cal}
    keys = {k for r in cal for k in B.line_keys(r["state"])}
    train, _ = B.fill_train(gids, groups, 10_000, cal_g, ex, keys)
    assert not ({r["group_id"] for r in train} & cal_g) and not ({r["example_id"] for r in train} & ex)
    if "e0" in ex:
        assert all(r["id"] != "pack" for r in train)


def test_rebalance_noul():
    items = []
    for i in range(100):
        yes = i < 20
        items.append({"options": ["yes", "no"], "gold": 0 if yes else 1, "meta": {"qtype": "noul", "soft": False, "task": "t"}})
    kept, dropped = B.rebalance_noul(items, seed="s")
    n_yes = sum(it["gold"] == 0 for it in kept)
    assert n_yes == 20 and len(kept) - n_yes == 30 and dropped == 50


def test_noul_hint_only_for_non_yes_no_questions():
    it = B.convert_row(row(kind="noul", options=[], target=[0.2], question="How likely is text_B given text_A?"),
                       set_name="train", source="ts:t", meta_extra={})
    assert it["question"].endswith(B.NOUL_HINT)
    it = B.convert_row(row(kind="noul", options=[], target=[1.0], question="Is the review positive?"),
                       set_name="train", source="ts:t", meta_extra={})
    assert it["question"] == "Is the review positive?"


def test_assign_sources_limit():
    items = [{"meta": {"family": f"fam{i}", "bucket": f"b{i % 5}"}} for i in range(100) for _ in range(i + 1)]
    m = B.assign_sources(items, max_sources=20)
    assert len(set(m.values())) <= 20 and m["fam99"] == "ts:fam99" and m["fam0"] == "ts:b0"


def test_content_overlap():
    assert B.content_overlap_match("dgen") and B.content_overlap_match("dgen-x") is None
