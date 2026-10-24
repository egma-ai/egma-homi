import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "inference"))

import build as B  # noqa: E402


def test_number_label():
    assert B.number_label("It earns 2.00% interest for 6 years.") is True
    assert B.number_label("There were twenty people at the party.") is True
    assert B.number_label("Which one would you pick?") is None
    assert B.number_label("Where would you find a fox?") is False


def test_question_label():
    assert B.question_label("Where do you keep milk?") is True
    assert B.question_label("The screen impressed me.") is False
    assert B.question_label("what time is it") is None


def test_words_and_distractors():
    rng = random.Random(0)
    text = "The camera disappointed me but the delivery impressed me."
    w = B.said_word(text, rng)
    assert w in B.content_words(text) and w not in B.STOPWORDS
    vocab = ["camera", "cameras", "banana", "garden", "rocket", "pencil", "delivered", "violin"]
    ds = B.distractor_words(text, vocab, rng, 3)
    assert ds and all(d not in ("camera", "cameras", "delivered") for d in ds)


def test_csqa_specs_read_keeps_spoken_order_and_gold():
    rows = [{"id": f"r{i}", "question": f"Where would you put item {i}?", "question_concept": "x",
             "choices": {"label": list("ABCDE"), "text": ["shelf", "ocean", "pocket", "cloud", "river"]},
             "answerKey": "C"} for i in range(30)]
    specs, _ = B.csqa_specs(rows, 10, 10, 10)
    read = [s for s in specs if s["variant"] == "read"]
    assert read and all(s["qa"]["options"] == ["shelf", "ocean", "pocket", "cloud", "river"] for s in read)
    assert all(s["qa"]["gold"] == 2 and not s["qa"]["shuffle"] for s in read)
    assert all(("A. shelf." in s["text"]) or ("Option A: shelf." in s["text"]) for s in read)
    yn = [s for s in specs if s["variant"] == "yn"]
    c = Counter(s["qa"]["gold"] for s in yn)
    assert abs(c[0] - c[1]) <= 1
    for s in yn:  # yes <=> the asked option is the right one
        assert (s["qa"]["asked"] == "pocket") == (s["qa"]["gold"] == 0)


def _hvb_rows():
    rows = []
    intents = sorted(B.HVB_INTENTS)
    for c in range(40):
        for u in range(6):
            spk = str(c % 9) if u % 2 == 0 else str(9 + c % 7)
            acts = [["statement_problem"], ["acknowledge"], ["thanks"], ["question_general"], ["answer_general", "thanks"], ["apology"]][u]
            rows.append({"issue_id": f"call{c}", "speaker_id": spk, "text": "hello", "utt_index": u,
                         "channel": 1, "role": "caller" if u % 2 == 0 else "agent", "start_ms": 0,
                         "duration_ms": 1500, "intent": intents[c % 8], "dialog_acts": acts})
    return rows


def test_hvb_items_cal_disjoint_and_balanced():
    rows = _hvb_rows()
    durs = {B.hvb_key(r): 1.5 for r in rows}
    items, stats = B.build_hvb_items(rows, durs, adi_speakers={"0", "1", "2"})
    tr_spk = {it["meta"]["speaker"] for it in items if it["set"] == "train"}
    cal_spk = {it["meta"]["speaker"] for it in items if it["set"] == "CAL"}
    tr_calls = {it["meta"]["call"] for it in items if it["set"] == "train"}
    cal_calls = {it["meta"]["call"] for it in items if it["set"] == "CAL"}
    assert cal_spk and not (tr_spk & cal_spk) and not (tr_calls & cal_calls)
    for it in items:
        assert abs(sum(it["target"]) - 1) < 1e-9 and it["target"][it["gold"]] == 1.0
        if it["meta"]["qtype"] == "choice" and it["source"] == "hvb_intent":
            assert len(it["options"]) == 8
            assert it["options"][it["gold"]] == B.HVB_INTENTS[it["meta"]["intent"]]
        if it["meta"]["qtype"] == "choice" and it["source"] == "hvb_dialog_act":
            assert len(it["options"]) == 18
        if it["source"] == "hvb_dialog_act" and it["meta"]["qtype"] == "noul":
            yes = it["options"][it["gold"]] == "yes"
            assert yes == (it["meta"]["asked_act"] in it["meta"]["dialog_acts"])
        if it["source"] == "hvb_intent" and it["meta"]["qtype"] == "noul":
            yes = it["options"][it["gold"]] == "yes"
            assert yes == (it["meta"]["asked_intent"] == it["meta"]["intent"])
    yn = Counter(it["options"][it["gold"]] for it in items if it["source"] == "hvb_dialog_act" and it["meta"]["qtype"] == "noul")
    assert yn["yes"] == yn["no"]
    assert stats["max_items_per_clip"] <= B.MAX_ITEMS_PER_CLIP
