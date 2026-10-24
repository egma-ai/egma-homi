"""Unit tests for sb_calls pure logic (plan, labels, allocation, clip assembly).

    uv run --python 3.12 --with pytest --with numpy --with scipy pytest data/sb_calls/tests
"""
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "inference"))

import audio_fx  # noqa: E402
import build  # noqa: E402
import common  # noqa: E402
import scripts  # noqa: E402


def _fake_segs(p):
    return [np.full(int(0.06 * len(s["text"]) * 16000), 0.3, np.float32) for s in p["segments"]]


def test_plan_deterministic_and_split_disjoint():
    a, b = scripts.build_plan(0, 0.05), scripts.build_plan(0, 0.05)
    assert a == b
    tr = [p for p in a if p["split"] == "train"]
    cal = [p for p in a if p["split"] == "CAL"]
    assert {p["voice"] for p in tr if p["voice"]} <= set(scripts.TRAIN_VOICES)
    assert {p["voice"] for p in cal if p["voice"]} <= set(scripts.CAL_VOICES)
    assert not ({t for p in tr for t in p["templates"]} & {t for p in cal for t in p["templates"]})
    assert not ({s["text"] for p in tr for s in p["segments"]} & {s["text"] for p in cal for s in p["segments"]})


def test_voices_balanced_per_class():
    plan = scripts.build_plan(0, 1.0)
    for cls in scripts.SPEECH_CLASSES:
        c = Counter(p["voice"] for p in plan if p["split"] == "train" and p["cls"] == cls)
        assert set(c) == set(scripts.TRAIN_VOICES)
        assert max(c.values()) - min(c.values()) <= 1


def test_labels():
    assert build.label_yes("live", "live_person", {}) is True
    assert build.label_yes("live", "personal_voicemail", {}) is False
    assert build.label_yes("vm", "carrier_voicemail", {}) is True
    assert build.label_yes("auto", "carrier_voicemail", {}) is None
    assert build.label_yes("tone", "personal_voicemail", {"beep_present": True}) is True
    assert build.label_yes("tone", "live_person", {}) is None
    assert build.label_yes("leave", "personal_voicemail", {"beep_present": False, "truncated": True}) is False
    assert build.label_yes("leave", "personal_voicemail", {"beep_present": False, "truncated": False,
                                                          "mentions_tone": False}) is None
    assert build.label_yes("busy", "not_in_service", {}) is True
    assert build.label_yes("busy", "no_answer", {}) is False


def test_assemble_beep_and_truncation_facts():
    plan = scripts.build_plan(0, 0.05)
    for p in plan:
        segs = _fake_segs(p) if p["voice"] else None
        x, f = audio_fx.assemble(p, segs, np.random.default_rng(p["tone_seed"]))
        d = len(x) / 16000
        assert 1.0 <= d <= 15.01 and np.abs(x).max() <= 0.99 + 1e-6
        if f["beep_present"]:
            assert f["beep_span"][1] <= d + 1e-6 and d - f["beep_span"][1] <= 1.5 + 1e-3
        if f["truncated"]:
            assert d >= f["cue_end_s"]  # the class-defining part is always kept


def test_allocation_balanced_and_capped():
    plan = scripts.build_plan(0, 0.1)
    clips = []
    for p in plan:
        if p["split"] != "train":
            continue
        _, f = audio_fx.assemble(p, _fake_segs(p) if p["voice"] else None, np.random.default_rng(p["tone_seed"]))
        clips.append({"key": p["key"], "cls": p["cls"], "split": "train", "facts": f})
    alloc = build.allocate(clips, 0, 0.1)
    assert max(len(v) for v in alloc.values()) <= build.MAX_ITEMS_PER_CLIP
    per_q = {}
    for v in alloc.values():
        for q, _, y in v:
            if q != "who":
                per_q.setdefault(q, Counter())[y] += 1
    for q, c in per_q.items():
        assert c[True] == c[False] > 0, (q, c)
    items = build.make_items(clips, alloc, 0, common.make_item, {c["key"]: "b" for c in clips},
                             {p["key"]: p for p in plan}, "kokoro")
    rep = build.validate(items, {"b": {c["key"] for c in clips}})
    assert rep["n_items"] == len(items)
    for it in items:
        if it["meta"]["qtype"] == "choice":
            assert it["meta"]["option_classes"][it["gold"]] == it["meta"]["class"]


def test_tones_shapes():
    rng = np.random.default_rng(0)
    x, _ = audio_fx.busy(rng, 3.0)
    assert len(x) == 48000
    s = audio_fx.sit(rng)
    assert 0.8 < len(s) / 16000 < 1.2
    f, v = audio_fx.fax_modem(rng, 5.0)
    assert abs(len(f) - 80000) <= 1 and v in ("fax", "modem")
    # 1 kHz beep really is 1 kHz
    b = audio_fx.beep(rng, 0.5)
    spec = np.abs(np.fft.rfft(b))
    assert abs(np.fft.rfftfreq(len(b), 1 / 16000)[spec.argmax()] - 1000) < 15
    assert random.Random(0).random() == random.Random(0).random()


def test_trim_silence():
    x = np.concatenate([np.zeros(8000), 0.5 * np.sin(np.arange(16000) * 0.2), np.zeros(12000)]).astype(np.float32)
    y = audio_fx.trim_silence(x)
    assert abs(len(y) / 16000 - 1.0) < 0.1
