"""Unit tests for the pure label / conversion logic of the sb_sounds builder."""
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "inference"))

import labels as L  # noqa: E402

# a tiny ontology: Animal > Dog > Bark, Animal > Cat > Meow; Music > Guitar; Vehicle > Car
ONT = [
    {"id": "/a", "name": "Animal", "child_ids": ["/x"]},
    {"id": "/x", "name": "Domestic (not in vocab)", "child_ids": ["/dog", "/cat"]},
    {"id": "/dog", "name": "Dog", "child_ids": ["/bark"]},
    {"id": "/bark", "name": "Bark", "child_ids": []},
    {"id": "/cat", "name": "Cat", "child_ids": ["/meow"]},
    {"id": "/meow", "name": "Meow", "child_ids": []},
    {"id": "/m", "name": "Music", "child_ids": ["/g"]},
    {"id": "/g", "name": "Guitar", "child_ids": []},
    {"id": "/v", "name": "Vehicle", "child_ids": ["/car"]},
    {"id": "/car", "name": "Car", "child_ids": []},
]
VOCAB = {"/a": "Animal", "/dog": "Dog", "/bark": "Bark", "/cat": "Cat", "/meow": "Meow", "/m": "Music",
         "/g": "Guitar", "/v": "Vehicle", "/car": "Car"}


def tax():
    return L.Taxonomy(VOCAB, ONT)


def test_taxonomy_walks_through_non_vocab_nodes():
    t = tax()
    assert t.parents["Dog"] == {"Animal"}  # via the non-vocabulary node
    assert t.ancestors["Bark"] == {"Dog", "Animal"}
    assert t.descendants["Animal"] == {"Dog", "Bark", "Cat", "Meow"}
    assert t.siblings("Dog") == {"Cat"}
    assert t.leaves(["Bark", "Dog", "Animal", "Music", "Guitar"]) == ["Bark", "Guitar"]


def test_absent_class_never_related_or_confusable():
    t = tax()
    cooc = L.cooccurrence([["Bark", "Dog", "Animal"], ["Car", "Vehicle"], ["Guitar", "Music"]])
    labels = ["Bark", "Dog", "Animal"]
    rng = random.Random(0)
    for _ in range(50):
        for hard in (True, False):
            x = L.pick_absent(labels, t, cooc, rng, hard)
            assert x is not None
            assert x not in labels and not any(t.related(x, l) for l in labels)
            assert x not in L.group_blocked(labels)  # Cat/Meow share a confusion group with Dog
    assert not L.is_valid_absent("Animal", ["Bark", "Dog"], t, cooc)  # ancestor
    assert not L.is_valid_absent("Meow", ["Dog"], t, cooc)  # same confusion group
    assert L.is_valid_absent("Guitar", ["Dog"], t, cooc)


def test_cooccurrence_blocks_frequent_companions():
    t = tax()
    cooc = L.cooccurrence([["Car", "Guitar"]] * 5 + [["Car"]] * 5)
    assert abs(cooc["Car"]["Guitar"] - 0.5) < 1e-9
    assert not L.is_valid_absent("Guitar", ["Car"], t, cooc)


def test_clip_category_uses_leaves_and_rejects_mixed():
    t = tax()
    assert L.clip_category(["Bark", "Dog", "Animal"], t) == L.AN
    assert L.clip_category(["Bark", "Dog", "Animal", "Car", "Vehicle"], t) is None


def test_scene_answers():
    assert L.speech_answer(["Speech", "Human_voice"]) == "yes"
    assert L.speech_answer(["Laughter", "Human_voice"]) is None  # voice but not clearly speech
    assert L.speech_answer(["Bark", "Dog"]) == "no"
    L.MUSIC_SUBTREE.clear()
    L.MUSIC_SUBTREE.update({"Music", "Musical_instrument", "Guitar", "Bell", "Church_bell"})
    assert L.music_answer(["Guitar", "Musical_instrument", "Music"]) == "yes"
    assert L.music_answer(["Church_bell", "Bell", "Musical_instrument", "Music"]) is None  # bells only
    assert L.music_answer(["Bark", "Dog"]) == "no"


def test_pitch_register_has_gaps():
    assert L.pitch_register(40) == "low"
    assert L.pitch_register(48) is None
    assert L.pitch_register(60) == "middle"
    assert L.pitch_register(72) is None
    assert L.pitch_register(84) == "high"


def test_signal_rules():
    sr = 16000
    rng = np.random.default_rng(0)
    tone = (0.1 * np.sin(2 * np.pi * 440 * np.arange(2 * sr) / sr)).astype(np.float32)
    assert abs(L.active_rms_db(tone) - 20 * np.log10(0.1 / np.sqrt(2))) < 0.5
    assert L.active_until_end(tone)
    padded = np.concatenate([tone, 1e-4 * rng.standard_normal(2 * sr).astype(np.float32)])
    assert not L.active_until_end(padded)
    assert "loud" in L.feasible_levels(tone)  # crest factor of a sine is low
    spike = np.zeros(sr, np.float32)
    spike[100] = 0.9
    spike += 1e-3 * rng.standard_normal(sr).astype(np.float32)
    assert "loud" not in L.feasible_levels(spike)  # would clip


def test_question_bank_sizes_and_class_table():
    for qs in (L.Q_PRESENCE, L.Q_IDENT, L.Q_CATEGORY, L.Q_MUSIC, L.Q_SPEECH, L.Q_COUNT, L.Q_LEVEL, L.Q_TRAIL,
               L.Q_NS_FAMILY, L.Q_NS_SOURCE, L.Q_NS_REGISTER, *L.Q_NS_QUALITY.values()):
        assert len(qs) >= 5 and len(set(qs)) == len(qs)
    assert len(L.CLASSES) == 200
    for name, (opt, np_, cat, pres, ident) in L.CLASSES.items():
        assert opt and np_ and (cat is None or cat in L.CATEGORIES)
        assert "_" not in opt


def test_presence_question_number_agreement():
    assert L.presence_question("Is there {np} in this recording?", "Cheering") == "Are there people cheering in this recording?"
    assert L.presence_question("Listen carefully: is there {np} in this audio?", "Coin_(dropping)") == \
        "Listen carefully: are there coins dropping in this audio?"
    assert L.presence_question("Is there {np} in this recording?", "Bark") == "Is there a dog barking in this recording?"
    assert all(c in L.CLASSES for c in L.PLURAL)
