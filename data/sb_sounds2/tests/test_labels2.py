import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fsd_labels as FL  # noqa: E402
import labels2 as L2  # noqa: E402

# toy AudioSet-like ontology with real FSD50K class names
VOCAB = {"/m/a": "Animal", "/m/d": "Domestic_animals_and_pets", "/m/dog": "Dog", "/m/bark": "Bark",
         "/m/growl": "Growling", "/m/cat": "Cat", "/m/meow": "Meow", "/m/purr": "Purr"}
NODES = [{"id": "/m/a", "name": "Animal", "child_ids": ["/m/d"]},
         {"id": "/m/d", "name": "Domestic animals, pets", "child_ids": ["/m/dog", "/m/cat"]},
         {"id": "/m/dog", "name": "Dog", "child_ids": ["/m/bark", "/m/growl"]},
         {"id": "/m/cat", "name": "Cat", "child_ids": ["/m/meow", "/m/purr"]},
         {"id": "/m/bark", "name": "Bark", "child_ids": []}, {"id": "/m/growl", "name": "Growling", "child_ids": []},
         {"id": "/m/meow", "name": "Meow", "child_ids": []}, {"id": "/m/purr", "name": "Purr", "child_ids": []}]


def toy():
    return FL.Taxonomy(VOCAB, NODES), L2.FullOntology(VOCAB, NODES)


def test_semantic_block_names_are_fsd_classes():
    names = set(FL.CLASSES)
    assert not ({c for g in L2.SEMANTIC_BLOCK for c in g} - names)


def test_sibling_of_leaf_is_a_valid_hard_negative_but_not_descendant_of_leaf():
    tax, onto = toy()
    labels = ["Bark", "Dog", "Domestic_animals_and_pets", "Animal"]
    assert L2.valid_negative("Growling", labels, ["Bark"], tax, {})          # sibling under smeared parent "Dog"
    assert not L2.valid_negative("Dog", labels, ["Bark"], tax, {})           # a label
    assert L2.valid_negative("Meow", labels, ["Bark"], tax, {})
    # clip labelled only "Dog": its descendants are ambiguous -> not allowed
    assert not L2.valid_negative("Growling", ["Dog", "Domestic_animals_and_pets", "Animal"], ["Dog"], tax, {})
    # co-occurrence with a leaf blocks it
    assert not L2.valid_negative("Growling", labels, ["Bark"], tax, {"Bark": {"Growling": 0.1}})
    near = L2.near_negatives(labels, ["Bark"], tax, onto, {})
    assert near.get("Growling") == "sibling"
    assert onto.official["Domestic_animals_and_pets"] == "Domestic animals, pets"


def test_pick_negative_near_vs_random():
    tax, onto = toy()
    labels = ["Bark", "Dog", "Domestic_animals_and_pets", "Animal"]
    near = L2.near_negatives(labels, ["Bark"], tax, onto, {})
    c, kind = L2.pick_negative(labels, ["Bark"], tax, onto, {}, random.Random(0), want_near=True)
    assert c in near and kind == near[c] and kind in ("sibling", "confusable", "broad_sibling")
    c, kind = L2.pick_negative(labels, ["Bark"], tax, onto, {}, random.Random(0), want_near=False)
    assert (c is None and kind == "none") or (kind == "random" and c not in near)


def test_licence_classes():
    assert L2.licence_class("http://creativecommons.org/licenses/by/4.0/") == "CC-BY"
    assert L2.licence_class("http://creativecommons.org/licenses/by-sa/3.0/us/") == "CC-BY-SA"
    assert L2.licence_class("http://creativecommons.org/licenses/by-nc-sa/3.0/") is None
    assert L2.licence_class("http://creativecommons.org/licenses/by-nd/4.0/") is None
    assert L2.licence_class("http://creativecommons.org/publicdomain/zero/1.0/") == "CC0"
    assert L2.licence_class("http://freemusicarchive.org/Sound_Recording_Common_Law") is None
    assert L2.mirror_licence_class("CC-BY 4.0") == "CC-BY"
    assert L2.mirror_licence_class("CC-BY-SA 3.0") == "CC-BY-SA"
    assert L2.mirror_licence_class("CC-BY-NC-SA 3.0") is None
    assert L2.mirror_licence_class("CC-Sampling+ 1.0") is None


def test_sdd_artist_parse():
    txt = ("26/226.mp3\nJLP vs momomatic by David TMX from Jamendo: http://www.jamendo.com/track/226\n"
           "Available under ...: http://creativecommons.org/licenses/by-nc-sa/2.0/\n--\n")
    arts, ids = L2.parse_sdd_artists(txt)
    assert arts == {"davidtmx"} and ids == {"226"}


ROWS = [{"genre_id": "12", "parent": "0", "title": "Rock"}, {"genre_id": "25", "parent": "12", "title": "Punk"},
        {"genre_id": "109", "parent": "25", "title": "Hardcore"}, {"genre_id": "26", "parent": "12", "title": "Post-Rock"},
        {"genre_id": "31", "parent": "12", "title": "Metal"}, {"genre_id": "4", "parent": "0", "title": "Jazz"},
        {"genre_id": "10", "parent": "0", "title": "Pop"}]


def test_genre_negatives():
    tree = L2.GenreTree(ROWS)
    track = {12, 25}  # Punk (+ ancestor Rock)
    assert not L2.genre_valid_negative(109, track, tree)   # descendant of a track genre
    assert not L2.genre_valid_negative(12, track, tree)    # own genre
    assert L2.genre_valid_negative(26, track, tree)        # sibling Post-Rock
    assert L2.genre_valid_negative(4, track, tree)         # other root
    assert not L2.genre_valid_negative(10, track, tree)    # Pop blocked with Rock (overlap)
    assert 26 in L2.genre_near(track, tree)


def click_track(bpm, seconds=12, sr=16000):
    x = np.zeros(int(seconds * sr), np.float32)
    step = int(round(60 / bpm * sr))
    rng = np.random.default_rng(0)
    for s in range(0, len(x) - 400, step):
        x[s: s + 400] += rng.standard_normal(400).astype(np.float32) * np.hanning(400).astype(np.float32)
    return x * 0.3


def test_tempo_classes_on_click_tracks():
    for bpm, want in ((70, "slow"), (110, "medium")):
        t = L2.estimate_tempo(click_track(bpm))
        assert abs(t["bpm"] - bpm) / bpm < 0.04, (bpm, t)
        assert L2.tempo_class(t) == want, (bpm, t)
    # a strictly periodic 160 BPM click is octave-ambiguous (80 vs 160): it must be skipped, not mislabelled
    t = L2.estimate_tempo(click_track(160))
    assert L2.tempo_class(t) in ("fast", None)
    assert L2.tempo_class({"bpm": 150.0, "ambiguity": 0.3, "halves": [151.0, 149.0], "peak": 3}) == "fast"
    assert L2.tempo_class({"bpm": 90.0, "ambiguity": 0.1, "halves": [90.0, 90.0], "peak": 3}) is None  # gap
    assert L2.tempo_class({"bpm": 110.0, "ambiguity": 0.9, "halves": [110.0, 110.0], "peak": 3}) is None


def test_instrument_thresholds():
    pos, neg = L2.instrument_labels([("piano", 0.9), ("organ", 0.1), ("violin", 0.5), ("kazoo", 1.0)])
    assert pos == {"piano"} and neg == {"organ"}
    assert "organ" in L2.instr_near("piano")


def test_question_paraphrases():
    for q in (L2.Q_PRES_NAME, L2.Q_WHICH, L2.Q_GENRE_TOP, L2.Q_GENRE_SUB, L2.Q_GENRE_NOUL, L2.Q_INSTR, L2.Q_INSTR_WHICH,
              L2.Q_VOCALS, L2.Q_INSTRUMENTAL, L2.Q_USE_BED, L2.Q_TEMPO, L2.Q_LEVEL, L2.Q_LIVE):
        assert len(set(q)) >= 5
    adi = 'Does this recording contain the sound "{name}"?'
    assert adi not in L2.Q_PRES_NAME and "Which sound is most prominent in this recording?" not in L2.Q_WHICH
    assert all(g in L2.GENRE_DISPLAY for g in L2.TOP_GENRES)
