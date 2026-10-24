import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "inference"))

import build  # noqa: E402
import lexicon as L  # noqa: E402

CMU = L.parse_cmudict("""banana  B AH0 N AE1 N AH0
record  R EH1 K ER0 D
record(2)  R IH0 K AO1 R D
every  EH1 V ER0 IY0
every(2)  EH1 V R IY0
ship  SH IH1 P
sheep  SH IY1 P
cat  K AE1 T
""")


def test_cmudict_syllables_and_stress():
    assert L.syllables(CMU["banana"][0]) == 3
    assert L.primary_index(CMU["banana"][0]) == 1
    assert L.consistent(CMU["every"], L.syllables) is None  # 3 vs 2 -> unusable
    assert L.consistent(CMU["record"], L.primary_index) is None  # noun/verb disagree
    assert L.coda(CMU["cat"][0]) == ("T",) and L.onset(CMU["ship"][0]) == ("SH",)


def test_misaki_nuclei():
    assert L.misaki_syllables("bənˈænə") == 3
    assert L.misaki_primary("bənˈænə") == 1
    assert L.misaki_primary("ɹˈɛkəɹd") == 0 and L.misaki_primary("ɹəkˈɔɹd") == 1
    assert L.misaki_syllables("Idˈiə") == 3
    assert L.misaki_syllables("nˈɪə", gb=True) == 1
    assert L.misaki_primary("kˈɑmpˌækt") == 0


def test_minimal_groups():
    com = {"ship": ("SH", "IH", "P"), "sheep": ("SH", "IY", "P"), "cat": ("K", "AE", "T")}
    g = build.minimal_groups(com)
    assert len(g) == 1 and g[0]["words"] == ["sheep", "ship"] and g[0]["is_vowel"]


def test_disfluent_text_types():
    rng = random.Random(3)
    for _ in range(200):
        text, info = build.make_disfluent(rng)
        low = text.lower()
        if "filled" in info["types"]:
            assert low.count("um,") + low.count("uh,") == info["n_fill"] >= 1
        else:
            assert info["n_fill"] == 0 and " um," not in low and " uh," not in low
        if "restart" in info["types"]:
            assert "—" in text
        if not info["types"]:
            assert text.rstrip(".").split() == [w for w in info["words"]] or text[0].isupper()


def test_balance_noul_exact():
    items = []
    for i in range(30):
        yes = i < 20
        items.append({"id": f"x{i}", "set": "train", "meta": {"qname": "q", "qtype": "noul",
                                                               "answer_class": "yes" if yes else "no"}})
    out = build.balance(items, 0)
    assert sum(it["meta"]["answer_class"] == "yes" for it in out) == 10 == sum(it["meta"]["answer_class"] == "no" for it in out)
