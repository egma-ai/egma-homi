"""Unit tests for sb_dialog pure logic (labels, layouts, selection).

    uv run --python 3.12 --with pytest --with numpy --with scipy pytest data/sb_dialog/tests
"""
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "inference"))

import build  # noqa: E402
import common  # noqa: E402
import hvb  # noqa: E402
import labels as L  # noqa: E402
import questions as Q  # noqa: E402


def test_talkover_rule_and_margins():
    caller = [(0.0, 5.0)]
    # agent starts at 4.0 while caller has talked 4 s; both talk 1.0 s -> strong
    assert L.talked_over([(4.0, 7.0)], caller, 60.0) == ("yes", "first third")
    # agent starts right after caller ends -> no event
    assert L.talked_over([(5.3, 7.0)], caller, 60.0) == ("no", "it never did")
    # overlap only 0.1 s -> negligible -> no
    assert L.talked_over([(4.9, 7.0)], caller, 60.0)[0] == "no"
    # overlap 0.3 s (between 0.15 and 0.4) -> ambiguous -> not asked
    assert L.talked_over([(4.7, 7.0)], caller, 60.0) == (None, None)
    # simultaneous start (prior 0.05 s) -> not an interruption
    assert L.talked_over([(0.05, 3.0)], caller, 60.0)[0] == "no"
    # event at 20.5 s of a 60 s clip is within 1 s of the 1/3 boundary -> when unknown, yes/no known
    assert L.talked_over([(20.5, 23.0)], [(15.0, 22.0)], 60.0) == ("yes", None)
    assert L.talked_over([(45.0, 47.0)], [(40.0, 46.5)], 60.0) == ("yes", "last third")


def test_silence_and_response_and_share():
    sp = [(0.5, 5.0), (5.5, 11.5), (16.0, 20.0), (20.4, 29.5)]
    f = L.silence_facts(sp, sp, 30.0)
    assert f["long"] == "yes" and f["where"] == "middle third" and abs(f["max_gap"] - 4.5) < 1e-6
    f2 = L.silence_facts([(0.2, 10.0), (11.0, 29.8)], [(0.2, 10.0), (11.0, 29.8)], 30.0)
    assert f2["long"] == "no"
    f3 = L.silence_facts([(0.2, 10.0), (13.0, 29.8)], [(0.2, 10.0), (13.0, 29.8)], 30.0)
    assert f3["long"] is None  # 3.0 s gap is inside the 2.5-3.5 margin
    caller = [(0.0, 2.0), (6.0, 8.0)]
    agent = [(2.5, 5.0), (11.0, 13.0)]
    assert L.response_delays(agent, caller) == [0.5, 3.0]
    assert L.slow_response(agent, caller) == "yes"
    assert L.slow_response([(2.5, 5.0), (8.6, 10.0)], caller) == "no"
    assert L.talk_share([(0, 10)], [(10, 13)]) == "a"
    assert L.talk_share([(0, 10)], [(10, 20)]) == "equal"
    assert L.talk_share([(0, 10)], [(10, 25)]) == "b"
    assert L.talk_share([(0, 10)], [(10, 22.5)]) is None  # share 0.444 is in the margin


def test_multi_party():
    sp = {"x": [(0.5, 10.0)], "y": [(12.0, 20.0)], "z": [(22.0, 29.0)]}
    yn, when, amt = L.any_overlap(sp, 30.0)
    assert (yn, when, amt) == ("no", "never", "not at all")
    assert L.n_speakers(sp) == "three or more"
    assert L.first_last(sp) == ("x", "z")
    sp2 = {"x": [(0.5, 10.0)], "y": [(8.0, 20.0)]}
    assert L.any_overlap(sp2, 30.0)[:2] == ("yes", "first third")  # first overlap starts at 8 s
    assert L.any_overlap({"x": [(0.5, 10.0)], "y": [(9.5, 20.0)]}, 30.0)[:2] == ("yes", None)  # 9.5 s: at boundary
    sp3 = {"x": [(0.5, 25.0)], "y": [(22.0, 29.0)], "z": [(1.0, 1.4)]}
    assert L.n_speakers(sp3) is None  # z has 0.4 s: ambiguous
    assert L.dominant({"x": [(0, 20)], "y": [(20, 25)]}) == "yes"


def _fake_conv(n_turns=14, seed=0):
    rng = random.Random(seed)
    segs, t = [], 1.0
    for i in range(n_turns):
        role = "agent" if i % 2 == 0 else "caller"
        d = rng.uniform(2.0, 6.0)
        words, w = [], t
        while w < t + d - 0.3:
            words.append((w, w + 0.3))
            w += 0.35
        segs.append({"role": role, "s": t, "e": words[-1][1], "words": words, "speech": True, "text": "x"})
        t = words[-1][1] + rng.uniform(0.3, 1.0)
    return segs


def test_relaid_layout_labels_are_consistent():
    segs = _fake_conv()
    turns = hvb.make_turns(segs)
    rng = random.Random(0)
    lay = hvb.relaid_layout(turns, rng, {"start_turn": 0, "target_len": 200, "style": "tight", "agent_over": True,
                                         "caller_over": False, "long_sil": False, "slow": False, "many_over": False})
    assert lay is not None and any(e["type"] == "agent_over" for e in lay["events"])
    lab, _ = build.two_party_labels(lay["speech"], lay["incl"], lay["T"], None, [(0.0, lay["T"])])
    assert lab["a_over"] == "yes" and lab["c_over"] == "no" and lab["over_who"] == "a"
    clean = hvb.relaid_layout(turns, random.Random(1), {"start_turn": 0, "target_len": 200, "style": "tight",
                                                        "agent_over": False, "caller_over": False,
                                                        "long_sil": False, "slow": False, "many_over": False})
    lab2, _ = build.two_party_labels(clean["speech"], clean["incl"], clean["T"], None, [])
    assert lab2["a_over"] == "no" and lab2["c_over"] == "no" and lab2["long_sil"] == "no"
    assert lab2.get("slow_resp") == "no"


def test_render_tracks_places_pieces():
    sr = 8000
    chans = {"agent": np.zeros(sr * 30, np.float32), "caller": np.zeros(sr * 30, np.float32)}
    chans["agent"][sr * 2: sr * 3] = 0.5
    lay = {"T": 10.0, "bed": True, "pieces": [("agent", 2.0, 3.0, 5.0)]}
    beds = {"agent": np.zeros(sr, np.float32), "caller": np.full(sr, 0.001, np.float32)}
    tr = hvb.render_tracks(lay, chans, beds, sr, random.Random(0))
    assert len(tr["agent"]) == 10 * sr
    assert abs(tr["agent"][int(5.5 * sr)] - 0.5) < 1e-6 and tr["agent"][int(4.0 * sr)] == 0
    assert np.allclose(tr["caller"], 0.001)
    ev = hvb.energy_vad(tr["agent"], sr)
    assert len(ev) == 1 and abs(ev[0][0] - 5.0) < 0.05 and abs(ev[0][1] - 6.0) < 0.05


def test_select_items_balances_yes_no_and_caps():
    rng = random.Random(0)
    clips = []
    for i in range(400):
        lab = {"long_sil": "yes" if i % 4 == 0 else "no", "a_over": "yes" if i % 3 == 0 else "no",
               "first_spk": "a" if i % 5 else "b", "n_spk": "two", "ov_amount": "a little" if i % 2 else "a lot"}
        clips.append({"key": f"k{i}", "source": "hvb", "split": "train", "labels": lab})
    picked = build.select_items(clips, rng)
    per_clip = Counter(c["key"] for c, _, _ in picked)
    assert max(per_clip.values()) <= 4
    cnt = Counter((f, lab) for _, f, lab in picked)
    assert cnt[("long_sil", "yes")] == cnt[("long_sil", "no")] > 0
    assert cnt[("a_over", "yes")] == cnt[("a_over", "no")] > 0
    assert ("n_spk", "two") not in cnt  # a single-label choice family is never asked


def test_make_items_schema():
    clips = [{"key": "hvb_x_v0", "source": "hvb", "split": "train", "labels": {}, "bank": "b", "dur": 40.0,
              "licence": "CC BY 4.0", "source_id": "x", "speakers": ["hvb_spk:1"], "kind": "relaid", "events": []}]
    picked = [(clips[0], "agent_share", "a"), (clips[0], "a_over", "yes"), (clips[0], "a_over_when", "middle third")]
    items = build.make_items(picked, random.Random(0), common)
    sc = items[0]
    assert sc["meta"]["qtype"] == "score" and sc["options"][2].startswith("more than") and sc["gold"] == 2
    yn = items[1]
    assert sorted(yn["options"]) == ["no", "yes"] and yn["options"][yn["gold"]] == "yes"
    wh = items[2]
    assert wh["options"][wh["gold"]] == "middle third" and len(wh["options"]) == 4
    for it in items:
        assert abs(sum(it["target"]) - 1) < 1e-9 and it["gold"] == int(np.argmax(it["target"]))


def test_paraphrase_counts():
    for fam, ts in Q.T.items():
        assert len(ts) >= 5, fam
    assert set(Q.QTYPE) <= set(Q.T) | {"c_over", "c_over_when"}
    rng = random.Random(0)
    for fam in Q.QTYPE:
        lab = {"noul": "yes", "score": None}.get(Q.QTYPE[fam])
        lab = lab or {"a_over_when": "first third", "c_over_when": "it never did", "over_who": "both",
                      "more_talk": "equal", "first_spk": "a", "last_spk": "b", "sil_where": "last third",
                      "mp_overlap_when": "never", "n_spk": "two", "agent_share": "b", "ov_amount": "a lot"}[fam]
        q, opts, gold, qt = Q.build_question(fam, lab, ("the agent", "the caller"), "call", rng)
        assert qt == Q.QTYPE[fam] and 0 <= gold < len(opts) and "{" not in q
    q, *_ = Q.build_question("c_over", "yes", ("the agent", "the caller"), "call", random.Random(3))
    assert q.lower().index("the caller") < q.lower().index("the agent")


def test_parsers():
    xml = ('<w nite:id="a" starttime="1.0" endtime="1.5">Hi</w><w nite:id="b" starttime="1.5" endtime="1.5" '
           'punc="true">,</w><vocalsound nite:id="c" starttime="3.0" endtime="4.0" type="laugh"/>'
           '<w nite:id="d" starttime="1.52" endtime="2.0">there</w>')
    sp, inc = build.parse_ami_words(xml)
    assert sp == [(1.0, 2.0)] and inc == [(1.0, 2.0), (3.0, 4.0)]
    r = build.parse_rttm("SPEAKER f 1 3.0 2.0 <NA> <NA> 0 <NA> <NA>\nSPEAKER f 1 6.0 1.0 <NA> <NA> 1 <NA> <NA>\n")
    assert r == {"0": [(3.0, 5.0)], "1": [(6.0, 7.0)]}
    wins = build.plan_windows({"0": [(i * 3.0, i * 3.0 + 2.5) for i in range(400)]}, 1200.0, random.Random(0))
    assert wins and all(20 <= b - a <= 180 for a, b in wins)
    assert all(wins[i][1] <= wins[i + 1][0] for i in range(len(wins) - 1))


def test_hvb_speaker_split_disjoint():
    rng = random.Random(0)
    conv = {f"c{i}": {f"s{rng.randrange(12)}", f"s{rng.randrange(12, 30)}"} for i in range(300)}
    split, S = build.hvb_speaker_split(conv, 0.03, random.Random(1))
    cal_spk = {s for c, v in conv.items() if split[c] == "CAL" for s in v}
    tr_spk = {s for c, v in conv.items() if split[c] == "train" for s in v}
    assert cal_spk and not (cal_spk & tr_spk)
