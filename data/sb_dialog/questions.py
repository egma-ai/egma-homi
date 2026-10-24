"""sb_dialog question families: paraphrases, options and item construction.

Two-party (HVB) and multi-party (AMI, Earnings-21) family keys are listed in README.md.
"""
from __future__ import annotations

import random

import numpy as np

QTYPE = {
    "a_over": "noul", "c_over": "noul", "long_sil": "noul", "slow_resp": "noul", "mp_overlap": "noul",
    "same_first_last": "noul", "dominant": "noul",
    "a_over_when": "choice", "c_over_when": "choice", "over_who": "choice", "more_talk": "choice",
    "first_spk": "choice", "last_spk": "choice", "sil_where": "choice", "mp_overlap_when": "choice",
    "n_spk": "choice",
    "agent_share": "score", "ov_amount": "score",
}

# role wordings for two-party calls: (agent term, caller term, short agent option, short caller option)
ROLE_SETS_BANK = [("the agent", "the caller"), ("the agent", "the customer"), ("the bank agent", "the customer"),
                  ("the representative", "the caller"), ("the bank representative", "the customer")]
ROLE_SETS_AI = [("the AI receptionist", "the caller"), ("the AI agent", "the caller"),
                ("the receptionist", "the caller"), ("the voice agent", "the customer")]

T = {
    "a_over": [
        "Did {A} talk over {C} at any point?",
        "Did {A} interrupt {C} during this {X}?",
        "Does {A} ever start talking while {C} is still speaking?",
        "Is there a moment where {A} cuts {C} off?",
        "Did {A} speak over {C}?",
        "At any point, does {A} begin speaking before {C} has finished?",
    ],
    "a_over_when": [
        "When did {A} first talk over {C}?",
        "When does {A} first interrupt {C}?",
        "At what point in the {X} did {A} first start speaking while {C} was still talking?",
        "In which part of the {X} does {A} first cut {C} off?",
        "Where in the {X} is the first time {A} talks over {C}?",
    ],
    "over_who": [
        "Who talked over whom in this {X}?",
        "Which best describes the interruptions in this {X}?",
        "Did anyone talk over the other person? Pick the best description.",
        "Which of these is true about the two sides talking over each other?",
        "How did the two speakers interrupt each other, if at all?",
    ],
    "more_talk": [
        "Who talked more in this {X}?",
        "Who did most of the talking?",
        "Which side spoke for longer overall?",
        "Who held the floor more in this conversation?",
        "Overall, who spoke more?",
    ],
    "agent_share": [
        "How much did {A} talk compared with {C}?",
        "Compared with {C}, how much talking did {A} do?",
        "Rate how much {A} spoke relative to {C}.",
        "Relative to {C}, how much of the conversation was {A} speaking?",
        "How does {A}'s amount of talking compare with {C}'s?",
    ],
    "long_sil": [
        "Was there a long silence (over 3 seconds) in the {X}?",
        "Is there a stretch of more than three seconds where nobody speaks?",
        "Does the {X} contain a pause longer than 3 seconds with no one talking?",
        "Was there any dead air longer than three seconds?",
        "At some point, does everyone stay silent for more than 3 seconds?",
    ],
    "sil_where": [
        "In which part of the {X} does the longest silence happen?",
        "Where is the longest pause in this {X}?",
        "When does the longest stretch of silence occur?",
        "In which third of the {X} is the longest gap with nobody talking?",
        "Where in the {X} is the longest dead air?",
    ],
    "slow_resp": [
        "Did {A} respond slowly at some point (a pause of over 2 seconds before answering)?",
        "Was there a time when {A} took more than two seconds to reply after {C} finished?",
        "Did {C} ever have to wait more than 2 seconds for {A} to answer?",
        "Is there at least one delay of over two seconds before {A} responds to {C}?",
        "Did {A} ever leave a long pause (over 2 s) before replying?",
    ],
    "first_spk": [
        "Who spoke first?",
        "Who is the first to speak in this {X}?",
        "Who talks first in this clip?",
        "Which side starts speaking first?",
        "Who opens this part of the conversation?",
    ],
    "last_spk": [
        "Who spoke last?",
        "Who has the last word in this clip?",
        "Who is the last person to speak?",
        "Which side speaks last in this {X}?",
        "Who talks last before the clip ends?",
    ],
    "ov_amount": [
        "How much do the speakers talk over each other?",
        "How much overlapping speech is there in this {X}?",
        "How often do people speak at the same time in this {X}?",
        "Rate the amount of cross-talk in this {X}.",
        "How much simultaneous talking is there?",
    ],
    "mp_overlap": [
        "Do two people speak at the same time at any point in this {X}?",
        "Is there any overlapping speech, with two people talking at once?",
        "Do any two speakers talk at the same time?",
        "At some point, are two voices heard speaking simultaneously?",
        "Is there cross-talk (people speaking at the same time) in this {X}?",
    ],
    "mp_overlap_when": [
        "When do two people first speak at the same time?",
        "In which part of the {X} is there overlapping speech for the first time?",
        "When is the first moment that two speakers talk at once?",
        "Where in the {X} does cross-talk first happen?",
        "At what point do two voices first overlap?",
    ],
    "n_spk": [
        "How many different people speak in this {X}?",
        "How many speakers can be heard?",
        "How many distinct voices talk in this clip?",
        "How many people take part in this stretch of the conversation?",
        "Count the speakers: how many different people talk?",
    ],
    "same_first_last": [
        "Is the person who speaks last the same person who spoke first?",
        "Does the clip end with the same speaker it started with?",
        "Is the first voice in the {X} also the last one heard?",
        "Does the same person both open and close this clip?",
        "Is the last speaker the same as the first speaker?",
    ],
    "dominant": [
        "Does one person do more than half of the talking?",
        "Does a single speaker account for most of the speech in this {X}?",
        "Is there one person who talks more than everyone else combined?",
        "Does one speaker hold the floor for most of this {X}?",
        "Is most of the talking done by just one person?",
    ],
}

WHEN_OPTS = ["it never did", "first third", "middle third", "last third"]
MP_WHEN_OPTS = ["never", "first third", "middle third", "last third"]
THIRD_OPTS = ["first third", "middle third", "last third"]
N_SPK_OPTS = ["one", "two", "three or more"]
OV_LEVELS = ["not at all", "a little", "a lot"]


class NoShuffle(random.Random):
    def shuffle(self, x, *a, **k):  # keep score levels in their natural (low -> high) order
        return None


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def _short(term: str) -> str:
    return term  # options keep the article ("the agent"), short and readable


def build_question(family: str, label: str, roles: tuple | None, ctx: str, rng: random.Random):
    """-> (question, options, gold index, qtype). `label` is the canonical label value from labels.py
    (two-party direction-free labels use 'a' = agent side, 'b' = caller side)."""
    A, C = roles if roles else ("", "")
    tfam = family
    if family in ("c_over", "c_over_when"):  # same templates with the roles swapped
        tfam = "a_over" if family == "c_over" else "a_over_when"
        A, C = C, A
    q = rng.choice(T[tfam])
    q = q.replace("{A}", A).replace("{C}", C).replace("{X}", ctx)
    q = _cap(q)
    qt = QTYPE[family]
    if qt == "noul":
        opts = ["yes", "no"]
        gold = 0 if label == "yes" else 1
    elif family in ("a_over_when", "c_over_when"):
        opts, gold = WHEN_OPTS, WHEN_OPTS.index(label)
    elif family == "mp_overlap_when":
        opts, gold = MP_WHEN_OPTS, MP_WHEN_OPTS.index(label)
    elif family == "sil_where":
        opts, gold = THIRD_OPTS, THIRD_OPTS.index(label)
    elif family == "n_spk":
        opts, gold = N_SPK_OPTS, N_SPK_OPTS.index(label)
    elif family == "ov_amount":
        opts, gold = OV_LEVELS, OV_LEVELS.index(label)
    elif family in ("first_spk", "last_spk"):
        opts = [_short(A), _short(C)]
        gold = 0 if label == "a" else 1
    elif family == "more_talk":
        opts = [_short(A), _short(C), "about the same"]
        gold = {"a": 0, "b": 1, "equal": 2}[label]
    elif family == "agent_share":
        opts = [f"less than {C}", f"about the same as {C}", f"more than {C}"]
        gold = {"b": 0, "equal": 1, "a": 2}[label]
    elif family == "over_who":
        opts = [f"only {A} talked over {C}", f"only {C} talked over {A}", "both talked over each other",
                "neither talked over the other"]
        gold = {"a": 0, "b": 1, "both": 2, "neither": 3}[label]
    else:
        raise KeyError(family)
    return q, list(opts), gold, qt


def one_hot(k: int, gold: int) -> np.ndarray:
    t = np.zeros(k)
    t[gold] = 1.0
    return t
