"""sb_aicall question families, paraphrases, options and role words.

Placeholders: {A} = agent side, {O} = other side, {N} = threshold ("3 seconds")."""
from __future__ import annotations

import random

QTYPE = {
    "live_in": "noul", "live_out": "noul", "waited": "noul", "silence": "noul", "barge": "noul",
    "barge_stop": "noul", "aiover": "noul", "vm_left": "noul", "tone": "noul", "auto_in": "noul", "auto_out": "noul",
    "live_choice": "choice", "aiover_when": "choice", "more_talk": "choice", "first": "choice", "last": "choice",
    "wait_bin": "score_or_choice", "sil_bin": "score_or_choice", "share": "score_or_choice",
    "barge_count": "score_or_choice",
}
FAMILIES = list(QTYPE)
NOUL = {f for f, q in QTYPE.items() if q == "noul"}

# role words: (agent phrase, short agent option), (other phrase, short other option)
AGENT_IN = ["the AI receptionist", "the AI agent", "the voice agent", "the assistant", "the bot",
            "the AI receptionist", "the AI"]
AGENT_OUT = ["the AI agent", "the voice agent", "the assistant", "the bot", "the AI"]
AGENT_HUMAN = ["the agent", "the representative", "the bank representative", "the bank agent",
               "the customer service agent"]
OTHER_IN = ["the caller", "the customer", "the person calling", "the other party", "the caller"]
OTHER_OUT = ["the person who answered", "the customer", "the other party", "the person on the line"]
OTHER_E = ["the caller", "the customer"]


def roles_for(scenario: str, direction: str, rng: random.Random):
    if scenario == "E":
        return rng.choice(AGENT_HUMAN), rng.choice(OTHER_E)
    if direction == "out":
        return rng.choice(AGENT_OUT), rng.choice(OTHER_OUT)
    return rng.choice(AGENT_IN), rng.choice(OTHER_IN)


T = {
    "waited": [
        "Did {O} ever wait more than {N} for {A} to respond?",
        "After {O} finished speaking, was there ever a pause of more than {N} before {A} answered?",
        "Was {A} ever slower than {N} to reply after {O} stopped talking?",
        "Is there a point where {O} stops talking and {A} takes longer than {N} to start its reply?",
        "Did any of {A}'s responses come more than {N} after {O} had finished?",
        "Was there a gap of over {N} between the end of something {O} said and {A}'s answer?",
        "Did {O} have to wait longer than {N} for a reply from {A} at any point?",
    ],
    "wait_bin": [
        "How long was the longest wait for {A}'s reply after {O} finished speaking?",
        "What was {A}'s slowest response time in this call?",
        "When {O} stopped talking, what was the longest {A} took to respond?",
        "Roughly how long was {A}'s longest delay before answering {O}?",
        "What is the longest pause between {O} finishing and {A} replying?",
        "How slow was {A}'s slowest reply to {O}?",
    ],
    "silence": [
        "Is there a silence longer than {N} where nobody speaks?",
        "Does the call contain more than {N} of dead air at any point?",
        "At any moment, do both sides stay silent for over {N}?",
        "Is there a stretch of more than {N} with no one talking?",
        "Does the line go quiet for longer than {N} at some point in this call?",
        "Is there any pause longer than {N} in which neither {A} nor {O} says anything?",
    ],
    "sil_bin": [
        "How long was the longest silence in this call?",
        "What is the longest stretch of dead air in this call?",
        "How long is the longest pause where nobody speaks?",
        "Roughly how long did the longest silence last?",
        "What best describes the longest gap with no one talking?",
        "How long was the call's longest quiet moment?",
    ],
    "barge": [
        "Did {O} cut in while {A} was still talking (ignoring short sounds like 'mm-hm')?",
        "Did {O} interrupt {A} at any point?",
        "Does {O} ever start speaking over {A} before {A} has finished?",
        "Is there a barge-in, where {O} starts talking in the middle of what {A} is saying?",
        "Did {O} talk over {A}? Brief acknowledgements like 'okay' or 'mm-hm' don't count.",
        "At some point, does {O} begin a sentence while {A} is still speaking?",
    ],
    "barge_stop": [
        "When {O} interrupted, did {A} stop talking?",
        "Did {A} yield and go quiet when {O} cut in?",
        "After {O} started talking over {A}, did {A} stop speaking right away?",
        "Did {A} let {O} take over when interrupted, rather than keep talking?",
        "When {O} barged in, did {A} cut its own sentence short?",
        "Did {A} stop talking once {O} interrupted?",
    ],
    "barge_count": [
        "How many times did {O} interrupt {A}?",
        "How often did {O} cut in while {A} was speaking?",
        "How many times does {O} start talking over {A}?",
        "Count the times {O} barged in on {A}. How many were there?",
        "How many interruptions of {A} by {O} are there in this call?",
        "How frequently did {O} talk over {A}?",
    ],
    "aiover": [
        "Did {A} talk over {O} at any point?",
        "Did {A} start speaking while {O} was still talking?",
        "Does {A} ever interrupt {O}?",
        "Is there a moment where {A} cuts {O} off mid-sentence?",
        "Did {A} jump in before {O} had finished speaking?",
        "At any point, does {A} begin talking over {O}?",
    ],
    "aiover_when": [
        "When did {A} first talk over {O}?",
        "At what point in the call did {A} first interrupt {O}?",
        "In which part of the call does {A} first start speaking while {O} is still talking?",
        "Where in the call is the first time {A} cuts {O} off?",
        "When does {A} first speak over {O}, if ever?",
        "Which part of the call contains {A}'s first interruption of {O}?",
    ],
    "share": [
        "How much of the talking in this call was done by {O}?",
        "What share of the speaking time belongs to {O}?",
        "Roughly what fraction of the conversation is {O} talking?",
        "Compared with {A}, how much of the call's speech comes from {O}?",
        "What portion of all the talking did {O} do?",
        "How large is {O}'s share of the talking time?",
    ],
    "more_talk": [
        "Who talked more in this call?",
        "Who did most of the talking?",
        "Which side spoke for longer overall?",
        "Did {A} or {O} talk more?",
        "Overall, who spoke more?",
        "Who held the floor for more of the call?",
    ],
    "first": [
        "Who spoke first in this call?",
        "Who is heard first?",
        "Who starts the conversation?",
        "Which side says something first?",
        "Who opens the call?",
        "Who was the first to talk?",
    ],
    "last": [
        "Who has the last word in this recording?",
        "Who is the last to speak?",
        "Who says the final words?",
        "Which side speaks last before the recording ends?",
        "Who talks last?",
        "Who is heard last in this call?",
    ],
    "live_in": [
        "Is the caller a real person?",
        "Is a live human calling, rather than a recording or a machine?",
        "Is there a live person on the other end of the line?",
        "Was this call placed by a real human being?",
        "Is the person calling actually a live human (not automated)?",
        "Is the caller a live human rather than a recording?",
        "Is {O} a real, live person?",
    ],
    "live_out": [
        "Did the AI reach a live human?",
        "Did a real person pick up the call?",
        "Is there a live person on the other end of the line?",
        "Did {A} get through to an actual human?",
        "Was the call answered by a live person?",
        "Is a real human on the line with {A}?",
    ],
    "live_choice": [
        "Who or what was on the other end of the line?",
        "What best describes the other side of this call?",
        "What is on the other end of the line with {A}?",
        "Who or what is {A} talking to in this call?",
        "What kind of party is on the other side of this call?",
        "Which best describes who or what {A} is connected to?",
    ],
    "vm_left": [
        "Did the AI leave a voicemail message?",
        "Did {A} leave a message after the greeting?",
        "Does {A} record a message on the voicemail?",
        "After reaching voicemail, did {A} say a message?",
        "Was a message left on the voicemail by {A}?",
        "Did {A} speak a message for the recipient after the voicemail greeting?",
    ],
    "tone": [
        "Did the voicemail tone play?",
        "Is there a beep after the voicemail greeting?",
        "Can you hear the voicemail beep?",
        "Did the answering machine play its tone?",
        "Does a beep sound after the greeting ends?",
        "Was the record tone (beep) heard?",
    ],
    "auto_in": [
        "Is the caller an automated system?",
        "Is the caller a robocall or a recorded message?",
        "Is this call coming from a machine rather than a person?",
        "Is the caller an automated or prerecorded voice?",
        "Did an automated system place this call?",
        "Is the voice calling in an automated recording?",
    ],
    "auto_out": [
        "Did the AI reach an automated system?",
        "Was the call answered by an automated phone menu?",
        "Did a machine or a phone menu pick up?",
        "Did {A} end up talking to an automated menu?",
        "Is the other end an automated system rather than a person?",
        "Did an automated system answer the call {A} made?",
    ],
}

LIVE_OPTS = {"live": "a live person", "voicemail": "a voicemail greeting or answering machine",
             "automated": "an automated recording or phone menu", "nobody": "nobody (only silence or ringing)"}
LIVE_ORDER = ["live", "voicemail", "automated", "nobody"]
WAIT_OPTS = ["under 1 second", "1 to 3 seconds", "3 to 6 seconds", "over 6 seconds"]
SIL_OPTS = ["under 2 seconds", "2 to 5 seconds", "5 to 10 seconds", "over 10 seconds"]
COUNT_OPTS = ["none", "once", "two or more times"]
WHEN_MAP = {"it never did": "it never did", "first third": "in the first third", "middle third":
            "in the middle third", "last third": "in the last third"}
WHEN_OPTS = list(WHEN_MAP.values())


def fmt_n(n: float) -> str:
    if n == 1:
        return "1 second"
    return f"{n:g} seconds"


def short(role_phrase: str) -> str:
    return role_phrase


def cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def build_question(fam: str, answer: str, param, roles, rng: random.Random):
    """-> (question, options (canonical order), gold index, qtype)."""
    A, O = roles
    tmpl = rng.choice(T[fam])
    q = tmpl.format(A=A, O=O, N=fmt_n(param) if isinstance(param, (int, float)) else "")
    q = cap(q)
    qt = QTYPE[fam]
    if qt == "score_or_choice":
        qt = "score" if rng.random() < 0.5 else "choice"
    if qt == "noul":
        opts = ["yes", "no"]
        return q, opts, opts.index(answer), "noul"
    if fam == "live_choice":
        opts = [LIVE_OPTS[k] for k in LIVE_ORDER]
        return q, opts, LIVE_ORDER.index(answer), qt
    if fam == "wait_bin":
        return q, list(WAIT_OPTS), WAIT_OPTS.index(answer), qt
    if fam == "sil_bin":
        return q, list(SIL_OPTS), SIL_OPTS.index(answer), qt
    if fam == "barge_count":
        return q, list(COUNT_OPTS), COUNT_OPTS.index(answer), qt
    if fam == "aiover_when":
        return q, list(WHEN_OPTS), WHEN_OPTS.index(WHEN_MAP[answer]), qt
    if fam == "share":
        import aic_labels as AL
        labs = AL.SHARE_BINNINGS[param][1]
        opts = [f"{x} of the talking" for x in labs]
        return q, opts, labs.index(answer), qt
    if fam in ("first", "last", "more_talk"):
        opts = [A, O] + (["about the same"] if fam == "more_talk" else [])
        key = {"agent": 0, "other": 1, "equal": 2}[answer]
        return q, opts, key, qt
    raise KeyError(fam)


class NoShuffle:
    def shuffle(self, x):
        return None


def one_hot(n: int, k: int):
    return [1.0 if i == k else 0.0 for i in range(n)]
