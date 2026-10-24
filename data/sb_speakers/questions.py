"""sb_speakers question paraphrases (our own wording, >= 8 per family); none may equal an MMSU question."""
from __future__ import annotations

CHOICE = [
    "How many distinct people can be heard talking in this audio?",
    "Count every person who speaks at least once in this recording. How many are there?",
    "How many separate voices take part in this clip, counting each individual only once?",
    "In total, how many different individuals say something in this recording?",
    "Listening to the whole clip, how many different talkers do you count?",
    "How many unique speakers appear in this audio, no matter how briefly they talk?",
    "Some people may speak more than once. How many different people speak altogether?",
    "Counting each voice a single time, how many speakers are in this audio?",
    "What is the total number of different people speaking here, including anyone who only says a word or two?",
    "How many individual voices do you hear across the entire recording?",
    "Tally the distinct speakers in this clip. What do you get?",
    "How many people contribute speech to this recording?",
]
NOUL = {
    "exactly": [
        "Does this recording contain exactly {n} different speakers?",
        "Are precisely {n} distinct people talking in this clip?",
        "Counting each person once, is the number of speakers here exactly {n}?",
        "Would you say exactly {n} separate voices are heard in this audio?",
        "Is it true that {n} different people, no more and no fewer, speak in this recording?",
        "Do exactly {n} unique talkers appear in this clip?",
        "Is the count of distinct voices in this audio equal to {n}?",
        "Are there {n} different individuals speaking here, exactly?",
        "Does the clip feature exactly {n} different people talking?",
    ],
    "more": [
        "Does this recording contain more than {n} different speakers?",
        "Are more than {n} distinct people talking in this clip?",
        "Counting each person once, is the number of speakers here greater than {n}?",
        "Would you count over {n} separate voices in this audio?",
        "Do more than {n} unique talkers appear in this clip?",
        "Is the count of distinct voices in this audio above {n}?",
        "Are there more than {n} different individuals speaking here?",
        "Does the clip feature more than {n} different people talking?",
        "Is the number of different people heard here larger than {n}?",
    ],
    "fewer": [
        "Does this recording contain fewer than {n} different speakers?",
        "Are fewer than {n} distinct people talking in this clip?",
        "Counting each person once, is the number of speakers here smaller than {n}?",
        "Would you count under {n} separate voices in this audio?",
        "Do fewer than {n} unique talkers appear in this clip?",
        "Is the count of distinct voices in this audio below {n}?",
        "Are there fewer than {n} different individuals speaking here?",
        "Does the clip feature less than {n} different people talking?",
    ],
    "atleast": [
        "Does this recording contain at least {n} different speakers?",
        "Are {n} or more distinct people talking in this clip?",
        "Counting each person once, is the number of speakers here {n} or higher?",
        "Would you count at least {n} separate voices in this audio?",
        "Do {n} or more unique talkers appear in this clip?",
        "Is the count of distinct voices in this audio at least {n}?",
        "Are there at least {n} different individuals speaking here?",
        "Does the clip feature {n} or more different people talking?",
    ],
}
SCORE = [
    "Roughly how many different people talk in this clip?",
    "Which of these best describes the number of distinct speakers in the recording?",
    "How many separate voices are present here?",
    "Pick the option that matches how many different people speak in this audio.",
    "How many individuals take part in this recording, counting each once?",
    "Choose the speaker count for this clip.",
    "How many unique talkers does this audio contain?",
    "Classify this recording by its number of different speakers.",
    "How many people can be heard speaking, each counted a single time?",
]
SCORE_LEVELS = [
    ["one", "two", "three", "four or more"],
    ["1 speaker", "2 speakers", "3 speakers", "4 or more speakers"],
    ["a single person", "two people", "three people", "four or more people"],
    ["just one voice", "two voices", "three voices", "four or more voices"],
    ["1", "2", "3", "4+"],
]
# choice option wording; {n} digit, {w} number word
COUNT_FMT = ["{n}", "{w}", "{n} people", "{w} speakers", "{n} speakers", "{w} different voices", "{n} voices",
             "{w} people", "{n} different speakers", "{w}"]


def fmt_count(v: int, fm: str) -> str:
    from labels import NUM_WORDS
    w = NUM_WORDS[v] if v < len(NUM_WORDS) else str(v)
    s = fm.format(n=v, w=w)
    if v == 1:
        for a, b in (("people", "person"), ("speakers", "speaker"), ("voices", "voice")):
            s = s.replace(a, b)
    return s


import re as _re

_SING = _re.compile(r"\b(1|one) (different |distinct |separate |unique )?(speakers|people|voices|talkers|individuals)\b")
_SING_MAP = {"speakers": "speaker", "people": "person", "voices": "voice", "talkers": "talker", "individuals": "individual"}


def noul_question(template: str, n: int, as_word: bool) -> str:
    """Fill a yes/no template with the comparison number (digit or word), singular after 'one'."""
    from labels import NUM_WORDS
    s = template.format(n=(NUM_WORDS[n] if as_word and n < len(NUM_WORDS) else str(n)))
    if "exactly" in template or "precisely" in template or "equal to" in template or "no more and no fewer" in template:
        s = _SING.sub(lambda m: f"{m.group(1)} {m.group(2) or ''}{_SING_MAP[m.group(3)]}", s)
    return s


def all_templates() -> list[str]:
    out = list(CHOICE) + list(SCORE)
    for v in NOUL.values():
        out += v
    return out


def expand_all(max_n: int = 10) -> list[str]:
    """Every concrete question string the builder can emit."""
    out = list(CHOICE) + list(SCORE)
    for v in NOUL.values():
        for t in v:
            for n in range(0, max_n + 1):
                out.append(noul_question(t, n, False))
                out.append(noul_question(t, n, True))
    return out
