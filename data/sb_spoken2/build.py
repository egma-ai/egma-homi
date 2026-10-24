"""Pure logic for sb_spoken2: multiple-choice reasoning questions read aloud by Kokoro TTS.

Formats: read, stem, yn, valid, passage (see README.md). Sources: CommonsenseQA, QASC, QuaRTz and generated puzzles.
"""
from __future__ import annotations

import hashlib
import itertools
import random
import re
from collections import Counter, defaultdict

import common as C

BUILDER = "sb_spoken2"
MAX_CLIP_S = 20.0
CAL_FRAC = 0.03
PHONE_FRAC = 0.10
MAX_ITEMS_PER_CLIP = 4
MAX_TTS_WORDS = 46
YES_NO = ["yes", "no"]


class _NoShuffle:
    def shuffle(self, x):  # keeps canonical order (spoken order)
        return None


NO_SHUFFLE = _NoShuffle()


def h01(s: str) -> float:
    """Deterministic hash of a string -> [0, 1)."""
    return int(hashlib.sha1(s.encode()).hexdigest()[:12], 16) / float(16 ** 12)


def rng_for(*parts) -> random.Random:
    return random.Random(f"{C.SEED}:" + ":".join(str(p) for p in parts))


def balanced_flags(n: int, rng: random.Random) -> list[bool]:
    f = [i % 2 == 0 for i in range(n)]
    rng.shuffle(f)
    return f


def n_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9$%']+", text))


def cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


# Numbers -> words (for the spoken side; typed options keep digits)
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
        "twelve": "twelfth"}


def num_words(n: int) -> str:
    if n < 0:
        return "minus " + num_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " and " + num_words(n % 100))
    if n < 10000:
        return num_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + num_words(n % 1000))
    return str(n)  # large numbers: left to the TTS front end


def ordinal_words(n: int) -> str:
    w = num_words(n)
    last = re.split(r"[- ]", w)[-1]
    if last in _ORD:
        new = _ORD[last]
    elif last.endswith("y"):
        new = last[:-1] + "ieth"
    else:
        new = last + "th"
    return w[: len(w) - len(last)] + new


def clock_words(h: int, m: int) -> str:
    if m == 0:
        return f"{num_words(h)} o'clock"
    if m < 10:
        return f"{num_words(h)} oh {num_words(m)}"
    return f"{num_words(h)} {num_words(m)}"


def spoken_option(o: str) -> str:
    """Typed option -> how it is read in the `read` format (numbers as words)."""
    m = re.fullmatch(r"(\d{1,2}):(\d\d)", o)
    if m:
        return clock_words(int(m.group(1)), int(m.group(2)))
    m = re.fullmatch(r"([A-Z][a-z]+) (\d{1,2})", o)
    if m and m.group(1) in MONTHS:
        return f"{m.group(1)} {ordinal_words(int(m.group(2)))}"
    return re.sub(r"\d+", lambda mm: num_words(int(mm.group())), o)


# Typed question paraphrases (>= 5 each)
READ_Q = [  # VoiceBench wording first
    "The audio is a spoken multiple-choice question. Which option is the correct answer?",
    "Listen to the spoken multiple-choice question. Which option is correct?",
    "The speaker reads a question and its answer choices. Which choice is right?",
    "Which of the options read aloud correctly answers the spoken question?",
    "Answer the multiple-choice question in the recording.",
    "A question with lettered answer choices is read aloud. Pick the correct one.",
]
STEM_Q = [
    "Answer the question in the audio.",
    "Answer the question asked in the audio.",
    "Which option answers the spoken question?",
    "What is the best answer to the question you hear?",
    "The speaker asks a question. Pick the best answer.",
    "Respond to the question in the recording.",
]
YN_Q = [  # Big Bench Audio wording first
    "Answer the spoken yes/no question.",
    "The speaker asks a yes-or-no question. What is the right answer?",
    "Answer the question in the audio with yes or no.",
    "What is the correct answer to the yes/no question in the recording?",
    "Listen to the question and answer it: yes or no?",
]
VALID_Q = [  # Big Bench Audio formal_fallacies wording first
    "Answer the spoken question: is the argument deductively valid or invalid?",
    "Is the argument in the audio deductively valid or invalid?",
    "The speaker presents an argument. Is it logically valid or invalid?",
    "Judge the spoken argument: is it valid or invalid?",
    "Does the reasoning in the recording form a valid or an invalid argument?",
]
VALID_YN_Q = [
    "Does the conclusion in the audio follow logically from the premises?",
    "The speaker makes an argument. Does the conclusion necessarily follow?",
    "Is the conclusion the speaker draws logically guaranteed by what was said before it?",
    "Listen to the argument. Must the conclusion be true if the premises are true?",
    "Does the spoken argument's conclusion follow from its premises?",
]
VALID_TAILS = ["Is this argument valid or invalid?", "Is the argument valid?", "Valid or invalid?",
               "Is this reasoning valid?", "Does the conclusion follow?"]
PASSAGE_PREFIX = ["{q}", "Based on what the speaker says: {q}", "Listen to the recording. {q}", "From the audio: {q}",
                  "Using only what is said in the audio, answer: {q}", "According to the speaker, {ql}"]
CONCL_Q = [
    "Based on the context, which conclusion is most suitable?",
    "Which conclusion follows from what the speaker says?",
    "Given the statements in the audio, which of these must be true?",
    "What can be concluded from the spoken statements?",
    "Which statement is logically implied by what you hear?",
    "Based on the speaker's statements, which conclusion is correct?",
]
CONCL_SPOKEN_Q = ["What can we conclude?", "Which conclusion follows?", "What must be true?",
                  "So what follows from this?", "What can be concluded?"]
FOLLOW_YN = ["Based on the audio, does it follow that {s}?", "Given what the speaker says, must it be true that {s}?",
             "From the spoken statements, can we conclude that {s}?", "Does it follow from the audio that {s}?",
             "Is it necessarily true, given the statements in the audio, that {s}?"]


def passage_q(rng, q: str) -> str:
    t = rng.choice(PASSAGE_PREFIX)
    return t.format(q=q, ql=q[:1].lower() + q[1:])


# Names and vocab (neutral, ordinary)
FEMALE = ["Anna", "Clara", "Diana", "Emma", "Grace", "Hannah", "Julia", "Laura", "Maya", "Nina", "Olivia", "Rosa",
          "Sarah", "Tara", "Vera", "Zoe", "Alice", "Bella", "Chloe", "Ella", "Fiona", "Helen", "Irene", "Lucy",
          "Megan", "Paula", "Ruby", "Sofia", "Wendy", "Amy", "Kate", "Mia", "Lily", "Jane", "Leah", "Eva"]
MALE = ["Ben", "Carlos", "David", "Ethan", "Frank", "George", "Henry", "Ivan", "Jack", "Kevin", "Leo", "Mark",
        "Nathan", "Oscar", "Peter", "Ryan", "Samuel", "Tom", "Victor", "Adam", "Brian", "Daniel", "Eric", "Felix",
        "Hugo", "Liam", "Max", "Owen", "Paul", "Simon", "James", "Luke", "Noah", "Sam", "Alex", "Joe"]
_FEM = set(FEMALE)


def pron(name: str) -> str:
    return "she" if name in _FEM else "he"


def poss(name: str) -> str:
    return "her" if name in _FEM else "his"


def names(rng, k: int, pool=None) -> list[str]:
    return rng.sample(pool or (FEMALE + MALE), k)


DIRS = ["north", "east", "south", "west"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
MONTH_DAYS = {"January": 31, "March": 31, "April": 30, "May": 31, "June": 30, "July": 31, "August": 31,
              "September": 30, "October": 31, "November": 30, "December": 31}  # February skipped (leap years)


# Puzzle generators. Each returns {"text": spoken premises, "choice": [Q...], "yn": Q | None, "valid": bool | None}
#   choice Q = {"q": question (spoken or typed), "options": [...], "gold": int}
#   yn Q     = {"q": yes/no question, "yes": bool}
ORDER_RELS = [
    dict(more="{a} is taller than {b}.", less="{a} is shorter than {b}.", top="Who is the tallest?",
         bottom="Who is the shortest?", ynm="Is {a} taller than {b}?", ynl="Is {a} shorter than {b}?"),
    dict(more="{a} is older than {b}.", less="{a} is younger than {b}.", top="Who is the oldest?",
         bottom="Who is the youngest?", ynm="Is {a} older than {b}?", ynl="Is {a} younger than {b}?"),
    dict(more="{a} runs faster than {b}.", less="{a} runs slower than {b}.", top="Who runs the fastest?",
         bottom="Who runs the slowest?", ynm="Does {a} run faster than {b}?", ynl="Does {a} run slower than {b}?"),
    dict(more="{a} arrived earlier than {b}.", less="{a} arrived later than {b}.", top="Who arrived first?",
         bottom="Who arrived last?", ynm="Did {a} arrive earlier than {b}?", ynl="Did {a} arrive later than {b}?"),
    dict(more="{a} scored more points than {b}.", less="{a} scored fewer points than {b}.",
         top="Who scored the most points?", bottom="Who scored the fewest points?",
         ynm="Did {a} score more points than {b}?", ynl="Did {a} score fewer points than {b}?"),
    dict(more="{a} lives closer to the school than {b}.", less="{a} lives farther from the school than {b}.",
         top="Who lives closest to the school?", bottom="Who lives farthest from the school?",
         ynm="Does {a} live closer to the school than {b}?", ynl="Does {a} live farther from the school than {b}?"),
    dict(more="{a}'s bag is heavier than {b}'s.", less="{a}'s bag is lighter than {b}'s.",
         top="Whose bag is the heaviest?", bottom="Whose bag is the lightest?",
         ynm="Is {a}'s bag heavier than {b}'s?", ynl="Is {a}'s bag lighter than {b}'s?"),
]


def _chain_yn(rng, rank: list[str], yes: bool, ynm: str, ynl: str) -> str:
    """yes/no question about a pair (non-adjacent when possible, so it needs the chain), label `yes`."""
    n = len(rank)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if j - i >= 2] or [(0, 1)]
    i, j = rng.choice(pairs)
    hi, lo = rank[i], rank[j]  # hi ranks above lo
    if yes:
        return rng.choice([ynm.format(a=hi, b=lo), ynl.format(a=lo, b=hi)])
    return rng.choice([ynm.format(a=lo, b=hi), ynl.format(a=hi, b=lo)])


def gen_order(rng, want_yes: bool) -> dict:
    rel = rng.choice(ORDER_RELS)
    n = 3 if rng.random() < 0.6 else 4
    rank = names(rng, n)
    prem = []
    for i in range(n - 1):
        a, b = rank[i], rank[i + 1]
        prem.append(rel["more"].format(a=a, b=b) if rng.random() < 0.5 else rel["less"].format(a=b, b=a))
    rng.shuffle(prem)
    opts = rank[:]
    rng.shuffle(opts)
    top = rng.random() < 0.5
    q = {"q": rel["top"] if top else rel["bottom"], "options": opts, "gold": opts.index(rank[0] if top else rank[-1])}
    return {"text": " ".join(prem), "choice": [q], "yn": {"q": _chain_yn(rng, rank, want_yes, rel["ynm"], rel["ynl"]),
                                                          "yes": want_yes}, "valid": None,
            "extra": {"rank": rank, "rel": ORDER_RELS.index(rel)}}


COLORS = ["red", "blue", "green", "yellow", "white", "black", "orange", "brown", "gray", "pink"]
ROW_OBJECTS = [("cup", "cups", "on a shelf"), ("book", "books", "on a shelf"), ("box", "boxes", "on the floor"),
               ("chair", "chairs", "against the wall"), ("vase", "vases", "on a table"), ("car", "cars", "in a parking row"),
               ("lamp", "lamps", "on a long desk"), ("bike", "bikes", "at the rack")]
COUNT_WORD = {3: "Three", 4: "Four"}


def gen_spatial(rng, want_yes: bool) -> dict:
    obj, objs, where = rng.choice(ROW_OBJECTS)
    n = 3 if rng.random() < 0.6 else 4
    cols = rng.sample(COLORS, n)  # left -> right
    th = [f"the {c} {obj}" for c in cols]
    prem = [f"{COUNT_WORD[n]} {objs} stand in a row {where}."]
    rel = []
    for i in range(n - 1):
        a, b = th[i], th[i + 1]
        rel.append(f"{cap(a)} is to the left of {b}." if rng.random() < 0.5 else f"{cap(b)} is to the right of {a}.")
    rng.shuffle(rel)
    opts = th[:]
    rng.shuffle(opts)
    kind = rng.choice(["left", "right"] + (["middle"] if n == 3 else []))
    if kind == "left":
        q, g = f"Which {obj} is furthest to the left?", th[0]
    elif kind == "right":
        q, g = f"Which {obj} is furthest to the right?", th[-1]
    else:
        q, g = f"Which {obj} is in the middle?", th[1]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if j - i >= 2] or [(0, 1)]
    i, j = rng.choice(pairs)
    if want_yes:
        yq = rng.choice([f"Is {th[i]} to the left of {th[j]}?", f"Is {th[j]} to the right of {th[i]}?"])
    else:
        yq = rng.choice([f"Is {th[j]} to the left of {th[i]}?", f"Is {th[i]} to the right of {th[j]}?"])
    return {"text": " ".join(prem + rel), "choice": [{"q": q, "options": opts, "gold": opts.index(g)}],
            "yn": {"q": yq, "yes": want_yes}, "valid": None}


ACTIVITIES = [("washed the car", "wash the car"), ("made lunch", "make lunch"), ("called a friend", "call a friend"),
              ("read the newspaper", "read the newspaper"), ("walked the dog", "walk the dog"),
              ("watered the plants", "water the plants"), ("did the laundry", "do the laundry"),
              ("went to the bank", "go to the bank"), ("cleaned the kitchen", "clean the kitchen"),
              ("baked a cake", "bake a cake"), ("fixed the bike", "fix the bike"), ("wrote an email", "write an email"),
              ("went for a run", "go for a run"), ("bought groceries", "buy groceries"),
              ("paid the bills", "pay the bills"), ("ironed a shirt", "iron a shirt")]


def gen_temporal(rng, want_yes: bool) -> dict:
    who = names(rng, 1)[0]
    p = pron(who)
    n = 3 if rng.random() < 0.6 else 4
    acts = rng.sample(ACTIVITIES, n)  # in time order
    prem = []
    for i in range(n - 1):
        (pa, _), (pb, _) = acts[i], acts[i + 1]
        if rng.random() < 0.5:
            prem.append(f"{who} {pa} before {p} {pb}.")
        else:
            prem.append(f"{who} {pb} after {p} {pa}.")
    rng.shuffle(prem)
    opts = [a[0] for a in acts]
    rng.shuffle(opts)
    first = rng.random() < 0.5
    q = {"q": f"What did {who} do {'first' if first else 'last'}?", "options": opts,
         "gold": opts.index(acts[0][0] if first else acts[-1][0])}
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if j - i >= 2] or [(0, 1)]
    i, j = rng.choice(pairs)  # i earlier than j
    if want_yes:
        yq = rng.choice([f"Did {who} {acts[i][1]} before {p} {acts[j][0]}?", f"Did {who} {acts[j][1]} after {p} {acts[i][0]}?"])
    else:
        yq = rng.choice([f"Did {who} {acts[j][1]} before {p} {acts[i][0]}?", f"Did {who} {acts[i][1]} after {p} {acts[j][0]}?"])
    return {"text": " ".join(prem), "choice": [q], "yn": {"q": yq, "yes": want_yes}, "valid": None}


TRUTH_FIRST = {True: ["{a} tells the truth.", "{a} always tells the truth.", "{a} is honest."],
               False: ["{a} lies.", "{a} always lies.", "{a} is a liar."]}
TRUTH_CLAIM = {True: ["{b} says that {a} tells the truth.", "{b} says {a} is honest.", "{b} claims that {a} is telling the truth."],
               False: ["{b} says that {a} lies.", "{b} says {a} is lying.", "{b} claims that {a} is a liar."]}
TRUTH_ASK = {True: ["Does {x} tell the truth?", "Is {x} telling the truth?", "Is {x} honest?"],
             False: ["Does {x} lie?", "Is {x} lying?", "Is {x} a liar?"]}


def truth_values(first_truthful: bool, claims_truthful: list[bool]) -> list[bool]:
    """claims_truthful[i] = person i+1 says person i tells the truth (True) or lies (False)."""
    t = [first_truthful]
    for c in claims_truthful:
        t.append(c == t[-1])
    return t


def gen_truth(rng, want_yes: bool) -> dict:
    n = rng.choice([3, 4, 4, 5])
    ppl = names(rng, n)
    first = rng.random() < 0.5
    claims = [rng.random() < 0.5 for _ in range(n - 1)]
    t = truth_values(first, claims)
    sents = [rng.choice(TRUTH_FIRST[first]).format(a=ppl[0])]
    for i, c in enumerate(claims):
        sents.append(rng.choice(TRUTH_CLAIM[c]).format(a=ppl[i], b=ppl[i + 1]))
    x = n - 1 if rng.random() < 0.8 else rng.randrange(1, n - 1)
    # ask about truthfulness or lying so that the answer matches want_yes
    ask_truth = (t[x] == want_yes)
    yq = rng.choice(TRUTH_ASK[ask_truth]).format(x=ppl[x])
    opts = ["telling the truth", "lying"]
    cq = {"q": rng.choice([f"Is {ppl[x]} telling the truth or lying?", f"Is {ppl[x]} honest or lying?"]),
          "options": opts, "gold": 0 if t[x] else 1}
    return {"text": " ".join(sents), "choice": [cq], "yn": {"q": yq, "yes": want_yes}, "valid": None,
            "extra": {"truth_values": t, "asked": ppl[x]}}


TURNS = {"left": -1, "right": 1, "around": 2}


def gen_facing(rng, want_yes: bool) -> dict:
    d0 = rng.randrange(4)
    k = rng.choice([2, 3, 3, 4])
    turns = [rng.choice(["left", "right", "left", "right", "around"]) for _ in range(k)]
    d = d0
    for tt in turns:
        d = (d + TURNS[tt]) % 4
    parts = [f"turn {tt}" for tt in turns]
    seq = ", then ".join(parts)
    start = rng.choice(["You are facing {d}.", "You stand facing {d}.", "Imagine you are facing {d}."]).format(d=DIRS[d0])
    text = f"{start} You {seq}."
    opts = DIRS[:]
    q = {"q": rng.choice(["Which direction are you facing now?", "Which way are you facing now?",
                          "In which direction are you facing at the end?"]), "options": opts, "gold": d}
    asked = DIRS[d] if want_yes else rng.choice([x for x in DIRS if x != DIRS[d]])
    return {"text": text, "choice": [q], "yn": {"q": f"Are you facing {asked} now?", "yes": want_yes}, "valid": None}


PLACES = ["library", "station", "park", "bakery", "museum", "post office", "school", "market", "hotel", "bus stop"]
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
DVEC = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}


def gen_grid(rng, want_yes: bool) -> dict:
    """Walk blocks along streets. want_yes -> you end at the start; else you end displaced along one axis."""
    place = rng.choice(PLACES)
    for _ in range(200):
        if want_yes:
            if rng.random() < 0.4:
                d = rng.choice(DIRS)
                k = rng.randint(1, 6)
                legs = [(d, k), (OPP[d], k)]
            else:
                d1 = rng.choice(DIRS)
                d2 = rng.choice([x for x in DIRS if x not in (d1, OPP[d1])])
                a, b = rng.randint(1, 6), rng.randint(1, 6)
                legs = [(d1, a), (d2, b), (OPP[d1], a), (OPP[d2], b)]
                if rng.random() < 0.5:  # walk the box the other way round
                    legs = [(d1, a), (d2, b), (OPP[d2], b), (OPP[d1], a)]
        else:
            k = rng.choice([2, 3, 3])
            legs = [(rng.choice(DIRS), rng.randint(1, 6)) for _ in range(k)]
        if any(legs[i][0] == legs[i + 1][0] for i in range(len(legs) - 1)):
            continue
        x = sum(DVEC[d][0] * k for d, k in legs)
        y = sum(DVEC[d][1] * k for d, k in legs)
        if want_yes and (x, y) == (0, 0):
            break
        if not want_yes and (x == 0) != (y == 0):  # displaced along exactly one axis
            break
    else:
        raise RuntimeError("grid")

    def leg(d, k):
        return f"{num_words(k)} block{'s' if k > 1 else ''} {d}"

    text = f"Starting at the {place}, you walk " + ", then ".join(leg(d, k) for d, k in legs) + "."
    yq = rng.choice([f"Are you back at the {place}?", f"Do you end up where you started?",
                     f"Are you at the {place} again?"])
    out = {"text": text, "yn": {"q": yq, "yes": want_yes}, "valid": None, "choice": []}
    if want_yes:  # back at the start: distance 0
        nums = [0] + rng.sample(range(1, 9), 3)
        rng.shuffle(nums)
        out["choice"].append({"q": f"How many blocks are you from the {place} now?", "options": [str(v) for v in nums],
                              "gold": nums.index(0)})
    else:
        if x:
            back = "west" if x > 0 else "east"
        else:
            back = "south" if y > 0 else "north"
        dist = abs(x) + abs(y)
        out["choice"].append({"q": rng.choice([f"In which direction is the {place} from where you are now?",
                                               f"Which way would you walk to get back to the {place}?"]),
                              "options": DIRS[:], "gold": DIRS.index(back)})
        nums = [dist] + rng.sample([v for v in range(1, 13) if v != dist], 3)
        rng.shuffle(nums)
        out["choice"].append({"q": f"How many blocks are you from the {place} now?", "options": [str(v) for v in nums],
                              "gold": nums.index(dist)})
    return out


def _num_options(rng, gold: int, cands: list[int], k: int = 4, lo: int = 0) -> tuple[list[str], int]:
    pool = [c for c in dict.fromkeys(cands) if c != gold and c >= lo]
    rng.shuffle(pool)
    opts = [gold] + pool[: k - 1]
    d = 1
    while len(opts) < k:
        for c in (gold + d, gold - d):
            if c >= lo and c not in opts and len(opts) < k:
                opts.append(c)
        d += 1
    rng.shuffle(opts)  # random order: sorted numbers would put the gold at predictable positions in `read`
    return [str(o) for o in opts], opts.index(gold)


def gen_weekday(rng, want_yes: bool) -> dict:
    kind = rng.choice(["future", "past", "rel", "rel"])
    if kind == "future":
        d0, n = rng.randrange(7), rng.randint(2, 20)
        text = rng.choice(["Today is {d}.", "It is {d} today."]).format(d=WEEKDAYS[d0])
        q = f"What day of the week will it be in {num_words(n)} days?"
        g = (d0 + n) % 7
        yq = "Will it be {x} in " + num_words(n) + " days?"
    elif kind == "past":
        d0, n = rng.randrange(7), rng.randint(2, 20)
        text = rng.choice(["Today is {d}.", "It is {d} today."]).format(d=WEEKDAYS[d0])
        q = f"What day of the week was it {num_words(n)} days ago?"
        g = (d0 - n) % 7
        yq = "Was it {x} " + num_words(n) + " days ago?"
    else:
        rels = [("Yesterday was {d}.", -1), ("Tomorrow is {d}.", 1), ("The day before yesterday was {d}.", -2),
                ("The day after tomorrow is {d}.", 2), ("Three days ago it was {d}.", -3)]
        asks = [("What day is it today?", 0, "Is today {x}?"), ("What day will it be tomorrow?", 1, "Is tomorrow {x}?"),
                ("What day was it yesterday?", -1, "Was yesterday {x}?"),
                ("What day will it be the day after tomorrow?", 2, "Is the day after tomorrow {x}?")]
        (rt, ro), (q, ao, yq) = rng.choice(rels), rng.choice(asks)
        if ro == ao:
            ao, q, yq = 0, "What day is it today?", "Is today {x}?"
            if ro == 0:
                raise RuntimeError
        today = rng.randrange(7)
        text = rt.format(d=WEEKDAYS[(today + ro) % 7])
        g = (today + ao) % 7
    others = [i for i in range(7) if i != g]
    opts_i = sorted([g] + rng.sample(others, 3))
    opts = [WEEKDAYS[i] for i in opts_i]
    asked = WEEKDAYS[g] if want_yes else WEEKDAYS[rng.choice(others)]
    return {"text": text, "choice": [{"q": q, "options": opts, "gold": opts.index(WEEKDAYS[g])}],
            "yn": {"q": yq.format(x=asked), "yes": want_yes}, "valid": None}


def gen_month(rng, want_yes: bool) -> dict:
    m0, n = rng.randrange(12), rng.randint(2, 15)
    fwd = rng.random() < 0.6
    text = rng.choice(["It is {m} now.", "The current month is {m}.", "We are in {m}."]).format(m=MONTHS[m0])
    if fwd:
        q, g, yq = f"Which month will it be in {num_words(n)} months?", (m0 + n) % 12, f"Will it be {{x}} in {num_words(n)} months?"
    else:
        q, g, yq = f"Which month was it {num_words(n)} months ago?", (m0 - n) % 12, f"Was it {{x}} {num_words(n)} months ago?"
    others = [i for i in range(12) if i != g]
    oi = sorted([g] + rng.sample(others, 3))
    opts = [MONTHS[i] for i in oi]
    asked = MONTHS[g] if want_yes else MONTHS[rng.choice(others)]
    return {"text": text, "choice": [{"q": q, "options": opts, "gold": opts.index(MONTHS[g])}],
            "yn": {"q": yq.format(x=asked), "yes": want_yes}, "valid": None}


CLOCK_EVENTS = [("A train leaves at {t} and the trip takes {d}.", "When does the train arrive?", "Does the train arrive at {x}?"),
                ("A film starts at {t} and lasts {d}.", "When does the film end?", "Does the film end at {x}?"),
                ("{n} starts cooking at {t}. The meal takes {d} to prepare.", "At what time is the meal ready?", "Is the meal ready at {x}?"),
                ("The meeting begins at {t} and runs for {d}.", "When does the meeting finish?", "Does the meeting finish at {x}?"),
                ("{n} puts a cake in the oven at {t}. It needs to bake for {d}.", "When should the cake come out?", "Should the cake come out at {x}?")]


def dur_words(mins: int) -> str:
    h, m = divmod(mins, 60)
    parts = []
    if h:
        parts.append(f"{num_words(h)} hour{'s' if h > 1 else ''}")
    if m:
        parts.append(f"{num_words(m)} minutes")
    return " and ".join(parts)


def gen_clock(rng, want_yes: bool) -> dict:
    for _ in range(100):
        h, m = rng.randint(1, 10), rng.choice(range(0, 60, 5))
        d = rng.choice([15, 20, 25, 30, 35, 40, 45, 50, 55, 70, 75, 80, 90, 95, 100, 105, 110, 120, 135, 150])
        tot = h * 60 + m + d
        if tot < 12 * 60:  # stay within the same half-day (no am/pm ambiguity)
            break
    tpl, q, yq = rng.choice(CLOCK_EVENTS)
    text = tpl.format(t=clock_words(h, m), d=dur_words(d), n=names(rng, 1)[0])
    g = tot

    def fmt(v):
        return f"{v // 60}:{v % 60:02d}"

    cands = [tot + 10, tot - 10, tot + 60, tot - 60, tot + 5, tot - 5, h * 60 + m + d % 60, tot + 30, tot - 30]
    cands = [c for c in cands if 60 <= c < 12 * 60 and c != g]
    pool = list(dict.fromkeys(cands))
    rng.shuffle(pool)
    vals = [g] + pool[:3]
    rng.shuffle(vals)
    opts = [fmt(v) for v in vals]
    asked = g if want_yes else pool[3 % len(pool)] if len(pool) > 3 else pool[0]
    return {"text": text, "choice": [{"q": q, "options": opts, "gold": vals.index(g)}],
            "yn": {"q": yq.format(x=clock_words(asked // 60, asked % 60)), "yes": want_yes}, "valid": None}


_CAL = [(m, d) for m in MONTHS for d in range(1, (MONTH_DAYS.get(m) or 28) + 1)]  # a non-leap year


def shift_date(month: str, day: int, k: int) -> tuple[str, int]:
    """Move k days (k may be negative) in a non-leap year calendar (wraps Dec -> Jan)."""
    i = _CAL.index((month, day))
    return _CAL[(i + k) % len(_CAL)]


def gen_date(rng, want_yes: bool) -> dict:
    mon = rng.choice(list(MONTH_DAYS))  # never February (leap-year ambiguity)
    day = rng.randint(1, MONTH_DAYS[mon])
    n = rng.randint(2, 12)
    later = rng.random() < 0.6
    gm, gd = shift_date(mon, day, n if later else -n)
    if gm == "February":
        raise RuntimeError("feb")
    ev = rng.choice(["The meeting", "The school trip", "The dentist appointment", "The concert", "The football match"])
    text = f"{ev} was planned for {mon} {ordinal_words(day)}, but it was moved {num_words(n)} days {'later' if later else 'earlier'}."
    q = rng.choice(["On what date does it take place now?", "What is the new date?"])
    gold = f"{gm} {gd}"
    cands = []
    for k in (1, -1, 2, -2, 7, -7):
        cm, cd = shift_date(gm, gd, k)
        if cm != "February":
            cands.append(f"{cm} {cd}")
    if gm != mon:  # the classic slip: keep the old month
        cands.append(f"{mon} {gd}")
    pool = [c for c in dict.fromkeys(cands) if c != gold]
    rng.shuffle(pool)
    opts = [gold] + pool[:3]
    rng.shuffle(opts)
    am, ad = (gm, gd) if want_yes else (pool[3 % len(pool)].split(" ")[0], int(pool[3 % len(pool)].split(" ")[1]))
    yq = f"Does it now take place on {am} {ordinal_words(ad)}?"
    return {"text": text, "choice": [{"q": q, "options": opts, "gold": opts.index(gold)}],
            "yn": {"q": yq, "yes": want_yes}, "valid": None}


ARITH_OBJ = ["stickers", "marbles", "apples", "pencils", "stamps", "postcards", "cookies", "coins", "books", "shells"]


def gen_arith(rng, want_yes: bool) -> dict:
    who, other = names(rng, 2)
    p = pron(who)
    obj = rng.choice(ARITH_OBJ)
    kind = rng.choice(["buy_give", "give_buy", "times", "share", "more_than", "per_day", "cost"])
    if kind == "buy_give":
        a, b = rng.randint(4, 25), rng.randint(2, 15)
        c = rng.randint(1, a + b - 1)
        g = a + b - c
        text = f"{who} has {num_words(a)} {obj}. {cap(p)} gets {num_words(b)} more and then gives {num_words(c)} to {other}."
        q = f"How many {obj} does {who} have now?"
        cands = [a + b + c, a - b + c, a + b, a - c, g + 1, g - 1, g + 2]
    elif kind == "give_buy":
        a = rng.randint(6, 25)
        c = rng.randint(1, a - 1)
        b = rng.randint(2, 15)
        g = a - c + b
        text = f"{who} had {num_words(a)} {obj}. {cap(p)} gave away {num_words(c)} and later found {num_words(b)} more."
        q = f"How many {obj} does {who} have now?"
        cands = [a + c + b, a - c - b, a + b, a - c, g + 1, g - 1]
    elif kind == "times":
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        g = a * b
        cont = rng.choice(["box", "bag", "jar", "packet"])
        text = f"Each {cont} holds {num_words(a)} {obj}. {who} has {num_words(b)} full {cont}{'es' if cont == 'box' else 's'}."
        q = f"How many {obj} does {who} have in total?"
        cands = [a + b, a * (b + 1), a * (b - 1), g + 1, g - 1, g + a]
    elif kind == "share":
        b, each = rng.randint(2, 8), rng.randint(2, 9)
        a = b * each
        g = each
        text = f"{who} shares {num_words(a)} {obj} equally among {num_words(b)} friends."
        q = "How many does each friend get?"
        cands = [a - b, b, each + 1, each - 1, each + 2, a // 2]
    elif kind == "more_than":
        a, b = sorted(rng.sample(range(3, 40), 2))
        g = b - a
        text = f"{who} has {num_words(a)} {obj} and {other} has {num_words(b)}."
        q = f"How many more {obj} does {other} have than {who}?"
        cands = [a + b, g + 1, g - 1, g + 2, g + 10, b]
    elif kind == "per_day":
        a, b = rng.randint(2, 15), rng.randint(2, 7)
        g = a * b
        text = f"{who} reads {num_words(a)} pages every day."
        q = f"How many pages does {p} read in {num_words(b)} days?"
        cands = [a + b, a * (b + 1), a * (b - 1), g + 1, g - 1, g + 10]
    else:
        a, b = rng.randint(2, 12), rng.randint(2, 6)
        g = a * b
        item = rng.choice(["ticket", "notebook", "sandwich", "plant", "mug"])
        text = f"One {item} costs {num_words(a)} dollars. {who} buys {num_words(b)} of them."
        q = f"How many dollars does {who} spend?"
        cands = [a + b, a * (b + 1), a * (b - 1), g + 1, g - 1, g + 2]
    opts, gi = _num_options(rng, g, cands, lo=0)
    asked = g if want_yes else int(rng.choice([o for o in opts if o != str(g)]))
    yq = rng.choice(["{q} Is it {x}?", "{q} Is the answer {x}?"]).format(q=q, x=num_words(asked))
    return {"text": text, "choice": [{"q": q, "options": opts, "gold": gi}], "yn": {"q": yq, "yes": want_yes,
                                                                                      "self_contained": True},
            "valid": None}


COUNT_SETS = [
    ("fruit", ["apples", "pears", "oranges", "bananas", "plums", "peaches", "lemons"],
     ["carrots", "potatoes", "onions", "leeks"],
     "A basket holds {items}.", "How many pieces of fruit are in the basket?",
     "How many vegetables are in the basket?"),
    ("birds", ["ducks", "hens", "geese", "swans"], ["cows", "horses", "sheep", "goats", "pigs"],
     "On the farm there are {items}.", "How many birds are on the farm?", "How many of the animals are not birds?"),
]
SING = {"apples": "apple", "pears": "pear", "oranges": "orange", "bananas": "banana", "plums": "plum", "peaches": "peach",
        "lemons": "lemon", "carrots": "carrot", "potatoes": "potato", "onions": "onion", "leeks": "leek",
        "ducks": "duck", "hens": "hen", "geese": "goose", "swans": "swan", "cows": "cow", "horses": "horse",
        "sheep": "sheep", "goats": "goat", "pigs": "pig"}


def gen_count(rng, want_yes: bool) -> dict:
    _, cat_a, cat_b, tpl, qa, qb = rng.choice(COUNT_SETS)
    na, nb = rng.randint(1, 3), rng.randint(1, 2)
    ka, kb = rng.sample(cat_a, na), rng.sample(cat_b, nb)
    counts = {w: rng.randint(1, 6) for w in ka + kb}
    order = ka + kb
    rng.shuffle(order)
    parts = [f"{num_words(counts[w])} {w if counts[w] > 1 else SING[w]}" for w in order]
    items = ", ".join(parts[:-1]) + " and " + parts[-1]
    text = tpl.format(items=items)
    which = rng.random() < 0.5
    if rng.random() < 0.25:
        q, g = rng.choice(["How many are there in total?", "How many are there altogether?"]), sum(counts.values())
    else:
        q, g = (qa, sum(counts[w] for w in ka)) if which else (qb, sum(counts[w] for w in kb))
    cands = [sum(counts.values()), sum(counts[w] for w in ka), sum(counts[w] for w in kb), g + 1, g - 1, g + 2]
    opts, gi = _num_options(rng, g, cands, lo=1)
    asked = g if want_yes else int(rng.choice([o for o in opts if o != str(g)]))
    yq = f"{q} Is it {num_words(asked)}?"
    return {"text": text, "choice": [{"q": q, "options": opts, "gold": gi}],
            "yn": {"q": yq, "yes": want_yes, "self_contained": True}, "valid": None}


# family: (template, relation of A to C by gender of A, distractor relations by gender)
FAMILY = [
    ("{A} is {B}'s {parentA}. {B} is {C}'s {parentB}.", {"f": "grandmother", "m": "grandfather"}),
    ("{A} and {B} are {sibs}. {B} is {C}'s {parentB}.", {"f": "aunt", "m": "uncle"}),
    ("{A} is {B}'s {childA}. {B} is {C}'s {sibB}.", {"f": "niece", "m": "nephew"}),
    ("{A} is {B}'s {childA}. {C} is {B}'s {childC}.", {"f": "sister", "m": "brother"}),
    ("{A}'s {parentWord} and {C}'s {parentWord} are {sibsP}.", {"f": "cousin", "m": "cousin"}),
]
REL_POOL = {"f": ["grandmother", "mother", "aunt", "sister", "cousin", "niece", "daughter"],
            "m": ["grandfather", "father", "uncle", "brother", "cousin", "nephew", "son"]}


def gen_family(rng, want_yes: bool) -> dict:
    ti = rng.randrange(len(FAMILY))
    tpl, rel = FAMILY[ti]
    ga = "f" if rng.random() < 0.5 else "m"
    gb = "f" if rng.random() < 0.5 else "m"
    gc = "f" if rng.random() < 0.5 else "m"
    pick = {"f": FEMALE, "m": MALE}
    A = rng.choice(pick[ga])
    B = rng.choice([x for x in pick[gb] if x != A])
    Cn = rng.choice([x for x in pick[gc] if x not in (A, B)])
    fill = {
        "A": A, "B": B, "C": Cn,
        "parentA": "mother" if ga == "f" else "father",
        "parentB": "mother" if gb == "f" else "father",
        "childA": "daughter" if ga == "f" else "son",
        "childC": "daughter" if gc == "f" else "son",
        "sibB": "sister" if gb == "f" else "brother",
        "sibs": "sisters" if ga == gb == "f" else "brothers" if ga == gb == "m" else "siblings",
        "parentWord": rng.choice(["mother", "father"]),
    }
    fill["sibsP"] = "sisters" if fill["parentWord"] == "mother" else "brothers"
    text = tpl.format(**fill)
    gold = rel[ga]
    pool = [r for r in REL_POOL[ga] if r != gold]
    rng.shuffle(pool)
    opts = [gold] + pool[:3]
    rng.shuffle(opts)
    q = rng.choice([f"What is {A} to {Cn}?", f"How is {A} related to {Cn}?", f"What relation is {A} to {Cn}?"])
    asked = gold if want_yes else rng.choice(pool[:3])
    art = "an" if asked[0] in "aeiou" else "a"
    yq = rng.choice([f"Is {A} {Cn}'s {asked}?", f"Is {A} {art} {asked} of {Cn}?"])
    return {"text": text, "choice": [{"q": q, "options": opts, "gold": opts.index(gold)}],
            "yn": {"q": yq, "yes": want_yes}, "valid": None}


# Logic: categorical syllogisms
ROLES = ["painters", "cyclists", "teachers", "gardeners", "swimmers", "bakers", "runners", "singers", "chess players",
         "musicians", "writers", "dancers", "hikers", "cooks", "students", "volunteers", "photographers", "climbers",
         "readers", "drivers"]
SYL_CTX = ["In a small town,", "At one school,", "In a certain club,", "In this village,", "At the community center,",
           "On one street,"]
FORMS = ["A", "E", "I", "O"]


def stmt_true(form: str, x: int, y: int, nonempty: frozenset) -> bool:
    """Truth of a categorical statement over Venn regions (region = 3-bit membership mask; nonempty = set of masks)."""
    xy = [r for r in nonempty if r >> x & 1 and r >> y & 1]
    x_not_y = [r for r in nonempty if r >> x & 1 and not r >> y & 1]
    return {"A": not x_not_y, "E": not xy, "I": bool(xy), "O": bool(x_not_y)}[form]


def syllogism_valid(prem: list[tuple], concl: tuple, existential_import: bool) -> bool:
    """prem/concl: (form, x, y) with terms 0=S, 1=M, 2=P. Valid iff every model of the premises satisfies the conclusion.
    existential_import=True restricts to models where S, M and P are all non-empty (traditional logic)."""
    for bits in range(256):
        nonempty = frozenset(r for r in range(8) if bits >> r & 1)
        if existential_import and not all(any(r >> t & 1 for r in nonempty) for t in range(3)):
            continue
        if all(stmt_true(f, x, y, nonempty) for f, x, y in prem) and not stmt_true(*concl, nonempty):
            return False
    return True


def stmt_text(form: str, X: str, Y: str) -> str:
    return {"A": f"all {X} are {Y}", "E": f"no {X} are {Y}", "I": f"some {X} are {Y}", "O": f"some {X} are not {Y}"}[form]


def gen_syllogism(rng, want_valid: bool) -> dict:
    for _ in range(5000):
        f1, f2, fc = rng.choice(FORMS), rng.choice(FORMS), rng.choice(FORMS)
        p1 = (f1, 1, 2) if rng.random() < 0.5 else (f1, 2, 1)  # M-P or P-M
        p2 = (f2, 0, 1) if rng.random() < 0.5 else (f2, 1, 0)  # S-M or M-S
        cc = (fc, 0, 2)
        v1, v2 = syllogism_valid([p1, p2], cc, False), syllogism_valid([p1, p2], cc, True)
        if v1 == v2 == want_valid:
            break
    else:
        raise RuntimeError("syllogism")
    terms = rng.sample(ROLES, 3)
    ctx = rng.choice(SYL_CTX)
    prem = [p1, p2] if rng.random() < 0.5 else [p2, p1]
    s1 = f"{ctx} {stmt_text(prem[0][0], terms[prem[0][1]], terms[prem[0][2]])}."
    s2 = cap(stmt_text(prem[1][0], terms[prem[1][1]], terms[prem[1][2]])) + "."
    concl = stmt_text(cc[0], terms[0], terms[2])
    lead = rng.choice(["Therefore, ", "So, ", "It follows that "])
    text = f"{s1} {s2} {lead}{concl}."
    return {"text": text, "choice": [], "yn": None, "valid": want_valid, "concl": concl,
            "extra": {"forms": [list(p1), list(p2), list(cc)], "terms": terms}}


# Logic: propositional
PROPS = [("it rains", "it does not rain"), ("the match is cancelled", "the match is not cancelled"),
         ("Maya takes the bus", "Maya does not take the bus"), ("the store is open", "the store is not open"),
         ("the alarm rings", "the alarm does not ring"), ("Leo finishes his homework", "Leo does not finish his homework"),
         ("the road is wet", "the road is not wet"), ("the train is late", "the train is not late"),
         ("the lights are on", "the lights are not on"), ("Nina bakes bread", "Nina does not bake bread"),
         ("the window is open", "the window is not open"), ("the plants get water", "the plants do not get water"),
         ("the heater is on", "the heater is not on"), ("Sam goes to the gym", "Sam does not go to the gym"),
         ("the library is quiet", "the library is not quiet"), ("the bus is full", "the bus is not full"),
         ("Ella buys a ticket", "Ella does not buy a ticket"), ("the phone is charged", "the phone is not charged"),
         ("the soup is hot", "the soup is not hot"), ("Tom walks to work", "Tom does not walk to work"),
         ("the garden gate is locked", "the garden gate is not locked"), ("the kettle is boiling", "the kettle is not boiling"),
         ("the meeting starts on time", "the meeting does not start on time"), ("Ruby plays the piano", "Ruby does not play the piano")]
PROP_FORMS = {  # name -> (valid, builder)
    "modus_ponens": True, "modus_tollens": True, "disjunctive_syllogism": True, "chain": True,
    "hypothetical_syllogism": True, "affirming_consequent": False, "denying_antecedent": False,
    "chain_converse": False, "affirming_disjunct": False,
}


def _if(p, q):
    return f"If {p}, then {q}."


def gen_prop(rng, want_valid: bool) -> dict:
    name = rng.choice([k for k, v in PROP_FORMS.items() if v == want_valid])
    p, q, r = rng.sample(PROPS, 3)
    if name == "modus_ponens":
        prem, concl = [_if(p[0], q[0]), cap(p[0]) + "."], q[0]
    elif name == "modus_tollens":
        prem, concl = [_if(p[0], q[0]), cap(q[1]) + "."], p[1]
    elif name == "disjunctive_syllogism":
        prem, concl = [f"Either {p[0]} or {q[0]}, or both.", cap(p[1]) + "."], q[0]
    elif name == "chain":
        prem, concl = [_if(p[0], q[0]), _if(q[0], r[0]), cap(p[0]) + "."], r[0]
    elif name == "hypothetical_syllogism":
        prem, concl = [_if(p[0], q[0]), _if(q[0], r[0])], f"if {p[0]}, then {r[0]}"
    elif name == "affirming_consequent":
        prem, concl = [_if(p[0], q[0]), cap(q[0]) + "."], p[0]
    elif name == "denying_antecedent":
        prem, concl = [_if(p[0], q[0]), cap(p[1]) + "."], q[1]
    elif name == "chain_converse":
        prem, concl = [_if(p[0], q[0]), _if(q[0], r[0]), cap(r[0]) + "."], p[0]
    else:  # affirming a disjunct of an inclusive or ("or both" is said explicitly)
        prem, concl = [f"Either {p[0]} or {q[0]}, or both.", cap(p[0]) + "."], q[1]
    lead = rng.choice(["Therefore, ", "So, ", "It follows that "])
    text = " ".join(prem) + f" {lead}{concl}."
    return {"text": text, "choice": [], "yn": None, "valid": want_valid, "concl": concl, "extra": {"form": name}}


HABITS = [("plays chess", "does not play chess"), ("reads novels", "does not read novels"),
          ("drinks tea", "does not drink tea"), ("rides a bike", "does not ride a bike"),
          ("keeps a diary", "does not keep a diary"), ("grows tomatoes", "does not grow tomatoes"),
          ("sings in a choir", "does not sing in a choir"), ("swims on Sundays", "does not swim on Sundays"),
          ("bakes bread", "does not bake bread"), ("walks to work", "does not walk to work"),
          ("speaks Spanish", "does not speak Spanish"), ("owns a cat", "does not own a cat")]


def gen_conclusion(rng, want_yes: bool) -> dict:
    """Premises spoken; which conclusion follows (exactly one option is entailed)."""
    form = rng.choice(["mp", "mt", "ds", "chain", "universal", "universal_mt"])
    if form.startswith("universal"):
        a, b = rng.sample(HABITS, 2)
        x, y = names(rng, 2)
        if form == "universal":
            prem = [f"Everyone who {a[0]} also {b[0]}.", f"{x} {a[0]}."]
            correct = f"{x} {b[0]}"
            distract = [f"{x} {b[1]}", f"everyone who {b[0]} also {a[0]}", f"{y} {b[0]}"]
        else:
            prem = [f"Everyone who {a[0]} also {b[0]}.", f"{x} {b[1]}."]
            correct = f"{x} {a[1]}"
            distract = [f"{x} {a[0]}", f"{x} {b[0]}", f"{y} {a[1]}"]
    else:
        p, q, r, s = rng.sample(PROPS, 4)
        if form == "mp":
            prem, correct, distract = [_if(p[0], q[0]), cap(p[0]) + "."], q[0], [q[1], p[1], r[0]]
        elif form == "mt":
            prem, correct, distract = [_if(p[0], q[0]), cap(q[1]) + "."], p[1], [p[0], q[0], r[0]]
        elif form == "ds":
            prem, correct, distract = [f"Either {p[0]} or {q[0]}, or both.", cap(p[1]) + "."], q[0], [q[1], p[0], r[0]]
        else:
            prem, correct, distract = [_if(p[0], q[0]), _if(q[0], r[0]), cap(p[0]) + "."], r[0], [r[1], q[1], s[0]]
    opts = [cap(correct)] + [cap(d) for d in distract]
    rng.shuffle(opts)
    stmt = correct if want_yes else rng.choice(distract)
    yq = rng.choice(FOLLOW_YN).format(s=stmt)
    return {"text": " ".join(prem), "choice": [{"q": rng.choice(CONCL_Q), "options": opts, "gold": opts.index(cap(correct)),
                                                "spoken_q": rng.choice(CONCL_SPOKEN_Q), "no_prefix": True}],
            "yn": {"q": yq, "yes": want_yes, "typed_only": True}, "valid": None, "extra": {"form": form}}


# family -> (source, generator, kind, target clips)
GEN_FAMILIES = {
    "order": ("spoken_gen_order", gen_order, "qa", 650),
    "spatial": ("spoken_gen_order", gen_spatial, "qa", 350),
    "temporal": ("spoken_gen_order", gen_temporal, "qa", 450),
    "truth": ("spoken_gen_truth", gen_truth, "qa", 650),
    "facing": ("spoken_gen_nav", gen_facing, "qa", 350),
    "grid": ("spoken_gen_nav", gen_grid, "qa", 450),
    "weekday": ("spoken_gen_time", gen_weekday, "qa", 400),
    "month": ("spoken_gen_time", gen_month, "qa", 180),
    "clock": ("spoken_gen_time", gen_clock, "qa", 250),
    "date": ("spoken_gen_time", gen_date, "qa", 250),
    "arith": ("spoken_gen_math", gen_arith, "qa", 600),
    "count": ("spoken_gen_math", gen_count, "qa", 300),
    "family": ("spoken_gen_family", gen_family, "qa", 450),
    "syllogism": ("spoken_gen_logic", gen_syllogism, "valid", 500),
    "propositional": ("spoken_gen_logic", gen_prop, "valid", 400),
    "conclusion": ("spoken_gen_logic", gen_conclusion, "concl", 550),
}
QA_FORMATS = [("passage", 0.30), ("stem", 0.25), ("read", 0.25), ("yn", 0.20)]
SHORT_FORMATS = [("stem", 0.40), ("read", 0.35), ("yn", 0.25)]  # spoken premise is one short sentence -> no passage
FORMAT_OVERRIDE = {"weekday": SHORT_FORMATS, "month": SHORT_FORMATS}
VALID_FORMATS = [("valid", 0.65), ("valid_yn", 0.20), ("valid_stem", 0.15)]
CONCL_FORMATS = [("passage", 0.50), ("read", 0.25), ("stem", 0.25)]
LETTERS = "ABCDEFGHIJ"


def weighted(rng, pairs):
    x, acc = rng.random(), 0.0
    for k, w in pairs:
        acc += w
        if x < acc:
            return k
    return pairs[-1][0]


def read_text(stem: str, options: list[str], rng, convert: bool = True) -> str:
    """'<stem> A. x. B. y. ...' (misaki reads 'A.' as the letter name)."""
    opt = " ".join(f"{LETTERS[i]}. {spoken_option(o) if convert else o}." for i, o in enumerate(options))
    return f"{stem} {opt}"


def qa_dict(question, options, gold, qtype, shuffle, extra=None, state_text=None):
    return {"question": question, "options": list(options), "gold": gold, "qtype": qtype, "shuffle": shuffle,
            "state_text": state_text, "extra": extra or {}}


def render_generated(fam: str, fmt: str, pz: dict, rng) -> tuple[str, list[dict]] | None:
    """Puzzle + format -> (spoken text, [qa dicts]). None if the spoken text would be too long."""
    text = pz["text"]
    qas = []
    if fmt in ("valid", "valid_yn", "valid_stem"):
        v = pz["valid"]
        if fmt == "valid":
            spoken = text if rng.random() < 0.7 else f"{text} {rng.choice(VALID_TAILS)}"
            qas.append(qa_dict(rng.choice(VALID_Q), ["valid", "invalid"], 0 if v else 1, "choice",
                               shuffle=rng.random() < 0.3, extra={"label": "valid" if v else "invalid"}))
        elif fmt == "valid_yn":
            spoken = text
            qas.append(qa_dict(rng.choice(VALID_YN_Q), YES_NO, 0 if v else 1, "noul", True, {"label": v}))
        else:
            spoken = f"{text} Is this argument valid or invalid?"
            qas.append(qa_dict(rng.choice(STEM_Q), ["valid", "invalid"], 0 if v else 1, "choice", True,
                               {"label": "valid" if v else "invalid"}))
    elif fmt == "passage":
        spoken = text
        cq = pz["choice"][0] if pz["choice"] else None
        if cq is not None:
            qas.append(qa_dict(cq["q"] if cq.get("no_prefix") else passage_q(rng, cq["q"]), cq["options"], cq["gold"],
                               "choice", True))
        yn = pz.get("yn")
        if yn is not None and not yn.get("self_contained"):
            qas.append(qa_dict(yn["q"] if yn.get("typed_only") else passage_q(rng, yn["q"]), YES_NO,
                               0 if yn["yes"] else 1, "noul", True, {"label": yn["yes"]}))
        if len(pz["choice"]) > 1 and len(qas) < 3:
            c2 = pz["choice"][1]
            qas.append(qa_dict(passage_q(rng, c2["q"]), c2["options"], c2["gold"], "choice", True))
    elif fmt == "stem":
        cq = pz["choice"][0]
        spoken = f"{text} {cq.get('spoken_q', cq['q'])}"
        qas.append(qa_dict(rng.choice(STEM_Q), cq["options"], cq["gold"], "choice", True))
    elif fmt == "read":
        cq = pz["choice"][0]
        opts, gold = cq["options"], cq["gold"]
        if len(opts) > 5:
            keep = [gold] + rng.sample([i for i in range(len(opts)) if i != gold], 3)
            keep.sort()
            gold = keep.index(gold)
            opts = [opts[i] for i in keep]
        spoken = read_text(f"{text} {cq.get('spoken_q', cq['q'])}", opts, rng)
        qas.append(qa_dict(rng.choice(READ_Q), opts, gold, "choice", False))
    elif fmt == "yn":
        yn = pz["yn"]
        spoken = f"{text} {yn['q']}"
        qas.append(qa_dict(rng.choice(YN_Q), YES_NO, 0 if yn["yes"] else 1, "noul", True, {"label": yn["yes"]}))
    else:
        raise ValueError(fmt)
    if n_words(spoken) > MAX_TTS_WORDS:
        return None
    return spoken, qas


def generated_specs(scale: float = 1.0) -> tuple[list[dict], Counter]:
    """All generated puzzles -> TTS clip specs (each spec carries its 1-3 items)."""
    stats = Counter()
    specs, seen = [], set()
    for fam, (source, fn, kind, n_target) in GEN_FAMILIES.items():
        n = max(4, int(round(n_target * scale)))
        frng = rng_for("genfam", fam)
        fmts_tab = FORMAT_OVERRIDE.get(fam) or (VALID_FORMATS if kind == "valid" else CONCL_FORMATS if kind == "concl" else QA_FORMATS)
        fmts = [weighted(frng, fmts_tab) for _ in range(n)]
        flags = [False] * n  # yes/no (or valid/invalid) label, balanced 50/50 within every (family, format)
        for f in sorted(set(fmts)):
            idx = [i for i in range(n) if fmts[i] == f]
            for i, fl in zip(idx, balanced_flags(len(idx), rng_for("genflags", fam, f))):
                flags[i] = fl
        made = 0
        for i in range(n):
            fmt, flag = fmts[i], flags[i]
            for attempt in range(60):
                rng = rng_for("gen", fam, i, attempt)
                try:
                    pz = fn(rng, flag)
                except RuntimeError:
                    stats[f"{fam}:gen_retry"] += 1
                    continue
                f = fmt
                if f in ("stem", "read") and not pz["choice"]:
                    f = "yn"  # e.g. grid walks that end at the start have no choice question
                if f == "yn" and pz.get("yn") is None:
                    continue
                r = render_generated(fam, f, pz, rng)
                if r is None and f in ("read", "stem"):
                    f = "passage"
                    r = render_generated(fam, f, pz, rng)
                if r is None:
                    stats[f"{fam}:too_long"] += 1
                    continue
                spoken, qas = r
                if not qas:
                    continue
                key = spoken.lower()
                if key in seen:
                    stats[f"{fam}:dup"] += 1
                    continue
                seen.add(key)
                pid = f"{fam}{i:05d}"
                specs.append({"clip": f"gen/{fam}/{pid}", "source": source, "family": fam, "row_id": pid, "group": pid,
                              "cal": h01("gencal:" + pid) < CAL_FRAC, "variant": f, "licence": "generated (sb_spoken2, CC0)",
                              "dataset": "generated puzzle (exact label by construction)", "tts_text": spoken,
                              "text": spoken, "stem": pz["text"], "qas": qas, "puzzle_extra": pz.get("extra", {})})
                made += 1
                stats[f"{fam}:{f}"] += 1
                break
        stats[f"{fam}:made"] = made
    return specs, stats


# Dataset rows -> specs (CSQA, QASC, QuaRTz)
YN_SPOKEN = ["{q} Is the answer {c}?", "{q} Would the answer be {c}?", "{q} Is it {c}?",
             "{q} Is {c} the right answer?", "{q} Could the answer be {c}?"]


def clean_opt(o: str) -> str:
    o = re.sub(r"\s+", " ", o).strip()
    return o[:-1].strip() if o.endswith(".") else o


def speak_blanks(q: str) -> str:
    return re.sub(r"_{2,}", "blank", q)


def dataset_specs(rows: list[dict], *, ds: str, source: str, licence: str, dataset: str, n_max: int,
                  fmt_weights: list[tuple[str, float]], read_k: int | None, exclude: set = frozenset(),
                  group_of=None) -> tuple[list[dict], Counter]:
    """Generic MCQ rows ({id, q, choices, gold}) -> specs. read_k: options read aloud (None = all, else gold + k-1
    random others, spoken order kept); stem uses all options (shuffled); yn asks about gold or a random wrong option."""
    stats = Counter()
    good = []
    for r in rows:
        if r["id"] in exclude:
            stats["excluded_v1_row"] += 1
            continue
        ch = [clean_opt(c) for c in r["choices"]]
        if len(ch) < 2 or len(set(c.lower() for c in ch)) != len(ch) or any(not c for c in ch) or max(len(c) for c in ch) > 90:
            stats["bad_row"] += 1
            continue
        q = re.sub(r"\s+", " ", r["q"]).strip()
        if n_words(q) > 36:
            stats["too_long"] += 1
            continue
        good.append({**r, "q": q, "choices": ch})
    good.sort(key=lambda g: h01(f"{ds}:" + g["id"]))
    good = good[:n_max]
    fr = rng_for(ds, "fmts")
    fmts = [weighted(fr, fmt_weights) for _ in good]
    yn_idx = [i for i, f in enumerate(fmts) if f == "yn"]
    yflags = dict(zip(yn_idx, balanced_flags(len(yn_idx), rng_for(ds, "ynflags"))))
    specs = []
    for i, (g, fmt) in enumerate(zip(good, fmts)):
        rng = rng_for(ds, g["id"], fmt)
        grp = group_of(g) if group_of else g["id"]
        q, ch, gold = g["q"], g["choices"], g["gold"]
        spoken_q = speak_blanks(q)
        extra = {}
        if fmt == "read":
            idx = list(range(len(ch)))
            if read_k and len(ch) > read_k:
                idx = sorted([gold] + rng.sample([j for j in idx if j != gold], read_k - 1))
            opts = [ch[j] for j in idx]
            gi = idx.index(gold)
            spoken = read_text(spoken_q, opts, rng, convert=False)
            if n_words(spoken) > MAX_TTS_WORDS:
                fmt = "stem"
            else:
                qas = [qa_dict(rng.choice(READ_Q), opts, gi, "choice", False)]
        if fmt == "stem":
            spoken = spoken_q
            qas = [qa_dict(rng.choice(STEM_Q), ch, gold, "choice", True)]
        elif fmt == "yn":
            yes = yflags[i]
            asked = ch[gold] if yes else rng.choice([c for j, c in enumerate(ch) if j != gold])
            spoken = rng.choice(YN_SPOKEN).format(q=spoken_q, c=asked)
            extra = {"asked_option": asked, "label": yes}
            qas = [qa_dict(rng.choice(YN_Q), YES_NO, 0 if yes else 1, "noul", True, extra)]
        if n_words(spoken) > MAX_TTS_WORDS + 4:
            stats["too_long_spoken"] += 1
            continue
        stats[f"fmt:{fmt}"] += 1
        specs.append({"clip": f"{ds}/{g['id']}", "source": source, "family": ds, "row_id": g["id"], "group": grp,
                      "cal": h01(f"{ds}cal:" + str(grp)) < CAL_FRAC, "variant": fmt, "licence": licence,
                      "dataset": dataset + (f" {g['split']}" if g.get("split") else ""), "tts_text": spoken,
                      "text": spoken, "stem": q, "qas": qas, "puzzle_extra": {}})
    stats["specs"] = len(specs)
    return specs, stats


def csqa_rows(raw: list[dict]) -> list[dict]:
    out = []
    for r in raw:
        labels = list(r["choices"]["label"])
        if r["answerKey"] not in labels:
            continue
        out.append({"id": r["id"], "q": r["question"], "choices": list(r["choices"]["text"]),
                    "gold": labels.index(r["answerKey"]), "split": "train"})
    return out


def qasc_rows(raw: list[dict], split: str) -> list[dict]:
    out = []
    for r in raw:
        labels = list(r["choices"]["label"])
        if not r.get("answerKey") or r["answerKey"] not in labels:
            continue
        out.append({"id": r["id"], "q": r["question"], "choices": list(r["choices"]["text"]),
                    "gold": labels.index(r["answerKey"]), "fact1": r.get("fact1"), "split": split})
    return out


def quartz_rows(raw: list[dict], split: str) -> list[dict]:
    out = []
    for r in raw:
        labels = list(r["choices"]["label"])
        if not r.get("answerKey") or r["answerKey"] not in labels:
            continue
        out.append({"id": r["id"], "q": r["question"], "choices": list(r["choices"]["text"]),
                    "gold": labels.index(r["answerKey"]), "para_id": r.get("para_id"), "split": split})
    return out


# Voices
def english_voices(voice_wer: dict, max_wer: float = 0.06) -> list[str]:
    return sorted(v for v, w in voice_wer.items() if v[:3] in ("af_", "am_", "bf_", "bm_") and w <= max_wer)


def split_voices(voices: list[str]) -> tuple[list[str], list[str]]:
    """One held-out (CAL) voice per accent/gender group (same hash as sb_spoken, so the same 4 CAL voices)."""
    cal = []
    for g in ("af_", "am_", "bf_", "bm_"):
        grp = sorted((v for v in voices if v.startswith(g)), key=lambda v: h01("voicecal:" + v))
        if len(grp) > 1:
            cal.append(grp[0])
    return [v for v in voices if v not in cal], cal


def assign_voices(specs: list[dict], train_voices: list[str], cal_voices: list[str]) -> None:
    for s in specs:
        rng = rng_for("voice2", s["clip"])
        s["voice"] = rng.choice(cal_voices if s["cal"] else train_voices)
        s["speed"] = round(rng.uniform(0.9, 1.1), 3)


# Items
def make_items(spec: dict, bank: str, duration_s: float) -> list[dict]:
    out = []
    for k, qa in enumerate(spec["qas"][:MAX_ITEMS_PER_CLIP]):
        iid = f"{BUILDER}/{spec['source']}/{spec['clip']}/{spec['variant']}/{k}"
        rng = rng_for("item", iid)
        meta = {"transcript": spec["text"], "voice": spec["voice"], "speed": spec["speed"], "variant": spec["variant"],
                "family": spec["family"], "row_id": spec["row_id"], "group": spec["group"], "licence": spec["licence"],
                "dataset": spec["dataset"], "duration_s": round(duration_s, 3), "tts": "Kokoro-82M", "qtype": qa["qtype"],
                **qa["extra"]}
        if spec.get("puzzle_extra"):
            meta["puzzle"] = spec["puzzle_extra"]
        target = [0.0] * len(qa["options"])
        target[qa["gold"]] = 1.0
        channel = "phone" if h01("phone:" + iid) < PHONE_FRAC else "clean"
        out.append(C.make_item(item_id=iid, set_name="CAL" if spec["cal"] else "train", source=spec["source"],
                               question=qa["question"], options=list(qa["options"]), target=target, gold=qa["gold"],
                               rng=rng if qa["shuffle"] else NO_SHUFFLE, audio={"bank": bank, "key": spec["clip"]},
                               channel=channel, state_text=qa["state_text"], meta=meta))
    return out


# Leakage helpers + validation
def norm_text(t: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", (t or "").lower()).split())


def shingles(t: str, n: int = 8) -> set:
    w = norm_text(t).split()
    if len(w) < n:
        return {" ".join(w)} if len(w) >= 5 else set()
    return {" ".join(w[i : i + n]) for i in range(len(w) - n + 1)}


KEYS = {"id", "set", "source", "question", "options", "target", "gold", "audio", "channel", "state_text", "meta"}


def validate(items: list[dict], bank_index: dict[str, dict]) -> dict:
    """bank_index: bank -> {key: [start, length]}. Returns problems + balance stats."""
    probs = Counter()
    ids = Counter(it["id"] for it in items)
    probs["duplicate_ids"] = sum(v - 1 for v in ids.values() if v > 1)
    yes = defaultdict(lambda: [0, 0])
    valid_bal = defaultdict(lambda: [0, 0])
    gold_pos = defaultdict(Counter)
    durs = []
    per_clip = Counter()
    for it in items:
        if set(it) != KEYS:
            probs["schema"] += 1
        n = len(it["options"])
        if not 2 <= n <= 20 or len(it["target"]) != n:
            probs["n_options"] += 1
        if abs(sum(it["target"]) - 1) > 1e-6:
            probs["target_sum"] += 1
        if max(it["target"]) == 1.0 and it["target"].index(1.0) != it["gold"]:
            probs["gold_argmax"] += 1
        if len(set(o.lower() for o in it["options"])) != n:
            probs["dup_options"] += 1
        if any(not str(o).strip() for o in it["options"]):
            probs["empty_option"] += 1
        q = it["meta"].get("qtype")
        if q not in ("choice", "noul", "score"):
            probs["qtype"] += 1
        if it["channel"] not in ("clean", "phone"):
            probs["channel"] += 1
        if q == "noul":
            if sorted(it["options"]) != ["no", "yes"]:
                probs["noul_options"] += 1
            y = it["options"][it["gold"]] == "yes"
            yes[(it["source"], it["meta"].get("family"))][0 if y else 1] += 1
        else:
            gold_pos[it["source"]][it["gold"]] += 1
        if it["options"] in (["valid", "invalid"], ["invalid", "valid"]):
            valid_bal[it["source"]][0 if it["options"][it["gold"]] == "valid" else 1] += 1
        a = it["audio"]
        if a is None or a["bank"] not in bank_index or a["key"] not in bank_index[a["bank"]]:
            probs["audio_missing"] += 1
            continue
        per_clip[(a["bank"], a["key"])] += 1
        L = bank_index[a["bank"]][a["key"]][1] / C.SAMPLE_RATE
        durs.append(L)
        if L > MAX_CLIP_S + 1e-6 or L <= 0.3:
            probs["duration"] += 1
        if it["set"] not in ("train", "CAL"):
            probs["set"] += 1
    probs["clips_over_cap"] = sum(1 for v in per_clip.values() if v > MAX_ITEMS_PER_CLIP)
    return {"problems": {k: v for k, v in probs.items() if v}, "n_items": len(items),
            "yes_no_balance": {f"{s}|{k}": {"yes": v[0], "no": v[1]} for (s, k), v in sorted(yes.items())},
            "valid_invalid_balance": {s: {"valid": v[0], "invalid": v[1]} for s, v in sorted(valid_bal.items())},
            "gold_position_choice": {s: dict(sorted(v.items())) for s, v in sorted(gold_pos.items())},
            "clip_duration_s": {"max": round(max(durs), 2) if durs else 0, "mean": round(sum(durs) / max(1, len(durs)), 2),
                                "min": round(min(durs), 2) if durs else 0},
            "max_items_per_clip": max(per_clip.values()) if per_clip else 0}
