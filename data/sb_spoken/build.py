"""sb_spoken item logic: HarperValleyBank intents and dialog acts, spoken questions read by Kokoro TTS, and listening
checks on those clips. Modal plumbing is in modal_build.py.
"""
from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict

import common as C

BUILDER = "sb_spoken"
MAX_CLIP_S = 20.0
CAL_FRAC = 0.03
MAX_ITEMS_PER_CLIP = 4
YES_NO = ["yes", "no"]


class _NoShuffle:
    def shuffle(self, x):  # keeps canonical order (spoken order / score order)
        return None


NO_SHUFFLE = _NoShuffle()


def h01(s: str) -> float:
    """Deterministic hash of a string -> [0, 1)."""
    return int(hashlib.sha1(s.encode()).hexdigest()[:12], 16) / float(16 ** 12)


def rng_for(*parts) -> random.Random:
    return random.Random(f"{C.SEED}:" + ":".join(str(p) for p in parts))


def item(*, iid, cal, source, question, options, gold, rng, audio, qtype, meta, state_text=None, shuffle=True):
    target = [0.0] * len(options)
    target[gold] = 1.0
    m = dict(meta)
    m["qtype"] = qtype
    return C.make_item(item_id=f"{BUILDER}/{iid}", set_name="CAL" if cal else "train", source=source,
                       question=question, options=list(options), target=target, gold=gold,
                       rng=rng if shuffle else NO_SHUFFLE, audio=audio, channel="clean", state_text=state_text, meta=m)


# HarperValleyBank
HVB_INTENTS = {  # all 8 always offered
    "check balance": "check their account balance",
    "get branch hours": "find out the branch opening hours",
    "order checks": "order new checks",
    "pay bill": "pay a bill",
    "replace card": "replace their card",
    "reset password": "reset their password",
    "schedule appointment": "schedule an appointment",
    "transfer money": "transfer money between accounts",
}
HVB_DIALOG_ACTS = {  # act -> (choice option text, yes/no predicate in base form); all 18 always offered
    "acknowledge": ("acknowledges what the other person said", "acknowledge what the other person said"),
    "answer_agree": ("agrees or says yes", "agree or say yes"),
    "answer_dis": ("disagrees or says no", "disagree or say no"),
    "answer_general": ("answers a question or gives requested information", "answer a question or give requested information"),
    "apology": ("apologizes", "apologize"),
    "backchannel": ("makes a backchannel sound (uh-huh, mm-hmm)", "make a backchannel sound such as uh-huh or mm-hmm"),
    "disfluency": ("produces a disfluency or false start", "produce a disfluency or false start"),
    "other": ("something else", None),
    "question_check": ("asks a question to check or confirm information", "ask a question to check or confirm information"),
    "question_general": ("asks a general question", "ask a general question"),
    "question_repeat": ("asks the other person to repeat", "ask the other person to repeat something"),
    "self": ("talks to themselves", "talk to themselves"),
    "statement_close": ("closes the conversation (goodbye)", "close the conversation, for example by saying goodbye"),
    "statement_general": ("makes a general statement", "make a general statement"),
    "statement_instruct": ("gives an instruction", "give an instruction"),
    "statement_open": ("opens the conversation (greeting or introduction)", "open the conversation with a greeting or introduction"),
    "statement_problem": ("states the problem or the reason for the call", "state the problem or the reason for the call"),
    "thanks": ("thanks the other person", "thank the other person"),
}
HVB_MIN_MS = 300
HVB_INTENT_Q = [
    "What does the caller want to do?",
    "Why is this customer calling the bank?",
    "What is the caller's reason for calling?",
    "Which request is the caller making?",
    "What does the customer need help with?",
    "Based on what the caller says, what do they want?",
]
HVB_INTENT_YN = [
    "Does the caller want to {x}?",
    "Is the caller asking to {x}?",
    "Is the customer calling to {x}?",
    "Is the caller trying to {x}?",
    "Does the person calling the bank want to {x}?",
]
HVB_DA_Q = [
    "In this turn of a bank customer-service phone call, what is the speaker doing?",
    "What is the speaker doing in this turn of the call?",
    "Which best describes what the speaker does in this phone-call turn?",
    "This is one turn from a call to a bank. What is its function?",
    "What kind of conversational move does the speaker make here?",
    "Listen to this turn of the conversation. What is the speaker doing?",
]
HVB_DA_YN = [
    "Does the speaker {p}?",
    "In this turn of the call, does the speaker {p}?",
    "Listen to this turn from a bank call. Does the speaker {p}?",
    "In this clip, does the person speaking {p}?",
    "Is the speaker trying to {p} in this turn?",
]
HVB_DA_CAP_PER_ACT = 450      # choice items per act (balance the long tail)
HVB_DA_NOUL_TARGET = 2600     # yes/no dialog-act items (50/50)
HVB_PROBLEM_PER_CALL = 3      # intent items: caller 'statement_problem' turns per call


def hvb_cal_speakers(utts_by_speaker: dict[str, int], adi_speakers: set, frac: float = CAL_FRAC) -> set:
    """Hold out speakers for CAL until >= frac of utterances. Speakers that also occur in our evaluation set go first
    (smallest first, then hash), so the evaluation-free speakers stay in train."""
    total = sum(utts_by_speaker.values())
    order = sorted(utts_by_speaker, key=lambda s: (s not in adi_speakers, utts_by_speaker[s], h01(f"hvbcal:{s}")))
    chosen, n = set(), 0
    for s in order:
        if n >= frac * total:
            break
        chosen.add(s)
        n += utts_by_speaker[s]
    return chosen


def hvb_key(r) -> str:
    return f"{r['issue_id']}_{int(r['utt_index']):03d}"


def build_hvb_items(rows: list[dict], durations: dict[str, float], adi_speakers: set) -> tuple[list[dict], dict]:
    """rows: SLUE-HVB train rows (no audio); durations: key -> seconds for rows whose audio is in the bank."""
    rows = [r for r in rows if hvb_key(r) in durations]
    spk_count = Counter(str(r["speaker_id"]) for r in rows)
    cal_spk = hvb_cal_speakers(spk_count, adi_speakers)
    call_spk = defaultdict(set)
    for r in rows:
        call_spk[r["issue_id"]].add(str(r["speaker_id"]))
    cal_calls = {c for c, s in call_spk.items() if s & cal_spk}

    def split_of(r):
        spk = str(r["speaker_id"])
        if spk in cal_spk:
            return "CAL"
        if r["issue_id"] in cal_calls:
            return None  # other speaker of a CAL call: dropped (recording-disjoint CAL)
        return "train"

    stats = Counter()
    keep = []
    for r in rows:
        s = split_of(r)
        if s is None:
            stats["dropped_cal_call_partner"] += 1
            continue
        if int(r["duration_ms"]) < HVB_MIN_MS:
            stats["too_short"] += 1
            continue
        keep.append((r, s))
    per_clip = Counter()
    items = []

    def meta_of(r):
        return {"call": r["issue_id"], "speaker": str(r["speaker_id"]), "role": r["role"], "intent": r["intent"],
                "dialog_acts": list(r["dialog_acts"]), "utt_index": int(r["utt_index"]), "transcript": r["text"],
                "duration_s": round(durations[hvb_key(r)], 3), "licence": "CC BY 4.0",
                "adi_speaker_overlap": str(r["speaker_id"]) in adi_speakers, "dataset": "asapp/slue-phase-2 hvb train"}

    def add(it, r):
        per_clip[hvb_key(r)] += 1
        items.append(it)

    # ---- intent: caller turns that state the problem (up to 3 per call, longest first)
    intent_names = sorted(HVB_INTENTS)
    by_call = defaultdict(list)
    for r, s in keep:
        if r["role"] == "caller" and "statement_problem" in r["dialog_acts"] and r["intent"] in HVB_INTENTS:
            by_call[r["issue_id"]].append((r, s))
    problem = []
    for c in sorted(by_call):
        problem += sorted(by_call[c], key=lambda t: -int(t[0]["duration_ms"]))[:HVB_PROBLEM_PER_CALL]
    flags = balanced_flags(len(problem), rng_for("hvb_intent_yn_flags"))
    for (r, s), yes in zip(problem, flags):
        k = hvb_key(r)
        rng = rng_for("hvb_intent", k)
        gold = intent_names.index(r["intent"])
        add(item(iid=f"hvb_intent/{k}/choice", cal=s == "CAL", source="hvb_intent", question=rng.choice(HVB_INTENT_Q),
                 options=[HVB_INTENTS[n] for n in intent_names], gold=gold, rng=rng,
                 audio={"bank": "sb_spoken_hvb", "key": k}, qtype="choice", meta=meta_of(r)), r)
        asked = r["intent"] if yes else rng.choice([n for n in intent_names if n != r["intent"]])
        m = meta_of(r)
        m["asked_intent"] = asked
        add(item(iid=f"hvb_intent/{k}/yn", cal=s == "CAL", source="hvb_intent",
                 question=rng.choice(HVB_INTENT_YN).format(x=HVB_INTENTS[asked]), options=YES_NO, gold=0 if yes else 1,
                 rng=rng, audio={"bank": "sb_spoken_hvb", "key": k}, qtype="noul", meta=m), r)
    stats["intent_turns"] = len(problem)

    # ---- dialog act, choice: single-act turns, capped per act
    da_names = sorted(HVB_DIALOG_ACTS)
    single = defaultdict(list)
    for r, s in keep:
        if len(r["dialog_acts"]) == 1 and r["dialog_acts"][0] in HVB_DIALOG_ACTS:
            single[r["dialog_acts"][0]].append((r, s))
    for act in da_names:
        pool = sorted(single[act], key=lambda t: h01("da:" + hvb_key(t[0])))[:HVB_DA_CAP_PER_ACT]
        for r, s in pool:
            k = hvb_key(r)
            rng = rng_for("hvb_da", k)
            add(item(iid=f"hvb_dialog_act/{k}/choice", cal=s == "CAL", source="hvb_dialog_act",
                     question=rng.choice(HVB_DA_Q), options=[HVB_DIALOG_ACTS[n][0] for n in da_names],
                     gold=da_names.index(act), rng=rng, audio={"bank": "sb_spoken_hvb", "key": k}, qtype="choice",
                     meta=meta_of(r)), r)
        stats[f"da_choice:{act}"] = len(pool)

    # ---- dialog act, yes/no: any turn whose acts are all known; asked act spread over acts, 50/50 yes/no
    askable = [a for a in da_names if HVB_DIALOG_ACTS[a][1]]
    cands = [(r, s) for r, s in keep if r["dialog_acts"] and all(a in HVB_DIALOG_ACTS for a in r["dialog_acts"])
             and per_clip[hvb_key(r)] < MAX_ITEMS_PER_CLIP]
    cands.sort(key=lambda t: h01("dayn:" + hvb_key(t[0])))
    yes_pool = defaultdict(list)  # act -> turns that have it
    for r, s in cands:
        for a in r["dialog_acts"]:
            if a in askable:
                yes_pool[a].append((r, s))
    used = set()
    n_half = min(HVB_DA_NOUL_TARGET // 2, len(cands) // 2)
    # yes items: round-robin over acts so rare acts are asked as often as possible
    yes_items = []
    ptr = {a: 0 for a in askable}
    while len(yes_items) < n_half and any(ptr[a] < len(yes_pool[a]) for a in askable):
        for a in askable:
            while ptr[a] < len(yes_pool[a]) and hvb_key(yes_pool[a][ptr[a]][0]) in used:
                ptr[a] += 1
            if ptr[a] < len(yes_pool[a]) and len(yes_items) < n_half:
                yes_items.append((yes_pool[a][ptr[a]], a))
                used.add(hvb_key(yes_pool[a][ptr[a]][0]))
                ptr[a] += 1
    no_items = []
    rest = [t for t in cands if hvb_key(t[0]) not in used]
    for i, (r, s) in enumerate(rest[: len(yes_items)]):
        absent = [a for a in askable if a not in r["dialog_acts"]]
        # spread the asked acts: cycle through acts, fall back to random absent act
        a = askable[i % len(askable)]
        if a not in absent:
            a = rng_for("dayn_no", hvb_key(r)).choice(absent)
        no_items.append(((r, s), a))
    yes_items = yes_items[: len(no_items)]
    for ((r, s), a), yes in [(t, True) for t in yes_items] + [(t, False) for t in no_items]:
        k = hvb_key(r)
        rng = rng_for("hvb_da_yn", k)
        m = meta_of(r)
        m["asked_act"] = a
        add(item(iid=f"hvb_dialog_act/{k}/yn", cal=s == "CAL", source="hvb_dialog_act",
                 question=rng.choice(HVB_DA_YN).format(p=HVB_DIALOG_ACTS[a][1]), options=YES_NO, gold=0 if yes else 1,
                 rng=rng, audio={"bank": "sb_spoken_hvb", "key": k}, qtype="noul", meta=m), r)
    stats["da_yn_yes"], stats["da_yn_no"] = len(yes_items), len(no_items)
    stats["cal_speakers"] = sorted(cal_spk)
    stats["max_items_per_clip"] = max(per_clip.values()) if per_clip else 0
    return items, dict(stats)


def balanced_flags(n: int, rng: random.Random) -> list[bool]:
    f = [i % 2 == 0 for i in range(n)]
    rng.shuffle(f)
    return f


# Spoken questions: TTS scripts
MAX_TTS_WORDS = 45

QA_READ_Q = [  # options are read aloud too; option order = spoken order (VoiceBench format)
    "The audio is a spoken multiple-choice question. Which option is the correct answer?",
    "Listen to the spoken multiple-choice question. Which option is correct?",
    "The speaker reads a question and its answer choices. Which choice is right?",
    "Which of the options read aloud correctly answers the spoken question?",
    "Answer the multiple-choice question in the recording.",
    "The recording contains a quiz question with lettered options. Pick the correct option.",
]
QA_STEM_Q = [  # only the question is spoken; options are text only
    "Answer the question asked in the audio.",
    "Which option answers what the speaker asked?",
    "What is the best answer to the spoken question?",
    "The speaker asks a question. Which option is the best answer?",
    "Pick the best answer to the question in the recording.",
    "Respond to the question you hear: which option fits best?",
]
QA_YN_Q = [  # spoken yes/no question (Big Bench Audio format)
    "Answer the spoken yes/no question.",
    "The speaker asks a yes-or-no question. What is the right answer?",
    "Answer the question in the audio with yes or no.",
    "What is the correct answer to the yes/no question in the recording?",
    "Listen to the question and answer it: yes or no?",
]
QA_YN_SPOKEN = ["{q} Is the answer {c}?", "{q} Would the answer be {c}?", "{q} Is it {c}?",
                "{q} Is {c} the right answer?", "{q} Could the answer be {c}?"]
PASSAGE_Q = [  # JADE state read aloud, JADE question typed
    "{q}",
    "Based on what the speaker says: {q}",
    "Listen to the recording. {q}",
    "From the audio: {q}",
    "The speaker describes a situation. {q}",
    "Using only what is said in the audio, answer: {q}",
]


def n_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9$%']+", text))


def speakable(text: str) -> str:
    """Light normalisation for TTS: newlines -> spaces, e-mail/URL punctuation spelled out."""
    def fix(tok: str) -> str:
        if "@" in tok or "/" in tok or re.fullmatch(r"[A-Za-z][\w-]*\.(example|com|net|org)\S*", tok):
            tail = ""
            if tok[-1] in ".,;:":
                tok, tail = tok[:-1], tok[-1]
            tok = tok.replace("@", " at ").replace("/", " slash ").replace(".", " dot ").replace("-", " ")
            return re.sub(r"\s+", " ", tok).strip() + tail
        return tok
    text = text.replace("\n", ". ").replace("..", ".")
    return " ".join(fix(t) for t in text.split())


def csqa_read_texts(q: str, choices: list[str], option_word: bool = False) -> tuple[str, str]:
    """(tts_text, transcript) for a question with lettered options. misaki reads "A." as the letter name (checked with
    Whisper), so no phoneme overrides are needed."""
    letters = "ABCDE"[: len(choices)]
    if option_word:
        t = q + " " + " ".join(f"Option {l}: {c}." for l, c in zip(letters, choices))
    else:
        t = q + " " + " ".join(f"{l}. {c}." for l, c in zip(letters, choices))
    return t, t


def csqa_specs(rows: list[dict], n_read: int, n_stem: int, n_yn: int) -> tuple[list[dict], Counter]:
    """CommonsenseQA train rows -> TTS clip specs (each with its main QA item)."""
    stats = Counter()
    good = []
    for r in rows:
        labels, texts = list(r["choices"]["label"]), [t.strip() for t in r["choices"]["text"]]
        q = r["question"].strip()
        if labels != list("ABCDE") or r["answerKey"] not in labels or len(set(t.lower() for t in texts)) != 5:
            stats["bad_row"] += 1
            continue
        if n_words(q) > 35:
            stats["too_long"] += 1
            continue
        good.append({"id": r["id"], "q": q, "choices": texts, "gold": labels.index(r["answerKey"]),
                     "concept": r.get("question_concept")})
    good.sort(key=lambda g: h01("csqa:" + g["id"]))
    specs = []
    plan = ["read"] * n_read + ["stem"] * n_stem + ["yn"] * n_yn
    yn_flags = balanced_flags(n_yn, rng_for("csqa_yn_flags"))
    yi = 0
    for g, variant in zip(good, plan):
        rng = rng_for("csqa", g["id"], variant)
        cal = h01("csqacal:" + g["id"]) < CAL_FRAC
        base = {"clip": f"csqa/{g['id']}", "source": "spoken_qa_csqa", "row_id": g["id"], "cal": cal, "variant": variant,
                "licence": "MIT", "dataset": "tau/commonsense_qa train"}
        if variant == "read":
            tts, plain = csqa_read_texts(g["q"], g["choices"], option_word=rng.random() < 0.3)
            if n_words(plain) > MAX_TTS_WORDS:
                stats["read_too_long"] += 1
                continue
            qa = {"question": rng.choice(QA_READ_Q), "options": g["choices"], "gold": g["gold"], "qtype": "choice",
                  "shuffle": False, "state_text": None}
        elif variant == "stem":
            tts = plain = g["q"]
            qa = {"question": rng.choice(QA_STEM_Q), "options": g["choices"], "gold": g["gold"], "qtype": "choice",
                  "shuffle": True, "state_text": None}
        else:
            yes = yn_flags[yi]
            yi += 1
            asked = g["choices"][g["gold"]] if yes else rng.choice([c for i, c in enumerate(g["choices"]) if i != g["gold"]])
            tts = plain = rng.choice(QA_YN_SPOKEN).format(q=g["q"], c=asked)
            qa = {"question": rng.choice(QA_YN_Q), "options": YES_NO, "gold": 0 if yes else 1, "qtype": "noul",
                  "shuffle": True, "state_text": None, "asked": asked}
        specs.append({**base, "tts_text": tts, "text": plain, "stem": g["q"], "qa": qa})
    stats["specs"] = len(specs)
    return specs, stats


JADE_SUITES = {  # suite -> (max rows, question may be spoken too)
    "retrieval.intent": (1300, False),
    "language.aspects": (700, True),
    "language.pragmatics": (800, True),
    "language.security": (600, False),
    "knowledge.math": (450, True),
}


def jade_options(row: dict) -> tuple[list[str], int] | None:
    q = row["question"]
    crit = q.get("criteria") or {}
    keys = sorted(crit)
    if q.get("type") != "choice" or not 2 <= len(keys) <= 20 or not isinstance(row.get("target"), str) or row["target"] not in crit:
        return None
    opts = [str(crit[k]).strip() for k in keys]
    if len(set(o.lower() for o in opts)) != len(opts) or max(len(o) for o in opts) > 80:
        return None
    return opts, keys.index(row["target"])


def jade_specs(rows: list[dict], exclude_ids: set) -> tuple[list[dict], Counter]:
    """JADE-Data rows (chosen suites, string states) -> TTS clip specs. One row per family; rows used by the sb_jade
    text builder are skipped."""
    stats = Counter()
    by_suite = defaultdict(dict)
    for r in rows:
        suite = r.get("suite")
        if suite not in JADE_SUITES or not isinstance(r.get("state"), str):
            continue
        fam = r.get("family") or r["id"]
        if r["id"] in exclude_ids:
            stats["sb_jade_row_skipped"] += 1
            continue
        if fam in by_suite[suite]:
            continue
        by_suite[suite][fam] = r
    specs = []
    for suite, (cap, spoken_q_ok) in JADE_SUITES.items():
        fams = sorted(by_suite[suite], key=lambda f: h01("jade:" + f))
        n = 0
        for fam in fams:
            if n >= cap:
                break
            r = by_suite[suite][fam]
            oo = jade_options(r)
            if oo is None:
                stats[f"bad_options:{suite}"] += 1
                continue
            opts, gold = oo
            state = speakable(r["state"].strip())
            jq = r["question"]["instructions"].strip()
            rng = rng_for("jade", r["id"])
            variant = "spoken_q" if spoken_q_ok and n % 2 == 1 else "passage"
            text = state + " " + jq if variant == "spoken_q" else state
            if n_words(text) > MAX_TTS_WORDS:
                stats[f"too_long:{suite}"] += 1
                continue
            if variant == "spoken_q":
                qa = {"question": rng.choice(QA_STEM_Q), "options": opts, "gold": gold, "qtype": "choice",
                      "shuffle": True, "state_text": None}
            else:
                qa = {"question": rng.choice(PASSAGE_Q).format(q=jq), "options": opts, "gold": gold, "qtype": "choice",
                      "shuffle": True, "state_text": None}
            specs.append({"clip": f"jade/{r['id']}", "source": "spoken_qa_jade", "row_id": r["id"], "family": fam,
                          "suite": suite, "cal": h01("jadecal:" + fam) < CAL_FRAC, "variant": variant,
                          "licence": "Apache-2.0", "dataset": "theunnecessarythings/JADE-Data train.jsonl",
                          "tts_text": text, "text": text, "stem": state, "qa": qa})
            n += 1
        stats[f"suite:{suite}"] = n
    return specs, stats


# Voices
def english_voices(voice_wer: dict, max_wer: float = 0.06) -> list[str]:
    return sorted(v for v, w in voice_wer.items() if v[:3] in ("af_", "am_", "bf_", "bm_") and w <= max_wer)


def split_voices(voices: list[str]) -> tuple[list[str], list[str]]:
    """One held-out (CAL) voice per accent/gender group; the rest are train voices."""
    cal = []
    for g in ("af_", "am_", "bf_", "bm_"):
        grp = sorted((v for v in voices if v.startswith(g)), key=lambda v: h01("voicecal:" + v))
        if len(grp) > 1:
            cal.append(grp[0])
    return [v for v in voices if v not in cal], cal


def assign_voices(specs: list[dict], train_voices: list[str], cal_voices: list[str]) -> None:
    for s in specs:
        rng = rng_for("voice", s["clip"])
        s["voice"] = rng.choice(cal_voices if s["cal"] else train_voices)
        s["speed"] = round(rng.uniform(0.88, 1.12), 3)


# Listening checks (labels from the synthesised text)
NUMBER_WORDS = {"two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
                "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty",
                "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "hundreds", "thousand", "thousands",
                "million", "millions", "billion", "dozen", "dozens", "zero"}
AMBIGUOUS_NUMBER_WORDS = {"one", "ones", "first", "second", "seconds", "third", "fourth", "fifth", "once", "twice",
                          "half", "single", "pair", "couple", "double", "triple", "none", "nine-to-five", "tenth",
                          "quarter", "many", "few", "several", "number", "numbers", "dozenth"}


def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def number_label(text: str):
    """True = a number is said; False = clearly none; None = ambiguous (skip)."""
    if re.search(r"\d", text):
        return True
    w = set(words(text))
    if w & NUMBER_WORDS:
        return True
    if w & AMBIGUOUS_NUMBER_WORDS:
        return None
    return False


def question_label(text: str):
    """True = the speaker asks a question (a '?' sentence); False = statements / requests only."""
    if "?" in text:
        return True
    first = words(text)[:1]
    if first and first[0] in {"what", "which", "who", "whom", "whose", "where", "when", "why", "how", "is", "are",
                              "do", "does", "did", "can", "could", "would", "will", "should"}:
        return None
    return False


STOPWORDS = set("""a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each else ever every few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just let like made make many may
me might more most much must my myself near need no nor not now of off often on once only or other ought our ours
ourselves out over own per please put really same say says said see she should so some such than that the their theirs
them themselves then there these they this those though through thus to too under until up upon us use used very want
was way we well were what when where which while who whom whose why will with within without would yet you your yours
yourself yourselves answer option options question questions right correct thing things something someone
person people place what's it's that's there's""".split())


def content_words(text: str) -> list[str]:
    seen, out = set(), []
    for w in words(text):
        if 4 <= len(w) <= 12 and w.isalpha() and w not in STOPWORDS and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def related(a: str, b: str) -> bool:
    if a == b or a in b or b in a:
        return True
    n = min(5, len(a), len(b))
    return a[:n] == b[:n]


def said_word(text: str, rng: random.Random) -> str | None:
    cw = content_words(text)
    return rng.choice(cw) if cw else None


def distractor_words(text: str, vocab: list[str], rng: random.Random, n: int, avoid=()) -> list[str] | None:
    tw = set(words(text)) | set(avoid)
    out = []
    for _ in range(60 * n):
        if len(out) == n:
            break
        d = rng.choice(vocab)
        if any(related(d, t) for t in tw) or any(related(d, o) for o in out):
            continue
        out.append(d)
    return out if len(out) == n else None


NUM_Q = ["Does the speaker mention a number?", "Is any number said in the recording?",
         "Does the speaker say a number at any point?", "Is there a number in what the speaker says?",
         "Do you hear the speaker mention any numeric value?"]
ASK_Q = ["Does the speaker ask a question?", "Is the speaker asking a question?", "Does the recording contain a question?",
         "Is a question asked in the audio?", "Does the person speaking pose a question?"]
WORD_CHOICE_Q = ["Which of these words did the speaker say?", "Which word appears in the recording?",
                 "One of these words is spoken in the audio. Which one?", "Which of the following words do you hear?",
                 "Which word did the speaker use?"]
WORD_YN_Q = ['Did the speaker say the word "{w}"?', 'Is the word "{w}" spoken in the audio?',
             'Does the speaker use the word "{w}"?', 'Do you hear the word "{w}" in the recording?',
             'Is "{w}" one of the words the speaker says?']


def plan_checks(clips: list[dict], vocab: list[str], n_num: int, n_ask: int, n_word: int, n_word_yn: int,
                used: Counter) -> tuple[list[dict], Counter]:
    """clips: dicts with clip, text, cal, voice, bank, key. Returns check items (balanced yes/no per check type).
    `used` counts items already attached to each clip (cap MAX_ITEMS_PER_CLIP, and at most 2 checks per clip)."""
    stats = Counter()
    checks = Counter()
    out = []
    order = sorted(clips, key=lambda c: h01("chk:" + c["clip"]))

    def free(c):
        return used[c["clip"]] < MAX_ITEMS_PER_CLIP and checks[c["clip"]] < 2

    def emit(c, kind, question, options, gold, qtype, extra):
        rng = rng_for("chkitem", c["clip"], kind)
        meta = {"check": kind, "transcript": c["text"], "voice": c["voice"], "tts_source_row": c["row_id"],
                "licence": c["licence"], "duration_s": c["duration_s"], **extra}
        out.append(item(iid=f"listen/{c['clip']}/{kind}", cal=c["cal"], source="listen_check", question=question,
                        options=options, gold=gold, rng=rng, audio={"bank": c["bank"], "key": c["key"]}, qtype=qtype,
                        meta=meta))
        used[c["clip"]] += 1
        checks[c["clip"]] += 1

    for kind, fn, qs, n in (("number", number_label, NUM_Q, n_num), ("asks_question", question_label, ASK_Q, n_ask)):
        pools = {True: [], False: []}
        for c in order:
            lab = fn(c["text"])
            if lab is not None:
                pools[lab].append(c)
        half = min(n // 2, len(pools[True]), len(pools[False]))
        for lab in (True, False):
            k = 0
            for c in pools[lab]:
                if k >= half:
                    break
                if not free(c):
                    continue
                emit(c, kind, rng_for("chkq", c["clip"], kind).choice(qs), YES_NO, 0 if lab else 1, "noul", {"label": lab})
                k += 1
            stats[f"{kind}:{lab}"] = k
    k = 0
    for c in sorted(order, key=lambda c: h01("wc:" + c["clip"])):
        if k >= n_word:
            break
        if not free(c):
            continue
        rng = rng_for("word", c["clip"])
        w = said_word(c["text"], rng)
        ds = distractor_words(c["text"], vocab, rng, 3) if w else None
        if not ds:
            continue
        emit(c, "word_choice", rng.choice(WORD_CHOICE_Q), [w] + ds, 0, "choice", {"said": w, "distractors": ds})
        k += 1
    stats["word_choice"] = k
    flags = balanced_flags(n_word_yn, rng_for("word_yn_flags"))
    k = 0
    for c in sorted(order, key=lambda c: h01("wyn:" + c["clip"])):
        if k >= n_word_yn:
            break
        if not free(c):
            continue
        rng = rng_for("wordyn", c["clip"])
        yes = flags[k]
        if yes:
            w = said_word(c["text"], rng)
        else:
            d = distractor_words(c["text"], vocab, rng, 1)
            w = d[0] if d else None
        if not w:
            continue
        emit(c, "word_yn", rng.choice(WORD_YN_Q).format(w=w), YES_NO, 0 if yes else 1, "noul", {"word": w, "label": yes})
        stats[f"word_yn:{yes}"] += 1
        k += 1
    return out, stats


def qa_item(spec: dict, bank: str, duration_s: float) -> dict:
    qa = spec["qa"]
    rng = rng_for("qaitem", spec["clip"])
    meta = {"transcript": spec["text"], "voice": spec["voice"], "speed": spec["speed"], "variant": spec["variant"],
            "row_id": spec["row_id"], "licence": spec["licence"], "dataset": spec["dataset"],
            "duration_s": round(duration_s, 3), "tts": "Kokoro-82M"}
    for k in ("suite", "family"):
        if k in spec:
            meta[k] = spec[k]
    if "asked" in qa:
        meta["asked_option"] = qa["asked"]
    return item(iid=f"{spec['source']}/{spec['clip']}/{spec['variant']}", cal=spec["cal"], source=spec["source"],
                question=qa["question"], options=qa["options"], gold=qa["gold"], rng=rng,
                audio={"bank": bank, "key": spec["clip"]}, qtype=qa["qtype"], meta=meta,
                state_text=qa["state_text"], shuffle=qa["shuffle"])


# Validation
KEYS = {"id", "set", "source", "question", "options", "target", "gold", "audio", "channel", "state_text", "meta"}


def validate(items: list[dict], bank_index: dict[str, dict]) -> dict:
    """bank_index: bank -> {key: [start, length]}. Returns problems + balance stats."""
    probs = Counter()
    ids = Counter(it["id"] for it in items)
    probs["duplicate_ids"] = sum(v - 1 for v in ids.values() if v > 1)
    yes = defaultdict(lambda: [0, 0])
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
        q = it["meta"].get("qtype")
        if q not in ("choice", "noul", "score"):
            probs["qtype"] += 1
        if q == "noul":
            if sorted(it["options"]) != ["no", "yes"]:
                probs["noul_options"] += 1
            y = it["options"][it["gold"]] == "yes"
            yes[(it["source"], it["meta"].get("check") or ("intent" if "asked_intent" in it["meta"] else "x"))][0 if y else 1] += 1
        a = it["audio"]
        if a is None or a["bank"] not in bank_index or a["key"] not in bank_index[a["bank"]]:
            probs["audio_missing"] += 1
            continue
        per_clip[(a["bank"], a["key"])] += 1
        L = bank_index[a["bank"]][a["key"]][1] / C.SAMPLE_RATE
        durs.append(L)
        if L > MAX_CLIP_S + 1e-6 or L <= 0.05:
            probs["duration"] += 1
        if it["set"] not in ("train", "CAL"):
            probs["set"] += 1
    probs["clips_over_cap"] = sum(1 for v in per_clip.values() if v > MAX_ITEMS_PER_CLIP)
    return {"problems": {k: v for k, v in probs.items() if v}, "n_items": len(items),
            "yes_no_balance": {f"{s}|{k}": {"yes": v[0], "no": v[1]} for (s, k), v in sorted(yes.items())},
            "clip_duration_s": {"max": round(max(durs), 2) if durs else 0, "mean": round(sum(durs) / max(1, len(durs)), 2)},
            "max_items_per_clip": max(per_clip.values()) if per_clip else 0}
