"""Pure logic for sb_calltext: transcripts, LLM label questions, the two-run agreement rule and item building.

Each label question is asked twice (options in canonical and reversed order); see `agree`.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter, defaultdict

import numpy as np

SEED = 0
AGREE_MIN = 0.70
MAX_ITEMS_PER_CLIP = 4
CAL_FRAC = 0.03
PHONE_FRAC = 0.70
NOUL_MAX_RATIO = 1.15  # majority class kept at <= 1.15 x minority per noul question type (~53/47)

# Label questions asked to the LLM (canonical option order)
HVB_INTENTS = ["check balance", "get branch hours", "order checks", "pay bill", "replace card", "reset password",
               "schedule appointment", "transfer money"]
INTENT_TEXT = {  # option text shown to the LLM and (shortened) to the audio model
    "check balance": "checking an account balance",
    "get branch hours": "asking about branch opening hours",
    "order checks": "ordering new checks",
    "pay bill": "paying a bill",
    "replace card": "replacing a lost or damaged card",
    "reset password": "resetting a password",
    "schedule appointment": "scheduling an appointment",
    "transfer money": "transferring money between accounts",
}
EMOTIONS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
SENTIMENT = ["negative", "neutral", "positive"]  # natural (score) order low -> high

HVB_LABELS = {
    # key: (question for the LLM, canonical options)
    "sentiment": ("Overall, how did the caller (customer) feel during this call?", SENTIMENT),
    "emotion": ("What is the caller's (customer's) main emotion in this call?", EMOTIONS),
    "frustrated": ("Did the caller (customer) sound frustrated or annoyed at any point in the call?", ["yes", "no"]),
    "resolved": ("Was the caller's issue resolved by the end of the call?", ["yes", "no", "unclear"]),
    "reason": ("What was the main reason for the call?", [INTENT_TEXT[k] for k in HVB_INTENTS]),
    "personal": ("Did the agent ask the caller for personal details (for example their name, phone number, "
                 "account number or other identifying information)?", ["yes", "no"]),
    "price": ("Did the caller ask about a price, fee or cost of something?", ["yes", "no"]),
    "nextstep": ("Did the call end with a concrete next step agreed (for example an appointment, a callback, a "
                 "transfer to someone else, or something being sent to the caller)?", ["yes", "no"]),
}
E22_TOPICS = ["financial results for the period", "guidance and outlook for the future",
              "strategy, products or customers", "costs, margins or efficiency",
              "cash, debt, dividends or share buybacks", "call logistics (introductions, safe-harbor notice, operator)"]
E22_LABELS = {
    "qa": ("Is the main speaker in this excerpt answering an analyst's question or giving prepared remarks?",
           ["answering an analyst's question", "giving prepared remarks", "neither (operator or analyst speaking)"]),
    "tone": ("Does the company executive speaking in this excerpt come across as confident or cautious?",
             ["confident", "cautious", "no company executive speaks"]),
    "news": ("Overall, is what is said in this excerpt good, neutral or bad news for the company?",
             ["bad news", "neutral", "good news"]),
    "topic": ("What is the main topic of this excerpt?", E22_TOPICS),
    "figures": ("Does a speaker in this excerpt state specific financial figures (amounts, percentages or growth "
                "rates)?", ["yes", "no"]),
}

AMI_TOPICS = ["the product's look, design or materials", "technical components or functions",
              "cost, budget or price", "users, market research or trends", "meeting organisation, agenda or roles",
              "small talk or something unrelated"]
AMI_LABELS = {
    "mood": ("Overall, what is the mood of the discussion in this excerpt?", ["negative or tense", "neutral", "positive or friendly"]),
    "disagree": ("Do the participants disagree with each other about something in this excerpt?", ["yes", "no"]),
    "frustrated": ("Does any participant sound frustrated, annoyed or impatient in this excerpt?", ["yes", "no"]),
    "decision": ("Does the group reach a decision or agree on something concrete in this excerpt?", ["yes", "no"]),
    "topic": ("What is the main topic of this excerpt?", AMI_TOPICS),
    "emotion": ("What is the main emotion expressed by the speakers in this excerpt?", EMOTIONS),
}

SYSTEM = {
    "hvb": ("You are a careful annotator of customer-service phone calls to a bank (Harper Valley Bank). You read a "
            "transcript and answer a multiple-choice question about the call. AGENT = the bank employee, CALLER = "
            "the customer. Use only what the transcript shows; when the transcript gives no sign of something, "
            "prefer the plain / default answer (e.g. neutral, no). Answer with only the letter of the best option."),
    "e22": ("You are a careful annotator of company earnings-call recordings. You read the transcript of a short "
            "excerpt (speakers numbered S1, S2, ... in order of appearance; names and roles are not given) and answer "
            "a multiple-choice question about the excerpt. Use only what the transcript shows. Answer with only the "
            "letter of the best option."),
    "ami": ("You are a careful annotator of recorded business meetings (a design team of 3-5 people). You read the "
            "transcript of a short excerpt (speakers labelled by their role or as Speaker A-E) and answer a "
            "multiple-choice question about the excerpt. Use only what the transcript shows; when it gives no sign of "
            "something, prefer the plain / default answer (e.g. neutral, no). Answer with only the letter of the best "
            "option."),
}
LETTERS = "ABCDEFGHIJ"


def run_order(n: int, run: int) -> list[int]:
    """Display order of canonical option indices for LLM run 0 (canonical) and run 1 (reversed)."""
    order = list(range(n))
    return order if run == 0 else order[::-1]


def label_request(rid: str, domain: str, transcript: str, question: str, options: list[str], run: int) -> dict:
    order = run_order(len(options), run)
    lines = "\n".join(f"{LETTERS[j]}. {options[c]}" for j, c in enumerate(order))
    head = ("Transcript of the call:" if domain == "hvb" else "Transcript of the excerpt:")
    user = f"{head}\n{transcript}\n\nQuestion: {question}\nOptions:\n{lines}\n\nAnswer with only the letter."
    return {"rid": rid, "system": SYSTEM[domain], "user": user, "letters": list(LETTERS[: len(options)]),
            "order": order}


def to_canonical(probs: list[float], order: list[int]) -> np.ndarray:
    out = np.zeros(len(order))
    for j, c in enumerate(order):
        out[c] = probs[j]
    return out


def agree(p0: np.ndarray, p1: np.ndarray, thr: float = AGREE_MIN) -> tuple[bool, int, np.ndarray]:
    """Both runs pick the same top option, each with probability >= thr -> (kept, top, mean distribution)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    mean = (p0 + p1) / 2.0
    mean = mean / mean.sum() if mean.sum() > 0 else mean
    t0, t1 = int(np.argmax(p0)), int(np.argmax(p1))
    ok = t0 == t1 and p0[t0] >= thr and p1[t1] >= thr
    return ok, t0, mean


def restrict(dist: np.ndarray, keep: list[int]) -> np.ndarray:
    """Distribution over a subset of the canonical options (renormalised), with a tiny floor."""
    q = np.asarray([dist[i] for i in keep], float) + 1e-4
    return q / q.sum()


# Transcripts
def hvb_turns(segments: list[dict]) -> list[dict]:
    """HVB segments -> role turns in conversation order, merging consecutive same-role segments."""
    segs = []
    for s in segments:
        text = (s.get("human_transcript") or s.get("transcript") or "").strip()
        text = re.sub(r"\[[^\]]*\]|<[^>]*>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        st = int(s["start_ms"])
        segs.append({"role": s["speaker_role"], "start_ms": st, "end_ms": st + int(s["duration_ms"]), "text": text})
    segs.sort(key=lambda x: (x["start_ms"], x["role"]))
    turns = []
    for s in segs:
        if turns and turns[-1]["role"] == s["role"]:
            turns[-1]["text"] += " " + s["text"]
            turns[-1]["end_ms"] = max(turns[-1]["end_ms"], s["end_ms"])
        else:
            turns.append(dict(s))
    return turns


def hvb_transcript(turns: list[dict]) -> str:
    return "\n".join(f"{t['role'].upper()}: {t['text']}" for t in turns)


def channel_shift_ms(segments: list[dict], role: str) -> int:
    """Offset (ms) of a channel recording relative to conversation time: recording_ms = conversation_ms + shift."""
    d = [int(s["offset_ms"]) - int(s["start_ms"]) for s in segments if s.get("speaker_role") == role
         and s.get("offset_ms") is not None and s.get("start_ms") is not None]
    return int(np.median(d)) if d else 0


def parse_nlp(text: str) -> list[dict]:
    """Earnings-22 force-aligned .nlp -> tokens {tok, spk, ts, te, punct} (ts/te None when not aligned)."""
    lines = text.splitlines()
    head = lines[0].split("|")
    col = {h: i for i, h in enumerate(head)}
    out = []
    for ln in lines[1:]:
        p = ln.split("|")
        if len(p) < len(head) - 2:
            continue

        def f(name):
            i = col.get(name)
            if i is None or i >= len(p) or p[i] == "":
                return None
            try:
                return float(p[i])
            except ValueError:
                return None

        out.append({"tok": p[col["token"]], "spk": p[col["speaker"]], "ts": f("ts"), "te": f("endTs"),
                    "punct": p[col["punctuation"]] if "punctuation" in col and col["punctuation"] < len(p) else ""})
    return out


def nlp_sentences(tokens: list[dict]) -> list[dict]:
    """Tokens -> sentences (split at . ? ! or speaker change) with start/end time from aligned tokens."""
    sents, cur = [], []

    def flush():
        if not cur:
            return
        ts = [t["ts"] for t in cur if t["ts"] is not None]
        te = [t["te"] for t in cur if t["te"] is not None]
        if ts and te:
            words = []
            for t in cur:
                words.append(t["tok"] + (t["punct"] or ""))
            sents.append({"spk": cur[0]["spk"], "start": min(ts), "end": max(te), "text": " ".join(words),
                          "n_tok": len(cur), "n_aligned": len(ts)})
        cur.clear()

    for t in tokens:
        if cur and t["spk"] != cur[-1]["spk"]:
            flush()
        cur.append(t)
        if (t["punct"] or "").strip() in {".", "?", "!"}:
            flush()
    flush()
    # drop sentences with broken timing
    return [s for s in sents if s["end"] > s["start"] and s["end"] - s["start"] < 60 and s["n_aligned"] >= 0.5 * s["n_tok"]]


def make_windows(sents: list[dict], rng: random.Random, lo: float = 30.0, hi: float = 100.0,
                 max_len: float = 180.0, gap: tuple = (3.0, 30.0)) -> list[dict]:
    """Non-overlapping windows of consecutive sentences, target length ~U(lo, hi) s, cut at sentence ends."""
    wins, i, n = [], 0, len(sents)
    while i < n:
        target = rng.uniform(lo, hi)
        j = i
        while j < n and sents[j]["end"] - sents[i]["start"] < target:
            j += 1
        if j >= n:
            break
        start, end = sents[i]["start"], sents[j]["end"]
        if lo <= end - start <= max_len:
            wins.append({"i0": i, "i1": j, "start": max(0.0, start - 0.3), "end": end + 0.4})
        # next window starts after a random gap
        nxt_t = end + rng.uniform(*gap)
        k = j + 1
        while k < n and sents[k]["start"] < nxt_t:
            k += 1
        i = k
    return wins


def window_transcript(sents: list[dict], i0: int, i1: int) -> str:
    """Speakers renumbered S1, S2, ... by first appearance; consecutive same-speaker sentences merged."""
    names, lines = {}, []
    for s in sents[i0 : i1 + 1]:
        if s["spk"] not in names:
            names[s["spk"]] = f"S{len(names) + 1}"
        tag = names[s["spk"]]
        if lines and lines[-1][0] == tag:
            lines[-1][1].append(s["text"])
        else:
            lines.append((tag, [s["text"]]))
    return "\n".join(f"{tag}: {' '.join(t)}" for tag, t in lines)


def is_cal(key: str, frac: float = CAL_FRAC, salt: str = "sb_calltext") -> bool:
    h = int(hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()[:8], 16)
    return (h % 10_000) < frac * 10_000


def choose_cal_groups(group_of_each_clip: list[str], frac: float = CAL_FRAC, salt: str = "sb_calltext") -> set:
    """Whole groups (e.g. earnings calls) for CAL: groups in sha256 order until >= frac of the clips are covered."""
    cnt = Counter(group_of_each_clip)
    order = sorted(cnt, key=lambda g: hashlib.sha256(f"{salt}:{g}".encode()).hexdigest())
    out, n = set(), 0
    for g in order:
        if n >= frac * len(group_of_each_clip):
            break
        out.add(g)
        n += cnt[g]
    return out


# Audio-model questions (>= 5 paraphrases per type) and views of each label
def _roles(rng: random.Random, receptionist_frac: float = 0.15) -> tuple[str, str]:
    """(caller word, business-side word) for HVB items; some items use 'AI receptionist' / 'caller' phrasing."""
    if rng.random() < receptionist_frac:
        return "caller", rng.choice(["AI receptionist", "receptionist"])
    return rng.choice([("caller", "agent"), ("customer", "agent"), ("caller", "bank agent"),
                       ("customer", "bank representative"), ("customer", "agent")])


Q_HVB = {
    "sentiment_choice": ["Overall, how did the {c} feel during this call?",
                         "What was the {c}'s overall sentiment in this call?",
                         "How would you describe the {c}'s mood over the whole call?",
                         "Taking the whole call into account, how did the {c} come across?",
                         "Was the {c} positive, neutral or negative during this call?",
                         "Listen to the whole call. What best describes how the {c} felt?"],
    "sentiment_score": ["Rate the {c}'s overall sentiment in this call.",
                        "How positive was the {c} during this call?",
                        "On the whole, how did the {c} feel about this call, from negative to positive?",
                        "Overall caller sentiment for this call?",
                        "How satisfied did the {c} sound over the course of the call?"],
    "sentiment_neg": ["Was the {c} in a negative mood during this call?",
                      "Did the {c} come across as unhappy overall?",
                      "Overall, was the {c}'s sentiment negative?",
                      "Would you say the {c} was displeased during this call?",
                      "Did the {c} sound upset or dissatisfied overall?"],
    "sentiment_pos": ["Did the {c} come across as positive and pleased overall?",
                      "Was the {c}'s overall sentiment positive?",
                      "Did the {c} sound happy with how the call went?",
                      "Overall, was the {c} in a good mood during this call?",
                      "Would you describe the {c} as positive in this call?"],
    "emotion": ["What is the {c}'s main emotion in this call?",
                "Which emotion best describes the {c} during this call?",
                "What emotion does the {c} mostly express in this call?",
                "How does the {c} feel for most of the call? Pick the closest emotion.",
                "Listen to the {c}. What is their dominant emotion?",
                "Which emotion comes through most in the {c}'s voice and words?"],
    "frustrated": ["Did the {c} sound frustrated at any point?",
                   "Was the {c} frustrated or annoyed at some point in the call?",
                   "Did the {c} show any frustration during this call?",
                   "At any moment, did the {c} get irritated?",
                   "Is there a point where the {c} sounds annoyed or fed up?"],
    "resolved": ["Was the {c}'s issue resolved by the end of the call?",
                 "Did the {a} solve the {c}'s problem?",
                 "By the end of the call, did the {c} get what they called for?",
                 "Was the reason for the call taken care of?",
                 "Did this call end with the {c}'s request handled?",
                 "Was the {c}'s problem fixed during the call?"],
    "reason": ["What was the main reason for the call?",
               "Why did the {c} call?",
               "What did the {c} call about?",
               "What was the {c}'s request in this call?",
               "What is this call mainly about?",
               "What did the {c} need help with?"],
    "personal": ["Did the {a} ask the {c} for personal details (name, phone number, account)?",
                 "Did the {a} request any identifying information from the {c}?",
                 "Was the {c} asked for their name, phone number or account details?",
                 "Did the {a} ask the {c} to verify who they are?",
                 "Did the {a} collect any personal information from the {c}?"],
    "price": ["Did the {c} ask about price or cost?",
              "Did the {c} ask how much something costs?",
              "Was there a question from the {c} about fees or prices?",
              "Did the {c} enquire about a fee or charge?",
              "Did pricing or cost come up as a question from the {c}?"],
    "nextstep": ["Did the call end with a next step agreed (appointment, callback, transfer)?",
                 "Was a follow-up action agreed before the call ended?",
                 "Did the {a} and the {c} agree on a next step?",
                 "By the end of the call, was something scheduled, sent or arranged?",
                 "Did the call finish with a concrete next step for the {c}?"],
}
Q_E22 = {
    "qa": ["Is the speaker answering an analyst's question or giving prepared remarks?",
           "Is this part of the earnings call prepared remarks or an answer to an analyst?",
           "Listen to this earnings-call excerpt. Is the main speaker reading prepared remarks or answering a question?",
           "Which part of the earnings call is this: prepared remarks or the Q&A answer?",
           "Is the executive here responding to an analyst or delivering scripted remarks?"],
    "tone": ["Does the tone of the executive sound confident or cautious?",
             "How does the executive come across in this excerpt: confident or cautious?",
             "Is the speaking executive confident or cautious here?",
             "Would you describe the executive's tone as confident or cautious?",
             "Listen to the executive. Confident or cautious?"],
    "news_choice": ["Is what is said here good news, neutral, or bad news for the company?",
                    "Overall, is this excerpt good or bad news for the company?",
                    "What is the overall sentiment of this earnings-call excerpt for the company?",
                    "Does this part of the call deliver good news, bad news, or neither?",
                    "How does this excerpt sound for the company's business: good, neutral or bad news?"],
    "news_score": ["Rate how positive this earnings-call excerpt is for the company.",
                   "How good is the news in this excerpt for the company?",
                   "Sentiment of this excerpt for the company, from bad to good?",
                   "How positive is the message in this part of the earnings call?",
                   "Overall, how favourable is what the speakers say here?"],
    "topic": ["What is the main topic of this excerpt?",
              "What are the speakers mainly talking about here?",
              "Which topic best describes this part of the earnings call?",
              "What is this excerpt of the earnings call mostly about?",
              "What subject does this part of the call cover?"],
    "figures": ["Does the speaker give specific financial figures?",
                "Are specific numbers (amounts, percentages, growth rates) stated in this excerpt?",
                "Does anyone in this excerpt quote concrete financial figures?",
                "Are there specific financial numbers mentioned here?",
                "Does this excerpt include exact figures such as revenue amounts or percentages?"],
}
Q_AMI = {
    "mood_choice": ["What is the mood of the discussion in this recording?",
                    "How would you describe the atmosphere of this meeting excerpt?",
                    "Overall, is this discussion positive, neutral or tense?",
                    "Listen to the meeting. What best describes the mood?",
                    "How does the group come across in this part of the meeting?"],
    "mood_score": ["Rate the overall mood of this meeting excerpt.",
                   "How positive is the atmosphere in this part of the meeting?",
                   "From tense to friendly, how does this discussion feel?",
                   "Overall mood of the participants here?",
                   "How pleasant is the discussion in this recording?"],
    "disagree": ["Do the participants disagree about something here?",
                 "Is there any disagreement between the speakers in this excerpt?",
                 "Does anyone in the meeting push back on what another person said?",
                 "Are the speakers in conflict about something in this recording?",
                 "Do the people in this meeting have differing opinions here?"],
    "frustrated": ["Does anyone in the meeting sound frustrated?",
                   "Is any participant annoyed or impatient in this excerpt?",
                   "Does a speaker show frustration at some point?",
                   "Is there a moment where someone sounds irritated?",
                   "Does any of the participants get impatient here?"],
    "decision": ["Does the group reach a decision in this excerpt?",
                 "Do the participants agree on something concrete here?",
                 "Is a decision made in this part of the meeting?",
                 "By the end of this excerpt, has the team settled on something?",
                 "Does this part of the meeting end with an agreement?"],
    "topic": ["What is the main topic of this meeting excerpt?",
              "What are the participants mainly discussing?",
              "What is this part of the meeting about?",
              "Which topic best describes this discussion?",
              "What subject is the team talking about here?"],
    "emotion": ["What is the main emotion expressed by the speakers here?",
                "Which emotion best describes the speakers in this excerpt?",
                "What emotion comes through most in this part of the meeting?",
                "Listen to the participants. What is their dominant emotion?",
                "How do the speakers mostly feel in this recording? Pick the closest emotion."],
}
REASON_OPTIONS = {k: v for k, v in zip(HVB_INTENTS, ["check an account balance", "ask about branch opening hours",
                                                       "order new checks", "pay a bill", "replace a card",
                                                       "reset a password", "schedule an appointment",
                                                       "transfer money"])}
STATE_HVB = ["Recording of a phone call between a bank's customer-service agent and a customer.",
             "Customer-service call to a bank.", "A recorded support call.", None]
STATE_E22 = ["Excerpt from a company's quarterly earnings call.", "Earnings conference call recording.", None]
STATE_AMI = ["Excerpt from a recorded team meeting.", "Recording of a product-design meeting.", None]


class _NoShuffle(random.Random):
    def shuffle(self, x, *a, **k):  # keeps score levels in their natural order
        return None


def views_for(domain: str, key: str, dist: np.ndarray) -> list[dict]:
    """Label distribution over the LLM options -> candidate training views ([] when the label is not usable)."""
    top = int(np.argmax(dist))
    out = []

    def noul(view, yes_idx_set):
        keep_yes = sum(dist[i] for i in yes_idx_set)
        t = np.array([keep_yes, 1.0 - keep_yes])
        t = (t + 1e-4) / (t + 1e-4).sum()
        out.append({"view": view, "qtype": "noul", "options": ["yes", "no"], "target": t,
                    "label": "yes" if top in yes_idx_set else "no"})

    if domain == "hvb":
        if key == "sentiment":
            opts = SENTIMENT
            out.append({"view": "sentiment_choice", "qtype": "choice", "options": opts, "target": restrict(dist, [0, 1, 2]), "label": opts[top]})
            out.append({"view": "sentiment_score", "qtype": "score", "options": opts, "target": restrict(dist, [0, 1, 2]), "label": opts[top]})
            noul("sentiment_neg", {0})
            noul("sentiment_pos", {2})
        elif key == "emotion":
            out.append({"view": "emotion", "qtype": "choice", "options": EMOTIONS, "target": restrict(dist, list(range(6))), "label": EMOTIONS[top]})
        elif key in ("frustrated", "personal", "price", "nextstep"):
            noul(key, {0})
        elif key == "resolved":
            if top == 2:  # "unclear" -> not used as a yes/no question
                return []
            q = restrict(dist, [0, 1])
            out.append({"view": "resolved", "qtype": "noul", "options": ["yes", "no"], "target": q, "label": ["yes", "no"][top]})
        elif key == "reason":
            opts = [REASON_OPTIONS[k] for k in HVB_INTENTS]
            out.append({"view": "reason", "qtype": "choice", "options": opts, "target": restrict(dist, list(range(8))), "label": opts[top]})
    elif domain == "ami":
        if key == "mood":
            opts = ["negative or tense", "neutral", "positive or friendly"]
            out.append({"view": "mood_choice", "qtype": "choice", "options": opts, "target": restrict(dist, [0, 1, 2]), "label": opts[top]})
            out.append({"view": "mood_score", "qtype": "score", "options": opts, "target": restrict(dist, [0, 1, 2]), "label": opts[top]})
        elif key in ("disagree", "frustrated", "decision"):
            noul(key, {0})
        elif key == "topic":
            out.append({"view": "topic", "qtype": "choice", "options": list(AMI_TOPICS), "target": restrict(dist, list(range(len(AMI_TOPICS)))), "label": AMI_TOPICS[top]})
        elif key == "emotion":
            out.append({"view": "emotion", "qtype": "choice", "options": EMOTIONS, "target": restrict(dist, list(range(6))), "label": EMOTIONS[top]})
    else:
        if key == "qa":
            if top == 2:
                return []
            opts = ["answering an analyst's question", "giving prepared remarks"]
            out.append({"view": "qa", "qtype": "choice", "options": opts, "target": restrict(dist, [0, 1]), "label": opts[top]})
        elif key == "tone":
            if top == 2:
                return []
            opts = ["confident", "cautious"]
            out.append({"view": "tone", "qtype": "choice", "options": opts, "target": restrict(dist, [0, 1]), "label": opts[top]})
        elif key == "news":
            opts = ["bad news", "neutral", "good news"]
            out.append({"view": "news_choice", "qtype": "choice", "options": opts, "target": restrict(dist, [0, 1, 2]), "label": opts[top]})
            out.append({"view": "news_score", "qtype": "score", "options": opts, "target": restrict(dist, [0, 1, 2]), "label": opts[top]})
        elif key == "topic":
            out.append({"view": "topic", "qtype": "choice", "options": list(E22_TOPICS), "target": restrict(dist, list(range(len(E22_TOPICS)))), "label": E22_TOPICS[top]})
        elif key == "figures":
            noul("figures", {0})
    return out


def balance_noul(cands: list[dict], rng: random.Random, ratio: float = NOUL_MAX_RATIO) -> list[dict]:
    """Keep all choice/score views; per yes/no view keep the minority class and <= ratio x as many of the majority."""
    keep = [c for c in cands if c["qtype"] != "noul"]
    by = defaultdict(lambda: defaultdict(list))
    for c in cands:
        if c["qtype"] == "noul":
            by[c["view"]][c["label"]].append(c)
    for view, d in sorted(by.items()):
        ys, ns = d.get("yes", []), d.get("no", [])
        m = min(len(ys), len(ns))
        cap = int(math.floor(m * ratio))
        for lst in (ys, ns):
            lst = list(lst)
            rng.shuffle(lst)
            keep.extend(lst[: max(m, min(len(lst), cap))] if len(lst) > m else lst)
    return keep


def allocate(cands_by_group: dict, slots_by_group: dict, rng: random.Random, max_per_clip: int = MAX_ITEMS_PER_CLIP,
             max_views_per_label: int = 2, caps: dict | None = None) -> list[dict]:
    """Assign views to the clips of each group, rarer views first, with per-clip and per-label caps."""
    freq = Counter(c["view"] for cs in cands_by_group.values() for c in cs)
    out = []
    for g in sorted(cands_by_group):
        cs = list(cands_by_group[g])
        rng.shuffle(cs)
        cs.sort(key=lambda c: freq[c["view"]])
        clips = slots_by_group[g]
        cap = (caps or {}).get(g, max_per_clip)
        used = Counter()
        per_label = Counter()
        for c in cs:
            if per_label[c["label_key"]] >= max_views_per_label:
                continue
            clip = min(clips, key=lambda k: (used[k], clips.index(k)))
            if used[clip] >= cap:
                break
            used[clip] += 1
            per_label[c["label_key"]] += 1
            out.append(dict(c, clip=clip))
    return out


def question_text(domain: str, view: str, rng: random.Random) -> str:
    if domain == "hvb":
        c, a = _roles(rng)
        return rng.choice(Q_HVB[view]).format(c=c, a=a)
    if domain == "ami":
        return rng.choice(Q_AMI[view])
    return rng.choice(Q_E22[view])


LABELS = {"hvb": HVB_LABELS, "e22": E22_LABELS, "ami": AMI_LABELS}


def ami_words(xml_text: str, spk: str) -> list[dict]:
    """AMI NXT words file -> tokens {tok, spk, ts, te, punct} (punctuation tokens attached to the previous word)."""
    import re as _re
    import html

    out = []
    for m in _re.finditer(r"<w ([^>]*)>([^<]*)</w>", xml_text):
        attrs, word = m.group(1), html.unescape(m.group(2))
        st = _re.search(r'starttime="([0-9.]+)"', attrs)
        en = _re.search(r'endtime="([0-9.]+)"', attrs)
        if 'punc="true"' in attrs:
            if out:
                out[-1]["punct"] += word
            continue
        if not st or not en:
            continue
        out.append({"tok": word, "spk": spk, "ts": float(st.group(1)), "te": float(en.group(1)), "punct": ""})
    return out


def ami_sentences(tokens: list[dict], pause: float = 1.2) -> list[dict]:
    """All speakers' tokens (any order) -> utterances: split at . ? !, speaker change or a pause > `pause` s."""
    toks = sorted(tokens, key=lambda t: (t["ts"], t["spk"]))
    by_spk = {}
    for t in toks:
        by_spk.setdefault(t["spk"], []).append(t)
    sents = []
    for spk, ts in by_spk.items():
        cur = []
        for t in ts:
            if cur and t["ts"] - cur[-1]["te"] > pause:
                sents.append(cur)
                cur = []
            cur.append(t)
            if any(p in t["punct"] for p in ".?!"):
                sents.append(cur)
                cur = []
        if cur:
            sents.append(cur)
    out = [{"spk": c[0]["spk"], "start": c[0]["ts"], "end": max(x["te"] for x in c),
            "text": " ".join(x["tok"] + x["punct"] for x in c), "n_tok": len(c), "n_aligned": len(c)} for c in sents]
    out.sort(key=lambda s: s["start"])
    return out


def ami_transcript(sents: list[dict], i0: int, i1: int, roles: dict) -> str:
    lines = []
    for s in sents[i0 : i1 + 1]:
        tag = roles.get(s["spk"], f"Speaker {s['spk']}")
        if lines and lines[-1][0] == tag:
            lines[-1][1].append(s["text"])
        else:
            lines.append((tag, [s["text"]]))
    return "\n".join(f"{tag}: {' '.join(t)}" for tag, t in lines)


# Validation of written items
REQUIRED_KEYS = {"id", "set", "source", "question", "options", "target", "gold", "audio", "channel", "state_text", "meta"}


def validate_item(it: dict) -> list[str]:
    errs = []
    if set(it) - {"majority"} != REQUIRED_KEYS:
        errs.append(f"keys {sorted(set(it) ^ REQUIRED_KEYS)}")
    k = len(it["options"])
    if not 2 <= k <= 20:
        errs.append("n options")
    if len(it["target"]) != k:
        errs.append("target len")
    if abs(sum(it["target"]) - 1.0) > 1e-3:
        errs.append("target sum")
    if int(np.argmax(it["target"])) != it["gold"]:
        errs.append("gold != argmax")
    if len(set(it["options"])) != k:
        errs.append("dup options")
    q = it["meta"].get("qtype")
    if q not in ("choice", "noul", "score"):
        errs.append("qtype")
    if q == "noul" and sorted(it["options"]) != ["no", "yes"]:
        errs.append("noul options")
    if it["channel"] not in ("clean", "phone"):
        errs.append("channel")
    if it["set"] not in ("train", "CAL"):
        errs.append("set")
    return errs
