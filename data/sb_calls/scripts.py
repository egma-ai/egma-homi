"""sb_calls call-outcome classes, script templates and the seeded per-clip plan.

All people and businesses are fictional. `cue_idx` in a plan is the segment by whose end the class is clear."""
from __future__ import annotations

import random

# Classes
CLASSES = ["live_person", "personal_voicemail", "carrier_voicemail", "business_ivr", "call_screening",
           "no_answer", "busy", "not_in_service", "fax_modem"]
SPEECH_CLASSES = ["live_person", "personal_voicemail", "carrier_voicemail", "business_ivr", "call_screening",
                  "not_in_service"]  # classes that need TTS (not_in_service: the recorded notice after the SIT)
TONE_ONLY = ["no_answer", "busy", "fax_modem"]
VOICEMAIL = ["personal_voicemail", "carrier_voicemail"]

# Natural option strings for the "who/what answered" question (two phrasings per class; one set per item).
OPTION_TEXT = {
    "live_person": ["a live person", "a real person picked up"],
    "personal_voicemail": ["a personal voicemail greeting", "someone's recorded voicemail greeting"],
    "carrier_voicemail": ["a carrier's default voicemail message", "a generic automated voicemail message"],
    "business_ivr": ["an automated business phone menu", "a company's automated menu"],
    "call_screening": ["a call-screening assistant", "an automated call screener"],
    "no_answer": ["no answer (it just rang)", "nobody answered, only ringing"],
    "busy": ["a busy signal", "a busy tone"],
    "not_in_service": ["a not-in-service / disconnected number message", "a number-not-in-service recording"],
    "fax_modem": ["a fax or modem tone", "a fax machine or modem"],
}

# Kokoro-82M English voices (intelligibility checked with Whisper, see voice_wer.json)
EN_VOICES = ["af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova",
             "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
             "am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
             "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]
CAL_VOICES = ["af_nova", "am_eric", "bf_lily", "bm_lewis"]
TRAIN_VOICES = [v for v in EN_VOICES if v not in CAL_VOICES]

# Fictional names and businesses (the last few of each list are CAL-only)
NAMES = ["Dana", "Sam", "Maria", "Priya", "Jordan", "Alex", "Taylor", "Morgan", "Chris", "Elena", "Marcus",
         "Nina", "Owen", "Leah", "Raj", "Sofia", "Ben", "Grace", "Hector", "Ivy", "Kenji", "Laura", "Miguel",
         "Nora", "Pete", "Rosa", "Theo", "Uma", "Victor", "Wendy", "Yusuf", "Zoe", "Carmen", "Derek",
         # CAL-only
         "Felix", "Hannah", "Isaac", "Julia", "Kofi", "Lena"]
N_CAL_NAMES = 6
BIZ_PLACES = ["Riverside", "Maple Street", "Northgate", "Lakeview", "Oak Hill", "Silver Creek", "Pinecrest",
              "Harbor View", "Willow Park", "Sunnyvale Road", "Brookfield", "Cedar Grove", "Eastwood",
              "Fairmont", "Granite Peak", "Hollow Pine",
              # CAL-only
              "Juniper Bay", "Kestrel Point", "Larchmont"]
N_CAL_PLACES = 3
BIZ_TYPES = ["Dental", "Auto Repair", "Family Clinic", "Plumbing", "Insurance Agency", "Pet Hospital",
             "Hardware", "Bakery", "Law Office", "Realty", "Pharmacy", "Eye Care", "Landscaping",
             "Physical Therapy", "Print Shop",
             # CAL-only
             "Tutoring Center", "Bike Shop", "Florist"]
N_CAL_TYPES = 3

# Script templates; placeholders {name} {name2} {biz}. The last `n_cal` entries of each list are CAL-only.
LIVE_GREETINGS = [
    "Hello?", "Hi, this is {name}.", "Yes, who's calling?", "Hello, {name} speaking.", "Uh, hello?",
    "Hi there.", "Yeah, hello?", "Good morning, this is {name}.", "Hello, who is this?", "Hi, can I help you?",
    "This is {name}, go ahead.", "Hey, hello?", "Good afternoon, {biz}, this is {name}. How can I help you?",
    "{biz}, {name} speaking.", "Hi, thanks for calling {biz}, this is {name}.", "Um, hi, who's this?",
    "Hello, yes?", "Hello, {name} here.", "Hi, sorry, I was just walking in. Hello?",
    "Hello, this is {name}. Who am I speaking with?",
    # CAL-only
    "Hey, it's {name}.", "Morning, {biz}, {name} here.", "Yes, hello, can I help?",
]
LIVE_FOLLOWUPS = [
    "Hello?", "Hello? Anyone there?", "Hello? Hello?", "Can you hear me?", "Hi, is someone there?",
    "Sorry, who's calling?", "I can't hear you.", "Hello, are you there?", "Hmm, hello?",
    # CAL-only
    "Is anybody on the line?", "Hello? I think we got cut off.",
]
N_CAL_LIVE_G, N_CAL_LIVE_F = 3, 2

PVM_FIRST = [
    "Hi, you've reached {name}.", "Hey, this is {name}, I can't get to the phone right now.",
    "Hello, you've reached the voicemail of {name}.", "Hi, it's {name}. Sorry I missed your call.",
    "You've reached {name} and {name2}.", "This is {name}'s phone. I'm not available right now.",
    "Hi, you've reached {name} at {biz}.", "Hello, this is {name}. I'm away from my phone at the moment.",
    "Hey, you've reached {name}'s voicemail.", "Hi, this is {name}, and you've reached my voicemail.",
    "Thanks for calling {name}. I can't answer right now.", "Hello, you've reached the desk of {name}.",
    # CAL-only
    "Hi, {name} here, I can't pick up right now.", "You've reached the voicemail box of {name}.",
]
PVM_MIDDLE = [
    "I'm either on another call or away from my desk.", "I'll get back to you as soon as I can.",
    "Sorry I couldn't take your call.", "I'm out of the office today.", "I check my messages often.",
    "Thanks for calling.",
    # CAL-only
    "I'm probably driving right now.",
]
PVM_END = [
    "Please leave a message.", "Leave a message and I'll call you back.", "Please leave a message after the beep.",
    "Leave a message after the tone and I'll get back to you.", "Leave me a message and have a great day.",
    "Please leave a short message after the tone.", "Go ahead and leave a message.",
    # CAL-only
    "Leave a message at the beep.", "Talk to you soon, leave a message.",
]
N_CAL_PVM_F, N_CAL_PVM_M, N_CAL_PVM_E = 2, 1, 2

CVM_FIRST = [
    "The person you are calling is not available.", "The number you have reached is not available right now.",
    "Your call has been forwarded to an automated voice message system.",
    "The subscriber you have called is not available.",
    "The mailbox you have reached is not available to take your call.",
    "The party you are trying to reach is unavailable.", "You have reached the voice mailbox of {name}.",
    # CAL-only
    "The customer you are calling cannot take your call right now.", "The person at this number is unavailable.",
]
CVM_MIDDLE = [
    "At the tone, please record your message.", "When you have finished recording, you may hang up.",
    "Please leave a message after the tone.",
    # CAL-only
    "After the tone, record your message.",
]
CVM_END = [
    "Please leave your message after the tone.", "Record your message at the tone.",
    "When you are finished, simply hang up.", "Please begin speaking after the tone.",
    # CAL-only
    "Start your message after the tone.",
]
N_CAL_CVM_F, N_CAL_CVM_M, N_CAL_CVM_E = 2, 1, 1

IVR_GREET = [
    "Thank you for calling {biz}.", "Welcome to {biz}.", "You have reached {biz}.",
    "Hello, and thanks for calling {biz}.", "Thanks for calling {biz}, we appreciate your call.",
    # CAL-only
    "Hi, you've reached the main line for {biz}.",
]
IVR_PREAMBLE = [
    "Please listen carefully, as our menu options have changed.", "Please choose from the following options.",
    "If you know your party's extension, you may dial it at any time.",
    # CAL-only
    "Please select from the menu below.",
]
IVR_ITEM_FORMS = ["For {topic}, press {num}.", "To {verb}, press {num}.", "Press {num} for {topic}.",
                  # CAL-only
                  "For {topic}, please press {num}."]
IVR_TOPICS = ["hours and location", "appointments", "billing questions", "directions", "our mailing address",
              "prescription refills", "new customers", "scheduling", "general questions", "store hours",
              # CAL-only
              "holiday hours", "returns"]
IVR_VERBS = ["schedule an appointment", "hear our hours", "reach the front desk", "speak with a representative",
             "check on an order", "hear these options again",
             # CAL-only
             "reach our main office"]
IVR_TAIL = ["To speak with a team member, press zero.", "To repeat these options, press nine.",
            "Otherwise, please stay on the line.",
            # CAL-only
            "To hear this menu again, press star."]
N_CAL_IVR = {"greet": 1, "pre": 1, "form": 1, "topic": 2, "verb": 1, "tail": 1}
NUM_WORDS = ["one", "two", "three", "four", "five", "six"]

SCR_FIRST = [
    "Hi, the person you're calling is using a screening service.",
    "Hello, this call is being screened by an automated assistant.",
    "The person you're trying to reach is using a call screening assistant.",
    "Hello, I'm an assistant screening calls for {name}.", "Hi, {name} is using a call assistant to screen calls.",
    "Hi, this is a call screening service.",
    # CAL-only
    "Hello, you've reached an automated screening assistant for this number.",
]
SCR_SECOND = [
    "Please say your name and the reason for your call.", "Go ahead and say your name and why you're calling.",
    "Please state your name and what you're calling about.", "Who's calling, and what is this regarding?",
    "Tell me your name and the purpose of your call, and I'll see if they're available.",
    # CAL-only
    "Can you tell me your name and why you're calling?",
]
SCR_THIRD = ["Please stay on the line.", "I'll pass it along.", "They'll see what you say on their screen.",
             # CAL-only
             "One moment after you answer."]
N_CAL_SCR_F, N_CAL_SCR_S, N_CAL_SCR_T = 1, 1, 1

NIS_FIRST = [
    "The number you have dialed is not in service.", "We're sorry, the number you have reached is not in service.",
    "The number you have called has been disconnected or is no longer in service.",
    "We're sorry, your call cannot be completed as dialed.", "The number you have dialed is not a working number.",
    # CAL-only
    "This number is no longer in service.",
]
NIS_SECOND = [
    "Please check the number and try again.",
    "If you feel you have reached this recording in error, please check the number and try your call again.",
    "Please hang up and try again.", "This is a recording.",
    # CAL-only
    "Please verify the number and dial again.",
]
N_CAL_NIS_F, N_CAL_NIS_S = 1, 1


def split_list(lst, n_cal, split):
    """Train part = all but the last n_cal entries; CAL part = the last n_cal entries."""
    return lst[: len(lst) - n_cal] if split == "train" else lst[len(lst) - n_cal:]


def _pick(rng, lst, n_cal, split):
    part = split_list(lst, n_cal, split)
    i = rng.randrange(len(part))
    offset = 0 if split == "train" else len(lst) - n_cal
    return part[i], offset + i


def _biz(rng, split):
    p, _ = _pick(rng, BIZ_PLACES, N_CAL_PLACES, split)
    t, _ = _pick(rng, BIZ_TYPES, N_CAL_TYPES, split)
    return f"{p} {t}"


def _fill(text, rng, split, ctx):
    if "{name}" in text and "name" not in ctx:
        ctx["name"], _ = _pick(rng, NAMES, N_CAL_NAMES, split)
    if "{name2}" in text and "name2" not in ctx:
        while True:
            n2, _ = _pick(rng, NAMES, N_CAL_NAMES, split)
            if n2 != ctx.get("name"):
                ctx["name2"] = n2
                break
    if "{biz}" in text and "biz" not in ctx:
        ctx["biz"] = _biz(rng, split)
    return text.format(**{k: ctx.get(k, "") for k in ("name", "name2", "biz")})


def mentions_tone(text: str) -> bool:
    t = text.lower()
    return "tone" in t or "beep" in t


def make_script(cls: str, rng: random.Random, split: str) -> tuple[list[dict], int, list[str]]:
    """-> (segments [{"text", "kind"}], cue_idx, template ids). Pure function of (cls, rng state, split)."""
    ctx: dict = {}
    segs: list[dict] = []
    tids: list[str] = []

    def add(lst, n_cal, kind, tag):
        t, i = _pick(rng, lst, n_cal, split)
        segs.append({"text": _fill(t, rng, split, ctx), "kind": kind})
        tids.append(f"{tag}{i}")

    if cls == "live_person":
        add(LIVE_GREETINGS, N_CAL_LIVE_G, "speech", "lg")
        if rng.random() < 0.6:
            add(LIVE_FOLLOWUPS, N_CAL_LIVE_F, "followup", "lf")
            if rng.random() < 0.35:
                add(LIVE_FOLLOWUPS, N_CAL_LIVE_F, "followup", "lf")
        return segs, 0, tids
    if cls == "personal_voicemail":
        add(PVM_FIRST, N_CAL_PVM_F, "speech", "pf")
        if rng.random() < 0.6:
            add(PVM_MIDDLE, N_CAL_PVM_M, "speech", "pm")
        add(PVM_END, N_CAL_PVM_E, "speech", "pe")
        return segs, 0, tids
    if cls == "carrier_voicemail":
        add(CVM_FIRST, N_CAL_CVM_F, "speech", "cf")
        if rng.random() < 0.4:
            add(CVM_MIDDLE, N_CAL_CVM_M, "speech", "cm")
        add(CVM_END, N_CAL_CVM_E, "speech", "ce")
        return segs, 0, tids
    if cls == "business_ivr":
        add(IVR_GREET, N_CAL_IVR["greet"], "speech", "ig")
        if rng.random() < 0.5:
            add(IVR_PREAMBLE, N_CAL_IVR["pre"], "speech", "ip")
        n_items = rng.randint(2, 4)
        topics = split_list(IVR_TOPICS, N_CAL_IVR["topic"], split)
        verbs = split_list(IVR_VERBS, N_CAL_IVR["verb"], split)
        forms = split_list(IVR_ITEM_FORMS, N_CAL_IVR["form"], split)
        used_t = rng.sample(range(len(topics)), k=min(n_items, len(topics)))
        used_v = rng.sample(range(len(verbs)), k=min(n_items, len(verbs)))
        for k in range(n_items):
            fi = rng.randrange(len(forms))
            text = forms[fi].format(topic=topics[used_t[k % len(used_t)]], verb=verbs[used_v[k % len(used_v)]],
                                    num=NUM_WORDS[k])
            segs.append({"text": text, "kind": "speech"})
            tids.append(f"if{fi + (0 if split == 'train' else len(IVR_ITEM_FORMS) - N_CAL_IVR['form'])}")
        if rng.random() < 0.5:
            add(IVR_TAIL, N_CAL_IVR["tail"], "speech", "it")
        return segs, 1, tids  # cue = segment 1 (the menu preamble or the first menu option)
    if cls == "call_screening":
        add(SCR_FIRST, N_CAL_SCR_F, "speech", "sf")
        add(SCR_SECOND, N_CAL_SCR_S, "speech", "ss")
        if rng.random() < 0.3:
            add(SCR_THIRD, N_CAL_SCR_T, "speech", "st")
        return segs, 0, tids
    if cls == "not_in_service":
        add(NIS_FIRST, N_CAL_NIS_F, "speech", "nf")
        if rng.random() < 0.7:
            add(NIS_SECOND, N_CAL_NIS_S, "speech", "ns")
        return segs, 0, tids
    return [], -1, []


# Plan: train clip counts per class; CAL adds ~3.3% per class on top.
TRAIN_COUNTS = {"live_person": 1750, "personal_voicemail": 1700, "carrier_voicemail": 1150, "business_ivr": 1350,
                "call_screening": 1050, "no_answer": 450, "busy": 450, "not_in_service": 550, "fax_modem": 400}
CAL_FRACTION = 0.033


def draw_aug(rng: random.Random, cls: str) -> dict:
    """Augmentation draws, the same distributions for every class. no_answer is never truncated (its label needs
    the whole ring-then-silence pattern); the ringback prefix is only used where something answers."""
    aug = {
        "speed": round(rng.uniform(0.85, 1.15), 3),
        "lead_s": round(rng.uniform(0.0, 1.5), 3),
        "peak": round(rng.uniform(0.2, 0.95), 3),
        "noise": rng.choice(["none", "none", "white", "pink", "hum", "pink"]) if rng.random() < 0.65 else "none",
        "noise_dbfs": round(rng.uniform(-60.0, -32.0), 1),
        "truncate": rng.random() < 0.31,
        "cut_s": round(rng.uniform(1.5, 4.0), 3),
        "ring_prefix": rng.random() < 0.15,
        "beep": rng.random() < 0.55,
        "beep_s": round(rng.uniform(0.3, 1.0), 3),
        "beep_gap_s": round(rng.uniform(0.15, 0.8), 3),
        "tail_s": round(rng.uniform(0.2, 1.5), 3),
        "phone": rng.random() < 0.8,
    }
    if cls == "no_answer":
        aug["truncate"] = False
    if cls not in ("live_person", "personal_voicemail", "carrier_voicemail", "business_ivr", "call_screening",
                   "fax_modem"):
        aug["ring_prefix"] = False
    if cls not in VOICEMAIL:
        aug["beep"] = False
    return aug


def build_plan(seed: int = 0, scale: float = 1.0) -> list[dict]:
    """All clips (train + CAL). Voices go round-robin over a per-class shuffled order, so every voice appears
    about equally often in every speech class."""
    plan = []
    for split in ("train", "CAL"):
        voices = TRAIN_VOICES if split == "train" else CAL_VOICES
        for ci, cls in enumerate(CLASSES):
            n = TRAIN_COUNTS[cls]
            n = max(2, round(n * scale)) if split == "train" else max(2, round(n * scale * CAL_FRACTION / (1 - CAL_FRACTION)) + 1)
            rng = random.Random(f"{seed}-{split}-{cls}")
            order = list(voices)
            rng.shuffle(order)
            for j in range(n):
                segs, cue, tids = make_script(cls, rng, split)
                aug = draw_aug(rng, cls)
                voice = order[j % len(order)] if cls in SPEECH_CLASSES else None
                if cls in SPEECH_CLASSES and j % len(order) == len(order) - 1:
                    rng.shuffle(order)
                plan.append({
                    "key": f"{'t' if split == 'train' else 'c'}{ci}_{j:05d}",
                    "split": split, "cls": cls, "voice": voice, "segments": segs, "cue_idx": cue,
                    "templates": tids, "aug": aug, "tone_seed": rng.randrange(2**31),
                })
    return plan
