"""sb_aicall line templates, voices and HVB text clean-up. All businesses and people are fictional.

Voicemail greetings and IVR menus come from data/sb_calls/scripts.py."""
from __future__ import annotations

import random
import re

import scripts as SC  # data/sb_calls/scripts.py

EN_VOICES = list(SC.EN_VOICES)  # 28 English Kokoro-82M voices
CAL_VOICES = ["af_sky", "bm_george", "am_puck"]  # held out for CAL
TRAIN_VOICES = [v for v in EN_VOICES if v not in CAL_VOICES]

AGENT_NAMES = ["Ava", "Mia", "Leo", "Nova", "Aria", "Max", "Ella", "Sam", "Iris", "Kai", "Luna", "Remy", "Jade",
               "Theo", "Ruby", "Finn", "Clara", "Milo", "Zara", "Owen"]
CUST_NAMES = ["Dana", "Maria", "Priya", "Jordan", "Alex", "Taylor", "Morgan", "Chris", "Elena", "Marcus", "Nina",
              "Leah", "Raj", "Sofia", "Ben", "Grace", "Hector", "Kenji", "Laura", "Miguel", "Nora", "Rosa", "Uma",
              "Victor", "Wendy", "Yusuf", "Carmen", "Derek", "Mr. Patel", "Ms. Garcia", "Mrs. Chen", "Mr. Brooks"]
BIZ = ["Bright Smile Dental", "Glow Hair Studio", "Riverside Family Clinic", "Oak Hill Dermatology", "Serenity Day Spa",
       "Maple Street Salon", "Northgate Auto Care", "Lakeview Pediatrics", "Silver Creek Physical Therapy",
       "Harbor View Eye Care", "Pinecrest Animal Hospital", "Willow Park Chiropractic", "Cedar Grove Orthodontics",
       "Sunny Day Nail Studio", "Granite Peak Tire and Auto", "Fairmont Hearing Center", "Eastwood Music School",
       "Brookfield Dental Group", "Golden Hour Photography", "Juniper Bay Wellness"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "tomorrow", "Saturday"]
TIMES = ["nine thirty", "ten a.m.", "eleven fifteen", "one o'clock", "two thirty", "three forty-five", "four p.m."]
SERVICES = ["cleaning", "consultation", "follow-up visit", "facial", "massage", "oil change", "check-up",
            "haircut", "piano lesson", "eye exam", "therapy session", "brake inspection"]

# AI-agent lines
INBOUND_GREET = [
    "Thank you for calling {biz}, this is {agent}. How can I help you today?",
    "Hi, thanks for calling {biz}. This is {agent}, the virtual assistant. What can I do for you?",
    "Hello, you've reached {biz}. My name is {agent}. How may I help you?",
    "Good morning, {biz}, this is {agent} speaking. How can I help?",
    "Thanks for calling {biz}! I'm {agent}, an automated assistant. How can I help you today?",
    "Hi there, this is {agent} at {biz}. What can I help you with?",
    "{biz}, this is {agent}. How can I assist you?",
    "Hello and thank you for calling {biz}. I'm {agent}. Are you calling to book, change, or cancel an appointment?",
]
OUTBOUND_OPEN = [
    "Hi, this is {agent} calling from {biz}. Am I speaking with {cust}?",
    "Hello, this is {agent}, a virtual assistant from {biz}, calling for {cust}.",
    "Hi {cust}, it's {agent} from {biz}, calling about your {service} on {day}.",
    "Good afternoon, this is {agent} with {biz}. I'm calling to confirm your appointment for {day} at {time}.",
    "Hi, this is {agent} from {biz}, returning your call. How can I help you today?",
    "Hello, this is {agent} calling on behalf of {biz}. Is now a good time to talk?",
    "Hi there, this is {agent} from {biz}. I'm following up on your recent visit.",
    "Hello, may I speak with {cust}? This is {agent} from {biz}.",
]
CALLBACK_OPEN = [  # outbound opener that fits a following HVB caller request (callback framing)
    "Hi, this is {agent} from Harper Valley Bank, returning your call. How can I help you today?",
    "Hello, this is {agent}, the virtual assistant at Harper Valley Bank, calling you back. What can I do for you?",
    "Hi, it's {agent} from Harper Valley Bank. You asked for a call back. How can I help?",
    "Good afternoon, this is {agent} calling from Harper Valley Bank about your earlier request. What do you need?",
    "Hello, this is {agent} with Harper Valley Bank, returning your call from this morning. How can I help?",
]
VM_MESSAGE = [
    "Hi {cust}, this is {agent} from {biz}. Sorry we missed you. We're calling to confirm your {service} on {day} "
    "at {time}. Please give us a call back when you can. Thanks, bye!",
    "Hello, this message is for {cust}. This is {agent} calling from {biz} about your upcoming appointment. "
    "Please call us back at your convenience. Have a great day.",
    "Hi {cust}, it's {agent} with {biz}. Just a friendly reminder about your {service} on {day}. "
    "If you need to reschedule, give us a call. Thank you!",
    "Hi, this is {agent} from {biz} returning your call. We'd be happy to help you book a {service}. "
    "Please call us back, or reply to our text. Bye for now.",
    "Hello {cust}, this is {agent}, the virtual assistant for {biz}. We have an opening on {day} at {time} if "
    "you'd like it. Call us back to confirm. Thanks!",
    "Hi {cust}, {agent} here from {biz}. Your {service} is coming up on {day}. Please call us if anything changes. "
    "Take care.",
]
ARE_YOU_THERE = [
    "Hello? Are you there?", "Hello? I can't hear you.", "Hi, is anyone there?", "Hello? Can you hear me?",
    "Sorry, I didn't catch that. Are you still on the line?", "Hello? I'm not hearing anything.",
    "Are you still there?", "Hello? This is {agent} from {biz}.",
]
GOODBYE_NOBODY = [
    "It seems we have a bad connection. Please call us back. Goodbye.",
    "I'm not able to hear you, so I'll end the call now. Please feel free to call again. Goodbye.",
    "I'll try you again later. Have a nice day.", "Sorry, I can't hear anything. I'll hang up now. Goodbye.",
]
ROBO_REPLY = [
    "Sorry, I didn't catch that. How can I help you today?", "Hello? Is there something I can help you with?",
    "I'm sorry, could you repeat that?", "Hello? Can I help you book an appointment?",
    "I'm sorry, I'm not able to help with that. Is there anything else?",
    "It sounds like this might be a recording. I'll end the call now. Goodbye.",
]
IVR_NAV = [
    "Representative.", "Operator, please.", "Hi, I'm calling from {biz} to confirm an appointment.",
    "Front desk, please.", "Speak to a person.", "Hi, this is {agent} calling on behalf of {biz}.",
    "Agent.", "I'd like to speak with someone about an existing appointment.",
]
TALKOVER_VM = [  # AI mistakes the start of a greeting for a person and starts its opener
    "Hi, is this {cust}?", "Hi {cust}, this is {agent} from {biz}.", "Hello! This is {agent} calling from {biz}.",
    "Hi there, am I speaking with {cust}?",
]

# Automated inbound callers
AUTO_CALLER = [
    "This is an automated message from {pharm}. A prescription is ready for pickup. To hear this message again, "
    "press one.",
    "Hello. This is an automated courtesy call from {util}. Service in your area may be interrupted on {day} "
    "for scheduled maintenance. Thank you.",
    "This is a recorded message from the {school} office. School will close early on {day}. No action is needed.",
    "Hello, this is an automated reminder from {lab}. Your test results are available online. Goodbye.",
    "This is an automated call from {util}. We are calling to confirm a delivery scheduled for {day}. "
    "Press one to confirm, or press two to reschedule.",
    "Attention. This is a pre-recorded message from {pharm}. Our store hours will change starting {day}.",
]
PHARM = ["Main Street Pharmacy", "Corner Drug", "Valley Health Pharmacy"]
UTIL = ["City Water and Power", "Greenline Delivery", "Metro Gas"]
SCHOOL = ["Lincoln Elementary", "Westside Middle School"]
LAB = ["Clearview Labs", "Northside Diagnostics"]


def fill(t: str, rng: random.Random, ctx: dict | None = None) -> str:
    ctx = ctx if ctx is not None else {}
    for k, lst in (("agent", AGENT_NAMES), ("cust", CUST_NAMES), ("biz", BIZ), ("day", DAYS), ("time", TIMES),
                   ("service", SERVICES), ("pharm", PHARM), ("util", UTIL), ("school", SCHOOL), ("lab", LAB)):
        if "{" + k + "}" in t and k not in ctx:
            ctx[k] = rng.choice(lst)
    return t.format(**{k: ctx.get(k, "") for k in ("agent", "cust", "biz", "day", "time", "service", "pharm",
                                                    "util", "school", "lab")})


def n_agent_templates() -> int:
    return sum(len(x) for x in (INBOUND_GREET, OUTBOUND_OPEN, CALLBACK_OPEN, VM_MESSAGE, ARE_YOU_THERE,
                                GOODBYE_NOBODY, ROBO_REPLY, IVR_NAV, TALKOVER_VM))


def other_script(kind: str, rng: random.Random) -> list[str]:
    """Other-side texts via sb_calls templates (train-side template lists; CAL is split by voice here)."""
    cls = {"pvm": "personal_voicemail", "cvm": "carrier_voicemail", "ivr": "business_ivr"}[kind]
    segs, _cue, _t = SC.make_script(cls, rng, "train")
    return [s["text"] for s in segs]


# HVB transcript text -> TTS text
_TAG = re.compile(r"\[[^\]]*\]|<[^>]*>")
_FILL = re.compile(r"\b(uhm|um+|uh+|hmm+|mm+|erm|er|ah)\b")
_Q = ("what", "how", "can", "could", "would", "is", "are", "do", "does", "did", "may", "will", "when", "where",
      "which", "who", "why", "should", "have", "was")


def clean_text(t: str) -> str:
    t = _TAG.sub(" ", t.lower())
    t = _FILL.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip(" ,.")
    return t


def tts_sentence(t: str) -> str:
    t = clean_text(t)
    if not t:
        return ""
    t = re.sub(r"\bi\b", "I", t)
    t = re.sub(r"\bi'", "I'", t)
    first = t.split(" ", 1)[0]
    end = "?" if first in _Q else "."
    return t[0].upper() + t[1:] + end


def turn_text(seg_texts: list[str]) -> str:
    return " ".join(s for s in (tts_sentence(x) for x in seg_texts) if s)


BACKCHANNEL_WORDS = {"okay", "ok", "mhm", "mm-hm", "uh-huh", "yeah", "yes", "right", "sure", "alright", "great",
                     "perfect", "yep", "cool"}
