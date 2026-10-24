"""Phone-family label logic, copied from data/sb_phone/labels.py with the same rules plus extra question
paraphrases: transcript rules, degradation recipes, labels by construction. No I/O."""
from __future__ import annotations

import random
import re

# Transcript rules
NUM_WORDS = {
    "ZERO", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "NINER", "TEN", "ELEVEN", "TWELVE",
    "THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN", "TWENTY", "THIRTY", "FORTY",
    "FOURTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY", "HUNDRED", "THOUSAND", "MILLION", "BILLION",
}
# "ONE" is also a pronoun ("this one", "no one"): it only counts when it stands next to another number word,
# after "NUMBER", or as a digit.
ONE_AMBIG = "ONE"
TOKEN_RE = re.compile(r"[A-Z0-9']+")


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").upper())


def number_label(text: str) -> bool | None:
    """True = the speaker clearly says a number; False = clearly no number; None = ambiguous (not asked).
    Ambiguous: a lone 'ONE' (pronoun or number?), 'OH'/'O' only (letter or zero?), 'HALF', 'DOZEN', 'COUPLE'."""
    t = tokens(text)
    if not t:
        return None
    if any(any(ch.isdigit() for ch in w) for w in t):
        return True
    if any(w in NUM_WORDS for w in t):
        return True
    for i, w in enumerate(t):
        if w == ONE_AMBIG:
            prev = t[i - 1] if i else ""
            nxt = t[i + 1] if i + 1 < len(t) else ""
            if prev in NUM_WORDS or nxt in NUM_WORDS or prev in ("NUMBER", "ONE") or nxt == ONE_AMBIG:
                return True
            return None
    if any(w in ("OH", "O", "HALF", "DOZEN", "COUPLE", "FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH", "SINGLE", "DOUBLE",
                 "TRIPLE", "NUMBER", "NUMBERS", "ONCE", "TWICE", "EIGHTH", "NINTH", "TENTH", "HUNDREDS", "THOUSANDS")
           for w in t):
        return None
    return False


MONTHS = {"JANUARY", "FEBRUARY", "APRIL", "JUNE", "JULY", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"}
WEEKDAYS = {"MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"}
DATE_AMBIG = {"MAY", "MARCH", "AUGUST", "TODAY", "TOMORROW", "YESTERDAY", "TONIGHT", "WEEK", "WEEKEND", "MONTH",
              "DATE", "DAY", "DAYS", "NIGHT", "MORNING", "AFTERNOON", "EVENING", "MIDNIGHT", "NOON", "TIME", "HOUR",
              "HOURS", "MINUTE", "MINUTES", "MONDAYS", "SUNDAYS", "SATURDAYS", "FRIDAYS", "YEAR", "YEARS"}
ORDINAL_DAY = {"FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH", "SIXTH", "SEVENTH", "EIGHTH", "NINTH", "TENTH",
               "ELEVENTH", "TWELFTH", "THIRTEENTH", "FOURTEENTH", "FIFTEENTH", "TWENTIETH", "THIRTIETH"}


def date_label(text: str) -> bool | None:
    """True = a calendar date or day is named (a month name, or a weekday); False = no date/time-like words at all;
    None = ambiguous ('may', 'today', ordinals, times...)."""
    t = tokens(text)
    if not t:
        return None
    if any(w in MONTHS or w in WEEKDAYS for w in t):
        return True
    if any(w in DATE_AMBIG or w in ORDINAL_DAY or w.endswith("TH") and w[:-2] in NUM_WORDS for w in t):
        return None
    return False


# Recipes
PHONE_CODECS = ["mulaw", "alaw", "gsm", "opus", "g726", "speex"]
LEVELS = ["very poor", "some problems", "clear"]  # score order: low -> high quality

# scenario -> (quality level or None when not a quality scenario)
SCENARIOS_CLEAN = {
    "clean": "clear", "noise_mod": "some problems", "noise_sev": "very poor", "drop_mod": "some problems",
    "drop_sev": "very poor", "clip_mod": "some problems", "clip_sev": "very poor", "lowvol": "some problems",
    "far": "some problems", "combo_sev": "very poor", "music": None, "talker": None,
}
SCENARIO_W_CLEAN = {"clean": 2.6, "noise_mod": 1.0, "noise_sev": 1.0, "drop_mod": 1.0, "drop_sev": 1.0,
                    "clip_mod": 0.6, "clip_sev": 0.7, "lowvol": 0.7, "far": 1.3, "combo_sev": 0.6, "music": 1.3,
                    "talker": 1.3}
# real phone recordings (unknown base quality): only one-sided, by-construction impairments
SCENARIO_W_REAL = {"orig": 3.0, "drop_mod": 0.8, "drop_sev": 0.8, "music": 1.0, "talker": 1.0, "noise_mod": 0.7}


def sample_codec(rng: random.Random, p_none: float) -> dict:
    if rng.random() < p_none:
        return {}
    c = rng.choice(PHONE_CODECS)
    d = {"codec": c}
    if c == "opus":
        d["bitrate_k"] = rng.choice([6, 8, 10, 12])
    elif c == "g726":
        d["bitrate_k"] = rng.choice([24, 32])
    return d


def sample_recipe(scenario: str, rng: random.Random, p_codec_none: float = 0.3) -> dict:
    """One clip's degradation recipe. Parameter ranges leave gaps between levels so labels are unambiguous:
    noise SNR 12-20 dB = 'some problems', -3..4 dB = 'very poor'; dropouts 4-9 % of speech time = mild,
    15-30 % = heavy; clipping 2-5 % of samples = moderate, 10-25 % = severe; quiet = peak -34..-42 dBFS."""
    r = {"scenario": scenario, **sample_codec(rng, p_codec_none), "peak_db": round(rng.uniform(-9.0, -1.0), 1)}
    sev_pick = None
    if scenario == "combo_sev":
        sev_pick = rng.choice(["noise_sev", "drop_sev", "clip_sev"])
        mod_pick = rng.choice([m for m in ("noise_mod", "drop_mod", "lowvol", "far") if m[:4] != sev_pick[:4]])
        parts = [sev_pick, mod_pick]
    else:
        parts = [scenario]
    for p in parts:
        if p == "noise_mod":
            r["noise"] = {"snr": round(rng.uniform(12.0, 20.0), 1)}
        elif p == "noise_sev":
            r["noise"] = {"snr": round(rng.uniform(-3.0, 4.0), 1)}
        elif p == "drop_mod":
            r["drop"] = round(rng.uniform(0.04, 0.09), 3)
        elif p == "drop_sev":
            r["drop"] = round(rng.uniform(0.15, 0.30), 3)
            r["plc"] = 0.3 if rng.random() < 0.3 else 0.0
        elif p == "clip_mod":
            r["clip"] = round(rng.uniform(0.02, 0.05), 3)
        elif p == "clip_sev":
            r["clip"] = round(rng.uniform(0.10, 0.25), 3)
        elif p == "lowvol":
            r["lowvol_db"] = round(rng.uniform(-42.0, -34.0), 1)
        elif p == "far":
            r["rir"] = "far"
        elif p == "music":
            r["music"] = {"snr": round(rng.uniform(0.0, 12.0), 1)}
        elif p == "talker":
            r["talker"] = {"snr": round(rng.uniform(6.0, 14.0), 1)}
    return r


def quality_level(recipe: dict) -> str | None:
    """By-construction call-audio quality. None when the clip has music / a background talker (not a 'quality'
    impairment) or an unexpected combination."""
    if recipe.get("music") or recipe.get("talker"):
        return None
    sev, mod = 0, 0
    n = recipe.get("noise")
    if n:
        if n["snr"] <= 5:
            sev += 1
        elif n["snr"] >= 12:
            mod += 1
        else:
            return None
    d = recipe.get("drop", 0)
    if d:
        if d >= 0.15:
            sev += 1
        elif 0.04 <= d <= 0.09:
            mod += 1
        else:
            return None
    c = recipe.get("clip", 0)
    if c:
        if c >= 0.10:
            sev += 1
        elif c <= 0.05:
            mod += 1
        else:
            return None
    if recipe.get("lowvol_db") is not None:
        mod += 1
    if recipe.get("rir"):
        mod += 1
    if sev:
        return "very poor"
    if mod == 1:
        return "some problems"
    if mod == 0:
        return "clear"
    return None


def applicable(recipe: dict, base_clean: bool) -> dict:
    """Yes/no or level answers this clip can be asked about (by construction). base_clean=False for real phone
    recordings, whose own noise/quality is unknown: only one-sided added impairments are asked there."""
    out = {}
    out["breakup"] = bool(recipe.get("drop", 0) >= 0.04) if (recipe.get("drop", 0) >= 0.04 or not recipe.get("drop")) else None
    out["music"] = bool(recipe.get("music"))
    out["talker"] = bool(recipe.get("talker"))
    if base_clean:
        q = quality_level(recipe)
        if q is not None:
            out["quality"] = q
            if q in ("clear", "very poor"):
                out["poor"] = q == "very poor"
                out["voice_clear"] = q == "clear"
        if recipe.get("noise"):
            out["noise"] = True
        elif not recipe.get("music") and not recipe.get("talker"):
            out["noise"] = False
        out["far"] = bool(recipe.get("rir"))
    return {k: v for k, v in out.items() if v is not None}


# Paraphrases
Q = {
    "live_noul": ["Is this a live person speaking?", "Is a real person talking on this call (not a recording)?",
                  "Is the caller a live human rather than an automated message?",
                  "Is this a live person on the line?", "Am I hearing a live person here?",
                  "Is a human speaking live on this phone call?"],
    "auto_noul": ["Is this an automated or recorded message?", "Is this a robocall or prerecorded message?",
                  "Is the audio an automated recording rather than a live person?",
                  "Is this call playing a recorded message?", "Is a machine or recording speaking on this call?",
                  "Is this a prerecorded or automated call?"],
    "live_choice": ["Is this a live person or an automated/recorded message?", "Who is speaking on this call?",
                    "What is on the line?", "Is the speaker a live person or a recording?",
                    "Live caller or automated message?", "What kind of call audio is this?"],
    "lang_choice": ["What language is the caller speaking?", "Which language is spoken in this call?",
                    "What language is this?", "In which language is the speaker talking?",
                    "Which language do you hear?", "What language is the person on the phone using?"],
    "lang_noul": ["Is the caller speaking {lang}?", "Is this call in {lang}?", "Is the speaker talking in {lang}?",
                  "Is the language spoken here {lang}?", "Does the caller speak {lang} in this clip?"],
    "number_noul": ["Does the speaker mention a number?", "Is any number said in this clip?",
                    "Does the caller say a number?", "Is a number mentioned?",
                    "Does the speaker say any numbers (digits, counts or amounts)?",
                    "Do you hear a number being said?"],
    "date_noul": ["Does the speaker mention a date?", "Is a date or day of the week mentioned?",
                  "Does the caller name a day or a month?", "Is any date said in this clip?",
                  "Does the speaker refer to a specific day or date?"],
    "quality_score": ["How good is the call audio?", "Rate the audio quality of this call.",
                      "How clear is the phone line?", "How is the sound quality of this call?",
                      "How well can the speaker be heard on this line?", "Rate the call's audio quality."],
    "quality_choice": ["How good is the call audio?", "Which best describes the audio quality of this call?",
                       "How does the phone line sound?", "What is the audio quality like?",
                       "Describe the call quality.", "How is the connection quality?"],
    "poor_noul": ["Is the audio quality poor (noisy, distorted or breaking up)?", "Is this a bad phone line?",
                  "Is the call audio badly degraded?", "Is the sound quality of this call poor?",
                  "Is the connection so bad that the speaker is hard to understand?", "Is the call quality bad?"],
    "voice_clear_noul": ["Is the speaker's voice clear?", "Can the speaker be heard clearly?",
                         "Is the voice on this call clear?", "Does the caller sound clear?",
                         "Is the speech easy to hear on this line?", "Is the speaker coming through clearly?"],
    "noise_noul": ["Is there background noise?", "Is there noise in the background of this call?",
                   "Can you hear background noise behind the speaker?", "Is the call noisy?",
                   "Is there any background noise on the line?", "Does the caller have noise around them?"],
    "breakup_noul": ["Is the call breaking up (parts of speech missing)?", "Is the audio cutting in and out?",
                     "Are there dropouts in the speech?", "Is the connection choppy?",
                     "Are bits of the speech missing because the line is breaking up?",
                     "Does the call keep cutting out?"],
    "music_noul": ["Is there music playing in the background?", "Can you hear music behind the speaker?",
                   "Is music playing during this call?", "Is there background music?",
                   "Does the caller have music on in the background?", "Do you hear any music?"],
    "talker_noul": ["Is there another person talking in the background?", "Can you hear someone else speaking in the background?",
                    "Is a second voice audible behind the main speaker?", "Is anyone else talking in the background?",
                    "Are there background voices on this call?", "Is another person speaking behind the caller?"],
    "far_noul": ["Is the speaker far from the phone (speakerphone/echo)?", "Does the caller sound like they are on speakerphone?",
                 "Is there room echo, as if the speaker is far from the microphone?",
                 "Does the speaker sound distant or echoey?", "Is the caller talking from across the room?",
                 "Is the speaker on speakerphone in an echoey room?"],
}
LIVE_OPTIONS = [("a live person", "an automated or recorded message"), ("live person", "recording / automated message"),
                ("a real person", "a prerecorded message"), ("live human", "automated message")]
QUALITY_OPTIONS = {"very poor": "very poor", "some problems": "some problems", "clear": "clear"}
QUALITY_OPTIONS_ALT = {"very poor": "very poor (hard to understand)", "some problems": "some problems",
                       "clear": "clear (no problems)"}
LANG_NAMES = {"en": "English", "cs-CZ": "Czech", "de-DE": "German", "es-ES": "Spanish", "fr-FR": "French",
              "it-IT": "Italian", "ko-KR": "Korean", "nl-NL": "Dutch", "pl-PL": "Polish", "pt-PT": "Portuguese",
              "ru-RU": "Russian", "zh-CN": "Chinese (Mandarin)"}


# Extra paraphrases written for sb_phone_pitch (>= 8 per family), appended to the sb_phone pools above
_EXTRA = {
    "live_noul": ["Is there an actual person speaking on the line right now?", "Is the voice on this call a live human?",
                  "Would you say a real person is talking here, not a machine?", "Is someone talking live rather than a playback?"],
    "auto_noul": ["Is the caller's audio a playback of a recorded announcement?", "Is this an automated voice message rather than a live talker?",
                  "Does this sound like a recorded or machine-generated call?", "Is a prerecorded announcement playing on this line?"],
    "live_choice": ["Is the voice on this line a live human or a recording?", "Which describes the speaker on this call?",
                    "Is this a person talking live, or an automated message?", "What is speaking on this phone line?"],
    "lang_choice": ["Which language is the person on this call speaking?", "Identify the language of the speech in this call.",
                    "The caller is speaking which language?", "What is the spoken language on this line?"],
    "lang_noul": ["Is the language of this call {lang}?", "Is the person on the line using {lang}?",
                  "Would you say this caller is speaking {lang}?", "Is {lang} the language being spoken here?"],
    "number_noul": ["Does the person say any number in this clip?", "Is a numeral or amount spoken by the caller?",
                    "Do you hear any numbers mentioned on this call?", "Does a number come up in what is said?"],
    "date_noul": ["Is a calendar date or weekday named here?", "Does the speaker name a day of the week or a month?",
                  "Is a specific date given on this call?"],
    "quality_score": ["How would you grade the audio on this phone line?", "What is the sound quality of this call like?",
                      "Grade the clarity of this phone connection.", "How usable is the audio of this call?"],
    "quality_choice": ["Which describes the sound of this phone line best?", "How would you sum up the audio quality here?",
                       "What is the state of this call's audio?", "How good is the line on this call?"],
    "poor_noul": ["Is the line so degraded that the call sounds bad?", "Does this call have poor audio quality?",
                  "Is the phone audio here of low quality?", "Is the call audio rough enough to make listening hard?"],
    "voice_clear_noul": ["Does the voice come through cleanly on this call?", "Is the speaker easy to make out on this line?",
                         "Is the caller's speech coming across clearly?", "Can you hear the person without any trouble?"],
    "noise_noul": ["Is there noise behind the speaker on this call?", "Do you hear any background noise on this line?",
                   "Is the speaker surrounded by noise?", "Is a noisy background audible during the call?"],
    "breakup_noul": ["Does the speech drop out in places?", "Are pieces of the caller's words missing because of a bad connection?",
                     "Is the line choppy, with gaps in the speech?", "Does the audio keep dropping for a moment?"],
    "music_noul": ["Is some music audible on this call?", "Do you hear music playing along with the speech?",
                   "Is there music in the background of the line?", "Is music mixed in behind the speaker?"],
    "talker_noul": ["Can you hear a second person talking behind the main speaker?", "Is there someone else's voice in the background?",
                    "Are other people talking in the background of this call?", "Is another voice audible behind the caller?"],
    "far_noul": ["Does the speaker sound far away from the microphone?", "Is the caller on speakerphone in a reverberant room?",
                 "Does the voice sound distant, with echo from the room?", "Is the person talking at a distance from the phone?"],
}
for _k, _v in _EXTRA.items():
    Q[_k] = list(Q[_k]) + [q for q in _v if q not in Q[_k]]
LIVE_OPTIONS = LIVE_OPTIONS + [("a person talking live", "a recorded or automated message"),
                               ("human, live", "machine / recording")]


def qtext(kind: str, rng: random.Random, **fmt) -> str:
    return rng.choice(Q[kind]).format(**fmt)
