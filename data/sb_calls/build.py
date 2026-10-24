"""sb_calls labels, questions and items: up to 4 questions per clip, yes/no balanced 50/50 per question.

Label rules are in `label_yes`; `who` is a choice over the 9 classes (or a 4-6 option subset containing the gold)."""
from __future__ import annotations

import random
from collections import Counter, defaultdict

from scripts import CLASSES, OPTION_TEXT, VOICEMAIL

SOURCE_TTS = "sb_calls_tts"
SOURCE_TONES = "sb_calls_tones"
MAX_ITEMS_PER_CLIP = 4

QUESTIONS = {
    "who": [
        "Who or what answered the call?", "What picked up on the other end of this call?",
        "Which best describes what the caller reached?", "What is on the other end of the line?",
        "Listen to the start of this outbound call. What answered?", "How was this call answered?",
        "What did this phone call connect to?",
    ],
    "live": [
        "Is a live person on the line?", "Did a real person answer?", "Is there a human speaking live on this call?",
        "Has a live human picked up the phone?", "Is the caller connected to a live person right now?",
        "Is someone actually on the line in real time?",
    ],
    "vm": [
        "Is this a voicemail greeting?", "Did the call go to voicemail?", "Is this a recorded voicemail message?",
        "Has the call reached a voicemail box?", "Is the audio a voicemail greeting rather than a live answer?",
        "Is this the greeting of a voicemail box?",
    ],
    "auto": [
        "Is this an automated system?", "Did an automated phone system answer, such as a menu or a call screener?",
        "Is the caller hearing an automated menu or assistant?", "Is a machine-run phone system handling the call?",
        "Is this an automated attendant rather than a person?", "Is an automated system talking to the caller?",
    ],
    "tone": [
        "Has the voicemail tone played?", "Did the beep sound yet?", "Has the voicemail beep already played?",
        "Was the record tone heard at the end?", "Has the tone that starts the recording played?",
        "Did the greeting end with the beep?",
    ],
    "leave": [
        "Should the caller start leaving a message now?", "Is it time to start speaking the voicemail message?",
        "Is the voicemail recording now, ready for the caller's message?",
        "Can the caller begin the message at this point?", "Should the caller begin talking now to leave a message?",
    ],
    "busy": [
        "Is the line busy or out of service?",
        "Did the call fail because the line is busy or the number is not in service?",
        "Is this a busy signal or a disconnected-number message?",
        "Is the call unable to go through because of a busy line or a dead number?",
        "Is the caller hearing a busy or out-of-service signal?",
    ],
}

# Hard negatives first in the round-robin of each "no" pool.
NO_POOLS = {
    "live": ["personal_voicemail", "call_screening", "business_ivr", "carrier_voicemail", "not_in_service",
             "no_answer", "fax_modem", "busy"],
    "vm": ["live_person", "call_screening", "business_ivr", "not_in_service", "no_answer", "fax_modem", "busy"],
    "auto": ["live_person", "personal_voicemail"],
    "busy": ["no_answer", "carrier_voicemail", "fax_modem", "live_person", "business_ivr", "call_screening",
             "personal_voicemail"],
}


def label_yes(q: str, cls: str, facts: dict) -> bool | None:
    """True/False = the answer to yes/no question `q` for this clip; None = do not ask it on this clip."""
    if q == "live":
        return cls == "live_person"
    if q == "vm":
        return cls in VOICEMAIL
    if q == "auto":
        if cls in ("business_ivr", "call_screening"):
            return True
        return False if cls in ("live_person", "personal_voicemail") else None
    if q == "tone":
        return bool(facts.get("beep_present")) if cls in VOICEMAIL else None
    if q == "leave":
        if cls in VOICEMAIL:
            if facts.get("beep_present"):
                return True
            if facts.get("truncated") or facts.get("mentions_tone"):
                return False
            return None
        return False if cls in ("live_person", "business_ivr", "call_screening") else None
    if q == "busy":
        return cls in ("busy", "not_in_service")
    raise ValueError(q)


NOUL_ORDER = ["tone", "leave", "busy", "auto", "live", "vm"]  # scarcest first
NOUL_CAP = {"tone": 1500, "leave": 1500, "busy": 1100, "auto": 2000, "live": 1750, "vm": 2300}  # yes per q (train)


def allocate(clips: list[dict], seed: int, cap_scale: float = 1.0) -> dict[str, list[tuple[str, int, bool | None]]]:
    """clips [{key, cls, facts}] -> {clip key: [(question, paraphrase index, yes?)]}, <= MAX_ITEMS_PER_CLIP per
    clip and exactly 50/50 yes/no per question."""
    rng = random.Random(f"{seed}-alloc")
    load = Counter()
    out: dict[str, list] = defaultdict(list)
    order = list(clips)
    rng.shuffle(order)
    for q in NOUL_ORDER:
        yes, no_by_cls = [], defaultdict(list)
        for c in order:
            y = label_yes(q, c["cls"], c["facts"])
            if y is None:
                continue
            (yes if y else no_by_cls[c["cls"]]).append(c)
        no_classes = [k for k in (NO_POOLS.get(q) or sorted(no_by_cls)) if no_by_cls.get(k)]
        cap = max(1, int(NOUL_CAP[q] * cap_scale))

        def avail(c):
            return load[c["key"]] < MAX_ITEMS_PER_CLIP - 1  # keep one slot for the "who" question

        yes_av = sorted([c for c in yes if avail(c)], key=lambda c: load[c["key"]])
        no_av = {k: sorted([c for c in no_by_cls[k] if avail(c)], key=lambda c: load[c["key"]]) for k in no_classes}
        n_no_total = sum(len(v) for v in no_av.values())
        n = min(cap, len(yes_av), n_no_total)
        picked_no, ptr = [], {k: 0 for k in no_classes}
        while len(picked_no) < n:
            progressed = False
            for k in no_classes:
                if len(picked_no) >= n:
                    break
                if ptr[k] < len(no_av[k]):
                    picked_no.append(no_av[k][ptr[k]])
                    ptr[k] += 1
                    progressed = True
            if not progressed:
                break
        n = min(n, len(picked_no))
        for c, y in [(c, True) for c in yes_av[:n]] + [(c, False) for c in picked_no[:n]]:
            out[c["key"]].append((q, rng.randrange(len(QUESTIONS[q])), y))
            load[c["key"]] += 1
    for c in order:  # the "who" choice question on every clip
        out[c["key"]].append(("who", rng.randrange(len(QUESTIONS["who"])), None))
        load[c["key"]] += 1
    return out


def who_options(cls: str, rng: random.Random) -> tuple[list[str], int, list[str]]:
    """-> (options, gold index, class per option) in canonical order (make_item shuffles them)."""
    style = rng.randrange(2)
    if rng.random() < 0.6:
        classes = list(CLASSES)
    else:
        others = [c for c in CLASSES if c != cls]
        classes = sorted([cls] + rng.sample(others, rng.randint(3, 5)), key=CLASSES.index)
    return [OPTION_TEXT[c][style] for c in classes], classes.index(cls), classes


def make_items(clips: list[dict], alloc: dict, seed: int, make_item, bank_of: dict, plan_by_key: dict,
               kokoro_ref: str) -> list[dict]:
    items = []
    for c in clips:
        key, cls, facts = c["key"], c["cls"], c["facts"]
        p = plan_by_key[key]
        rng = random.Random(f"{seed}-items-{key}")
        source = SOURCE_TONES if p["voice"] is None else SOURCE_TTS
        base_meta = {
            "class": cls, "clip": key, "voice": p["voice"], "templates": p["templates"],
            "script": " | ".join(s["text"] for s in p["segments"][: facts.get("segments_kept", 0)]) or None,
            "speed": p["aug"]["speed"], "duration_s": facts["duration_s"], "truncated": facts["truncated"],
            "beep_present": facts["beep_present"], "ring_prefix": facts["ring_prefix"], "noise": p["aug"]["noise"],
            "variant": facts.get("variant"),
            "licence": ("Apache-2.0 (Kokoro-82M TTS output; synthetic tones)" if p["voice"] else
                        "synthetic (numpy-generated tones, no external data)"),
            "tts": kokoro_ref if p["voice"] else None,
        }
        channel = "phone" if p["aug"]["phone"] else "clean"
        audio = {"bank": bank_of[key], "key": key}
        for q, pi, y in alloc.get(key, []):
            item_id = f"sb_calls_{key}_{q}"
            question = QUESTIONS[q][pi]
            if q == "who":
                opts, gold, classes = who_options(cls, rng)
                target = [1.0 if i == gold else 0.0 for i in range(len(opts))]
                meta = dict(base_meta, qtype="choice", qname="who", option_classes=classes)
                it = make_item(item_id=item_id, set_name=c["split"], source=source, question=question,
                               options=opts, target=target, gold=gold, rng=rng, audio=audio, channel=channel,
                               meta=meta)
                # keep option_classes aligned with the shuffled options
                it["meta"]["option_classes"] = [classes[opts.index(o)] for o in it["options"]]
            else:
                gold = 0 if y else 1
                meta = dict(base_meta, qtype="noul", qname=q, answer="yes" if y else "no")
                it = make_item(item_id=item_id, set_name=c["split"], source=source, question=question,
                               options=["yes", "no"], target=[1.0, 0.0] if y else [0.0, 1.0], gold=gold, rng=rng,
                               audio=audio, channel=channel, meta=meta)
            items.append(it)
    return items


def validate(items: list[dict], bank_index: dict[str, dict]) -> dict:
    """Schema + label sanity checks. Returns a report; raises on hard failures."""
    keys = {"id", "set", "source", "question", "options", "target", "gold", "audio", "channel", "state_text", "meta"}
    ids = set()
    bal = defaultdict(Counter)
    for it in items:
        assert keys <= set(it), it["id"]
        assert it["id"] not in ids, it["id"]
        ids.add(it["id"])
        assert it["set"] in ("train", "CAL")
        assert 2 <= len(it["options"]) <= 20 and len(it["target"]) == len(it["options"])
        assert abs(sum(it["target"]) - 1) < 1e-6
        assert it["gold"] == max(range(len(it["target"])), key=lambda i: it["target"][i])
        assert it["channel"] in ("clean", "phone")
        assert it["meta"]["qtype"] in ("choice", "noul")
        if it["meta"]["qtype"] == "noul":
            assert sorted(it["options"]) == ["no", "yes"]
            bal[(it["set"], it["meta"]["qname"])][it["options"][it["gold"]]] += 1
        a = it["audio"]
        assert a["key"] in bank_index[a["bank"]], a
    per_clip = Counter(it["audio"]["key"] for it in items)
    assert max(per_clip.values()) <= MAX_ITEMS_PER_CLIP
    return {"n_items": len(items), "max_items_per_clip": max(per_clip.values()),
            "noul_balance": {f"{s}/{q}": dict(v) for (s, q), v in sorted(bal.items())}}
