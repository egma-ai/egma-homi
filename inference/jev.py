"""
From probabilities to a Jev-shaped answer.

`systemone(state, questions, ...)` follows TypeSafe's request/response shape
(POST https://api.typesafe.ai/v1/systemone; field names as documented at https://docs.typesafe.ai/api
on 2026-10-06).

Request
  state      str | object | array, as in TypeSafe. Our audio extension:
             {"audio": <16 kHz float array or a path to an audio file>, "text": optional str/object}
  questions  {name: {"type": "noul" | "choice" | "score", "instructions": ..., "criteria": ...}}
             noul   criteria optional {"true": ..., "false": ...}
             choice criteria {option: description or null}   (2..255 options = our 255 answer codes)
             score  criteria [level 0, level 1, ...]           (2..10 ordered levels)
Response
  {"model": ..., "answers": {name: answer}, "usage": {"input_tokens": n, "output_tokens": 0}}
             noul   {"type": "noul", "noul": P(yes)}
             choice {"type": "choice", "choice": top option, "probabilities": {option: p}, "confidence": max p}
             score  {"type": "score", "score": sum_i i * p_i, "legend": {"0": level, ...},
                     "probabilities": {"0": p, ...}, "confidence": max p}

Notes
  * Score levels are 0-based, as in TypeSafe's docs (their example: legend {"0": "Calm", "1": ...},
    score 1.05). "confidence" here is simply the top probability; TypeSafe says theirs is "derived
    from the probability distribution" without giving the formula.
  * Each question is one letter-readout prompt (audio first). All questions of a request run in one
    batched forward pass; each one re-reads the audio.
"""

from __future__ import annotations

import json

import numpy as np

import common as C

MAX_CHOICE_OPTIONS = C.N_ANSWER_CODES  # 255, same cap as TypeSafe
SCORE_LEVELS = (2, 10)

EXAMPLE_QUESTIONS = {  # a small example request
    "sounds_angry": {"type": "noul", "instructions": "Does the speaker sound angry?"},
    "emotion": {
        "type": "choice",
        "instructions": "Which emotion does the speaker express?",
        "criteria": {"anger": None, "disgust": None, "fear": None, "happy": None, "neutral": None, "sad": None},
    },
    "intensity": {
        "type": "score",
        "instructions": "How intense is the emotion in the speaker's voice?",
        "criteria": ["Calm, no clear emotion", "Some emotion", "Strong emotion"],
    },
}


def _text(x) -> str:
    """TypeSafe fields may be a string, an object or an array; objects are shown to the model as JSON."""
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def _option(key: str, description) -> str:
    return key if description is None else f"{key}: {_text(description)}"


def load_audio_file(path) -> np.ndarray:
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return C.resample(x.mean(axis=1), sr, C.SAMPLE_RATE)


def parse_state(state) -> tuple[np.ndarray | None, str | None]:
    """-> (16 kHz waveform or None, text state or None)."""
    if isinstance(state, dict) and "audio" in state:
        audio = state["audio"]
        if not isinstance(audio, np.ndarray):
            audio = load_audio_file(audio)
        audio = np.asarray(audio, dtype=np.float32)[: C.MAX_AUDIO_SECONDS * C.SAMPLE_RATE]
        text = state.get("text")
        return audio, None if text is None else _text(text)
    return None, _text(state)


def question_to_item(name: str, question: dict, has_audio: bool, state_text: str | None) -> tuple[dict, list[str]]:
    """One TypeSafe question -> (an item for the letter readout, answer keys in option order)."""
    qtype = question.get("type")
    instructions = _text(question["instructions"])
    criteria = question.get("criteria")
    if qtype == "noul":
        criteria = criteria or {}
        options = [_option("yes", criteria.get("true")), _option("no", criteria.get("false"))]
        keys = ["yes", "no"]
    elif qtype == "choice":
        if not criteria or not 2 <= len(criteria) <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"{name}: a choice needs 2..{MAX_CHOICE_OPTIONS} options in criteria")
        keys = list(criteria)
        options = [_option(k, v) for k, v in criteria.items()]
    elif qtype == "score":
        if not criteria or not SCORE_LEVELS[0] <= len(criteria) <= SCORE_LEVELS[1]:
            raise ValueError(f"{name}: a score needs {SCORE_LEVELS[0]}..{SCORE_LEVELS[1]} levels in criteria")
        keys = [str(i) for i in range(len(criteria))]
        options = [_text(level) for level in criteria]
    else:
        raise ValueError(f"{name}: unknown question type {qtype!r} (noul | choice | score)")
    item = {"id": name, "set": "request", "question": instructions, "options": options,
            "audio": {"bank": "request"} if has_audio else None, "state_text": state_text}
    return item, keys


def probs_to_answer(qtype: str, keys: list[str], probs, criteria=None) -> dict:
    """Probabilities over the options (same order as keys) -> a TypeSafe-shaped answer."""
    p = np.asarray(probs, dtype=float)
    p = p / p.sum()
    if qtype == "noul":
        return {"type": "noul", "noul": round(float(p[0]), 4)}  # option 0 is "yes"
    top = int(np.argmax(p))
    probabilities = {k: round(float(v), 4) for k, v in zip(keys, p)}
    confidence = round(float(p[top]), 4)
    if qtype == "choice":
        return {"type": "choice", "choice": keys[top], "probabilities": probabilities, "confidence": confidence}
    if qtype == "score":
        return {"type": "score", "score": round(float(np.dot(np.arange(len(p)), p)), 4),
                "legend": {str(i): _text(level) for i, level in enumerate(criteria)},
                "probabilities": probabilities, "confidence": confidence}
    raise ValueError(f"unknown question type {qtype!r}")


def systemone(state, questions: dict, *, model, head, processor, temperature: float = 1.0,
              model_name: str = "homi") -> dict:
    """Answer every question about one state through the generic common.encode_batch path. For homi on
    Qwen3-Omni use predict.AudioDecider.ask, which returns the same shape. `temperature`: the fitted one."""
    import torch

    import readout

    audio, text = parse_state(state)
    built = [(name, q, *question_to_item(name, q, audio is not None, text)) for name, q in questions.items()]
    items = [item for _, _, item, _ in built]
    enc = C.encode_batch(processor, items, [audio] * len(items))
    with torch.inference_mode():
        log_probs, _ = readout.decision_log_probs(model, head, enc, [len(it["options"]) for it in items])
    log_probs = log_probs.float().cpu().numpy()
    answers = {}
    for (name, q, item, keys), row in zip(built, log_probs):
        probs = C.apply_temperature(row[: len(keys)], temperature)
        answers[name] = probs_to_answer(q["type"], keys, probs, q.get("criteria"))
    return {"model": model_name, "answers": answers,
            "usage": {"input_tokens": int(enc["attention_mask"].sum()), "output_tokens": 0}}
