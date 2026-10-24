"""
Shared helpers (plain Python + numpy):
  1. Constants
  2. Items: one decision question = state + question + lettered options + target
  3. Prompt building (the exact text the model sees)
  4. Answer codes (A, B, ..., AA, AB, ...) and batching (the decision head is in readout.py)
  5. Metrics (accuracy, chance-corrected skill, ECE, Brier, NLL) and temperature scaling
  6. Audio: 16 kHz audio banks on disk and a phone-channel (G.711 mu-law) simulation
  7. results.md builder
"""

from __future__ import annotations

import csv
import io
import json
import math
import string
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# 1. Constants
# --------------------------------------------------------------------------------------

SEED = 0
SAMPLE_RATE = 16_000  # all audio is 16 kHz mono
MAX_AUDIO_SECONDS = 30  # longer clips are cut

# Answer codes label the options: A..Z, then two-letter codes (AA, AB, ...) that are single tokens, 255 in total
# (as in Perplexity's pplx-decider). answer_codes.json is the default list; a model builds its own with
# build_answer_codes(tokenizer).
N_ANSWER_CODES = 255
ANSWER_CODES_FILE = Path(__file__).with_name("answer_codes.json")
ANSWER_CODES: list[str] = json.loads(ANSWER_CODES_FILE.read_text())["codes"]

# the audio goes first, then the text
AUDIO_FIRST = True
INSTRUCTION = "Answer with only the letter of the best option."

# CREMA-D: the six emotions, in the same order as the vote columns A,D,F,H,N,S
EMOTIONS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
VOTE_COLUMNS = ["A", "D", "F", "H", "N", "S"]
EMOTION_CODES = {"ANG": "anger", "DIS": "disgust", "FEA": "fear", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}

# Evaluation sets. CAL is the calibration split used to fit the temperature.
TEST_SETS = ["T1", "T2", "T3", "T4", "T5", "T6"]
TEXT_CONTROL_SETS = {"T1": "T1_text", "T2": "T2_text", "T4": "T4_text"}
GUARD_SETS = ["G1", "G2", "G3", "G4"]  # out-of-domain text guardrail (never trained on)
ALL_EVAL_SETS = ["CAL"] + TEST_SETS + list(TEXT_CONTROL_SETS.values()) + GUARD_SETS
SOFT_LABEL_SETS = {"T1", "T3", "T1_text"}  # scored against the listener-vote distribution too
NOUL_SETS = {"T2", "T2_text", "G2"}  # yes/no sets, scored with balanced accuracy


# --------------------------------------------------------------------------------------
# 2. Items
# --------------------------------------------------------------------------------------
# An item is a plain dict (one JSON line in the manifests):
#   id, set, source, question, options (in display order; option i gets code ANSWER_CODES[i]),
#   target (probabilities aligned with options, sums to 1), gold (index of the correct option),
#   audio ({"bank", "key"} or None), channel ("clean" | "phone"), state_text (str or None),
#   majority (CREMA-D only: indices of the options with the most listener votes), meta (dict).


def smooth_one_hot(k: int, gold: int, eps: float) -> np.ndarray:
    """Label smoothing: (1 - eps) on the gold option + eps spread evenly over all k options."""
    q = np.full(k, eps / k)
    q[gold] += 1.0 - eps
    return q


def votes_to_distribution(counts, eps: float = 0.01) -> np.ndarray:
    """Vote counts -> distribution, then add eps to every option and renormalise."""
    counts = np.asarray(counts, dtype=float)
    p = counts / counts.sum()
    p = p + eps
    return p / p.sum()


def parse_voice_votes(csv_text: str) -> dict[str, np.ndarray]:
    """Parse CREMA-D processedResults/tabulatedVotes.csv -> {fileName: counts in VOTE_COLUMNS order}.

    The file has 3 rows per clip. Rows whose first column (rater-batch id) starts with "1" are the
    voice-only (audio) ratings; "2" = face-only, "3" = audio+video. We keep only the voice rows.
    """
    reader = csv.reader(io.StringIO(csv_text))
    header = next(reader)
    col = {name: i for i, name in enumerate(header)}
    out = {}
    for row in reader:
        if not row or not row[0].startswith("1"):
            continue
        out[row[col["fileName"]]] = np.array([float(row[col[c]]) for c in VOTE_COLUMNS])
    return out


def make_item(*, item_id, set_name, source, question, options, target, gold, rng,
              audio=None, channel="clean", state_text=None, majority=None, meta=None) -> dict:
    """Build an item, shuffling the option order with `rng` (a random.Random).

    `options`, `target`, `gold`, `majority` are given in a canonical order; after shuffling they are
    all re-indexed consistently, so target[i] is always the probability of options[i].
    """
    perm = list(range(len(options)))
    rng.shuffle(perm)  # perm[j] = canonical index of the option shown at position j
    new_pos = {canon: j for j, canon in enumerate(perm)}
    item = {
        "id": item_id,
        "set": set_name,
        "source": source,
        "question": question,
        "options": [options[c] for c in perm],
        "target": [round(float(target[c]), 6) for c in perm],
        "gold": new_pos[gold],
        "audio": audio,
        "channel": channel,
        "state_text": state_text,
        "meta": meta or {},
    }
    if majority is not None:
        item["majority"] = sorted(new_pos[c] for c in majority)
    return item


def training_target(item: dict, crema_intent_mix: float = 0.0, ordinal_smooth: float = 0.0) -> np.ndarray:
    """The probability target used in training. For CREMA-D emotion items it can blend the listener-vote
    distribution with the actor's intended emotion: (1 - mix) x votes + mix x one-hot(intended).
    For ordered-level ("score") items with a hard label, `ordinal_smooth` moves that share of the probability
    to the neighbouring levels, so being one level off costs less than being two levels off."""
    t = np.asarray(item["target"], dtype=np.float64)
    if crema_intent_mix and item.get("source") == "crema_d":
        intent = np.zeros_like(t)
        intent[item["gold"]] = 1.0
        t = (1.0 - crema_intent_mix) * t + crema_intent_mix * intent
    if ordinal_smooth and (item.get("meta") or {}).get("qtype") == "score" and len(t) > 2 and t.max() >= 0.9:
        g = int(item["gold"])
        nb = [j for j in (g - 1, g + 1) if 0 <= j < len(t)]
        spread = np.zeros_like(t)
        spread[nb] = 1.0 / len(nb)
        t = (1.0 - ordinal_smooth) * t + ordinal_smooth * spread
    return t


def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# --------------------------------------------------------------------------------------
# 3. Prompt building
# --------------------------------------------------------------------------------------


def build_prompt_text(item: dict, codes: list[str] | None = None) -> str:
    """The text part of the user turn. Example:

    STATE: <the audio above>
    QUESTION: How does the speaker sound?
    OPTIONS:
    A) sad
    B) anger
    Answer with only the letter of the best option.
    """
    codes = codes or ANSWER_CODES  # another model's tokenizer may have different single-token codes
    if not 2 <= len(item["options"]) <= len(codes):
        raise ValueError(f"need 2..{len(codes)} options, got {len(item['options'])}")
    state_parts = []
    if item.get("audio") is not None:
        state_parts.append("<the audio above>" if AUDIO_FIRST else "<the audio below>")
    if item.get("state_text"):
        state_parts.append(item["state_text"])
    lines = [f"STATE: {chr(10).join(state_parts)}", f"QUESTION: {item['question']}", "OPTIONS:"]
    lines += [f"{codes[i]}) {opt}" for i, opt in enumerate(item["options"])]
    lines.append(INSTRUCTION)
    return "\n".join(lines)


def build_messages(item: dict) -> list[dict]:
    """Chat messages for the processor's chat template. The audio itself is passed separately
    (same order as the items) when the processor is called; here it is only a placeholder."""
    text = {"type": "text", "text": build_prompt_text(item)}
    if item.get("audio") is None:
        content = [text]
    else:
        content = [{"type": "audio"}, text] if AUDIO_FIRST else [text, {"type": "audio"}]
    return [{"role": "user", "content": content}]


# --------------------------------------------------------------------------------------
# 4. Answer codes + batching for the model (the torch decision head itself lives in readout.py)
# --------------------------------------------------------------------------------------


def build_answer_codes(tokenizer, n: int = N_ANSWER_CODES) -> tuple[list[str], list[int]]:
    """A..Z first, then AA, AB, ..., ZZ in order, keeping only codes that encode to exactly ONE
    token (with a token id not used yet). Stops at n codes; fails loudly if there are fewer."""
    candidates = list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    codes, ids = [], []
    for code in candidates:
        toks = tokenizer.encode(code, add_special_tokens=False)
        if len(toks) == 1 and toks[0] not in ids:
            codes.append(code)
            ids.append(toks[0])
            if len(codes) == n:
                return codes, ids
    raise ValueError(f"only {len(codes)} single-token answer codes in this tokenizer; need {n}")


def encode_batch(processor, items: list[dict], audios: list):
    """Render the chat template for each item and run the processor (tokenise + chunk audio).
    `audios[i]` is a 16 kHz float32 array for audio items, None for text-only items.
    The tokenizer pads on the LEFT, so the last position of every row is the answer position."""
    texts = [
        processor.apply_chat_template(build_messages(it), tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)
        for it in items
    ]
    audio_list = [a for a in audios if a is not None]
    return processor(text=texts, audio=audio_list or None, return_tensors="pt", padding=True)


# --------------------------------------------------------------------------------------
# 5. Metrics and temperature scaling
# --------------------------------------------------------------------------------------


def chance_corrected(acc: float, chance: float) -> float:
    """Skill on a 0-100 scale: 0 = guessing, 100 = perfect. (acc - chance) / (1 - chance) * 100."""
    return (acc - chance) / (1.0 - chance) * 100.0


def balanced_accuracy(pred_labels, gold_labels) -> float:
    """Mean recall over the classes present in gold (robust to class imbalance)."""
    pred = np.asarray(pred_labels)
    gold = np.asarray(gold_labels)
    recalls = [np.mean(pred[gold == c] == c) for c in sorted(set(gold.tolist()))]
    return float(np.mean(recalls))


def expected_calibration_error(conf, correct, n_bins: int = 10) -> float:
    """10-bin ECE on top-option confidence: sum_b (n_b / N) * |accuracy_b - confidence_b|."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.digitize(conf, edges[1:-1], right=True)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.sum() * abs(correct[m].mean() - conf[m].mean())
    return float(total / len(conf))


def brier(probs, targets) -> float:
    """Multi-class Brier score: mean over items of sum_k (p_k - q_k)^2. Lower is better."""
    return float(np.mean([np.sum((np.asarray(p) - np.asarray(q)) ** 2) for p, q in zip(probs, targets)]))


def nll(probs, targets) -> float:
    """Cross-entropy -sum_k q_k log p_k averaged over items (q may be soft). Lower is better."""
    vals = [-np.sum(np.asarray(q) * np.log(np.clip(np.asarray(p), 1e-12, 1.0))) for p, q in zip(probs, targets)]
    return float(np.mean(vals))


def apply_temperature(log_probs, temperature: float) -> np.ndarray:
    """p_T = softmax(log p / T). T > 1 softens (less confident), T < 1 sharpens. Keeps the argmax."""
    z = np.asarray(log_probs, dtype=float) / temperature
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def fit_temperature(log_probs_list, targets_list, lo: float = 0.05, hi: float = 20.0, weights=None) -> float:
    """Fit one scalar temperature by minimising mean NLL (grid search over log T, then a finer grid around
    the best point). Items may have different option counts. Optional `weights` (one per item) give a
    weighted mean, e.g. so that every data source counts equally."""
    k = max(len(lp) for lp in log_probs_list)
    n = len(log_probs_list)
    lp = np.full((n, k), -np.inf)
    q = np.zeros((n, k))
    for i, (a, b) in enumerate(zip(log_probs_list, targets_list)):
        lp[i, : len(a)] = a
        q[i, : len(b)] = b
    w = None if weights is None else np.asarray(weights, dtype=float)

    def mean_nll(log_t):
        z = lp / math.exp(log_t)
        z = z - z.max(axis=1, keepdims=True)
        log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        safe = np.where(q > 0, log_p, 0.0)  # padded options have q = 0 and log_p = -inf
        per_item = -(q * safe).sum(axis=1)
        return float(np.average(per_item, weights=w))

    grid = np.linspace(math.log(lo), math.log(hi), 401)
    best = grid[int(np.argmin([mean_nll(g) for g in grid]))]
    step = grid[1] - grid[0]
    fine = np.linspace(best - step, best + step, 201)
    return float(math.exp(fine[int(np.argmin([mean_nll(g) for g in fine]))]))


def reliability_bins(conf, correct, edges=(0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0001)) -> list[dict]:
    """For each confidence band: how many answers, average stated confidence, and how often right.
    Calibrated = "how often right" is close to "average stated confidence" in every band."""
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        out.append({"band": f"{lo:.0%}–{min(hi, 1.0):.0%}", "n": int(m.sum()),
                    "stated": float(conf[m].mean()) if m.any() else None,
                    "actual": float(correct[m].mean()) if m.any() else None})
    return out


def compute_set_metrics(items: list[dict], log_probs: list, temperature: float = 1.0) -> dict:
    """All metrics for one evaluation set. `log_probs[i]` aligns with items[i]["options"]."""
    probs = [apply_temperature(lp, temperature) for lp in log_probs]
    k = len(items[0]["options"])
    pred = np.array([int(np.argmax(p)) for p in probs])
    gold = np.array([it["gold"] for it in items])
    conf = np.array([float(np.max(p)) for p in probs])
    correct = pred == gold
    m = {"n": len(items), "k": k, "acc": float(correct.mean()), "ece": expected_calibration_error(conf, correct),
         "reliability": reliability_bins(conf, correct)}
    mixed_k = len({len(it["options"]) for it in items}) > 1  # e.g. the CAL split
    m["skill"] = None if mixed_k else chance_corrected(m["acc"], 1.0 / k)
    m["brier"] = brier(probs, [it["target"] for it in items])
    m["nll"] = nll(probs, [it["target"] for it in items])
    set_name = items[0]["set"]
    if set_name in SOFT_LABEL_SETS:  # also score against the listeners' majority vote
        maj = np.array([p in it["majority"] for p, it in zip(pred, items)])
        m["acc_majority"] = float(maj.mean())
        m["skill_majority"] = chance_corrected(m["acc_majority"], 1.0 / k)
    if set_name in NOUL_SETS:  # yes/no: balanced accuracy over the yes/no labels
        pred_lab = [it["options"][p] for p, it in zip(pred, items)]
        gold_lab = [it["options"][it["gold"]] for it in items]
        m["bal_acc"] = balanced_accuracy(pred_lab, gold_lab)
        m["skill"] = chance_corrected(m["bal_acc"], 0.5)
    return m


def uniform_metrics(items: list[dict]) -> dict:
    """The 'random' row: a predictor that gives every option the same probability."""
    k = len(items[0]["options"])
    probs = [np.full(k, 1.0 / k) for _ in items]
    return {"skill": 0.0, "skill_majority": 0.0,
            "brier": brier(probs, [it["target"] for it in items]),
            "nll": nll(probs, [it["target"] for it in items])}


# --------------------------------------------------------------------------------------
# 6. Audio: banks on disk + phone-channel simulation
# --------------------------------------------------------------------------------------


def write_audio_bank(out_dir, name: str, clips: dict[str, np.ndarray]) -> None:
    """Store all clips of one dataset as one float32 array (<name>.npy) + an index
    (<name>.json: key -> [start, length]). One big file loads fast from a Modal Volume."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index, start = {}, 0
    for key, x in clips.items():
        index[key] = [start, int(len(x))]
        start += len(x)
    data = np.concatenate([np.asarray(x, dtype=np.float32) for x in clips.values()]) if clips else np.zeros(0, np.float32)
    np.save(out_dir / f"{name}.npy", data)
    (out_dir / f"{name}.json").write_text(json.dumps(index))


class AudioBank:
    def __init__(self, out_dir, name: str):
        self.data = np.load(Path(out_dir) / f"{name}.npy", mmap_mode="r")
        self.index = json.loads((Path(out_dir) / f"{name}.json").read_text())

    def get(self, key: str) -> np.ndarray:
        start, length = self.index[key]
        return np.array(self.data[start : start + length], dtype=np.float32)


def load_banks(data_dir, names=("crema_d", "minds14")) -> dict[str, AudioBank]:
    return {n: AudioBank(Path(data_dir) / "audio", n) for n in names if (Path(data_dir) / "audio" / f"{n}.npy").exists()}


def item_audio(item: dict, banks: dict) -> np.ndarray | None:
    """The 16 kHz waveform for an item (None for text-only items), with the phone channel applied
    on the fly for T3 items."""
    if item.get("audio") is None:
        return None
    x = banks[item["audio"]["bank"]].get(item["audio"]["key"])
    if item.get("channel") == "phone":
        x = phone_channel(x)
    return x


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resampling with an anti-aliasing filter (scipy)."""
    from scipy.signal import resample_poly

    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    g = math.gcd(sr_in, sr_out)
    return resample_poly(np.asarray(x, dtype=np.float64), sr_out // g, sr_in // g).astype(np.float32)


_MULAW_BIAS, _MULAW_CLIP = 132, 32635


def mulaw_encode(x: np.ndarray) -> np.ndarray:
    """G.711 mu-law encoder (the classic 8-segment version used by telephone networks).
    float waveform in [-1, 1] -> 8-bit codes (uint8)."""
    pcm = np.clip(np.round(np.asarray(x, dtype=np.float64) * 32768.0), -32768, 32767).astype(np.int32)
    sign = (pcm < 0).astype(np.int32)
    mag = np.minimum(np.abs(pcm), _MULAW_CLIP) + _MULAW_BIAS
    exponent = np.clip(np.floor(np.log2(mag)).astype(np.int32) - 7, 0, 7)
    mantissa = (mag >> (exponent + 3)) & 0x0F
    return (~((sign << 7) | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8)


def mulaw_decode(codes: np.ndarray) -> np.ndarray:
    """G.711 mu-law decoder: 8-bit codes -> float waveform in [-1, 1]."""
    u = ~np.asarray(codes, dtype=np.int32) & 0xFF
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    mag = (((mantissa << 3) + _MULAW_BIAS) << exponent) - _MULAW_BIAS
    pcm = np.where(u & 0x80, -mag, mag)
    return (pcm / 32768.0).astype(np.float32)


def phone_channel(x16k: np.ndarray) -> np.ndarray:
    """Simulate a phone line: 16 kHz -> 8 kHz -> G.711 mu-law encode/decode -> back to 16 kHz.
    Output has the same length as the input."""
    x8 = resample(x16k, SAMPLE_RATE, 8000)
    x8 = mulaw_decode(mulaw_encode(np.clip(x8, -1.0, 1.0)))
    y = resample(x8, 8000, SAMPLE_RATE)[: len(x16k)]
    if len(y) < len(x16k):
        y = np.pad(y, (0, len(x16k) - len(y)))
    return y.astype(np.float32)


# --------------------------------------------------------------------------------------
# 7. results.md
# --------------------------------------------------------------------------------------

SET_TITLES = {
    "T1": "T1 emotion",
    "T2": "T2 'angry?' (yes/no, never trained)",
    "T3": "T3 emotion over phone line",
    "T4": "T4 spoken intent",
    "T5": "T5 BANKING77 text",
    "T6": "T6 BANKING77 text, all 77 options",
    "G1": "G1 CLINC150 (151 options)",
    "G2": "G2 BoolQ (yes/no)",
    "G3": "G3 HellaSwag (4 endings)",
    "G4": "G4 WinoGrande (2 options)",
}


def evaluate_predictions(items: list[dict], preds: list[dict], temperature: float) -> dict:
    """{set: {"raw": metrics, "temp": metrics}} for one system. `preds` rows: {"id", "log_probs"}."""
    by_id = {p["id"]: p for p in preds}
    out = {}
    for s in ALL_EVAL_SETS:
        its = [it for it in items if it["set"] == s and it["id"] in by_id]
        if not its:
            continue
        lps = [by_id[it["id"]]["log_probs"] for it in its]
        out[s] = {"raw": compute_set_metrics(its, lps, 1.0), "temp": compute_set_metrics(its, lps, temperature)}
    return out


def systems_in(metrics: dict) -> list[str]:
    """stock first, then lora, then any other systems (e.g. external baselines) in a stable order."""
    first = [s for s in ("stock", "lora") if s in metrics]
    return first + sorted(s for s in metrics if s not in first)


def results_markdown(run: str, items: list[dict], metrics: dict, temperatures: dict, latency: dict | None = None,
                     temp_details: dict | None = None) -> str:
    """Build results.md. `metrics[system]` comes from evaluate_predictions; systems: stock, lora."""

    def f(x, nd=1):
        return "—" if x is None else f"{x:.{nd}f}"

    def get(system, s, key, kind="raw"):
        return metrics.get(system, {}).get(s, {}).get(kind, {}).get(key)

    rows = []  # (label, system, use_text_controls)
    for system in systems_in(metrics):
        rows.append((system, system, False))
        if any(t in metrics[system] for t in TEXT_CONTROL_SETS.values()):
            rows.append((f"{system} · transcript-only", system, True))

    def set_for(s, text):
        return TEXT_CONTROL_SETS.get(s) if text else s

    uniform = {s: uniform_metrics([it for it in items if it["set"] == s]) for s in TEST_SETS
               if any(it["set"] == s for it in items)}
    L = [f"# Results: run `{run}`", ""]
    L.append("Model: google/gemma-4-12B-it with a 255-code decision head. `stock` = zero-shot (head at init = plain "
             "letter readout), `lora` = LoRA adapter + trained head. `transcript-only` = same model, no audio, the "
             "transcript as the text state.")
    L.append("")
    temps = ", ".join(f"{k} T = {v:.3f}" for k, v in temperatures.items())
    n_cal = sum(1 for it in items if it["set"] == "CAL")
    L.append(f"Temperatures fitted on the held-out calibration split (CAL, {n_cal} items, every source weighted "
             f"equally): {temps or '—'}")
    for system, d in (temp_details or {}).items():
        if d.get("per_source"):
            per = ", ".join(f"{k} {v:.2f}" for k, v in d["per_source"].items())
            L.append(f"Diagnostic only (not applied): best temperature per source for {system}: {per}")
    L.append("")

    if "stock" in metrics and "lora" in metrics:
        L.append("## 0. Headline: what did LoRA training change? (lora minus stock)")
        L.append("")
        L.append("| | " + " | ".join(SET_TITLES[s] for s in TEST_SETS) + " |")
        L.append("|" + "---|" * (len(TEST_SETS) + 1))
        deltas = [get("lora", s, "skill") - get("stock", s, "skill") if get("lora", s, "skill") is not None
                  and get("stock", s, "skill") is not None else None for s in TEST_SETS]
        L.append("| skill change (points) | " + " | ".join("—" if d is None else f"{d:+.1f}" for d in deltas) + " |")
        eces = [(get("stock", s, "ece", "temp"), get("lora", s, "ece", "temp")) for s in TEST_SETS]
        L.append("| ECE after temperature, stock → lora | "
                 + " | ".join("—" if a is None or b is None else f"{a:.3f} → {b:.3f}" for a, b in eces) + " |")
        L.append("")
        g = [(get("stock", x, "skill"), get("lora", x, "skill")) for x in GUARD_SETS]
        g = [(a, b) for a, b in g if a is not None and b is not None]
        if g:
            L.append(f"| out-of-domain text guardrail (G1–G4), mean skill change | "
                     f"{sum(b - a for a, b in g) / len(g):+.1f} points (never trained on these) |")
        L.append("")
        L.append("Positive skill change = better than the zero-shot model. Compare against the transcript-only rows "
                 "below: audio skill far above transcript-only skill means the model is really using the audio.")
        L.append("")

    L.append("## 1. Skill: chance-corrected accuracy (0 = random guessing, 100 = perfect)")
    L.append("Temperature scaling never changes which option is on top, so these are the same raw and scaled.")
    L.append("")
    hdr = ["system", "T1 vs actor's intent", "T1 vs listener majority", "T2 angry? (balanced)",
           "T3 phone line", "T4 spoken intent", "T5 BANKING77 text", "T6 BANKING77, 77 options"]
    L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    L.append("| random | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |")
    t1_items = [it for it in items if it["set"] == "T1" and "votes" in it.get("meta", {})]
    if t1_items:  # how often ONE human listener (voice only) picked the actor's intended emotion
        human = float(np.mean([it["meta"]["votes"][EMOTIONS.index(it["options"][it["gold"]])] / sum(it["meta"]["votes"])
                               for it in t1_items]))
        L.append(f"| reference: one human listener (CREMA-D votes) | {chance_corrected(human, 1 / 6):.1f} | — | — | — | — | — | — |")
    for label, system, text in rows:
        cells = [
            get(system, set_for("T1", text), "skill"),
            get(system, set_for("T1", text), "skill_majority"),
            get(system, set_for("T2", text), "skill"),
            None if text else get(system, "T3", "skill"),
            get(system, set_for("T4", text), "skill"),
            None if text else get(system, "T5", "skill"),
            None if text else get(system, "T6", "skill"),
        ]
        L.append(f"| {label} | " + " | ".join(f(c) for c in cells) + " |")
    L.append("")

    present_guard = [x for x in GUARD_SETS if any(x in metrics.get(sy, {}) for sy in metrics)]
    if present_guard:
        L.append("## 1b. Out-of-domain text guardrail: public text decision benchmarks NEVER used in training")
        L.append("Skill (chance-corrected; BoolQ = balanced accuracy) and ECE after temperature. A drop here would mean "
                 "the audio training damaged general text decision-making.")
        L.append("")
        hdr = ["system"] + [SET_TITLES[x] for x in present_guard] + ["mean skill"]
        L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for system in systems_in(metrics):
            sk = [get(system, x, "skill") for x in present_guard]
            ece = [get(system, x, "ece", "temp") for x in present_guard]
            cells = [("—" if a is None else f"{a:.1f} (ECE {b:.3f})") for a, b in zip(sk, ece)]
            vals = [a for a in sk if a is not None]
            mean = f"{sum(vals) / len(vals):.1f}" if vals else "—"
            L.append(f"| {system} | " + " | ".join(cells) + f" | {mean} |")
        L.append("")

    L.append("## 2. Calibration: ECE (10 bins, top-option confidence; lower is better). raw → temperature-scaled")
    L.append("")
    hdr = ["system"] + [SET_TITLES[s] for s in TEST_SETS]
    L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for label, system, text in rows:
        cells = []
        for s in TEST_SETS:
            s2 = set_for(s, text)
            raw, tmp = get(system, s2, "ece") if s2 else None, get(system, s2, "ece", "temp") if s2 else None
            cells.append("—" if raw is None else f"{raw:.3f} → {tmp:.3f}")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## 3. Probability quality: Brier and NLL (lower is better). raw → temperature-scaled")
    L.append("T1/T3 are scored against the listener-vote distribution; the others against the gold answer.")
    L.append("")
    hdr = ["system"] + [f"{s} Brier" for s in TEST_SETS] + [f"{s} NLL" for s in TEST_SETS]
    L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    L.append("| random (uniform) | " + " | ".join(f(uniform.get(s, {}).get("brier"), 3) for s in TEST_SETS) + " | "
             + " | ".join(f(uniform.get(s, {}).get("nll"), 3) for s in TEST_SETS) + " |")
    for label, system, text in rows:
        cells = []
        for key in ["brier", "nll"]:
            for s in TEST_SETS:
                s2 = set_for(s, text)
                raw = get(system, s2, key) if s2 else None
                cells.append("—" if raw is None else f"{raw:.3f} → {get(system, s2, key, 'temp'):.3f}")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## 4. Plain accuracies (%)")
    L.append("")
    hdr = ["system", "T1 vs intent", "T1 vs majority", "T2 balanced acc", "T3", "T4", "T5", "T6"]
    L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    chance = []
    for s, key in [("T1", "acc"), ("T1", "acc_majority"), ("T2", "bal_acc"), ("T3", "acc"), ("T4", "acc"), ("T5", "acc"), ("T6", "acc")]:
        its = [it for it in items if it["set"] == s]
        chance.append(None if not its else (50.0 if s == "T2" else 100.0 / len(its[0]["options"])))
    L.append("| random (chance) | " + " | ".join(f(c) for c in chance) + " |")
    for label, system, text in rows:
        cells = [get(system, set_for("T1", text), "acc"), get(system, set_for("T1", text), "acc_majority"),
                 get(system, set_for("T2", text), "bal_acc"), None if text else get(system, "T3", "acc"),
                 get(system, set_for("T4", text), "acc"), None if text else get(system, "T5", "acc"),
                 None if text else get(system, "T6", "acc")]
        L.append(f"| {label} | " + " | ".join(f(None if c is None else 100 * c) for c in cells) + " |")
    L.append("")

    L.append("## 4b. Reliability: when the model states a confidence, how often is it right? (after temperature)")
    L.append("Each cell is `stated → actual (n)`. Calibrated means the two numbers are close in every band.")
    L.append("")
    for system in systems_in(metrics):
        bands = None
        rows_rel = []
        for s in TEST_SETS:
            rel = get(system, s, "reliability", "temp")
            if not rel:
                continue
            bands = bands or [b["band"] for b in rel]
            cells = ["—" if b["n"] == 0 else f"{b['stated']:.0%} → {b['actual']:.0%} ({b['n']})" for b in rel]
            rows_rel.append(f"| {SET_TITLES[s]} | " + " | ".join(cells) + " |")
        if bands:
            L += [f"**{system}**", "", "| set | " + " | ".join(bands) + " |", "|" + "---|" * (len(bands) + 1)]
            L += rows_rel + [""]
    L.append("")
    L.append("## 5. Latency: milliseconds per request, batch size 1, warm model")
    latency = latency or {}
    if latency:
        any_lat = next(iter(latency.values()))
        L.append(f"GPU per system is shown in each column header (compare only like with like). "
                 f"{any_lat['runs']} timed runs after {any_lat['warmup']} warm-ups. "
                 "End-to-end = prompt + processor (CPU) + forward + readout; model-only excludes the CPU part. "
                 "`lora` rows run the adapter unmerged (extra cost); `lora merged` rows have it merged into the weights, as in deployment.")
        L.append("")
        systems = [s for s in ("stock", "lora", "lora_merged") if s in latency] + sorted(
            s for s in latency if s not in ("stock", "lora", "lora_merged"))
        hdr = ["scenario", "decisions", "prompt tokens"] + [
            f"{s} on {latency[s].get('device', '?')}: median / p95 (model-only median)" for s in systems]
        L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for name, sc in any_lat["scenarios"].items():
            cells = [name, str(sc["decisions"]), str(sc["prompt_tokens"])]
            for s in systems:
                r = latency[s]["scenarios"].get(name)
                cells.append("—" if r is None else f"{r['median_ms']:.0f} / {r['p95_ms']:.0f} ({r['model_median_ms']:.0f})")
            L.append("| " + " | ".join(cells) + " |")
    else:
        L.append("(no latency measurements found)")
    L.append("")

    L.append("## 6. Items per set")
    L.append("")
    counts = {s: sum(1 for it in items if it["set"] == s) for s in ALL_EVAL_SETS}
    L += ["| " + " | ".join(counts) + " |", "|" + "---|" * len(counts), "| " + " | ".join(str(v) for v in counts.values()) + " |"]
    L.append("")
    return "\n".join(L)
