"""
Calibration of egma-homi, the base model, Gemini and OpenAI on multiple-choice audio benchmarks in the MMSU layout
(truth.jsonl, items<name>.jsonl, runs/q<name>/<system>/...), from the saved answers.

    uv run --no-project --python 3.12 --with numpy python eval/calibration_report.py \
        --dir eval/results/mmsu eval/results/vocalgrad --name mmsu vocalgrad \
        --temperature 1.0084233286300535 --out eval/results/CALIBRATION.md

egma-homi = softmax(log_probs / T) at its calibration temperature; the base model at T = 1. API models = the
probabilities they report with their answer, renormalised; a row counts only if every option gets a non-negative
number. Confidence = probability of the argmax option (ties -> first). ECE uses 15 equal-width bins (lo, hi].
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

SYSTEMS = [("egma-homi", "egma-homi"), ("qwen-stock", "Qwen3-Omni untouched (base)"),
           ("gemini:gemini-3.8-flash", "Gemini 3.8 Flash"), ("gemini:gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite"),
           ("openai:gpt-audio", "OpenAI gpt-audio")]
TITLES = {"mmsu": "MMSU", "vocalgrad": "VocalGrad"}
LETTERS = "ABCDEFGHIJ"
BINS = 15
THRESHOLDS = (0.5, 0.7, 0.9, 0.95)
MIN_BIN = 20


def parse_answer(a: dict, options: list) -> int | None:
    n, ans = len(options), str(a.get("answer", "")).strip()
    if ans and ans[:1].upper() in LETTERS[:n] and (len(ans) == 1 or not ans[1:2].isalpha()):
        return LETTERS.index(ans[:1].upper())
    if ans and ans in options:
        return options.index(ans)
    return None


def load_ours(path: Path, items: dict, temperature: float) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        p = json.loads(line)
        z = np.asarray(p["log_probs"], dtype=float) / temperature
        z = np.exp(z - z.max())
        out[items[p["id"]]["meta"]["call"]] = z / z.sum()
    return out


def load_api(path: Path, truth: dict) -> tuple[dict, int]:
    """-> ({id: probabilities over the options}, rows with an answer but no usable probabilities)"""
    out, answer_only = {}, 0
    for line in path.read_text().splitlines():
        r = json.loads(line)
        a = r.get("answers")
        if r.get("id") not in truth or r.get("refused") or not isinstance(a, dict):
            continue
        options = truth[r["id"]]["options"]
        pr = a.get("probabilities")
        pr = {str(k).strip().upper(): v for k, v in pr.items()} if isinstance(pr, dict) else {}
        p = [pr.get(k) for k in LETTERS[:len(options)]]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 for v in p) \
                and sum(p) > 0:
            out[r["id"]] = np.asarray(p, dtype=float) / sum(p)
        elif parse_answer(a, options) is not None:
            answer_only += 1
    return out, answer_only


def score(probs: dict, truth: dict) -> dict:
    ids = [i for i in truth if i in probs]
    gold = [truth[i]["gold"] for i in ids]
    conf = np.array([probs[i].max() for i in ids])
    hit = np.array([int(np.argmax(probs[i])) == g for i, g in zip(ids, gold)], dtype=float)
    brier = np.mean([((probs[i] - np.eye(len(probs[i]))[g]) ** 2).sum() for i, g in zip(ids, gold)])
    nll = np.mean([-math.log(max(probs[i][g], 1e-6)) for i, g in zip(ids, gold)])
    b = np.clip(np.ceil(np.round(conf * BINS, 6)).astype(int) - 1, 0, BINS - 1)
    bins = [(k, int((b == k).sum()), conf[b == k].mean(), hit[b == k].mean()) for k in range(BINS) if (b == k).any()]
    ece = sum(n * abs(c - a) for _, n, c, a in bins) / len(ids)
    sel = {}
    for t in THRESHOLDS:
        m = conf >= t - 1e-9
        sel[t] = (m.sum() / len(truth), hit[m].mean() if m.any() else float("nan"))
    return {"n": len(ids), "acc": hit.mean(), "ece": ece, "brier": brier, "nll": nll, "bins": bins, "sel": sel}


def report(d: Path, name: str, temperature: float) -> list[str]:
    truth = {json.loads(l)["id"]: json.loads(l) for l in (d / "truth.jsonl").read_text().splitlines()}
    items = {json.loads(l)["id"]: json.loads(l) for l in (d / f"items{name}.jsonl").read_text().splitlines()}
    res, answer_only = {}, {}
    for s, _ in SYSTEMS:
        base = d / "runs" / f"q{name}" / s
        if not base.exists():  # folder names with ":" replaced by "_" (portable checkouts)
            base = base.with_name(s.replace(":", "_"))
        if ":" in s and (base / "responses.jsonl").exists():
            probs, answer_only[s] = load_api(base / "responses.jsonl", truth)
        elif ":" not in s and (base / "preds.jsonl").exists():
            probs, answer_only[s] = load_ours(base / "preds.jsonl", items, temperature if s == "egma-homi" else 1.0), 0
        else:
            continue
        if probs:
            res[s] = score(probs, truth)
    ran = [(s, n) for s, n in SYSTEMS if s in res]
    pct = lambda v: f"{100 * v:.1f}"  # noqa: E731
    L = [f"## {TITLES.get(name, name)} ({len(truth)} questions)", "",
         "| model | coverage | accuracy | ECE | Brier | NLL | share at conf ≥ 0.9 | accuracy at conf ≥ 0.9 | "
         "answered, no usable probabilities |", "|---|---|---|---|---|---|---|---|---|"]
    for s, n in ran:
        r = res[s]
        share, acc = r["sel"][0.9]
        L.append(f"| {n} | {r['n']}/{len(truth)} | {pct(r['acc'])} | {r['ece']:.3f} | {r['brier']:.3f} | "
                 f"{r['nll']:.3f} | {pct(share)} | {pct(acc)} | {answer_only[s]} |")
    L += ["", "| model | " + " | ".join(f"share / accuracy at conf ≥ {t}" for t in THRESHOLDS) + " |",
          "|---|" + "---|" * len(THRESHOLDS)]
    for s, n in ran:
        L.append(f"| {n} | " + " | ".join(f"{pct(a)} / {pct(b)}" for a, b in res[s]["sel"].values()) + " |")
    if "egma-homi" in res:
        L += ["", f"### egma-homi reliability (T = {temperature:.4f}; bins with ≥ {MIN_BIN} questions)", "",
              "| confidence bin | n | mean confidence | accuracy |", "|---|---|---|---|"]
        for k, n, c, a in res["egma-homi"]["bins"]:
            if n >= MIN_BIN:
                L.append(f"| {k / BINS:.3f}–{(k + 1) / BINS:.3f} | {n} | {pct(c)} | {pct(a)} |")
    return L + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", nargs="+", required=True)
    ap.add_argument("--name", nargs="+", required=True)
    ap.add_argument("--temperature", type=float, required=True, help="egma-homi's calibration temperature")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert len(args.dir) == len(args.name), "one --name per --dir"
    L = ["# Calibration", "",
         f"egma-homi: softmax(log-probs / T), T = {args.temperature:.4f} (its calibration temperature); the untouched base "
         "model at T = 1. API models: the probabilities they self-report with their answer, renormalised; a row without a "
         "non-negative number for every option is not covered (last column: such rows that still named an answer).",
         "Confidence = probability of the argmax option, which is the answer scored. Accuracy, ECE (15 equal-width bins), "
         "Brier (sum over options) and NLL (p clipped at 1e-6) are on covered questions; selective = share of all "
         "questions at or above a confidence threshold / accuracy on them (%).", ""]
    for d, name in zip(args.dir, args.name):
        L += report(Path(d), name, args.temperature)
    Path(args.out).write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
