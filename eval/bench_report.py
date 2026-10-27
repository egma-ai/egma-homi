"""
Scoreboard for a multiple-choice audio benchmark in the MMSU layout (truth.jsonl, items<name>.jsonl,
runs/q<name>/<system>/...): homi, the base model, Gemini and OpenAI on identical questions.

    uv run --python 3.12 --with numpy python eval/bench_report.py --dir <local copy of /vol/bench_<name>> --name <name> \
        --out <NAME>_RESULTS.md

Main score = accuracy; an unanswered or refused question counts as wrong. Also shown: chance level, a 95% interval,
accuracy on answered questions, refusals and cost per 10,000 questions.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

SYSTEMS = [("homi", "homi"), ("qwen-stock", "Qwen3-Omni untouched (base)"),
           ("gemini:gemini-3.8-flash", "Gemini 3.8 Flash"), ("gemini:gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite"),
           ("openai:gpt-audio", "OpenAI gpt-audio")]
PRICES = {"gemini:gemini-3.8-flash": {"audio": 0.75, "text": 0.75, "output": 3.75},  # $ / 1M tokens, 2026-10-21
          "gemini:gemini-3.5-flash-lite": {"audio": 0.30, "text": 0.30, "output": 2.50},
          "openai:gpt-audio": {"audio": 32.0, "text": 2.5, "output": 10.0}}
H200_PER_SECOND = 0.001261  # Modal H200, $4.54/h
LETTERS = "ABCDEFGHIJ"


def load_ours(path: Path, items: dict) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            p = json.loads(line)
            out[items[p["id"]]["meta"]["call"]] = int(np.argmax(p["log_probs"]))
    return out


def load_api(path: Path, truth: dict) -> tuple[dict, set]:
    """-> ({id: chosen index}, {ids the model refused / gave no answer to})"""
    out, refused = {}, set()
    if not path.exists():
        return out, refused
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("id") not in truth:
            continue
        if r.get("refused"):
            refused.add(r["id"])
            continue
        a = r.get("answers")
        if not isinstance(a, dict):
            continue
        n = len(truth[r["id"]]["options"])
        ans = str(a.get("answer", "")).strip()
        idx = None
        if ans and ans[:1].upper() in LETTERS[:n] and (len(ans) == 1 or not ans[1:2].isalpha()):
            idx = LETTERS.index(ans[:1].upper())
        elif ans and ans in truth[r["id"]]["options"]:
            idx = truth[r["id"]]["options"].index(ans)
        elif isinstance(a.get("probabilities"), dict):
            pr = {k.strip()[:1].upper(): v for k, v in a["probabilities"].items() if isinstance(v, (int, float))}
            pr = {k: v for k, v in pr.items() if k in LETTERS[:n]}
            idx = LETTERS.index(max(pr, key=pr.get)) if pr else None
        if idx is None:
            refused.add(r["id"])
        else:
            out[r["id"]] = idx
    return out, refused


def ci95(p: float, n: int) -> float:
    return 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / max(n, 1)) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    d = Path(args.dir)
    truth = {json.loads(l)["id"]: json.loads(l) for l in (d / "truth.jsonl").read_text().splitlines()}
    items = {json.loads(l)["id"]: json.loads(l) for l in (d / f"items{args.name}.jsonl").read_text().splitlines()}
    preds, refused, cost = {}, {}, {}
    for s, _ in SYSTEMS:
        base = d / "runs" / f"q{args.name}" / s
        if not base.exists():  # folder names with ":" replaced by "_" (portable checkouts)
            base = base.with_name(s.replace(":", "_"))
        if ":" in s:
            preds[s], refused[s] = load_api(base / "responses.jsonl", truth)
            if (base / "responses.jsonl").exists() and s in PRICES:
                pr, tot, n = PRICES[s], 0.0, 0
                for line in (base / "responses.jsonl").read_text().splitlines():
                    u = json.loads(line).get("usage")
                    if u:
                        n += 1
                        audio = (u.get("prompt_by_modality") or {}).get("AUDIO", 0)
                        tot += (audio * pr["audio"] + (u["prompt"] - audio) * pr["text"]
                                + (u["output"] + u.get("thinking", 0)) * pr["output"]) / 1e6
                if n:
                    cost[s] = tot / n * 10000
        else:
            preds[s], refused[s] = load_ours(base / "preds.jsonl", items), set()
            if (base / "timing.json").exists():
                t = json.loads((base / "timing.json").read_text())
                cost[s] = t["inference_seconds"] * H200_PER_SECOND / t["questions"] * 10000
    ran = [(s, n) for s, n in SYSTEMS if preds[s] or refused[s]]
    tasks = defaultdict(list)
    for t in truth.values():
        tasks[t["task"]].append(t)
    chance = 100 * float(np.mean([1 / len(t["options"]) for t in truth.values()]))

    def acc(s, ts):
        return 100 * float(np.mean([preds[s].get(t["id"]) == t["gold"] for t in ts])) if ts else float("nan")

    allt = list(truth.values())
    L = [f"# {args.title or args.name}: one harness, every model", "",
         f"{len(truth)} questions, {len(tasks)} tasks; chance level {chance:.1f}%. Same audio, question and options (published "
         "order) for every model; an unanswered or refused question counts as wrong. ± = 95% interval.", "",
         "| model | accuracy | ± | answered | accuracy on answered | refused / no answer | cost per 10k questions |",
         "|---|---|---|---|---|---|---|"]
    for s, name in ran:
        a = acc(s, allt)
        answered = [t for t in allt if t["id"] in preds[s]]
        L.append(f"| {name} | {a:.1f} | {ci95(a / 100, len(allt)):.1f} | {len(answered)}/{len(allt)} | "
                 f"{acc(s, answered):.1f} | {len(refused[s])} | " + (f"${cost[s]:.2f}" if s in cost else "—") + " |")
    L += ["", "## Per task (accuracy %, n questions)", "", "| task | n | " + " | ".join(n for _, n in ran) + " |",
          "|---|---|" + "---|" * len(ran)]
    for k, ts in sorted(tasks.items()):
        L.append(f"| {k} | {len(ts)} | " + " | ".join(f"{acc(s, ts):.0f}" for s, _ in ran) + " |")
    if any(refused[s] for s, _ in ran):
        L += ["", "## Refusals / no answer by task", "", "| task | " + " | ".join(n for s, n in ran if refused[s]) + " |",
              "|---|" + "---|" * sum(1 for s, _ in ran if refused[s])]
        for k, ts in sorted(tasks.items()):
            ids = {t["id"] for t in ts}
            L.append(f"| {k} | " + " | ".join(str(len(refused[s] & ids)) for s, _ in ran if refused[s]) + " |")
    L += ["", "Ours = measured GPU seconds on one H200 at $4.54/h, batched. Gemini / OpenAI = their reported tokens at "
          "paid-tier list prices (Gemini 3.8 Flash at its promotional price until 2026-12-31)."]
    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L[:14]))


if __name__ == "__main__":
    main()
