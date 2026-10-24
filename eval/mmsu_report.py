"""
MMSU (ICLR 2026, MIT) scoreboard: egma-homi, the base model, Gemini and OpenAI on identical questions, scored in the
paper's groups, next to published numbers.

    uv run --python 3.12 --with numpy python eval/mmsu_report.py --dir <local copy of /vol/bench_mmsu> --out MMSU.md

<dir> holds truth.jsonl, itemsmmsu.jsonl, runs/qmmsu/<system>/preds.jsonl (ours) and
runs/qmmsu/<api>:<model>/responses.jsonl (API models). An unanswered question counts as wrong.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SYSTEMS = [("egma-homi", "egma-homi"), ("qwen-stock", "Qwen3-Omni untouched (our base)"),
           ("gemini:gemini-3.8-flash", "Gemini 3.8 Flash"), ("gemini:gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite"),
           ("openai:gpt-audio", "OpenAI gpt-audio")]
# $ per 1M tokens, paid tier, ai.google.dev/gemini-api/docs/pricing (checked 2026-10-21). 3.8 Flash is a promotional
# price through 2026-12-31 (then 1.50 / 7.50).
PRICES = {"gemini:gemini-3.8-flash": {"audio": 0.75, "text": 0.75, "output": 3.75},
          "gemini:gemini-3.5-flash-lite": {"audio": 0.30, "text": 0.30, "output": 2.50},
          "openai:gpt-audio": {"audio": 32.0, "text": 2.5, "output": 10.0},
          "gemini:gemini-2.5-flash-lite": {"audio": 0.30, "text": 0.10, "output": 0.40}}
H200_PER_SECOND = 0.001261  # Modal H200, $4.54/h
LETTERS = "ABCDEFGHIJ"
# Published full-MMSU results, each from its authors' own setup (not directly comparable with ours).
PUBLISHED = [  # name, overall, perception avg, reasoning avg, voice perception, voice reasoning, source
    ("Gemini 2.5 Pro", 77.7, None, None, None, None, "Qwen3-Omni model card"),
    ("Gemini 2.5 Flash", 70.2, None, None, None, None, "Qwen3-Omni model card"),
    ("Qwen3-Omni-30B-A3B (our base)", 69.0, None, None, None, None, "Qwen3-Omni model card"),
    ("Gemini 1.5 Pro", 60.68, 46.10, 76.16, 31.23, 46.33, "MMSU paper leaderboard"),
    ("Qwen2.5-Omni-7B", 60.57, 42.50, 79.83, 39.35, 48.36, "MMSU paper leaderboard"),
    ("Kimi-Audio", 59.28, 43.52, 76.03, 35.74, 55.22, "MMSU paper leaderboard"),
    ("GPT-4o Audio", 56.38, 39.67, 71.96, 21.44, 26.25, "MMSU paper leaderboard"),
    ("Gemini 2.0 Flash", 51.03, 40.83, 47.83, 30.62, 36.16, "MMSU paper leaderboard"),
]


def load_ours(path: Path, items: dict) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        p = json.loads(line)
        out[items[p["id"]]["meta"]["call"]] = int(np.argmax(p["log_probs"]))
    return out


def load_gemini(path: Path, truth: dict) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        r = json.loads(line)
        a = r.get("answers")
        if not isinstance(a, dict) or r["id"] not in truth:
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
        if idx is not None:
            out[r["id"]] = idx
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    truth = {json.loads(l)["id"]: json.loads(l) for l in (d / "truth.jsonl").read_text().splitlines()}
    items = {json.loads(l)["id"]: json.loads(l) for l in (d / "itemsmmsu.jsonl").read_text().splitlines()}
    preds, cost = {}, {}
    for s, _ in SYSTEMS:
        base = d / "runs" / "qmmsu" / s
        if not base.exists():  # folder names with ":" replaced by "_" (portable checkouts)
            base = base.with_name(s.replace(":", "_"))
        preds[s] = (load_gemini(base / "responses.jsonl", truth) if ":" in s
                    else load_ours(base / "preds.jsonl", items))
        if ":" in s and (base / "responses.jsonl").exists():
            pr, tot = PRICES[s], 0.0
            for line in (base / "responses.jsonl").read_text().splitlines():
                u = json.loads(line).get("usage")
                if u:
                    audio = (u.get("prompt_by_modality") or {}).get("AUDIO", 0)
                    tot += (audio * pr["audio"] + (u["prompt"] - audio) * pr["text"] + (u["output"] + u["thinking"]) * pr["output"]) / 1e6
            cost[s] = tot / len(truth) * 10000
        elif (base / "timing.json").exists():
            t = json.loads((base / "timing.json").read_text())
            cost[s] = t["inference_seconds"] * H200_PER_SECOND / t["questions"] * 10000

    groups = {
        "Overall": lambda t: True,
        "Perception avg": lambda t: t["category"] == "Perception",
        "Reasoning avg": lambda t: t["category"] == "Reasoning",
        "Voice perception (paralinguistic)": lambda t: t["category"] == "Perception" and t["sub_category"] == "Paralinguistics",
        "Voice reasoning (paralinguistic)": lambda t: t["category"] == "Reasoning" and t["sub_category"] == "Paralinguistics",
        "Words & sounds perception": lambda t: t["category"] == "Perception" and t["sub_category"] == "Linguistics",
        "Meaning reasoning": lambda t: t["category"] == "Reasoning" and t["sub_category"] == "Linguistics",
    }

    def acc(s, sel):
        ts = [t for t in truth.values() if sel(t)]
        return 100 * np.mean([preds[s].get(t["id"]) == t["gold"] for t in ts]), len(ts)

    L = ["# MMSU, all questions, one harness", "",
         f"{len(truth)} questions (MMSU, ICLR 2026, MIT licence; never used for training). Same audio, same question, same "
         "options in the published order for every model. Score = accuracy against the authors' answer key; an unanswered "
         "question counts as wrong.", "",
         "## 1. Our runs (same setup for every model)", "",
         "| model | answered | " + " | ".join(groups) + " | task average (47 tasks) |", "|---|---|" + "---|" * (len(groups) + 1)]
    tasks = defaultdict(list)
    for t in truth.values():
        tasks[t["task"]].append(t)
    for s, name in SYSTEMS:
        if not preds[s]:
            L.append(f"| {name} | — |" + " |" * (len(groups) + 1))
            continue
        cells = [f"{acc(s, sel)[0]:.1f}" for sel in groups.values()]
        tavg = np.mean([np.mean([preds[s].get(t["id"]) == t["gold"] for t in ts]) for ts in tasks.values()]) * 100
        L.append(f"| {name} | {len(preds[s])}/{len(truth)} | " + " | ".join(cells) + f" | {tavg:.1f} |")
    L += ["", "## Accuracy vs cost (Pareto view)", "",
          "| model | MMSU, all | cost per 10,000 questions |", "|---|---|---|"]
    for s, name in SYSTEMS:
        if preds[s]:
            c = f"${cost[s]:.2f}" if s in cost else "—"
            L.append(f"| {name} | {acc(s, lambda t: True)[0]:.1f} | {c} |")
    L += ["", "Ours = measured GPU seconds on one H200 at $4.54/h with the GPU kept busy (batch); Gemini = its reported tokens "
          "at Google's paid-tier prices (3.8 Flash at its promotional price until 2026-12-31; it doubles after)."]
    L += ["", "Questions per group: " + ", ".join(f"{g} {acc(SYSTEMS[0][0], sel)[1]}" for g, sel in groups.items()), "",
          "## 2. Published results (full MMSU; each from its authors' own setup)", "",
          "| model | Overall | Perception avg | Reasoning avg | Voice perception | Voice reasoning | source |",
          "|---|---|---|---|---|---|---|"]
    f = lambda v: "–" if v is None else f"{v:.1f}"  # noqa: E731
    for name, o, pa, ra, vp, vr, src in PUBLISHED:
        L.append(f"| {name} | {f(o)} | {f(pa)} | {f(ra)} | {f(vp)} | {f(vr)} | {src} |")
    L += ["", "Published numbers come from different setups (they generate a text answer that is then parsed); compare "
          "them with ours through the untouched Qwen3-Omni row, which appears in both: published 69.0 vs our run above.", "",
          "## 3. Per task (accuracy %)", "", "| task | n | " + " | ".join(n for _, n in SYSTEMS) + " |",
          "|---|---|" + "---|" * len(SYSTEMS)]
    for k, ts in sorted(tasks.items(), key=lambda kv: (kv[1][0]["category"], kv[1][0]["sub_category"], kv[0])):
        L.append(f"| {k} ({ts[0]['category'][:4]}/{ts[0]['sub_category'][:6]}) | {len(ts)} | " + " | ".join(
            f"{100 * np.mean([preds[s].get(t['id']) == t['gold'] for t in ts]):.0f}" if preds[s] else "—" for s, _ in SYSTEMS) + " |")
    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
