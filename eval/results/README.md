# Raw per-model answers

Every model's raw answers on MMSU and VocalGrad, so anyone can re-score them without running a model. No audio is
stored here; it comes from the benchmarks themselves (`eval/modal_eval.py::mmsu --step prepare`,
`eval/bench_prepare.py::main`).

```
results/<bench>/truth.jsonl                               question, options, answer key, task (from the benchmark)
results/<bench>/items<bench>.jsonl                        our items (question + options in the published order)
results/<bench>/runs/q<bench>/homi/preds.jsonl       homi: log-probabilities per option (T = 1)
results/<bench>/runs/q<bench>/homi/timing.json       homi: GPU seconds, for the cost column
results/<bench>/runs/q<bench>/qwen-stock/...              the untouched base model, same harness
results/<bench>/runs/q<bench>/gemini_<model>/responses.jsonl   Gemini: JSON answer, probabilities, token usage
results/<bench>/runs/q<bench>/openai_gpt-audio/responses.jsonl OpenAI: JSON answer or the reply it gave instead
```

Re-score:

```bash
python eval/mmsu_report.py  --dir eval/results/mmsu --out MMSU.md                               # needs numpy
python eval/bench_report.py --dir eval/results/vocalgrad --name vocalgrad --out VOCALGRAD.md --title VocalGrad
```

Our reports from these files: [MMSU_RESULTS.md](MMSU_RESULTS.md), [VOCALGRAD_RESULTS.md](VOCALGRAD_RESULTS.md).
MMSU (ddwang2000/MMSU) is MIT-licensed; VocalGrad (ynklab/vocalgrad) is CC BY 4.0. Their question text in truth.jsonl
belongs to those benchmarks.
