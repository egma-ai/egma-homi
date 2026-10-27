# teacher

Soft labels for the text-only items, from a text decision model, used for distillation.

## Labels
- Every text item of `sb_jade` and `sb_tasksource` (train split) is scored; audio items, items with more than 40
  options and prompts over 8192 tokens are skipped.
- Yes/no items are asked as teacher `noul` questions, all others as `choice` questions with our options as criteria;
  probabilities are mapped back to our option order. A label is kept only if its fingerprint (state, question, option
  order) matches the current item.

## Sources
| Source | Use | Licence | Revision |
|---|---|---|---|
| perplexity-ai/pplx-decider-v1.1-27b | teacher, run with its own `source/src/autojev/model.py` and pinned requirements (torch 2.14.0, transformers 5.17.0, flash-linear-attention 0.5.2); saved temperature 1.00874 | Apache-2.0 | `5cd25e3f8ae59ba65d1ebea74d8a06fa4a5f60fd` |

The teacher was trained on tasksource data, so its accuracy on sb_tasksource items is likely inflated by memorisation.

## Run
    modal run data/teacher/modal_teacher.py::smoke_main         # 50 sb_jade CAL items: accuracy and latency
    modal run --detach data/teacher/modal_teacher.py::main      # sb_jade, then sb_tasksource once it is final

Writes `/vol/stageb/teacher/` on the `homi` volume: `soft_labels.jsonl` (`id`, `probs`, `teacher_acc_gold`),
`summary.json` (written last), `smoke.json` and raw per-shard scores in `parts/`.

## Tests
    uv run --python 3.12 --with pytest --with numpy --with scipy --with soundfile pytest data/teacher/tests
