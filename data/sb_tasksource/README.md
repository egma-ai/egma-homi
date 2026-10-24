# sb_tasksource

Text-only decision questions (no audio) from tasksource, filtered to commercially usable licences.

## Questions
- choice, yes/no and score (ordered levels) rows converted with `common.make_item`; options that cite other options
  by position ("all of the above") keep their order.
- Hard labels become one-hot targets (0.05 smoothing in train); votes, mean ratings and probabilities stay soft.
  Groups (`group_id`) are never split between train and CAL.

## Sources
| Source | Use | Licence | Revision |
|---|---|---|---|
| [tasksource/tasksource-jev-typed-decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions) | config `full`, rows whose original split is `train` | card `other`; filtered per task | `5d3d8a557f62809d690f22c2c696b44c266d1ada` |
| [tasksource](https://github.com/sileod/tasksource) (Damien Sileo, LREC-COLING 2024) | `task_licenses()` snapshot (`task_licenses.json`, via `licences_snapshot.py`), `jev_source_scores.csv`, bucket regexes from `metadata/jev_mixes.py` (copied into `build.py` with attribution) | tasksource code | `d4f939cdaa900a7b70c6d4dd9f4f77a10a6306b7` |

A task is kept only if tasksource, `sources.yaml` and every row say `commercial` and every licence recorded for it is
CC0 / public domain, CC BY, CC BY-SA (flagged), Apache-2.0, MIT, BSD, ODC-BY or ODbL. Also dropped: leaderboard
benchmarks (`LEADERBOARD_PATTERNS`), tasks built from closed-model outputs, `dgen`, and items that share text or an
option set with our evaluation set. Each dropped task and its reason is listed in `summary.json`.

## Run
    modal run --detach data/sb_tasksource/modal_build.py::main   # CPU only; ::smoke writes /vol/stageb_smoke/, ::validate re-checks
Writes `/vol/stageb/sb_tasksource/` on the `egma-homi` volume: `items_train.jsonl`, `items_cal.jsonl`, empty `audio/`,
`summary.json`.

## Tests
    uv run --python 3.12 --with pytest --with numpy --with scipy --with soundfile pytest data/sb_tasksource/tests
