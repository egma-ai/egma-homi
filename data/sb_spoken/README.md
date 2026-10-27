# sb_spoken

Questions about what was said: intents and dialog acts in real bank calls, spoken questions read by TTS, and listening
checks.

## Questions
- HarperValleyBank turns: caller intent (all 8 intents) and dialog act (all 18 acts), as choice and yes/no items;
  labels are the corpus annotations. CAL holds out whole speakers and their calls.
- Kokoro reads CommonsenseQA questions (with or without the lettered options, or as a yes/no question) and JADE-Data
  passages; CAL uses one held-out voice per accent/gender group.
- Listening checks labelled from the synthesised text: is a number said, is a question asked, which word was said.
  Yes/no items are 50/50; at most 4 items per clip.

## Sources
| Source | Use | Licence | Revision |
|---|---|---|---|
| SLUE-HVB (`asapp/slue-phase-2`, config `hvb`) | train split only | CC BY 4.0 (Gridspace-Stanford Harper Valley) | `e2989c55a53593a8e39b8f8ebdb47ccaccbe484a`, parquet `c3e476fce94cbd006aafb52187125b748b2c037e` |
| CommonsenseQA (`tau/commonsense_qa`) | train questions | MIT | `94630fe30dad47192a8546eb75f094926d47e155` |
| JADE-Data (`theunnecessarythings/JADE-Data`) | `train.jsonl`, rows not used by sb_jade | Apache-2.0 | `ca75bcb94afd60c4e9c0c17133da4c731e527ea2` |
| Kokoro-82M (`hexgrad/Kokoro-82M`, kokoro 0.9.4) | TTS; voices with Whisper WER <= 0.06 | Apache-2.0 | `f3ff3571791e39611d31c381e3a41a3af07b4987` |

No call from our evaluation set is used; `items_train_strict.jsonl` also drops HVB speakers that occur in it.

## Run
    modal run --detach data/sb_spoken/modal_build.py::main   # TTS on a GPU; ::smoke for a small run

Writes `/vol/stageb/sb_spoken/` on the `homi` volume: `items_train.jsonl`, `items_cal.jsonl`,
`items_train_strict.jsonl`, `audio/`, `summary.json`.

## Tests
    uv run --python 3.12 --with pytest --with numpy --with scipy --with soundfile pytest data/sb_spoken/tests
