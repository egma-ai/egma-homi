# sb_spoken2

Multiple-choice reasoning questions read aloud by TTS; the typed text only says e.g. "Answer the question in the audio."

## Questions
- Formats: `read` (question and lettered options spoken), `stem` (question spoken, options typed), `yn`, `valid` (is
  the spoken argument valid?) and `passage` (premises spoken, question typed). Yes/no answers are 50/50 per format.
- CommonsenseQA (rows not used by sb_spoken), QASC and QuaRTz questions with their gold answers, plus generated
  puzzles with exact labels (ordering, truth-tellers and liars, navigation, calendar and clock arithmetic, word
  problems, family relations, syllogisms and propositional arguments); the tests re-derive the puzzle labels.

## Sources
| Source | Use | Licence | Revision |
|---|---|---|---|
| CommonsenseQA | train rows not used by sb_spoken | MIT (HF cardData) | `tau/commonsense_qa@94630fe30dad47192a8546eb75f094926d47e155` |
| QASC | train + validation | CC BY 4.0 (HF cardData) | `allenai/qasc@a34ba204eb9a33b919c10cc08f4f1c8dae5ec070` |
| QuaRTz | train + validation + test | CC BY 4.0 (HF cardData) | `allenai/quartz@28c1dbb56caf81799296cb17892fa73402e23464` |
| Kokoro-82M (`kokoro==0.9.4`, `misaki[en]==0.9.4`) | TTS, 27 voices with Whisper WER <= 0.06 | Apache-2.0 | `hexgrad/Kokoro-82M@f3ff3571791e39611d31c381e3a41a3af07b4987` |

CAL (~3 %) uses held-out rows / puzzles and 4 held-out voices (af_kore, am_liam, bf_lily, bm_george). Clips whose text
matches our evaluation set (exact match or a shared 8-word shingle) are dropped.

## Run and test
    modal run --detach data/sb_spoken2/modal_build.py::main      # ::smoke for a ~100-item run
    uv run --python 3.12 --with pytest --with numpy --with scipy --with soundfile pytest data/sb_spoken2/tests
The build writes `/vol/stageb/sb_spoken2/` on the `egma-homi` volume: `items_train.jsonl`, `items_cal.jsonl`,
`audio/`, `summary.json`.
