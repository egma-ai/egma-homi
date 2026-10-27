# sb_disfluency

Naming the combination of disfluencies in an utterance, and picking the written line that continues a spoken line
with the same meter and rhyme. All audio is Kokoro TTS of text generated in code; labels are exact by construction
and re-checked on the TTS words and phonemes.

## Questions
- `sb_disfluency_disfl`: 90 base sentences scripted with any combination of filled pauses, discourse markers,
  restarts and editing terms (uniform over the 16 combinations). Asked as a choice over combinations, yes/no per type
  (exactly 50/50), a none-to-four count, and fluent yes/no. Plain pauses are added to fluent stretches too.
- `sb_disfluency_verse`: generated lines in iambic, trochaic, anapestic or dactylic meter. Candidates follow a 2x2
  design (meter kept or broken x rhyme kept or broken); also rhyme, syllable-count and meter-name questions.
- Stress and rhyme come from CMUdict and misaki, and must agree across CMUdict, misaki US and misaki GB.

## Sources
| Source | Use | Licence | Revision |
|---|---|---|---|
| Kokoro-82M (`kokoro==0.9.4`) | all speech, 28 US/GB voices | Apache-2.0 (weights and code) | `hexgrad/Kokoro-82M@f3ff3571791e39611d31c381e3a41a3af07b4987` |
| misaki 0.9.4 | G2P and gold lexicons; espeak-ng (GPL) fallback disabled, clips with unknown words dropped | Apache-2.0 | 0.9.4 |
| spaCy en_core_web_sm | POS tags for misaki | MIT | 3.8.0 |
| CMUdict | syllables, stress, rhyme keys | BSD-2 | `cmusphinx/cmudict@74790861f652b15e4ac49015a90074ad62a27690` |

CAL uses held-out voices af_nova, am_eric and bf_lily. About 20% of clips get a phone channel via `data/sb_phone/fx.py`.

## Run
    modal run --detach data/sb_disfluency/modal_build.py::smoke   # small run in /vol/stageb_work/sb_disfluency/smoke
    modal run --detach data/sb_disfluency/modal_build.py::full    # one L4 for TTS, CPU for the rest

Writes `/vol/stageb/sb_disfluency/` on the `homi` volume: `items_train.jsonl`, `items_cal.jsonl`, `audio/`,
`summary.json` (written last).

## Tests
    uv run --python 3.12 --with pytest --with numpy --with scipy --with soundfile pytest data/sb_disfluency/tests

The MMSU question-equality test is a no-op outside Modal.
