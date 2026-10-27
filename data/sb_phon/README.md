# sb_phon

Word-sound questions on synthetic speech: syllables, stress, emphasis, intonation, disfluencies, minimal pairs,
consonant and vowel sounds, and final-stop release.

## Questions
- Choice (2-8 options), yes/no (exactly 50/50 per question and split) and ordered score items; up to 4 per clip.
- Labels come from CMUdict and are kept only when they match the misaki phonemes Kokoro actually spoke. Signal edits
  (emphasis, contours, list rises, removed releases) are measured back; clips with too weak an edit are dropped.
- About 20% of clips get a baked-in phone channel (band-limit plus mu-law / A-law / GSM / Opus).

## Sources
| Source | Use | Licence | Revision |
|---|---|---|---|
| Kokoro-82M (28 English voices) | all speech, word and phoneme timings | Apache-2.0 | `hexgrad/Kokoro-82M@f3ff3571791e39611d31c381e3a41a3af07b4987` |
| CMUdict | syllable counts, stress, phones | BSD-2-clause style | `cmusphinx/cmudict@74790861f652b15e4ac49015a90074ad62a27690` |
| misaki gold lexicons | vocabulary filter, noun / verb heteronyms | Apache-2.0 | `misaki==0.9.4` |
| pyworld / WORLD | pitch, energy and duration edits | MIT / modified BSD | `pyworld==0.3.5` |

Sentences and templates are written in `lexicon.py`. CAL uses held-out voices `af_nova`, `am_eric`, `bf_lily`. No MMSU
audio or text is used; its question strings are read only to check that no item question equals one.

## Run
    modal run --detach data/sb_phon/modal_build.py::full       # ::smoke for a small run

Writes `/vol/stageb/sb_phon/` on the `homi` volume (`items_train.jsonl`, `items_cal.jsonl`, `audio/`, `summary.json`).

## Tests
    uv run --python 3.12 --with pytest --with numpy --with scipy --with soundfile pytest data/sb_phon/tests
