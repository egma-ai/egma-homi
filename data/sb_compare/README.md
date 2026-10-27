# sb_compare

Comparisons between parts of a clip (pitch, speed, volume, vocal range), total speaking time, and pauses.

## Questions
- Pitch / speed / volume / range of 2-4 LibriTTS-R segments: pattern ("high-low-medium"), highest or lowest part, "is part
  j higher than part i", lower / same / higher. Pitch, speed and volume are shifted by known amounts; range is natural.
- Duration: total speaking time, silence between sentences, which utterance is longer. Pauses inserted into Kokoro
  sentences: is there one, after which word, how many, how long, where, and which reading of an ambiguous sentence.
- Every label is re-measured on the final clip (energy VAD, pYIN F0, syllable rate, power); clips that miss the margins
  or disagree with the plan are dropped. About 20 % of clips get a baked-in phone codec.

## Sources
| Source | Use | Licence | Revision |
|---|---|---|---|
| LibriTTS-R dev-clean + train-clean-100 | comparison and duration clips | CC BY 4.0 | `mythicinfinity/libritts_r@0d1718db351207d5a493135a62d5ed8cc4733830` |
| Kokoro-82M | pause sentences with word timings | Apache-2.0 | `hexgrad/Kokoro-82M@f3ff3571791e39611d31c381e3a41a3af07b4987` |
| Generated | tones, silence, noise; Rubber Band and ffmpeg as tools | CC0 | - |

CAL: 10 LibriTTS-R speakers and Kokoro voices af_sky, bm_george, am_puck. sb_voice's CAL speakers, VCTK, CREMA-D, Common
Voice and MMSU audio are not used; no question equals an MMSU question.

## Run
    modal run --detach data/sb_compare/modal_build.py::main    # --smoke for a ~3 % build; Kokoro runs on L4 GPUs
Writes `/vol/stageb/sb_compare/` on the `homi` volume: `items_train.jsonl`, `items_cal.jsonl`, `audio/`, `summary.json`.

## Tests
    uv run --python 3.12 --with pytest --with numpy --with scipy --with soundfile pytest data/sb_compare/tests
