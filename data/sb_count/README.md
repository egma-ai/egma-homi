# sb_count

Counting turns and speakers, and telling speakers apart.

## Questions
- How many turns or speaker changes, how many speakers, who talks the longest or takes the most turns (choice, yes/no,
  score). A turn is one stretch by one speaker; a pause inside one person's talk is not a new turn.
- Which of 3-6 numbered clips matches clip N (or none), which two clips share a speaker, same speaker or not.
- Labels are exact by construction (we place every utterance, or use AMI's manual word alignments) and re-checked with an
  energy VAD per speaker track. About 20 % of non-HVB clips get a baked-in phone codec (`data/sb_phone/fx.py`).

## Sources
| Source | Use | Licence | Revision |
|---|---|---|---|
| LibriTTS-R | train.clean / train.other -> train, dev.clean -> CAL | CC BY 4.0 | `mythicinfinity/libritts_r@0d1718db351207d5a493135a62d5ed8cc4733830` |
| HarperValleyBank | per-speaker call channels, word timings | CC BY 4.0 | `cricketclub/gridspace-stanford-harper-valley@0bd721e877c4a85d8c13ff837e68661ea6200a98` |
| AMI Meeting Corpus | Mix-Headset audio, manual word alignments | CC BY 4.0 | official mirror; HF `edinburghcstr/ami@46f28f2503e2ec48f8867a84eef356c70476beab` |
| Kokoro-82M | announcer phrases ("Clip three.") only | Apache-2.0 | `hexgrad/Kokoro-82M@f3ff3571791e39611d31c381e3a41a3af07b4987` |

Speaker gender comes from LibriSpeech SPEAKERS.TXT (CC BY 4.0). Excluded: every SLUE-HVB test call and every HVB call in
our evaluation set. CAL (~3 %) has its own speakers, meetings, calls and announcer voices. The second training stage uses
sb_speakers instead of this builder's speaker-counting clips, whose length grew with the number of speakers.

## Run
    modal run --detach data/sb_count/modal_build.py::prep_all   # LibriTTS-R pools + Kokoro announcer (one L4 GPU)
    modal run --detach data/sb_count/modal_build.py::full       # ::smoke for one tiny shard, ::check to re-check
Writes `/vol/stageb/sb_count/` on the `homi` volume: `items_train.jsonl`, `items_cal.jsonl`, `audio/`, `summary.json`.

## Tests
    uv run --python 3.12 --with pytest --with numpy --with scipy --with soundfile pytest data/sb_count/tests
