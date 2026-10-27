# sb_speakers

Counting how many different people speak in a clip; clip length is drawn independently of the number of speakers.

## Questions
- choice "how many different speakers" (3-6 consecutive counts), yes/no "exactly / more than / fewer than / at least
  N" (50/50 per kind), and score "one / two / three / four or more"; at most 4 questions per clip.
- Labels are exact: N is the number of distinct real speakers placed in the clip, each re-checked to be audible on its
  own track (energy VAD). Mean clip duration is equalised across N before publishing.

## Sources
| Source | Use | Licence | Revision |
|---|---|---|---|
| LibriTTS-R | utterance pools shared with sb_count | CC BY 4.0 | `mythicinfinity/libritts_r@0d1718db351207d5a493135a62d5ed8cc4733830` |
| LibriSpeech (original audio) | utterances; reader gender from SPEAKERS.TXT (`librispeech_speaker_gender.json`) | CC BY 4.0 | `openslr/librispeech_asr@71cacbfb7e2354c4226d01e70d77d5fca3d04ba1` |
| HarperValleyBank | per-speaker channels and word timings (sb_dialog raw copy) | CC BY 4.0 | `cricketclub/gridspace-stanford-harper-valley@0bd721e877c4a85d8c13ff837e68661ea6200a98` |
| AMI | Mix-Headset stretches where exactly one participant speaks | CC BY 4.0 | HF `edinburghcstr/ami@46f28f2503e2ec48f8867a84eef356c70476beab` |
| MUSAN noise | background noise; only Public Domain / CC BY 3.0 files, none with voices | MUSAN CC BY 4.0 | openslr 17 |

CAL speakers, chapters, calls and meetings are disjoint from train. SLUE-HVB test calls
(`asapp/slue-phase-2@e2989c55a53593a8e39b8f8ebdb47ccaccbe484a`) and HVB calls in our evaluation set are excluded.

## Run
    modal run --detach data/sb_speakers/modal_build.py::full   # CPU only; ::smoke for a small run, ::check to re-check
Writes `/vol/stageb/sb_speakers/` on the `homi` volume: `items_train.jsonl`, `items_cal.jsonl`, `audio/`,
`summary.json`.

## Tests
    uv run --python 3.12 --with pytest --with numpy --with scipy --with soundfile pytest data/sb_speakers/tests
`test_mmsu_questions.py` is skipped locally (it needs the volume); `::tests` runs all tests in Modal.
