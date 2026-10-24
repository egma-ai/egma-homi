# sb_intonation

Intonation contours and pause meaning. Every label is set by construction and measured back on the rendered audio.

## Questions
- Intonation (`sb_intonation_inton_*`, `sb_intonation_final_*`): real and Kokoro speech resynthesised with WORLD and an
  imposed contour (falling, rising, fall-rise, rise-fall) or final rise / fall; which contour, yes/no, final direction.
  Each clip is re-measured with Harvest F0 and dropped if it does not match.
- Pause meaning (`sb_intonation_pause_*`): 206 own sentences (`sentences.py`) whose meaning depends on where the pause
  falls; which meaning, where the pause is, and yes/no versions. Pauses are checked with an energy VAD.

## Sources
| Source | Use | Licence | Revision |
|---|---|---|---|
| LibriTTS-R dev-clean + train-clean-100 (via `/vol/stageb/sb_voice/work2`, sb_voice CAL speakers excluded) | real intonation clips; transcripts as TTS text | CC BY 4.0 | `mythicinfinity/libritts_r@0d1718db351207d5a493135a62d5ed8cc4733830` |
| Kokoro-82M (`kokoro==0.9.4`) | TTS intonation clips, all pause clips | Apache-2.0 (code and weights) | `hexgrad/Kokoro-82M@f3ff3571791e39611d31c381e3a41a3af07b4987` |
| misaki | G2P and word timestamps | Apache-2.0 | 0.9.4 |
| pyworld / WORLD | contour editing, Harvest verification (tool only) | MIT / modified BSD | 0.3.5 |
| own text | sentences, meanings, foils, question wording | own | - |

CAL (~3%) uses 8 held-out LibriTTS-R speakers and Kokoro voices af_nova, bm_lewis, am_echo. ~20% of clips get a phone
channel from `data/sb_phone/fx.py`.

## Run
    modal run --detach data/sb_intonation/modal_build.py::full   # Kokoro on one L4 GPU, the rest on CPU; ::smoke for a small run
Writes `/vol/stageb/sb_intonation/` on the `egma-homi` volume: `items_train.jsonl`, `items_cal.jsonl`, `audio/`,
`summary.json`.

## Tests
    uv run --python 3.12 --with pytest --with numpy --with scipy --with soundfile pytest data/sb_intonation/tests
`test_no_question_equals_mmsu` needs /vol/bench_mmsu/truth.jsonl and is skipped locally; the Modal build runs it.
