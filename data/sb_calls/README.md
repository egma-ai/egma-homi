# sb_calls

Synthetic outbound-call outcomes (1-15 s): a live person, a personal or carrier voicemail, a business phone menu, a
call screener, no answer, busy, not in service, or a fax / modem.

## Questions
- `who` (choice over the 9 classes or a 4-6 option subset) on every clip.
- Yes/no: live person, voicemail, automated system, tone played, start leaving a message now, busy or out of service.
  Each is exactly 50/50 per split; ambiguous cases are not asked; up to 4 items per clip.
- Labels are exact by construction; truncation never cuts the part that defines the class.

## Sources
| Source | Use | Licence | Revision |
|---|---|---|---|
| Kokoro-82M (28 English voices) | all speech; scripts generated from templates with fictional names | Apache-2.0 (HF `cardData.license`, not gated) | `hexgrad/Kokoro-82M@f3ff3571791e39611d31c381e3a41a3af07b4987` |
| numpy (`audio_fx.py`) | ringback, busy, SIT, fax / modem, beep, line noise | own code | - |

CAL uses held-out voices (`af_nova`, `am_eric`, `bf_lily`, `bm_lewis`) and the last templates of each list in
`scripts.py`. `voice_wer.json` holds the Whisper-small WER per voice (all <= 0.10, from `probe.py`).

## Run
    modal run --detach data/sb_calls/modal_build.py::full      # ::smoke for a ~1% run

Writes `/vol/stageb/sb_calls/` on the `egma-homi` volume (`items_train.jsonl`, `items_cal.jsonl`, `audio/`,
`summary.json`). `check_asr.py` (Whisper spot check, `asr_check.json`) and `probe.py` are not part of the build.

## Tests
    uv run --python 3.12 --with pytest --with numpy --with scipy --with soundfile pytest data/sb_calls/tests
