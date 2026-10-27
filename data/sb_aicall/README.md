# sb_aicall

Two-party calls (8-180 s) where one side is an AI voice agent and the other is a live caller, a voicemail greeting, an
automated system or nobody; some calls have two humans and no AI.

## Questions
- Who or what is on the other end, voicemail tone and message left, automated caller (yes/no and choice).
- Timing: waits and silences longer than N seconds, barge-in, AI talk-over, talk share, first and last speaker.
- Timings are laid out by the builder; labels use margins, are re-checked with an energy VAD of each track, and
  ambiguous cases are not asked. Yes/no answers are balanced per family and threshold.

## Sources
| Source | Use | Licence | Revision |
|---|---|---|---|
| HarperValleyBank | live callers, re-voiced agent turns, two-human calls | CC BY 4.0 | `github.com/cricketclub/gridspace-stanford-harper-valley@0bd721e877c4a85d8c13ff837e68661ea6200a98` |
| Kokoro-82M | AI agent, voicemail, IVR and notification voices | Apache-2.0 | `hexgrad/Kokoro-82M@f3ff3571791e39611d31c381e3a41a3af07b4987` |
| FTC robocall audio | automated callers | public domain | `github.com/wspr-ncsu/robocall-audio-dataset@5aa6f3bfa8563ce8c1c75ebf8a2271e6ff6b4272` |
| numpy (`sb_calls/audio_fx.py`) | ringback, beep, line noise | own code | - |

SLUE-HVB test calls (`asapp/slue-phase-2@e2989c55a53593a8e39b8f8ebdb47ccaccbe484a`) are excluded. CAL uses held-out HVB
speaker pairs, voices `af_sky`, `bm_george`, `am_puck` and FTC cases. Five fictional business names and one caller line
in `aic_scripts.py` were reworded after training; a rebuild differs only there.

## Run
Needs the raw HVB and robocall copies made by `sb_dialog` and `sb_phone`; TTS uses up to 4 L4 GPUs.

    modal run --detach data/sb_aicall/modal_build.py::full     # ::smoke for a small run, ::check to re-validate

Writes `/vol/stageb/sb_aicall/` on the `homi` volume (`items_train.jsonl`, `items_cal.jsonl`, `audio/`, `summary.json`).

## Tests
    uv run --python 3.12 --with pytest --with numpy --with scipy --with soundfile pytest data/sb_aicall/tests
