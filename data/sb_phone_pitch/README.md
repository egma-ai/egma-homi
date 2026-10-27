# sb_phone_pitch

Phone-line and pitch / prosody questions built without VCTK or Vystadial. Labels come from the degradation recipe,
dataset metadata or human transcripts, or from signal measurements with margins; no model output is used as a label.

## Questions
- Phone (`sb_phone_pitch_phone_<family>`): the sb_phone families (live or recorded, language, number, quality, noise,
  breakup, music, background talker, far-field, call outcome), degraded with `fx.py` (a copy of `data/sb_phone/fx.py`).
- Prosody: `_rel_*` (same speaker twice, joined by a tone: loudness, rate, pitch, pitch variation of the second part),
  `_abs_*` (pitch, pitch variation, rate of one clip, in bins with gaps) and `_cmp_*` (two different speakers).
- Yes/no exactly 50/50 per family and split; choice / score answers capped at 2x the rarest; at most 4 items per clip.

## Sources
| Source | Use | Licence | Revision |
|---|---|---|---|
| HarperValleyBank | real 8 kHz calls, both channels | CC BY 4.0 | `cricketclub/gridspace-stanford-harper-valley@0bd721e877c4a85d8c13ff837e68661ea6200a98` |
| Apollo telephony EN | phone speech | CC BY 4.0 | `BrunoHays/apollo-telephony-en@b9d3a68a9faa550ae1f2069eb1eb7634cdc8966c` |
| MInDS-14, 11 non-English locales | phone speech, language | CC BY 4.0 | `PolyAI/minds14@40ce77cb32a384e4d50a568e1ec39ac804019d33` |
| FTC robocall audio | recorded calls | public domain | `wspr-ncsu/robocall-audio-dataset@5aa6f3bfa8563ce8c1c75ebf8a2271e6ff6b4272` |
| LibriTTS-R | clean base for phone families; prosody | CC BY 4.0 | `mythicinfinity/libritts_r@0d1718db351207d5a493135a62d5ed8cc4733830` |
| CREMA-D, training actors only | prosody | ODbL 1.0 + DbCL 1.0 | `cfahlgren1/crema-d@b7d5beded69285eb5eb6a1c6894a5b0ad17cab92`, GitHub `@1658cd342dff90010aa843eaeebd53610a08b1dc` |
| Kokoro-82M clips from sb_calls | call outcomes, re-degraded | Apache-2.0 | `hexgrad/Kokoro-82M@f3ff3571791e39611d31c381e3a41a3af07b4987` |
| MUSAN, CC BY / CC0 / public-domain files only | noise, music, background speech | CC BY 4.0 corpus, per-file filter | OpenSLR 17 |
| OpenSLR 28 simulated RIRs | far-field | Apache-2.0 | OpenSLR 28 |

Inputs are read from the sb_phone, sb_voice, sb_relative, sb_dialog and sb_calls outputs on the volume. Excluded:
SLUE-HVB test calls (`asapp/slue-phase-2@e2989c55a53593a8e39b8f8ebdb47ccaccbe484a`), calls and actors in our
evaluation set, and sb_voice CAL speakers. CAL uses held-out speakers, calls, MUSAN files and RIR rooms.

## Run
    modal run --detach data/sb_phone_pitch/modal_build.py::main   # CPU only; add --smoke for a 3% run

Writes `/vol/stageb/sb_phone_pitch/` on the `homi` volume: `items_train.jsonl`, `items_cal.jsonl`, `audio/`,
`summary.json` (written last).

## Tests
    uv run --python 3.12 --with pytest --with numpy --with scipy --with soundfile pytest data/sb_phone_pitch/tests
