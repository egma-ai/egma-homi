# sb_phone

Real phone and VoIP speech, plus clean speech degraded by known amounts, with questions a caller would ask about the
line.

## Questions
- Live person or recorded message, spoken language (12 choices), whether a number is said (human-transcript rule).
- Quality (very poor / some problems / clear), noise, music, background talker, breaking up, speakerphone echo.
  Degradations follow a recipe (ffmpeg codecs, packet loss, clipping, low volume, MUSAN backgrounds at a set SNR,
  simulated RIRs), so labels come from the recipe; real phone recordings only get questions about added impairments.
- sb_calls clips are re-degraded and keep their own questions. Yes/no families are 50/50 per split; at most 4 per clip.

## Sources
| Source | Use | Licence | Revision |
|---|---|---|---|
| Apollo telephony EN | mission-control loops; train partition -> train, dev -> CAL | CC BY 4.0 | `BrunoHays/apollo-telephony-en@b9d3a68a9faa550ae1f2069eb1eb7634cdc8966c` |
| Vystadial 2013 EN | VoIP caller utterances | CC BY-SA 3.0 (share-alike; excluded from training, see DATA.md) | `BrunoHays/vystadial-telephony-en@b7454512a3547c829be201d9afceb64af8c28e61` |
| MInDS-14 | phone requests, 11 non-English locales only | CC BY 4.0 | `PolyAI/minds14@40ce77cb32a384e4d50a568e1ec39ac804019d33` |
| FTC robocall audio | robocall recordings (remote-party channel) | public domain (the repo's CC BY-ND covers its documents only) | `wspr-ncsu/robocall-audio-dataset@5aa6f3bfa8563ce8c1c75ebf8a2271e6ff6b4272` |
| VCTK 0.92, LibriTTS-R | clean speech from the sb_voice banks | CC BY 4.0 | see sb_voice |
| sb_calls | Kokoro-82M call clips and tones | Apache-2.0 | see sb_calls |
| MUSAN (OpenSLR 17) | noise, music and speech backgrounds; per-file licences parsed, NC / ND / Sampling+ dropped | CC BY 4.0 corpus | `musan.tar.gz` |
| RIRS_NOISES (OpenSLR 28) | simulated RIRs only | Apache-2.0 | `rirs_noises.zip` |

Not used: MInDS-14 English locales (evaluation data), real RIRs (third-party licences), robocall Whisper transcripts.
CAL holds whole Vystadial speakers, the Apollo dev partition, whole robocall cases, sb_voice / sb_calls CAL clips, and
disjoint MUSAN files and rooms.

## Run
    modal run --detach data/sb_phone/modal_build.py::prep    # download and decode the sources once
    modal run --detach data/sb_phone/modal_build.py::build   # --smoke for a ~4 % run

Output on the `egma-homi` volume: `/vol/stageb/sb_phone/` (`items_train.jsonl`, `items_cal.jsonl`, `audio/`,
`summary.json`).

## Tests
    uv run --python 3.12 --with pytest --with numpy --with scipy --with soundfile pytest data/sb_phone/tests
