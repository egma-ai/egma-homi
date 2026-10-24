# sb_voice

Questions about how something is said (loudness, speed, pitch, pauses, emotion) and who is speaking.

## Questions
- Measured prosody: loudness, speaking rate, pitch, pitch variation and pauses, each relative to the same speaker's
  other clips (per-speaker tertiles; a soft 0.7 / 0.3 target within 5 percentile points of a tertile boundary).
- CREMA-D intensity (from the file name) and emotion (target = 0.4 x listener vote share + 0.6 x intended emotion).
- Speaker traits from corpus metadata: gender, age bracket, child or adult, accent (VCTK accent groups).
- Same speaker or two (two utterances joined at matched levels), vocal sounds (6 VocalSound classes), pronunciation
  (speechocean762 scores in 3 levels), background noise (synthetic noise mixed in at 0-20 dB SNR).
- Choice, yes/no (50/50) and score views; at most 4 questions per clip; ~15 % of items use the phone channel.

## Sources
| Source | Use | Licence | Revision |
|---|---|---|---|
| VCTK 0.92 (mic1) | prosody, gender, age, accent, pairs, noise | CC BY 4.0 (excluded from training as an MMSU source, see DATA.md) | `sanchit-gandhi/vctk@73ef4ee7d49a6fed4ea1efd65f82b4c95faeb9de` |
| LibriTTS-R | prosody, gender, pairs, noise; gender from LibriSpeech SPEAKERS.TXT (CC BY 4.0) | CC BY 4.0 | `mythicinfinity/libritts_r@0d1718db351207d5a493135a62d5ed8cc4733830` |
| CREMA-D | training actors only; demographics from VideoDemographics.csv | ODbL 1.0 + DbCL 1.0 | `cfahlgren1/crema-d@b7d5beded69285eb5eb6a1c6894a5b0ad17cab92`, `CheyneyComputerScience/CREMA-D@1658cd342dff90010aa843eaeebd53610a08b1dc` |
| VocalSound | official train split speakers, 44.1 kHz release zip | CC BY-SA 4.0 (share-alike; excluded from training, see DATA.md) | `YuxinXie/VocalSound@dad0fcfa060d5291948f23fad33117068b28e121`; official repo `YuanGongND/vocalsound@337ac5526f392987ab6e9214bf986bc923639338` |
| speechocean762 | train split only | Apache-2.0 | `mispeech/speechocean762@06385584fad212b26134c656fdd3ccf9f093f33e` |
| Synthetic noise | white, pink, brown, hum, rumble (`labels.colored_noise`) | CC0 (generated) | - |

Held out: CREMA-D clips and actors in the evaluation items, VocalSound val/test, speechocean762 test. CAL uses held-out
speakers only. `librispeech_speaker_gender.json` and `crema_demographics.json` are small metadata extracts.

## Run
    modal run --detach data/sb_voice/modal_build.py::main    # --skip-stage1 re-assembles from the work files
    modal run data/sb_voice/modal_build.py::check             # re-validate the written files

Output on the `egma-homi` volume: `/vol/stageb/sb_voice/` (`items_train.jsonl`, `items_cal.jsonl`, `audio/`,
`summary.json`).

## Tests
    uv run --python 3.12 --with pytest --with numpy --with scipy --with soundfile pytest data/sb_voice/tests
