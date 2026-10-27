# sb_relative

Two-part recordings of one speaker (normal voice, short tone, target clip): how does the second part compare?

## Questions
- Loudness, speaking rate, pitch and pitch variation of the second part relative to the first, as 3 ordered levels
  (score) or yes/no (louder / quieter / about the same, ...); up to 4 questions per recording.
- Labels are the measured difference. Clear: >= 3 dB, 12 % rate, 1.5 semitones, 40 % pitch spread; "about the same":
  <= 1 dB, 4 %, 0.5 st, 10 %; pairs in between are dropped. Classes are balanced with known gain, tempo and pitch
  changes (Rubber Band) to the target, which is then re-measured.

## Sources
| Source | Use | Licence | Revision |
|---|---|---|---|
| VCTK 0.92 (mic1) | sb_voice train speakers, 1-8 s clips | CC BY 4.0 | `sanchit-gandhi/vctk@73ef4ee7d49a6fed4ea1efd65f82b4c95faeb9de` |
| LibriTTS-R | sb_voice train speakers, 1-8 s clips | CC BY 4.0 | `mythicinfinity/libritts_r@0d1718db351207d5a493135a62d5ed8cc4733830` |
| CREMA-D | training actors only; reference = a neutral clip | ODbL 1.0 + DbCL | `cfahlgren1/crema-d@b7d5beded69285eb5eb6a1c6894a5b0ad17cab92`, `CheyneyComputerScience/CREMA-D@1658cd342dff90010aa843eaeebd53610a08b1dc` |
| separator tone | generated 1 kHz tone | CC0 | `relabels.py` |

Clips and features come from sb_voice (`/vol/stageb/sb_voice/work2`); its CAL speakers are never used, and CAL here
(~3 %) uses held-out speakers. VCTK items are left out of homi training (see DATA.md).

## Run and test
    modal run --detach data/sb_relative/modal_build.py::main     # --smoke for a 3 % build
    uv run --python 3.12 --with pytest --with numpy --with scipy --with soundfile pytest data/sb_relative/tests
The build writes `/vol/stageb/sb_relative/` on the `homi` volume (`items_train.jsonl`, `items_cal.jsonl`,
`audio/`, `summary.json`) and an extra evaluation set from the evaluation-set CREMA-D actors to `/vol/adi/extra/`.
