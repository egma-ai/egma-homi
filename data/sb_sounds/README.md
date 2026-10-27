# sb_sounds

Questions about non-speech audio: sound events and categories, music, mixtures, signal level and musical notes.

## Questions
- FSD50K clips: is there <sound> (negatives are ontology siblings or random classes, never related or confusable ones),
  which sound, what kind of sound (11 categories), is there music, is someone talking.
- Constructed clips: one or two kinds of sound, speech under a sound, quiet / moderate / loud, trailing silence.
- NSynth notes: instrument family, acoustic / electronic / synthesizer, register, note qualities.
- Labels are FSD50K's human-verified dev labels, NSynth metadata, or hold by construction. About 10 % of items use the
  phone channel.

## Sources
| Source | Use | Licence | Revision |
|---|---|---|---|
| FSD50K dev split | clips, labels | dataset CC BY 4.0; only clips licensed CC0 1.0 or CC BY 3.0 (CC BY-NC, Sampling+ dropped) | labels `hf:Fhrozen/FSD50k@ccf1acaff12f3f4a4c10052dddafbb6d22152c9b` (Zenodo 4060432); audio `hf:quinnlue/FSD50K-16k@2a60d475f4e2f0db624a902881af2df79d656145` |
| NSynth valid + test | notes, metadata | CC BY 4.0 | `download.magenta.tensorflow.org/datasets/nsynth/nsynth-{valid,test}.jsonwav.tar.gz` |
| LibriSpeech dev-clean | speech under sounds | CC BY 4.0 | `openslr.org/resources/12/dev-clean.tar.gz` |
| AudioSet ontology | class hierarchy only, not training content | CC BY-SA 4.0 | `github.com/audioset/ontology@d417d32bf59c711abb5910fd2f76a0eb44697991` |

FSD50K eval clips are never used. CAL (~3 %) is chosen by whole uploaders, NSynth instruments and LibriSpeech speakers.

## Run
    modal run --detach data/sb_sounds/modal_build.py::main    # --smoke for a small build; ::verify, ::check_leakage
Writes `/vol/stageb/sb_sounds/` on the `homi` volume: `items_train.jsonl`, `items_cal.jsonl`, `audio/`, `summary.json`.

## Tests
    uv run --python 3.12 --with pytest --with numpy --with scipy --with soundfile pytest data/sb_sounds/tests
