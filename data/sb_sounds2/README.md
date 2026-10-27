# sb_sounds2

Hard-negative sound questions on FSD50K, and music questions on licence-clean FMA and OpenMIC-2018 audio.

## Questions
- FSD50K presence (yes/no, 50/50) and which-of-these (choice): most "no" answers and all distractors are near
  negatives (ontology siblings, confusable classes, cousins), never a label, its ancestor or descendant, or a class
  that often co-occurs with it.
- FMA: genre and sub-genre (choice among sibling genres), genre yes/no, tempo (measured, kept only when unambiguous),
  level (set by construction), live vs studio. OpenMIC-2018: instruments (present >= 0.75 relevance, absent <= 0.25),
  vocals, vocal-free use.
- At most 3 items per clip, about 10% on the phone channel. CAL holds out whole Freesound uploaders and artists.

## Sources
| Source | Use | Licence | Revision |
|---|---|---|---|
| FSD50K, dev split only | sound questions | dataset CC BY 4.0; only clips under CC0 1.0 or CC BY 3.0 kept | labels `Fhrozen/FSD50k@ccf1acaff12f3f4a4c10052dddafbb6d22152c9b`, audio `quinnlue/FSD50K-16k@2a60d475f4e2f0db624a902881af2df79d656145` |
| FMA "large" (30 s clips) | genre, tempo, level, live | metadata CC BY 4.0; a track is kept only if its official licence and the repack's agree on CC BY, CC0, public domain or CC BY-SA (flagged `meta.share_alike`) | audio `benjamin-paine/free-music-archive-large@d4cb3e133a7c5a007ddd20458aa30aef8968cf27`, metadata `fma_metadata.zip` (github.com/mdeff/fma) |
| OpenMIC-2018 v1.0.0 | instruments, vocals | labels CC BY 4.0; audio is FMA tracks under the same licence rule | Zenodo 1432913, md5 `e4ccf187e2bb5ab2e115416e8aafe7f4` |
| AudioSet ontology | class hierarchy only | CC BY-SA 4.0 (not training content) | `audioset/ontology@d417d32bf59c711abb5910fd2f76a0eb44697991` |
| Song Describer `audio_licenses.txt` | exclusion list only | - | Zenodo 10072001 |

Not used: MTG-Jamendo (non-commercial labels), MusicCaps and AudioSet audio (YouTube), FSD50K eval. FMA artists in
Song Describer are dropped, and music excerpts are scanned for near-duplicates of our evaluation set's MuChoMusic tracks.

## Run
    modal run --detach data/sb_sounds2/modal_build.py::prep   # FMA shards (cached)
    modal run --detach data/sb_sounds2/modal_build.py::main   # full build, CPU only
    modal run data/sb_sounds2/modal_build.py::finalize        # second-stage near-duplicate check

Writes `/vol/stageb/sb_sounds2/` on the `homi` volume: `items_train.jsonl`, `items_cal.jsonl`, `audio/`,
`summary.json`.

## Tests
    uv run --python 3.12 --with pytest --with numpy --with scipy --with soundfile pytest data/sb_sounds2/tests
