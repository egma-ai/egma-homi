# Training data

498,491 training passes in total. Everything below is public and permissively licensed, or
synthetic audio generated for this project.

## Rules

- Licences: CC0, public domain, CC BY, Apache-2.0, MIT, BSD, and CREMA-D (ODbL). No share-alike, non-commercial,
  no-derivatives or GPL data, with the one exception noted at the end.
- None of the datasets MMSU builds its audio from (MELD, GigaSpeech, Common Voice, Emilia, CoVoST 2, EDACC, VCTK,
  CHILDES, SLURP, SEAME, Fake-or-Real, RAVDESS, Switchboard), and no MMSU audio or text.
- Nothing from VocalGrad, VoxParadox or MSU-Bench.
- No customer data. No outputs from Gemini, OpenAI or Anthropic models.

## Sources

| Source | Licence | Used for |
|---|---|---|
| [LibriTTS-R](https://www.openslr.org/141/) | CC BY 4.0 | voice features, comparisons, counting speakers, intonation, phone audio |
| [LibriSpeech](https://www.openslr.org/12/) | CC BY 4.0 | real speech for counting and real-vs-synthetic, speech in sound mixtures |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) (training actors only) | ODbL 1.0 + DbCL 1.0 | emotion, pitch, loudness, rate |
| [speechocean762](https://www.openslr.org/101/) | Apache-2.0 | pronunciation, child speech |
| [HarperValleyBank](https://github.com/cricketclub/gridspace-stanford-harper-valley) (SLUE test calls excluded) | CC BY 4.0 | calls, dialogue timing, intents, counting speakers |
| [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) | CC BY 4.0 | meetings, dialogue timing, counting speakers |
| [Apollo telephony](https://huggingface.co/datasets/BrunoHays/apollo-telephony-en) | CC BY 4.0 | phone audio, real-vs-synthetic |
| [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) | CC BY 4.0 | spoken intents, phone audio, language |
| [FTC robocall audio](https://github.com/wspr-ncsu/robocall-audio-dataset) | public domain | robocalls, IVR |
| [MUSAN](https://www.openslr.org/17/) (per-file licence filtered) | CC BY 4.0 / per file | background noise and music |
| [OpenSLR 28](https://www.openslr.org/28/) simulated room responses | Apache-2.0 | room echo |
| [FSD50K](https://zenodo.org/records/4060432) (CC0 and CC BY 3.0 clips) | per clip | sound events |
| [NSynth](https://magenta.tensorflow.org/datasets/nsynth) | CC BY 4.0 | instruments, notes |
| [FMA](https://github.com/mdeff/fma) (CC BY, CC0, public-domain tracks) | per track | music |
| [OpenMIC-2018](https://github.com/cosmir/openmic-2018) labels | CC BY 4.0 | instruments |
| [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa) | MIT | spoken questions |
| [QASC](https://huggingface.co/datasets/allenai/qasc), [QuaRTz](https://huggingface.co/datasets/allenai/quartz) | CC BY 4.0 | spoken questions |
| [JADE-Data](https://huggingface.co/datasets/theunnecessarythings/JADE-Data) | Apache-2.0 | text decisions, spoken questions |
| [tasksource](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions) (commercial, non-share-alike tasks) | CC BY / MIT / Apache-2.0 / BSD / CC0 / ODC-BY | text decisions |
| [CMUdict](https://github.com/cmusphinx/cmudict) | BSD-2 | syllables, stress, rhyme |
| [misaki](https://github.com/hexgrad/misaki) | Apache-2.0 | pronunciation for TTS |

Synthetic speech: [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0), [Piper](https://github.com/rhasspy/piper) 1.2.0 with voices trained on CC BY or public-domain data (MIT),
[Parler-TTS mini v1](https://huggingface.co/parler-tts/parler-tts-mini-v1) (Apache-2.0), [SpeechT5](https://huggingface.co/microsoft/speecht5_tts) with [HiFi-GAN](https://huggingface.co/microsoft/speecht5_hifigan) and [CMU ARCTIC x-vectors](https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors) (MIT), [Chatterbox](https://huggingface.co/ResembleAI/chatterbox) (MIT), [Vocos](https://github.com/gemelo-ai/vocos)
(MIT). Pitch and timing edits use [WORLD](https://github.com/mmorise/World) / [pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) (modified BSD / MIT) and [Rubber Band](https://breakfastquay.com/rubberband/). Kokoro and Piper call the
[espeak-ng](https://github.com/espeak-ng/espeak-ng) phonemizer (GPL-3) as a tool; none of it is in the model.

Text teacher: soft labels on text questions from
[perplexity-ai/pplx-decider-v1.1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) (Apache-2.0), used
only where its answer is correct. Call-content labels for HarperValleyBank come from an open Qwen model (two runs that
agree).

## Removed

| Removed | Questions | Reason |
|---|---|---|
| VCTK | 34,539 | MMSU uses it |
| Earnings-21, Earnings-22 | 22,686 | CC BY-SA |
| share-alike tasksource tasks | 17,179 | CC BY-SA |
| VocalSound | 9,000 | CC BY-SA |
| Vystadial | 4,051 | CC BY-SA |
| share-alike FMA / OpenMIC tracks | 3,982 | CC BY-SA |
| BANKING77 | 6,491 | Decision Index benchmark |

## Credits

Per-clip credits for FSD50K, per-track credits for FMA and per-file licences for MUSAN are in the `ATTRIBUTION_*.txt`
files on Hugging Face. CREMA-D: "This model was trained using CREMA-D (Cao et al., 2014), made available under the
Open Database License (ODbL v1.0); contents under the Database Contents License (DbCL v1.0)." We treat the weights as a
Produced Work under ODbL §4.3.

## Known exception

MUSAN's licence file lists 213 of its music tracks as CC BY-SA. They were mixed in as background music in 1,368
training questions, all in the first training pass. The speech and labels in those questions are not
share-alike.
