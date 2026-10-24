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
| LibriTTS-R | CC BY 4.0 | voice features, comparisons, counting speakers, intonation, phone audio |
| LibriSpeech | CC BY 4.0 | real speech for counting and real-vs-synthetic, speech in sound mixtures |
| CREMA-D (training actors only) | ODbL 1.0 + DbCL 1.0 | emotion, pitch, loudness, rate |
| speechocean762 | Apache-2.0 | pronunciation, child speech |
| HarperValleyBank (SLUE test calls excluded) | CC BY 4.0 | calls, dialogue timing, intents, counting speakers |
| AMI | CC BY 4.0 | meetings, dialogue timing, counting speakers |
| Apollo telephony | CC BY 4.0 | phone audio, real-vs-synthetic |
| MInDS-14 | CC BY 4.0 | spoken intents, phone audio, language |
| FTC robocall audio | public domain | robocalls, IVR |
| MUSAN (per-file licence filtered) | CC BY 4.0 / per file | background noise and music |
| OpenSLR 28 simulated room responses | Apache-2.0 | room echo |
| FSD50K (CC0 and CC BY 3.0 clips) | per clip | sound events |
| NSynth | CC BY 4.0 | instruments, notes |
| FMA (CC BY, CC0, public-domain tracks) | per track | music |
| OpenMIC-2018 labels | CC BY 4.0 | instruments |
| CommonsenseQA | MIT | spoken questions |
| QASC, QuaRTz | CC BY 4.0 | spoken questions |
| JADE-Data | Apache-2.0 | text decisions, spoken questions |
| tasksource (commercial, non-share-alike tasks) | CC BY / MIT / Apache-2.0 / BSD / CC0 / ODC-BY | text decisions |
| CMUdict | BSD-2 | syllables, stress, rhyme |
| misaki | Apache-2.0 | pronunciation for TTS |

Synthetic speech: Kokoro-82M (Apache-2.0), Piper 1.2.0 with voices trained on CC BY or public-domain data (MIT),
Parler-TTS mini v1 (Apache-2.0), SpeechT5 with HiFi-GAN and CMU ARCTIC x-vectors (MIT), Chatterbox (MIT), Vocos
(MIT). Pitch and timing edits use WORLD / pyworld (modified BSD / MIT) and Rubber Band. Kokoro and Piper call the
espeak-ng phonemizer (GPL-3) as a tool; none of it is in the model.

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
