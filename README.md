# homi - an open decision model for audio

A Pareto-frontier open decision model that answers typed questions about a given audio clip (inspired by [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)). Unlike
Jev, [Clef](https://blog.cloudflare.com/clef-decision-models/) and [pplx-decider](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b), which only take in text, homi takes an audio clip and a question, and returns a
probability for every option. It beats Gemini Flash on [VocalGrad](https://github.com/ynklab/vocalgrad) and is comparable to Gemini
Flash on [MMSU](https://github.com/dingdongwang/MMSU). The model can be used
for monitoring and evaluating large-scale audio at low cost.

Weights: [`egma-ai/homi`](https://huggingface.co/egma-ai/homi).


## Problem

A lot of paralinguistic and phonological questions about audio cannot be answered by looking at a transcript. This
model takes in audio natively and answers those questions, e.g. "How does the speaker feel?" with the options
angry / happy / sad / fear / disgust.


## Model

It is [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) with frozen weights, a
decision head in place of the language-model head, 17M parameters of [LoRA](https://arxiv.org/abs/2106.09685), and one calibration temperature. It does
not generate text.

Weights: [`egma-ai/homi`](https://huggingface.co/egma-ai/homi). This repository has the code to run
it, evaluate it, rebuild its training data and train it.

Just like Jev, there are three question types:
- `noul`: yes/no, returns P(yes)
- `choice`: 2 to 255 options, returns a probability per option
- `score`: 2 to 10 ordered levels (e.g. slow / normal / fast), returns a probability per level

## Results

Same audio, question and options for every model. Unanswered counts as wrong. Our cost is GPU time on one H200
($4.54/h, batched); API costs are reported tokens at list prices (Gemini 3.8 Flash at its promotional price).

**[VocalGrad](https://github.com/ynklab/vocalgrad)** (7,200 questions: does pitch, speed or volume rise or fall within a clip? Chance is 50%.)

| Model | VocalGrad | Speed | Pitch | Volume | $ per 10k questions |
|---|---|---|---|---|---|
| homi | 73.9 | 74 | 61 | 87 | 0.12 |
| Gemini 3.8 Flash | 63.0 | 78 | 56 | 55 | 12.10 |
| Qwen3-Omni-30B-A3B | 50.3 | 50 | 50 | 50 | 0.11 |
| Gemini 3.5 Flash-Lite | 50.9 | 51 | 51 | 50 | 1.30 |
| OpenAI gpt-audio | 33.8 | 32 | 35 | 34 | 37.03 |

**[MMSU](https://github.com/dingdongwang/MMSU)** (4,981 questions, 47 tasks)

| Model | MMSU | Perception | Reasoning | $ per 10k questions |
|---|---|---|---|---|
| Gemini 3.8 Flash | 85.7 | 82.4 | 89.2 | 14.80 |
| homi | 83.1 | 80.4 | 86.0 | 0.28 |
| Qwen3-Omni-30B-A3B | 72.2 | 60.5 | 84.5 | 0.19 |
| Gemini 3.5 Flash-Lite | 67.4 | 55.1 | 80.3 | 2.07 |
| OpenAI gpt-audio | 60.3 | 46.5 | 74.9 | 41.69 |



gpt-audio did not answer 13% of MMSU and 34% of VocalGrad (it declined or asked us to "play the audio"); on the
questions it did answer it scores 69.5 and 51.2. Raw answers and per-task tables: [`eval/results/`](eval/results/).

**Calibration** (does confidence match accuracy?). "Sure" means confidence of at least 0.9.

| Model | VocalGrad ECE | VocalGrad Brier | VocalGrad: accuracy when sure (share sure) | MMSU ECE | MMSU Brier | MMSU: accuracy when sure (share sure) |
|---|---|---|---|---|---|---|
| homi | 0.157 | 0.399 | 82.8% (67%) | 0.076 | 0.259 | 91.7% (73%) |
| Gemini 3.8 Flash | 0.237 | 0.570 | 70.2% (37%) | 0.052 | 0.238 | 93.7% (63%) |
| Qwen3-Omni-30B-A3B | 0.337 | 0.740 | 52.5% (38%) | 0.182 | 0.439 | 84.4% (73%) |
| Gemini 3.5 Flash-Lite | 0.229 | 0.622 | 55.0% (15%) | 0.133 | 0.512 | 72.1% (14%) |
| OpenAI gpt-audio | 0.269 | 0.657 | 50.1% (8%) | 0.081 | 0.459 | 84.8% (11%) |

Gemini and OpenAI probabilities are the ones the models report in their answers. gpt-audio is scored only on the
questions it answered. On VocalGrad, homi is the only model with a
Brier score better than guessing evenly, but it is overconfident: its most confident answers (98% on average) are
right 85% of the time. Full tables:
[`eval/results/CALIBRATION.md`](eval/results/CALIBRATION.md).

No MMSU audio or text is in the training data, and neither are the datasets MMSU builds its audio from.

Weak spots: counting distinct speakers (it usually answers one too few), long conversations with several people,
rhyming couplets, reasoning from pauses.

## Layout

| Folder | Contents |
|---|---|
| `inference/` | `predict.py` (`AudioDecider` and a CLI), the decision head, prompt and answer-code helpers |
| `eval/` | benchmark preparation, runners for our model, Gemini and OpenAI, scoring scripts, `results/` |
| `training/` | data loading and mixture, training loop, temperature fit (Modal) |
| `data/` | one folder per training-data builder, and [`DATA.md`](data/DATA.md) with every source and licence |

Data building, training and evaluation run on [Modal](https://modal.com), using one Volume (`homi`, mounted at
`/vol`).

## Run it

You need a GPU with about 70 GB free (bf16).

```bash
pip install torch==2.14.1 transformers==5.18.0 peft==0.21.2 accelerate==1.15.0 safetensors==0.8.0 \
    soundfile==0.14.0 scipy==1.18.1 numpy==2.5.3 huggingface_hub

hf download egma-ai/homi --local-dir homi

python homi/predict.py clip.wav
```

```python
import sys; sys.path.insert(0, "homi")
from predict import AudioDecider

decider = AudioDecider("homi")
decider.ask("clip.wav", {
    "angry": {"type": "noul", "instructions": "Does the speaker sound angry?"},
    "emotion": {"type": "choice", "instructions": "Which emotion does the speaker express?",
                "criteria": {"anger": None, "happy": None, "neutral": None, "sad": None}},
    "pace": {"type": "score", "instructions": "How fast is the person speaking?",
             "criteria": ["slow", "normal pace", "fast"]},
})
```

All questions about a clip run in one batched forward pass (about 75 ms for three questions on a short clip on an
H200). Load the adapter with `predict.py`; it targets the Qwen3-Omni thinker, so `AutoPeftModel` will not find it.

## Evaluate

```bash
pip install modal && modal setup
modal volume put homi homi /models/homi       # the downloaded weights

modal run eval/modal_eval.py::mmsu --step prepare
modal run --detach eval/modal_eval.py::mmsu --step run --which homi --adapter-dir /vol/models/homi
modal run --detach eval/modal_eval.py::mmsu --step run --which qwen-stock,gemini-3.8-flash,gemini-3.5-flash-lite
modal volume get homi /bench_mmsu ./bench_mmsu
python eval/mmsu_report.py --dir ./bench_mmsu --out MMSU.md

modal run --detach eval/bench_prepare.py::main                # VocalGrad (also VoxParadox, MSU-Bench)
modal run --detach eval/modal_eval.py::bench --name vocalgrad --which homi --adapter-dir /vol/models/homi
python eval/bench_report.py --dir ./bench_vocalgrad --name vocalgrad --out VOCALGRAD.md
```

Gemini needs a Modal secret `gemini` (`GEMINI_API_KEY`), OpenAI a secret `openai` (`OPENAI_API_KEY`). The exact
prompts, revisions and scoring are in `eval/bench_lock.json` and `eval/modal_eval.py`. [VoxParadox](https://github.com/ihp-lab/VoxParadox) and [MSU-Bench](https://github.com/ASLP-lab/MSU-Bench) are
research-only licences; check them before publishing scores.

## Training data

Every training question has optional audio, an optional text context, a question, 2 to 40 options and a target
probability per option. The data is public and permissively licensed, or synthetic audio whose labels are exact by
construction. [`data/DATA.md`](data/DATA.md) lists every source and licence.

| Builder | Teaches | Main sources |
|---|---|---|
| `training/prepare_data.py` | emotion, spoken intent | [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D), [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) |
| `sb_voice` | emotion, pitch, rate, loudness, pauses, gender, age, pronunciation | CREMA-D, [LibriTTS-R](https://www.openslr.org/141/), [speechocean762](https://www.openslr.org/101/) |
| `sb_relative` | louder / faster / higher than the speaker's usual voice | LibriTTS-R, CREMA-D |
| `sb_count`, `sb_speakers` | counting turns and speakers, same speaker | LibriTTS-R, [LibriSpeech](https://www.openslr.org/12/), [HarperValleyBank](https://github.com/cricketclub/gridspace-stanford-harper-valley), [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) |
| `sb_compare` | pitch / speed / volume / range comparisons, durations, pauses | LibriTTS-R, [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) |
| `sb_phon` | syllables, stress, emphasis, minimal pairs, consonants and vowels | Kokoro-82M, [CMUdict](https://github.com/cmusphinx/cmudict) |
| `sb_intonation` | intonation contours, meaning of pauses | LibriTTS-R, Kokoro-82M, [WORLD](https://github.com/mmorise/World) |
| `sb_disfluency` | disfluency types, verse rhythm and rhyme | Kokoro-82M, CMUdict |
| `sb_synthetic` | real vs synthetic speech | LibriSpeech, [Apollo](https://huggingface.co/datasets/BrunoHays/apollo-telephony-en), HarperValleyBank, AMI; Kokoro, [Piper](https://github.com/rhasspy/piper), [Parler-TTS](https://huggingface.co/parler-tts/parler-tts-mini-v1), [SpeechT5](https://huggingface.co/microsoft/speecht5_tts), [Chatterbox](https://huggingface.co/ResembleAI/chatterbox), [Vocos](https://github.com/gemelo-ai/vocos) |
| `sb_sounds`, `sb_sounds2` | sound events, music genre, instruments | [FSD50K](https://zenodo.org/records/4060432), [NSynth](https://magenta.tensorflow.org/datasets/nsynth), [FMA](https://github.com/mdeff/fma), [OpenMIC-2018](https://github.com/cosmir/openmic-2018) |
| `sb_calls`, `sb_aicall` | who answered a call; AI agent with a person, voicemail or machine | Kokoro-82M, HarperValleyBank, [FTC robocalls](https://github.com/wspr-ncsu/robocall-audio-dataset) |
| `sb_dialog`, `sb_calltext` | talk-over, silences, turn order; call content | HarperValleyBank, AMI |
| `sb_phone`, `sb_phone_pitch` | phone line quality, noise, live vs recorded, language; pitch on phone audio | Apollo, MInDS-14, FTC robocalls, LibriTTS-R, CREMA-D, [MUSAN](https://www.openslr.org/17/) |
| `sb_spoken`, `sb_spoken2` | spoken questions, intents, dialog acts | HarperValleyBank, [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa), [QASC](https://huggingface.co/datasets/allenai/qasc), [QuaRTz](https://huggingface.co/datasets/allenai/quartz), [JADE-Data](https://huggingface.co/datasets/theunnecessarythings/JADE-Data) |
| `sb_tasksource`, JADE (`training/`) | text-only decisions | [tasksource](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions) (commercial, non-share-alike tasks), JADE-Data |
| `teacher` | soft labels for text questions | [perplexity-ai/pplx-decider-v1.1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) |

Each builder runs with its `modal_build.py` and writes `/vol/stageb/<builder>/`. Some builders reuse raw copies made by
others (e.g. `sb_aicall` reads `sb_dialog`'s HarperValleyBank copy), so build in table order. Rebuilt data follows the same
recipe, but the seeded samples will not be byte-identical to ours.

## Training

```bash
modal run training/modal_train.py::prepare                     # base items, Qwen3-Omni weights
modal run training/modal_train.py::jade                        # JADE-Data text items
# run the data builders and data/teacher
modal run training/modal_train.py::audit --run pass1           # counts; fails if an excluded item is present
modal run --detach training/modal_train.py::train --run pass1  # 8 x H200 or B200
modal run --detach training/modal_train.py::train --run pass2
modal run --detach training/modal_train.py::calibrate --run pass2
```

- LoRA rank 16 (alpha 32, dropout 0.05) on q/k/v/o of the language model's 48 layers and q/k/v of the audio
  encoder's 32 layers. All base weights frozen.
- Decision head initialised from the base model's output rows for the answer codes; read at the last prompt
  position; soft-target cross-entropy.
- AdamW, 64 questions per step, cosine schedule, 3% warmup, gradient clipping at 1.0, data parallel on 8 GPUs.
- Two passes: 5,654 steps at learning rate 1e-4 on the core builders, then 2,136 steps at 5e-5 on the newer builders
  plus a 20% replay of the core data. Mixtures and exclusions are in `training/recipe.py`.
- Text targets blend 50/50 with the teacher's probabilities where the teacher is right.
- One temperature, fitted on every builder's calibration split with sources weighted equally.

Outputs go to `/vol/runs/<run>/adapter/` and `/vol/runs/<run>/temperature_lora.json`. Put those files next to
`inference/` (temperature file renamed to `temperature.json`) to get the Hugging Face folder.

## Licence

Code: Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). Weights: Apache-2.0 on Hugging Face. Datasets keep their own
licences ([data/DATA.md](data/DATA.md)).

```bibtex
@misc{homi_2026,
  title        = {homi: typed questions about audio, with probabilities},
  author       = {{Outlier Humans Inc.}},
  year         = {2026},
  howpublished = {\url{https://github.com/egma-ai/homi}}
}
```

## References

**Inspiration**
- The name is a nod to [Homi J. Bhabha](https://en.wikipedia.org/wiki/Homi_J._Bhabha).
- Jev (TypeSafe AI): [launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- Clef (Cloudflare): [blog post](https://blog.cloudflare.com/clef-decision-models/), [weights](https://huggingface.co/Cloudflare/clef)
- pplx-decider (Perplexity): [weights](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b)

**Model and method**
- Qwen3-Omni-30B-A3B-Instruct: [weights](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct), [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765)
- Hu et al., 2021. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

**Benchmarks**
- MMSU (ICLR 2026): [paper](https://arxiv.org/abs/2506.04779), [code](https://github.com/dingdongwang/MMSU), [data](https://huggingface.co/datasets/ddwang2000/MMSU)
- VocalGrad: [code](https://github.com/ynklab/vocalgrad), [data](https://huggingface.co/datasets/ynklab/vocalgrad)
- VoxParadox: [paper](https://arxiv.org/abs/2605.27772), [code](https://github.com/ihp-lab/VoxParadox)
- MSU-Bench: [paper](https://arxiv.org/abs/2606.22868), [code](https://github.com/ASLP-lab/MSU-Bench)

**Compared models**
- Gemini 3.8 Flash, Gemini 3.5 Flash-Lite: [Gemini API models](https://ai.google.dev/gemini-api/docs/models)
- OpenAI gpt-audio: [model page](https://platform.openai.com/docs/models/gpt-audio)

**Training data** (licences in [data/DATA.md](data/DATA.md))
- Speech: [LibriTTS-R](https://www.openslr.org/141/), [LibriSpeech](https://www.openslr.org/12/), [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) (Cao et al., 2014), [speechocean762](https://www.openslr.org/101/), [HarperValleyBank](https://github.com/cricketclub/gridspace-stanford-harper-valley), [AMI](https://groups.inf.ed.ac.uk/ami/corpus/), [Apollo telephony](https://huggingface.co/datasets/BrunoHays/apollo-telephony-en), [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14), [FTC robocall audio](https://github.com/wspr-ncsu/robocall-audio-dataset)
- Sounds and music: [FSD50K](https://zenodo.org/records/4060432), [NSynth](https://magenta.tensorflow.org/datasets/nsynth), [FMA](https://github.com/mdeff/fma), [OpenMIC-2018](https://github.com/cosmir/openmic-2018), [MUSAN](https://www.openslr.org/17/), [OpenSLR 28 room responses](https://www.openslr.org/28/)
- Text: [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa), [QASC](https://huggingface.co/datasets/allenai/qasc), [QuaRTz](https://huggingface.co/datasets/allenai/quartz), [JADE-Data](https://huggingface.co/datasets/theunnecessarythings/JADE-Data), [tasksource](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions)
- Pronunciation: [CMUdict](https://github.com/cmusphinx/cmudict), [misaki](https://github.com/hexgrad/misaki)
- Synthetic speech and audio tools: [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), [Piper](https://github.com/rhasspy/piper), [Parler-TTS mini v1](https://huggingface.co/parler-tts/parler-tts-mini-v1), [SpeechT5](https://huggingface.co/microsoft/speecht5_tts) with [HiFi-GAN](https://huggingface.co/microsoft/speecht5_hifigan) and [CMU ARCTIC x-vectors](https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors), [Chatterbox](https://huggingface.co/ResembleAI/chatterbox), [Vocos](https://github.com/gemelo-ai/vocos), [WORLD](https://github.com/mmorise/World) / [pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder), [Rubber Band](https://breakfastquay.com/rubberband/)
