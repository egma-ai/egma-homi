# VocalGrad: one harness, every model

7200 questions, 3 tasks; chance level 50.0%. Same audio, question and options (published order) for every model; an unanswered or refused question counts as wrong. ± = 95% interval.

| model | accuracy | ± | answered | accuracy on answered | refused / no answer | cost per 10k questions |
|---|---|---|---|---|---|---|
| egma-homi | 73.9 | 1.0 | 7200/7200 | 73.9 | 0 | $0.12 |
| Qwen3-Omni untouched (base) | 50.3 | 1.2 | 7200/7200 | 50.3 | 0 | $0.11 |
| Gemini 3.8 Flash | 63.0 | 1.1 | 7200/7200 | 63.0 | 0 | $12.10 |
| Gemini 3.5 Flash-Lite | 50.9 | 1.2 | 7200/7200 | 50.9 | 0 | $1.30 |
| OpenAI gpt-audio | 33.8 | 1.1 | 4747/7200 | 51.2 | 2453 | $37.03 |

## Per task (accuracy %, n questions)

| task | n | egma-homi | Qwen3-Omni untouched (base) | Gemini 3.8 Flash | Gemini 3.5 Flash-Lite | OpenAI gpt-audio |
|---|---|---|---|---|---|---|
| speaking_speed | 2400 | 74 | 50 | 78 | 51 | 32 |
| voice_pitch | 2400 | 61 | 50 | 56 | 51 | 35 |
| volume | 2400 | 87 | 50 | 55 | 50 | 34 |

## Refusals / no answer by task

| task | OpenAI gpt-audio |
|---|---|
| speaking_speed | 890 |
| voice_pitch | 778 |
| volume | 785 |

Ours = measured GPU seconds on one H200 at $4.54/h, batched. Gemini / OpenAI = their reported tokens at paid-tier list prices (Gemini 3.8 Flash at its promotional price until 2026-12-31).
