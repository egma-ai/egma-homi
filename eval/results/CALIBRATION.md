# Calibration

homi: softmax(log-probs / T), T = 1.0084 (its calibration temperature); the untouched base model at T = 1. API models: the probabilities they self-report with their answer, renormalised; a row without a non-negative number for every option is not covered (last column: such rows that still named an answer).
Confidence = probability of the argmax option, which is the answer scored. Accuracy, ECE (15 equal-width bins), Brier (sum over options) and NLL (p clipped at 1e-6) are on covered questions; selective = share of all questions at or above a confidence threshold / accuracy on them (%).

## MMSU (4981 questions)

| model | coverage | accuracy | ECE | Brier | NLL | share at conf ≥ 0.9 | accuracy at conf ≥ 0.9 | answered, no usable probabilities |
|---|---|---|---|---|---|---|---|---|
| homi | 4981/4981 | 83.1 | 0.076 | 0.259 | 0.539 | 73.2 | 91.7 | 0 |
| Qwen3-Omni untouched (base) | 4981/4981 | 72.2 | 0.182 | 0.439 | 1.033 | 72.5 | 84.4 | 0 |
| Gemini 3.8 Flash | 4980/4981 | 85.7 | 0.052 | 0.238 | 0.465 | 62.9 | 93.7 | 0 |
| Gemini 3.5 Flash-Lite | 4979/4981 | 67.3 | 0.133 | 0.512 | 1.146 | 13.8 | 72.1 | 2 |
| OpenAI gpt-audio | 4324/4981 | 69.5 | 0.081 | 0.459 | 0.894 | 10.8 | 84.8 | 0 |

| model | share / accuracy at conf ≥ 0.5 | share / accuracy at conf ≥ 0.7 | share / accuracy at conf ≥ 0.9 | share / accuracy at conf ≥ 0.95 |
|---|---|---|---|---|
| homi | 96.7 / 84.4 | 86.5 / 88.0 | 73.2 / 91.7 | 66.5 / 93.5 |
| Qwen3-Omni untouched (base) | 96.5 / 73.7 | 85.4 / 78.4 | 72.5 / 84.4 | 66.8 / 87.4 |
| Gemini 3.8 Flash | 100.0 / 85.7 | 98.3 / 86.3 | 62.9 / 93.7 | 52.7 / 95.4 |
| Gemini 3.5 Flash-Lite | 97.6 / 68.0 | 85.7 / 71.6 | 13.8 / 72.1 | 10.8 / 71.1 |
| OpenAI gpt-audio | 86.3 / 69.7 | 74.8 / 73.6 | 10.8 / 84.8 | 4.5 / 89.7 |

### homi reliability (T = 1.0084; bins with ≥ 20 questions)

| confidence bin | n | mean confidence | accuracy |
|---|---|---|---|
| 0.333–0.400 | 41 | 37.2 | 46.3 |
| 0.400–0.467 | 61 | 43.3 | 37.7 |
| 0.467–0.533 | 136 | 50.4 | 44.1 |
| 0.533–0.600 | 178 | 56.6 | 52.2 |
| 0.600–0.667 | 167 | 63.1 | 58.7 |
| 0.667–0.733 | 179 | 70.1 | 59.8 |
| 0.733–0.800 | 193 | 76.8 | 62.7 |
| 0.800–0.867 | 232 | 83.5 | 69.0 |
| 0.867–0.933 | 338 | 90.3 | 75.7 |
| 0.933–1.000 | 3451 | 99.0 | 92.7 |

## VocalGrad (7200 questions)

| model | coverage | accuracy | ECE | Brier | NLL | share at conf ≥ 0.9 | accuracy at conf ≥ 0.9 | answered, no usable probabilities |
|---|---|---|---|---|---|---|---|---|
| homi | 7200/7200 | 73.9 | 0.157 | 0.399 | 0.669 | 67.4 | 82.8 | 0 |
| Qwen3-Omni untouched (base) | 7200/7200 | 50.3 | 0.337 | 0.740 | 1.101 | 37.8 | 52.5 | 0 |
| Gemini 3.8 Flash | 7200/7200 | 63.0 | 0.237 | 0.570 | 0.842 | 37.3 | 70.2 | 0 |
| Gemini 3.5 Flash-Lite | 7200/7200 | 50.9 | 0.229 | 0.622 | 0.923 | 15.0 | 55.0 | 0 |
| OpenAI gpt-audio | 4746/7200 | 51.2 | 0.269 | 0.657 | 0.909 | 8.0 | 50.1 | 1 |

| model | share / accuracy at conf ≥ 0.5 | share / accuracy at conf ≥ 0.7 | share / accuracy at conf ≥ 0.9 | share / accuracy at conf ≥ 0.95 |
|---|---|---|---|---|
| homi | 100.0 / 73.9 | 87.6 / 77.0 | 67.4 / 82.8 | 55.7 / 86.3 |
| Qwen3-Omni untouched (base) | 100.0 / 50.3 | 87.2 / 51.2 | 37.8 / 52.5 | 11.6 / 53.3 |
| Gemini 3.8 Flash | 100.0 / 63.0 | 98.6 / 63.1 | 37.3 / 70.2 | 23.0 / 75.8 |
| Gemini 3.5 Flash-Lite | 100.0 / 50.9 | 65.8 / 51.4 | 15.0 / 55.0 | 11.4 / 56.0 |
| OpenAI gpt-audio | 65.9 / 51.2 | 63.6 / 51.0 | 8.0 / 50.1 | 0.3 / 45.5 |

### homi reliability (T = 1.0084; bins with ≥ 20 questions)

| confidence bin | n | mean confidence | accuracy |
|---|---|---|---|
| 0.467–0.533 | 150 | 51.6 | 50.0 |
| 0.533–0.600 | 306 | 56.7 | 51.3 |
| 0.600–0.667 | 274 | 63.4 | 53.3 |
| 0.667–0.733 | 348 | 70.1 | 51.7 |
| 0.733–0.800 | 420 | 76.8 | 54.8 |
| 0.800–0.867 | 515 | 83.5 | 59.2 |
| 0.867–0.933 | 829 | 90.4 | 65.0 |
| 0.933–1.000 | 4358 | 98.3 | 84.7 |
