# MMSU, all questions, one harness

4981 questions (MMSU, ICLR 2026, MIT licence; never used for training). Same audio, same question, same options in the published order for every model. Score = accuracy against the authors' answer key; an unanswered question counts as wrong.

## 1. Our runs (same setup for every model)

| model | answered | Overall | Perception avg | Reasoning avg | Voice perception (paralinguistic) | Voice reasoning (paralinguistic) | Words & sounds perception | Meaning reasoning | task average (47 tasks) |
|---|---|---|---|---|---|---|---|---|---|
| homi | 4981/4981 | 83.1 | 80.4 | 86.0 | 75.3 | 64.8 | 83.7 | 89.4 | 83.3 |
| Qwen3-Omni untouched (our base) | 4981/4981 | 72.2 | 60.5 | 84.5 | 49.4 | 49.6 | 67.8 | 90.1 | 72.5 |
| Gemini 3.8 Flash | 4980/4981 | 85.7 | 82.4 | 89.2 | 76.1 | 55.8 | 86.4 | 94.6 | 85.9 |
| Gemini 3.5 Flash-Lite | 4981/4981 | 67.4 | 55.1 | 80.3 | 48.6 | 46.0 | 59.3 | 85.9 | 67.6 |
| OpenAI gpt-audio | 4324/4981 | 60.3 | 46.5 | 74.9 | 34.1 | 41.5 | 54.7 | 80.3 | 60.7 |

## Accuracy vs cost (Pareto view)

| model | MMSU, all | cost per 10,000 questions |
|---|---|---|
| homi | 83.1 | $0.28 |
| Qwen3-Omni untouched (our base) | 72.2 | $0.19 |
| Gemini 3.8 Flash | 85.7 | $14.80 |
| Gemini 3.5 Flash-Lite | 67.4 | $2.07 |
| OpenAI gpt-audio | 60.3 | $41.69 |

Ours = measured GPU seconds on one H200 at $4.54/h with the GPU kept busy (batch); Gemini = its reported tokens at Google's paid-tier prices (3.8 Flash at its promotional price until 2026-12-31; it doubles after).

Questions per group: Overall 4981, Perception avg 2561, Reasoning avg 2420, Voice perception (paralinguistic) 1010, Voice reasoning (paralinguistic) 335, Words & sounds perception 1551, Meaning reasoning 2085

## 2. Published results (full MMSU; each from its authors' own setup)

| model | Overall | Perception avg | Reasoning avg | Voice perception | Voice reasoning | source |
|---|---|---|---|---|---|---|
| Gemini 2.5 Pro | 77.7 | – | – | – | – | Qwen3-Omni model card |
| Gemini 2.5 Flash | 70.2 | – | – | – | – | Qwen3-Omni model card |
| Qwen3-Omni-30B-A3B (our base) | 69.0 | – | – | – | – | Qwen3-Omni model card |
| Gemini 1.5 Pro | 60.7 | 46.1 | 76.2 | 31.2 | 46.3 | MMSU paper leaderboard |
| Qwen2.5-Omni-7B | 60.6 | 42.5 | 79.8 | 39.4 | 48.4 | MMSU paper leaderboard |
| Kimi-Audio | 59.3 | 43.5 | 76.0 | 35.7 | 55.2 | MMSU paper leaderboard |
| GPT-4o Audio | 56.4 | 39.7 | 72.0 | 21.4 | 26.2 | MMSU paper leaderboard |
| Gemini 2.0 Flash | 51.0 | 40.8 | 47.8 | 30.6 | 36.2 | MMSU paper leaderboard |

Published numbers come from different setups (they generate a text answer that is then parsed); compare them with ours through the untouched Qwen3-Omni row, which appears in both: published 69.0 vs our run above.

## 3. Per task (accuracy %)

| task | n | homi | Qwen3-Omni untouched (our base) | Gemini 3.8 Flash | Gemini 3.5 Flash-Lite | OpenAI gpt-audio |
|---|---|---|---|---|---|---|
| accent_identification (Perc/Lingui) | 114 | 70 | 49 | 77 | 61 | 50 |
| consonant_vowel_perception (Perc/Lingui) | 103 | 83 | 86 | 95 | 85 | 87 |
| content_grounding (Perc/Lingui) | 109 | 99 | 97 | 100 | 98 | 99 |
| dialogue_turn_counting (Perc/Lingui) | 99 | 91 | 14 | 100 | 36 | 52 |
| disfluency_detection (Perc/Lingui) | 87 | 79 | 61 | 85 | 38 | 63 |
| intonation_perception (Perc/Lingui) | 111 | 57 | 45 | 64 | 56 | 40 |
| near_homophone_perception (Perc/Lingui) | 99 | 70 | 67 | 75 | 54 | 22 |
| non_verbal_sound_detection (Perc/Lingui) | 117 | 91 | 97 | 91 | 73 | 85 |
| pause_perception (Perc/Lingui) | 107 | 100 | 52 | 84 | 50 | 31 |
| plosive_sound_identification (Perc/Lingui) | 108 | 51 | 52 | 56 | 52 | 16 |
| prolonged_sound_perception (Perc/Lingui) | 83 | 96 | 95 | 96 | 80 | 73 |
| speech_act_classification (Perc/Lingui) | 98 | 97 | 100 | 99 | 89 | 87 |
| speech_duration_estimation (Perc/Lingui) | 106 | 95 | 58 | 96 | 20 | 39 |
| speech_stress_perception (Perc/Lingui) | 110 | 96 | 75 | 91 | 47 | 35 |
| syllable_perception (Perc/Lingui) | 100 | 82 | 71 | 92 | 50 | 47 |
| age_prediction (Perc/Parali) | 104 | 82 | 64 | 71 | 71 | 35 |
| emotion_recognition (Perc/Parali) | 117 | 85 | 65 | 58 | 57 | 34 |
| gender_prediction (Perc/Parali) | 120 | 98 | 98 | 95 | 93 | 52 |
| pitch_comparison (Perc/Parali) | 108 | 69 | 25 | 81 | 26 | 19 |
| speaker_identity_recognition (Perc/Parali) | 114 | 72 | 39 | 75 | 33 | 27 |
| speed_comparison (Perc/Parali) | 109 | 72 | 24 | 77 | 22 | 28 |
| total_speaker_counting (Perc/Parali) | 120 | 19 | 49 | 92 | 55 | 44 |
| vocal_range_comparison (Perc/Parali) | 108 | 91 | 54 | 80 | 45 | 31 |
| volume_comparison (Perc/Parali) | 110 | 95 | 22 | 53 | 30 | 35 |
| background_scene_recognition (Reas/Lingui) | 110 | 66 | 65 | 66 | 45 | 46 |
| casual_reasoning (Reas/Lingui) | 116 | 99 | 100 | 99 | 99 | 99 |
| code_switch_question_answering (Reas/Lingui) | 111 | 93 | 97 | 98 | 97 | 85 |
| continuation_writing (Reas/Lingui) | 110 | 99 | 100 | 100 | 98 | 100 |
| couplet_matching (Reas/Lingui) | 107 | 52 | 79 | 93 | 78 | 83 |
| deixis_resolution (Reas/Lingui) | 103 | 90 | 91 | 97 | 85 | 75 |
| dialogue_reasoning (Reas/Lingui) | 114 | 95 | 94 | 99 | 89 | 96 |
| homophone_based_reasoning (Reas/Lingui) | 101 | 90 | 97 | 100 | 89 | 75 |
| idiom_reasoning (Reas/Lingui) | 102 | 99 | 99 | 98 | 98 | 89 |
| intent_detection (Reas/Lingui) | 115 | 92 | 90 | 92 | 83 | 72 |
| intonation_based_reasoning (Reas/Lingui) | 78 | 86 | 88 | 90 | 81 | 78 |
| logical_reasoning (Reas/Lingui) | 104 | 86 | 86 | 89 | 81 | 53 |
| long_speech_summarization (Reas/Lingui) | 106 | 99 | 100 | 100 | 99 | 99 |
| pause_based_reasoning (Reas/Lingui) | 90 | 61 | 64 | 76 | 59 | 39 |
| polysemy_reasoning (Reas/Lingui) | 98 | 100 | 99 | 99 | 95 | 88 |
| prolonged_sound_reasoning (Reas/Lingui) | 84 | 98 | 98 | 98 | 100 | 95 |
| puns_interpretation (Reas/Lingui) | 109 | 100 | 98 | 100 | 99 | 94 |
| speech_translation (Reas/Lingui) | 119 | 98 | 98 | 100 | 94 | 98 |
| stress_based_reasoning (Reas/Lingui) | 110 | 92 | 67 | 94 | 53 | 44 |
| syntactic_structure_matching (Reas/Lingui) | 98 | 88 | 89 | 100 | 95 | 92 |
| emotional_context_reasoning (Reas/Parali) | 110 | 65 | 36 | 44 | 28 | 27 |
| sarcasm_detection (Reas/Parali) | 116 | 67 | 66 | 78 | 62 | 60 |
| synthetic_speech_detection (Reas/Parali) | 109 | 62 | 45 | 45 | 47 | 36 |
