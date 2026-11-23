# Goal 2: Baseline evaluation of an ALM on WorldSpeech (English)

## 1. Setup
- Dataset / subset / split: TODO
- Filters (duration, quality): TODO
- Sample size and seed: TODO (IDs in `data/sample_ids.json`)
- Model(s) and exact prompt: TODO
- Text normalizer and metric: TODO (jiwer WER)
- Hardware: TODO

## 2. Results

| Run | Model | Samples | WER | CER |
|---|---|---|---|---|
| baseline_en_whisper | TODO | TODO | TODO | TODO |
| baseline_en_qwen2audio | TODO | TODO | TODO | TODO |

WER by duration and by quality: TODO (output of `src/error_analysis.py`)

## 3. Error examples
5 examples (reference vs. prediction). Mark names of people and places, numbers, and rare words.

## 4. Discussion
- Which kinds of errors dominate?
- Effect of normalization and prompt
- Limitations (sample size, dataset overlap with model training data)

## 5. Next steps
Step 3: failure cases on named entities.
