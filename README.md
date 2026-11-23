# Audio-03: NER for ASR with Audio Language Models
 
Portfolio exam project (Multimodal AI, Hochschule Hof, Winter Term 2026/27).
We evaluate how well an Audio Language Model (ALM) transcribes speech, with a focus on named entities (people, places), and later improve it by finetuning.
 
**Current scope: steps 1 and 2** (dataset review and baseline evaluation on WorldSpeech, English).
 
## Project steps
 
| # | Step | Status |
|---|---|---|
| 1 | Review multilingual ASR/TTS datasets (MLS, VoxPopuli, WorldSpeech) | in progress |
| 2 | Evaluate an ALM on the chosen dataset (EN) | in progress |
| 3 | Identify failure cases (names of people, places, ...) | todo |
| 4 | Named Entity Recognition on the text data (spaCy) | todo |
| 5 | Evaluate the ALM preconditioned on the extracted entities | todo |
| 6 | Finetune the model with pre-conditioning | todo |
| 7 | Compare pre/post training, with and without named entities | todo |
| 8 | Measure degradation on a second dataset | todo |
| 9 | Personal learning experience write-ups (4x) | todo |
 
## Team
 
| Person | Role | Main files |
|---|---|---|
| TODO name | Data and repository | `src/dataset.py`, `src/inspect_dataset.py`, `requirements.txt` |
| TODO name | Model and evaluation | `src/model.py`, `src/metrics.py`, `src/evaluate.py`, `src/error_analysis.py` |
| TODO name | Documentation | `README.md`, `docs/dataset_review.md`, `docs/evaluation_report.md` |
 
## Repository structure
 
```
Audio-03/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/        run settings, one YAML per run
├── src/            code (data loading, models, metrics, evaluation)
├── data/           frozen sample IDs (no audio in git)
├── results/        predictions and metrics per run
├── notebooks/      exploration
└── docs/           dataset review, evaluation report, learning diaries
```
 
## Installation
 
```bash
git clone https://gitlab.hof-university.de/nochoamorales/Audio-03.git
cd Audio-03
pip install -r requirements.txt
huggingface-cli login      # your token stays out of git
```
 
## Usage
 
Run the steps in this order:
 
```bash
# 1. Check the real column names of the dataset and fix configs/baseline_en_whisper.yaml if needed
python src/inspect_dataset.py
 
# 2. Quick end-to-end test on 20 samples
python src/evaluate.py --limit 20
 
# 3. Full baseline run (Whisper)
python src/evaluate.py --config configs/baseline_en_whisper.yaml
 
# 4. Inspect the errors (WER by duration/quality, 30 worst samples)
python src/error_analysis.py results/step2_baseline/baseline_en_whisper
```
 
### Running the ALM (Qwen2-Audio)
 
Copy the config and change three values, keep everything else the same so the comparison is fair:
 
```yaml
run_name: baseline_en_qwen2audio
model:
  type: "qwen2audio"
  name: "Qwen/Qwen2-Audio-7B-Instruct"
```
 
Then run `python src/evaluate.py --config configs/baseline_en_qwen2audio.yaml`.
The Qwen wrapper depends on the installed `transformers` version, so test it with `--limit 5` first.
 
## Configuration
 
All settings live in `configs/*.yaml`: dataset and subset, column names, quality and duration filters, sample size, seed, model, prompt and text normalizer.
Each run saves a copy of its config next to its results, so every number can be reproduced.
 
## Results
 
Each run writes to `results/step2_baseline/<run_name>/`:
 
- `metrics_summary.json`: WER, CER, substitutions, deletions, insertions
- `predictions.csv`: reference, prediction and WER for every sample
- `config_used.yaml`: the exact settings of the run
- `worst_30.csv`: created by `error_analysis.py`
| Run | Model | Samples | WER |
|---|---|---|---|
| baseline_en_whisper | TODO | TODO | TODO |
| baseline_en_qwen2audio | TODO | TODO | TODO |
 
## Team workflow
 
- One branch per person (`feature/data`, `feature/model`, `feature/docs`), merge into `main` with a merge request.
- Never commit audio files, model weights or tokens (see `.gitignore`).
- The test set is frozen: the same seed and filters are used in steps 2, 5, 7 and 8.
- Use one text normalizer for all runs and state it in the report.
## Troubleshooting
 
- **Cannot clone on JupyterHub** (`Failed to connect ... port 443`): the pod cannot reach the university GitLab. Clone on your computer (VPN or campus network) and upload a zip of the repo to JupyterHub.
- **Column not found**: run `python src/inspect_dataset.py` and update the `columns` section of the config.
- **Audio decoding error**: the `datasets` version changed its audio format. Install `torchcodec` or use an older `datasets` version.
## Roadmap
 
Steps 3 to 9 from the table above: failure analysis on named entities, spaCy NER, entity-conditioned evaluation, finetuning, comparison, second dataset, learning diaries.
 
## Authors
 
TODO: names of the three team members.
 
## Project status
 
Active, working towards the first hand-in (steps 1 and 2).
 # Prove_project_Audio03
