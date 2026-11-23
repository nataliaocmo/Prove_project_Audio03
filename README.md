# Audio-03: WorldSpeech English ASR evaluation

This repository evaluates `Qwen/Qwen3-ASR-1.7B-hf` on the `en_us` subset of
`disco-eth/WorldSpeech`, using the official test split. The baseline compares
transcriptions with and without named-entity context from spaCy.

## Repository layout

```text
Audio-03/
├── configs/eval_config.yaml  # dataset, filters, model, and output paths
├── data/                     # frozen sample IDs; no audio is stored here
├── docs/                     # dataset review and evaluation report
├── notebooks/                # exploratory notebooks
├── results/                  # generated metrics and predictions
├── scripts/scan_metadata.py  # metadata-only WorldSpeech scanner
└── src/
    ├── dataset.py            # streaming load, filters, and sample manifest
    ├── model.py              # Qwen3-ASR and spaCy helpers
    ├── metrics.py            # Lab 6 text normalization, WER, and CER
    └── evaluate.py           # evaluation CLI and result export
```

## Setup

Use Python 3.10+ and install the project dependencies:

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

The evaluator falls back to `en_core_web_sm` if the transformer model is
unavailable; install it with `python -m spacy download en_core_web_sm` if needed.
For gated Hugging Face access, authenticate without placing a token in the
repository.

## Run the evaluation

From the repository root, run a small test without replacing the official
sample-ID manifest:

```bash
python src/evaluate.py --config configs/eval_config.yaml --max-samples 10
```

Run the configured 500-sample evaluation:

```bash
python src/evaluate.py --config configs/eval_config.yaml
```

The first full run streams `disco-eth/WorldSpeech` (`en_us`, `test`), applies
the configured transcript, DNSMOS, and duration filters, then saves the chosen
`segment_id` values in `data/sample_ids.json`. Later runs reuse those IDs.
Audio is decoded into memory as mono 16 kHz arrays; audio files are not saved.
The `--max-samples` option uses a temporary subset manifest and does not alter
`data/sample_ids.json`.

The evaluator uses the Lab 6 normalizer (lowercase, remove punctuation, and
collapse whitespace) with `jiwer` WER/CER. Results are written to:

```text
results/step2_baseline/baseline_en_qwen3_asr/predictions.csv
results/step2_baseline/baseline_en_qwen3_asr/metrics_summary.json
```

Output directories are created automatically. The CSV contains one row per
sample with the reference, entity context, predictions, WER, and CER for both
conditions.

## Dataset metadata scan

To scan WorldSpeech metadata without materializing audio:

```bash
python scripts/scan_metadata.py --split test --shards 3
```

Omit `--shards 3` to scan all shards for the requested split. Metadata caches
are written under `data/metadata/` and are excluded from Git.
