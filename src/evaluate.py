"""Run Qwen3-ASR with and without spaCy named-entity context."""

import argparse
import copy
import csv
import json
import logging
import os
import tempfile
from statistics import mean, median
from typing import Any, Dict, List

import yaml
from tqdm.auto import tqdm

if __package__:
    from .dataset import load_samples
    from .metrics import compute_corpus_metrics, compute_sample_metrics, normalize_text
    from .model import (
        extract_entities,
        load_asr_model,
        load_spacy_model,
        transcribe_sample,
    )
else:
    from dataset import load_samples
    from metrics import compute_corpus_metrics, compute_sample_metrics, normalize_text
    from model import (
        extract_entities,
        load_asr_model,
        load_spacy_model,
        transcribe_sample,
    )


logger = logging.getLogger(__name__)


CSV_COLUMNS = [
    "wav_path",
    "label",
    "entities",
    "pred_without_context",
    "wer_without_context",
    "cer_without_context",
    "pred_with_context",
    "wer_with_context",
    "cer_with_context",
]


def _positive_int(value: str) -> int:
    limit = int(value)
    if limit <= 0:
        raise argparse.ArgumentTypeError("--max-samples must be a positive integer")
    return limit


def _load_quick_subset(cfg: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    """Load a limited subset without creating or replacing the official manifest."""
    quick_cfg = copy.deepcopy(cfg)
    dataset_cfg = quick_cfg["dataset"]
    official_manifest = dataset_cfg["sample_ids_file"]
    dataset_cfg["max_samples"] = min(limit, dataset_cfg["max_samples"])

    with tempfile.TemporaryDirectory(prefix="audio03-sample-") as temp_dir:
        temporary_manifest = os.path.join(temp_dir, "sample_ids.json")
        if os.path.exists(official_manifest) and os.path.getsize(official_manifest):
            with open(official_manifest, encoding="utf-8") as fh:
                manifest = json.load(fh)
            frozen_ids = manifest if isinstance(manifest, list) else manifest.get("ids")
            if not isinstance(frozen_ids, list):
                raise ValueError(
                    f"Sample manifest at {official_manifest} must contain a list of IDs"
                )
            quick_manifest = (
                manifest[:limit]
                if isinstance(manifest, list)
                else {**manifest, "ids": frozen_ids[:limit]}
            )
            with open(temporary_manifest, "w", encoding="utf-8") as fh:
                json.dump(quick_manifest, fh, indent=2)
        dataset_cfg["sample_ids_file"] = temporary_manifest
        return load_samples(quick_cfg)


def _distribution(values: List[float], prefix: str) -> Dict[str, float]:
    return {
        f"mean_{prefix}": mean(values),
        f"median_{prefix}": median(values),
        f"min_{prefix}": min(values),
        f"max_{prefix}": max(values),
    }


def _context_summary(
    references: List[str], predictions: List[str], sample_metrics: List[Dict[str, Any]]
) -> Dict[str, Any]:
    corpus = compute_corpus_metrics(references, predictions)
    wers = [metrics["wer"] for metrics in sample_metrics]
    cers = [metrics["cer"] for metrics in sample_metrics]
    perfect_count = sum(wer == 0 for wer in wers)
    return {
        **_distribution(wers, "wer"),
        **_distribution(cers, "cer"),
        "perfect_wer_count": perfect_count,
        "perfect_wer_ratio": perfect_count / len(wers),
        "corpus_wer": corpus["wer"],
        "corpus_cer": corpus["cer"],
        "substitutions": corpus["substitutions"],
        "deletions": corpus["deletions"],
        "insertions": corpus["insertions"],
        "hits": corpus["hits"],
        "N": corpus["N"],
    }


def evaluate(cfg: Dict[str, Any], max_samples: int = None) -> Dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    samples = (
        _load_quick_subset(cfg, max_samples)
        if max_samples is not None
        else load_samples(cfg)
    )
    if not samples:
        raise ValueError("The configured dataset selection contains no samples")

    processor, model = load_asr_model(
        model_name=cfg["model"]["name"],
        device=cfg["model"].get("device", "cuda"),
    )
    nlp = load_spacy_model()
    language = cfg["model"].get("language", "en")
    max_new_tokens = cfg["model"].get("max_new_tokens", 256)
    rows = []
    references = []
    without_context_predictions = []
    with_context_predictions = []
    without_context_metrics = []
    with_context_metrics = []

    for sample in tqdm(samples, desc="Evaluating ASR", unit="samples"):
        reference = sample["reference"]
        pred_without_context = transcribe_sample(
            sample["audio"],
            sample["sampling_rate"],
            processor,
            model,
            language=language,
            prompt=None,
            max_new_tokens=max_new_tokens,
        )
        without_metrics = compute_sample_metrics(reference, pred_without_context)

        entities = extract_entities(reference, nlp)
        pred_with_context = transcribe_sample(
            sample["audio"],
            sample["sampling_rate"],
            processor,
            model,
            language=language,
            prompt=entities,
            max_new_tokens=max_new_tokens,
        )
        with_metrics = compute_sample_metrics(reference, pred_with_context)

        references.append(reference)
        without_context_predictions.append(pred_without_context)
        with_context_predictions.append(pred_with_context)
        without_context_metrics.append(without_metrics)
        with_context_metrics.append(with_metrics)
        rows.append(
            {
                "wav_path": sample["id"],
                "label": normalize_text(reference),
                "entities": entities or "",
                "pred_without_context": normalize_text(pred_without_context),
                "wer_without_context": without_metrics["wer"],
                "cer_without_context": without_metrics["cer"],
                "pred_with_context": normalize_text(pred_with_context),
                "wer_with_context": with_metrics["wer"],
                "cer_with_context": with_metrics["cer"],
            }
        )

    output_cfg = cfg["output"]
    predictions_path = output_cfg["predictions_csv"]
    summary_path = output_cfg["summary_json"]
    output_dir = output_cfg.get("output_dir")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(predictions_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)

    with open(predictions_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved predictions CSV to %s", predictions_path)

    summary = {
        "model": cfg["model"]["name"],
        "dataset": {
            "hf_repo": cfg["dataset"]["hf_repo"],
            "subset": cfg["dataset"]["subset"],
            "split": cfg["dataset"]["split"],
        },
        "samples": len(samples),
        "without_context": _context_summary(
            references, without_context_predictions, without_context_metrics
        ),
        "with_context": _context_summary(
            references, with_context_predictions, with_context_metrics
        ),
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Saved metrics summary to %s", summary_path)
    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/eval_config.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "--max-samples",
        type=_positive_int,
        help="Evaluate a temporary subset without modifying the saved sample IDs",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration file {args.config} must contain a YAML mapping")

    summary = evaluate(cfg, max_samples=args.max_samples)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()