"""Text normalization and WER/CER calculation for ASR evaluation."""

import re
import string
from typing import Any, Dict, Iterable, List

import jiwer


def normalize_text(text: str) -> str:
    """Apply the Lab 6 lowercase, punctuation-removal and whitespace rules."""
    text = text.translate(str.maketrans("", "", string.punctuation)).lower()
    return re.sub(r"\s+", " ", text).strip()


def _word_counts(processed: Any) -> Dict[str, int]:
    substitutions = int(processed.substitutions)
    deletions = int(processed.deletions)
    insertions = int(processed.insertions)
    hits = int(processed.hits)
    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "hits": hits,
        "N": substitutions + deletions + hits,
    }


def compute_sample_metrics(reference: str, prediction: str) -> Dict[str, Any]:
    """Compute normalized sample-level WER, CER and word error counts."""
    normalized_reference = normalize_text(reference)
    normalized_prediction = normalize_text(prediction)
    words = jiwer.process_words(normalized_reference, normalized_prediction)
    characters = jiwer.process_characters(
        normalized_reference, normalized_prediction
    )
    return {
        "wer": float(words.wer),
        "cer": float(characters.cer),
        **_word_counts(words),
    }


def compute_corpus_metrics(
    references: Iterable[str], predictions: Iterable[str]
) -> Dict[str, Any]:
    """Compute corpus-level WER, CER and aggregate word error counts."""
    normalized_references = [normalize_text(text) for text in references]
    normalized_predictions = [normalize_text(text) for text in predictions]
    if len(normalized_references) != len(normalized_predictions):
        raise ValueError("references and predictions must have the same length")
    if not normalized_references:
        raise ValueError("cannot compute corpus metrics for an empty evaluation")

    words = jiwer.process_words(normalized_references, normalized_predictions)
    characters = jiwer.process_characters(
        normalized_references, normalized_predictions
    )
    return {
        "wer": float(words.wer),
        "cer": float(characters.cer),
        **_word_counts(words),
    }