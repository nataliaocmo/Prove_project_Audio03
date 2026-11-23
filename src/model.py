"""Qwen3-ASR inference and named-entity context helpers."""

from typing import Any, Optional, Tuple

import numpy as np
import torch
from transformers import AutoProcessor

MODEL_NAME = "Qwen/Qwen3-ASR-1.7B-hf"


def load_asr_model(
    model_name: str = MODEL_NAME, device: str = "cuda"
) -> Tuple[Any, Any]:
    """Load the Qwen3-ASR processor and model, supporting CPU-only inference."""
    try:
        from transformers import AutoModelForMultimodalLM
    except ImportError as exc:
        raise RuntimeError(
            "Qwen3-ASR requires a Transformers version that provides "
            "AutoModelForMultimodalLM; the workshop uses transformers>=5.13.0."
        ) from exc

    processor = AutoProcessor.from_pretrained(model_name)
    if device.lower() == "cpu":
        model = AutoModelForMultimodalLM.from_pretrained(model_name)
        model = model.to("cpu")
    else:
        model = AutoModelForMultimodalLM.from_pretrained(
            model_name, device_map="auto"
        )
    model.eval()
    return processor, model


def transcribe_sample(
    audio_array: np.ndarray,
    sampling_rate: int,
    processor: Any,
    model: Any,
    language: str = "en",
    prompt: Optional[str] = None,
    max_new_tokens: int = 256,
) -> str:
    """Transcribe one waveform using the Qwen3-ASR request pattern."""
    audio = np.asarray(audio_array, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)

    target_sr = getattr(
        getattr(processor, "feature_extractor", None), "sampling_rate", 16000
    )
    if int(sampling_rate) != target_sr:
        import librosa

        audio = librosa.resample(
            audio, orig_sr=int(sampling_rate), target_sr=target_sr
        )

    kwargs = {"audio": audio}
    if prompt:
        kwargs["prompt"] = prompt
    if language:
        kwargs["language"] = language

    inputs = processor.apply_transcription_request(**kwargs).to(
        model.device, model.dtype
    )
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    return processor.decode(
        generated_ids, return_format="transcription_only"
    )[0].strip()


def load_spacy_model() -> Any:
    """Load the English transformer NER model, falling back to the small model."""
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError(
            "Entity-conditioned evaluation requires spaCy and an English model "
            "(en_core_web_trf or en_core_web_sm)."
        ) from exc

    try:
        return spacy.load("en_core_web_trf")
    except OSError:
        try:
            return spacy.load("en_core_web_sm")
        except OSError as exc:
            raise RuntimeError(
                "Install en_core_web_trf or en_core_web_sm to run entity-conditioned "
                "evaluation."
            ) from exc


def extract_entities(text: str, nlp: Any) -> Optional[str]:
    """Return unique named entities in source order, or None when absent."""
    seen = set()
    entities = []
    for entity in nlp(text).ents:
        if entity.text not in seen:
            entities.append(entity.text)
            seen.add(entity.text)
    return ", ".join(entities) if entities else None