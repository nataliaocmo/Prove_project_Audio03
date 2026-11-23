"""Load, filter and sample WorldSpeech for the evaluation."""
import json
import logging
import math
import os
from typing import Any, Dict, List

import numpy as np
import torch
import torchaudio
from datasets import load_dataset

logger = logging.getLogger(__name__)
TARGET_SAMPLE_RATE = 16000


def _get_audio(audio) -> tuple:
    """Return a float32 mono array resampled to 16 kHz for both datasets audio APIs."""
    if isinstance(audio, dict) and "array" in audio:
        data = np.asarray(audio["array"], dtype=np.float32)
        sample_rate = int(audio["sampling_rate"])
    else:
        samples = audio.get_all_samples()
        data = samples.data.numpy().astype(np.float32)
        sample_rate = int(samples.sample_rate)

    if data.ndim > 1:
        channel_axis = 0 if data.shape[0] <= data.shape[1] else 1
        data = data.mean(axis=channel_axis)
    if data.ndim != 1 or sample_rate <= 0:
        raise ValueError("Audio must have a valid sample rate and mono-convertible array")

    waveform = torch.from_numpy(np.ascontiguousarray(data))
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
        )
    return waveform.numpy().astype(np.float32), TARGET_SAMPLE_RATE


def _read_sample_ids(path: str) -> List[str]:
    """Return frozen sample IDs, or an empty list when no non-empty manifest exists."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []

    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    ids = manifest if isinstance(manifest, list) else manifest.get("ids")
    if not isinstance(ids, list):
        raise ValueError(f"Sample manifest at {path} must contain a list of IDs")
    if not ids:
        return []

    normalized_ids = [str(sample_id) for sample_id in ids]
    if any(not sample_id for sample_id in normalized_ids):
        raise ValueError(f"Sample manifest at {path} contains an empty ID")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError(f"Sample manifest at {path} contains duplicate IDs")
    return normalized_ids


def load_samples(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    d, f, c = cfg["dataset"], cfg["filtering"], cfg["columns"]
    streaming = d.get("streaming", True)
    manifest_path = d["sample_ids_file"]
    frozen_ids = _read_sample_ids(manifest_path)
    remaining_ids = set(frozen_ids)

    logger.info("Loading %s (%s, split=%s)", d["hf_repo"], d["subset"], d["split"])
    ds = load_dataset(d["hf_repo"], d["subset"], split=d["split"], streaming=streaming)
    if streaming:
        ds = ds.shuffle(seed=d.get("seed", 42), buffer_size=d.get("shuffle_buffer", 10000))
    else:
        ds = ds.shuffle(seed=d.get("seed", 42))

    samples_by_id: Dict[str, Dict[str, Any]] = {}
    min_quality = f.get("min_quality")
    for row in ds:
        sample_id_value = row.get(c["id"])
        if sample_id_value is None or not str(sample_id_value):
            raise ValueError(f"Dataset row is missing required ID column '{c['id']}'")
        sample_id = str(sample_id_value)
        if frozen_ids and sample_id not in remaining_ids:
            continue

        text = (row.get(c["transcript"]) or "").strip()
        if not text:
            continue

        quality = row.get(c["quality"])
        if min_quality is not None:
            try:
                quality = float(quality)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(quality) or quality < min_quality:
                continue

        duration = row.get(c["duration"])
        if duration is not None and not f["min_duration_sec"] <= duration <= f["max_duration_sec"]:
            continue

        audio, sr = _get_audio(row[c["audio"]])
        if duration is None:
            duration = len(audio) / sr
            if not f["min_duration_sec"] <= duration <= f["max_duration_sec"]:
                continue

        samples_by_id[sample_id] = {
            "id": sample_id,
            "audio": audio,
            "sampling_rate": sr,
            "reference": text,
            "quality": quality,
            "duration": float(duration),
        }
        remaining_ids.discard(sample_id)
        if not frozen_ids and len(samples_by_id) >= d["max_samples"]:
            break

    if frozen_ids:
        if remaining_ids:
            missing = sorted(remaining_ids)
            raise ValueError(
                f"Could not reload {len(missing)} frozen samples from the dataset "
                f"after filtering; first missing IDs: {missing[:10]}"
            )
        samples = [samples_by_id[sample_id] for sample_id in frozen_ids]
    else:
        samples = list(samples_by_id.values())
        save_manifest(samples, cfg)

    logger.info("Collected %d samples", len(samples))
    return samples


def save_manifest(samples: List[Dict[str, Any]], cfg: Dict[str, Any]) -> None:
    """Persist selected IDs so subsequent runs reuse the same evaluation samples."""
    path = cfg["dataset"]["sample_ids_file"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    manifest = {
        "hf_repo": cfg["dataset"]["hf_repo"],
        "subset": cfg["dataset"]["subset"],
        "split": cfg["dataset"]["split"],
        "seed": cfg["dataset"].get("seed"),
        "filtering": cfg["filtering"],
        "ids": [s["id"] for s in samples],
    }
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Saved %d sample IDs to %s", len(manifest["ids"]), path)
