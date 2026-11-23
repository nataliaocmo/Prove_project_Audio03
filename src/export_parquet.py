"""Export the frozen WorldSpeech English evaluation set to Parquet."""

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
from datasets import Audio, Dataset, Features, Value

if __package__:
    from .dataset import load_samples
else:
    from dataset import load_samples


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "eval_config.yaml"
MANIFEST_PATH = REPO_ROOT / "data" / "sample_ids.json"
EXPECTED_SAMPLES = 1000
OUTPUT_PATH = "data/worldspeech_en_1000_prepared.parquet"


def _read_entities(path: Path) -> Dict[str, str]:
    required_columns = {"wav_path", "entities"}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path} must contain columns {sorted(required_columns)}"
            )
        entities_by_id = {}
        for row in reader:
            sample_id = row["wav_path"]
            if not sample_id or sample_id in entities_by_id:
                raise ValueError(f"Missing or duplicate prediction ID: {sample_id!r}")
            entities_by_id[sample_id] = row["entities"] or ""
    if len(entities_by_id) != EXPECTED_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLES} prediction rows, found {len(entities_by_id)}"
        )
    return entities_by_id


def _load_config() -> Dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    dataset_cfg = cfg["dataset"]
    if (
        dataset_cfg["hf_repo"] != "disco-eth/WorldSpeech"
        or dataset_cfg["subset"] != "en_us"
        or dataset_cfg["split"] != "test"
        or dataset_cfg["seed"] != 42
        or dataset_cfg["max_samples"] != EXPECTED_SAMPLES
        or Path(dataset_cfg["sample_ids_file"]).as_posix() != "data/sample_ids.json"
    ):
        raise ValueError(
            "eval_config.yaml must select en_us, seed 42, and the 1,000-ID manifest"
        )

    with MANIFEST_PATH.open(encoding="utf-8") as file:
        manifest = json.load(file)
    manifest_ids = manifest.get("ids") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest_ids, list)
        or manifest.get("hf_repo") != "disco-eth/WorldSpeech"
        or manifest.get("subset") != "en_us"
        or manifest.get("split") != "test"
        or manifest.get("seed") != 42
        or len(manifest_ids) != EXPECTED_SAMPLES
        or len(set(manifest_ids)) != EXPECTED_SAMPLES
    ):
        raise ValueError("sample_ids.json must contain 1,000 unique en_us IDs for seed 42")

    dataset_cfg["sample_ids_file"] = str(MANIFEST_PATH)
    return cfg


def _export() -> None:
    os.chdir(REPO_ROOT)
    cfg = _load_config()
    samples = load_samples(cfg)
    if len(samples) != EXPECTED_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_SAMPLES} loaded samples, got {len(samples)}")

    predictions_path = REPO_ROOT / cfg["output"]["predictions_csv"]
    entities_by_id = _read_entities(predictions_path)
    sample_ids = {sample["id"] for sample in samples}
    if sample_ids != set(entities_by_id):
        raise ValueError("Prediction IDs do not match the frozen evaluation samples")

    records = {
        "id": [],
        "language": [],
        "audio": [],
        "transcript": [],
        "entities": [],
        "duration": [],
        "quality": [],
    }
    for sample in samples:
        audio = np.asarray(sample["audio"], dtype=np.float32)
        if audio.ndim != 1 or sample["sampling_rate"] != 16000:
            raise ValueError(f"Sample {sample['id']} is not mono audio at 16 kHz")

        quality = float(sample["quality"])
        duration = float(sample["duration"])
        if not math.isfinite(quality) or not math.isfinite(duration):
            raise ValueError(f"Sample {sample['id']} has invalid quality or duration")

        records["id"].append(sample["id"])
        records["language"].append("en_us")
        records["audio"].append(
            {"array": audio, "sampling_rate": sample["sampling_rate"]}
        )
        records["transcript"].append(sample["reference"])
        records["entities"].append(entities_by_id[sample["id"]])
        records["duration"].append(duration)
        records["quality"].append(quality)

    features = Features(
        {
            "id": Value("string"),
            "language": Value("string"),
            "audio": Audio(sampling_rate=16000),
            "transcript": Value("string"),
            "entities": Value("string"),
            "duration": Value("float32"),
            "quality": Value("float32"),
        }
    )
    dataset = Dataset.from_dict(records, features=features)
    dataset.to_parquet(OUTPUT_PATH)

    parquet_dataset = Dataset.from_parquet(OUTPUT_PATH, features=features)
    if parquet_dataset.num_rows != EXPECTED_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLES} Parquet rows, got {parquet_dataset.num_rows}"
        )

    preview = {
        column: parquet_dataset[column][0]
        for column in parquet_dataset.column_names
        if column != "audio"
    }
    audio_storage = parquet_dataset.data.column("audio")[0].as_py()
    preview["audio"] = {
        "path": audio_storage.get("path"),
        "bytes": f"<{len(audio_storage['bytes'])} bytes omitted>"
        if audio_storage.get("bytes") is not None
        else None,
        "sampling_rate": 16000,
    }

    size_mb = (REPO_ROOT / OUTPUT_PATH).stat().st_size / (1024 * 1024)
    print(f"Rows: {parquet_dataset.num_rows}")
    print(f"Columns: {parquet_dataset.column_names}")
    print(f"Schema: {parquet_dataset.features}")
    print(f"File size: {size_mb:.2f} MB")
    print(f"First row: {preview}")


if __name__ == "__main__":
    _export()
