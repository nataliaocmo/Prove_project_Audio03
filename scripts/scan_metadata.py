#!/usr/bin/env python3
"""Scan WorldSpeech en_us parquet metadata WITHOUT downloading audio.

Reads the train and/or test shards of disco-eth/WorldSpeech config `en_us`
via HTTP range requests, projecting all columns EXCEPT `audio` (DR-1 in
docs/specs/dataset-package-submission.md). The 109 GB audio payload is never
materialised; only the small metadata column chunks are fetched.

Per-shard results are cached as parquet files under data/metadata/shards/
so the scan is resumable. The combined tables are written to
data/metadata/en_us_{train,test}_metadata.parquet (git-ignored) and stats
per split to data/metadata/scan_stats.txt.

Usage:
    python scripts/scan_metadata.py                  # scan train + test
    python scripts/scan_metadata.py --split train    # train only
    python scripts/scan_metadata.py --shards 3       # smoke: first 3 shards
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from huggingface_hub import HfFileSystem

REPO = "disco-eth/WorldSpeech"
CONFIG = "en_us"

# Everything except `audio` (DR-1: metadata-first filtering).
META_COLS = [
    "human_transcript", "asr_transcript", "cer", "snr",
    "dnsmos_sig", "dnsmos_bak", "dnsmos_ovr", "dnsmos_p808",
    "duration", "source", "source_url", "source_start_s", "source_end_s",
    "session_date", "segment_id", "language", "country",
]

META_DIR = Path(__file__).resolve().parent.parent / "data" / "metadata"
SHARD_DIR = META_DIR / "shards"


def list_shards(split: str) -> list[str]:
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    return sorted(
        f for f in files
        if f.startswith(f"data/{CONFIG}/") and f.endswith(".parquet")
        and split in Path(f).name
    )


def scan_shard(hffs: HfFileSystem, shard: str) -> pa.Table:
    path = f"datasets/{REPO}/{shard}"  # HfFileSystem needs the repo-type prefix
    pf = pq.ParquetFile(hffs.open(path))
    cols = [c for c in META_COLS if c in pf.schema_arrow.names]
    table = pf.read(columns=cols)
    # DR-1 (spec §5): the audio payload must never be materialised.
    assert "audio" not in table.column_names, (
        f"{shard}: audio column was projected — DR-1 violated"
    )
    return table


def scan_split(hffs: HfFileSystem, split: str, shard_limit: int, fresh: bool) -> Path:
    """Scan one split's shards (resumable), combine, write parquet. Returns out path."""
    shards = list_shards(split)
    if shard_limit:
        shards = shards[:shard_limit]
    print(f"[{split}] {len(shards)} shards to scan (config {CONFIG})", flush=True)

    t0 = time.time()
    for i, shard in enumerate(shards):
        cache = SHARD_DIR / (Path(shard).stem + ".meta.parquet")
        if cache.exists() and not fresh:
            continue
        table = scan_shard(hffs, shard)
        pq.write_table(table, cache)
        print(f"[{split}] [{i + 1}/{len(shards)}] {Path(shard).name}: "
              f"{table.num_rows} rows ({time.time() - t0:.0f}s elapsed)",
              flush=True)

    # Combine cache into one table; record which shard each row came from
    # (needed later to locate audio bytes without re-scanning).
    tables = []
    for p in sorted(SHARD_DIR.glob(f"{split}-*.meta.parquet")):
        t = pq.read_table(p)
        n = t.num_rows
        tables.append(t.append_column("shard", pa.array([p.stem] * n)))
    combined = pa.concat_tables(tables)
    out = META_DIR / f"{CONFIG}_{split}_metadata.parquet"
    pq.write_table(combined, out)
    print(f"[{split}] combined: {combined.num_rows} rows -> {out}", flush=True)
    return out


def write_stats(paths: dict[str, Path]) -> None:
    """Per-split headline stats, appended to data/metadata/scan_stats.txt."""
    import pandas as pd  # local import: only needed for the summary

    lines = []
    for split, out in paths.items():
        df = pq.read_table(out).to_pandas()
        hours = df["duration"].sum() / 3600
        lines.append(f"=== split: {split} | rows: {len(df)} | hours: {hours:.1f} ===")
        if "session_date" in df.columns:
            lines.append(
                f"session_date: {df['session_date'].min()} .. {df['session_date'].max()}"
                f" ({df['session_date'].nunique()} distinct,"
                f" {df['session_date'].isna().sum()} null)"
            )
        if "source" in df.columns:
            src = df.groupby("source")["duration"].agg(["count", "sum"])
            src["hours"] = (src["sum"] / 3600).round(1)
            lines.append("per-source rows/hours:")
            lines.append(src[["count", "hours"]].sort_values("count", ascending=False).to_string())
        for col in ["cer", "snr", "dnsmos_ovr", "duration"]:
            if col in df.columns:
                lines.append(f"\n{col} deciles:")
                lines.append(df[col].describe(percentiles=[.1, .25, .5, .75, .9]).to_string())
        lines.append("")
    text = "\n".join(lines)
    print(text, flush=True)
    stats_path = META_DIR / "scan_stats.txt"
    with open(stats_path, "a") as f:
        f.write(text + "\n")
    print(f"stats appended -> {stats_path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["train", "test", "both"], default="both",
                    help="which official split(s) to scan")
    ap.add_argument("--shards", type=int, default=0,
                    help="only scan the first N shards per split (smoke mode)")
    ap.add_argument("--fresh", action="store_true", help="ignore shard cache")
    args = ap.parse_args()

    splits = ["train", "test"] if args.split == "both" else [args.split]
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    hffs = HfFileSystem()
    paths = {}
    for split in splits:
        paths[split] = scan_split(hffs, split, args.shards, args.fresh)
    write_stats(paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
