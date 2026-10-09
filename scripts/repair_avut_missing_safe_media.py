#!/usr/bin/env python3
"""Restore missing AVUT safe-media targets referenced by an existing JSONL."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from prepare_worldsense_safe_media import MIB, probe, target_dimensions, transcode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--original-video-dir", type=Path, required=True)
    parser.add_argument("--threshold-mib", type=float, default=30.0)
    parser.add_argument("--max-output-mib", type=float, default=18.0)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input_jsonl.expanduser().resolve().open(encoding="utf-8")
        if line.strip()
    ]
    missing: dict[str, Path] = {}
    for row in rows:
        supervision = row["supervisions"][0]
        custom_path = Path(supervision["custom"]["video_path"]).expanduser().resolve()
        source_path = Path(row["recording"]["sources"][0]["source"]).expanduser().resolve()
        if custom_path != source_path:
            raise ValueError(f"Conflicting paths for {row['id']}: {custom_path} != {source_path}")
        if not source_path.is_file():
            missing[str(supervision["recording_id"])] = source_path

    if not missing:
        print("All referenced AVUT media files already exist.")
        return

    originals: dict[str, tuple[Path, dict[str, float | int]]] = {}
    for video_id, destination in missing.items():
        source = args.original_video_dir.expanduser().resolve() / destination.name
        if not source.is_file():
            raise FileNotFoundError(f"Original AVUT video not found: {source}")
        originals[video_id] = (source, probe(source))

    existing_safe_paths = {
        Path(row["recording"]["sources"][0]["source"]).expanduser().resolve()
        for row in rows
        if Path(row["recording"]["sources"][0]["source"]).expanduser().resolve().is_file()
    }
    safe_areas = [probe(path)["pixel_area"] for path in sorted(existing_safe_paths)]
    target_area = int(statistics.median(safe_areas))
    threshold_bytes = int(args.threshold_mib * MIB)

    def restore(video_id: str) -> tuple[str, str, float]:
        destination = missing[video_id]
        source, metadata = originals[video_id]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if int(metadata["bytes"]) <= threshold_bytes:
            shutil.copy2(source, destination)
            result = probe(destination)
            mode = "copied"
        else:
            dimensions = target_dimensions(
                int(metadata["width"]), int(metadata["height"]), target_area
            )
            result = transcode(
                source=source,
                destination=destination,
                dimensions=dimensions,
                duration=float(metadata["duration_seconds"]),
                max_output_mib=args.max_output_mib,
            )
            mode = "transcoded"
        return video_id, mode, float(result["mebibytes"])

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(restore, video_id): video_id for video_id in missing}
        for index, future in enumerate(as_completed(futures), 1):
            video_id, mode, size_mib = future.result()
            print(
                f"Restored {index}/{len(futures)} {video_id}: {mode}, {size_mib:.2f} MiB",
                flush=True,
            )

    unresolved = [str(path) for path in missing.values() if not path.is_file()]
    if unresolved:
        raise RuntimeError(f"Failed to restore {len(unresolved)} media files: {unresolved[:3]}")
    print(f"Restored all {len(missing)} missing AVUT media targets.")


if __name__ == "__main__":
    main()
