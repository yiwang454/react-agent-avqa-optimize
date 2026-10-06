#!/usr/bin/env python3
"""Create a size-safe WorldSense media set and a rewritten full JSONL.

Original MP4s and the source JSONL are never modified. Videos strictly larger
than 30 MiB are transcoded to at most 18 MiB. Their display aspect ratio and
full duration are retained, while pixel area is capped at the median pixel area
of WorldSense videos that are already at most 30 MiB.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


MIB = 1024 * 1024
DEFAULT_INPUT = Path(
    "/mnt/ceph_rbd/data/avqa_project/WorldSense/worldsense_test_cut_old.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/ceph_rbd/data/avqa_project/WorldSense/"
    "worldsense_reencoded_over30m_max18m"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False,
        prefix=f".{path.name}.",
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False,
        prefix=f".{path.name}.",
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def probe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration,size",
        "-of", "json", str(path),
    ]
    result = json.loads(subprocess.check_output(command, text=True))
    stream = result["streams"][0]
    media = result["format"]
    width = int(stream["width"])
    height = int(stream["height"])
    return {
        "width": width,
        "height": height,
        "pixel_area": width * height,
        "duration_seconds": float(media["duration"]),
        "bytes": int(media["size"]),
        "mebibytes": int(media["size"]) / MIB,
    }


def target_dimensions(width: int, height: int, target_area: int) -> tuple[int, int]:
    """Return even dimensions without upscaling, preserving display aspect."""
    if width * height <= target_area:
        return max(2, width // 2 * 2), max(2, height // 2 * 2)
    scale = math.sqrt(target_area / (width * height))
    return max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = Counter((item["width"], item["height"]) for item in records)
    return {
        "videos": len(records),
        "median_width": statistics.median(item["width"] for item in records),
        "median_height": statistics.median(item["height"] for item in records),
        "median_pixel_area": statistics.median(item["pixel_area"] for item in records),
        "median_duration_seconds": statistics.median(
            item["duration_seconds"] for item in records
        ),
        "median_mebibytes": statistics.median(item["bytes"] for item in records) / MIB,
        "min_mebibytes": min(item["bytes"] for item in records) / MIB,
        "max_mebibytes": max(item["bytes"] for item in records) / MIB,
        "dimension_counts": [
            {"width": width, "height": height, "videos": count}
            for (width, height), count in dimensions.most_common()
        ],
    }


def transcode(
    *, source: Path, destination: Path, dimensions: tuple[int, int],
    duration: float, max_output_mib: float,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    target_bytes = int(max_output_mib * MIB)
    audio_kbps = 48
    video_kbps = max(64, int(target_bytes * 8 / duration / 1000) - audio_kbps - 8)
    width, height = dimensions
    for attempt in range(1, 5):
        temporary = destination.with_suffix(f".attempt{attempt}.mp4")
        temporary.unlink(missing_ok=True)
        command = [
            "ffmpeg", "-y", "-v", "error", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a?",
            "-vf", f"scale={width}:{height}:flags=lanczos",
            "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-b:v", f"{video_kbps}k", "-maxrate", f"{video_kbps}k",
            "-bufsize", f"{video_kbps * 2}k",
            "-c:a", "aac", "-b:a", f"{audio_kbps}k", "-ac", "1",
            "-movflags", "+faststart", str(temporary),
        ]
        subprocess.run(command, check=True)
        encoded = probe(temporary)
        if encoded["bytes"] <= target_bytes:
            os.replace(temporary, destination)
            encoded.update({"video_kbps": video_kbps, "attempt": attempt})
            return encoded
        temporary.unlink(missing_ok=True)
        video_kbps = max(48, int(video_kbps * target_bytes / encoded["bytes"] * 0.90))
    raise RuntimeError(
        f"Could not fit {source} below {max_output_mib:.2f} MiB after four attempts"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--oversized-threshold-mib", type=float, default=30.0)
    parser.add_argument("--max-output-mib", type=float, default=18.0)
    parser.add_argument("--probe-workers", type=int, default=16)
    parser.add_argument("--encode-workers", type=int, default=2)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.oversized_threshold_mib <= 0
        or args.max_output_mib <= 0
        or args.max_output_mib >= args.oversized_threshold_mib
        or args.probe_workers < 1
        or args.encode_workers < 1
    ):
        raise ValueError("Thresholds and workers are invalid")
    if not shutil.which("ffprobe") or not shutil.which("ffmpeg"):
        raise RuntimeError("ffprobe and ffmpeg must be installed")

    input_jsonl = args.input_jsonl.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    rows = read_jsonl(input_jsonl)
    cut_ids = [str(row["id"]) for row in rows]
    if len(cut_ids) != 3172 or len(set(cut_ids)) != len(cut_ids):
        raise ValueError(
            f"Expected 3,172 unique WorldSense cut IDs, got "
            f"rows={len(cut_ids)} unique={len(set(cut_ids))}"
        )

    videos: dict[str, dict[str, Any]] = {}
    for row in rows:
        supervision = row["supervisions"][0]
        recording_id = str(supervision["recording_id"])
        custom_path = Path(supervision["custom"]["video_path"]).expanduser().resolve()
        source_path = Path(row["recording"]["sources"][0]["source"]).expanduser().resolve()
        if custom_path != source_path:
            raise ValueError(f"Conflicting video paths for {row['id']}: {custom_path} != {source_path}")
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing WorldSense video: {source_path}")
        record = videos.setdefault(
            recording_id,
            {"recording_id": recording_id, "source_path": str(source_path), "cut_ids": []},
        )
        if record["source_path"] != str(source_path):
            raise ValueError(f"Multiple source paths for recording {recording_id}")
        record["cut_ids"].append(str(row["id"]))
    if len(videos) != 1662:
        raise ValueError(f"Expected 1,662 unique videos, got {len(videos)}")

    with ThreadPoolExecutor(max_workers=args.probe_workers) as executor:
        futures = {
            executor.submit(probe, Path(record["source_path"])): recording_id
            for recording_id, record in videos.items()
        }
        completed = 0
        for future in as_completed(futures):
            videos[futures[future]].update(future.result())
            completed += 1
            if completed % 200 == 0 or completed == len(futures):
                print(f"Probed {completed}/{len(futures)} videos", flush=True)

    threshold_bytes = int(args.oversized_threshold_mib * MIB)
    within_limit = [record for record in videos.values() if record["bytes"] <= threshold_bytes]
    oversized = [record for record in videos.values() if record["bytes"] > threshold_bytes]
    target_area = int(statistics.median(record["pixel_area"] for record in within_limit))
    oversized_ids = {record["recording_id"] for record in oversized}
    oversized_cut_count = sum(len(record["cut_ids"]) for record in oversized)

    output_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "schema_version": 1,
        "input_jsonl": str(input_jsonl),
        "question_count": len(rows),
        "video_count": len(videos),
        "within_30_mib": summarize(within_limit),
        "over_30_mib": summarize(oversized),
        "oversized_question_count": oversized_cut_count,
        "policy": {
            "oversized_threshold_mebibytes": args.oversized_threshold_mib,
            "max_output_mebibytes": args.max_output_mib,
            "target_pixel_area": target_area,
            "target_pixel_area_source": (
                "median pixel area among WorldSense source videos at or below 30 MiB"
            ),
            "aspect_ratio": "preserved",
            "upscaling": False,
            "source_media_modified": False,
        },
    }
    atomic_write_json(output_dir / "media_audit.json", audit)

    csv_path = output_dir / "oversized_videos.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "recording_id", "cut_count", "width", "height", "pixel_area",
            "duration_seconds", "bytes", "mebibytes", "source_path",
            "target_width", "target_height", "output_path",
        ])
        writer.writeheader()
        for record in sorted(oversized, key=lambda item: item["bytes"], reverse=True):
            target_width, target_height = target_dimensions(
                record["width"], record["height"], target_area
            )
            writer.writerow({
                "recording_id": record["recording_id"],
                "cut_count": len(record["cut_ids"]),
                "width": record["width"],
                "height": record["height"],
                "pixel_area": record["pixel_area"],
                "duration_seconds": f"{record['duration_seconds']:.6f}",
                "bytes": record["bytes"],
                "mebibytes": f"{record['mebibytes']:.6f}",
                "source_path": record["source_path"],
                "target_width": target_width,
                "target_height": target_height,
                "output_path": str(output_dir / "videos" / f"{record['recording_id']}.mp4"),
            })

    print(
        f"Audit: {len(videos)} videos / {len(rows)} questions; "
        f"over {args.oversized_threshold_mib:g} MiB: {len(oversized)} videos / "
        f"{oversized_cut_count} questions; target median area={target_area} px",
        flush=True,
    )
    if args.audit_only:
        return

    def encode_one(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        recording_id = record["recording_id"]
        destination = output_dir / "videos" / f"{recording_id}.mp4"
        dimensions = target_dimensions(record["width"], record["height"], target_area)
        if destination.is_file():
            existing = probe(destination)
            if (
                existing["bytes"] <= int(args.max_output_mib * MIB)
                and (existing["width"], existing["height"]) == dimensions
            ):
                existing.update({"reused": True})
                return recording_id, existing
        encoded = transcode(
            source=Path(record["source_path"]),
            destination=destination,
            dimensions=dimensions,
            duration=record["duration_seconds"],
            max_output_mib=args.max_output_mib,
        )
        encoded.update({"reused": False})
        return recording_id, encoded

    encoded: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.encode_workers) as executor:
        futures = {executor.submit(encode_one, record): record for record in oversized}
        for index, future in enumerate(as_completed(futures), start=1):
            recording_id, result = future.result()
            encoded[recording_id] = result
            print(
                f"Encoded {index}/{len(futures)} {recording_id}: "
                f"{result['width']}x{result['height']}, {result['mebibytes']:.2f} MiB, "
                f"reused={result['reused']}",
                flush=True,
            )

    if set(encoded) != oversized_ids:
        raise ValueError("Encoded recording set does not match oversized recording set")
    replacements = {
        recording_id: str(output_dir / "videos" / f"{recording_id}.mp4")
        for recording_id in encoded
    }
    rewritten: list[dict[str, Any]] = []
    replaced_rows = 0
    for source_row in rows:
        row = json.loads(json.dumps(source_row))
        recording_id = str(row["supervisions"][0]["recording_id"])
        replacement = replacements.get(recording_id)
        if replacement is not None:
            row["supervisions"][0]["custom"]["video_path"] = replacement
            row["recording"]["sources"][0]["source"] = replacement
            replaced_rows += 1
        rewritten.append(row)

    output_jsonl = output_dir / "worldsense_test_cut_reencoded_over30m_max18m.jsonl"
    atomic_write_jsonl(output_jsonl, rewritten)
    output_ids = [str(row["id"]) for row in rewritten]
    if output_ids != cut_ids:
        raise ValueError("Output JSONL changed source row IDs or ordering")
    for path in replacements.values():
        result = probe(Path(path))
        if result["bytes"] > int(args.max_output_mib * MIB):
            raise ValueError(f"Re-encoded file exceeds limit: {path}")

    manifest = {
        "schema_version": 1,
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "question_count": len(rewritten),
        "unique_question_ids": len(set(output_ids)),
        "video_count": len(videos),
        "reencoded_video_count": len(encoded),
        "replaced_question_count": replaced_rows,
        "policy": audit["policy"],
        "videos": {recording_id: encoded[recording_id] for recording_id in sorted(encoded)},
    }
    atomic_write_json(output_dir / "reencoded_videos_manifest.json", manifest)
    print(
        f"Wrote {output_jsonl}: {len(rewritten)} rows, {len(set(output_ids))} unique IDs, "
        f"{replaced_rows} rows use {len(encoded)} re-encoded videos.",
        flush=True,
    )


if __name__ == "__main__":
    main()
