#!/usr/bin/env python3
"""Prepare one Val125 manifest with original AV and OmniAgent sepAV aliases."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def custom(row: dict) -> dict:
    return row["supervisions"][0]["custom"]


def ensure_link(link: Path, target: Path) -> None:
    target = target.resolve(strict=True)
    if link.is_symlink():
        if link.resolve() != target:
            raise ValueError(f"Conflicting media alias: {link}")
        return
    if link.exists():
        raise ValueError(f"Media alias already exists and is not a symlink: {link}")
    link.symlink_to(target)


def duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
         "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def prepare(original: Path, separated: Path, output: Path, *, check_durations: bool = False) -> None:
    original_rows = read_rows(original)
    sep_rows = {str(row["id"]): row for row in read_rows(separated)}
    ids = [str(row["id"]) for row in original_rows]
    if len(ids) != len(set(ids)) or any(sample_id not in sep_rows for sample_id in ids):
        raise ValueError("Original and sepAV manifests must contain the same unique selected IDs")
    alias_dir = output.parent / "sepav_media"
    alias_dir.mkdir(parents=True, exist_ok=True)
    prepared = []
    checked_videos: set[str] = set()
    max_duration_delta = (0.0, "")
    for source in original_rows:
        sample_id = str(source["id"])
        if Path(sample_id).name != sample_id:
            raise ValueError(f"Unsafe sample ID: {sample_id!r}")
        original_custom = custom(source)
        sep_custom = custom(sep_rows[sample_id])
        if original_custom.get("video_id") != sep_custom.get("video_id"):
            raise ValueError(f"Video ID mismatch for {sample_id}")
        original_video = Path(original_custom["video_path"])
        sep_video = Path(sep_custom["video_path"])
        sep_audio = Path(sep_custom["audio_path"])
        original_video.resolve(strict=True)
        if check_durations and original_custom["video_id"] not in checked_videos:
            original_duration = duration(original_video)
            separated_duration = duration(sep_video)
            duration_delta = abs(original_duration - separated_duration)
            if duration_delta > max_duration_delta[0]:
                max_duration_delta = (duration_delta, sample_id)
            if duration_delta > 0.25:
                raise ValueError(
                    f"Original/sepAV timeline differs for {sample_id}: "
                    f"{original_duration:.3f}s vs {separated_duration:.3f}s"
                )
            checked_videos.add(original_custom["video_id"])
        video_alias = alias_dir / f"{sample_id}.mp4"
        audio_alias = alias_dir / f"{sample_id}.wav"
        ensure_link(video_alias, sep_video)
        ensure_link(audio_alias, sep_audio)
        row = copy.deepcopy(source)
        row_custom = custom(row)
        row_custom["specialized_video_path"] = str(video_alias)
        row_custom["specialized_audio_path"] = str(audio_alias)
        prepared.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reject stale manifests so checkpoint/resume cannot silently change media.
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prepared)
    if output.exists() and output.read_text(encoding="utf-8") != content:
        raise ValueError(f"Existing prepared manifest differs: {output}")
    if not output.exists():
        output.write_text(content, encoding="utf-8")
    print(f"Prepared {len(prepared)} selected samples: {output}")
    if check_durations:
        print(
            f"Verified original/sepAV duration for {len(checked_videos)} unique videos; "
            f"largest delta={max_duration_delta[0]:.3f}s ({max_duration_delta[1]})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--separated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-durations", action="store_true")
    args = parser.parse_args()
    prepare(args.original, args.separated, args.output, check_durations=args.check_durations)
