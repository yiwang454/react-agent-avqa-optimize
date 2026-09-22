#!/usr/bin/env python3
"""Make a WorldSense Cut JSONL point at locally extracted MP4 files.

The published/previously generated WorldSense Cut file contains machine-local
absolute paths.  The DSPy AVQA loader uses both ``custom.video_path`` and the
Lhotse recording source, so update both fields in a run-local copy and verify
every unique video before an expensive inference job starts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source_jsonl.expanduser().resolve()
    video_dir = args.video_dir.expanduser().resolve()
    output = args.output_jsonl.expanduser().resolve()

    if not source.is_file():
        raise SystemExit(f"Missing WorldSense source JSONL: {source}")
    if not video_dir.is_dir():
        raise SystemExit(
            f"Missing extracted WorldSense video directory: {video_dir}\n"
            "The downloaded worldsense_videos_*.zip archives must be extracted "
            "there first (each archive contains flat <video_id>.mp4 files)."
        )

    rows: list[dict] = []
    required_videos: set[Path] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                cut = json.loads(line)
                supervision = (cut.get("supervisions") or [])[0]
                custom = supervision.get("custom") or {}
                video_id = str(custom["video_id"])
                sources = (cut.get("recording") or {}).get("sources") or []
                source_entry = sources[0]
            except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise SystemExit(f"Malformed WorldSense cut at line {line_number}: {exc}") from exc

            local_video = video_dir / f"{video_id}.mp4"
            custom["video_path"] = str(local_video)
            source_entry["source"] = str(local_video)
            required_videos.add(local_video)
            rows.append(cut)

    missing = sorted(path for path in required_videos if not path.is_file())
    if missing:
        examples = "\n".join(f"  {path}" for path in missing[:20])
        suffix = "\n  ..." if len(missing) > 20 else ""
        raise SystemExit(
            f"WorldSense media validation failed: {len(missing)}/{len(required_videos)} "
            f"required MP4 files are missing under {video_dir}.\n{examples}{suffix}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, delete=False, prefix=f".{output.name}."
    ) as handle:
        temp_output = Path(handle.name)
        for cut in rows:
            handle.write(json.dumps(cut, ensure_ascii=False) + "\n")
    os.replace(temp_output, output)
    print(
        f"Prepared {output}: {len(rows)} cuts, {len(required_videos)} unique MP4 files; "
        "rewrote custom.video_path and recording.sources[0].source."
    )


if __name__ == "__main__":
    main()
