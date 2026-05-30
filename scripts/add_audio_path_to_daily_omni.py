#!/usr/bin/env python3
import json
from pathlib import Path


INPUT_PATH = Path(
    "/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_v2.jsonl"
)
OUTPUT_PATH = Path(
    "/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_v3.jsonl"
)


def build_audio_path(video_path: str) -> str:
    return video_path.replace("_video.mp4", "_audio.wav")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    updated_supervisions = 0

    with INPUT_PATH.open("r", encoding="utf-8") as fin, OUTPUT_PATH.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            total_lines += 1

            for supervision in obj.get("supervisions", []):
                custom = supervision.get("custom")
                if not isinstance(custom, dict):
                    raise ValueError(f"Custom is not a dict: {custom}")

                video_path = custom.get("video_path")
                if not isinstance(video_path, str):
                    raise ValueError(f"Video path is not a string: {video_path}")

                custom["audio_path"] = build_audio_path(video_path)
                updated_supervisions += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Done. lines={total_lines}, updated_supervisions={updated_supervisions}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
