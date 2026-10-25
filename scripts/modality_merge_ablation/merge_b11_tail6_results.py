#!/usr/bin/env python3
"""Replace five B11 rows/checkpoints with completed six-call reruns, with backup."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DATA = Path("/mnt/ceph_rbd/data/avqa_project/daily_omni")
ORIGINAL = DATA / "modality_merge_ablation_b11_selectedVal125_repeats"
RERUN = DATA / "modality_merge_ablation_b11_over6_budget6_reruns"
BACKUP = DATA / "modality_merge_ablation_b11_pre_tail6_merge_backup_20261013"
TARGETS = {
    1: {"37UUg0fwzsw-1", "TwsSrD8g3HA-2"},
    2: {"AYW_zgrLc6U-1", "TwsSrD8g3HA-2"},
    3: {"TwsSrD8g3HA-2"},
}
REPO = Path(__file__).resolve().parents[2]


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def row_id(row: dict) -> str:
    question_id = row["question_data"]["question_id"]
    if row["video_id"] != question_id:
        raise ValueError(f"Mismatched video_id/question_id: {question_id}")
    return question_id


def atomic_write(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(content)
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> None:
    if sys.argv[1:] != ["--apply"]:
        raise SystemExit(f"Usage: {sys.argv[0]} --apply")
    if BACKUP.exists():
        raise SystemExit(f"Backup already exists; refusing a second merge: {BACKUP}")

    prepared = {}
    for repeat, expected_ids in TARGETS.items():
        original_dir = ORIGINAL / f"repeat{repeat}" / "b11"
        rerun_dir = RERUN / f"repeat{repeat}" / "b11-tail6"
        original_path = original_dir / "output_test.jsonl"
        rerun_path = rerun_dir / "output_test.jsonl"
        original_rows = read_rows(original_path)
        rerun_rows = read_rows(rerun_path)
        if len(original_rows) != 125 or len({row_id(r) for r in original_rows}) != 125:
            raise ValueError(f"Original repeat{repeat} is not a unique 125-question set")
        replacements = {row_id(row): row for row in rerun_rows}
        if len(replacements) != len(rerun_rows) or replacements.keys() != expected_ids:
            raise ValueError(f"Unexpected rerun IDs in repeat{repeat}")
        original_by_id = {row_id(row): row for row in original_rows}
        for question_id, rerun_row in replacements.items():
            old_row = original_by_id[question_id]
            for key in ("question", "options", "answer"):
                if old_row["question_data"][key] != rerun_row["question_data"][key]:
                    raise ValueError(f"Changed {key} for repeat{repeat}/{question_id}")
            if old_row["metadata"] != rerun_row["metadata"]:
                raise ValueError(f"Changed metadata for repeat{repeat}/{question_id}")
            metrics = rerun_row["question_data"]["metrics"]
            if metrics["budget"]["limit"] != 6 or metrics["tool_calls"]["budgeted_total"] > 6:
                raise ValueError(f"Invalid six-call budget for repeat{repeat}/{question_id}")
            if metrics["evaluation"]["parseable"] is not True:
                raise ValueError(f"Unparseable rerun for repeat{repeat}/{question_id}")
            checkpoint = json.loads((rerun_dir / f"{question_id}.json").read_text())
            if checkpoint != rerun_row["question_data"]:
                raise ValueError(f"Rerun checkpoint/JSONL mismatch for repeat{repeat}/{question_id}")
        if not (original_dir / "metrics_summary.json").is_file():
            raise ValueError(f"Missing original summary for repeat{repeat}")
        for question_id in expected_ids:
            if not (original_dir / f"{question_id}.json").is_file():
                raise ValueError(f"Missing original checkpoint repeat{repeat}/{question_id}")
        merged_rows = [replacements.get(row_id(row), row) for row in original_rows]
        prepared[repeat] = (original_dir, rerun_dir, merged_rows, expected_ids)

    # Back up every file that will be replaced before changing any original.
    for repeat, (original_dir, _, _, expected_ids) in prepared.items():
        backup_dir = BACKUP / f"repeat{repeat}" / "b11"
        backup_dir.mkdir(parents=True)
        for name in ("output_test.jsonl", "metrics_summary.json"):
            shutil.copy2(original_dir / name, backup_dir / name)
        for question_id in expected_ids:
            shutil.copy2(original_dir / f"{question_id}.json", backup_dir / f"{question_id}.json")
    manifest = {
        "original": str(ORIGINAL), "rerun": str(RERUN),
        "replacements": {f"repeat{r}": sorted(ids) for r, ids in TARGETS.items()},
        "note": "Only the listed question records use max_turns=6; all other B11 records retain max_turns=30.",
    }
    atomic_write(BACKUP / "replacement_manifest.json", json.dumps(manifest, indent=2) + "\n")

    for repeat, (original_dir, rerun_dir, merged_rows, expected_ids) in prepared.items():
        output_path = original_dir / "output_test.jsonl"
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged_rows)
        atomic_write(output_path, content)
        for question_id in expected_ids:
            source = rerun_dir / f"{question_id}.json"
            atomic_write(original_dir / f"{question_id}.json", source.read_text(encoding="utf-8"))
        subprocess.run(
            [sys.executable, str(REPO / "scripts/summarize_modality_merge_ablation.py"),
             str(output_path), "--expected-count", "125", "--output-json",
             str(original_dir / "metrics_summary.json")],
            check=True, stdout=subprocess.DEVNULL,
        )
        print(f"Merged repeat{repeat}: {len(expected_ids)} question record(s)")
    print(f"Original files backed up at {BACKUP}")


if __name__ == "__main__":
    main()
