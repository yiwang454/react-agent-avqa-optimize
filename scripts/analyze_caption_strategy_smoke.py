#!/usr/bin/env python3
"""Descriptive raw-output audit; timestamp coverage is not semantic accuracy."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics

TAG = re.compile(r"^\s*\[\[(?:start_local=|start_s=)?([^;]+);(?:end_local=|end_s=)?([^\]]+)\]\]\s*(.*)")
CLOCK = r"\d{1,3}:\d{2}(?::\d{2,3})?(?:\.\d+)?"
RANGE = re.compile(r"(" + CLOCK + r")\s*[-–—]\s*(" + CLOCK + r")")
DECIMAL_RANGE = re.compile(r"^\s*[-*]?\s*\*{0,2}(\d+\.\d+)\s*[-–—]\s*(\d+\.\d+)")
POINT = re.compile(r"^\s*[-*]?\s*\*{0,2}(" + CLOCK + r")\s*(?:\([^)]*\))?\*{0,2}[: ]\s*(.*)")


def seconds(value):
    result = 0.
    for part in value.strip().split(":"):
        result = result * 60 + float(part)
    return result


def valid_clock(value):
    parts = value.split(":")
    return len(parts) == 1 or all(float(p) < 60 for p in parts[1:])


def segments(raw):
    clean = raw.strip().removeprefix("```json").removesuffix("```").strip()
    if clean.startswith("[") and not clean.startswith("[["):
        try:
            payload = json.loads(clean)
            if isinstance(payload, list):
                return [[seconds(str(p["start"])), seconds(str(p["end"])),
                         str(p.get("caption", p.get("label", "")))] for p in payload]
        except (ValueError, KeyError, TypeError):
            pass
    result = []
    for line in raw.splitlines():
        match = TAG.match(line)
        if match:
            result.append([seconds(match[1]), seconds(match[2]), match[3]])
            continue
        match = RANGE.search(line) or DECIMAL_RANGE.search(line)
        if match:
            # Preserve the segment but exclude malformed ends from span statistics.
            end = seconds(match[2]) if valid_clock(match[2]) else None
            result.append([seconds(match[1]), end, line[match.end():].strip(" *:]-")])
        elif (match := POINT.match(line)):
            result.append([seconds(match[1]), None, match[2]])
        elif result and line.strip():
            result[-1][2] += " " + line.strip()
    return result


def audit(paths, local):
    all_durations, repeats, all_text = [], 0, []
    chunk_rows, longest = [], 0
    for path in sorted(paths):
        row = json.loads(path.read_text())
        segs = segments(row.get("raw_response", ""))
        norm = [re.sub(r"\W+", " ", s[2].lower()).strip() for s in segs]
        counts = Counter(norm)
        duplicate = sum(n-1 for n in counts.values())
        repeats += duplicate
        run = best = 0
        previous = None
        for text in norm:
            run = run + 1 if text == previous else 1
            best = max(best, run)
            previous = text
        longest = max(longest, best)
        ds = [e-s for s,e,_ in segs if e is not None and e >= s]
        all_durations.extend(ds)
        all_text.extend(norm)
        lo, hi = row["chunk_start_s"], row["chunk_end_s"]
        offset = lo if local else 0
        outside = sum(s+offset < lo-1.5 or (e if e is not None else s)+offset > hi+1.5 for s,e,_ in segs)
        quoted = sum(bool(re.search(r'["“”]', text)) for _,_,text in segs)
        chunk_rows.append({"path": str(path), "segments": len(segs), "duplicate_excess": duplicate,
                           "longest_identical_run": best, "outside_chunk": outside,
                           "point_segments": sum(e is None for _,e,_ in segs),
                           "quoted_segments": quoted,
                           "most_common_caption": counts.most_common(1),
                           "raw_tail": row.get("raw_response", "")[-180:],
                           "latency": row.get("latency"),
                           "usage": row.get("token_usage") or row.get("metadata", {}).get("token_usage"),
                           "status": row.get("status")})
    latency = [c["latency"]["retry_adjusted_seconds"] for c in chunk_rows if c.get("latency") and c["latency"].get("retry_adjusted_seconds") is not None]
    return {"calls": len(chunk_rows), "segments": len(all_text), "duplicate_excess": repeats,
            "duplicate_percent": round(100*repeats/len(all_text), 1) if all_text else None,
            "longest_identical_run": longest,
            "median_segment_seconds": statistics.median(all_durations) if all_durations else None,
            "at_most_1s": sum(d <= 1 for d in all_durations),
            "outside_chunk": sum(c["outside_chunk"] for c in chunk_rows),
            "quoted_segments": sum(c["quoted_segments"] for c in chunk_rows),
            "latency_max": max(latency) if latency else None,
            "latency_sum": round(sum(latency),3), "chunks": chunk_rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    args = parser.parse_args()
    variants = {
        "structured_single_A": (args.data_root / "chunked_caption_timestamp_smoke_20261004/local_offset/video_artifacts", True),
        "structured_split_A": (args.data_root / "chunked_caption_timestamp_smoke_split_avs_20261004/local_offset/video_artifacts", True),
        "coarse_chunks_B": (args.new_root / "coarse_chunks", False),
        "whole_video": (args.new_root / "whole_video", False),
        "structured_single_B_control": (args.data_root / "chunked_caption_timestamp_smoke_20261004/model_original/video_artifacts", False),
        "structured_split_B_control": (args.data_root / "chunked_caption_timestamp_smoke_split_avs_20261004/model_original/video_artifacts", False),
    }
    results = {}
    lines = ["# Caption strategy smoke audit", "", "Exact duplication is measured within each call after stripping timestamps and normalizing punctuation/case. Quoted segments are a transcription proxy, not verified speech accuracy; split AVS often has unquoted transcripts, so do not compare quotation counts as speech coverage. Span statistics exclude point-only segments and malformed end times. Three-part clock values use literal HH:MM:SS interpretation; no guessed correction is applied. Counts are descriptive, from one run per condition. A completed API call does not prove a complete or useful caption.", "",
             "| Strategy | Video | Segments | Duplicate excess (%) | Longest identical run | Median span s | ≤1s | Out-of-chunk | Quoted segments | Max / sum latency s |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant, (root, local) in variants.items():
        results[variant] = {}
        for video in ("ayCJscCe", "NUhenCVt", "sKzAjbxS"):
            metric = audit((root / video).glob("chunk_*.json"), local)
            results[variant][video] = metric
            lines.append(f"| {variant} | {video} | {metric['segments']} | {metric['duplicate_excess']} ({metric['duplicate_percent']}%) | {metric['longest_identical_run']} | {metric['median_segment_seconds']} | {metric['at_most_1s']} | {metric['outside_chunk']} | {metric['quoted_segments']} | {metric['latency_max']} / {metric['latency_sum']} |")
    (args.new_root / "strategy_audit.json").write_text(json.dumps(results, indent=2))
    (args.new_root / "strategy_audit.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
