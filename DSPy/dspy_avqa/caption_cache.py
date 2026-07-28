"""Deterministic, question-scoped caption cache utilities."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


CACHE_SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class CachedCaption:
    question_id: str
    response: str
    caption_prompt: str
    path: Path
    source_results_file: str
    source_rank: int
    source_backend: str
    source_model: str
    source_token_usage: Any


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_cache_path(cache_dir: Path, question_id: str) -> Path:
    normalized_id = str(question_id or "").strip()
    if not normalized_id:
        raise ValueError("Caption cache lookup requires a non-empty question_id")
    if Path(normalized_id).name != normalized_id or "/" in normalized_id or "\\" in normalized_id:
        raise ValueError(f"Unsafe caption cache question_id: {question_id!r}")
    return cache_dir / f"{normalized_id}.json"


def effective_caption_prompt_from_yaml(prompt_yaml: Path) -> str:
    """Render the fixed caption prompt represented by a V8 prompt YAML."""
    path = prompt_yaml.expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, Mapping):
        raise ValueError(f"Prompt YAML must contain a mapping: {path}")
    captioner = config.get("captioner")
    if not isinstance(captioner, Mapping):
        raise ValueError(f"Prompt YAML has no captioner mapping: {path}")
    instruction = str(captioner.get("default_caption_instruction") or "").strip()
    if not instruction:
        raise ValueError(f"Prompt YAML has no captioner.default_caption_instruction: {path}")
    template = str(captioner.get("caption_prompt_template") or "{caption_instruction}")
    try:
        return template.format(caption_instruction=instruction).strip()
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid captioner.caption_prompt_template in {path}: {exc}") from exc


def load_cached_caption(
    cache_dir: Path,
    question_id: str,
    *,
    expected_prompt: str,
) -> CachedCaption:
    """Load one exact question-scoped cache entry and validate its prompt."""
    root = cache_dir.expanduser().resolve()
    path = _safe_cache_path(root, question_id)
    if not path.is_file():
        raise FileNotFoundError(f"Caption cache entry not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Caption cache entry must be a JSON object: {path}")

    cached_id = str(payload.get("question_id") or "").strip()
    if cached_id != str(question_id).strip():
        raise ValueError(
            f"Caption cache question_id mismatch at {path}: "
            f"expected {question_id!r}, found {cached_id!r}"
        )
    response = str(payload.get("response") or "").strip()
    if not response:
        raise ValueError(f"Caption cache response is empty: {path}")
    caption_prompt = str(payload.get("caption_prompt") or "").strip()
    active_prompt = str(expected_prompt or "").strip()
    if not caption_prompt:
        raise ValueError(f"Caption cache caption_prompt is empty: {path}")
    if caption_prompt != active_prompt:
        raise ValueError(
            f"Caption cache prompt mismatch for question_id={question_id}: "
            f"cache_sha256={_sha256_text(caption_prompt)} "
            f"active_sha256={_sha256_text(active_prompt)}"
        )

    try:
        source_rank = int(payload.get("source_rank"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Caption cache source_rank is invalid: {path}") from exc
    return CachedCaption(
        question_id=cached_id,
        response=response,
        caption_prompt=caption_prompt,
        path=path,
        source_results_file=str(payload.get("source_results_file") or ""),
        source_rank=source_rank,
        source_backend=str(payload.get("source_backend") or ""),
        source_model=str(payload.get("source_model") or ""),
        source_token_usage=payload.get("source_token_usage"),
    )


def load_caption_cache_manifest(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir.expanduser().resolve() / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Caption cache manifest not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Caption cache manifest must be a JSON object: {path}")
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported caption cache schema at {path}: "
            f"{payload.get('schema_version')!r}"
        )
    return payload


def _cache_content_sha256(cache_dir: Path) -> tuple[str, int]:
    """Recompute the manifest content digest and validate entry filenames."""
    digest = hashlib.sha256()
    entry_count = 0
    for path in sorted(cache_dir.glob("*.json"), key=lambda value: value.name):
        if path.name == MANIFEST_NAME:
            continue
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Caption cache entry must be a JSON object: {path}")
        question_id = str(payload.get("question_id") or "").strip()
        expected_path = _safe_cache_path(cache_dir, question_id)
        if path != expected_path:
            raise ValueError(
                f"Caption cache filename/question_id mismatch: {path} "
                f"(expected {expected_path.name})"
            )
        entry_text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(question_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry_text.encode("utf-8"))
        digest.update(b"\n")
        entry_count += 1
    return digest.hexdigest(), entry_count


def validate_caption_cache_coverage(
    cache_dir: Path,
    question_ids: Iterable[str],
    *,
    expected_prompt: str,
) -> dict[str, Any]:
    """Fail fast unless every requested ID has a valid exact cache entry."""
    root = cache_dir.expanduser().resolve()
    manifest = load_caption_cache_manifest(root)
    manifest_prompt = str(manifest.get("caption_prompt") or "").strip()
    if manifest_prompt != str(expected_prompt or "").strip():
        raise ValueError(
            "Caption cache manifest prompt mismatch: "
            f"cache_sha256={_sha256_text(manifest_prompt)} "
            f"active_sha256={_sha256_text(str(expected_prompt or '').strip())}"
        )
    content_sha256, entry_count = _cache_content_sha256(root)
    if content_sha256 != str(manifest.get("content_sha256") or ""):
        raise ValueError(
            "Caption cache content hash mismatch: "
            f"manifest={manifest.get('content_sha256')!r} actual={content_sha256!r}"
        )
    try:
        manifest_count = int(manifest.get("question_count"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Caption cache manifest question_count is invalid") from exc
    if entry_count != manifest_count:
        raise ValueError(
            "Caption cache entry count mismatch: "
            f"manifest={manifest_count} actual={entry_count}"
        )
    ids = [str(value or "").strip() for value in question_ids]
    if any(not value for value in ids):
        raise ValueError("Caption cache coverage check received an empty question_id")
    duplicates = sorted(value for value, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate question IDs in cache coverage request, e.g. {duplicates[:5]}")
    for question_id in ids:
        load_cached_caption(root, question_id, expected_prompt=expected_prompt)
    return {
        "cache_dir": str(root),
        "requested": len(ids),
        "validated": len(ids),
        "cache_entries": entry_count,
        "manifest_sha256": hashlib.sha256(
            (root / MANIFEST_NAME).read_bytes()
        ).hexdigest(),
        "content_sha256": content_sha256,
        "prompt_sha256": manifest.get("prompt_sha256"),
    }


def _load_input_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
        for row_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"Expected JSON object at {path}:{row_number}")
            question_id = str(payload.get("id") or "").strip()
            if not question_id:
                supervisions = payload.get("supervisions")
                if (
                    isinstance(supervisions, Sequence)
                    and supervisions
                    and isinstance(supervisions[0], Mapping)
                ):
                    question_id = str(supervisions[0].get("id") or "").strip()
            if not question_id:
                raise ValueError(f"No cut/question ID at {path}:{row_number}")
            ids.append(question_id)
    duplicates = sorted(value for value, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate input IDs, e.g. {duplicates[:5]}")
    return ids


def _caption_turn(row: Mapping[str, Any], *, path: Path, row_number: int) -> tuple[str, Mapping[str, Any]]:
    question_data = row.get("question_data")
    question_data = question_data if isinstance(question_data, Mapping) else row
    question_id = str(
        question_data.get("question_id") or row.get("video_id") or ""
    ).strip()
    if not question_id:
        raise ValueError(f"Result row has no question_id at {path}:{row_number}")
    turns = question_data.get("turn_trace")
    turns = turns if isinstance(turns, Sequence) and not isinstance(turns, (str, bytes)) else []
    caption_turns = [
        turn
        for turn in turns
        if isinstance(turn, Mapping) and str(turn.get("tool_name") or "") == "ask_caption"
    ]
    if len(caption_turns) != 1:
        raise ValueError(
            f"Expected exactly one ask_caption turn for {question_id} at "
            f"{path}:{row_number}, found {len(caption_turns)}"
        )
    return question_id, caption_turns[0]


def _load_result_captions(
    path: Path,
    *,
    expected_ids: set[str],
    expected_prompt: str,
) -> dict[str, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    rows: dict[str, dict[str, Any]] = {}
    with resolved.open("r", encoding="utf-8") as stream:
        for row_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"Expected JSON object at {resolved}:{row_number}")
            question_id, turn = _caption_turn(payload, path=resolved, row_number=row_number)
            if question_id in rows:
                raise ValueError(f"Duplicate result question_id {question_id!r} in {resolved}")
            prompt = str(turn.get("perception_prompt") or "").strip()
            if prompt != expected_prompt:
                raise ValueError(
                    f"Caption prompt mismatch for {question_id} in {resolved}: "
                    f"source_sha256={_sha256_text(prompt)} "
                    f"expected_sha256={_sha256_text(expected_prompt)}"
                )
            rows[question_id] = {
                "response": str(turn.get("tool_observation") or "").strip(),
                "caption_prompt": prompt,
                "source_backend": str(turn.get("perception_backend") or ""),
                "source_model": str(turn.get("perception_model") or ""),
                "source_token_usage": turn.get("perception_token_usage"),
            }
    actual_ids = set(rows)
    if actual_ids != expected_ids:
        raise ValueError(
            f"Result ID coverage mismatch for {resolved}: "
            f"missing={len(expected_ids - actual_ids)} extra={len(actual_ids - expected_ids)}"
        )
    return rows


def extract_caption_cache(
    *,
    primary_results: Path,
    fallback_results: Sequence[Path],
    input_jsonl: Path,
    prompt_yaml: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Extract a deterministic cache, using fallbacks only for blank primary rows."""
    input_ids = _load_input_ids(input_jsonl)
    expected_ids = set(input_ids)
    expected_prompt = effective_caption_prompt_from_yaml(prompt_yaml)
    sources = [primary_results, *fallback_results]
    if not sources:
        raise ValueError("At least one result source is required")
    resolved_sources = [path.expanduser().resolve() for path in sources]
    source_rows = [
        _load_result_captions(
            path,
            expected_ids=expected_ids,
            expected_prompt=expected_prompt,
        )
        for path in resolved_sources
    ]

    selected: dict[str, dict[str, Any]] = {}
    source_counts: Counter[int] = Counter()
    for question_id in input_ids:
        for source_rank, (source_path, rows) in enumerate(
            zip(resolved_sources, source_rows), 1
        ):
            source = rows[question_id]
            if not source["response"]:
                continue
            entry = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "question_id": question_id,
                "response": source["response"],
                "caption_prompt": source["caption_prompt"],
                "source_results_file": str(source_path),
                "source_rank": source_rank,
                "source_backend": source["source_backend"],
                "source_model": source["source_model"],
                "source_token_usage": source["source_token_usage"],
            }
            selected[question_id] = entry
            source_counts[source_rank] += 1
            break
    missing = [question_id for question_id in input_ids if question_id not in selected]
    if missing:
        raise ValueError(
            f"No non-empty caption found for {len(missing)} input IDs, e.g. {missing[:5]}"
        )

    root = output_dir.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Caption cache output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    for question_id in sorted(selected):
        entry_text = json.dumps(
            selected[question_id],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(question_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry_text.encode("utf-8"))
        digest.update(b"\n")
        path = _safe_cache_path(root, question_id)
        path.write_text(
            json.dumps(selected[question_id], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "input_jsonl": str(input_jsonl.expanduser().resolve()),
        "prompt_yaml": str(prompt_yaml.expanduser().resolve()),
        "caption_prompt": expected_prompt,
        "prompt_sha256": _sha256_text(expected_prompt),
        "source_results_files": [str(path) for path in resolved_sources],
        "source_counts": {
            str(rank): source_counts.get(rank, 0)
            for rank in range(1, len(resolved_sources) + 1)
        },
        "question_count": len(input_ids),
        "content_sha256": digest.hexdigest(),
    }
    (root / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
