#!/usr/bin/env python3
"""Generate per-question densified supervision labels for DailyOmni."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, field_validator


DAILY_OMNI_ROOT = Path("/mnt/ceph_rbd/data/avqa_project/daily_omni")
DEFAULT_INPUT_JSONL = DAILY_OMNI_ROOT / "daily_omni_cuts_selectedTrain125.jsonl"
DEFAULT_OUTPUT_DIR = (
    DAILY_OMNI_ROOT
    / "daily_omni_densified_labels_selectedTrain125_gpt-5.4_high"
)
DEFAULT_PROMPT_YAML = (
    Path(__file__).resolve().parent
    / "dspy_avqa"
    / "yamls"
    / "generating_densified_label.yaml"
)
EXPECTED_TRAINING_QUESTIONS = 125
MODEL = "gpt-5.4"
REASONING_EFFORT = "high"
MAX_OUTPUT_TOKENS = 4096
DEFAULT_TIMEOUT_SECONDS = 180.0

CAPTION_FILENAMES = {
    "av_alignment_captions": "av_alignment_captions.txt",
    "video_consistent_captions": "video_consistent_captions.txt",
    "audio_revised_captions": "audio_revised_captions.txt",
}
LABEL_KEYS = {"key_evidence", "ideal_perception_target"}


class DensifiedLabel(BaseModel):
    """The exact JSON payload written for one question."""

    model_config = ConfigDict(extra="forbid")

    key_evidence: str
    ideal_perception_target: str

    @field_validator("key_evidence", "ideal_perception_target")
    @classmethod
    def validate_nonempty_text(cls, value: str) -> str:
        """Strip label text and reject empty model outputs."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("label text must not be empty")
        return normalized


@dataclass(frozen=True)
class CaptionBundle:
    """Privileged DailyOmni annotations shared by questions from one video."""

    av_alignment_captions: str
    video_consistent_captions: str
    audio_revised_captions: str


@dataclass(frozen=True)
class QuestionRecord:
    """Validated model input for one DailyOmni question."""

    cut_id: str
    video_id: str
    question: str
    options: tuple[str, ...]
    gold_answer: str
    captions: CaptionBundle


@dataclass(frozen=True)
class GenerationSummary:
    """Counts reported after one generation run."""

    total: int
    generated: int
    skipped: int
    failed: int
    valid_outputs: int


class PreflightError(ValueError):
    """Raised when dataset validation finds one or more errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__("\n".join(self.errors))


class ModelResponseError(RuntimeError):
    """Raised when an API response cannot produce a valid label."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and report malformed rows with line numbers."""
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, raw_line in enumerate(source, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{path}:{line_number}: invalid JSON: {exc}")
                    continue
                if not isinstance(row, dict):
                    errors.append(
                        f"{path}:{line_number}: expected a JSON object, "
                        f"got {type(row).__name__}"
                    )
                    continue
                rows.append(row)
    except OSError as exc:
        raise PreflightError([f"Cannot read input JSONL {path}: {exc}"]) from exc
    if errors:
        raise PreflightError(errors)
    return rows


def load_prompt_template(path: Path) -> str:
    """Load the densified-label prompt template from YAML."""
    try:
        with path.open("r", encoding="utf-8") as source:
            config = yaml.safe_load(source)
    except (OSError, yaml.YAMLError) as exc:
        raise PreflightError([f"Cannot load prompt YAML {path}: {exc}"]) from exc
    if not isinstance(config, dict):
        raise PreflightError([f"Prompt YAML must contain a mapping: {path}"])
    template = config.get("densified_label_prompt")
    if not isinstance(template, str) or not template.strip():
        raise PreflightError(
            [f"Prompt YAML has no non-empty densified_label_prompt: {path}"]
        )
    return template


def _fallback_video_id(cut_id: str) -> str:
    """Remove a numeric question suffix from a cut ID when present."""
    base, separator, suffix = cut_id.rpartition("-")
    if separator and base and suffix.isdigit():
        return base
    return cut_id


def _safe_cut_id(raw_cut_id: Any) -> str:
    """Validate that a cut ID is safe to use as a single filename."""
    cut_id = str(raw_cut_id or "").strip()
    if (
        not cut_id
        or cut_id in {".", ".."}
        or "/" in cut_id
        or "\\" in cut_id
        or Path(cut_id).name != cut_id
    ):
        raise ValueError(f"invalid or unsafe cut id: {cut_id!r}")
    return cut_id


def parse_cut_metadata(cut: dict[str, Any]) -> tuple[str, str, str, tuple[str, ...], str]:
    """Extract the question fields needed for densified-label generation."""
    cut_id = _safe_cut_id(cut.get("id"))
    supervisions = cut.get("supervisions")
    if not isinstance(supervisions, list) or not supervisions:
        raise ValueError("missing first supervision")
    supervision = supervisions[0]
    if not isinstance(supervision, dict):
        raise ValueError("first supervision is not an object")
    custom = supervision.get("custom") or {}
    if not isinstance(custom, dict):
        raise ValueError("supervision custom field is not an object")

    question = str(
        supervision.get("text")
        or custom.get("question")
        or custom.get("Question")
        or ""
    ).strip()
    raw_options = custom.get("options") or custom.get("Choice")
    raw_answer = custom.get("answer") or custom.get("Answer")
    raw_video_id = (
        custom.get("video_id")
        or supervision.get("recording_id")
        or (cut.get("recording") or {}).get("id")
        or _fallback_video_id(cut_id)
    )

    if not question:
        raise ValueError("question is empty")
    if not isinstance(raw_options, list) or not raw_options:
        raise ValueError("options are missing or empty")
    options = tuple(str(option).strip() for option in raw_options)
    if any(not option for option in options):
        raise ValueError("options contain an empty value")
    gold_answer = str(raw_answer or "").strip()
    if not gold_answer:
        raise ValueError("gold answer is empty")
    video_id = str(raw_video_id or "").strip()
    if not video_id or "/" in video_id or "\\" in video_id:
        raise ValueError(f"invalid video id: {video_id!r}")

    return cut_id, video_id, question, options, gold_answer


def _read_caption_file(path: Path) -> str:
    """Read one privileged caption file, allowing an empty annotation source."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc


def load_caption_bundle(
    daily_omni_root: Path,
    video_id: str,
    cache: dict[str, CaptionBundle],
) -> CaptionBundle:
    """Load all privileged captions once per video."""
    cached = cache.get(video_id)
    if cached is not None:
        return cached

    video_dir = daily_omni_root / "Videos" / video_id
    values = {
        field_name: _read_caption_file(video_dir / filename)
        for field_name, filename in CAPTION_FILENAMES.items()
    }
    bundle = CaptionBundle(**values)
    cache[video_id] = bundle
    return bundle


def preflight_dataset(
    cuts: Sequence[dict[str, Any]],
    daily_omni_root: Path,
) -> list[QuestionRecord]:
    """Validate every selected cut and load all annotation inputs before API use."""
    records: list[QuestionRecord] = []
    errors: list[str] = []
    seen_cut_ids: set[str] = set()
    caption_cache: dict[str, CaptionBundle] = {}

    for index, cut in enumerate(cuts, start=1):
        display_id = str(cut.get("id") or f"row-{index}")
        try:
            cut_id, video_id, question, options, gold_answer = parse_cut_metadata(cut)
            if cut_id in seen_cut_ids:
                raise ValueError(f"duplicate cut id: {cut_id}")
            seen_cut_ids.add(cut_id)
            captions = load_caption_bundle(daily_omni_root, video_id, caption_cache)
            records.append(
                QuestionRecord(
                    cut_id=cut_id,
                    video_id=video_id,
                    question=question,
                    options=options,
                    gold_answer=gold_answer,
                    captions=captions,
                )
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"{display_id}: {exc}")

    if errors:
        raise PreflightError(errors)
    return records


def render_prompt(template: str, record: QuestionRecord) -> str:
    """Render one model prompt using the validated question and annotations."""
    values = {
        "question": record.question,
        "options": json.dumps(record.options, ensure_ascii=False, indent=2),
        "gold_answer": record.gold_answer,
        "av_alignment_captions": record.captions.av_alignment_captions,
        "video_consistent_captions": record.captions.video_consistent_captions,
        "audio_revised_captions": record.captions.audio_revised_captions,
    }
    missing = [name for name in values if f"{{{name}}}" not in template]
    if missing:
        raise PreflightError(
            [f"Prompt template is missing placeholders: {', '.join(missing)}"]
        )
    rendered = template
    for name, value in values.items():
        rendered = rendered.replace(f"{{{name}}}", value)
    return rendered


def resolve_api_key() -> str:
    """Resolve an OpenAI key without exposing it in logs."""
    for env_name in ("OPENAI_API_KEY", "PLANNER_API_KEY", "ELM_API_KEY"):
        value = os.environ.get(env_name, "").strip()
        if value and value.upper() != "EMPTY":
            return value
    raise EnvironmentError(
        "Set OPENAI_API_KEY, PLANNER_API_KEY, or ELM_API_KEY before generation."
    )


def create_openai_client() -> OpenAI:
    """Create the shared OpenAI client used by worker threads."""
    return OpenAI(
        api_key=resolve_api_key(),
        max_retries=0,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )


def _response_refusals(response: Any) -> list[str]:
    """Extract any refusal messages from a Responses API result."""
    refusals: list[str] = []
    for output_item in getattr(response, "output", []) or []:
        if getattr(output_item, "type", None) != "message":
            continue
        for content_item in getattr(output_item, "content", []) or []:
            if getattr(content_item, "type", None) == "refusal":
                refusal = str(getattr(content_item, "refusal", "") or "").strip()
                refusals.append(refusal or "model refusal")
    return refusals


def call_model_once(client: Any, prompt: str) -> DensifiedLabel:
    """Call GPT-5.4 once and validate its Structured Output."""
    response = client.responses.parse(
        model=MODEL,
        input=prompt,
        text_format=DensifiedLabel,
        reasoning={"effort": REASONING_EFFORT},
        max_output_tokens=MAX_OUTPUT_TOKENS,
        store=False,
    )
    refusals = _response_refusals(response)
    if refusals:
        raise ModelResponseError("; ".join(refusals))
    status = getattr(response, "status", None)
    if status != "completed":
        details = getattr(response, "incomplete_details", None)
        raise ModelResponseError(
            f"response status is {status!r}; incomplete_details={details!r}"
        )
    parsed = getattr(response, "output_parsed", None)
    if parsed is None:
        raise ModelResponseError("response has no parsed structured output")
    try:
        return DensifiedLabel.model_validate(parsed)
    except (TypeError, ValueError) as exc:
        raise ModelResponseError(f"invalid structured output: {exc}") from exc


def call_model_with_retries(
    client: Any,
    prompt: str,
    *,
    cut_id: str,
    max_attempts: int,
    sleep_fn: Callable[[float], None] = time.sleep,
    random_fn: Callable[[], float] = random.random,
) -> DensifiedLabel:
    """Retry transient API and response-validation failures with backoff."""
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return call_model_once(client, prompt)
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            delay = min(30.0, 2.0 ** (attempt - 1)) + 0.25 * random_fn()
            print(
                f"[warn] {cut_id}: attempt {attempt}/{max_attempts} failed: "
                f"{exc}; retrying in {delay:.2f}s",
                file=sys.stderr,
                flush=True,
            )
            sleep_fn(delay)
    assert last_error is not None
    raise last_error


def load_existing_label(path: Path) -> DensifiedLabel | None:
    """Return a valid existing label, or None for missing/invalid files."""
    try:
        with path.open("r", encoding="utf-8") as source:
            payload = json.load(source)
        if not isinstance(payload, dict) or set(payload) != LABEL_KEYS:
            return None
        return DensifiedLabel.model_validate(payload)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def write_label_atomic(path: Path, label: DensifiedLabel) -> None:
    """Atomically write one exact two-field label JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(
                label.model_dump(),
                temp_file,
                ensure_ascii=False,
                indent=2,
            )
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _output_path(output_dir: Path, record: QuestionRecord) -> Path:
    return output_dir / f"{record.cut_id}.json"


def _count_valid_outputs(records: Sequence[QuestionRecord], output_dir: Path) -> int:
    return sum(
        load_existing_label(_output_path(output_dir, record)) is not None
        for record in records
    )


def generate_batch(
    records: Sequence[QuestionRecord],
    template: str,
    output_dir: Path,
    *,
    client: Any,
    max_workers: int,
    max_attempts: int,
    overwrite: bool,
) -> GenerationSummary:
    """Generate all missing labels while preserving valid existing outputs."""
    pending: list[QuestionRecord] = []
    skipped = 0
    for record in records:
        path = _output_path(output_dir, record)
        if not overwrite and load_existing_label(path) is not None:
            skipped += 1
        else:
            pending.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)

    def worker(record: QuestionRecord) -> str:
        prompt = render_prompt(template, record)
        label = call_model_with_retries(
            client,
            prompt,
            cut_id=record.cut_id,
            max_attempts=max_attempts,
        )
        write_label_atomic(_output_path(output_dir, record), label)
        return record.cut_id

    generated = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, record): record for record in pending}
        for future in as_completed(futures):
            record = futures[future]
            try:
                future.result()
                generated += 1
                print(
                    f"[ok] {record.cut_id} ({generated + skipped}/{len(records)})",
                    flush=True,
                )
            except Exception as exc:
                failed += 1
                print(
                    f"[error] {record.cut_id}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    valid_outputs = _count_valid_outputs(records, output_dir)
    return GenerationSummary(
        total=len(records),
        generated=generated,
        skipped=skipped,
        failed=failed,
        valid_outputs=valid_outputs,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the standalone command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate one strict densified pseudo-label JSON file per DailyOmni "
            "training question."
        )
    )
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--daily-omni-root", type=Path, default=DAILY_OMNI_ROOT)
    parser.add_argument("--prompt-yaml", type=Path, default=DEFAULT_PROMPT_YAML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum total API attempts per question (default: 3).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N questions for a smoke test.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate valid existing label files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and captions without creating files or calling the API.",
    )
    parser.add_argument(
        "--print-first-prompt",
        action="store_true",
        help="Print the fully rendered prompt for the first selected question.",
    )
    return parser


def _validate_cli_args(args: argparse.Namespace) -> None:
    errors: list[str] = []
    if args.max_workers < 1:
        errors.append("--max-workers must be >= 1")
    if args.max_retries < 1:
        errors.append("--max-retries must be >= 1")
    if args.limit is not None and args.limit < 1:
        errors.append("--limit must be >= 1")
    if errors:
        raise PreflightError(errors)


def run(args: argparse.Namespace, *, client: Any | None = None) -> int:
    """Execute preflight and optional generation; return a process exit code."""
    _validate_cli_args(args)
    template = load_prompt_template(args.prompt_yaml)
    cuts = read_jsonl(args.input_jsonl)
    if args.limit is None and len(cuts) != EXPECTED_TRAINING_QUESTIONS:
        raise PreflightError(
            [
                f"Expected {EXPECTED_TRAINING_QUESTIONS} questions in "
                f"{args.input_jsonl}, found {len(cuts)}"
            ]
        )
    selected_cuts = cuts if args.limit is None else cuts[: args.limit]
    records = preflight_dataset(selected_cuts, args.daily_omni_root)
    if not records:
        raise PreflightError(["No questions were selected for generation."])
    rendered_prompts = [render_prompt(template, record) for record in records]

    skipped = sum(
        not args.overwrite
        and load_existing_label(_output_path(args.output_dir, record)) is not None
        for record in records
    )
    pending = len(records) - skipped
    print(
        f"Preflight passed: source_questions={len(cuts)}, selected={len(records)}, "
        f"pending={pending}, valid_existing={skipped}",
        flush=True,
    )
    print(
        f"Model: {MODEL}; reasoning_effort: {REASONING_EFFORT}; "
        f"output_dir: {args.output_dir}",
        flush=True,
    )
    if args.print_first_prompt:
        print(
            f"===== FIRST RENDERED PROMPT: {records[0].cut_id} =====",
            flush=True,
        )
        print(rendered_prompts[0], flush=True)
        print("===== END FIRST RENDERED PROMPT =====", flush=True)

    if args.dry_run:
        print("Dry run complete: no files created and no API calls made.", flush=True)
        return 0

    api_client = client if client is not None else create_openai_client()
    summary = generate_batch(
        records,
        template,
        args.output_dir,
        client=api_client,
        max_workers=args.max_workers,
        max_attempts=args.max_retries,
        overwrite=args.overwrite,
    )
    print(
        "Summary: "
        f"total={summary.total} generated={summary.generated} "
        f"skipped={summary.skipped} failed={summary.failed} "
        f"valid_outputs={summary.valid_outputs}/{summary.total}",
        flush=True,
    )
    if summary.failed or summary.valid_outputs != summary.total:
        return 1
    if args.limit is None:
        print(
            f"Confirmed {EXPECTED_TRAINING_QUESTIONS} valid densified label files.",
            flush=True,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (EnvironmentError, PreflightError) as exc:
        print(f"[fatal] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
