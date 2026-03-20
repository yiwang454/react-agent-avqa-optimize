from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

import requests
from langchain.tools import tool
from pydantic import BaseModel, Field


class PerceptionInput(BaseModel):
    video_path: str = Field(description="Absolute or server-visible path of the target video.")
    perceptual_question: str = Field(
        description="A concrete perceptual question about the video's audio/visual content."
    )
    start_time: float | None = Field(
        default=None,
        description="Optional start time in seconds for a focused clip."
    )
    end_time: float | None = Field(
        default=None,
        description="Optional end time in seconds for a focused clip."
    )
    # modality_hint: str | None = Field(
    #     default=None,
    #     description="Optional hint: visual, audio, speech, audiovisual, temporal."
    # )


class TemporalGroundingInput(BaseModel):
    video_path: str = Field(description="Absolute or server-visible path of the target video.")
    target_question: str = Field(
        description="Question whose relevant evidence span should be grounded in the video."
    )


def _encode_video(video_path: str) -> tuple[str, str]:
    path = Path(video_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "video/mp4"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return mime_type, data


def _build_gemini_url() -> str:
    base_url = os.environ.get("GEMINI_BASE_URL", "https://yinli.one").rstrip("/")
    model = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
    return f"{base_url}/v1beta/models/{model}:generateContent"


def _gemini_headers() -> dict[str, str]:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _call_gemini_with_video(video_path: str, text_prompt: str) -> str:
    mime_type, video_b64 = _encode_video(video_path)
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inlineData": {"mimeType": mime_type, "data": video_b64}},
                    {"text": text_prompt},
                ],
            }
        ]
    }
    response = requests.post(
        _build_gemini_url(),
        headers=_gemini_headers(),
        json=payload,
        timeout=int(os.environ.get("GEMINI_TIMEOUT", "180")),
    )
    response.raise_for_status()
    data = response.json()
    candidate = data["candidates"][0]
    parts = candidate["content"]["parts"]
    text_chunks = [part["text"] for part in parts if "text" in part]
    return "\n".join(text_chunks).strip()


@tool(args_schema=PerceptionInput)
def ask_qwen_perception(
    video_path: str,
    perceptual_question: str,
    start_time: float | None = None,
    end_time: float | None = None,
    # modality_hint: str | None = None,
) -> str:
    """Ask perceptual backend a concrete AV question.

    Tool name remains qwen-prefixed for planner compatibility.
    """
    clip_note = ""
    if start_time is not None and end_time is not None:
        clip_note = f"Focus only on the clip from {start_time:.2f}s to {end_time:.2f}s."

    # modality_note = f"Primary modality: {modality_hint}." if modality_hint else ""

    prompt = f"""
        You are a perceptual audio-visual assistant.
        Answer ONLY from what is observable in the provided video/audio.
        Do not guess beyond the evidence.

        Video path: {video_path}
        {clip_note}

        Perceptual question: {perceptual_question}

        Return JSON with keys:
        - answer: short direct answer
        - evidence: list of concrete observations
        - uncertainty: short note about ambiguity / missing evidence, or empty string
        - used_time_range: object with start_time and end_time if relevant, else null
        """.strip()

    return _call_gemini_with_video(video_path, prompt)


@tool(args_schema=TemporalGroundingInput)
def temporal_ground_video(video_path: str, target_question: str) -> str:
    """Ask perceptual backend to identify relevant time spans."""
    prompt = f"""
        You are a temporal grounding assistant for video question answering.
        Given the question, decide whether temporal grounding is helpful.
        If not helpful, return <no_grounding>.
        If helpful, return one or more relevant spans in seconds.

        Video path: {video_path}
        Question: {target_question}

        Output format:
        <grounding>[{{"start": 0.0, "end": 3.2}}, {{"start": 7.4, "end": 11.0}}]</grounding>
        Then provide one-sentence justification.
        """.strip()

    return _call_gemini_with_video(video_path, prompt)


TOOLS = [ask_qwen_perception, temporal_ground_video]
