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
    audio_path: str | None = Field(
        default=None,
        description="Absolute or server-visible path of the target audio.",
    )
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
    audio_path: str | None = Field(
        default=None,
        description="Absolute or server-visible path of the target audio.",
    )
    perceptual_question: str = Field(
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


def _build_qwen_url() -> str:
    base_url = os.environ.get("QWEN_BASE_URL", "http://29.232.225.71:8000").rstrip("/")
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


def _gemini_headers() -> dict[str, str]:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _qwen_headers() -> dict[str, str]:
    api_key = os.environ.get("QWEN_API_KEY", "").strip()
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.upper() != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


# def _to_video_data_url(video_path: str) -> str:
#     mime_type, video_b64 = _encode_video(video_path)
#     return f"data:{mime_type};base64,{video_b64}"
def _to_data_url(path: str) -> str:
    suffix = Path(path).suffix.lower()
    mime_overrides = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".flac": "audio/flac",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
    }
    mime_type = mime_overrides.get(suffix) or mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{data}"

def _call_qwen_with_video(video_path: str, audio_path: str | None, text_prompt: str) -> str:
    if not audio_path:
        raise ValueError("audio_path is required for qwen perception calls.")
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a perceptual audio-visual assistant.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": _to_data_url(audio_path),
                        },
                    },
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": _to_data_url(video_path),
                        },
                    },
                    {"type": "text", "text": text_prompt},
                ],
            },
        ],
    }
    qwen_model = os.environ.get("QWEN_MODEL", "").strip()
    if qwen_model and qwen_model.upper() != "EMPTY":
        payload["model"] = qwen_model
    response = requests.post(
        _build_qwen_url(),
        headers=_qwen_headers(),
        json=payload,
        timeout=int(os.environ.get("QWEN_TIMEOUT", "300")),
    )
    if not response.ok:
        body_preview = response.text[:1200]
        raise RuntimeError(
            "Qwen request failed. "
            f"status={response.status_code}, url={response.url}, "
            f"audio_path={audio_path!r}, video_path={video_path!r}, "
            f"content_type={response.headers.get('Content-Type', '')!r}, "
            f"body_preview={body_preview!r}"
        )
    data = response.json()
    return str(data["choices"][0]["message"]["content"]).strip()


def _selected_perception_model() -> str:
    backend = os.environ.get("PERCEPTION_MODEL", "gemini").strip().lower()
    return backend or "gemini"


def _call_perception_with_video(
    video_path: str,
    audio_path: str | None,
    text_prompt: str,
) -> str:
    backend = _selected_perception_model()
    if backend == "qwen":
        return _call_qwen_with_video(video_path, audio_path, text_prompt)
    return _call_gemini_with_video(video_path, text_prompt)


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
    audio_path: str | None = None,
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

    return _call_perception_with_video(video_path, audio_path, prompt)


@tool(args_schema=TemporalGroundingInput)
def temporal_ground_video(
    video_path: str,
    perceptual_question: str,
    audio_path: str | None = None,
) -> str:
    """Ask perceptual backend to identify relevant time spans."""
    prompt = f"""
        You are a temporal grounding assistant for video question answering.
        Given the question, decide whether temporal grounding is helpful.
        If not helpful, return <no_grounding>.
        If helpful, return one or more relevant spans in seconds.

        Video path: {video_path}
        Question: {perceptual_question}

        Output format:
        <grounding>[{{"start": 0.0, "end": 3.2}}, {{"start": 7.4, "end": 11.0}}]</grounding>
        Then provide one-sentence justification.
        """.strip()

    return _call_perception_with_video(video_path, audio_path, prompt)


TOOLS = [ask_qwen_perception, temporal_ground_video]
