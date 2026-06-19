from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


API_KEY = os.getenv("GEMINI_API_KEY", "")
BASE_URL = os.getenv("GEMINI_BASE_URL", "https://api.apiplus.org")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
_PRINTED_FIRST_PROMPT = False


def _mime_type(path: str) -> str:
    return mimetypes.guess_type(path)[0] or "application/octet-stream"


def _inline_data_from_path(path: str, mime_type: Optional[str] = None) -> Dict[str, Any]:
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return {
        "inlineData": {
            "mimeType": mime_type or _mime_type(path),
            "data": data,
        }
    }


def extract_choice_letter(text: str) -> str:
    if not text:
        return ""
    patterns = [
        r"answer\s*(?:is|:)\s*([ABCD])\b",
        r"final\s*answer\s*(?:is|:)?\s*([ABCD])\b",
        r"选择\s*[:：]?\s*([ABCD])\b",
        r"答案\s*[:：]?\s*([ABCD])\b",
        r"option\s*([ABCD])\b",
        r"^\s*([ABCD])\s*$",
        r"(?<![A-Z])([ABCD])(?![A-Z])",
    ]
    text = text.strip()
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.I)
        if matches:
            return matches[-1].upper()
    return ""

def extract_last_option(response: str) -> str:
    response = response.strip()
    if not response:
        return ""
    last_line = response.split("\n")[-1].strip()
    # .splitlines()[-1].strip()
    answer_match = re.search(
        r"(?i:\b(?:final\s+answer|correct\s+answer|answer)\b\s*(?:is|:)?\s*\**)\s*([A-F])\b",
        last_line,
    )
    if answer_match:
        return answer_match.group(1).upper()
    matches = re.findall(r"\b([A-F])\b", last_line)
    if not matches:
        return ""
    return matches[-1].upper()


def _response_text_parts(resp_json: Dict[str, Any]) -> Tuple[str, str]:
    candidates = resp_json.get("candidates") or []
    if not candidates:
        return "", ""
    parts = candidates[0].get("content", {}).get("parts", [])
    answer_texts: List[str] = []
    thought_texts: List[str] = []
    for part in parts:
        text = part.get("text") if isinstance(part, dict) else None
        if isinstance(text, str):
            if part.get("thought"):
                thought_texts.append(text)
            else:
                answer_texts.append(text)
    return "".join(answer_texts), "".join(thought_texts)


def _response_text(resp_json: Dict[str, Any]) -> str:
    # necessary for parsing _response_text
    return _response_text_parts(resp_json)[0]


def _usage(resp_json: Dict[str, Any]) -> Dict[str, Any]:
    usage = resp_json.get("usageMetadata")
    return usage if isinstance(usage, dict) else {}


def _short_error(exc: Exception, limit: int = 500) -> str:
    text = str(exc).replace("\n", " ")
    if len(text) > limit:
        return text[:limit] + "..."
    return text



def _env_flag(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _redact_inline_data(value: Any) -> Any:
    if isinstance(value, list):
        return [_redact_inline_data(item) for item in value]
    if isinstance(value, dict):
        redacted: Dict[str, Any] = {}
        for key, item in value.items():
            if key == "data" and isinstance(item, str):
                redacted[key] = f"<base64 redacted; {len(item)} chars>"
            else:
                redacted[key] = _redact_inline_data(item)
        return redacted
    return value


def _maybe_print_first_prompt(data: Dict[str, Any]) -> None:
    global _PRINTED_FIRST_PROMPT
    if _PRINTED_FIRST_PROMPT or not _env_flag("GEMINI_PRINT_FIRST_PROMPT"):
        return
    print("[Gemini first prompt]", flush=True)
    print(json.dumps(_redact_inline_data(data), ensure_ascii=False, indent=2), flush=True)
    _PRINTED_FIRST_PROMPT = True

def call_gemini_messages(
    contents: List[Dict[str, Any]],
    *,
    system_prompt: str = "",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 180,
    max_retries: int = 3,
    retry_delay_s: float = 5.0,
    include_thoughts: bool = False,
    return_thinking: bool = False,
    temperature: float = 0.6,
    gemini_seed: Optional[int] = None,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 1024,
) -> Tuple[str, Dict[str, Any]] | Tuple[str, Dict[str, Any], str]:
    resolved_model = model or os.getenv("GEMINI_MODEL", MODEL)
    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY", API_KEY)
    resolved_base_url = (base_url or os.getenv("GEMINI_BASE_URL", BASE_URL)).rstrip("/")
    url = f"{resolved_base_url}/v1beta/models/{resolved_model}:generateContent"
    headers = {
        "Authorization": f"Bearer {resolved_api_key}",
        "Content-Type": "application/json",
    }
    data: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p,
            "topK": top_k,
            "maxOutputTokens": max_tokens,
        },
    }
    if gemini_seed is not None:
        data["generationConfig"]["seed"] = gemini_seed
    if system_prompt:
        data["systemInstruction"] = {"parts": [{"text": system_prompt}]}
    if include_thoughts:
        data["generationConfig"]["thinkingConfig"] = {"includeThoughts": True}
    _maybe_print_first_prompt(data)

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=timeout)
            if not resp.ok:
                raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:2000]}")
            resp_json = resp.json()
            response_text, thinking_text = _response_text_parts(resp_json)
            token_usage = _usage(resp_json)
            if return_thinking:
                return response_text, token_usage, thinking_text
            return response_text, token_usage
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                print(
                    f"[warn] Gemini API attempt {attempt}/{max_retries} failed: {_short_error(exc)}. "
                    f"Retrying in {retry_delay_s}s.",
                    flush=True,
                )
                time.sleep(retry_delay_s)

    if last_err is not None:
        raise RuntimeError(f"Gemini API call failed: {_short_error(last_err, limit=2000)}") from last_err
    raise RuntimeError("Gemini API call failed.")
