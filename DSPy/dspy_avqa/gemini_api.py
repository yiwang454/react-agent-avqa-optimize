from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import requests

from .response_quality import degenerate_response_reason

API_KEY = os.getenv("GEMINI_API_KEY", "")
BASE_URL = os.getenv("GEMINI_BASE_URL", "https://api.apiplus.org")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
PROVIDER = os.getenv("GEMINI_PROVIDER", "apiplus")
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


def _retry_delay(retry_delay_s: float, attempt: int) -> float:
    """Return exponential backoff after a failed 1-based attempt."""
    return retry_delay_s * (2 ** (attempt - 1))



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

def _service_account_access_token(credentials_path: str) -> str:
    """Create an OAuth access token from a service-account JSON file."""
    try:
        from google.auth.transport.requests import Request as GoogleAuthRequest
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        credentials.refresh(GoogleAuthRequest())
        return str(credentials.token)
    except ImportError:
        pass

    try:
        import jwt
    except ImportError as exc:
        raise RuntimeError(
            "Vertex ADC authentication requires google-auth or PyJWT; alternatively set "
            "GEMINI_API_KEY to a Google Cloud authorization key."
        ) from exc

    with open(credentials_path, "r", encoding="utf-8") as f:
        service_account_info = json.load(f)
    now = int(time.time())
    assertion = jwt.encode(
        {
            "iss": service_account_info["client_email"],
            "scope": "https://www.googleapis.com/auth/cloud-platform",
            "aud": service_account_info.get("token_uri", "https://oauth2.googleapis.com/token"),
            "iat": now,
            "exp": now + 3600,
        },
        service_account_info["private_key"],
        algorithm="RS256",
    )
    token_response = requests.post(
        service_account_info.get("token_uri", "https://oauth2.googleapis.com/token"),
        data={
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": assertion,
        },
        timeout=30,
    )
    if not token_response.ok:
        raise RuntimeError(
            f"Google OAuth token error {token_response.status_code}: {token_response.text[:1000]}"
        )
    return str(token_response.json()["access_token"])


def _vertex_headers(api_key: str, auth_mode: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        if auth_mode == "bearer":
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers["x-goog-api-key"] = api_key
        return headers

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not credentials_path:
        raise RuntimeError(
            "Vertex authentication requires GEMINI_API_KEY or GOOGLE_APPLICATION_CREDENTIALS."
        )
    headers["Authorization"] = f"Bearer {_service_account_access_token(credentials_path)}"
    return headers


def _request_url_and_headers(
    provider: str,
    base_url: str,
    model: str,
    api_key: str,
    auth_mode: str,
) -> Tuple[str, Dict[str, str]]:
    if provider == "vertex":
        url = f"{base_url}/models/{urllib.parse.quote(model, safe='')}:generateContent"
        return url, _vertex_headers(api_key, auth_mode)
    if provider != "apiplus":
        raise ValueError(f"Unsupported GEMINI_PROVIDER: {provider!r}")
    url = f"{base_url}/v1beta/models/{urllib.parse.quote(model, safe='')}:generateContent"
    return url, {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def call_gemini_messages(
    contents: List[Dict[str, Any]],
    *,
    system_prompt: str = "",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    provider: Optional[str] = None,
    auth_mode: Optional[str] = None,
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
    retry_degenerate_response: bool = False,
) -> Tuple[str, Dict[str, Any]] | Tuple[str, Dict[str, Any], str]:
    resolved_model = model or os.getenv("GEMINI_MODEL", MODEL)
    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY", API_KEY)
    resolved_base_url = (base_url or os.getenv("GEMINI_BASE_URL", BASE_URL)).rstrip("/")
    resolved_provider = (provider or os.getenv("GEMINI_PROVIDER", PROVIDER)).strip().lower()
    resolved_auth_mode = (auth_mode or os.getenv("GEMINI_AUTH_MODE", "api_key")).strip().lower()
    url, headers = _request_url_and_headers(
        resolved_provider,
        resolved_base_url,
        resolved_model,
        resolved_api_key,
        resolved_auth_mode,
    )
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
            degenerate_reason = (
                degenerate_response_reason(response_text, token_usage, max_tokens=max_tokens)
                if retry_degenerate_response
                else None
            )
            if degenerate_reason and attempt < max_retries:
                delay_s = _retry_delay(retry_delay_s, attempt)
                print(
                    f"[warn] Gemini API attempt {attempt}/{max_retries} returned a degenerate "
                    f"perception response: {degenerate_reason}. Retrying in {delay_s}s.",
                    flush=True,
                )
                time.sleep(delay_s)
                continue
            if degenerate_reason:
                print(
                    f"[warn] Gemini API exhausted {max_retries} attempts after a degenerate "
                    f"perception response: {degenerate_reason}. Keeping the final response.",
                    flush=True,
                )
            if return_thinking:
                return response_text, token_usage, thinking_text
            return response_text, token_usage
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                delay_s = _retry_delay(retry_delay_s, attempt)
                print(
                    f"[warn] Gemini API attempt {attempt}/{max_retries} failed: {_short_error(exc)}. "
                    f"Retrying in {delay_s}s.",
                    flush=True,
                )
                time.sleep(delay_s)

    if last_err is not None:
        raise RuntimeError(f"Gemini API call failed: {_short_error(last_err, limit=2000)}") from last_err
    raise RuntimeError("Gemini API call failed.")
