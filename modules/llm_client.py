"""
LLM Client — Hugging Face Inference Providers
Primary: Qwen/Qwen3-32B
Fallback: deepseek-ai/DeepSeek-R1
"""

import os
import requests
import time
import json
import re
from typing import Optional

HF_URL = "https://router.huggingface.co/v1/chat/completions"
PRIMARY_MODEL = "deepseek-ai/DeepSeek-R1"
FALLBACK_MODEL = ""
FATAL_STATUS_CODES = {400, 401, 402, 403, 404}


def _load_key_from_files() -> str:
    """Scan local .env for HF_API_KEY."""
    for candidate in [".env", "../.env"]:
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("HF_API_KEY") and "=" in line:
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                return val
            except Exception:
                pass
    return ""


def _resolve_api_key(api_key: Optional[str] = None) -> str:
    """Resolve API key from argument, environment, or disk configuration."""
    if api_key and str(api_key).strip():
        return str(api_key).strip()
    env_key = os.environ.get("HF_API_KEY", "").strip()
    if env_key:
        return env_key
    file_key = _load_key_from_files()
    if file_key:
        return file_key
    return ""


def _resolve_models(primary_model: Optional[str] = None, fallback_model: Optional[str] = None) -> list[str]:
    """Return configured models in retry order."""
    p = (primary_model or os.environ.get("PRIMARY_MODEL", "")).strip() or PRIMARY_MODEL
    f = (fallback_model or os.environ.get("FALLBACK_MODEL", "")).strip()
    models = []
    for m in (p, f):
        m = str(m or "").strip()
        if m and m not in models:
            models.append(m)
    return models or [PRIMARY_MODEL]


def clean_response(text: str) -> str:
    """
    Remove reasoning traces and clean formatting.
    """

    # Remove Qwen think blocks
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Some reasoning models can be truncated before emitting </think>. Remove
    # the unfinished trace as well so it is never shown to the user.
    text = re.sub(
        r"<think>.*$",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _call_hf(
    model: str,
    messages: list,
    max_tokens: int,
    temperature: float,
    timeout: int,
    api_key: Optional[str] = None,
) -> str:

    resolved_key = _resolve_api_key(api_key)

    if not resolved_key:
        raise ValueError("No Hugging Face API key configured.")

    headers = {
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "extra_body": {
            "reasoning": False
        }
    }

    resp = requests.post(
        HF_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
    )

    if not resp.ok:
        http_error = requests.exceptions.HTTPError(
            f"HTTP {resp.status_code}: {resp.text}"
        )
        http_error.response = resp
        raise http_error

    data = resp.json()

    if "choices" in data and data["choices"]:

        message = data["choices"][0]["message"]

        # Ignore DeepSeek reasoning_content
        content = message.get("content", "")

        return clean_response(content)

    raise ValueError(f"Unexpected response: {data}")


def _friendly_http_error(status: int, body: str) -> str:
    if status == 402:
        return (
            "Hugging Face Inference Providers rejected the request because "
            "your monthly included credits are depleted. Add prepaid credits, "
            "upgrade to Pro, or switch to a local/free provider before running AI queries."
        )
    if status == 401:
        return "Hugging Face rejected the API key. Check or replace your HF_API_KEY."
    if status == 403:
        return "Hugging Face denied access to this model/provider for your account."
    if status == 404:
        return "The configured Hugging Face model was not found. Check the model name in Settings."
    return f"HTTP {status}: {body}"


_DEPLETED_KEYS: set[str] = set()

def reset_depleted_keys():
    _DEPLETED_KEYS.clear()


def query_llm(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    retries: int = 3,
    timeout: int = 20,
    api_key: Optional[str] = None,
    primary_model: Optional[str] = None,
    fallback_model: Optional[str] = None,
) -> tuple[str, str]:
    """
    Returns:
        (response_text, model_used)
    """
    resolved_key = _resolve_api_key(api_key)
    if not resolved_key:
        return ("❌ LLM error: No API key configured.", "none")

    if resolved_key in _DEPLETED_KEYS:
        return ("❌ LLM error: " + _friendly_http_error(402, ""), "none")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    models = _resolve_models(primary_model, fallback_model)
    last_error = ""

    for model in models:

        for attempt in range(retries):

            try:

                text = _call_hf(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    api_key=api_key,
                )

                return text, model

            except requests.exceptions.HTTPError as e:

                status = (
                    e.response.status_code
                    if e.response
                    else 0
                )
                body = e.response.text if e.response is not None else str(e)

                if status == 429:

                    wait = 2 ** attempt
                    time.sleep(wait)
                    continue

                elif status in FATAL_STATUS_CODES:

                    if status == 402:
                        _DEPLETED_KEYS.add(resolved_key)
                    last_error = _friendly_http_error(status, body)
                    return (
                        f"❌ LLM error: {last_error}",
                        "none",
                    )

                elif status in (500, 503):

                    last_error = (
                        f"Model {model} unavailable "
                        f"(HTTP {status})"
                    )
                    break

                else:

                    last_error = str(e)
                    break

            except requests.exceptions.Timeout:

                last_error = (
                    f"Timeout on {model} "
                    f"(attempt {attempt + 1})"
                )

                if attempt == retries - 1:
                    break

                time.sleep(2)

            except Exception as e:

                last_error = str(e)
                break

    return (
        f"❌ LLM error after all retries: {last_error}",
        "none",
    )


def extract_python_code(text: str) -> list[str]:
    """
    Extract all python code blocks from LLM response.
    """

    blocks = re.findall(
        r"```python\s*(.*?)```",
        text,
        re.DOTALL,
    )

    return [
        block.strip()
        for block in blocks
        if block.strip()
    ]


def extract_json(text: str) -> dict | list | None:
    """
    Try to extract JSON from LLM response.
    """

    blocks = re.findall(
        r"```json\s*(.*?)```",
        text,
        re.DOTALL,
    )

    for block in blocks:

        try:
            return json.loads(block.strip())
        except Exception:
            pass

    try:

        start = text.index("{")
        end = text.rindex("}") + 1

        return json.loads(text[start:end])

    except Exception:
        pass

    return None
