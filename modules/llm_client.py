"""
LLM Client — Hugging Face Inference Providers
Primary: Qwen/Qwen3-32B
Fallback: deepseek-ai/DeepSeek-R1
"""

import streamlit as st
import requests
import time
import json
import re

HF_URL = "https://router.huggingface.co/v1/chat/completions"
PRIMARY_MODEL = "deepseek-ai/DeepSeek-R1"
FALLBACK_MODEL = ""
FATAL_STATUS_CODES = {400, 401, 402, 403, 404}


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

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _call_hf(
    model: str,
    messages: list,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> str:

    api_key = st.session_state.get("hf_api_key", "")

    if not api_key:
        raise ValueError("No Hugging Face API key configured.")

    headers = {
        "Authorization": f"Bearer {api_key}",
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


def _get_model_chain() -> list[str]:
    """Return configured models in retry order, skipping blanks and duplicates."""
    primary = st.session_state.get("primary_model", PRIMARY_MODEL)
    fallback = st.session_state.get("fallback_model", FALLBACK_MODEL)

    models = []
    for model in (primary, fallback):
        model = str(model or "").strip()
        if model and model not in models:
            models.append(model)

    return models or [PRIMARY_MODEL]


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


def query_llm(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    retries: int = 3,
    timeout: int = 120,
) -> tuple[str, str]:
    """
    Returns:
        (response_text, model_used)
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    models = _get_model_chain()
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

                    last_error = _friendly_http_error(status, body)
                    return (
                        f"âŒ LLM error: {last_error}",
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
