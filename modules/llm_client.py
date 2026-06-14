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
PRIMARY_MODEL = "Qwen/Qwen3-32B"
FALLBACK_MODEL = "deepseek-ai/DeepSeek-R1"


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
        timeout=60,
    )

    if not resp.ok:
        raise Exception(
            f"HTTP {resp.status_code}: {resp.text}"
        )

    data = resp.json()

    if "choices" in data and data["choices"]:

        message = data["choices"][0]["message"]

        # Ignore DeepSeek reasoning_content
        content = message.get("content", "")

        return clean_response(content)

    raise ValueError(f"Unexpected response: {data}")


def query_llm(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    retries: int = 3,
) -> tuple[str, str]:
    """
    Returns:
        (response_text, model_used)
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    models = [PRIMARY_MODEL, FALLBACK_MODEL]
    last_error = ""

    for model in models:

        for attempt in range(retries):

            try:

                text = _call_hf(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                return text, model

            except requests.exceptions.HTTPError as e:

                status = (
                    e.response.status_code
                    if e.response
                    else 0
                )

                if status == 429:

                    wait = 2 ** attempt
                    time.sleep(wait)
                    continue

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

