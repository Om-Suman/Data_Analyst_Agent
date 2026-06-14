import requests
import streamlit as st

HF_TOKEN = st.session_state.get("hf_api_key", "")

response = requests.post(
    "https://router.huggingface.co/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    },
    json={
        "model": "deepseek-ai/DeepSeek-R1",
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    },
    timeout=60
)

print(response.status_code)
print(response.text)