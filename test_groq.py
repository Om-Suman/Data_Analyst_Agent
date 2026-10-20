import os
from modules.llm_client import query_llm

response, model = query_llm(
    system_prompt="You are a helpful data analyst.",
    user_prompt="Say hello and confirm you are working.",
    max_tokens=100,
)
print(f"Model used: {model}")
print(f"Response: {response}")

