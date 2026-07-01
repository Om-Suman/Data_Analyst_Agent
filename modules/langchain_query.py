"""LangChain-backed orchestration for dataframe questions."""

from __future__ import annotations

import pandas as pd

from modules.executor import execute_code
from modules.llm_client import extract_python_code, query_llm
from modules.query_engine import (
    REPAIR_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_context,
    build_repair_prompt,
    generate_final_insights,
)

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda

    LANGCHAIN_AVAILABLE = True
except Exception:
    ChatPromptTemplate = None
    RunnableLambda = None
    LANGCHAIN_AVAILABLE = False


def langchain_available() -> bool:
    return LANGCHAIN_AVAILABLE


def _build_codegen_chain():
    if not LANGCHAIN_AVAILABLE:
        return None

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            ("user", "{user_prompt}"),
        ]
    )

    def _call_model(prompt_value):
        messages = prompt_value.to_messages()
        system_prompt = messages[0].content
        user_prompt = messages[1].content
        response, model = query_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=2048,
            temperature=0.3,
        )
        return {"response": response, "model": model}

    return prompt | RunnableLambda(_call_model)


def _generate_code(df: pd.DataFrame, question: str, history: list | None, max_tokens: int):
    prompt_text = build_context(df, question, history)

    if not LANGCHAIN_AVAILABLE:
        response, model_used = query_llm(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt_text,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response, model_used, False

    chain = _build_codegen_chain()
    payload = chain.invoke(
        {
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": prompt_text,
        }
    )
    return payload["response"], payload["model"], True


def run_query_langchain(
    df: pd.DataFrame,
    question: str,
    history: list | None = None,
    max_tokens: int = 2048,
) -> dict:
    """Run the dataframe question pipeline using LangChain for orchestration."""
    result = {
        "question": question,
        "llm_response": "",
        "code_generation_response": "",
        "code_blocks": [],
        "execution_results": [],
        "insights": "",
        "model_used": "",
        "error": None,
        "orchestrator": "langchain" if LANGCHAIN_AVAILABLE else "fallback",
    }

    response, model_used, _ = _generate_code(df, question, history, max_tokens)
    result["code_generation_response"] = response
    result["llm_response"] = ""
    result["model_used"] = model_used

    if response.startswith("❌"):
        result["error"] = response
        return result

    code_blocks = extract_python_code(response)
    result["code_blocks"] = code_blocks

    exec_results = []
    repaired_any = False
    final_code_blocks = []

    for code in code_blocks:
        exec_result = execute_code(code, df)
        final_code = code

        if exec_result.error:
            repair_prompt = build_repair_prompt(
                df=df,
                question=question,
                failed_code=code,
                error=exec_result.error,
            )
            repair_response, repair_model = query_llm(
                system_prompt=REPAIR_SYSTEM_PROMPT,
                user_prompt=repair_prompt,
                max_tokens=max_tokens,
                temperature=0.1,
            )
            repaired_blocks = extract_python_code(repair_response)
            if repaired_blocks:
                repaired_result = execute_code(repaired_blocks[0], df)
                if repaired_result.success:
                    exec_result = repaired_result
                    final_code = repaired_blocks[0]
                    repaired_any = True
                    result["model_used"] = f"{model_used} + repair:{repair_model}"

        final_code_blocks.append(final_code)
        exec_results.append(exec_result)

    if repaired_any:
        result["code_blocks"] = final_code_blocks
        repair_note = (
            "\n\n## Repair Note\n"
            "A generated code block raised an execution error, so it was automatically corrected and re-run."
        )

    result["execution_results"] = exec_results

    final_insights, insights_model = generate_final_insights(
        question=question,
        code_blocks=final_code_blocks,
        exec_results=exec_results,
        repaired_any=repaired_any,
        max_tokens=max_tokens,
    )
    if repaired_any:
        final_insights = (final_insights + repair_note).strip()

    result["insights"] = final_insights
    result["llm_response"] = final_insights
    if result["model_used"]:
        result["model_used"] = f"{result['model_used']} + insights:{insights_model}"
    else:
        result["model_used"] = f"insights:{insights_model}"

    return result