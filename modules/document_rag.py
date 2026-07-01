"""LlamaIndex-backed document question answering helpers."""

from __future__ import annotations

from difflib import SequenceMatcher

import streamlit as st

from modules.llm_client import query_llm

try:
    from llama_index.core import Document, Settings, VectorStoreIndex

    try:
        from llama_index.core.embeddings import MockEmbedding
    except Exception:
        MockEmbedding = None

    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except Exception:
        HuggingFaceEmbedding = None

    LLAMAINDEX_AVAILABLE = True
except Exception:
    Document = None
    Settings = None
    VectorStoreIndex = None
    MockEmbedding = None
    HuggingFaceEmbedding = None
    LLAMAINDEX_AVAILABLE = False


DOCUMENT_SYSTEM_PROMPT = """You answer questions strictly from the provided document context.

Rules:
- Use only the supplied context.
- If the context does not contain the answer, say so clearly.
- Keep the answer concise and grounded.
- Do not invent facts or cite information that is not in the context.
"""


def llamaindex_available() -> bool:
    return LLAMAINDEX_AVAILABLE


def _split_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(text_length, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_document_bundle(text: str, metadata: dict | None = None) -> dict:
    metadata = metadata or {}
    chunks = _split_text(text)

    bundle = {
        "text": text,
        "metadata": metadata,
        "chunks": chunks,
        "index": None,
        "engine": "fallback",
    }

    if not LLAMAINDEX_AVAILABLE or not chunks:
        return bundle

    try:
        if HuggingFaceEmbedding is not None:
            Settings.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        elif MockEmbedding is not None:
            Settings.embed_model = MockEmbedding(embed_dim=384)

        documents = [
            Document(text=chunk, metadata={**metadata, "chunk_index": idx + 1})
            for idx, chunk in enumerate(chunks)
        ]
        index = VectorStoreIndex.from_documents(documents)
        bundle["index"] = index
        bundle["engine"] = "llamaindex"
    except Exception:
        bundle["index"] = None
        bundle["engine"] = "fallback"

    return bundle


def store_document_bundle(name: str, bundle: dict) -> dict:
    st.session_state.document_indexes[name] = bundle
    return bundle


def get_document_bundle(name: str) -> dict | None:
    return st.session_state.get("document_indexes", {}).get(name)


def _fallback_retrieve(question: str, chunks: list[str], top_k: int = 4) -> list[str]:
    scored = []
    question_words = set(question.lower().split())

    for chunk in chunks:
        chunk_lower = chunk.lower()
        overlap = sum(1 for word in question_words if word in chunk_lower)
        score = overlap + SequenceMatcher(None, question_lower := question.lower(), chunk_lower).ratio()
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k] if chunk.strip()]


def _format_context(chunks: list[str]) -> str:
    lines = []
    for idx, chunk in enumerate(chunks, start=1):
        lines.append(f"Chunk {idx}:\n{chunk.strip()}")
    return "\n\n".join(lines)


def answer_document_question(
    question: str,
    document_name: str,
    max_tokens: int = 1024,
) -> dict:
    bundle = get_document_bundle(document_name)
    if not bundle:
        return {
            "answer": "No document index is available for the active dataset.",
            "sources": [],
            "engine": "missing",
            "error": "document bundle not found",
        }

    retrieved_chunks: list[str] = []
    engine = bundle.get("engine", "fallback")

    try:
        index = bundle.get("index")
        if index is not None:
            retriever = index.as_retriever(similarity_top_k=4)
            nodes = retriever.retrieve(question)
            for node in nodes:
                try:
                    text = node.node.get_content(metadata_mode="none")
                except Exception:
                    text = getattr(node.node, "text", "") or str(node)
                if text.strip():
                    retrieved_chunks.append(text.strip())
        else:
            retrieved_chunks = _fallback_retrieve(question, bundle.get("chunks", []), top_k=4)
    except Exception:
        retrieved_chunks = _fallback_retrieve(question, bundle.get("chunks", []), top_k=4)
        engine = "fallback"

    if not retrieved_chunks:
        return {
            "answer": "I could not find relevant text in the uploaded document.",
            "sources": [],
            "engine": engine,
            "error": None,
        }

    context = _format_context(retrieved_chunks)
    user_prompt = f"""QUESTION:
{question}

CONTEXT:
{context}
"""

    response, model_used = query_llm(
        system_prompt=DOCUMENT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=0.2,
        retries=1,
        timeout=60,
    )

    if response.startswith("❌") or not response.strip():
        response = (
            "I found relevant context, but the model call was not available. "
            "Here is the retrieved text:\n\n" + context
        )

    return {
        "answer": response.strip(),
        "sources": retrieved_chunks,
        "engine": engine,
        "model_used": model_used,
        "error": None,
    }