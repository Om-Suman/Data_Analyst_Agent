"""Document QA page backed by LlamaIndex retrieval."""

import streamlit as st

from modules.document_rag import answer_document_question, get_document_bundle


def render():
    st.title("📚 Document QA")

    active_name = st.session_state.get("active_dataset")
    if not active_name:
        st.warning("Load a text document first.")
        return

    dataset = st.session_state.get("datasets", {}).get(active_name, {})
    text_content = dataset.get("text_content")

    if not text_content:
        st.warning("The active dataset is not a text document. Upload a PDF, DOCX, TXT, or OCR image first.")
        return

    bundle = get_document_bundle(active_name)
    if bundle:
        st.caption(f"Index engine: {bundle.get('engine', 'unknown')}")
    else:
        st.caption("No stored index found yet. It will be created when you ask a question.")

    question = st.text_area(
        "Ask a question about the document",
        placeholder="What are the main risks mentioned in this report?",
        height=90,
    )

    if st.button("🔎 Ask Document", type="primary") and question.strip():
        with st.spinner("Searching document..."):
            answer = answer_document_question(
                question=question,
                document_name=active_name,
                max_tokens=st.session_state.get("max_tokens", 1024),
            )

        if answer.get("error"):
            st.error(answer["error"])

        st.markdown("### Answer")
        st.markdown(answer.get("answer", "No answer returned."))

        sources = answer.get("sources", [])
        if sources:
            with st.expander("Retrieved Context", expanded=False):
                for idx, source in enumerate(sources, start=1):
                    st.markdown(f"**Chunk {idx}**")
                    st.write(source)