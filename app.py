import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mimetypes
import docx
import fitz  # PyMuPDF
from PIL import Image
from dotenv import load_dotenv
import pytesseract
import requests
import json
import re
import time
import io
import sys
import os
# --- Config ---
load_dotenv()
TOGETHER_API_KEY = os.getenv("MY_API_KEY") 
LLAMA_MODEL_ID = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

# --- File Parser ---
def parse_file(uploaded_file):
    suffix = uploaded_file.name.split(".")[-1].lower()
    if suffix == "csv":
        return pd.read_csv(uploaded_file), "dataframe"
    elif suffix == "xlsx":
        return pd.read_excel(uploaded_file), "dataframe"
    elif suffix == "txt":
        return uploaded_file.read().decode(), "text"
    elif suffix == "docx":
        doc = docx.Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs]), "text"
    elif suffix == "pdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])
        return text, "text"
    elif suffix in ["png", "jpg", "jpeg"]:
        image = Image.open(uploaded_file)
        return pytesseract.image_to_string(image), "text"
    else:
        return "Unsupported file type", None

# --- LLM Call ---
def query_llama_agent(prompt, retries=3):
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": LLAMA_MODEL_ID,
        "prompt": prompt,
        "max_tokens": 1024,
        "temperature": 0.7,
    }

    for attempt in range(retries):
        response = requests.post("https://api.together.xyz/v1/completions", headers=headers, json=payload)
        try:
            result = response.json()
        except:
            return "❌ Invalid response from API"

        if response.status_code == 200 and "choices" in result:
            return result["choices"][0]["text"].strip()
        elif response.status_code == 429:
            time.sleep(100)
            continue
        else:
            return f"❌ API Error: {result}"
    return "❌ LLM unavailable after retries"

# --- Execute Code Blocks and Capture Output ---
def extract_and_run_code(response_text, df):
    code_blocks = re.findall(r"```python(.*?)```", response_text, re.DOTALL)

    for i, code in enumerate(code_blocks):
        code = re.sub(r"df\s*=\s*pd\.read_csv\(.*?\)", "", code.strip())
        st.code(code.strip(), language="python")
        try:
            # Capture stdout to display print outputs in the app
            buffer = io.StringIO()
            sys.stdout = buffer

            exec_globals = {"pd": pd, "plt": plt, "df": df, "np": __import__('numpy')}
            exec(code.strip(), exec_globals)

            # Reset stdout
            sys.stdout = sys.__stdout__
            printed_output = buffer.getvalue()

            if printed_output:
                st.subheader("🖨️ Output")
                st.text(printed_output.strip())

            fig = plt.gcf()
            if fig.axes:
                st.subheader("📊 Visual Output")
                st.pyplot(fig)
            plt.clf()

        except Exception as e:
            sys.stdout = sys.__stdout__
            st.warning(f"⚠️ Could not execute code block #{i+1}: {e}")

# --- Streamlit UI ---
st.title("📊 Data Analyst Agent")
uploaded_file = st.file_uploader("Upload a file (.csv, .xlsx, .pdf, .txt, .docx, image)", type=["csv", "xlsx", "pdf", "txt", "docx", "png", "jpg", "jpeg"])

if uploaded_file:
    parsed_data, data_type = parse_file(uploaded_file)

    if data_type == "dataframe":
        st.subheader("🔍 Data Preview")
        st.dataframe(parsed_data.head())

    elif data_type == "text":
        st.subheader("📄 Document Preview")
        st.text_area("Content", parsed_data[:2000], height=300)

    st.subheader("💬 Ask a Question")
    question = st.text_input("Type your question")

    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            if data_type == "dataframe":
                context = parsed_data.head(10).to_markdown()
                prompt = f"""You are a data analyst.
Here is the dataset preview:
{context}

User's question: {question}
Answer in detail and include Python code if needed to support your analysis."""
            else:
                preview = parsed_data[:2000]
                prompt = f"""You are a document analysis assistant.
Here is the document excerpt:
\"\"\"{preview}\"\"\"

User's question: {question}
Answer in detail."""

            response = query_llama_agent(prompt)

        st.subheader("🧠 Full LLM Response")
        st.markdown(response)

        if data_type == "dataframe":
            extract_and_run_code(response, parsed_data)
        else:
            st.info("No tabular data available for code execution.")
