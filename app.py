import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mimetypes
import docx
import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path
import pytesseract
import requests
import json
import re
import time
import io
import sys
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration ---
st.set_page_config(
    page_title="Advanced Data Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'data_info' not in st.session_state:
    st.session_state.data_info = {}
if 'current_file_name' not in st.session_state:
    st.session_state.current_file_name = None

# --- Enhanced Configuration ---
try:
    api_key = st.secrets["API_KEY"]
except:
    api_key = st.sidebar.text_input("Enter API Key", type="password")
    if not api_key:
        st.warning("Please enter your API key to continue")
        st.stop()

LLAMA_MODEL_ID = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

# --- Enhanced File Parser ---
@st.cache_data
def parse_file(uploaded_file):
    """Parse uploaded file and return data with metadata"""
    try:
        suffix = uploaded_file.name.split(".")[-1].lower()
        file_size = uploaded_file.size
        
        if suffix == "csv":
            # Enhanced CSV parsing with error handling
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='latin-1')
            
            metadata = {
                'type': 'dataframe',
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'file_size': file_size,
                'missing_values': df.isnull().sum().to_dict()
            }
            return df, "dataframe", metadata
            
        elif suffix == "xlsx":
            df = pd.read_excel(uploaded_file)
            metadata = {
                'type': 'dataframe',
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'file_size': file_size,
                'missing_values': df.isnull().sum().to_dict()
            }
            return df, "dataframe", metadata
            
        elif suffix == "txt":
            content = uploaded_file.read().decode('utf-8')
            metadata = {
                'type': 'text',
                'length': len(content),
                'word_count': len(content.split()),
                'file_size': file_size
            }
            return content, "text", metadata
            
        elif suffix == "docx":
            doc = docx.Document(uploaded_file)
            content = "\n".join([p.text for p in doc.paragraphs])
            metadata = {
                'type': 'text',
                'paragraphs': len(doc.paragraphs),
                'length': len(content),
                'word_count': len(content.split()),
                'file_size': file_size
            }
            return content, "text", metadata
            
        elif suffix == "pdf":
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            content = "\n".join([page.get_text() for page in doc])
            metadata = {
                'type': 'text',
                'pages': len(doc),
                'length': len(content),
                'word_count': len(content.split()),
                'file_size': file_size
            }
            return content, "text", metadata
            
        elif suffix in ["png", "jpg", "jpeg"]:
            image = Image.open(uploaded_file)
            content = pytesseract.image_to_string(image)
            metadata = {
                'type': 'text',
                'image_size': image.size,
                'extracted_text_length': len(content),
                'file_size': file_size
            }
            return content, "text", metadata
            
        else:
            return "Unsupported file type", None, {}
            
    except Exception as e:
        st.error(f"Error parsing file: {str(e)}")
        return None, None, {}

# --- Enhanced LLM Call ---
def query_llama_agent(prompt, retries=3, max_tokens=1024):
    """Enhanced LLM query with better error handling"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": LLAMA_MODEL_ID,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    for attempt in range(retries):
        try:
            response = requests.post("https://api.together.xyz/v1/completions", headers=headers, json=payload, timeout=30)
            result = response.json()
            
            if response.status_code == 200 and "choices" in result:
                return result["choices"][0]["text"].strip()
            elif response.status_code == 429:
                st.warning(f"Rate limit hit, waiting... (attempt {attempt + 1})")
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                return f"❌ API Error: {result.get('error', 'Unknown error')}"
                
        except requests.exceptions.Timeout:
            st.warning(f"Request timed out (attempt {attempt + 1})")
            continue
        except Exception as e:
            st.error(f"Request failed: {str(e)}")
            continue
            
    return "❌ LLM unavailable after retries"

# --- Enhanced Code Execution ---
def extract_and_run_code(response_text, df):
    """Enhanced code execution with better error handling and visualization"""
    code_blocks = re.findall(r"```python(.*?)```", response_text, re.DOTALL)
    
    if not code_blocks:
        return
    
    st.subheader("🔧 Generated Code & Results")
    
    for i, code in enumerate(code_blocks):
        # Clean up code
        code = re.sub(r"df\s*=\s*pd\.read_csv\(.*?\)", "", code.strip())
        # Remove plt.show() calls as they cause warnings in Streamlit
        code = re.sub(r"plt\.show\(\)", "", code)
        code = code.strip()
        
        if not code:
            continue
            
        st.code(code, language="python")
        
        try:
            # Capture output
            buffer = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buffer
            
            # Clear any previous plots
            plt.clf()
            plt.close('all')
            
            # Setup execution environment - don't create fig/ax upfront
            exec_globals = {
                "pd": pd, 
                "np": np, 
                "plt": plt, 
                "sns": sns,
                "px": px,
                "go": go,
                "df": df,
                "st": st
            }
            
            # Execute code
            exec(code, exec_globals)
            
            # Restore stdout
            sys.stdout = old_stdout
            printed_output = buffer.getvalue()
            
            # Display text output
            if printed_output:
                st.subheader("📊 Output")
                st.text(printed_output.strip())
            
            # Check if matplotlib plots were created
            if plt.get_fignums():
                st.subheader("📈 Visualization")
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    st.pyplot(fig)
                    plt.close(fig)
            
        except Exception as e:
            sys.stdout = old_stdout
            st.error(f"❌ Execution error: {str(e)}")
            st.code(str(e))
        finally:
            # Clean up any remaining plots
            plt.close('all')

# --- Data Insights Generator ---
def generate_data_insights(df):
    """Generate automatic insights about the dataset"""
    insights = []
    
    # Basic info
    insights.append(f"📊 Dataset has {df.shape[0]:,} rows and {df.shape[1]} columns")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        insights.append(f"⚠️ Found {missing.sum():,} missing values across {(missing > 0).sum()} columns")
    
    # Data types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(numeric_cols) > 0:
        insights.append(f"🔢 {len(numeric_cols)} numeric columns: {', '.join(numeric_cols[:3])}{'...' if len(numeric_cols) > 3 else ''}")
    
    if len(categorical_cols) > 0:
        insights.append(f"📝 {len(categorical_cols)} categorical columns: {', '.join(categorical_cols[:3])}{'...' if len(categorical_cols) > 3 else ''}")
    
    # Duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        insights.append(f"🔄 Found {duplicate_count:,} duplicate rows")
    
    return insights

# --- Sidebar ---
with st.sidebar:
    st.header("🛠️ Configuration")
    
    # Model settings
    with st.expander("Model Settings"):
        max_tokens = st.slider("Max Tokens", 100, 2048, 1024)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    
    # Quick actions
    st.header("⚡ Quick Actions")
    if st.session_state.current_data is not None:
        if st.button("📈 Quick Data Summary"):
            st.session_state.chat_history.append({
                'question': 'Generate a comprehensive data summary',
                'timestamp': datetime.now(),
                'type': 'quick_action'
            })
        
        if st.button("🔍 Find Correlations"):
            st.session_state.chat_history.append({
                'question': 'Find and visualize correlations in the data',
                'timestamp': datetime.now(),
                'type': 'quick_action'
            })
    
    # Chat history
    if st.session_state.chat_history:
        st.header("💬 Chat History")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-8:])):  # Show last 8
            timestamp = chat['timestamp'].strftime("%H:%M")
            chat_preview = f"👤: {chat['question'][:35]}..."
            if st.button(f"[{timestamp}] {chat_preview}", key=f"history_{i}"):
                st.session_state.selected_question = chat['question']
    
    # Clear history button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()

# --- Main Interface ---
st.title("📊 Advanced Data Analyst Agent")
st.markdown("Upload your data and ask questions to get insights, analysis, and visualizations!")

# File upload
uploaded_file = st.file_uploader(
    "Upload a file", 
    type=["csv", "xlsx", "pdf", "txt", "docx", "png", "jpg", "jpeg"],
    help="Supported formats: CSV, Excel, PDF, Text, Word documents, and images"
)

if uploaded_file:
    # Check if this is a new file
    if st.session_state.current_file_name != uploaded_file.name:
        st.session_state.chat_history = []  # Clear history for new file
        st.session_state.current_file_name = uploaded_file.name
    
    with st.spinner("Processing file..."):
        parsed_data, data_type, metadata = parse_file(uploaded_file)
    
    if parsed_data is not None:
        st.session_state.current_data = parsed_data
        st.session_state.data_info = metadata
        
        # File information
        if data_type == "dataframe":
            st.subheader("📋 Dataset Overview")
            
            # Data preview
            st.dataframe(parsed_data.head(10), use_container_width=True)
            
            # Auto insights
            insights = generate_data_insights(parsed_data)
            st.subheader("🔍 Quick Insights")
            for insight in insights:
                st.info(insight)
            
        elif data_type == "text":
            st.subheader("📄 Document Preview")
            preview_text = parsed_data[:1000] + "..." if len(parsed_data) > 1000 else parsed_data
            st.text_area("Content Preview", preview_text, height=200)
        

        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("💬 Previous Conversations")
            with st.expander("View Chat History", expanded=False):
                for i, chat in enumerate(st.session_state.chat_history):
                    timestamp = chat['timestamp'].strftime("%H:%M:%S")
                    st.markdown(f"**[{timestamp}] 👤 User:** {chat['question']}")
                    if 'response' in chat:
                        st.markdown(f"**🤖 Agent:** {chat['response'][:300]}...")
                    st.markdown("---")
        
        # Question interface
        st.subheader("💬 Ask Your Question")
        
        # Suggested questions
        if data_type == "dataframe":
            suggested_questions = [
                "What are the main patterns in this data?",
                "Show me statistical summaries for all columns",
                "Create visualizations for the most important relationships",
                "Are there any outliers or anomalies?",
                "What insights can you derive from this dataset?"
            ]
        else:
            suggested_questions = [
                "Summarize the key points in this document",
                "What are the main themes discussed?",
                "Extract important facts and figures",
                "Identify any conclusions or recommendations"
            ]
        
        st.write("**Suggested questions:**")
        cols = st.columns(len(suggested_questions))
        for i, suggestion in enumerate(suggested_questions):
            if cols[i % len(cols)].button(suggestion, key=f"suggest_{i}"):
                st.session_state.selected_question = suggestion
        
        # Question input
        question = st.text_input(
            "Type your question:",
            value=getattr(st.session_state, 'selected_question', ''),
            placeholder="e.g., What trends do you see in the data?"
        )
        
        col1, col2 = st.columns([1, 4])
        ask_button = col1.button("🚀 Ask", type="primary")
        
        if ask_button and question:
            # Add to chat history
            chat_entry = {
                'question': question,
                'timestamp': datetime.now(),
                'type': 'user_question'
            }
            
            with st.spinner("🤔 Analyzing..."):
                if data_type == "dataframe":
                    # Build conversation context from chat history
                    conversation_context = ""
                    if st.session_state.chat_history:
                        conversation_context = "\n\nPrevious conversation context:\n"
                        for i, prev_chat in enumerate(st.session_state.chat_history[-3:]):  # Last 3 exchanges
                            conversation_context += f"User Q{i+1}: {prev_chat['question']}\n"
                            if 'response' in prev_chat:
                                conversation_context += f"Agent A{i+1}: {prev_chat['response'][:400]}...\n\n"
                    
                    # Enhanced context for dataframes
                    context = f"""
Dataset Info:
- Shape: {parsed_data.shape}
- Columns: {list(parsed_data.columns)}
- Data Types: {parsed_data.dtypes.to_dict()}

Data Sample:
{parsed_data.head(10).to_string()}

Statistical Summary:
{parsed_data.describe().to_string()}
{conversation_context}
"""
                    
                    # Check if user is asking for visualization
                    viz_keywords = ['plot', 'chart', 'graph', 'visualiz', 'show', 'display', 'draw', 'histogram', 'scatter', 'bar chart', 'line chart', 'heatmap', 'boxplot', 'distribution']
                    needs_visualization = any(keyword in question.lower() for keyword in viz_keywords)
                    
                    if needs_visualization:
                        prompt = f"""You are an expert data analyst. Analyze this dataset and answer the user's question with detailed insights and create the requested visualizations.

{context}

Current Question: {question}

The user is asking for visualizations. Please provide:
1. A detailed analysis answering the question
2. Python code using pandas, matplotlib, seaborn to create the requested visualizations
3. Key insights from the analysis

Important guidelines for code:
- Use 'df' as the variable name for the dataset
- Create visualizations using matplotlib/seaborn as requested
- Use plt.figure(figsize=(10, 6)) to create properly sized plots
- Do NOT use plt.show() - Streamlit will handle plot display
- Always include proper labels, titles, and legends for plots
- For multiple plots, create separate figure instances

Generate the specific visualizations requested by the user."""
                    else:
                        prompt = f"""You are an expert data analyst. Analyze this dataset and answer the user's question with detailed insights.

{context}

Current Question: {question}

Please provide:
1. A detailed analysis answering the question
2. Python code for data analysis ONLY if needed to answer the question (calculations, filtering, grouping, etc.)
3. Key insights and recommendations

Important guidelines:
- Use 'df' as the variable name for the dataset
- Focus on answering the question with data analysis
- Only include Python code if it's necessary for calculations or data manipulation
- Do NOT create visualizations unless specifically requested
- Provide clear, data-driven insights"""

                else:
                    # Build conversation context for text documents
                    conversation_context = ""
                    if st.session_state.chat_history:
                        conversation_context = "\n\nPrevious conversation context:\n"
                        for i, prev_chat in enumerate(st.session_state.chat_history[-3:]):
                            conversation_context += f"User Q{i+1}: {prev_chat['question']}\n"
                            if 'response' in prev_chat:
                                conversation_context += f"Agent A{i+1}: {prev_chat['response'][:400]}...\n\n"
                    
                    preview = parsed_data[:3000]
                    prompt = f"""You are a document analysis expert. Analyze this document and answer the user's question.

Document Content:
\"\"\"{preview}\"\"\"
{conversation_context}

Current Question: {question}

Please provide a detailed analysis answering the question with specific references to the document content."""

                response = query_llama_agent(prompt, max_tokens=max_tokens)
                
                # Add response to chat entry
                chat_entry['response'] = response
            
            # Add to chat history
            st.session_state.chat_history.append(chat_entry)
            
            # Display response
            st.subheader("🧠 Analysis Results")
            st.markdown(response)
            
            # Execute code if dataframe
            if data_type == "dataframe" and "```python" in response:
                extract_and_run_code(response, parsed_data)
            
            # Clear selected question
            if hasattr(st.session_state, 'selected_question'):
                del st.session_state.selected_question

else:
    st.info("👆 Please upload a file to begin analysis")
    
    # Show example
    st.subheader("📝 How to use this tool:")
    st.markdown("""
    1. **Upload your data** - CSV, Excel, PDF, Word docs, or images
    2. **Ask questions** - Use natural language to query your data
    3. **Get insights** - Receive analysis, visualizations, and code
    4. **Interact** - Run generated code blocks and explore further
    
    **Example questions:**
    - "What are the trends in sales over time?"
    - "Show me correlations between variables"
    - "Are there any outliers in the data?"
    - "Create a dashboard view of key metrics"
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit • Enhanced Data Analysis Agent")