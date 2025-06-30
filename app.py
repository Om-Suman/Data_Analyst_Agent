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
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = None

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
def extract_and_run_code(response_text, df, chat_id):
    """Enhanced code execution with better error handling and visualization"""
    code_blocks = re.findall(r"```python(.*?)```", response_text, re.DOTALL)
    
    if not code_blocks:
        return []
    
    st.subheader("🔧 Generated Code & Results")
    
    execution_results = []
    
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
            
            # Store execution results
            result = {
                'code': code,
                'output': printed_output.strip(),
                'has_plots': bool(plt.get_fignums()),
                'execution_time': datetime.now()
            }
            execution_results.append(result)
            
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
            error_msg = f"❌ Execution error: {str(e)}"
            st.error(error_msg)
            st.code(str(e))
            
            # Store error in results
            result = {
                'code': code,
                'error': str(e),
                'execution_time': datetime.now()
            }
            execution_results.append(result)
        finally:
            # Clean up any remaining plots
            plt.close('all')
    
    return execution_results

# --- Enhanced Chat History Management ---
def add_to_chat_history(question, response, data_type, execution_results=None):
    """Add a complete conversation entry to chat history"""
    chat_entry = {
        'id': f"chat_{len(st.session_state.chat_history)}_{int(time.time())}",
        'question': question,
        'response': response,
        'timestamp': datetime.now(),
        'data_type': data_type,
        'execution_results': execution_results or [],
        'file_name': st.session_state.current_file_name
    }
    st.session_state.chat_history.append(chat_entry)
    return chat_entry['id']

def display_chat_history():
    """Display the complete chat history with all messages"""
    if not st.session_state.chat_history:
        return
        
    st.subheader("💬 Conversation History")
    
    for i, chat in enumerate(st.session_state.chat_history):
        timestamp = chat['timestamp'].strftime("%H:%M:%S")
        
        with st.expander(f"[{timestamp}] 👤 {chat['question'][:60]}...", expanded=(i == len(st.session_state.chat_history) - 1)):
            # User question
            st.markdown(f"**👤 User:** {chat['question']}")
            
            # Agent response
            st.markdown(f"**🤖 Agent:**")
            st.markdown(chat['response'])
            
            # Show execution results if any
            if chat.get('execution_results'):
                st.markdown("**🔧 Code Execution Results:**")
                for j, result in enumerate(chat['execution_results']):
                    if 'error' in result:
                        st.error(f"Code block {j+1} failed: {result['error']}")
                    else:
                        st.success(f"Code block {j+1} executed successfully")
                        if result.get('output'):
                            st.text(result['output'])
                        if result.get('has_plots'):
                            st.info("📈 Visualization was generated")
            
            # Metadata
            st.caption(f"File: {chat.get('file_name', 'N/A')} | Type: {chat.get('data_type', 'N/A')}")
            
            # Re-run button
            if st.button(f"🔄 Re-run this query", key=f"rerun_{chat['id']}"):
                st.session_state.selected_question = chat['question']
                st.rerun()

def export_chat_history():
    """Export chat history as downloadable file"""
    if not st.session_state.chat_history:
        return None
    
    export_data = []
    for chat in st.session_state.chat_history:
        export_entry = {
            'timestamp': chat['timestamp'].isoformat(),
            'file_name': chat.get('file_name', ''),
            'question': chat['question'],
            'response': chat['response'],
            'data_type': chat.get('data_type', ''),
            'execution_count': len(chat.get('execution_results', []))
        }
        export_data.append(export_entry)
    
    return pd.DataFrame(export_data)

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
    
    # Chat history management
    if st.session_state.chat_history:
        st.header("💬 Chat Management")
        
        # Export chat history
        export_df = export_chat_history()
        if export_df is not None:
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Export Chat History",
                data=csv,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Chat statistics
        st.metric("Total Conversations", len(st.session_state.chat_history))
        
        # Clear history button
        if st.button("🗑️ Clear All History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Quick actions
    st.header("⚡ Quick Actions")
    if st.session_state.current_data is not None:
        if st.button("📈 Quick Data Summary"):
            st.session_state.selected_question = 'Generate a comprehensive data summary with key statistics and insights'
        
        if st.button("🔍 Find Correlations"):
            st.session_state.selected_question = 'Find and visualize correlations in the data with heatmap and scatter plots'
        
        if st.button("📊 Create Dashboard"):
            st.session_state.selected_question = 'Create a comprehensive dashboard with multiple visualizations showing key insights'
    
    # Recent questions quick access
    if st.session_state.chat_history:
        st.header("🕒 Recent Questions")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
            timestamp = chat['timestamp'].strftime("%H:%M")
            chat_preview = f"{chat['question'][:30]}..."
            if st.button(f"[{timestamp}] {chat_preview}", key=f"recent_{i}"):
                st.session_state.selected_question = chat['question']

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
        # Don't clear history automatically - let user decide
        if st.session_state.chat_history:
            if st.button("🗑️ New file detected. Clear chat history?"):
                st.session_state.chat_history = []
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
        
        # Display complete chat history
        if st.session_state.chat_history:
            display_chat_history()
        
        # Question interface
        st.subheader("💬 Ask Your Question")
        
        # Suggested questions
        if data_type == "dataframe":
            suggested_questions = [
                "What are the main patterns and trends in this data?",
                "Show me statistical summaries and distributions for all columns",
                "Create comprehensive visualizations for the most important relationships",
                "Are there any outliers, anomalies, or data quality issues?",
                "Generate a complete analysis report with key insights and recommendations"
            ]
        else:
            suggested_questions = [
                "Summarize the key points and main arguments in this document",
                "What are the main themes, topics, and conclusions discussed?",
                "Extract all important facts, figures, and data points",
                "Identify any recommendations, action items, or next steps mentioned"
            ]
        
        st.write("**Suggested questions:**")
        cols = st.columns(min(3, len(suggested_questions)))
        for i, suggestion in enumerate(suggested_questions):
            col_idx = i % len(cols)
            if cols[col_idx].button(suggestion, key=f"suggest_{i}"):
                st.session_state.selected_question = suggestion
        
        # Question input
        question = st.text_input(
            "Type your question:",
            value=getattr(st.session_state, 'selected_question', ''),
            placeholder="e.g., What trends do you see in the data? Create visualizations to show key insights."
        )
        
        col1, col2 = st.columns([1, 4])
        ask_button = col1.button("🚀 Ask", type="primary")
        
        if ask_button and question:
            with st.spinner("🤔 Analyzing..."):
                if data_type == "dataframe":
                    # Build conversation context from chat history
                    conversation_context = ""
                    if st.session_state.chat_history:
                        conversation_context = "\n\nPrevious conversation context:\n"
                        for i, prev_chat in enumerate(st.session_state.chat_history[-3:]):  # Last 3 exchanges
                            conversation_context += f"User Q{i+1}: {prev_chat['question']}\n"
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
                    viz_keywords = ['plot', 'chart', 'graph', 'visualiz', 'show', 'display', 'draw', 'histogram', 'scatter', 'bar chart', 'line chart', 'heatmap', 'boxplot', 'distribution', 'dashboard']
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
- Use plt.figure(figsize=(12, 8)) to create properly sized plots
- Do NOT use plt.show() - Streamlit will handle plot display
- Always include proper labels, titles, and legends for plots
- For multiple plots, create separate figure instances
- Use professional styling and clear, readable plots

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
                            conversation_context += f"Agent A{i+1}: {prev_chat['response'][:400]}...\n\n"
                    
                    preview = parsed_data[:3000]
                    prompt = f"""You are a document analysis expert. Analyze this document and answer the user's question.

Document Content:
\"\"\"{preview}\"\"\"
{conversation_context}

Current Question: {question}

Please provide a detailed analysis answering the question with specific references to the document content."""

                response = query_llama_agent(prompt, max_tokens=max_tokens)
                
                # Display response immediately
                st.subheader("🧠 Analysis Results")
                st.markdown(response)
                
                # Execute code if dataframe and get results
                execution_results = []
                if data_type == "dataframe" and "```python" in response:
                    execution_results = extract_and_run_code(response, parsed_data, st.session_state.current_chat_id)
                
                # Save complete conversation to history
                chat_id = add_to_chat_history(question, response, data_type, execution_results)
                st.session_state.current_chat_id = chat_id
            
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
    4. **Review history** - All conversations are saved and can be exported
    5. **Re-run queries** - Click on previous conversations to run them again
    
    **Example questions:**
    - "What are the trends in sales over time? Show me visualizations."
    - "Show me correlations between variables with heatmaps"
    - "Are there any outliers in the data? Create box plots to show them."
    - "Create a comprehensive dashboard view of key metrics"
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit • Enhanced Data Analysis Agent with Persistent Chat History")