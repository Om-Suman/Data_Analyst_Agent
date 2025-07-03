# 📊 Advanced Data Analyst Agent

A powerful Streamlit-based data analysis application that combines natural language processing with automated data analysis and visualization. Upload your data, ask questions in plain English, and get comprehensive insights with visualizations and executable code.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Features

### 🔍 **Multi-Format Data Support**

- **Structured Data**: CSV, Excel (XLSX)
- **Documents**: PDF, Word documents (DOCX), Plain text
- **Images**: PNG, JPG, JPEG with OCR text extraction
- **Automatic parsing** with metadata extraction and error handling

### 🤖 **AI-Powered Analysis**

- **Natural language queries** - Ask questions in plain English
- **Intelligent code generation** - Automatically generates Python code for analysis
- **Context-aware responses** - Uses conversation history for better insights
- **Visualization recommendations** - Suggests appropriate charts and graphs

### 📈 **Advanced Visualizations**

- **Interactive plots** using Matplotlib, Seaborn, and Plotly
- **Statistical analysis** with comprehensive summaries
- **Correlation analysis** and pattern detection
- **Outlier identification** and data quality assessment
- **Dashboard creation** with multiple visualization types

### 💬 **Persistent Chat History**

- **Complete conversation storage** - All questions and responses saved
- **Code execution tracking** - Monitors success/failure of generated code
- **Export functionality** - Download chat history as CSV
- **Re-run capability** - Replay any previous analysis
- **Conversation context** - Builds upon previous interactions

### 🛠️ **Advanced Features**

- **Real-time code execution** with error handling
- **Data insights generation** - Automatic data profiling
- **Quick action buttons** - Common analysis tasks
- **Configurable AI parameters** - Adjust model behavior
- **File change detection** - Smart history management

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Required Dependencies

```bash
pip install -r requirements.txt

```

## 🔧 Configuration

### API Key Setup

The application requires an API key for the LLM service. Set it up in one of these ways:

#### Option 1: Streamlit Secrets (Recommended)

Create `.streamlit/secrets.toml`:

```toml
API_KEY = "your_api_key_here"
```

#### Option 2: Environment Variable

```bash
export API_KEY="your_api_key_here"
```

#### Option 3: Manual Entry

Enter the API key directly in the sidebar when prompted.

### Model Configuration

The application uses the Llama model via Together AI. You can modify the model in the code:

```python
LLAMA_MODEL_ID = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
```

## 🚀 Usage

### Starting the Application

```bash
streamlit run app.py
```

### Basic Workflow

1. **Upload Data**

   - Click "Upload a file" and select your data file
   - Supported formats: CSV, Excel, PDF, Word, Images
   - View automatic data preview and insights

2. **Ask Questions**

   - Use the text input or click suggested questions
   - Examples:
     - "What are the main trends in this data?"
     - "Show me correlations between variables"
     - "Create a dashboard with key metrics"
     - "Are there any outliers in the data?"

3. **Review Results**

   - Get detailed analysis and insights
   - View automatically generated visualizations
   - Examine the generated Python code
   - All results are saved in chat history

4. **Export and Share**
   - Download chat history as CSV
   - Re-run previous analyses
   - Build upon previous insights

### Example Questions by Data Type

#### For Structured Data (CSV/Excel):

- "What are the key statistics for each column?"
- "Show me the distribution of [column_name]"
- "Create a correlation heatmap"
- "Find outliers in the dataset"
- "What patterns do you see over time?"
- "Compare [column_A] vs [column_B]"
- "Generate a comprehensive data report"

#### For Documents (PDF/Word/Text):

- "Summarize the key points in this document"
- "What are the main themes discussed?"
- "Extract important facts and figures"
- "Identify conclusions and recommendations"
- "What are the key takeaways?"

#### For Images:

- "Extract and analyze the text from this image"
- "What information can you find in this image?"
- "Summarize the content shown in the image"

## 🔄 Advanced Features

### Chat History Management

- **Persistent storage**: All conversations are saved during the session
- **Export functionality**: Download complete history as CSV
- **Re-run queries**: Click any previous question to run it again
- **Conversation context**: AI remembers previous interactions
- **Execution tracking**: Monitor code success/failure

### Code Execution

- **Automatic code generation**: Creates Python code for analysis
- **Safe execution**: Isolated environment with error handling
- **Visualization rendering**: Automatically displays plots
- **Output capture**: Shows print statements and results
- **Error reporting**: Clear error messages with debugging info

### Data Insights

- **Automatic profiling**: Immediate insights upon file upload
- **Missing value detection**: Identifies data quality issues
- **Data type analysis**: Automatic column type detection
- **Duplicate identification**: Finds duplicate records
- **Statistical summaries**: Basic descriptive statistics

## 🎨 Customization

### Sidebar Configuration

- **Model parameters**: Adjust max tokens and temperature
- **Quick actions**: Predefined analysis tasks
- **Chat management**: Export, clear, and view statistics
- **Recent questions**: Quick access to previous queries

### Visualization Settings

- **Plot styling**: Modify figure sizes and styles
- **Color schemes**: Customize visualization colors
- **Interactive plots**: Enable/disable interactivity
- **Export formats**: Configure output formats

## 📋 File Structure

```
├── app.py                   # Main application file
├── .streamlit/
│   └── secrets.toml         # API key configuration
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── data/                   # Sample data files (optional)
```

## 📞 Support

## 🔄 Updates and Roadmap

### Current Version: 1.0.0

- ✅ Multi-format file support
- ✅ AI-powered analysis
- ✅ Persistent chat history
- ✅ Code execution and visualization
- ✅ Export functionality

### Planned Features:

- 🔄 Database connectivity
- 🔄 Advanced statistical tests
- 🔄 Machine learning model suggestions
- 🔄 Collaborative analysis features
- 🔄 Custom visualization templates

---

**Happy Analyzing! 📊✨**
