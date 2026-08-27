# Data Analyst Agent (Production React + FastAPI Edition)

An enterprise-grade AI data analysis platform combining deterministic analytics with LLM reasoning. Built with **FastAPI**, **React**, **Vite**, **TypeScript**, **Tailwind CSS**, and **Hugging Face Inference models**.

---

## 🚀 Key Features

- **Multi-Format Ingestion**: Upload CSV, Excel (`.xlsx`, `.xls`), JSON, SQLite (`.db`, `.sqlite`), PDF, Word (`.docx`), plain text (`.txt`), and OCR images.
- **Exploratory Data Analysis (EDA)**: Interactive paginated grid, descriptive statistics, correlation heatmaps (Pearson, Spearman, Kendall), distribution analyzers (Histogram, Box Plot, Violin), and deep column profilers.
- **Automated Data Quality & Cleaning**:
  - Quality scoring algorithm (0–100, Grade A–F) with issue diagnosis.
  - Imputation (mean, median, mode, forward fill, backward fill, custom, drop).
  - Outlier detection & filtering via Z-Score or IQR.
  - Automatic dtype inference and column name normalization (snake_case).
  - Version history with instant snapshot rollback.
- **Natural Language AI Data Queries**:
  - LangChain-based heuristic and LLM intent routing.
  - Safe Python sandbox execution with AST syntax validation, restricted globals, and banned dangerous imports (`os`, `sys`, `subprocess`, `open`).
  - Automatic single-pass code repair for recoverable errors.
  - Dynamic Plotly visualization and DataFrame table generation.
- **Document QA (RAG)**: Retrieval-Augmented Generation over unstructured documents using LlamaIndex with keyword similarity fallback.
- **Interactive Visualizations Studio**: 14+ interactive chart types (Bar, Line, Scatter, Histogram, Box, Violin, Pie, Area, Heatmap, Treemap, Sunburst, Bubble, Funnel, KPI Dashboard).
- **Automated Business Intelligence & Insights**: Rule-based statistical pattern detection and deep AI executive summaries, findings, trends, opportunities, risks, recommendations, and narrative data stories.
- **Time Series Forecasting**: Moving Average, Linear Trend Regression (OLS), and Exponential Smoothing (Holt-Winters) with 95% confidence intervals and CSV export.
- **Anomaly Detection**: Multivariate Isolation Forest, Gaussian Z-Score, and IQR anomaly scans with interactive PCA 2D scatter projections.
- **Multi-Format Report Exports**:
  - Standalone interactive dark-mode HTML executive reports.
  - Multi-tab formatted Excel workbooks (`.xlsx`) with Data, Statistics, Missing Values, Column Info, and Query History sheets.
  - Cleaned data CSV downloads.

---

## 🏗️ Architecture

```text
DataAnalystAgent/
├── backend/                  # FastAPI REST API Backend
│   ├── api/                  # Modular APIRouters
│   │   ├── routes_config.py
│   │   ├── routes_datasets.py
│   │   ├── routes_cleaning.py
│   │   ├── routes_explorer.py
│   │   ├── routes_query.py
│   │   ├── routes_document.py
│   │   ├── routes_visualizations.py
│   │   ├── routes_insights.py
│   │   ├── routes_forecasting.py
│   │   ├── routes_anomalies.py
│   │   └── routes_reports.py
│   ├── session/              # Thread-safe SessionState & SessionManager
│   │   └── state.py
│   ├── schemas/              # Pydantic Request & Response Schemas
│   ├── services/             # Pure Business Logic & Serialization
│   ├── tests/                # Comprehensive Pytest Suite
│   └── main.py               # FastAPI App & Static File Serving
│
├── frontend/                 # Modern React + Vite + TypeScript Frontend
│   ├── src/
│   │   ├── api/              # Axios API Client & Session Handler
│   │   ├── components/       # PlotlyChart, DataTable, MetricCard, Navbar, Sidebar
│   │   ├── context/          # DatasetContext & State Management
│   │   ├── layouts/          # Responsive AppLayout Shell
│   │   ├── pages/            # 12 Feature Pages (Dashboard, Upload, Cleaning, etc.)
│   │   └── types/            # TypeScript Interface Definitions
│   ├── index.html
│   ├── tailwind.config.js
│   └── vite.config.ts
│
├── modules/                  # Decoupled Core Analytics & AI Engines
│   ├── ingestion.py
│   ├── cleaning.py
│   ├── executor.py
│   ├── llm_client.py
│   ├── insights.py
│   ├── forecasting.py
│   ├── anomaly_detection.py
│   ├── document_rag.py
│   ├── query_engine.py
│   └── langchain_query.py
│
├── requirements.txt
├── .env
├── .env.example
└── README.md
```

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### 2. Backend Setup
```bash
# Activate virtual environment
.\venv\Scripts\activate  # On Windows
source venv/bin/activate # On Linux/macOS

# Install Python dependencies
pip install -r requirements.txt

# Run backend test suite
pytest backend/tests

# Start FastAPI backend server (port 8000)
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger API documentation is available at `http://127.0.0.1:8000/docs`.

### 3. Frontend Setup
```bash
cd frontend

# Install Node packages
npm install

# Start Vite development server (port 5173 with API proxy to 8000)
npm run dev
```
Open `http://localhost:5173` in your browser.

### 4. Single-Port Production Build
```bash
cd frontend
npm run build
```
Once built into `frontend/dist/`, running `uvicorn backend.main:app --port 8000` automatically serves both the backend API and the React single-page application from `http://localhost:8000`.

---

## 🔒 Security & Code Execution Sandbox

- **AST Validation**: Code submitted for execution is parsed into an Abstract Syntax Tree (AST) to detect and block forbidden AST nodes (`Import`, `ImportFrom`, attribute access to private/dunder members, dangerous built-ins).
- **Restricted Namespace**: Code runs inside a restricted global namespace allowing only safe packages (`pandas`, `numpy`, `plotly`, `math`, `datetime`, `re`, `json`, `scikit-learn`).
- **DataFrame Immutability**: Code executions operate on deep copies of in-memory DataFrames to prevent accidental destructive state corruption.
- **Key Masking**: Hugging Face API keys are stored in backend memory sessions and are never transmitted in cleartext over API responses.

---

## 📄 License
MIT License.
