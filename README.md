# 📊 Data Analyst Agent

A **AI Data Analytics Agent** powered by Hugging Face LLMs. Think of it as a self-hosted mini-combination of Power BI + Julius AI + ChatGPT for your data.

---

## ✨ Features

| Module | Capabilities |
|--------|-------------|
| 📁 **Data Upload** | CSV, Excel, JSON, SQLite, PDF, Word, Images (OCR) |
| 🧹 **Data Cleaning** | Missing values, duplicates, outliers, dtype fixing, quality scoring |
| 🔍 **Data Explorer** | Browse, filter, correlations, column profiling |
| 🤖 **AI Query** | Natural language → Pandas code → results, auto-visualizations |
| 📈 **Visualizations** | 14 chart types with Plotly, interactive, downloadable |
| 💡 **Insights** | AI-generated business intelligence & executive summaries |
| 🔮 **Forecasting** | Moving average, linear trend, exponential smoothing |
| 🚨 **Anomaly Detection** | Isolation Forest, Z-Score, IQR |
| 📄 **Reports** | HTML, Excel, in-app profiling |
| ⚙️ **Settings** | Model config, version history, theme toggle |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Om-Suman/Data_Analyst_Agent
cd data-analyst-agent
pip install -r requirements.txt
```

### 2. Configure API Key

**Option A — Secrets file (recommended for local):**
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and set HF_API_KEY
```

**Option B — Environment variable:**
```bash
export HF_API_KEY="hf_your_key_here"
```

**Option C — Enter in sidebar at runtime.**

### 3. Run

```bash
streamlit run app.py
```

Open http://localhost:8501

---

## 🤖 AI Models

| Role | Model |
|------|-------|
| Primary | `Qwen/Qwen3-32B` |
| Fallback | `deepseek-ai/DeepSeek-R1` |

The system automatically falls back to the secondary model if the primary is unavailable. Both are accessed via the Hugging Face Inference Providers API.

---

## 📂 Project Structure

```
Data_Analyst_Agent/
├── app.py                    # Main entry point
├── requirements.txt
├── .streamlit/
│   ├── config.toml           # Streamlit theme & server config
│   └── secrets.toml.example  # API key template
├── pages/                    # One file per page
│   ├── dashboard.py
│   ├── upload.py
│   ├── cleaning.py
│   ├── explorer.py
│   ├── ai_query.py
│   ├── visualizations.py
│   ├── insights.py
│   ├── forecasting.py
│   ├── anomalies.py
│   ├── reports.py
│   └── settings.py
├── modules/                  # Core business logic
│   ├── llm_client.py         # HF API with retry & fallback
│   ├── executor.py           # Safe code sandbox
│   ├── ingestion.py          # File parsing & metadata
│   ├── cleaning.py           # Data quality & cleaning
│   ├── query_engine.py       # NL → Pandas pipeline
│   ├── anomaly_detection.py  # Isolation Forest, Z-Score, IQR
│   ├── forecasting.py        # MA, Linear, Exponential Smoothing
│   └── insights.py           # Auto-insights engine
└── utils/
    ├── config.py             # Configuration loader
    ├── session.py            # Session state management
    └── navigation.py         # Sidebar routing
```

---

## 🔐 Security

- **Sandboxed code execution** — AI-generated Pandas code runs in a restricted namespace with blocked imports (`os`, `subprocess`, `eval`, etc.)
- **API key isolation** — Keys stored in Streamlit secrets or environment variables, never in code
- **Input sanitization** — All user inputs validated before processing
- **File size limits** — Configurable via `.streamlit/config.toml`

---

## 🚀 Deployment

### Streamlit Cloud
1. Push to GitHub
2. Go to share.streamlit.io
3. Deploy from your repo
4. Add `HF_API_KEY` in Streamlit Cloud secrets

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t dataagent .
docker run -p 8501:8501 -e HF_API_KEY=hf_your_key dataagent
```

---

## 📊 Supported File Types

| Format | Details |
|--------|---------|
| CSV | Auto-encoding detection, large file support |
| Excel (.xlsx/.xls) | Multi-sheet, merged cells handled |
| JSON | Nested JSON auto-flattened |
| SQLite (.db) | All tables imported |
| PDF | Text extraction via PyMuPDF |
| Word (.docx) | Paragraph extraction |
| Images | OCR via Tesseract |

---

## 🤝 Contributing

PRs welcome! Please open an issue first to discuss major changes.

---

## 📄 License

MIT License
