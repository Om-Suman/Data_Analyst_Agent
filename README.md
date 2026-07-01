# Data Analyst Agent

An AI-powered Streamlit application for uploading data, exploring it, cleaning it, generating charts, forecasting values, detecting anomalies, asking natural-language questions, and extracting grounded answers from documents.

## What it does

- Upload CSV, Excel, JSON, SQLite, PDF, Word, text, or image files.
- Clean datasets and track version snapshots.
- Explore distributions, correlations, filters, and column profiles.
- Ask AI questions that turn into safe Pandas code and grounded insights.
- Ask questions over uploaded documents with LlamaIndex retrieval.
- Generate forecasts, anomaly reports, and exportable HTML/Excel reports.

## Tech Stack

- Python, Streamlit, Pandas, NumPy
- Plotly, Matplotlib, Seaborn
- scikit-learn, SciPy, statsmodels
- Hugging Face Inference Providers
- LangChain and LlamaIndex for orchestration and retrieval
- PyMuPDF, python-docx, Pillow, pytesseract, chardet

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Configure `HF_API_KEY` via `.streamlit/secrets.toml`, an environment variable, or the sidebar.

## Project Layout

- `app.py` - Streamlit entry point and page bootstrap.
- `app_pages/` - UI pages for upload, exploration, AI query, document QA, insights, forecasting, anomalies, reports, and settings.
- `modules/` - ingestion, cleaning, executor, AI client, query engine, LangChain wrapper, LlamaIndex retrieval, forecasting, anomaly detection, and insights logic.
- `utils/` - session state, config loading, and sidebar navigation.
- `requirements.txt` - runtime dependencies.
- `.streamlit/` - Streamlit theme and secrets template.

## Security Notes

- AI-generated code runs in a restricted sandbox.
- Secrets are loaded from Streamlit secrets or environment variables.
- Streamlit XSRF protection is enabled.

## Documentation

For a full technical deep dive, see [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md).

## Deployment

The app can run locally with Streamlit or be deployed to Streamlit Cloud. Add `HF_API_KEY` in the deployment secrets.

## License

MIT
