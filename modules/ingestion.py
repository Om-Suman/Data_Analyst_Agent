"""
Data Ingestion Module
Supports: CSV, Excel, JSON, SQLite, Text, PDF, Word, Images
"""
import pandas as pd
import numpy as np
import streamlit as st
import json
import sqlite3
import chardet
import io
from datetime import datetime
from pathlib import Path


def detect_encoding(file_bytes: bytes) -> str:
    result = chardet.detect(file_bytes)
    return result.get("encoding", "utf-8") or "utf-8"


def infer_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Try to optimally infer/cast dtypes."""
    for col in df.columns:
        # Try numeric
        if df[col].dtype == object:
            try:
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() / max(len(df), 1) > 0.8:
                    df[col] = converted
                    continue
            except Exception:
                pass
            # Try datetime
            try:
                converted = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                if converted.notna().sum() / max(len(df), 1) > 0.8:
                    df[col] = converted
                    continue
            except Exception:
                pass
    return df


def compute_metadata(df: pd.DataFrame, source: str = "", file_size: int = 0) -> dict:
    """Compute rich metadata about the dataframe."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    col_profiles = {}
    for col in df.columns:
        profile = {
            "dtype": str(df[col].dtype),
            "missing": int(missing[col]),
            "missing_pct": float(missing_pct[col]),
            "unique": int(df[col].nunique()),
        }
        if col in numeric_cols:
            profile.update({
                "min": float(df[col].min()) if not df[col].isna().all() else None,
                "max": float(df[col].max()) if not df[col].isna().all() else None,
                "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                "std": float(df[col].std()) if not df[col].isna().all() else None,
            })
        elif col in cat_cols:
            top = df[col].value_counts().head(5).to_dict()
            profile["top_values"] = {str(k): int(v) for k, v in top.items()}
        col_profiles[col] = profile

    return {
        "source": source,
        "file_size": file_size,
        "rows": len(df),
        "cols": len(df.columns),
        "columns": df.columns.tolist(),
        "numeric_cols": numeric_cols,
        "categorical_cols": cat_cols,
        "date_cols": date_cols,
        "missing_total": int(missing.sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "col_profiles": col_profiles,
        "loaded_at": datetime.now().isoformat(),
    }


def load_csv(file) -> tuple[pd.DataFrame, dict]:
    raw = file.read()
    encoding = detect_encoding(raw)
    df = pd.read_csv(io.BytesIO(raw), encoding=encoding, low_memory=False)
    df = infer_dtypes(df)
    meta = compute_metadata(df, source=file.name, file_size=len(raw))
    return df, meta


def load_excel(file) -> tuple[pd.DataFrame, dict]:
    raw = file.read()
    df = pd.read_excel(io.BytesIO(raw))
    df = infer_dtypes(df)
    meta = compute_metadata(df, source=file.name, file_size=len(raw))
    return df, meta


def load_json(file) -> tuple[pd.DataFrame, dict]:
    raw = file.read()
    data = json.loads(raw)
    if isinstance(data, list):
        df = pd.json_normalize(data)
    elif isinstance(data, dict):
        df = pd.json_normalize([data])
    else:
        raise ValueError("JSON must be a list or dict")
    df = infer_dtypes(df)
    meta = compute_metadata(df, source=file.name, file_size=len(raw))
    return df, meta


def load_sqlite(file) -> dict[str, pd.DataFrame]:
    """Returns dict of table_name -> DataFrame."""
    import tempfile, os
    raw = file.read()
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        conn = sqlite3.connect(tmp_path)
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
        result = {}
        for table in tables["name"]:
            result[table] = pd.read_sql(f"SELECT * FROM {table}", conn)
        conn.close()
    finally:
        os.unlink(tmp_path)
    return result


def parse_uploaded_file(file) -> tuple[pd.DataFrame | None, dict, str]:
    """
    Main entry point. Returns (df, meta, data_type).
    data_type: 'dataframe' | 'text'
    """
    suffix = Path(file.name).suffix.lower().lstrip(".")

    try:
        if suffix == "csv":
            df, meta = load_csv(file)
            return df, meta, "dataframe"
        elif suffix in ("xlsx", "xls"):
            df, meta = load_excel(file)
            return df, meta, "dataframe"
        elif suffix == "json":
            df, meta = load_json(file)
            return df, meta, "dataframe"
        elif suffix == "db":
            tables = load_sqlite(file)
            if tables:
                first = list(tables.keys())[0]
                df = tables[first]
                meta = compute_metadata(df, source=file.name)
                meta["tables"] = list(tables.keys())
                return df, meta, "dataframe"
        elif suffix == "txt":
            content = file.read().decode("utf-8", errors="replace")
            return None, {"type": "text", "content": content}, "text"
        elif suffix == "docx":
            import docx as docxlib
            doc = docxlib.Document(file)
            content = "\n".join(p.text for p in doc.paragraphs)
            return None, {"type": "text", "content": content}, "text"
        elif suffix == "pdf":
            import fitz
            raw = file.read()
            doc = fitz.open(stream=raw, filetype="pdf")
            content = "\n".join(page.get_text() for page in doc)
            return None, {"type": "text", "content": content}, "text"
        elif suffix in ("png", "jpg", "jpeg"):
            from PIL import Image
            import pytesseract
            img = Image.open(file)
            content = pytesseract.image_to_string(img)
            return None, {"type": "text", "content": content}, "text"
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    except Exception as e:
        raise RuntimeError(f"Failed to parse {file.name}: {e}") from e
