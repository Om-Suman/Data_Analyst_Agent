from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Any, Optional
import numpy as np
import pandas as pd

from backend.session.state import SessionState
from modules.ingestion import parse_uploaded_file, compute_metadata
from modules.document_rag import build_document_bundle, store_document_bundle


def to_json_compatible(obj: Any) -> Any:
    """Recursively convert numpy types, timestamps, NaNs, and custom objects to JSON-friendly primitives."""
    if obj is None:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, (list, tuple, set)):
        return [to_json_compatible(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_json_compatible(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return [to_json_compatible(x) for x in obj.tolist()]
    if pd.isna(obj):
        return None
    return str(obj)


def sanitize_dataframe_for_json(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert dataframe rows to JSON-serializable list of dicts, handling NaNs, Infs, timestamps, and numpy types."""
    if df is None or df.empty:
        return []

    records = []
    columns = list(df.columns)
    for row in df.itertuples(index=False):
        record = {str(col): to_json_compatible(val) for col, val in zip(columns, row)}
        records.append(record)
    return records


class UploadFileAdapter:
    """Wrapper around FastAPI UploadFile to match ingestion module interface."""
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data
        self._io = io.BytesIO(data)

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            return self._data
        return self._io.read(size)

    def seek(self, offset: int, whence: int = 0):
        return self._io.seek(offset, whence)


def make_sales_sample() -> pd.DataFrame:
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "product": np.random.choice(["Widget A", "Widget B", "Widget C", "Gadget X"], n),
        "sales": np.random.normal(1000, 300, n).clip(50).round(2),
        "units": np.random.randint(1, 100, n),
        "profit": np.random.normal(200, 80, n).round(2),
        "customer_id": np.random.randint(1000, 9999, n),
    })


def make_employee_sample() -> pd.DataFrame:
    np.random.seed(7)
    n = 300
    return pd.DataFrame({
        "employee_id": range(1, n + 1),
        "department": np.random.choice(["Engineering", "Sales", "Marketing", "HR", "Finance"], n),
        "salary": np.random.normal(75000, 20000, n).clip(30000).round(0),
        "years_experience": np.random.randint(0, 20, n),
        "performance_score": np.random.uniform(1, 5, n).round(1),
        "remote": np.random.choice([True, False], n),
        "hire_date": pd.date_range("2015-01-01", periods=n, freq="30D"),
    })


def make_finance_sample() -> pd.DataFrame:
    np.random.seed(99)
    n = 365
    base = 100
    returns = np.random.normal(0.001, 0.02, n)
    prices = base * (1 + returns).cumprod()
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n),
        "close": prices.round(2),
        "volume": np.random.randint(1_000_000, 10_000_000, n),
        "high": (prices * np.random.uniform(1.0, 1.03, n)).round(2),
        "low": (prices * np.random.uniform(0.97, 1.0, n)).round(2),
        "category": np.random.choice(["Tech", "Finance", "Healthcare", "Energy"], n),
    })


def process_and_register_file(
    session: SessionState,
    filename: str,
    file_bytes: bytes,
) -> dict[str, Any]:
    adapter = UploadFileAdapter(filename, file_bytes)
    df, meta, data_type = parse_uploaded_file(adapter)

    if data_type == "dataframe" and df is not None:
        session.register_dataset(filename, df, source=filename, meta=meta)
        return {
            "name": filename,
            "type": "dataframe",
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
            "source": filename,
        }
    elif data_type == "text":
        content = meta.get("content", "")
        session.register_text_dataset(filename, content, source=filename, meta=meta)
        bundle = build_document_bundle(content, metadata={"name": filename, **meta})
        store_document_bundle(filename, bundle, store=session.document_indexes)
        return {
            "name": filename,
            "type": "text",
            "characters": len(content),
            "source": filename,
        }
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def load_sample_dataset(session: SessionState, sample_name: str) -> dict[str, Any]:
    sample_generators = {
        "Sales Data": make_sales_sample,
        "Employee Data": make_employee_sample,
        "Finance Data": make_finance_sample,
    }
    if sample_name not in sample_generators:
        raise ValueError(f"Unknown sample dataset: {sample_name}. Available: {list(sample_generators.keys())}")

    df = sample_generators[sample_name]()
    meta = compute_metadata(df, source="sample")
    session.register_dataset(sample_name, df, source="sample", meta=meta)
    return {
        "name": sample_name,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "source": "sample",
    }


def get_dataset_preview(session: SessionState, limit: int = 50) -> dict[str, Any]:
    record = session.get_active_record()
    if not record:
        raise ValueError("No active dataset found.")

    name = record["name"]
    df = record.get("df")
    is_text = record.get("text_content") is not None
    meta_json = to_json_compatible(record.get("meta", {}))

    if is_text or df is None:
        return {
            "name": name,
            "rows": 0,
            "cols": 0,
            "columns": [],
            "data": [],
            "metadata": meta_json,
            "numeric_cols": [],
            "categorical_cols": [],
            "date_cols": [],
            "is_text": True,
            "text_content": record.get("text_content", ""),
        }

    numeric_cols = [str(c) for c in df.select_dtypes(include=np.number).columns.tolist()]
    cat_cols = [str(c) for c in df.select_dtypes(include="object").columns.tolist()]
    date_cols = [str(c) for c in df.select_dtypes(include=["datetime64"]).columns.tolist()]

    preview_df = df.head(limit)
    data = sanitize_dataframe_for_json(preview_df)

    return {
        "name": name,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "columns": [str(c) for c in df.columns],
        "data": data,
        "metadata": meta_json,
        "numeric_cols": numeric_cols,
        "categorical_cols": cat_cols,
        "date_cols": date_cols,
        "is_text": False,
        "text_content": None,
    }
