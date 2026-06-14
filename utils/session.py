"""Session state management."""
import streamlit as st
from datetime import datetime

DEFAULTS = {
    "active_page": "Dashboard",
    "datasets": {},           # registry: name -> DatasetRecord
    "active_dataset": None,   # name of currently active dataset
    "query_history": [],      # list of QueryRecord dicts
    "chat_messages": [],      # list of {role, content, timestamp}
    "dataset_versions": {},   # name -> list of version snapshots
    "anomaly_results": None,
    "forecast_results": None,
    "cleaning_log": [],
    "profile_report": None,
    "selected_question": None,
    "viz_history": [],
}

def init_session_state():
    for key, default in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default

def get_active_df():
    name = st.session_state.active_dataset
    if not name or name not in st.session_state.datasets:
        return None
    return st.session_state.datasets[name]["df"]

def get_active_meta():
    name = st.session_state.active_dataset
    if not name or name not in st.session_state.datasets:
        return {}
    return st.session_state.datasets[name].get("meta", {})

def register_dataset(name, df, source="upload", meta=None):
    """Register a dataset in the session registry."""
    if meta is None:
        meta = {}
    record = {
        "df": df,
        "name": name,
        "source": source,
        "uploaded_at": datetime.now(),
        "rows": len(df),
        "cols": len(df.columns),
        "meta": meta,
        "version": 1,
        "transformations": [],
    }
    st.session_state.datasets[name] = record
    st.session_state.active_dataset = name
    # Init version history
    if name not in st.session_state.dataset_versions:
        st.session_state.dataset_versions[name] = []
    _snapshot_version(name, "Initial load")
    return record

def _snapshot_version(name, description=""):
    record = st.session_state.datasets.get(name)
    if not record:
        return
    snapshot = {
        "version": record["version"],
        "timestamp": datetime.now(),
        "rows": record["rows"],
        "cols": record["cols"],
        "description": description,
        "df_snapshot": record["df"].copy(),
    }
    st.session_state.dataset_versions[name].append(snapshot)

def save_version(name, description=""):
    """Bump version and snapshot."""
    record = st.session_state.datasets.get(name)
    if not record:
        return
    record["version"] += 1
    _snapshot_version(name, description)

def add_query_to_history(question, code, result_summary, dataset_name):
    entry = {
        "id": f"q_{len(st.session_state.query_history)}",
        "timestamp": datetime.now(),
        "question": question,
        "code": code,
        "result_summary": result_summary,
        "dataset": dataset_name,
    }
    st.session_state.query_history.append(entry)
    return entry
