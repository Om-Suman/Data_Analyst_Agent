"""
Thread-safe session state manager replacing Streamlit's st.session_state.
Supports multi-user sessions with an isolated state per session_id.
"""
from __future__ import annotations

import os
import threading
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Any, Optional
import pandas as pd


from modules.llm_client import _resolve_api_key


class SessionState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self._lock = threading.RLock()

        # Core registries
        self.datasets: dict[str, dict[str, Any]] = {}
        self.active_dataset: Optional[str] = None
        self.dataset_versions: dict[str, list[dict[str, Any]]] = {}
        self.query_history: list[dict[str, Any]] = []
        self.document_indexes: dict[str, dict[str, Any]] = {}
        self.cleaning_log: list[dict[str, Any]] = []
        self.last_insights: dict[str, Any] = {}
        self.cached_forecast: Optional[Any] = None
        self.cached_anomaly: Optional[Any] = None
        self.viz_history: list[dict[str, Any]] = []
        self.pinned_charts: list[dict[str, Any]] = []

        # Configuration defaults
        self.config: dict[str, Any] = {
            "hf_api_key": _resolve_api_key(),
            "primary_model": os.environ.get("PRIMARY_MODEL", "deepseek-ai/DeepSeek-R1"),
            "fallback_model": os.environ.get("FALLBACK_MODEL", ""),
            "theme": "dark",
            "max_tokens": int(os.environ.get("MAX_TOKENS", 2048)),
            "temperature": float(os.environ.get("TEMPERATURE", 0.3)),
            "default_outlier": "none",
            "default_missing_threshold": 50,
            "auto_profile": False,
            "auto_insights": False,
        }

    def touch(self):
        self.last_accessed = datetime.now()

    def get_active_df(self) -> Optional[pd.DataFrame]:
        with self._lock:
            self.touch()
            if not self.active_dataset or self.active_dataset not in self.datasets:
                return None
            return self.datasets[self.active_dataset].get("df")

    def get_active_meta(self) -> dict[str, Any]:
        with self._lock:
            self.touch()
            if not self.active_dataset or self.active_dataset not in self.datasets:
                return {}
            return self.datasets[self.active_dataset].get("meta", {})

    def get_active_record(self) -> Optional[dict[str, Any]]:
        with self._lock:
            self.touch()
            if not self.active_dataset or self.active_dataset not in self.datasets:
                return None
            return self.datasets[self.active_dataset]

    def register_dataset(
        self,
        name: str,
        df: pd.DataFrame,
        source: str = "upload",
        meta: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        with self._lock:
            self.touch()
            meta = meta or {}
            record = {
                "df": df,
                "name": name,
                "source": source,
                "uploaded_at": datetime.now(),
                "rows": int(len(df)),
                "cols": int(len(df.columns)),
                "meta": meta,
                "text_content": None,
                "version": 1,
                "transformations": [],
            }
            self.datasets[name] = record
            self.active_dataset = name

            if name not in self.dataset_versions:
                self.dataset_versions[name] = []
            self._snapshot_version(name, "Initial load")
            return record

    def register_text_dataset(
        self,
        name: str,
        content: str,
        source: str = "upload",
        meta: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        with self._lock:
            self.touch()
            meta = meta or {}
            record = {
                "df": None,
                "name": name,
                "source": source,
                "uploaded_at": datetime.now(),
                "rows": 0,
                "cols": 0,
                "meta": meta,
                "text_content": content,
                "version": 1,
                "transformations": [],
            }
            self.datasets[name] = record
            self.active_dataset = name
            return record

    def _snapshot_version(self, name: str, description: str = ""):
        record = self.datasets.get(name)
        if not record or record.get("df") is None:
            return
        snapshot = {
            "version": int(record["version"]),
            "timestamp": datetime.now(),
            "rows": int(record["rows"]),
            "cols": int(record["cols"]),
            "description": description,
            "df_snapshot": record["df"].copy(),
        }
        self.dataset_versions[name].append(snapshot)

    def save_version(self, name: str, description: str = ""):
        with self._lock:
            self.touch()
            record = self.datasets.get(name)
            if not record or record.get("df") is None:
                return
            record["version"] += 1
            self._snapshot_version(name, description)

    def update_dataset(self, name: str, df: pd.DataFrame, description: str = ""):
        with self._lock:
            self.touch()
            if name in self.datasets:
                self.datasets[name]["df"] = df
                self.datasets[name]["rows"] = int(len(df))
                self.datasets[name]["cols"] = int(len(df.columns))
                self.save_version(name, description)

    def rollback_version(self, name: str, version_number: int) -> bool:
        with self._lock:
            self.touch()
            if name not in self.datasets or name not in self.dataset_versions:
                return False
            versions = self.dataset_versions[name]
            target_snap = None
            for snap in versions:
                if snap["version"] == version_number:
                    target_snap = snap
                    break
            if target_snap is None:
                return False

            df_restored = target_snap["df_snapshot"].copy()
            self.datasets[name]["df"] = df_restored
            self.datasets[name]["rows"] = int(len(df_restored))
            self.datasets[name]["cols"] = int(len(df_restored.columns))
            self.datasets[name]["version"] = int(target_snap["version"])
            return True

    def add_query_to_history(
        self,
        question: str,
        code: str,
        result_summary: str,
        dataset_name: str,
        execution_results: Optional[list] = None,
        route: Optional[str] = None,
        model_used: Optional[str] = None,
    ) -> dict[str, Any]:
        with self._lock:
            self.touch()
            entry = {
                "id": f"q_{uuid.uuid4().hex[:8]}",
                "timestamp": datetime.now(),
                "question": question,
                "code": code,
                "result_summary": result_summary,
                "dataset": dataset_name,
                "route": route or "dataframe_analysis",
                "model_used": model_used or "",
            }
            self.query_history.append(entry)
            return entry

    def add_pinned_chart(self, title: str, chart_type: str, figure_spec: dict[str, Any], source_page: str = "Visualizations", notes: str = "") -> dict[str, Any]:
        with self._lock:
            self.touch()
            item = {
                "id": str(uuid.uuid4())[:8],
                "title": title,
                "chart_type": chart_type,
                "figure_spec": figure_spec,
                "pinned_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_page": source_page,
                "notes": notes,
            }
            self.pinned_charts.insert(0, item)
            return item

    def remove_pinned_chart(self, pin_id: str) -> bool:
        with self._lock:
            self.touch()
            initial_len = len(self.pinned_charts)
            self.pinned_charts = [p for p in self.pinned_charts if p["id"] != pin_id]
            return len(self.pinned_charts) < initial_len

    def get_pinned_charts(self) -> list[dict[str, Any]]:
        with self._lock:
            self.touch()
            return list(self.pinned_charts)

    def clear_query_history(self):
        with self._lock:
            self.touch()
            self.query_history.clear()

    def reset(self):
        with self._lock:
            self.datasets.clear()
            self.active_dataset = None
            self.dataset_versions.clear()
            self.query_history.clear()
            self.document_indexes.clear()
            self.cleaning_log.clear()
            self.last_insights.clear()
            self.cached_forecast = None
            self.cached_anomaly = None
            self.viz_history.clear()
            self.pinned_charts.clear()
            self.touch()


class SessionManager:
    _instance: Optional[SessionManager] = None
    _lock = threading.Lock()

    def __new__(cls) -> SessionManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._sessions = {}
                cls._instance._manager_lock = threading.RLock()
            return cls._instance

    def get_session(self, session_id: str = "default") -> SessionState:
        with self._manager_lock:
            sid = session_id.strip() if session_id and session_id.strip() else "default"
            if sid not in self._sessions:
                self._sessions[sid] = SessionState(session_id=sid)
            return self._sessions[sid]

    def remove_session(self, session_id: str) -> bool:
        with self._manager_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False


def get_session(session_id: str = "default") -> SessionState:
    return SessionManager().get_session(session_id)
