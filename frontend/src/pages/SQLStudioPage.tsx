import React, { useState, useEffect, useRef } from "react";
import {
  Terminal,
  Play,
  Download,
  AlertCircle,
  Clock,
  Table,
  CheckCircle2,
  Copy,
  BookOpen,
} from "lucide-react";
import { useDataset } from "../context/DatasetContext";
import { queryApi } from "../api/client";
import { SQLQueryResponse } from "../types";
import { DataTable } from "../components/DataTable";
import { useToast } from "../components/Toast";

export const SQLStudioPage: React.FC = () => {
  const { preview, hasDataset, activeDataset } = useDataset();
  const { success, error: toastError } = useToast();

  const [query, setQuery] = useState("SELECT * FROM df LIMIT 25;");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SQLQueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const autoRunStarted = useRef(false);

  const columns = preview?.columns || [];

  const snippets = [
    { label: "Sample 20 Rows", sql: "SELECT * FROM df LIMIT 20;" },
    {
      label: "Count & Summary",
      sql: "SELECT COUNT(*) as total_records FROM df;",
    },
    {
      label: "Group By Aggregation",
      sql:
        columns.length >= 2
          ? `SELECT ${columns[0]}, COUNT(*) as count FROM df GROUP BY ${columns[0]} ORDER BY count DESC LIMIT 10;`
          : "SELECT * FROM df LIMIT 10;",
    },
    {
      label: "Numeric Stats",
      sql:
        preview?.numeric_cols && preview.numeric_cols.length > 0
          ? `SELECT AVG(${preview.numeric_cols[0]}) as avg_val, MIN(${preview.numeric_cols[0]}) as min_val, MAX(${preview.numeric_cols[0]}) as max_val FROM df;`
          : "SELECT * FROM df LIMIT 10;",
    },
  ];

  const handleRunSQL = async () => {
    if (!hasDataset || !query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await queryApi.runSQL({ query, limit: 500 });
      if (res.success) {
        setResult(res);
        success(
          "Query Executed",
          `Returned ${res.total_rows} rows in ${res.execution_time}ms`,
        );
      } else {
        setError(res.error || "SQL Execution failed.");
        toastError("SQL Error", res.error || "Syntax or validation error.");
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to execute query.");
      toastError("Execution Error", err.response?.data?.detail);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (hasDataset && !result && !autoRunStarted.current) {
      autoRunStarted.current = true;
      handleRunSQL();
    }
  }, [hasDataset]);

  if (!hasDataset) {
    return (
      <div className="p-8 text-center text-slate-500 border border-slate-200 dark:border-slate-800 rounded-2xl bg-slate-100/50 dark:bg-slate-900/30">
        Please load or select a dataset first to query it via SQL Studio.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white tracking-tight flex items-center gap-2.5">
          <Terminal className="h-6 w-6 text-emerald-500" />
          In-Memory SQL Studio
        </h2>
        <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-1">
          Execute fast, interactive SQL queries directly over{" "}
          <span className="font-mono text-emerald-600 dark:text-emerald-400 font-bold">
            df
          </span>{" "}
          /{" "}
          <span className="font-mono text-emerald-600 dark:text-emerald-400 font-bold">
            data
          </span>{" "}
          table in memory
        </p>
      </div>

      {/* SQL Editor Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Editor Box */}
        <div className="lg:col-span-3 space-y-4">
          <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 p-4 sm:p-5 shadow-sm space-y-3">
            {/* Snippet pills */}
            <div className="flex flex-wrap items-center gap-2 pb-2 border-b border-slate-200 dark:border-slate-800/80">
              <span className="text-xs font-semibold text-slate-400 flex items-center gap-1">
                <BookOpen className="h-3.5 w-3.5" />
                Templates:
              </span>
              {snippets.map((s, idx) => (
                <button
                  key={idx}
                  onClick={() => setQuery(s.sql)}
                  className="text-xs px-2.5 py-1 rounded-lg bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 font-medium transition-colors"
                >
                  {s.label}
                </button>
              ))}
            </div>

            {/* Query TextArea */}
            <div className="relative rounded-xl overflow-hidden border border-slate-300 dark:border-slate-700 bg-slate-950">
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                rows={5}
                placeholder="SELECT * FROM df WHERE ..."
                className="w-full p-4 bg-transparent font-mono text-sm text-emerald-300 placeholder-slate-600 focus:outline-none resize-y leading-relaxed"
              />
            </div>

            {/* Run Button & Status */}
            <div className="flex items-center justify-between pt-1">
              <span className="text-xs text-slate-400 font-mono">
                Dataset table name:{" "}
                <code className="text-emerald-500 font-bold">df</code>
              </span>

              <button
                onClick={handleRunSQL}
                disabled={loading || !query.trim()}
                className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white text-xs sm:text-sm font-bold shadow-md shadow-emerald-600/20 transition-all"
              >
                <Play className="h-4 w-4" />
                {loading ? "Running Query..." : "Run SQL Query"}
              </button>
            </div>
          </div>
        </div>

        {/* Schema Reference Panel */}
        <div className="lg:col-span-1 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 p-4 sm:p-5 space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-slate-200 dark:border-slate-800">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 flex items-center gap-1.5">
              <Table className="h-4 w-4 text-emerald-500" />
              Table Schema ({columns.length} cols)
            </span>
          </div>

          <div className="space-y-1.5 max-h-[260px] overflow-y-auto pr-1">
            {columns.map((col) => {
              const isNum = preview?.numeric_cols?.includes(col);
              const isDate = preview?.date_cols?.includes(col);
              return (
                <button
                  key={col}
                  onClick={() => setQuery((prev) => `${prev} ${col}`)}
                  title="Click to append column name to query"
                  className="w-full flex items-center justify-between p-2 rounded-lg bg-slate-50 dark:bg-slate-800/50 hover:bg-slate-100 dark:hover:bg-slate-800 border border-slate-200 dark:border-slate-800 text-left transition-colors"
                >
                  <span className="text-xs font-mono font-medium text-slate-800 dark:text-slate-200 truncate">
                    {col}
                  </span>
                  <span className="text-[10px] uppercase font-bold px-1.5 py-0.5 rounded bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-400">
                    {isNum ? "num" : isDate ? "date" : "str"}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="p-4 rounded-xl flex items-center gap-3 text-xs sm:text-sm font-medium border bg-rose-50 dark:bg-rose-950/40 border-rose-200 dark:border-rose-500/30 text-rose-800 dark:text-rose-300">
          <AlertCircle className="h-5 w-5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Results View */}
      {result && result.success && (
        <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 p-5 space-y-4 shadow-sm">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-200 dark:border-slate-800">
            <div>
              <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                Query Execution Results
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5 flex items-center gap-3">
                <span className="flex items-center gap-1 font-mono">
                  <Clock className="h-3.5 w-3.5 text-blue-500" />
                  {result.execution_time} ms
                </span>
                <span>•</span>
                <span className="font-mono">
                  {result.total_rows.toLocaleString()} total rows
                </span>
                <span>•</span>
                <span className="font-mono">
                  {result.columns.length} columns
                </span>
              </p>
            </div>
          </div>

          <DataTable
            data={result.rows}
            columns={result.columns}
            pageSize={10}
          />
        </div>
      )}
    </div>
  );
};
