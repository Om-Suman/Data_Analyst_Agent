import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  Database,
  Columns,
  AlertCircle,
  Copy,
  Sparkles,
  ArrowRight,
  TrendingUp,
  BarChart2,
  FileSpreadsheet,
  Terminal,
  Layers,
  Wrench,
  Lightbulb,
} from "lucide-react";
import { useDataset } from "../context/DatasetContext";
import { MetricCard } from "../components/MetricCard";
import { PlotlyChart } from "../components/PlotlyChart";
import { DataTable } from "../components/DataTable";
import { queryApi } from "../api/client";
import { QueryHistoryItem } from "../types";

export const DashboardPage: React.FC = () => {
  const { activeDataset, preview, previewLoading, hasDataset, pinnedCharts } =
    useDataset();
  const [history, setHistory] = useState<QueryHistoryItem[]>([]);

  useEffect(() => {
    queryApi
      .getHistory()
      .then((res) => setHistory(res.history.slice(0, 5)))
      .catch((err) => console.error(err));
  }, []);

  if (!hasDataset) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[70vh] text-center space-y-6">
        <div className="h-20 w-20 rounded-3xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-500 text-4xl shadow-xl shadow-blue-500/10">
          ⚡
        </div>
        <div className="space-y-2 max-w-lg">
          <h2 className="text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white">
            Enterprise Data Analytics Suite
          </h2>
          <p className="text-sm sm:text-base text-slate-500 dark:text-slate-400">
            To start querying, exploring, and building automated dashboards,
            please upload a dataset or load one of the instant demo datasets.
          </p>
        </div>
        <div className="flex flex-wrap items-center justify-center gap-3">
          <Link
            to="/upload"
            className="flex items-center gap-2 px-6 py-3 rounded-2xl bg-blue-600 hover:bg-blue-500 text-white text-sm font-bold shadow-lg shadow-blue-600/20 transition-all"
          >
            Upload Data File
            <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      </div>
    );
  }

  const meta = preview?.metadata || {};
  const missingTotal = meta.missing_total ?? 0;
  const duplicateTotal = meta.duplicate_rows ?? 0;

  // Dtype composition pie spec
  const numCount = preview?.numeric_cols?.length || 0;
  const catCount = preview?.categorical_cols?.length || 0;
  const dateCount = preview?.date_cols?.length || 0;
  const otherCount = Math.max(
    0,
    (preview?.cols || 0) - numCount - catCount - dateCount,
  );

  const dtypePieSpec = {
    data: [
      {
        values: [numCount, catCount, dateCount, otherCount].filter(
          (v) => v > 0,
        ),
        labels: ["Numeric", "Categorical", "Date/Time", "Other"].slice(
          0,
          [numCount, catCount, dateCount, otherCount].filter((v) => v > 0)
            .length,
        ),
        type: "pie",
        hole: 0.55,
        marker: { colors: ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"] },
        textinfo: "label+percent",
      },
    ],
    layout: {
      title: "Column Types Composition",
      height: 280,
      margin: { l: 20, r: 20, t: 40, b: 20 },
      showlegend: false,
    },
  };

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="relative overflow-hidden flex flex-col lg:flex-row items-start lg:items-center justify-between gap-5 p-6 sm:p-7 rounded-3xl bg-gradient-to-br from-slate-950 via-slate-900 to-blue-950 border border-blue-400/30 shadow-lg shadow-slate-900/10">
        <div className="relative min-w-0">
          <span className="text-[11px] font-bold text-blue-300 uppercase tracking-[0.16em] flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse"></span>
            Active Workspace
          </span>
          <h2 className="text-3xl sm:text-4xl font-black text-white tracking-tight leading-tight mt-2 drop-shadow-sm">
            {activeDataset?.name}
          </h2>
          <div className="flex flex-wrap items-center gap-2 mt-3 text-[11px] sm:text-xs font-medium text-slate-200">
            <span className="rounded-lg bg-white/10 border border-white/10 px-2.5 py-1">
              Source:{" "}
              <span className="text-white font-mono">
                {activeDataset?.source}
              </span>
            </span>
            <span className="rounded-lg bg-blue-400/15 border border-blue-300/20 px-2.5 py-1">
              Version{" "}
              <span className="text-blue-200 font-mono font-bold">
                v{activeDataset?.version}
              </span>
            </span>
            <span className="rounded-lg bg-emerald-400/10 border border-emerald-300/20 px-2.5 py-1 text-emerald-100">
              {activeDataset?.rows.toLocaleString()} records
            </span>
          </div>
        </div>

        <div className="relative flex flex-wrap items-center gap-2.5 w-full lg:w-auto">
          <Link
            to="/sql"
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs sm:text-sm font-semibold border border-slate-700 transition-all"
          >
            <Terminal className="h-4 w-4 text-emerald-400" />
            SQL Studio
          </Link>
          <Link
            to="/query"
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs sm:text-sm font-bold shadow-md shadow-blue-600/20 transition-all"
          >
            <Sparkles className="h-4 w-4" />
            AI Query
          </Link>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Total Rows"
          value={activeDataset?.rows.toLocaleString() || "0"}
          subtitle="Dataset record volume"
          icon={Database}
          color="blue"
        />
        <MetricCard
          title="Total Columns"
          value={activeDataset?.cols.toLocaleString() || "0"}
          subtitle={`${preview?.numeric_cols.length || 0} numeric, ${preview?.categorical_cols.length || 0} categorical`}
          icon={Columns}
          color="green"
        />
        <MetricCard
          title="Missing Values"
          value={missingTotal.toLocaleString()}
          subtitle={`${((missingTotal / Math.max(1, (activeDataset?.rows || 1) * (activeDataset?.cols || 1))) * 100).toFixed(1)}% of all cells`}
          icon={AlertCircle}
          color={missingTotal > 0 ? "amber" : "green"}
        />
        <MetricCard
          title="Duplicate Rows"
          value={duplicateTotal.toLocaleString()}
          subtitle="Identical row signatures"
          icon={Copy}
          color={duplicateTotal > 0 ? "red" : "green"}
        />
      </div>

      {/* Analytics Summary & Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 items-start gap-6">
        <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 p-5 space-y-3 shadow-sm">
          <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <BarChart2 className="h-4 w-4 text-blue-500" />
            Schema Breakdown
          </h3>
          <PlotlyChart spec={dtypePieSpec} height={260} showActions={false} />
        </div>

        {/* Quick Suite Navigation Hub */}
        <div className="lg:col-span-2 min-h-[450px] flex flex-col rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 p-5 space-y-4 shadow-sm">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-emerald-500" />
              Enterprise Analytics Hub
            </h3>
            {pinnedCharts.length > 0 && (
              <Link
                to="/custom-dashboard"
                className="text-xs text-blue-600 dark:text-blue-400 hover:underline font-semibold flex items-center gap-1"
              >
                View Pinned Canvas ({pinnedCharts.length})
                <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            )}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5">
            <Link
              to="/cleaning"
              className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/40 hover:bg-slate-100 dark:hover:bg-slate-800/80 border border-slate-200 dark:border-slate-700/60 transition-all space-y-2 block"
            >
              <div className="flex items-center gap-2 font-bold text-xs sm:text-sm text-slate-900 dark:text-white">
                <Wrench className="h-4 w-4 text-blue-500" />
                Cleaning Studio
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                Transform columns, impute nulls, and snapshot dataset versions.
              </p>
            </Link>

            <Link
              to="/sql"
              className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/40 hover:bg-slate-100 dark:hover:bg-slate-800/80 border border-slate-200 dark:border-slate-700/60 transition-all space-y-2 block"
            >
              <div className="flex items-center gap-2 font-bold text-xs sm:text-sm text-slate-900 dark:text-white">
                <Terminal className="h-4 w-4 text-emerald-500" />
                SQL Studio
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                Query dataset in memory with sub-millisecond SQLite execution.
              </p>
            </Link>

            <Link
              to="/visualizations"
              className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/40 hover:bg-slate-100 dark:hover:bg-slate-800/80 border border-slate-200 dark:border-slate-700/60 transition-all space-y-2 block"
            >
              <div className="flex items-center gap-2 font-bold text-xs sm:text-sm text-slate-900 dark:text-white">
                <BarChart2 className="h-4 w-4 text-indigo-500" />
                Visualizations
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                Generate 14+ interactive Plotly charts, heatmaps, and pin them.
              </p>
            </Link>
          </div>

          <div className="mt-auto pt-5 border-t border-slate-200 dark:border-slate-800">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3">
              <div>
                <p className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 flex items-center gap-1.5">
                  <Lightbulb className="h-3.5 w-3.5 text-amber-500" />
                  Workspace Pulse
                </p>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                  A quick read of the active dataset before you dive in.
                </p>
              </div>
              <Link
                to="/explorer"
                className="text-xs font-semibold text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 flex items-center gap-1"
              >
                Open Explorer <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
              <div className="rounded-xl bg-blue-50/70 dark:bg-blue-950/20 px-3.5 py-3">
                <span className="block text-slate-500 dark:text-slate-400">
                  Data volume
                </span>
                <strong className="block text-slate-900 dark:text-white mt-1 font-mono">
                  {activeDataset?.rows.toLocaleString()} rows
                </strong>
              </div>
              <div className="rounded-xl bg-emerald-50/70 dark:bg-emerald-950/20 px-3.5 py-3">
                <span className="block text-slate-500 dark:text-slate-400">
                  Schema mix
                </span>
                <strong className="block text-slate-900 dark:text-white mt-1">
                  {numCount} numeric / {catCount} categorical
                </strong>
              </div>
              <div className="rounded-xl bg-amber-50/70 dark:bg-amber-950/20 px-3.5 py-3">
                <span className="block text-slate-500 dark:text-slate-400">
                  Data quality
                </span>
                <strong className="block text-slate-900 dark:text-white mt-1">
                  {missingTotal === 0
                    ? "No missing values"
                    : `${missingTotal.toLocaleString()} missing values`}
                </strong>
              </div>
            </div>
          </div>

          {/* Latest Query Activity */}
          {history.length > 0 && (
            <div className="pt-3 border-t border-slate-200 dark:border-slate-800">
              <span className="text-xs text-slate-400 font-bold uppercase tracking-wider">
                Recent AI Query
              </span>
              <p className="text-xs sm:text-sm text-blue-600 dark:text-blue-300 font-semibold mt-1">
                "{history[0].question}"
              </p>
              <p className="text-xs text-slate-500 dark:text-slate-400 line-clamp-2 mt-0.5">
                {history[0].result_summary}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Dataset Preview Table */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 p-5 space-y-4 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-bold text-slate-900 dark:text-white">
              Dataset Preview (Active Snapshot)
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              First 50 records rendered from in-memory engine
            </p>
          </div>
          <Link
            to="/explorer"
            className="text-xs text-blue-600 dark:text-blue-400 hover:underline flex items-center gap-1 font-semibold"
          >
            Open Advanced Explorer
            <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>

        {previewLoading ? (
          <div className="h-40 flex items-center justify-center text-slate-500 text-xs">
            Loading preview...
          </div>
        ) : (
          <DataTable
            data={preview?.data || []}
            columns={preview?.columns}
            pageSize={10}
          />
        )}
      </div>
    </div>
  );
};
