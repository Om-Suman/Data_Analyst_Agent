import React, { useEffect, useState, useMemo, useRef } from "react";
import { Link } from "react-router-dom";
import Plotly from "plotly.js-dist-min";
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
  Download,
  Search,
  CheckCircle2,
  Activity,
  HardDrive,
  RefreshCw,
  ChevronRight,
  ShieldCheck,
  Zap,
} from "lucide-react";
import { useDataset } from "../context/DatasetContext";
import { MetricCard } from "../components/MetricCard";
import { PlotlyChart } from "../components/PlotlyChart";
import { DataTable } from "../components/DataTable";
import { datasetApi, queryApi } from "../api/client";
import { QueryHistoryItem } from "../types";

export const DashboardPage: React.FC = () => {
  const {
    activeDataset,
    preview,
    previewLoading,
    hasDataset,
    pinnedCharts,
    refreshDatasets,
    refreshPreview,
  } = useDataset();

  const [history, setHistory] = useState<QueryHistoryItem[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [loadingSample, setLoadingSample] = useState<string | null>(null);
  const schemaChartRef = useRef<HTMLDivElement>(null);

  const handleDownloadSchemaChart = () => {
    if (schemaChartRef.current) {
      const plotEl = schemaChartRef.current.querySelector(
        ".js-plotly-plot",
      ) as any;
      if (plotEl) {
        Plotly.downloadImage(plotEl, {
          format: "png",
          filename: `${activeDataset?.name || "dataset"}_schema_breakdown`,
          width: 900,
          height: 600,
          scale: 2,
        });
      }
    }
  };

  useEffect(() => {
    queryApi
      .getHistory()
      .then((res) => setHistory(res.history.slice(0, 5)))
      .catch((err) => console.error(err));
  }, []);

  const handleLoadDemo = async (sampleName: string) => {
    setLoadingSample(sampleName);
    try {
      await datasetApi.loadSample(sampleName);
      await refreshDatasets();
      await refreshPreview();
    } catch (err) {
      console.error("Failed to load demo sample", err);
    } finally {
      setLoadingSample(null);
    }
  };

  // Empty State when no dataset is loaded
  if (!hasDataset) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[75vh] text-center px-4 py-12 space-y-8">
        <div className="h-20 w-20 rounded-3xl bg-blue-500/10 dark:bg-blue-500/20 border border-blue-500/20 flex items-center justify-center text-blue-600 dark:text-blue-400 text-4xl shadow-xl shadow-blue-500/10">
          <Zap className="h-10 w-10 animate-pulse text-blue-600 dark:text-blue-400" />
        </div>

        <div className="space-y-3 max-w-xl">
          <h2 className="text-3xl sm:text-4xl font-black tracking-tight text-slate-900 dark:text-white">
            Enterprise Data Analytics Suite
          </h2>
          <p className="text-sm sm:text-base text-slate-600 dark:text-slate-400 leading-relaxed">
            Upload your data files or initialize an instant demo dataset to
            unlock natural language AI queries, in-memory SQL execution, and
            custom BI dashboards.
          </p>
        </div>

        <div className="flex flex-wrap items-center justify-center gap-4">
          <Link
            to="/upload"
            className="flex items-center gap-2 px-6 py-3.5 rounded-2xl bg-blue-600 hover:bg-blue-500 text-white text-sm font-bold shadow-lg shadow-blue-600/25 transition-all"
          >
            Upload Data File
            <ArrowRight className="h-4 w-4" />
          </Link>
        </div>

        {/* Demo Datasets Grid */}
        <div className="w-full max-w-4xl pt-6">
          <p className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-4">
            Or Quick-Start with Curated Demo Datasets
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-left">
            <button
              onClick={() => handleLoadDemo("Sales Data")}
              disabled={loadingSample !== null}
              className="p-5 rounded-2xl bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 hover:border-blue-400 dark:hover:border-blue-500/50 hover:shadow-md transition-all group flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold px-2 py-0.5 rounded-md bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300">
                    E-Commerce
                  </span>
                  <Database className="h-4 w-4 text-blue-500 group-hover:scale-110 transition-transform" />
                </div>
                <h4 className="font-bold text-sm text-slate-900 dark:text-white mt-3">
                  Sales & Revenue
                </h4>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 leading-relaxed">
                  500 daily sales transactions across products, categories, and
                  profit margins.
                </p>
              </div>
              <span className="text-xs font-semibold text-blue-600 dark:text-blue-400 mt-4 flex items-center gap-1 group-hover:translate-x-1 transition-transform">
                {loadingSample === "Sales Data" ? "Loading..." : "Load Sample"}
                <ArrowRight className="h-3 w-3" />
              </span>
            </button>

            <button
              onClick={() => handleLoadDemo("Employee Data")}
              disabled={loadingSample !== null}
              className="p-5 rounded-2xl bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 hover:border-emerald-400 dark:hover:border-emerald-500/50 hover:shadow-md transition-all group flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold px-2 py-0.5 rounded-md bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300">
                    HR & Org
                  </span>
                  <Database className="h-4 w-4 text-emerald-500 group-hover:scale-110 transition-transform" />
                </div>
                <h4 className="font-bold text-sm text-slate-900 dark:text-white mt-3">
                  Employee HR
                </h4>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 leading-relaxed">
                  300 employee records with salaries, departments, experience,
                  and remote status.
                </p>
              </div>
              <span className="text-xs font-semibold text-emerald-600 dark:text-emerald-400 mt-4 flex items-center gap-1 group-hover:translate-x-1 transition-transform">
                {loadingSample === "Employee Data"
                  ? "Loading..."
                  : "Load Sample"}
                <ArrowRight className="h-3 w-3" />
              </span>
            </button>

            <button
              onClick={() => handleLoadDemo("Finance Data")}
              disabled={loadingSample !== null}
              className="p-5 rounded-2xl bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 hover:border-amber-400 dark:hover:border-amber-500/50 hover:shadow-md transition-all group flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold px-2 py-0.5 rounded-md bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300">
                    Markets
                  </span>
                  <Database className="h-4 w-4 text-amber-500 group-hover:scale-110 transition-transform" />
                </div>
                <h4 className="font-bold text-sm text-slate-900 dark:text-white mt-3">
                  Finance & Stock
                </h4>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 leading-relaxed">
                  365 daily stock price movements with trading volume, high,
                  low, and closing values.
                </p>
              </div>
              <span className="text-xs font-semibold text-amber-600 dark:text-amber-400 mt-4 flex items-center gap-1 group-hover:translate-x-1 transition-transform">
                {loadingSample === "Finance Data"
                  ? "Loading..."
                  : "Load Sample"}
                <ArrowRight className="h-3 w-3" />
              </span>
            </button>
          </div>
        </div>
      </div>
    );
  }

  const meta = preview?.metadata || {};
  const missingTotal = meta.missing_total ?? 0;
  const duplicateTotal = meta.duplicate_rows ?? 0;
  const totalRows = activeDataset?.rows || 1;
  const totalCols = activeDataset?.cols || 1;
  const totalCells = totalRows * totalCols;
  const completenessPct = Math.max(
    0,
    100 - (missingTotal / Math.max(1, totalCells)) * 100,
  ).toFixed(1);
  const uniquenessPct = Math.max(
    0,
    100 - (duplicateTotal / Math.max(1, totalRows)) * 100,
  ).toFixed(1);

  // Dtype composition
  const numCount = preview?.numeric_cols?.length || 0;
  const catCount = preview?.categorical_cols?.length || 0;
  const dateCount = preview?.date_cols?.length || 0;
  const otherCount = Math.max(0, totalCols - numCount - catCount - dateCount);

  // Filter preview data by search query
  const filteredPreviewData = useMemo(() => {
    if (!preview?.data) return [];
    if (!searchQuery.trim()) return preview.data;
    const q = searchQuery.toLowerCase();
    return preview.data.filter((row) =>
      Object.values(row).some((val) => String(val).toLowerCase().includes(q)),
    );
  }, [preview?.data, searchQuery]);

  const dtypePieSpec = {
    data: [
      {
        values: [numCount, catCount, dateCount, otherCount].filter(
          (v) => v > 0,
        ),
        labels: ["Numeric", "Categorical", "Date/Time", "Other"].filter(
          (_, i) => [numCount, catCount, dateCount, otherCount][i] > 0,
        ),
        type: "pie",
        hole: 0.6,
        marker: { colors: ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"] },
        textinfo: "percent",
        hoverinfo: "label+value+percent",
      },
    ],
    layout: {
      height: 200,
      margin: { l: 15, r: 15, t: 10, b: 10 },
      showlegend: false,
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
    },
  };

  return (
    <div className="space-y-6">
      {/* Top Active Workspace Banner */}
      <div className="relative overflow-hidden flex flex-col xl:flex-row items-start xl:items-center justify-between gap-5 p-6 sm:p-7 rounded-3xl bg-gradient-to-r from-blue-50/80 via-white to-slate-50 dark:from-slate-950 dark:via-slate-900 dark:to-blue-950/70 border border-slate-200 dark:border-blue-500/20 shadow-sm dark:shadow-slate-950/40">
        <div className="relative min-w-0 max-w-3xl">
          <div className="flex items-center gap-2.5">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-100 dark:bg-emerald-500/15 border border-emerald-200 dark:border-emerald-500/30 text-[11px] font-bold text-emerald-700 dark:text-emerald-300 uppercase tracking-wider">
              <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
              Active Workspace
            </span>
            <span className="text-xs text-slate-500 dark:text-slate-400 font-mono">
              Version v{activeDataset?.version}
            </span>
          </div>

          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-black text-slate-900 dark:text-white tracking-tight leading-tight mt-2.5 truncate">
            {activeDataset?.name}
          </h1>

          <div className="flex flex-wrap items-center gap-2 mt-3 text-xs">
            <span className="rounded-lg bg-slate-100 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700/80 px-2.5 py-1 text-slate-700 dark:text-slate-300 font-medium">
              Source:{" "}
              <span className="font-mono text-slate-900 dark:text-white font-bold">
                {activeDataset?.source}
              </span>
            </span>
            <span className="rounded-lg bg-blue-50 dark:bg-blue-500/10 border border-blue-200 dark:border-blue-400/20 px-2.5 py-1 text-blue-700 dark:text-blue-300 font-semibold font-mono">
              {activeDataset?.rows.toLocaleString()} rows ×{" "}
              {activeDataset?.cols.toLocaleString()} columns
            </span>
            {meta.memory_mb && (
              <span className="rounded-lg bg-purple-50 dark:bg-purple-500/10 border border-purple-200 dark:border-purple-400/20 px-2.5 py-1 text-purple-700 dark:text-purple-300 font-medium flex items-center gap-1">
                <HardDrive className="h-3 w-3" />
                {meta.memory_mb} MB RAM
              </span>
            )}
            <span className="rounded-lg bg-emerald-50 dark:bg-emerald-500/10 border border-emerald-200 dark:border-emerald-400/20 px-2.5 py-1 text-emerald-700 dark:text-emerald-300 font-medium flex items-center gap-1">
              <ShieldCheck className="h-3.5 w-3.5 text-emerald-500" />
              {completenessPct}% Complete
            </span>
          </div>
        </div>

        {/* Quick Actions / Exports */}
        <div className="relative flex flex-wrap items-center gap-2.5 w-full xl:w-auto pt-2 xl:pt-0">
          <a
            href={datasetApi.getDownloadCsvUrl()}
            download={`${activeDataset?.name || "dataset"}.csv`}
            className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-white dark:bg-slate-800/80 hover:bg-slate-50 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 text-xs font-bold border border-slate-200 dark:border-slate-700 shadow-sm transition-all"
            title="Download full dataset as CSV"
          >
            <Download className="h-3.5 w-3.5 text-blue-500" />
            CSV
          </a>
          <a
            href={datasetApi.getDownloadExcelUrl()}
            download={`${activeDataset?.name || "dataset"}.xlsx`}
            className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-white dark:bg-slate-800/80 hover:bg-slate-50 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 text-xs font-bold border border-slate-200 dark:border-slate-700 shadow-sm transition-all"
            title="Download full dataset as Excel workbook"
          >
            <FileSpreadsheet className="h-3.5 w-3.5 text-emerald-500" />
            Excel
          </a>
          <Link
            to="/sql"
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 dark:bg-slate-800 dark:hover:bg-slate-700 text-white text-xs font-bold shadow-sm transition-all"
          >
            <Terminal className="h-3.5 w-3.5 text-emerald-400" />
            SQL Studio
          </Link>
          <Link
            to="/query"
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-bold shadow-md shadow-blue-600/20 transition-all"
          >
            <Sparkles className="h-3.5 w-3.5" />
            AI Query
          </Link>
        </div>
      </div>

      {/* Top 4 KPI Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Total Records"
          value={activeDataset?.rows.toLocaleString() || "0"}
          subtitle={
            meta.memory_mb
              ? `${meta.memory_mb} MB in RAM`
              : "Active in-memory records"
          }
          icon={Database}
          color="blue"
        />
        <MetricCard
          title="Total Columns"
          value={activeDataset?.cols.toLocaleString() || "0"}
          subtitle={`${numCount} numeric • ${catCount} categorical • ${dateCount} date`}
          icon={Columns}
          color="green"
        />
        <MetricCard
          title="Data Completeness"
          value={`${completenessPct}%`}
          subtitle={
            missingTotal === 0
              ? "0 missing values across all cells"
              : `${missingTotal.toLocaleString()} missing cells`
          }
          icon={CheckCircle2}
          color={Number(completenessPct) >= 98 ? "green" : "amber"}
        />
        <MetricCard
          title="Row Uniqueness"
          value={duplicateTotal === 0 ? "100%" : `${uniquenessPct}%`}
          subtitle={
            duplicateTotal === 0
              ? "No duplicate rows found"
              : `${duplicateTotal.toLocaleString()} identical duplicate rows`
          }
          icon={Copy}
          color={duplicateTotal > 0 ? "red" : "purple"}
        />
      </div>

      {/* Middle Section: Balanced 2-Column Grid eliminating empty space */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* Left Column (5 Cols): Schema Breakdown & Quality Profile */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          {/* Card 1: Schema Breakdown */}
          <div className="flex-1 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 p-5 space-y-4 shadow-sm flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
                  <BarChart2 className="h-4 w-4 text-blue-500" />
                  Schema Breakdown
                </h3>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300">
                    {totalCols} Columns
                  </span>
                  <button
                    onClick={handleDownloadSchemaChart}
                    title="Download Schema Breakdown PNG"
                    className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-slate-100 dark:bg-slate-800/90 hover:bg-blue-50 dark:hover:bg-blue-950/40 border border-slate-200 dark:border-slate-700/80 text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 text-xs font-bold transition-all shadow-sm group"
                  >
                    <Download className="h-3 w-3 text-blue-500 group-hover:scale-110 transition-transform" />
                    PNG
                  </button>
                </div>
              </div>

              <div ref={schemaChartRef} className="mt-1">
                <PlotlyChart
                  spec={dtypePieSpec}
                  height={190}
                  showActions={false}
                />
              </div>

              {/* Detailed Column Classification Badges */}
              <div className="grid grid-cols-2 gap-2 pt-2">
                <div className="p-2.5 rounded-xl bg-blue-50/60 dark:bg-blue-950/30 border border-blue-100 dark:border-blue-900/30">
                  <div className="flex items-center justify-between text-xs font-bold text-blue-900 dark:text-blue-300">
                    <span>Numeric</span>
                    <span className="font-mono">{numCount}</span>
                  </div>
                  <p className="text-[11px] text-blue-700/80 dark:text-blue-400/80 truncate mt-0.5">
                    {preview?.numeric_cols?.slice(0, 3).join(", ") || "None"}
                    {numCount > 3 ? "..." : ""}
                  </p>
                </div>

                <div className="p-2.5 rounded-xl bg-emerald-50/60 dark:bg-emerald-950/30 border border-emerald-100 dark:border-emerald-900/30">
                  <div className="flex items-center justify-between text-xs font-bold text-emerald-900 dark:text-emerald-300">
                    <span>Categorical</span>
                    <span className="font-mono">{catCount}</span>
                  </div>
                  <p className="text-[11px] text-emerald-700/80 dark:text-emerald-400/80 truncate mt-0.5">
                    {preview?.categorical_cols?.slice(0, 3).join(", ") ||
                      "None"}
                    {catCount > 3 ? "..." : ""}
                  </p>
                </div>

                <div className="p-2.5 rounded-xl bg-amber-50/60 dark:bg-amber-950/30 border border-amber-100 dark:border-amber-900/30">
                  <div className="flex items-center justify-between text-xs font-bold text-amber-900 dark:text-amber-300">
                    <span>Date / Time</span>
                    <span className="font-mono">{dateCount}</span>
                  </div>
                  <p className="text-[11px] text-amber-700/80 dark:text-amber-400/80 truncate mt-0.5">
                    {preview?.date_cols?.slice(0, 2).join(", ") || "None"}
                    {dateCount > 2 ? "..." : ""}
                  </p>
                </div>

                <div className="p-2.5 rounded-xl bg-purple-50/60 dark:bg-purple-950/30 border border-purple-100 dark:border-purple-900/30">
                  <div className="flex items-center justify-between text-xs font-bold text-purple-900 dark:text-purple-300">
                    <span>Other / Text</span>
                    <span className="font-mono">{otherCount}</span>
                  </div>
                  <p className="text-[11px] text-purple-700/80 dark:text-purple-400/80 truncate mt-0.5">
                    {otherCount > 0 ? `${otherCount} columns` : "None"}
                  </p>
                </div>
              </div>
            </div>

            <div className="pt-3 border-t border-slate-200 dark:border-slate-800 flex items-center justify-between text-xs">
              <span className="text-slate-500 dark:text-slate-400">
                Explore full statistical distribution
              </span>
              <Link
                to="/explorer"
                className="font-semibold text-blue-600 dark:text-blue-400 hover:underline flex items-center gap-1"
              >
                Data Explorer <ChevronRight className="h-3 w-3" />
              </Link>
            </div>
          </div>

          {/* Card 2: Dataset Health & Integrity */}
          <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 p-5 space-y-4 shadow-sm">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
                <ShieldCheck className="h-4 w-4 text-emerald-500" />
                Data Health & Integrity
              </h3>
              <span
                className={`text-xs font-bold px-2 py-0.5 rounded-md ${
                  Number(completenessPct) >= 98
                    ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-300"
                    : "bg-amber-100 text-amber-800 dark:bg-amber-950/50 dark:text-amber-300"
                }`}
              >
                {Number(completenessPct) >= 98
                  ? "High Quality"
                  : "Needs Review"}
              </span>
            </div>

            {/* Health Progress Bars */}
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                  <span>Data Completeness Rate</span>
                  <span className="font-mono font-bold text-slate-900 dark:text-white">
                    {completenessPct}%
                  </span>
                </div>
                <div className="h-2 w-full bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-emerald-500 rounded-full transition-all duration-500"
                    style={{ width: `${completenessPct}%` }}
                  ></div>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                  <span>Unique Row Integrity</span>
                  <span className="font-mono font-bold text-slate-900 dark:text-white">
                    {uniquenessPct}%
                  </span>
                </div>
                <div className="h-2 w-full bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-500 rounded-full transition-all duration-500"
                    style={{ width: `${uniquenessPct}%` }}
                  ></div>
                </div>
              </div>
            </div>

            {/* Health Metrics Strip */}
            <div className="grid grid-cols-3 gap-2 pt-1 text-center">
              <div className="p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-800">
                <span className="block text-[11px] text-slate-500 dark:text-slate-400 font-medium">
                  Total Cells
                </span>
                <span className="block font-mono font-bold text-xs text-slate-900 dark:text-white mt-0.5">
                  {totalCells.toLocaleString()}
                </span>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-800">
                <span className="block text-[11px] text-slate-500 dark:text-slate-400 font-medium">
                  Missing Cells
                </span>
                <span className="block font-mono font-bold text-xs text-slate-900 dark:text-white mt-0.5">
                  {missingTotal.toLocaleString()}
                </span>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-800">
                <span className="block text-[11px] text-slate-500 dark:text-slate-400 font-medium">
                  Duplicates
                </span>
                <span className="block font-mono font-bold text-xs text-slate-900 dark:text-white mt-0.5">
                  {duplicateTotal.toLocaleString()}
                </span>
              </div>
            </div>

            <div className="pt-2 flex items-center justify-between text-xs">
              <span className="text-slate-500 dark:text-slate-400">
                Transform columns or impute nulls
              </span>
              <Link
                to="/cleaning"
                className="font-semibold text-blue-600 dark:text-blue-400 hover:underline flex items-center gap-1"
              >
                Cleaning Studio <ChevronRight className="h-3 w-3" />
              </Link>
            </div>
          </div>
        </div>

        {/* Right Column (7 Cols): Enterprise Action Hub & Workspace Pulse */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          {/* Card 1: 6-Core Workflow Launchpad (2x3 Grid) */}
          <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 p-5 space-y-4 shadow-sm">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-emerald-500" />
                  Enterprise Analytics Hub
                </h3>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                  One-click access to core intelligence & transformation engines
                </p>
              </div>

              {pinnedCharts.length > 0 && (
                <Link
                  to="/custom-dashboard"
                  className="text-xs text-blue-600 dark:text-blue-400 hover:underline font-semibold flex items-center gap-1 px-2.5 py-1 rounded-lg bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-900/40"
                >
                  <Layers className="h-3 w-3" />
                  Pinned ({pinnedCharts.length})
                </Link>
              )}
            </div>

            {/* 6 Curated Analytical Tools in a 2x3 Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
              <Link
                to="/query"
                className="p-4 rounded-xl bg-slate-50/80 dark:bg-slate-800/40 hover:bg-slate-100 dark:hover:bg-slate-800/80 border border-slate-200 dark:border-slate-700/60 hover:border-blue-300 dark:hover:border-blue-500/50 hover:shadow-sm transition-all space-y-1.5 group block"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 font-bold text-xs sm:text-sm text-slate-900 dark:text-white">
                    <div className="p-1.5 rounded-lg bg-blue-500/10 text-blue-600 dark:text-blue-400">
                      <Sparkles className="h-4 w-4" />
                    </div>
                    AI Data Query
                  </div>
                  <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-blue-100 dark:bg-blue-950 text-blue-700 dark:text-blue-300">
                    Copilot
                  </span>
                </div>
                <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                  Ask questions in plain English & generate Plotly charts.
                </p>
              </Link>

              <Link
                to="/sql"
                className="p-4 rounded-xl bg-slate-50/80 dark:bg-slate-800/40 hover:bg-slate-100 dark:hover:bg-slate-800/80 border border-slate-200 dark:border-slate-700/60 hover:border-emerald-300 dark:hover:border-emerald-500/50 hover:shadow-sm transition-all space-y-1.5 group block"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 font-bold text-xs sm:text-sm text-slate-900 dark:text-white">
                    <div className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
                      <Terminal className="h-4 w-4" />
                    </div>
                    SQL Studio
                  </div>
                  <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-emerald-100 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300 font-mono">
                    SQLite
                  </span>
                </div>
                <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                  Query active dataset in-memory with sub-millisecond latency.
                </p>
              </Link>

              <Link
                to="/custom-dashboard"
                className="p-4 rounded-xl bg-slate-50/80 dark:bg-slate-800/40 hover:bg-slate-100 dark:hover:bg-slate-800/80 border border-slate-200 dark:border-slate-700/60 hover:border-purple-300 dark:hover:border-purple-500/50 hover:shadow-sm transition-all space-y-1.5 group block"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 font-bold text-xs sm:text-sm text-slate-900 dark:text-white">
                    <div className="p-1.5 rounded-lg bg-purple-500/10 text-purple-600 dark:text-purple-400">
                      <Layers className="h-4 w-4" />
                    </div>
                    Custom BI Canvas
                  </div>
                  <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-purple-100 dark:bg-purple-950 text-purple-700 dark:text-purple-300">
                    {pinnedCharts.length} Pinned
                  </span>
                </div>
                <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                  Assemble pinned charts into an interactive dashboard canvas.
                </p>
              </Link>

              <Link
                to="/cleaning"
                className="p-4 rounded-xl bg-slate-50/80 dark:bg-slate-800/40 hover:bg-slate-100 dark:hover:bg-slate-800/80 border border-slate-200 dark:border-slate-700/60 hover:border-amber-300 dark:hover:border-amber-500/50 hover:shadow-sm transition-all space-y-1.5 group block"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 font-bold text-xs sm:text-sm text-slate-900 dark:text-white">
                    <div className="p-1.5 rounded-lg bg-amber-500/10 text-amber-600 dark:text-amber-400">
                      <Wrench className="h-4 w-4" />
                    </div>
                    Cleaning Studio
                  </div>
                  <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-amber-100 dark:bg-amber-950 text-amber-700 dark:text-amber-300 font-mono">
                    v{activeDataset?.version}
                  </span>
                </div>
                <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                  Transform columns, cast types, and rollback version snapshots.
                </p>
              </Link>

              <Link
                to="/forecasting"
                className="p-4 rounded-xl bg-slate-50/80 dark:bg-slate-800/40 hover:bg-slate-100 dark:hover:bg-slate-800/80 border border-slate-200 dark:border-slate-700/60 hover:border-cyan-300 dark:hover:border-cyan-500/50 hover:shadow-sm transition-all space-y-1.5 group block"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 font-bold text-xs sm:text-sm text-slate-900 dark:text-white">
                    <div className="p-1.5 rounded-lg bg-cyan-500/10 text-cyan-600 dark:text-cyan-400">
                      <TrendingUp className="h-4 w-4" />
                    </div>
                    Forecasting Studio
                  </div>
                  <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-cyan-100 dark:bg-cyan-950 text-cyan-700 dark:text-cyan-300">
                    ML
                  </span>
                </div>
                <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                  Predict future metrics using Holt-Winters and Linear OLS
                  models.
                </p>
              </Link>

              <Link
                to="/anomalies"
                className="p-4 rounded-xl bg-slate-50/80 dark:bg-slate-800/40 hover:bg-slate-100 dark:hover:bg-slate-800/80 border border-slate-200 dark:border-slate-700/60 hover:border-rose-300 dark:hover:border-rose-500/50 hover:shadow-sm transition-all space-y-1.5 group block"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 font-bold text-xs sm:text-sm text-slate-900 dark:text-white">
                    <div className="p-1.5 rounded-lg bg-rose-500/10 text-rose-600 dark:text-rose-400">
                      <AlertCircle className="h-4 w-4" />
                    </div>
                    Anomaly Detection
                  </div>
                  <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-rose-100 dark:bg-rose-950 text-rose-700 dark:text-rose-300">
                    Outliers
                  </span>
                </div>
                <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                  Detect multidimensional anomalies via Isolation Forest &
                  Z-scores.
                </p>
              </Link>
            </div>
          </div>

          {/* Card 2: Workspace Pulse & Activity Stream */}
          <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 p-5 space-y-4 shadow-sm flex-1 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between">
                <p className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 flex items-center gap-1.5">
                  <Activity className="h-3.5 w-3.5 text-blue-500" />
                  Workspace Pulse & Insights
                </p>
                <span className="text-xs text-slate-400 font-medium">
                  Real-time sync
                </span>
              </div>

              {/* 3 Status Badges */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs mt-3">
                <div className="rounded-xl bg-blue-50/70 dark:bg-blue-950/20 border border-blue-100 dark:border-blue-900/30 px-3.5 py-3">
                  <span className="block text-slate-500 dark:text-slate-400 font-medium text-[11px]">
                    Volume Profile
                  </span>
                  <strong className="block text-slate-900 dark:text-white mt-1 font-mono">
                    {activeDataset?.rows.toLocaleString()} records
                  </strong>
                </div>

                <div className="rounded-xl bg-emerald-50/70 dark:bg-emerald-950/20 border border-emerald-100 dark:border-emerald-900/30 px-3.5 py-3">
                  <span className="block text-slate-500 dark:text-slate-400 font-medium text-[11px]">
                    Schema Balance
                  </span>
                  <strong className="block text-slate-900 dark:text-white mt-1">
                    {numCount} numeric / {catCount} cat
                  </strong>
                </div>

                <div className="rounded-xl bg-amber-50/70 dark:bg-amber-950/20 border border-amber-100 dark:border-amber-900/30 px-3.5 py-3">
                  <span className="block text-slate-500 dark:text-slate-400 font-medium text-[11px]">
                    Data Integrity
                  </span>
                  <strong className="block text-slate-900 dark:text-white mt-1">
                    {missingTotal === 0
                      ? "100% Clean (0 nulls)"
                      : `${missingTotal.toLocaleString()} nulls`}
                  </strong>
                </div>
              </div>
            </div>

            {/* Activity Stream or Suggested AI Prompts */}
            <div className="pt-3 border-t border-slate-200 dark:border-slate-800">
              {history.length > 0 ? (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-slate-500 dark:text-slate-400 font-bold uppercase tracking-wider">
                      Latest AI Inquiry
                    </span>
                    <Link
                      to="/query"
                      className="text-xs text-blue-600 dark:text-blue-400 hover:underline font-semibold flex items-center gap-1"
                    >
                      View All Queries ({history.length})
                      <ArrowRight className="h-3 w-3" />
                    </Link>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700/60 flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <p className="text-xs sm:text-sm text-blue-600 dark:text-blue-300 font-bold truncate">
                        "{history[0].question}"
                      </p>
                      <p className="text-xs text-slate-500 dark:text-slate-400 line-clamp-1 mt-0.5">
                        {history[0].result_summary}
                      </p>
                    </div>
                    <Link
                      to="/query"
                      className="px-2.5 py-1 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-[11px] font-bold flex-shrink-0"
                    >
                      Ask
                    </Link>
                  </div>
                </div>
              ) : (
                <div className="space-y-2">
                  <span className="text-xs text-slate-500 dark:text-slate-400 font-bold uppercase tracking-wider flex items-center gap-1.5">
                    <Lightbulb className="h-3.5 w-3.5 text-amber-500" />
                    Suggested AI Exploration Prompts
                  </span>
                  <div className="flex flex-wrap gap-2 pt-1">
                    <Link
                      to="/query"
                      className="text-xs px-3 py-1.5 rounded-xl bg-slate-50 hover:bg-blue-50 dark:bg-slate-800/50 dark:hover:bg-blue-950/40 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-300 transition-colors"
                    >
                      "Summarize key trends & distributions"
                    </Link>
                    <Link
                      to="/query"
                      className="text-xs px-3 py-1.5 rounded-xl bg-slate-50 hover:bg-blue-50 dark:bg-slate-800/50 dark:hover:bg-blue-950/40 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-300 transition-colors"
                    >
                      "Plot correlation heatmap of numeric columns"
                    </Link>
                    <Link
                      to="/query"
                      className="text-xs px-3 py-1.5 rounded-xl bg-slate-50 hover:bg-blue-50 dark:bg-slate-800/50 dark:hover:bg-blue-950/40 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-300 transition-colors"
                    >
                      "Detect potential outliers or anomalies"
                    </Link>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Dataset Preview Table Section */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 p-5 space-y-4 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-bold text-slate-900 dark:text-white">
                Dataset Preview (Active Snapshot)
              </h3>
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300">
                First 50 records
              </span>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Rendered directly from the in-memory engine • Total:{" "}
              {activeDataset?.rows.toLocaleString()} rows ×{" "}
              {activeDataset?.cols.toLocaleString()} columns
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {/* Realtime Search Input */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-slate-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Filter preview rows..."
                className="pl-8 pr-3 py-1.5 text-xs rounded-xl bg-slate-50 dark:bg-slate-950/80 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500/30 w-48 sm:w-56"
              />
            </div>

            <Link
              to="/explorer"
              className="text-xs font-bold text-blue-600 dark:text-blue-400 hover:underline flex items-center gap-1"
            >
              Advanced Explorer
              <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </div>

        {previewLoading ? (
          <div className="h-40 flex items-center justify-center text-slate-500 text-xs">
            <RefreshCw className="h-4 w-4 animate-spin mr-2 text-blue-500" />
            Loading preview...
          </div>
        ) : (
          <div className="overflow-x-auto">
            <DataTable
              data={filteredPreviewData}
              columns={preview?.columns}
              pageSize={10}
            />
          </div>
        )}
      </div>
    </div>
  );
};
