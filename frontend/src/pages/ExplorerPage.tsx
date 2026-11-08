import { DatasetEmptyState } from "../components/DatasetEmptyState";
import React, { useState, useEffect } from "react";
import {
  Compass,
  Table,
  GitCommit,
  BarChart,
  Layers,
  Search,
  Filter,
} from "lucide-react";
import { useDataset } from "../context/DatasetContext";
import { explorerApi } from "../api/client";
import {
  ColumnProfileResponse,
  CorrelationsResponse,
  ExplorerBrowseResponse,
} from "../types";
import { PlotlyChart } from "../components/PlotlyChart";
import { DataTable } from "../components/DataTable";

export const ExplorerPage: React.FC = () => {
  const { preview, hasDataset: hasActiveDataset, activeDataset: selectedDataset } = useDataset();
  const hasDataset = hasActiveDataset && !selectedDataset?.is_text;
  const [activeTab, setActiveTab] = useState<
    "browse" | "correlations" | "distributions" | "profiler"
  >("browse");

  // Browse state
  const [browseData, setBrowseData] = useState<ExplorerBrowseResponse | null>(
    null,
  );
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
  const [loadingBrowse, setLoadingBrowse] = useState(false);
  const [browseError, setBrowseError] = useState(false);
  const [browseRetry, setBrowseRetry] = useState(0);

  // Correlations state
  const [corrMethod, setCorrMethod] = useState<
    "pearson" | "spearman" | "kendall"
  >("pearson");
  const [corrData, setCorrData] = useState<CorrelationsResponse | null>(null);
  const [loadingCorr, setLoadingCorr] = useState(false);

  // Distributions state
  const [distCol, setDistCol] = useState<string>("");
  const [distChartType, setDistChartType] = useState<string>("Histogram");
  const [distGroupBy, setDistGroupBy] = useState<string>("None");
  const [distData, setDistData] = useState<any>(null);
  const [loadingDist, setLoadingDist] = useState(false);

  // Profiler state
  const [profileCol, setProfileCol] = useState<string>("");
  const [profileData, setProfileData] = useState<ColumnProfileResponse | null>(
    null,
  );
  const [loadingProfile, setLoadingProfile] = useState(false);

  useEffect(() => {
    if (preview?.columns && preview.columns.length > 0) {
      // Reset column selections when switching datasets — the previously
      // selected column may not exist in the new dataset, which would
      // cause a 400 error from the API.
      if (!distCol || !preview.columns.includes(distCol))
        setDistCol(preview.columns[0]);
      if (!profileCol || !preview.columns.includes(profileCol))
        setProfileCol(preview.columns[0]);
    }
  }, [preview]);

  // Fetch Browse
  useEffect(() => {
    if (!hasDataset) return;
    let current = true;
    setLoadingBrowse(true);
    setBrowseError(false);
    explorerApi
      .browse({
        page,
        page_size: pageSize,
        columns: selectedColumns.length > 0 ? selectedColumns : undefined,
      })
      .then((res) => { if (current) setBrowseData(res); })
      .catch(() => { if (current) setBrowseError(true); })
      .finally(() => { if (current) setLoadingBrowse(false); });
    return () => { current = false; };
  }, [hasDataset, page, pageSize, selectedColumns, browseRetry]);

  // Fetch Correlations
  useEffect(() => {
    if (
      activeTab !== "correlations" ||
      !preview?.numeric_cols ||
      preview.numeric_cols.length < 2
    )
      return;
    setLoadingCorr(true);
    explorerApi
      .getCorrelations(preview.numeric_cols, corrMethod)
      .then((res) => setCorrData(res))
      .catch((err) => console.error(err))
      .finally(() => setLoadingCorr(false));
  }, [activeTab, corrMethod, preview]);

  // Fetch Distribution
  useEffect(() => {
    if (activeTab !== "distributions" || !distCol) return;
    setLoadingDist(true);
    explorerApi
      .getDistribution({
        column: distCol,
        chart_type: distChartType,
        group_by: distGroupBy !== "None" ? distGroupBy : undefined,
      })
      .then((res) => setDistData(res))
      .catch((err) => console.error(err))
      .finally(() => setLoadingDist(false));
  }, [activeTab, distCol, distChartType, distGroupBy]);

  // Fetch Profiler
  useEffect(() => {
    if (activeTab !== "profiler" || !profileCol) return;
    setLoadingProfile(true);
    explorerApi
      .getColumnProfile(profileCol)
      .then((res) => setProfileData(res))
      .catch((err) => console.error(err))
      .finally(() => setLoadingProfile(false));
  }, [activeTab, profileCol]);

  if (!hasDataset) {
    return (
      <DatasetEmptyState
        title="Data Explorer"
        description="Browse rows, inspect columns, and see how values are distributed."
      />
    );
  }

  const numericCols = preview?.numeric_cols || [];
  const catCols = preview?.categorical_cols || [];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white tracking-tight">
          Data Explorer
        </h2>
        <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-1">
          Explore distributions, correlations, column profiles, and interactive
          data grids
        </p>
      </div>

      {/* Tabs */}
      <div className="flex flex-wrap items-center gap-2 border-b border-slate-200 dark:border-slate-800 pb-2">
        <button
          onClick={() => setActiveTab("browse")}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
            activeTab === "browse"
              ? "bg-blue-50 dark:bg-blue-600/20 text-blue-600 dark:text-blue-400 border border-blue-200 dark:border-blue-500/30 shadow-sm"
              : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800/40"
          }`}
        >
          <Table className="h-4 w-4" />
          Browse Table
        </button>

        <button
          onClick={() => setActiveTab("correlations")}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
            activeTab === "correlations"
              ? "bg-blue-50 dark:bg-blue-600/20 text-blue-600 dark:text-blue-400 border border-blue-200 dark:border-blue-500/30 shadow-sm"
              : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800/40"
          }`}
        >
          <GitCommit className="h-4 w-4" />
          Correlations ({numericCols.length})
        </button>

        <button
          onClick={() => setActiveTab("distributions")}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
            activeTab === "distributions"
              ? "bg-blue-50 dark:bg-blue-600/20 text-blue-600 dark:text-blue-400 border border-blue-200 dark:border-blue-500/30 shadow-sm"
              : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800/40"
          }`}
        >
          <BarChart className="h-4 w-4" />
          Distributions
        </button>

        <button
          onClick={() => setActiveTab("profiler")}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
            activeTab === "profiler"
              ? "bg-blue-50 dark:bg-blue-600/20 text-blue-600 dark:text-blue-400 border border-blue-200 dark:border-blue-500/30 shadow-sm"
              : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800/40"
          }`}
        >
          <Layers className="h-4 w-4" />
          Column Profiler
        </button>
      </div>

      {/* Tab 1: Browse Table */}
      {activeTab === "browse" && (
        <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 p-5 sm:p-6 space-y-4 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <h3 className="text-sm font-bold text-slate-900 dark:text-slate-200">
                Interactive Dataset Browser
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                {browseData && `Rows ${browseData.total_rows ? (browseData.page - 1) * browseData.page_size + 1 : 0}–${Math.min(browseData.page * browseData.page_size, browseData.total_rows)} of ${browseData.total_rows.toLocaleString()}. Search, sort, and CSV apply to this page.`}
              </p>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-xs text-slate-500 dark:text-slate-400">
                Page size:
              </span>
              <select
                aria-label="Rows fetched per page"
                value={pageSize}
                onChange={(e) => { setPageSize(Number(e.target.value)); setPage(1); }}
                className="bg-slate-50 dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-xl px-2.5 py-1 text-xs text-slate-900 dark:text-slate-200 shadow-sm"
              >
                <option value={10}>10</option>
                <option value={25}>25</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
              </select>
            </div>
          </div>

          {loadingBrowse ? (
            <div className="h-40 flex items-center justify-center text-xs text-slate-500 dark:text-slate-400">
              Loading dataset...
            </div>
          ) : browseError ? (
            <div role="alert" className="space-y-2"><p>Could not load this page of data.</p><button type="button" onClick={() => setBrowseRetry(value => value + 1)} className="underline">Retry</button></div>
          ) : (
            <DataTable
              key={`${page}-${pageSize}`}
              showPagination={false}
              data={browseData?.data || []}
              columns={browseData?.columns}
              pageSize={pageSize}
              showSearch={true}
            />
          )}
          {browseData && (
            <nav aria-label="Dataset pages" className="flex flex-wrap items-center justify-between gap-2 text-sm">
              <button type="button" disabled={loadingBrowse || page <= 1} onClick={() => setPage(page - 1)} className="min-h-11 rounded-lg border px-3 disabled:opacity-40">Previous dataset page</button>
              <span>Page {page} of {Math.max(1, Math.ceil(browseData.total_rows / pageSize))}</span>
              <button type="button" disabled={loadingBrowse || page * pageSize >= browseData.total_rows} onClick={() => setPage(page + 1)} className="min-h-11 rounded-lg border px-3 disabled:opacity-40">Next dataset page</button>
            </nav>
          )}
        </div>
      )}

      {/* Tab 2: Correlations Matrix */}
      {activeTab === "correlations" && (
        <div className="space-y-6">
          <div className="flex items-center justify-between gap-4 p-4 sm:p-5 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 shadow-sm">
            <div>
              <h3 className="text-sm font-bold text-slate-900 dark:text-slate-200">
                Pairwise Feature Correlation
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Calculates linear or rank relationships between numeric fields
              </p>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 dark:text-slate-400">
                Method:
              </span>
              <select
                value={corrMethod}
                onChange={(e) => setCorrMethod(e.target.value as any)}
                className="bg-slate-50 dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-1.5 text-xs text-slate-900 dark:text-slate-200 font-medium shadow-sm"
              >
                <option value="pearson">Pearson (Standard)</option>
                <option value="spearman">Spearman (Rank)</option>
                <option value="kendall">Kendall (Tau)</option>
              </select>
            </div>
          </div>

          {numericCols.length < 2 ? (
            <div className="p-8 text-center text-slate-500 dark:text-slate-400 border border-slate-200 dark:border-slate-800 rounded-2xl bg-white dark:bg-slate-900/30 shadow-sm">
              At least 2 numeric columns are required to generate correlation
              heatmaps.
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 p-5 shadow-sm">
                <PlotlyChart spec={corrData?.figure_spec} height={450} />
              </div>

              <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 p-5 space-y-3 shadow-sm">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-400">
                  Strongest Correlations
                </h4>
                <div className="space-y-2 overflow-y-auto max-h-[400px]">
                  {corrData?.top_pairs.map((p, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between text-xs p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700/60"
                    >
                      <div className="truncate pr-2">
                        <span className="font-mono font-medium text-slate-800 dark:text-slate-200">
                          {p.col_a}
                        </span>
                        <span className="text-slate-400 dark:text-slate-500 mx-1">
                          ↔
                        </span>
                        <span className="font-mono font-medium text-slate-800 dark:text-slate-200">
                          {p.col_b}
                        </span>
                      </div>
                      <span
                        className={`font-mono font-bold ${
                          p.correlation > 0.6
                            ? "text-emerald-600 dark:text-emerald-400"
                            : p.correlation < -0.6
                              ? "text-rose-600 dark:text-rose-400"
                              : "text-blue-600 dark:text-blue-400"
                        }`}
                      >
                        {p.correlation.toFixed(3)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tab 3: Distributions */}
      {activeTab === "distributions" && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 p-4 sm:p-5 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 text-xs shadow-sm">
            <div>
              <label className="block text-slate-700 dark:text-slate-300 font-medium mb-1">
                Target Column
              </label>
              <select
                value={distCol}
                onChange={(e) => setDistCol(e.target.value)}
                className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 text-slate-900 dark:text-slate-200 shadow-sm"
              >
                {preview?.columns.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-slate-700 dark:text-slate-300 font-medium mb-1">
                Chart Type
              </label>
              <select
                value={distChartType}
                onChange={(e) => setDistChartType(e.target.value)}
                className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 text-slate-900 dark:text-slate-200 shadow-sm"
              >
                <option value="Histogram">Histogram</option>
                <option value="Box Plot">Box Plot</option>
                <option value="Violin">Violin Plot</option>
              </select>
            </div>

            <div>
              <label className="block text-slate-700 dark:text-slate-300 font-medium mb-1">
                Group By Category (Optional)
              </label>
              <select
                value={distGroupBy}
                onChange={(e) => setDistGroupBy(e.target.value)}
                className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 text-slate-900 dark:text-slate-200 shadow-sm"
              >
                <option value="None">None</option>
                {catCols.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 p-5 shadow-sm">
            {loadingDist ? (
              <div className="h-72 flex items-center justify-center text-xs text-slate-500 dark:text-slate-400">
                Generating distribution...
              </div>
            ) : (
              <PlotlyChart spec={distData?.figure_spec} height={420} />
            )}
          </div>
        </div>
      )}

      {/* Tab 4: Column Profiler */}
      {activeTab === "profiler" && (
        <div className="space-y-6 min-h-[600px]">
          <div className="flex items-center gap-3 p-4 sm:p-5 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 shadow-sm">
            <span className="text-xs text-slate-700 dark:text-slate-300 font-medium">
              Select Column to Profile:
            </span>
            <select
              value={profileCol}
              onChange={(e) => setProfileCol(e.target.value)}
              className="bg-slate-50 dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-1.5 text-xs text-slate-900 dark:text-slate-200 max-w-xs font-mono shadow-sm"
            >
              {preview?.columns.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>

          {loadingProfile ? (
            <div className="min-h-[500px] flex items-center justify-center text-xs text-slate-500 dark:text-slate-400">
              <div className="flex flex-col items-center gap-3">
                <div className="h-8 w-8 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
                Calculating profile statistics...
              </div>
            </div>
          ) : profileData ? (
            <div className="space-y-6 overflow-hidden">
              {/* Summary Stats Cards */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
                <div className="p-4 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 shadow-sm min-h-[88px]">
                  <span className="text-slate-500 dark:text-slate-400">
                    Data Type
                  </span>
                  <p className="text-lg font-bold text-blue-600 dark:text-blue-400 font-mono mt-1 truncate">
                    {profileData.dtype}
                  </p>
                </div>
                <div className="p-4 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 shadow-sm min-h-[88px]">
                  <span className="text-slate-500 dark:text-slate-400">
                    Unique Values
                  </span>
                  <p className="text-lg font-bold text-slate-900 dark:text-white font-mono mt-1 truncate">
                    {profileData.unique_count.toLocaleString()}
                  </p>
                </div>
                <div className="p-4 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 shadow-sm min-h-[88px]">
                  <span className="text-slate-500 dark:text-slate-400">
                    Missing Values
                  </span>
                  <p className="text-lg font-bold text-amber-600 dark:text-amber-400 font-mono mt-1 truncate">
                    {profileData.missing_count} ({profileData.missing_pct}%)
                  </p>
                </div>
                <div className="p-4 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 shadow-sm min-h-[88px]">
                  <span className="text-slate-500 dark:text-slate-400">
                    Category Type
                  </span>
                  <p className="text-lg font-bold text-emerald-600 dark:text-emerald-400 font-mono mt-1 truncate">
                    {profileData.is_numeric
                      ? "Numeric Metric"
                      : "Categorical Dimension"}
                  </p>
                </div>
              </div>

              {/* Numeric Deep Stats — always reserve space to prevent layout shift */}
              <div
                className={`grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs transition-all duration-200 ${profileData.numeric_stats ? "opacity-100" : "opacity-0 h-0 overflow-hidden !gap-0 !m-0"}`}
              >
                <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700 min-h-[68px]">
                  <span className="text-slate-500 dark:text-slate-400">
                    Mean ± Std
                  </span>
                  <p className="text-sm font-bold text-slate-900 dark:text-white font-mono mt-1 truncate">
                    {profileData.numeric_stats?.mean ?? "—"} ±{" "}
                    {profileData.numeric_stats?.std ?? "—"}
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700 min-h-[68px]">
                  <span className="text-slate-500 dark:text-slate-400">
                    Min / Max Range
                  </span>
                  <p className="text-sm font-bold text-slate-900 dark:text-white font-mono mt-1 truncate">
                    [{profileData.numeric_stats?.min ?? "—"},{" "}
                    {profileData.numeric_stats?.max ?? "—"}]
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700 min-h-[68px]">
                  <span className="text-slate-500 dark:text-slate-400">
                    Median
                  </span>
                  <p className="text-sm font-bold text-slate-900 dark:text-white font-mono mt-1 truncate">
                    {profileData.numeric_stats?.median ?? "—"}
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700 min-h-[68px]">
                  <span className="text-slate-500 dark:text-slate-400">
                    Skewness
                  </span>
                  <p className="text-sm font-bold text-slate-900 dark:text-white font-mono mt-1 truncate">
                    {profileData.numeric_stats?.skew ?? "—"}
                  </p>
                </div>
              </div>

              {/* Profiler Chart */}
              <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 p-5 shadow-sm overflow-hidden">
                <PlotlyChart spec={profileData.figure_spec} height={320} />
              </div>
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
};
