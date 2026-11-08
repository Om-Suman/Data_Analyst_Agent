import React, { useState } from "react";
import { Link } from "react-router-dom";
import { Database, Zap, ArrowRight, Download } from "lucide-react";
import { useDataset } from "../context/DatasetContext";
import { DataTable } from "../components/DataTable";
import { PlotlyChart } from "../components/PlotlyChart";
import { datasetApi } from "../api/client";
import { useToast } from "../components/Toast";

export const DashboardPage: React.FC = () => {
  const {
    activeDataset,
    preview,
    loading,
    previewLoading,
    hasDataset,
    refreshDatasets,
    refreshPreview,
  } = useDataset();
  const [loadingSample, setLoadingSample] = useState<string | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const { success, error } = useToast();
  const handleLoadDemo = async (name: string) => {
    if (loadingSample) return;
    setLoadingSample(name);
    try {
      await datasetApi.loadSample(name);
      await refreshDatasets();
      success("Demo loaded", name);
    } catch {
      error("Could not load sample", "Try again or upload a file.");
    } finally {
      setLoadingSample(null);
    }
  };
  if (loading)
    return (
      <div
        role="status"
        className="py-20 text-center text-slate-600 dark:text-slate-300"
      >
        Loading datasets…
      </div>
    );
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

  const currentPreview = preview?.name === activeDataset?.name ? preview : null;
  const meta = currentPreview?.metadata;
  const missing =
    Number.isFinite(meta?.missing_total) && meta.missing_total >= 0
      ? meta.missing_total
      : null;
  const duplicates =
    Number.isFinite(meta?.duplicate_rows) && meta.duplicate_rows >= 0
      ? meta.duplicate_rows
      : null;
  const cells = (activeDataset?.rows || 0) * (activeDataset?.cols || 0);
  const populated =
    missing !== null && cells > 0
      ? Math.max(0, 100 - (missing / cells) * 100).toFixed(1)
      : null;
  const needsReview = (missing ?? 0) > 0 || (duplicates ?? 0) > 0;
  const isText = activeDataset?.is_text;
  const promptLink = (prompt: string) => ({
    pathname: "/query",
    search: `?${new URLSearchParams({ prompt, dataset: activeDataset?.name || "" })}`,
  });
  const prompts = [
    "Summarize key trends and distributions",
    "Plot a correlation heatmap of numeric columns",
    "Identify potential outliers in this dataset",
  ];
  const columnTypes = [
    {
      label: "Numeric",
      count: currentPreview?.numeric_cols?.length || 0,
      color: "#3b82f6",
    },
    {
      label: "Categorical",
      count: currentPreview?.categorical_cols?.length || 0,
      color: "#10b981",
    },
    {
      label: "Date / time",
      count: currentPreview?.date_cols?.length || 0,
      color: "#f59e0b",
    },
  ];
  columnTypes.push({
    label: "Other",
    count: Math.max(
      0,
      (activeDataset?.cols || 0) -
        columnTypes.reduce((sum, type) => sum + type.count, 0),
    ),
    color: "#8b5cf6",
  });
  const visibleTypes = columnTypes.filter((type) => type.count > 0);
  const schemaSpec = {
    data: [
      {
        type: "pie",
        hole: 0.6,
        labels: visibleTypes.map((type) => type.label),
        values: visibleTypes.map((type) => type.count),
        marker: { colors: visibleTypes.map((type) => type.color) },
        textinfo: "percent",
        hoverinfo: "label+value+percent",
      },
    ],
    layout: {
      height: 240,
      margin: { l: 10, r: 10, t: 10, b: 10 },
      showlegend: true,
    },
  };

  return (
    <div className="space-y-8 min-w-0">
      <header className="space-y-2">
        <h1 className="text-2xl sm:text-3xl font-bold break-words">
          {activeDataset?.name}
        </h1>
        <p className="text-sm text-slate-600 dark:text-slate-300">
          Version {activeDataset?.version} ·{" "}
          {isText
            ? "Document"
            : `${activeDataset?.rows.toLocaleString()} rows · ${activeDataset?.cols.toLocaleString()} columns`}
        </p>
      </header>

      <section
        aria-labelledby="readiness-title"
        className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 sm:p-6 space-y-5"
      >
        <div className="space-y-2">
          <h2 id="readiness-title" className="text-xl font-semibold">
            {isText
              ? "Explore your document"
              : needsReview
                ? "Review your data before exploring"
                : "Choose a question to investigate"}
          </h2>
          <p className="text-sm text-slate-600 dark:text-slate-300 max-w-prose">
            {isText
              ? "Ask questions about this document and inspect the supporting sources."
              : "Start with missing values and duplicate rows. These checks do not establish accuracy or suitability for analysis."}
          </p>
        </div>
        {!isText && (
          <dl className="grid gap-5 sm:grid-cols-2 border-y border-slate-200 dark:border-slate-800 py-5">
            <div>
              <dt className="text-sm text-slate-600 dark:text-slate-300">
                Cells populated
              </dt>
              <dd className="text-xl font-semibold tabular-nums mt-1">
                {previewLoading
                  ? "Checking…"
                  : populated === null
                    ? "Not available"
                    : `${populated}%`}
              </dd>
              <dd className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                {previewLoading
                  ? "Loading missing-value counts."
                  : missing === null
                    ? "Load the preview to check missing values."
                    : missing === 0
                      ? "No missing values"
                      : `${missing.toLocaleString()} missing cells`}
              </dd>
            </div>
            <div>
              <dt className="text-sm text-slate-600 dark:text-slate-300">
                Duplicate rows
              </dt>
              <dd className="text-xl font-semibold tabular-nums mt-1">
                {previewLoading
                  ? "Checking…"
                  : duplicates === null
                    ? "Not available"
                    : duplicates.toLocaleString()}
              </dd>
              <dd className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                Exact row matches; repeated rows may be intentional.
              </dd>
            </div>
          </dl>
        )}
        <Link
          to={isText ? "/document" : needsReview ? "/cleaning" : "/query"}
          className="inline-flex min-h-11 items-center gap-2 px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold text-sm"
        >
          {isText
            ? "Ask about this document"
            : needsReview
              ? "Review missing values and duplicates"
              : "Ask a question"}
          <ArrowRight className="h-4 w-4" aria-hidden="true" />
        </Link>
      </section>

      {!isText && (
        <section aria-labelledby="preview-title" className="space-y-4 min-w-0">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 id="preview-title" className="text-xl font-semibold">
                Data preview
              </h2>
              <p className="text-sm text-slate-600 dark:text-slate-300">
                {currentPreview
                  ? `First ${currentPreview.data.length} rows. Filtering and table export apply to this preview only.`
                  : "Preview unavailable. Retry below to load rows."}
              </p>
            </div>
            <Link
              to="/explorer"
              className="text-sm text-blue-700 dark:text-blue-300 underline underline-offset-4"
            >
              Explore all rows
            </Link>
          </div>
          {previewLoading ? (
            <p role="status" className="py-8">
              Loading preview…
            </p>
          ) : !currentPreview ? (
            <div
              role="alert"
              className="rounded-xl border border-slate-300 dark:border-slate-700 p-5 space-y-3"
            >
              <p>
                We could not load this dataset’s preview. Try again to inspect
                rows and checks.
              </p>
              <button
                onClick={() => refreshPreview()}
                className="min-h-11 px-4 rounded-lg bg-blue-600 text-white"
              >
                Retry preview
              </button>
            </div>
          ) : (
            <DataTable
              key={currentPreview.name + currentPreview.version}
              data={currentPreview.data}
              columns={currentPreview.columns}
              pageSize={10}
            />
          )}
        </section>
      )}

      {!isText && (
        <section aria-labelledby="questions-title" className="space-y-3">
          <h2 id="questions-title" className="text-lg font-semibold">
            Start with a question
          </h2>
          <p className="text-sm text-slate-600 dark:text-slate-300">
            Choose a prompt to edit before running it.
          </p>
          <ul className="divide-y divide-slate-200 dark:divide-slate-800">
            {prompts.map((prompt) => (
              <li key={prompt}>
                <Link
                  to={promptLink(prompt)}
                  className="flex items-center justify-between gap-4 py-3 min-h-11 text-sm text-blue-700 dark:text-blue-300 hover:underline"
                >
                  {prompt}
                  <ArrowRight className="h-4 w-4 shrink-0" aria-hidden="true" />
                </Link>
              </li>
            ))}
          </ul>
        </section>
      )}

      <details
        open={detailsOpen}
        onToggle={(event) => setDetailsOpen(event.currentTarget.open)}
        className="border-t border-slate-200 dark:border-slate-800 pt-4"
      >
        <summary className="cursor-pointer min-h-11 font-semibold text-sm">
          Dataset details and full exports
        </summary>
        <div className="space-y-4 py-3 text-sm">
          <p className="break-words">
            Source: {activeDataset?.source}
            {meta?.memory_mb ? ` · Memory: ${meta.memory_mb} MB` : ""}
          </p>
          {!isText && (
            <>
              <p>
                {currentPreview
                  ? `${currentPreview.numeric_cols?.length ?? 0} numeric · ${currentPreview.categorical_cols?.length ?? 0} categorical · ${currentPreview.date_cols?.length ?? 0} date columns`
                  : "Column types unavailable until the preview loads."}
              </p>
              <div className="flex flex-wrap gap-4">
                <a
                  className="text-blue-700 dark:text-blue-300 underline min-h-11 inline-flex items-center gap-2"
                  href={datasetApi.getDownloadCsvUrl()}
                  download
                >
                  <Download className="h-4 w-4" />
                  Full dataset CSV
                </a>
                <a
                  className="text-blue-700 dark:text-blue-300 underline min-h-11 inline-flex items-center gap-2"
                  href={datasetApi.getDownloadExcelUrl()}
                  download
                >
                  <Download className="h-4 w-4" />
                  Full Excel workbook
                </a>
              </div>
              {detailsOpen && meta && visibleTypes.length > 0 && (
                <div className="max-w-lg">
                  <h3 className="font-semibold">Column types</h3>
                  <PlotlyChart
                    spec={schemaSpec}
                    title="Schema breakdown"
                    chartType="Pie"
                    sourcePage="Overview"
                    height={240}
                  />
                </div>
              )}
            </>
          )}
        </div>
      </details>
    </div>
  );
};
