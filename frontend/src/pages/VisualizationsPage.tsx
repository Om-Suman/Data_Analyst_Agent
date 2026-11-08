import { DatasetEmptyState } from "../components/DatasetEmptyState";
import React, { useState, useEffect } from "react";
import {
  BarChart3,
  LineChart,
  PieChart,
  ScatterChart,
  AreaChart,
  Activity,
  Layers,
  Sparkles,
  Sliders,
  Play,
  AlertCircle,
} from "lucide-react";
import { useDataset } from "../context/DatasetContext";
import { visualizationApi } from "../api/client";
import { VisualizationResponse } from "../types";
import { PlotlyChart } from "../components/PlotlyChart";
import { useToast } from "../components/Toast";

export const VisualizationsPage: React.FC = () => {
  const { preview, hasDataset: hasActiveDataset, activeDataset: selectedDataset } = useDataset();
  const hasDataset = hasActiveDataset && !selectedDataset?.is_text;
  const { success, error: toastError } = useToast();

  const [chartType, setChartType] = useState("Bar Chart");
  const [xAxis, setXAxis] = useState<string>("");
  const [yAxis, setYAxis] = useState<string>("");
  const [colorCol, setColorCol] = useState("None");
  const [sizeCol, setSizeCol] = useState("None");
  const [topN, setTopN] = useState(20);
  const [nbins, setNbins] = useState(30);
  const [hole, setHole] = useState(0.0);
  const [template, setTemplate] = useState("plotly_dark");
  const [colorscale, setColorscale] = useState("Blues");
  const [trendline, setTrendline] = useState(false);
  const [title, setTitle] = useState("");

  const [loading, setLoading] = useState(false);
  const [renderedSettings, setRenderedSettings] = useState<string | null>(null);
  const settingsKey = JSON.stringify([chartType, xAxis, yAxis, colorCol, sizeCol, topN, nbins, hole, template, colorscale, trendline, title]);
  const presets = [
    { label: "Compare categories", chart: "Bar Chart" },
    { label: "Track a trend", chart: "Line Chart" },
    { label: "Compare two measures", chart: "Scatter Plot" },
    { label: "See a distribution", chart: "Histogram" },
  ];
  const [chartResponse, setChartResponse] =
    useState<VisualizationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const chartTypes = [
    { name: "Bar Chart", icon: BarChart3 },
    { name: "Line Chart", icon: LineChart },
    { name: "Scatter Plot", icon: ScatterChart },
    { name: "Histogram", icon: BarChart3 },
    { name: "Box Plot", icon: Activity },
    { name: "Violin Plot", icon: Activity },
    { name: "Pie Chart", icon: PieChart },
    { name: "Area Chart", icon: AreaChart },
    { name: "Heatmap", icon: Layers },
    { name: "Treemap", icon: Layers },
    { name: "Sunburst", icon: PieChart },
    { name: "Bubble Chart", icon: ScatterChart },
    { name: "Funnel Chart", icon: BarChart3 },
    { name: "KPI Dashboard", icon: Activity },
  ];

  const columns = preview?.columns || [];
  const numericCols = preview?.numeric_cols || [];
  const catCols = preview?.categorical_cols || [];

  const selectPreset = (chart: string) => {
    setChartType(chart);
    setColorCol("None");
    setSizeCol("None");
    setTitle("");
    if (chart === "Line Chart") {
      setXAxis(preview?.date_cols?.[0] || columns[0] || "");
      setYAxis(numericCols[0] || "");
    } else if (chart === "Scatter Plot" || chart === "Histogram") {
      setXAxis(numericCols[0] || "");
      setYAxis(numericCols[1] || numericCols[0] || "");
    } else {
      setXAxis(catCols[0] || columns[0] || "");
      setYAxis(numericCols[0] || "");
    }
  };

  useEffect(() => {
    if (columns.length > 0) {
      if (!xAxis) setXAxis(catCols[0] || columns[0]);
      if (!yAxis) setYAxis(numericCols[0] || columns[1] || columns[0]);
    }
  }, [preview]);

  const handleGenerate = async () => {
    if (!hasDataset || loading) return;
    setLoading(true);
    setError(null);
    try {
      const res = await visualizationApi.generateChart({
        chart_type: chartType,
        x: xAxis || undefined,
        y: yAxis || undefined,
        color: colorCol === "None" ? undefined : colorCol,
        size: sizeCol === "None" ? undefined : sizeCol,
        top_n: topN,
        nbins: nbins,
        hole: hole,
        template: template,
        colorscale: colorscale,
        trendline: trendline,
        title: title || undefined,
      });
      setChartResponse(res);
      setRenderedSettings(settingsKey);
      success("Chart Rendered", `Generated ${chartType} visualization.`);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to render visualization.");
      toastError("Render Error", err.response?.data?.detail);
    } finally {
      setLoading(false);
    }
  };

  if (!hasDataset) {
    return (
      <DatasetEmptyState
        title="Visualizations"
        description="Build charts from your columns and pin useful views to the BI Canvas."
      />
    );
  }

  const activeTitle =
    title ||
    `${chartType}: ${yAxis ? `${yAxis} by ${xAxis}` : xAxis || "Distribution"}`;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white tracking-tight flex items-center gap-2.5">
          <BarChart3 className="h-6 w-6 text-indigo-500" />
          Interactive Visualizations Studio
        </h2>
        <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-1">
          Build and customize 14+ chart architectures with Plotly.js, drilldown
          filters, high-res exports, and dashboard pinning
        </p>
      </div>

      {/* Chart Type Selector Pills */}
      <div className="flex flex-wrap items-center gap-2">
        {presets.map((preset) => (
          <button type="button" key={preset.chart} aria-pressed={chartType === preset.chart} onClick={() => selectPreset(preset.chart)} className={`min-h-11 rounded-xl border px-3 text-sm font-semibold ${chartType === preset.chart ? 'border-blue-600 bg-blue-600 text-white' : 'border-slate-300 dark:border-slate-700'}`}>
            {preset.label}
          </button>
        ))}
      </div>
      <details className="rounded-xl border border-slate-200 p-3 dark:border-slate-700">
        <summary className="cursor-pointer text-sm font-semibold">All chart types · {chartType}</summary>
        <div className="mt-3 flex flex-wrap gap-2">
        {chartTypes.map((ct) => (
          <button
            key={ct.name}
            onClick={() => setChartType(ct.name)}
            className={`flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs sm:text-sm font-semibold flex-shrink-0 transition-all ${
              chartType === ct.name
                ? "bg-blue-600 text-white shadow-md shadow-blue-600/20"
                : "bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 text-slate-700 dark:text-slate-300 hover:border-slate-300 dark:hover:border-slate-700"
            }`}
          >
            <ct.icon className="h-4 w-4" />
            <span>{ct.name}</span>
          </button>
        ))}
        </div>
      </details>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Controls Sidebar */}
        <div className="lg:col-span-1 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 p-5 space-y-4 shadow-sm">
          <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2 pb-2 border-b border-slate-200 dark:border-slate-800">
            <Sliders className="h-4 w-4 text-blue-500" />
            Chart Parameters
          </h3>

          <div className="space-y-3.5 text-xs sm:text-sm">
            <details className="space-y-2">
              <summary className="cursor-pointer font-semibold">Chart styling</summary>
              <label htmlFor="chart-title" className="block text-slate-700 dark:text-slate-300 font-semibold mb-1">
                Custom chart title
              </label>
              <input
                type="text"
                id="chart-title"
                placeholder={activeTitle}
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                className="w-full bg-slate-100 dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 text-slate-900 dark:text-slate-200 focus:outline-none focus:border-blue-500"
              />
              <label htmlFor="chart-palette" className="block font-semibold">Color palette</label>
              <select id="chart-palette" value={colorscale} onChange={event => setColorscale(event.target.value)} className="w-full rounded-xl border border-slate-300 bg-slate-100 px-3 py-2 dark:border-slate-700 dark:bg-slate-950">
                {['Blues', 'Viridis', 'Plasma', 'Turbo', 'RdBu'].map(palette => <option key={palette}>{palette}</option>)}
              </select>
            </details>

            {/* X-Axis / Primary Column */}
            {!["Correlation Heatmap", "KPI Dashboard"].includes(chartType) && (
              <div>
                <label className="block text-slate-700 dark:text-slate-300 font-semibold mb-1">
                  X-Axis / Category
                </label>
                <select
                  aria-label="X axis or category"
                  value={xAxis}
                  onChange={(e) => setXAxis(e.target.value)}
                  className="w-full bg-slate-100 dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 text-slate-900 dark:text-slate-200 focus:outline-none focus:border-blue-500 font-mono text-xs"
                >
                  <option value="">(Auto Detect)</option>
                  {columns.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {/* Y-Axis / Metric Column */}
            {![
              "Pie Chart",
              "Histogram",
              "Treemap",
              "Sunburst",
              "Heatmap",
              "KPI Dashboard",
            ].includes(chartType) && (
              <div>
                <label className="block text-slate-700 dark:text-slate-300 font-semibold mb-1">
                  Y-Axis / Metric
                </label>
                <select
                  aria-label="Y axis or metric"
                  value={yAxis}
                  onChange={(e) => setYAxis(e.target.value)}
                  className="w-full bg-slate-100 dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 text-slate-900 dark:text-slate-200 focus:outline-none focus:border-blue-500 font-mono text-xs"
                >
                  <option value="">(Auto Detect)</option>
                  {numericCols.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {/* Color / Grouping Column */}
            {!["Histogram", "Heatmap", "KPI Dashboard"].includes(chartType) && (
              <div>
                <label className="block text-slate-700 dark:text-slate-300 font-semibold mb-1">
                  Color / Segment Group
                </label>
                <select
                  aria-label="Color grouping"
                  value={colorCol}
                  onChange={(e) => setColorCol(e.target.value)}
                  className="w-full bg-slate-100 dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 text-slate-900 dark:text-slate-200 focus:outline-none focus:border-blue-500 font-mono text-xs"
                >
                  <option value="None">None</option>
                  {columns.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {["Scatter Plot", "Bubble Chart"].includes(chartType) && (
              <div>
                <label className="block text-slate-700 dark:text-slate-300 font-semibold mb-1">
                  Size Metric
                </label>
                <select
                  aria-label="Size metric"
                  value={sizeCol}
                  onChange={(e) => setSizeCol(e.target.value)}
                  className="w-full bg-slate-100 dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 text-slate-900 dark:text-slate-200 font-mono text-xs"
                >
                  <option value="None">None</option>
                  {numericCols.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>
            )}

            <button
              onClick={() => handleGenerate()}
              disabled={loading}
              className="w-full mt-4 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white font-bold text-xs sm:text-sm flex items-center justify-center gap-2 shadow-md shadow-blue-600/20 transition-all"
            >
              <Play className="h-4 w-4" />
              {loading ? "Rendering..." : "Render / Refresh Chart"}
            </button>
          </div>
        </div>

        {/* Chart View Area */}
        <div className="lg:col-span-3 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 p-5 space-y-4 shadow-sm">
          {error && (
            <div className="p-4 rounded-xl flex items-center gap-3 text-xs sm:text-sm font-medium border bg-rose-50 dark:bg-rose-950/40 border-rose-200 dark:border-rose-500/30 text-rose-800 dark:text-rose-300">
              <AlertCircle className="h-4 w-4 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {loading ? (
            <div className="h-[520px] flex items-center justify-center text-sm text-slate-400 animate-pulse">
              Generating {chartType}...
            </div>
          ) : chartResponse?.figure_spec ? (
            <>
              {renderedSettings !== settingsKey && <p role="status" className="rounded-lg bg-amber-50 p-3 text-sm text-amber-900 dark:bg-amber-950 dark:text-amber-200">Showing the previous chart. Render again to apply your changes.</p>}
              <PlotlyChart
                spec={chartResponse.figure_spec}
                height={520}
                title={chartResponse.title}
                chartType={chartResponse.chart_type}
                sourcePage="Visualizations Studio"
              />
            </>
          ) : (
            <div className="h-[520px] flex items-center justify-center text-sm text-slate-400">
              Configure parameters on the left and click Render Chart.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
