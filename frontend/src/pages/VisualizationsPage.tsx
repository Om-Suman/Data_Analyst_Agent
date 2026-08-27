import React, { useState, useEffect } from 'react';
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
} from 'lucide-react';
import { useDataset } from '../context/DatasetContext';
import { visualizationApi } from '../api/client';
import { VisualizationResponse } from '../types';
import { PlotlyChart } from '../components/PlotlyChart';

export const VisualizationsPage: React.FC = () => {
  const { preview, hasDataset } = useDataset();

  const [chartType, setChartType] = useState('Bar Chart');
  const [xAxis, setXAxis] = useState('');
  const [yAxis, setYAxis] = useState('');
  const [colorCol, setColorCol] = useState('None');
  const [sizeCol, setSizeCol] = useState('None');
  const [topN, setTopN] = useState(20);
  const [nbins, setNbins] = useState(30);
  const [hole, setHole] = useState(0.0);
  const [template, setTemplate] = useState('plotly_dark');
  const [colorscale, setColorscale] = useState('Blues');
  const [trendline, setTrendline] = useState(false);
  const [title, setTitle] = useState('');

  const [loading, setLoading] = useState(false);
  const [chartResponse, setChartResponse] = useState<VisualizationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const chartTypes = [
    { name: 'Bar Chart', icon: BarChart3 },
    { name: 'Line Chart', icon: LineChart },
    { name: 'Scatter Plot', icon: ScatterChart },
    { name: 'Histogram', icon: BarChart3 },
    { name: 'Box Plot', icon: Activity },
    { name: 'Violin Plot', icon: Activity },
    { name: 'Pie Chart', icon: PieChart },
    { name: 'Area Chart', icon: AreaChart },
    { name: 'Heatmap', icon: Layers },
    { name: 'Treemap', icon: Layers },
    { name: 'Sunburst', icon: PieChart },
    { name: 'Bubble Chart', icon: ScatterChart },
    { name: 'Funnel Chart', icon: BarChart3 },
    { name: 'KPI Dashboard', icon: Activity },
  ];

  const columns = preview?.columns || [];
  const numericCols = preview?.numeric_cols || [];
  const catCols = preview?.categorical_cols || [];

  useEffect(() => {
    if (columns.length > 0) {
      if (!xAxis) setXAxis(columns[0]);
      if (!yAxis) setYAxis(numericCols.length > 0 ? numericCols[0] : columns[0]);
    }
  }, [columns, numericCols]);

  const handleGenerate = async () => {
    if (!hasDataset) return;
    setLoading(true);
    setError(null);
    try {
      const res = await visualizationApi.generateChart({
        chart_type: chartType,
        x: xAxis || undefined,
        y: yAxis || undefined,
        color: colorCol !== 'None' ? colorCol : undefined,
        size: sizeCol !== 'None' ? sizeCol : undefined,
        top_n: topN,
        nbins: nbins,
        hole: hole,
        template: template,
        colorscale: colorscale,
        trendline: trendline,
        title: title || `${chartType} of ${yAxis || xAxis}`,
      });
      setChartResponse(res);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to generate visualization.');
    } finally {
      setLoading(false);
    }
  };

  // Auto-generate on load if fields are present
  useEffect(() => {
    if (hasDataset && xAxis && yAxis && !chartResponse) {
      handleGenerate();
    }
  }, [xAxis, yAxis, hasDataset]);

  if (!hasDataset) {
    return (
      <div className="p-8 text-center text-slate-500 border border-slate-800 rounded-xl bg-slate-900/30">
        Please load or select a dataset first to generate interactive visualizations.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
          <BarChart3 className="h-5 w-5 text-blue-400" />
          Interactive Visualizations Studio
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Create, customize, and export 14+ interactive chart types powered by Plotly
        </p>
      </div>

      {/* Chart Type Selector Pills */}
      <div className="flex flex-wrap items-center gap-2 pb-2 border-b border-slate-800">
        {chartTypes.map((t) => (
          <button
            key={t.name}
            onClick={() => {
              setChartType(t.name);
              setTimeout(handleGenerate, 50);
            }}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              chartType === t.name
                ? 'bg-blue-600 text-white shadow-md shadow-blue-600/20'
                : 'bg-slate-900/60 text-slate-400 hover:text-slate-200 border border-slate-800 hover:bg-slate-800/60'
            }`}
          >
            <t.icon className="h-3.5 w-3.5" />
            {t.name}
          </button>
        ))}
      </div>

      {/* Main Studio Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Controls Sidebar */}
        <div className="lg:col-span-1 rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4 text-xs">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800">
            <span className="font-semibold text-slate-200 flex items-center gap-1.5">
              <Sliders className="h-4 w-4 text-blue-400" />
              Chart Options
            </span>
          </div>

          <div className="space-y-3">
            <div>
              <label className="block text-slate-300 font-medium mb-1">X-Axis / Category</label>
              <select
                value={xAxis}
                onChange={(e) => setXAxis(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-200"
              >
                {columns.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-slate-300 font-medium mb-1">Y-Axis / Metric Value</label>
              <select
                value={yAxis}
                onChange={(e) => setYAxis(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-200"
              >
                {columns.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-slate-300 font-medium mb-1">Color Breakdown</label>
              <select
                value={colorCol}
                onChange={(e) => setColorCol(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-200"
              >
                <option value="None">None</option>
                {columns.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>

            {['Scatter Plot', 'Bubble Chart'].includes(chartType) && (
              <div>
                <label className="block text-slate-300 font-medium mb-1">Size Metric</label>
                <select
                  value={sizeCol}
                  onChange={(e) => setSizeCol(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-200"
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

            {chartType === 'Pie Chart' && (
              <div>
                <label className="block text-slate-300 font-medium mb-1">Donut Hole Size: {hole}</label>
                <input
                  type="range"
                  min="0.0"
                  max="0.8"
                  step="0.1"
                  value={hole}
                  onChange={(e) => setHole(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            )}

            <div>
              <label className="block text-slate-300 font-medium mb-1">Color Palette</label>
              <select
                value={colorscale}
                onChange={(e) => setColorscale(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-200"
              >
                <option value="Blues">Blues</option>
                <option value="Viridis">Viridis</option>
                <option value="Plasma">Plasma</option>
                <option value="Turbo">Turbo</option>
                <option value="RdBu">Red-Blue</option>
              </select>
            </div>

            <button
              onClick={handleGenerate}
              disabled={loading}
              className="w-full mt-4 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold flex items-center justify-center gap-2 shadow-md shadow-blue-600/20 transition-all"
            >
              <Play className="h-3.5 w-3.5" />
              {loading ? 'Rendering...' : 'Update Chart'}
            </button>
          </div>
        </div>

        {/* Chart View Area */}
        <div className="lg:col-span-3 rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
          {error && (
            <div className="p-4 rounded-xl flex items-center gap-3 text-xs font-medium border bg-rose-950/40 border-rose-500/30 text-rose-300">
              <AlertCircle className="h-4 w-4" />
              <span>{error}</span>
            </div>
          )}

          {loading ? (
            <div className="h-[500px] flex items-center justify-center text-xs text-slate-500">
              Generating {chartType}...
            </div>
          ) : chartResponse?.figure_spec ? (
            <PlotlyChart spec={chartResponse.figure_spec} height={500} />
          ) : (
            <div className="h-[500px] flex items-center justify-center text-xs text-slate-500">
              Configure options on the left and click Update Chart to render.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
