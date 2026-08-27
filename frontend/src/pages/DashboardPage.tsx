import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
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
} from 'lucide-react';
import { useDataset } from '../context/DatasetContext';
import { MetricCard } from '../components/MetricCard';
import { PlotlyChart } from '../components/PlotlyChart';
import { DataTable } from '../components/DataTable';
import { queryApi } from '../api/client';
import { QueryHistoryItem } from '../types';

export const DashboardPage: React.FC = () => {
  const { activeDataset, preview, previewLoading, hasDataset } = useDataset();
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
        <div className="h-16 w-16 rounded-2xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400 text-3xl shadow-lg shadow-blue-500/10">
          📊
        </div>
        <div className="space-y-2 max-w-md">
          <h2 className="text-2xl font-bold tracking-tight text-white">Welcome to Data Analyst Agent</h2>
          <p className="text-sm text-slate-400">
            To start exploring, querying, and visualizing data, please upload a dataset or load one of our demo datasets.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Link
            to="/upload"
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-sm font-semibold shadow-md shadow-blue-600/20 transition-all"
          >
            Upload Dataset
            <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      </div>
    );
  }

  const meta = preview?.metadata || {};
  const missingTotal = meta.missing_total ?? 0;
  const duplicateTotal = meta.duplicate_rows ?? 0;

  // Dtype pie chart spec
  const numCount = preview?.numeric_cols?.length || 0;
  const catCount = preview?.categorical_cols?.length || 0;
  const dateCount = preview?.date_cols?.length || 0;
  const otherCount = Math.max(0, (preview?.cols || 0) - numCount - catCount - dateCount);

  const dtypePieSpec = {
    data: [
      {
        values: [numCount, catCount, dateCount, otherCount].filter((v) => v > 0),
        labels: ['Numeric', 'Categorical', 'Date/Time', 'Other'].slice(0, [numCount, catCount, dateCount, otherCount].filter((v) => v > 0).length),
        type: 'pie',
        hole: 0.55,
        marker: { colors: ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6'] },
        textinfo: 'label+percent',
      },
    ],
    layout: {
      title: 'Column Types Composition',
      height: 260,
      margin: { l: 20, r: 20, t: 30, b: 20 },
      showlegend: false,
    },
  };

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-6 rounded-2xl bg-gradient-to-r from-blue-950/40 via-slate-900/60 to-slate-900/40 border border-blue-500/20">
        <div>
          <span className="text-xs font-semibold text-blue-400 uppercase tracking-wider">Active Workspace</span>
          <h2 className="text-2xl font-bold text-white tracking-tight mt-1">{activeDataset?.name}</h2>
          <p className="text-xs text-slate-400 mt-1">
            Source: <span className="text-slate-300 font-mono">{activeDataset?.source}</span> • Version: v{activeDataset?.version}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Link
            to="/query"
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow-md shadow-blue-600/20 transition-all"
          >
            <Sparkles className="h-3.5 w-3.5" />
            Ask AI Query
          </Link>
          <Link
            to="/reports"
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-all"
          >
            <FileSpreadsheet className="h-3.5 w-3.5" />
            Export Report
          </Link>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Total Rows"
          value={activeDataset?.rows.toLocaleString() || '0'}
          subtitle="Dataset volume"
          icon={Database}
          color="blue"
        />
        <MetricCard
          title="Total Columns"
          value={activeDataset?.cols.toLocaleString() || '0'}
          subtitle={`${preview?.numeric_cols.length || 0} numeric, ${preview?.categorical_cols.length || 0} categorical`}
          icon={Columns}
          color="green"
        />
        <MetricCard
          title="Missing Values"
          value={missingTotal.toLocaleString()}
          subtitle={`${((missingTotal / Math.max(1, (activeDataset?.rows || 1) * (activeDataset?.cols || 1))) * 100).toFixed(1)}% of all cells`}
          icon={AlertCircle}
          color={missingTotal > 0 ? 'amber' : 'green'}
        />
        <MetricCard
          title="Duplicate Rows"
          value={duplicateTotal.toLocaleString()}
          subtitle="Identical records"
          icon={Copy}
          color={duplicateTotal > 0 ? 'red' : 'green'}
        />
      </div>

      {/* Analytics Summary Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-3">
          <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
            <BarChart2 className="h-4 w-4 text-blue-400" />
            Schema Breakdown
          </h3>
          <PlotlyChart spec={dtypePieSpec} height={240} />
        </div>

        {/* Quick Insights & Feature Highlights */}
        <div className="lg:col-span-2 rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-emerald-400" />
              Quick Analytics Actions
            </h3>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <Link
              to="/cleaning"
              className="p-4 rounded-lg bg-slate-800/40 hover:bg-slate-800/80 border border-slate-700/60 transition-all space-y-2 block"
            >
              <p className="text-xs font-semibold text-slate-200">Data Cleaning</p>
              <p className="text-[11px] text-slate-400">Impute missing values, drop duplicates, and fix types.</p>
            </Link>
            <Link
              to="/visualizations"
              className="p-4 rounded-lg bg-slate-800/40 hover:bg-slate-800/80 border border-slate-700/60 transition-all space-y-2 block"
            >
              <p className="text-xs font-semibold text-slate-200">Visualizations</p>
              <p className="text-[11px] text-slate-400">Generate 14+ interactive Plotly charts and heatmaps.</p>
            </Link>
            <Link
              to="/forecasting"
              className="p-4 rounded-lg bg-slate-800/40 hover:bg-slate-800/80 border border-slate-700/60 transition-all space-y-2 block"
            >
              <p className="text-xs font-semibold text-slate-200">Forecasting</p>
              <p className="text-[11px] text-slate-400">Predict future metrics with confidence bounds.</p>
            </Link>
          </div>

          {/* Recent Query Snippet */}
          {history.length > 0 && (
            <div className="pt-2 border-t border-slate-800">
              <span className="text-[11px] text-slate-400 font-semibold uppercase tracking-wider">Latest Query</span>
              <p className="text-xs text-blue-300 font-medium mt-1">"{history[0].question}"</p>
              <p className="text-[11px] text-slate-400 line-clamp-2 mt-0.5">{history[0].result_summary}</p>
            </div>
          )}
        </div>
      </div>

      {/* Dataset Preview Table */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-slate-200">Dataset Preview (First 50 Rows)</h3>
            <p className="text-xs text-slate-500">Live preview of current version in memory</p>
          </div>
          <Link to="/explorer" className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1 font-medium">
            Open Full Explorer
            <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>

        {previewLoading ? (
          <div className="h-40 flex items-center justify-center text-slate-500 text-xs">Loading dataset preview...</div>
        ) : (
          <DataTable data={preview?.data || []} columns={preview?.columns} pageSize={10} />
        )}
      </div>
    </div>
  );
};
