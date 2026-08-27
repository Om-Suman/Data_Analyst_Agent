import React, { useState, useEffect } from 'react';
import {
  TrendingUp,
  Sliders,
  Play,
  Download,
  AlertCircle,
  Activity,
} from 'lucide-react';
import { useDataset } from '../context/DatasetContext';
import { forecastingApi } from '../api/client';
import { ForecastingResponse } from '../types';
import { PlotlyChart } from '../components/PlotlyChart';
import { DataTable } from '../components/DataTable';

export const ForecastingPage: React.FC = () => {
  const { preview, hasDataset } = useDataset();

  const [targetCol, setTargetCol] = useState('');
  const [method, setMethod] = useState('Moving Average');
  const [horizon, setHorizon] = useState(30);
  const [windowSize, setWindowSize] = useState(7);
  const [alpha, setAlpha] = useState(0.3);

  const [loading, setLoading] = useState(false);
  const [forecast, setForecast] = useState<ForecastingResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const numericCols = preview?.numeric_cols || [];

  useEffect(() => {
    if (numericCols.length > 0 && !targetCol) {
      setTargetCol(numericCols[0]);
    }
  }, [numericCols]);

  const handleRun = async () => {
    if (!hasDataset || !targetCol) return;
    setLoading(true);
    setError(null);
    try {
      const res = await forecastingApi.run({
        target_col: targetCol,
        method: method,
        horizon: horizon,
        window: windowSize,
        alpha: alpha,
      });
      setForecast(res);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to generate forecast.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (hasDataset && targetCol && !forecast) {
      handleRun();
    }
  }, [targetCol, hasDataset]);

  if (!hasDataset) {
    return (
      <div className="p-8 text-center text-slate-500 border border-slate-800 rounded-xl bg-slate-900/30">
        Please load or select a dataset first to run time series forecasting.
      </div>
    );
  }

  const tableData =
    forecast?.forecast_index?.map((period, idx) => ({
      Period: period,
      'Forecast Value': forecast.forecast_values[idx] !== null ? forecast.forecast_values[idx]?.toFixed(2) : '—',
      'Lower 95% CI': forecast.confidence_lower[idx] !== null ? forecast.confidence_lower[idx]?.toFixed(2) : '—',
      'Upper 95% CI': forecast.confidence_upper[idx] !== null ? forecast.confidence_upper[idx]?.toFixed(2) : '—',
    })) || [];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-emerald-400" />
          Time Series Forecasting Studio
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Predict future trends with Moving Average, Linear Trend Regression, or Exponential Smoothing
        </p>
      </div>

      {/* Control Configuration Card */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
        <div className="flex items-center gap-2 pb-2 border-b border-slate-800">
          <Sliders className="h-4 w-4 text-blue-400" />
          <span className="text-xs font-semibold text-slate-200 uppercase tracking-wider">Forecast Model Parameters</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          <div>
            <label className="block text-slate-300 font-medium mb-1">Target Numeric Column</label>
            <select
              value={targetCol}
              onChange={(e) => setTargetCol(e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-200 font-mono"
            >
              {numericCols.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-slate-300 font-medium mb-1">Forecasting Algorithm</label>
            <select
              value={method}
              onChange={(e) => setMethod(e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-200"
            >
              <option value="Moving Average">Moving Average</option>
              <option value="Linear Trend">Linear Trend (OLS)</option>
              <option value="Exponential Smoothing">Exponential Smoothing (Holt-Winters)</option>
            </select>
          </div>

          <div>
            <label className="block text-slate-300 font-medium mb-1">Forecast Horizon: {horizon} periods</label>
            <input
              type="range"
              min="7"
              max="180"
              value={horizon}
              onChange={(e) => setHorizon(parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          {method === 'Moving Average' && (
            <div>
              <label className="block text-slate-300 font-medium mb-1">Window Size: {windowSize} steps</label>
              <input
                type="range"
                min="3"
                max="60"
                value={windowSize}
                onChange={(e) => setWindowSize(parseInt(e.target.value))}
                className="w-full"
              />
            </div>
          )}

          {method === 'Exponential Smoothing' && (
            <div>
              <label className="block text-slate-300 font-medium mb-1">Alpha (Smoothing Factor): {alpha}</label>
              <input
                type="range"
                min="0.05"
                max="0.95"
                step="0.05"
                value={alpha}
                onChange={(e) => setAlpha(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
          )}
        </div>

        <div className="flex justify-end pt-2">
          <button
            onClick={handleRun}
            disabled={loading}
            className="px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold flex items-center gap-2 shadow-md shadow-blue-600/20 transition-all"
          >
            <Play className="h-3.5 w-3.5" />
            {loading ? 'Computing Forecast...' : 'Run Forecast'}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl flex items-center gap-3 text-xs font-medium border bg-rose-950/40 border-rose-500/30 text-rose-300">
          <AlertCircle className="h-4 w-4" />
          <span>{error}</span>
        </div>
      )}

      {/* Forecast Output Area */}
      {forecast && (
        <div className="space-y-6">
          {/* Chart Card */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold text-slate-200">
                  {forecast.method} Forecast for {forecast.column} (+{forecast.horizon} periods)
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">{forecast.interpretation}</p>
              </div>
              <a
                href={forecastingApi.getDownloadUrl()}
                download="forecast.csv"
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 text-xs font-medium transition-colors"
              >
                <Download className="h-3.5 w-3.5" />
                Export CSV
              </a>
            </div>

            <PlotlyChart spec={forecast.figure_spec} height={460} />
          </div>

          {/* Forecast Summary Table */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
              <Activity className="h-4 w-4 text-emerald-400" />
              Predicted Values with 95% Confidence Intervals
            </h4>
            <DataTable data={tableData} pageSize={10} showSearch={false} />
          </div>
        </div>
      )}
    </div>
  );
};
