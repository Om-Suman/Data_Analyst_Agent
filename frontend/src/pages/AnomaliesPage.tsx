import React, { useState, useEffect } from "react";
import {
  AlertTriangle,
  Sliders,
  Play,
  Download,
  AlertCircle,
  ShieldAlert,
  CheckCircle,
} from "lucide-react";
import { useDataset } from "../context/DatasetContext";
import { anomalyApi } from "../api/client";
import { AnomalyResponse } from "../types";
import { MetricCard } from "../components/MetricCard";
import { PlotlyChart } from "../components/PlotlyChart";
import { DataTable } from "../components/DataTable";

export const AnomaliesPage: React.FC = () => {
  const { hasDataset } = useDataset();

  const [method, setMethod] = useState("Isolation Forest");
  const [contamination, setContamination] = useState(0.05);
  const [threshold, setThreshold] = useState(3.0);
  const [factor, setFactor] = useState(1.5);

  const [loading, setLoading] = useState(false);
  const [anomalyResult, setAnomalyResult] = useState<AnomalyResponse | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);

  const handleRun = async () => {
    if (!hasDataset) return;
    setLoading(true);
    setError(null);
    try {
      const res = await anomalyApi.run({
        method: method,
        contamination: contamination,
        threshold: threshold,
        factor: factor,
      });
      setAnomalyResult(res);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to detect anomalies.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (hasDataset && !anomalyResult) {
      handleRun();
    }
  }, [hasDataset]);

  if (!hasDataset) {
    return (
      <div className="p-8 text-center text-slate-500 dark:text-slate-400 border border-slate-200 dark:border-slate-800 rounded-2xl bg-white dark:bg-slate-900/30 shadow-sm">
        Please load or select a dataset first to run anomaly detection.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white tracking-tight flex items-center gap-2">
          <AlertTriangle className="h-5 w-5 text-rose-500 dark:text-rose-400" />
          Anomaly & Outlier Detection
        </h2>
        <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-1">
          Detect outliers, erroneous transactions, and irregular patterns with
          Isolation Forest, Z-Score, or IQR
        </p>
      </div>

      {/* Model Parameter Configuration */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 p-5 sm:p-6 space-y-4 shadow-sm">
        <div className="flex items-center gap-2 pb-2 border-b border-slate-200 dark:border-slate-800">
          <Sliders className="h-4 w-4 text-blue-500 dark:text-blue-400" />
          <span className="text-xs font-bold text-slate-900 dark:text-slate-200 uppercase tracking-wider">
            Detection Parameters
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
          <div>
            <label className="block text-slate-700 dark:text-slate-300 font-medium mb-1">
              Detection Algorithm
            </label>
            <select
              value={method}
              onChange={(e) => setMethod(e.target.value)}
              className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 text-slate-900 dark:text-slate-200 shadow-sm"
            >
              <option value="Isolation Forest">
                Isolation Forest (Multivariate)
              </option>
              <option value="Z-Score">Z-Score (Gaussian Distribution)</option>
              <option value="IQR">IQR (Interquartile Range)</option>
            </select>
          </div>

          {method === "Isolation Forest" && (
            <div>
              <label className="block text-slate-700 dark:text-slate-300 font-medium mb-1">
                Contamination Rate: {(contamination * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0.01"
                max="0.25"
                step="0.01"
                value={contamination}
                onChange={(e) => setContamination(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
          )}

          {method === "Z-Score" && (
            <div>
              <label className="block text-slate-700 dark:text-slate-300 font-medium mb-1">
                Z-Score Threshold (σ): {threshold}
              </label>
              <input
                type="range"
                min="1.5"
                max="5.0"
                step="0.1"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
          )}

          {method === "IQR" && (
            <div>
              <label className="block text-slate-700 dark:text-slate-300 font-medium mb-1">
                IQR Factor: {factor}x
              </label>
              <input
                type="range"
                min="1.0"
                max="3.0"
                step="0.1"
                value={factor}
                onChange={(e) => setFactor(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
          )}

          <div className="flex items-end">
            <button
              onClick={handleRun}
              disabled={loading}
              className="w-full py-2.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-semibold flex items-center justify-center gap-2 shadow-md shadow-rose-600/20 transition-all text-xs"
            >
              <Play className="h-3.5 w-3.5" />
              {loading ? "Analyzing Data..." : "Run Anomaly Scan"}
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl flex items-center gap-3 text-xs font-medium border bg-rose-50 dark:bg-rose-950/40 border-rose-200 dark:border-rose-500/30 text-rose-800 dark:text-rose-300">
          <AlertCircle className="h-4 w-4" />
          <span>{error}</span>
        </div>
      )}

      {/* Anomaly Results */}
      {anomalyResult && (
        <div className="space-y-6">
          {/* KPI Row */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <MetricCard
              title="Flagged Anomalies"
              value={anomalyResult.n_anomalies.toLocaleString()}
              subtitle={`Algorithm: ${anomalyResult.method}`}
              icon={ShieldAlert}
              color="red"
            />
            <MetricCard
              title="Anomaly Ratio"
              value={`${anomalyResult.anomaly_rate}%`}
              subtitle="Percentage of dataset flagged"
              icon={AlertTriangle}
              color="amber"
            />
            <MetricCard
              title="Features Analyzed"
              value={anomalyResult.columns_used.length}
              subtitle="Numeric dimensions"
              icon={CheckCircle}
              color="blue"
            />
          </div>

          {/* Anomaly Chart */}
          <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 p-5 sm:p-6 space-y-4 shadow-sm">
            <h3 className="text-sm font-bold text-slate-900 dark:text-slate-200">
              Anomaly Projection Visualization (PCA 2D Cluster)
            </h3>
            <PlotlyChart spec={anomalyResult.figure_spec} height={460} />
          </div>

          {/* Flagged Rows Table */}
          <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 p-5 sm:p-6 space-y-4 shadow-sm">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-slate-900 dark:text-slate-200">
                  Flagged Anomalous Records
                </h3>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Inspecting records deviating significantly from expected
                  patterns
                </p>
              </div>
              <a
                href={anomalyApi.getDownloadUrl()}
                download="anomalies.csv"
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700 text-xs font-medium transition-colors"
              >
                <Download className="h-3.5 w-3.5" />
                Export Anomalies CSV
              </a>
            </div>

            <DataTable
              data={anomalyResult.anomalous_rows}
              pageSize={10}
              emptyMessage="No anomalies detected with current parameters."
            />
          </div>
        </div>
      )}
    </div>
  );
};
