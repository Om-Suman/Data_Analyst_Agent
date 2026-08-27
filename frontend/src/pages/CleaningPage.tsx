import React, { useState, useEffect } from 'react';
import {
  Sparkles,
  RotateCcw,
  CheckCircle2,
  AlertTriangle,
  History,
  Play,
  Check,
} from 'lucide-react';
import { useDataset } from '../context/DatasetContext';
import { cleaningApi } from '../api/client';
import { CleaningReportResponse, QualityScoreResponse, VersionItem } from '../types';
import { DataTable } from '../components/DataTable';

export const CleaningPage: React.FC = () => {
  const { activeDataset, hasDataset, refreshDatasets, refreshPreview } = useDataset();

  const [quality, setQuality] = useState<QualityScoreResponse | null>(null);
  const [versions, setVersions] = useState<VersionItem[]>([]);
  const [loadingQuality, setLoadingQuality] = useState(false);
  const [previewReport, setPreviewReport] = useState<CleaningReportResponse | null>(null);
  const [applying, setApplying] = useState(false);
  const [previewing, setPreviewing] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Form state
  const [missingStrategy, setMissingStrategy] = useState('mean');
  const [removeDuplicates, setRemoveDuplicates] = useState(true);
  const [fixDtypes, setFixDtypes] = useState(true);
  const [normalizeNames, setNormalizeNames] = useState(true);
  const [outlierMethod, setOutlierMethod] = useState('none');
  const [outlierThreshold, setOutlierThreshold] = useState(3.0);
  const [iqrFactor, setIqrFactor] = useState(1.5);

  const fetchQualityAndVersions = async () => {
    if (!hasDataset) return;
    setLoadingQuality(true);
    try {
      const q = await cleaningApi.getQuality();
      setQuality(q);
      const v = await cleaningApi.getVersions();
      setVersions(v.versions);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingQuality(false);
    }
  };

  useEffect(() => {
    fetchQualityAndVersions();
  }, [activeDataset]);

  const handlePreview = async () => {
    setPreviewing(true);
    setMessage(null);
    try {
      const rep = await cleaningApi.previewCleaning({
        missing_strategy: missingStrategy,
        remove_duplicates: removeDuplicates,
        fix_dtypes: fixDtypes,
        normalize_column_names: normalizeNames,
        outlier_method: outlierMethod,
        outlier_threshold: outlierThreshold,
        iqr_factor: iqrFactor,
      });
      setPreviewReport(rep);
    } catch (err: any) {
      setMessage({ type: 'error', text: err.response?.data?.detail || 'Failed to preview cleaning.' });
    } finally {
      setPreviewing(false);
    }
  };

  const handleApply = async () => {
    setApplying(true);
    setMessage(null);
    try {
      const rep = await cleaningApi.applyCleaning({
        missing_strategy: missingStrategy,
        remove_duplicates: removeDuplicates,
        fix_dtypes: fixDtypes,
        normalize_column_names: normalizeNames,
        outlier_method: outlierMethod,
        outlier_threshold: outlierThreshold,
        iqr_factor: iqrFactor,
      });
      setPreviewReport(rep);
      setMessage({ type: 'success', text: `Cleaning applied successfully! Dataset bumped to v${(activeDataset?.version || 1) + 1}.` });
      await refreshDatasets();
      await refreshPreview();
      await fetchQualityAndVersions();
    } catch (err: any) {
      setMessage({ type: 'error', text: err.response?.data?.detail || 'Failed to apply cleaning.' });
    } finally {
      setApplying(false);
    }
  };

  const handleRollback = async (versionNumber: number) => {
    if (!window.confirm(`Roll back dataset to version ${versionNumber}?`)) return;
    try {
      await cleaningApi.rollbackVersion(versionNumber);
      setMessage({ type: 'success', text: `Dataset restored to version ${versionNumber}!` });
      await refreshDatasets();
      await refreshPreview();
      await fetchQualityAndVersions();
    } catch (err: any) {
      setMessage({ type: 'error', text: err.response?.data?.detail || 'Rollback failed.' });
    }
  };

  if (!hasDataset) {
    return (
      <div className="p-8 text-center text-slate-500 border border-slate-800 rounded-xl bg-slate-900/30">
        Please load or select a dataset first to use the Data Cleaning tools.
      </div>
    );
  }

  const gradeColor =
    quality?.quality_grade === 'A'
      ? 'text-emerald-400 border-emerald-500/30 bg-emerald-950/30'
      : quality?.quality_grade === 'B'
      ? 'text-blue-400 border-blue-500/30 bg-blue-950/30'
      : quality?.quality_grade === 'C'
      ? 'text-amber-400 border-amber-500/30 bg-amber-950/30'
      : 'text-rose-400 border-rose-500/30 bg-rose-950/30';

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight">Data Quality & Cleaning</h2>
        <p className="text-xs text-slate-400 mt-1">
          Detect anomalies, impute missing entries, remove duplicates, normalize column formats, and snapshot versions
        </p>
      </div>

      {message && (
        <div
          className={`p-4 rounded-xl flex items-center gap-3 text-xs font-medium border ${
            message.type === 'success'
              ? 'bg-emerald-950/40 border-emerald-500/30 text-emerald-300'
              : 'bg-rose-950/40 border-rose-500/30 text-rose-300'
          }`}
        >
          {message.type === 'success' ? <CheckCircle2 className="h-4 w-4" /> : <AlertTriangle className="h-4 w-4" />}
          <span>{message.text}</span>
        </div>
      )}

      {/* Quality Score Banner */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className={`rounded-xl border p-5 flex items-center justify-between ${gradeColor}`}>
          <div>
            <p className="text-xs uppercase font-semibold tracking-wider text-slate-400">Quality Score</p>
            <p className="text-3xl font-bold mt-1 text-white">{quality?.quality_score ?? '—'}/100</p>
          </div>
          <div className="text-3xl font-black px-3 py-1 rounded-lg border bg-slate-900/60 font-mono">
            {quality?.quality_grade ?? '—'}
          </div>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
          <p className="text-xs uppercase font-semibold tracking-wider text-slate-400">Missing Values</p>
          <p className="text-2xl font-bold mt-1 text-white">{quality?.missing_total?.toLocaleString() ?? '0'}</p>
          <p className="text-xs text-slate-500 mt-0.5">{quality?.missing_pct ?? 0}% of all cells</p>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
          <p className="text-xs uppercase font-semibold tracking-wider text-slate-400">Duplicate Rows</p>
          <p className="text-2xl font-bold mt-1 text-white">{quality?.duplicate_rows?.toLocaleString() ?? '0'}</p>
          <p className="text-xs text-slate-500 mt-0.5">{quality?.duplicate_pct ?? 0}% duplicate rate</p>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
          <p className="text-xs uppercase font-semibold tracking-wider text-slate-400">Current Version</p>
          <p className="text-2xl font-bold mt-1 text-blue-400 font-mono">v{activeDataset?.version}</p>
          <p className="text-xs text-slate-500 mt-0.5">{versions.length} version snapshots</p>
        </div>
      </div>

      {/* Cleaning Config Form */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1 rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
          <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-blue-400" />
            Cleaning Rules
          </h3>

          <div className="space-y-3 text-xs">
            <div>
              <label className="block text-slate-300 font-medium mb-1">Missing Value Strategy</label>
              <select
                value={missingStrategy}
                onChange={(e) => setMissingStrategy(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-blue-500"
              >
                <option value="mean">Mean (numeric) / Mode (categorical)</option>
                <option value="median">Median (numeric) / Mode (categorical)</option>
                <option value="mode">Mode (most frequent value)</option>
                <option value="ffill">Forward Fill (time series)</option>
                <option value="bfill">Backward Fill</option>
                <option value="drop_rows">Drop rows with missing values</option>
                <option value="drop_cols">Drop columns with &gt;50% missing</option>
                <option value="none">Do not impute missing</option>
              </select>
            </div>

            <div>
              <label className="block text-slate-300 font-medium mb-1">Outlier Treatment</label>
              <select
                value={outlierMethod}
                onChange={(e) => setOutlierMethod(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-blue-500"
              >
                <option value="none">None (keep outliers)</option>
                <option value="zscore">Z-Score Filtering</option>
                <option value="iqr">IQR (Interquartile Range) Filtering</option>
              </select>
            </div>

            {outlierMethod === 'zscore' && (
              <div>
                <label className="block text-slate-400 mb-1">Z-Score Threshold (σ): {outlierThreshold}</label>
                <input
                  type="range"
                  min="1.5"
                  max="5.0"
                  step="0.1"
                  value={outlierThreshold}
                  onChange={(e) => setOutlierThreshold(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            )}

            {outlierMethod === 'iqr' && (
              <div>
                <label className="block text-slate-400 mb-1">IQR Multiplier Factor: {iqrFactor}x</label>
                <input
                  type="range"
                  min="1.0"
                  max="3.0"
                  step="0.1"
                  value={iqrFactor}
                  onChange={(e) => setIqrFactor(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            )}

            <div className="space-y-2 pt-2 border-t border-slate-800">
              <label className="flex items-center gap-2 cursor-pointer text-slate-300">
                <input
                  type="checkbox"
                  checked={removeDuplicates}
                  onChange={(e) => setRemoveDuplicates(e.target.checked)}
                  className="rounded bg-slate-900 border-slate-700 text-blue-500"
                />
                Remove Duplicate Rows
              </label>

              <label className="flex items-center gap-2 cursor-pointer text-slate-300">
                <input
                  type="checkbox"
                  checked={fixDtypes}
                  onChange={(e) => setFixDtypes(e.target.checked)}
                  className="rounded bg-slate-900 border-slate-700 text-blue-500"
                />
                Auto-Convert Inferred Data Types (Dates, Numbers)
              </label>

              <label className="flex items-center gap-2 cursor-pointer text-slate-300">
                <input
                  type="checkbox"
                  checked={normalizeNames}
                  onChange={(e) => setNormalizeNames(e.target.checked)}
                  className="rounded bg-slate-900 border-slate-700 text-blue-500"
                />
                Normalize Column Names (snake_case)
              </label>
            </div>

            <div className="flex items-center gap-2 pt-4">
              <button
                onClick={handlePreview}
                disabled={previewing}
                className="flex-1 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold border border-slate-700 flex items-center justify-center gap-1.5 transition-colors"
              >
                <Play className="h-3.5 w-3.5 text-blue-400" />
                {previewing ? 'Testing...' : 'Dry Run Preview'}
              </button>
              <button
                onClick={handleApply}
                disabled={applying}
                className="flex-1 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold flex items-center justify-center gap-1.5 shadow-md shadow-blue-600/20 transition-colors"
              >
                <Check className="h-3.5 w-3.5" />
                {applying ? 'Applying...' : 'Apply & Save'}
              </button>
            </div>
          </div>
        </div>

        {/* Cleaning Report / Preview */}
        <div className="lg:col-span-2 rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
          <h3 className="text-sm font-semibold text-slate-200">
            {previewReport ? 'Cleaning Impact Report' : 'Current Data Health Breakdown'}
          </h3>

          {previewReport ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                <div className="p-3 rounded-lg bg-slate-800/40 border border-slate-700">
                  <span className="text-slate-400">Rows After</span>
                  <p className="text-lg font-bold text-white mt-1">
                    {previewReport.rows_after.toLocaleString()}{' '}
                    <span className="text-xs text-rose-400">
                      ({previewReport.rows_after - previewReport.rows_before})
                    </span>
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-slate-800/40 border border-slate-700">
                  <span className="text-slate-400">Duplicates Dropped</span>
                  <p className="text-lg font-bold text-emerald-400 mt-1">{previewReport.duplicates_removed}</p>
                </div>
                <div className="p-3 rounded-lg bg-slate-800/40 border border-slate-700">
                  <span className="text-slate-400">Outliers Removed</span>
                  <p className="text-lg font-bold text-amber-400 mt-1">{previewReport.outliers_removed}</p>
                </div>
                <div className="p-3 rounded-lg bg-slate-800/40 border border-slate-700">
                  <span className="text-slate-400">Score Improvement</span>
                  <p className="text-lg font-bold text-blue-400 mt-1">
                    {previewReport.quality_score_before} → {previewReport.quality_score_after} ({previewReport.quality_grade_after})
                  </p>
                </div>
              </div>

              {previewReport.recommendations?.length > 0 && (
                <div className="p-3 rounded-lg bg-blue-950/20 border border-blue-500/20 text-xs text-blue-300 space-y-1">
                  <span className="font-semibold text-blue-200">Recommendations:</span>
                  <ul className="list-disc list-inside space-y-0.5 text-[11px]">
                    {previewReport.recommendations.map((rec, i) => (
                      <li key={i}>{rec}</li>
                    ))}
                  </ul>
                </div>
              )}

              {previewReport.preview_data && (
                <div className="space-y-2">
                  <p className="text-xs font-semibold text-slate-300">Cleaned Data Sample</p>
                  <DataTable data={previewReport.preview_data} pageSize={5} showSearch={false} />
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              <p className="text-xs text-slate-400">Columns with missing values:</p>
              {quality?.missing_by_column && quality.missing_by_column.length > 0 ? (
                <div className="space-y-2">
                  {quality.missing_by_column.map((col) => (
                    <div key={col.column} className="flex items-center justify-between text-xs p-2.5 rounded bg-slate-800/40 border border-slate-700">
                      <span className="font-mono text-slate-200">{col.column}</span>
                      <span className="text-amber-400 font-mono">
                        {col.missing_count} missing ({col.missing_pct}%)
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="p-4 rounded-lg bg-emerald-950/20 border border-emerald-500/20 text-xs text-emerald-400 flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4" />
                  No missing values detected in the active dataset!
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Version History & Rollback Timeline */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
        <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
          <History className="h-4 w-4 text-blue-400" />
          Version History & Snapshot Rollback
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-slate-800/60 text-[11px] uppercase tracking-wider text-slate-400">
              <tr>
                <th className="px-4 py-3">Version</th>
                <th className="px-4 py-3">Timestamp</th>
                <th className="px-4 py-3">Rows</th>
                <th className="px-4 py-3">Cols</th>
                <th className="px-4 py-3">Description</th>
                <th className="px-4 py-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {versions.map((v) => {
                const isCurrent = v.version === activeDataset?.version;
                return (
                  <tr key={v.version} className="hover:bg-slate-800/30 transition-colors">
                    <td className="px-4 py-3 font-mono font-bold text-white">v{v.version}</td>
                    <td className="px-4 py-3 text-slate-400">{new Date(v.timestamp).toLocaleString()}</td>
                    <td className="px-4 py-3 font-mono">{v.rows.toLocaleString()}</td>
                    <td className="px-4 py-3 font-mono">{v.cols}</td>
                    <td className="px-4 py-3 text-slate-300">{v.description || 'Snapshot'}</td>
                    <td className="px-4 py-3 text-right">
                      {isCurrent ? (
                        <span className="text-[11px] text-emerald-400 font-semibold">Active Version</span>
                      ) : (
                        <button
                          onClick={() => handleRollback(v.version)}
                          className="flex items-center gap-1 text-[11px] text-blue-400 hover:text-blue-300 font-medium ml-auto"
                        >
                          <RotateCcw className="h-3 w-3" />
                          Rollback
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
