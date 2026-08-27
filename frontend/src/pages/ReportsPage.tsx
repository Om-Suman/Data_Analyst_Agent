import React, { useState, useEffect } from 'react';
import {
  FileSpreadsheet,
  Download,
  FileText,
  FileCode,
  CheckCircle,
  Table,
} from 'lucide-react';
import { useDataset } from '../context/DatasetContext';
import { datasetApi, reportsApi } from '../api/client';
import { ProfileResponse } from '../types';

export const ReportsPage: React.FC = () => {
  const { activeDataset, hasDataset } = useDataset();

  const [includeSample, setIncludeSample] = useState(true);
  const [includeStats, setIncludeStats] = useState(true);
  const [includeInsights, setIncludeInsights] = useState(true);
  const [downloadingHtml, setDownloadingHtml] = useState(false);
  const [downloadingExcel, setDownloadingExcel] = useState(false);
  const [profile, setProfile] = useState<ProfileResponse | null>(null);

  useEffect(() => {
    if (!hasDataset) return;
    reportsApi
      .getProfile()
      .then((res) => setProfile(res))
      .catch((err) => console.error(err));
  }, [hasDataset]);

  const handleDownloadHtml = async () => {
    setDownloadingHtml(true);
    try {
      await reportsApi.downloadHtmlReport({
        include_sample: includeSample,
        include_stats: includeStats,
        include_insights: includeInsights,
      });
    } catch (err) {
      console.error(err);
      alert('Failed to generate HTML report.');
    } finally {
      setDownloadingHtml(false);
    }
  };

  const handleDownloadExcel = async () => {
    setDownloadingExcel(true);
    try {
      await reportsApi.downloadExcelReport();
    } catch (err) {
      console.error(err);
      alert('Failed to export Excel report.');
    } finally {
      setDownloadingExcel(false);
    }
  };

  if (!hasDataset) {
    return (
      <div className="p-8 text-center text-slate-500 border border-slate-800 rounded-xl bg-slate-900/30">
        Please load or select a dataset first to generate exports and reports.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
          <FileSpreadsheet className="h-5 w-5 text-blue-400" />
          Report Export & Data Profiling Center
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Export production-ready interactive HTML executive reports, comprehensive multi-tab Excel workbooks, or raw cleaned data
        </p>
      </div>

      {/* Export Options Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* HTML Report Card */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 flex flex-col justify-between space-y-4">
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <FileCode className="h-5 w-5 text-blue-400" />
              <h3 className="text-sm font-semibold text-slate-200">Interactive HTML Executive Report</h3>
            </div>
            <p className="text-xs text-slate-400">
              Self-contained HTML report with CSS dark styling, KPIs, statistical summaries, missingness breakdown, and AI insights.
            </p>

            <div className="space-y-2 pt-2 text-xs text-slate-300">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={includeSample}
                  onChange={(e) => setIncludeSample(e.target.checked)}
                  className="rounded bg-slate-900 border-slate-700 text-blue-500"
                />
                Include 20-row sample table
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={includeStats}
                  onChange={(e) => setIncludeStats(e.target.checked)}
                  className="rounded bg-slate-900 border-slate-700 text-blue-500"
                />
                Include numeric distribution stats
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={includeInsights}
                  onChange={(e) => setIncludeInsights(e.target.checked)}
                  className="rounded bg-slate-900 border-slate-700 text-blue-500"
                />
                Include AI Business Intelligence
              </label>
            </div>
          </div>

          <button
            onClick={handleDownloadHtml}
            disabled={downloadingHtml}
            className="w-full py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold flex items-center justify-center gap-2 text-xs shadow-md shadow-blue-600/20 transition-all"
          >
            <Download className="h-4 w-4" />
            {downloadingHtml ? 'Generating HTML...' : 'Download HTML Report'}
          </button>
        </div>

        {/* Multi-Tab Excel Workbook Card */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 flex flex-col justify-between space-y-4">
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <FileSpreadsheet className="h-5 w-5 text-emerald-400" />
              <h3 className="text-sm font-semibold text-slate-200">Multi-Sheet Excel Workbook</h3>
            </div>
            <p className="text-xs text-slate-400">
              Complete .xlsx workbook formatted across 5 specialized sheets:
            </p>

            <ul className="text-xs text-slate-300 space-y-1.5 list-disc list-inside">
              <li><span className="font-mono text-emerald-300">Data</span>: Full active dataset</li>
              <li><span className="font-mono text-emerald-300">Statistics</span>: Summary statistics</li>
              <li><span className="font-mono text-emerald-300">Missing Values</span>: Column missingness</li>
              <li><span className="font-mono text-emerald-300">Column Info</span>: Metadata & types</li>
              <li><span className="font-mono text-emerald-300">Query History</span>: All session AI queries</li>
            </ul>
          </div>

          <button
            onClick={handleDownloadExcel}
            disabled={downloadingExcel}
            className="w-full py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-semibold flex items-center justify-center gap-2 text-xs shadow-md shadow-emerald-600/20 transition-all"
          >
            <Download className="h-4 w-4" />
            {downloadingExcel ? 'Building Excel...' : 'Download Excel (.xlsx)'}
          </button>
        </div>

        {/* Cleaned CSV Card */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 flex flex-col justify-between space-y-4">
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <FileText className="h-5 w-5 text-amber-400" />
              <h3 className="text-sm font-semibold text-slate-200">Cleaned Data Export (CSV)</h3>
            </div>
            <p className="text-xs text-slate-400">
              Download the current in-memory version of your dataset with all active transformations, filters, and imputations applied.
            </p>
            <div className="p-3 rounded-lg bg-slate-800/40 border border-slate-700/60 text-xs text-slate-300 space-y-1 font-mono">
              <p>Dataset: {activeDataset?.name}</p>
              <p>Version: v{activeDataset?.version}</p>
              <p>Rows: {activeDataset?.rows.toLocaleString()}</p>
            </div>
          </div>

          <a
            href={datasetApi.getDownloadCsvUrl()}
            download={`${activeDataset?.name || 'dataset'}_cleaned.csv`}
            className="w-full py-2.5 rounded-xl bg-amber-600 hover:bg-amber-500 text-white font-semibold flex items-center justify-center gap-2 text-xs shadow-md shadow-amber-600/20 transition-all"
          >
            <Download className="h-4 w-4" />
            Download Clean CSV
          </a>
        </div>
      </div>

      {/* Deep Column Schema Profile Summary */}
      {profile && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
          <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
            <Table className="h-4 w-4 text-blue-400" />
            Dataset Schema Profile ({profile.numeric_columns.length + profile.categorical_columns.length} columns)
          </h3>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-slate-800/60 text-[11px] uppercase tracking-wider text-slate-400 font-semibold">
                <tr>
                  <th className="px-4 py-2.5">Column</th>
                  <th className="px-4 py-2.5">Type</th>
                  <th className="px-4 py-2.5">Missing</th>
                  <th className="px-4 py-2.5">Mean</th>
                  <th className="px-4 py-2.5">Std</th>
                  <th className="px-4 py-2.5">Min / Max</th>
                  <th className="px-4 py-2.5">Median</th>
                  <th className="px-4 py-2.5">Skewness</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {profile.numeric_columns.map((c) => (
                  <tr key={c.column} className="hover:bg-slate-800/30 font-mono">
                    <td className="px-4 py-2 text-white font-sans font-medium">{c.column}</td>
                    <td className="px-4 py-2 text-blue-400">numeric</td>
                    <td className="px-4 py-2">{c.missing} ({c.missing_pct}%)</td>
                    <td className="px-4 py-2">{c.mean ?? '—'}</td>
                    <td className="px-4 py-2">{c.std ?? '—'}</td>
                    <td className="px-4 py-2">[{c.min}, {c.max}]</td>
                    <td className="px-4 py-2">{c['50%'] ?? '—'}</td>
                    <td className="px-4 py-2">{c.skew ?? '—'}</td>
                  </tr>
                ))}
                {profile.categorical_columns.map((c) => (
                  <tr key={c.column} className="hover:bg-slate-800/30">
                    <td className="px-4 py-2 text-white font-medium">{c.column}</td>
                    <td className="px-4 py-2 text-emerald-400 font-mono">categorical</td>
                    <td className="px-4 py-2 font-mono">{c.missing} ({c.missing_pct}%)</td>
                    <td className="px-4 py-2 font-mono text-slate-400" colSpan={3}>
                      Top: "{c.most_common}" ({c.most_common_count}x)
                    </td>
                    <td className="px-4 py-2 font-mono" colSpan={2}>
                      {c.unique_values} unique
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
