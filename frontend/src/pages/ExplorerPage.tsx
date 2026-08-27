import React, { useState, useEffect } from 'react';
import {
  Compass,
  Table,
  GitCommit,
  BarChart,
  Layers,
  Search,
  Filter,
} from 'lucide-react';
import { useDataset } from '../context/DatasetContext';
import { explorerApi } from '../api/client';
import {
  ColumnProfileResponse,
  CorrelationsResponse,
  ExplorerBrowseResponse,
} from '../types';
import { PlotlyChart } from '../components/PlotlyChart';
import { DataTable } from '../components/DataTable';

export const ExplorerPage: React.FC = () => {
  const { preview, hasDataset } = useDataset();
  const [activeTab, setActiveTab] = useState<'browse' | 'correlations' | 'distributions' | 'profiler'>('browse');

  // Browse state
  const [browseData, setBrowseData] = useState<ExplorerBrowseResponse | null>(null);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
  const [loadingBrowse, setLoadingBrowse] = useState(false);

  // Correlations state
  const [corrMethod, setCorrMethod] = useState<'pearson' | 'spearman' | 'kendall'>('pearson');
  const [corrData, setCorrData] = useState<CorrelationsResponse | null>(null);
  const [loadingCorr, setLoadingCorr] = useState(false);

  // Distributions state
  const [distCol, setDistCol] = useState<string>('');
  const [distChartType, setDistChartType] = useState<string>('Histogram');
  const [distGroupBy, setDistGroupBy] = useState<string>('None');
  const [distData, setDistData] = useState<any>(null);
  const [loadingDist, setLoadingDist] = useState(false);

  // Profiler state
  const [profileCol, setProfileCol] = useState<string>('');
  const [profileData, setProfileData] = useState<ColumnProfileResponse | null>(null);
  const [loadingProfile, setLoadingProfile] = useState(false);

  useEffect(() => {
    if (preview?.columns) {
      if (!distCol && preview.columns.length > 0) setDistCol(preview.columns[0]);
      if (!profileCol && preview.columns.length > 0) setProfileCol(preview.columns[0]);
    }
  }, [preview]);

  // Fetch Browse
  useEffect(() => {
    if (!hasDataset) return;
    setLoadingBrowse(true);
    explorerApi
      .browse({
        page,
        page_size: pageSize,
        columns: selectedColumns.length > 0 ? selectedColumns : undefined,
      })
      .then((res) => setBrowseData(res))
      .catch((err) => console.error(err))
      .finally(() => setLoadingBrowse(false));
  }, [hasDataset, page, pageSize, selectedColumns]);

  // Fetch Correlations
  useEffect(() => {
    if (activeTab !== 'correlations' || !preview?.numeric_cols || preview.numeric_cols.length < 2) return;
    setLoadingCorr(true);
    explorerApi
      .getCorrelations(preview.numeric_cols, corrMethod)
      .then((res) => setCorrData(res))
      .catch((err) => console.error(err))
      .finally(() => setLoadingCorr(false));
  }, [activeTab, corrMethod, preview]);

  // Fetch Distribution
  useEffect(() => {
    if (activeTab !== 'distributions' || !distCol) return;
    setLoadingDist(true);
    explorerApi
      .getDistribution({
        column: distCol,
        chart_type: distChartType,
        group_by: distGroupBy !== 'None' ? distGroupBy : undefined,
      })
      .then((res) => setDistData(res))
      .catch((err) => console.error(err))
      .finally(() => setLoadingDist(false));
  }, [activeTab, distCol, distChartType, distGroupBy]);

  // Fetch Profiler
  useEffect(() => {
    if (activeTab !== 'profiler' || !profileCol) return;
    setLoadingProfile(true);
    explorerApi
      .getColumnProfile(profileCol)
      .then((res) => setProfileData(res))
      .catch((err) => console.error(err))
      .finally(() => setLoadingProfile(false));
  }, [activeTab, profileCol]);

  if (!hasDataset) {
    return (
      <div className="p-8 text-center text-slate-500 border border-slate-800 rounded-xl bg-slate-900/30">
        Please load or select a dataset first to explore data.
      </div>
    );
  }

  const numericCols = preview?.numeric_cols || [];
  const catCols = preview?.categorical_cols || [];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight">Data Explorer</h2>
        <p className="text-xs text-slate-400 mt-1">
          Explore distributions, correlations, column profiles, and interactive data grids
        </p>
      </div>

      {/* Tabs */}
      <div className="flex items-center gap-2 border-b border-slate-800 pb-2">
        <button
          onClick={() => setActiveTab('browse')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
            activeTab === 'browse'
              ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
          }`}
        >
          <Table className="h-4 w-4" />
          Browse Table
        </button>

        <button
          onClick={() => setActiveTab('correlations')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
            activeTab === 'correlations'
              ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
          }`}
        >
          <GitCommit className="h-4 w-4" />
          Correlations ({numericCols.length})
        </button>

        <button
          onClick={() => setActiveTab('distributions')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
            activeTab === 'distributions'
              ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
          }`}
        >
          <BarChart className="h-4 w-4" />
          Distributions
        </button>

        <button
          onClick={() => setActiveTab('profiler')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
            activeTab === 'profiler'
              ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
          }`}
        >
          <Layers className="h-4 w-4" />
          Column Profiler
        </button>
      </div>

      {/* Tab 1: Browse Table */}
      {activeTab === 'browse' && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <h3 className="text-sm font-semibold text-slate-200">Interactive Dataset Browser</h3>
              <p className="text-xs text-slate-400">
                Displaying {browseData?.total_rows.toLocaleString()} total rows
              </p>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-xs text-slate-400">Page size:</span>
              <select
                value={pageSize}
                onChange={(e) => setPageSize(parseInt(e.target.value))}
                className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-slate-200"
              >
                <option value={10}>10</option>
                <option value={25}>25</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
              </select>
            </div>
          </div>

          {loadingBrowse ? (
            <div className="h-40 flex items-center justify-center text-xs text-slate-500">Loading dataset...</div>
          ) : (
            <DataTable
              data={browseData?.data || []}
              columns={browseData?.columns}
              pageSize={pageSize}
              showSearch={true}
            />
          )}
        </div>
      )}

      {/* Tab 2: Correlations Matrix */}
      {activeTab === 'correlations' && (
        <div className="space-y-6">
          <div className="flex items-center justify-between gap-4 p-4 rounded-xl border border-slate-800 bg-slate-900/40">
            <div>
              <h3 className="text-sm font-semibold text-slate-200">Pairwise Feature Correlation</h3>
              <p className="text-xs text-slate-400">Calculates linear or rank relationships between numeric fields</p>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400">Method:</span>
              <select
                value={corrMethod}
                onChange={(e) => setCorrMethod(e.target.value as any)}
                className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-200 font-medium"
              >
                <option value="pearson">Pearson (Standard)</option>
                <option value="spearman">Spearman (Rank)</option>
                <option value="kendall">Kendall (Tau)</option>
              </select>
            </div>
          </div>

          {numericCols.length < 2 ? (
            <div className="p-8 text-center text-slate-500 border border-slate-800 rounded-xl">
              At least 2 numeric columns are required to generate correlation heatmaps.
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 rounded-xl border border-slate-800 bg-slate-900/40 p-5">
                <PlotlyChart spec={corrData?.figure_spec} height={450} />
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-3">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">Strongest Correlations</h4>
                <div className="space-y-2 overflow-y-auto max-h-[400px]">
                  {corrData?.top_pairs.map((p, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between text-xs p-2.5 rounded-lg bg-slate-800/40 border border-slate-700/60"
                    >
                      <div className="truncate pr-2">
                        <span className="font-mono text-slate-200">{p.col_a}</span>
                        <span className="text-slate-500 mx-1">↔</span>
                        <span className="font-mono text-slate-200">{p.col_b}</span>
                      </div>
                      <span
                        className={`font-mono font-bold ${
                          p.correlation > 0.6
                            ? 'text-emerald-400'
                            : p.correlation < -0.6
                            ? 'text-rose-400'
                            : 'text-blue-400'
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
      {activeTab === 'distributions' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 p-4 rounded-xl border border-slate-800 bg-slate-900/40 text-xs">
            <div>
              <label className="block text-slate-300 font-medium mb-1">Target Column</label>
              <select
                value={distCol}
                onChange={(e) => setDistCol(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-200"
              >
                {preview?.columns.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-slate-300 font-medium mb-1">Chart Type</label>
              <select
                value={distChartType}
                onChange={(e) => setDistChartType(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-200"
              >
                <option value="Histogram">Histogram</option>
                <option value="Box Plot">Box Plot</option>
                <option value="Violin">Violin Plot</option>
              </select>
            </div>

            <div>
              <label className="block text-slate-300 font-medium mb-1">Group By Category (Optional)</label>
              <select
                value={distGroupBy}
                onChange={(e) => setDistGroupBy(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-200"
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

          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
            {loadingDist ? (
              <div className="h-72 flex items-center justify-center text-xs text-slate-500">Generating distribution...</div>
            ) : (
              <PlotlyChart spec={distData?.figure_spec} height={420} />
            )}
          </div>
        </div>
      )}

      {/* Tab 4: Column Profiler */}
      {activeTab === 'profiler' && (
        <div className="space-y-6">
          <div className="flex items-center gap-3 p-4 rounded-xl border border-slate-800 bg-slate-900/40">
            <span className="text-xs text-slate-300 font-medium">Select Column to Profile:</span>
            <select
              value={profileCol}
              onChange={(e) => setProfileCol(e.target.value)}
              className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-200 max-w-xs font-mono"
            >
              {preview?.columns.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>

          {loadingProfile ? (
            <div className="h-40 flex items-center justify-center text-xs text-slate-500">Calculating profile statistics...</div>
          ) : profileData ? (
            <div className="space-y-6">
              {/* Summary Stats Cards */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
                <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40">
                  <span className="text-slate-400">Data Type</span>
                  <p className="text-lg font-bold text-blue-400 font-mono mt-1">{profileData.dtype}</p>
                </div>
                <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40">
                  <span className="text-slate-400">Unique Values</span>
                  <p className="text-lg font-bold text-white font-mono mt-1">
                    {profileData.unique_count.toLocaleString()}
                  </p>
                </div>
                <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40">
                  <span className="text-slate-400">Missing Values</span>
                  <p className="text-lg font-bold text-amber-400 font-mono mt-1">
                    {profileData.missing_count} ({profileData.missing_pct}%)
                  </p>
                </div>
                <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40">
                  <span className="text-slate-400">Category Type</span>
                  <p className="text-lg font-bold text-emerald-400 font-mono mt-1">
                    {profileData.is_numeric ? 'Numeric Metric' : 'Categorical Dimension'}
                  </p>
                </div>
              </div>

              {/* Numeric Deep Stats */}
              {profileData.numeric_stats && (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
                  <div className="p-3 rounded-lg bg-slate-800/40 border border-slate-700">
                    <span className="text-slate-400">Mean ± Std</span>
                    <p className="text-sm font-bold text-white font-mono mt-1">
                      {profileData.numeric_stats.mean} ± {profileData.numeric_stats.std}
                    </p>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-800/40 border border-slate-700">
                    <span className="text-slate-400">Min / Max Range</span>
                    <p className="text-sm font-bold text-white font-mono mt-1">
                      [{profileData.numeric_stats.min}, {profileData.numeric_stats.max}]
                    </p>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-800/40 border border-slate-700">
                    <span className="text-slate-400">Median</span>
                    <p className="text-sm font-bold text-white font-mono mt-1">{profileData.numeric_stats.median}</p>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-800/40 border border-slate-700">
                    <span className="text-slate-400">Skewness</span>
                    <p className="text-sm font-bold text-white font-mono mt-1">{profileData.numeric_stats.skew}</p>
                  </div>
                </div>
              )}

              {/* Profiler Chart */}
              <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
                <PlotlyChart spec={profileData.figure_spec} height={320} />
              </div>
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
};
