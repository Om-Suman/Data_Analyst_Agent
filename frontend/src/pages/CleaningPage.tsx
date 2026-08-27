import React, { useState, useEffect } from 'react';
import {
  Sparkles,
  RotateCcw,
  CheckCircle2,
  AlertTriangle,
  History,
  Play,
  Check,
  Wrench,
  Layers,
  ArrowRight,
  Trash2,
  Edit3,
  Binary,
  Type,
  Calculator,
  PlusCircle,
} from 'lucide-react';
import { useDataset } from '../context/DatasetContext';
import { cleaningApi } from '../api/client';
import { CleaningReportResponse, QualityScoreResponse, VersionItem } from '../types';
import { DataTable } from '../components/DataTable';
import { useToast } from '../components/Toast';

export const CleaningPage: React.FC = () => {
  const { activeDataset, hasDataset, preview, refreshDatasets, refreshPreview } = useDataset();
  const { success, error: toastError } = useToast();

  const [activeTab, setActiveTab] = useState<'pipeline' | 'columns' | 'history'>('pipeline');
  const [quality, setQuality] = useState<QualityScoreResponse | null>(null);
  const [versions, setVersions] = useState<VersionItem[]>([]);
  const [loadingQuality, setLoadingQuality] = useState(false);
  const [previewReport, setPreviewReport] = useState<CleaningReportResponse | null>(null);
  const [applying, setApplying] = useState(false);
  const [previewing, setPreviewing] = useState(false);

  // Automated Pipeline Form state
  const [missingStrategy, setMissingStrategy] = useState('mean');
  const [removeDuplicates, setRemoveDuplicates] = useState(true);
  const [fixDtypes, setFixDtypes] = useState(true);
  const [normalizeNames, setNormalizeNames] = useState(true);
  const [outlierMethod, setOutlierMethod] = useState('none');
  const [outlierThreshold, setOutlierThreshold] = useState(3.0);
  const [iqrFactor, setIqrFactor] = useState(1.5);

  // Column Engineering state
  const [selectedColumn, setSelectedColumn] = useState<string>('');
  const [colOperation, setColOperation] = useState<'rename' | 'cast' | 'string_case' | 'math_expr' | 'create_column' | 'drop'>('rename');
  const [newName, setNewName] = useState('');
  const [targetType, setTargetType] = useState('float');
  const [caseMode, setCaseMode] = useState('upper');
  const [expression, setExpression] = useState('');
  const [transforming, setTransforming] = useState(false);

  const columns = preview?.columns || [];

  useEffect(() => {
    if (columns.length > 0 && !selectedColumn) {
      setSelectedColumn(columns[0]);
    }
  }, [columns, selectedColumn]);

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
      success('Dry-Run Computed', 'Previewing cleaning transformations below.');
    } catch (err: any) {
      toastError('Preview Failed', err.response?.data?.detail || 'Failed to preview cleaning.');
    } finally {
      setPreviewing(false);
    }
  };

  const handleApply = async () => {
    setApplying(true);
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
      success('Cleaning Applied', `Dataset updated to v${(activeDataset?.version || 1) + 1}`);
      await refreshDatasets();
      await refreshPreview();
      await fetchQualityAndVersions();
    } catch (err: any) {
      toastError('Apply Failed', err.response?.data?.detail || 'Failed to apply cleaning.');
    } finally {
      setApplying(false);
    }
  };

  const handleTransformColumn = async () => {
    if (!selectedColumn && colOperation !== 'create_column') return;
    setTransforming(true);
    try {
      const res = await cleaningApi.transformColumn({
        column: selectedColumn,
        operation: colOperation,
        new_name: newName,
        target_type: targetType,
        case_mode: caseMode,
        expression: expression,
      });

      if (res.success) {
        success('Transformation Applied', res.message);
        setNewName('');
        setExpression('');
        await refreshDatasets();
        await refreshPreview();
        await fetchQualityAndVersions();
      }
    } catch (err: any) {
      toastError('Transform Failed', err.response?.data?.detail || 'Failed to execute column operation.');
    } finally {
      setTransforming(false);
    }
  };

  const handleRollback = async (versionNumber: number) => {
    if (!window.confirm(`Roll back dataset to version ${versionNumber}?`)) return;
    try {
      await cleaningApi.rollbackVersion(versionNumber);
      success('Dataset Restored', `Successfully rolled back to version ${versionNumber}!`);
      await refreshDatasets();
      await refreshPreview();
      await fetchQualityAndVersions();
    } catch (err: any) {
      toastError('Rollback Failed', err.response?.data?.detail || 'Rollback failed.');
    }
  };

  if (!hasDataset) {
    return (
      <div className="p-8 text-center text-slate-500 border border-slate-200 dark:border-slate-800 rounded-2xl bg-slate-100/50 dark:bg-slate-900/30">
        Please load or select a dataset first to use the Data Cleaning tools.
      </div>
    );
  }

  const gradeColor =
    quality?.quality_grade === 'A'
      ? 'border-emerald-500/30 bg-emerald-50 dark:bg-emerald-950/30 text-emerald-700 dark:text-emerald-400'
      : quality?.quality_grade === 'B'
      ? 'border-blue-500/30 bg-blue-50 dark:bg-blue-950/30 text-blue-700 dark:text-blue-400'
      : quality?.quality_grade === 'C'
      ? 'border-amber-500/30 bg-amber-50 dark:bg-amber-950/30 text-amber-700 dark:text-amber-400'
      : 'border-rose-500/30 bg-rose-50 dark:bg-rose-950/30 text-rose-700 dark:text-rose-400';

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white tracking-tight flex items-center gap-2.5">
          <Sparkles className="h-6 w-6 text-blue-500" />
          Data Quality & Cleaning Studio
        </h2>
        <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-1">
          Comprehensive data hygiene, automated imputation, custom column engineering, and lossless rollback snapshots
        </p>
      </div>

      {/* Quality Overview KPIs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className={`rounded-2xl border p-5 flex items-center justify-between shadow-sm ${gradeColor}`}>
          <div>
            <p className="text-xs uppercase font-bold tracking-wider text-slate-500 dark:text-slate-400">Quality Score</p>
            <p className="text-3xl font-extrabold mt-1 text-slate-900 dark:text-white font-mono">
              {quality?.quality_score ?? '—'}<span className="text-lg opacity-60">/100</span>
            </p>
          </div>
          <div className="text-3xl font-black px-3.5 py-1.5 rounded-xl border bg-white/80 dark:bg-slate-900/80 font-mono shadow-sm">
            {quality?.quality_grade ?? '—'}
          </div>
        </div>

        <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 p-5 shadow-sm">
          <p className="text-xs uppercase font-bold tracking-wider text-slate-400">Missing Values</p>
          <p className="text-2xl sm:text-3xl font-bold mt-1 text-slate-900 dark:text-white font-mono">
            {quality?.missing_total?.toLocaleString() ?? '0'}
          </p>
          <p className="text-xs text-slate-500 mt-0.5 font-medium">{quality?.missing_pct ?? 0}% of total dataset cells</p>
        </div>

        <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 p-5 shadow-sm">
          <p className="text-xs uppercase font-bold tracking-wider text-slate-400">Duplicate Rows</p>
          <p className="text-2xl sm:text-3xl font-bold mt-1 text-slate-900 dark:text-white font-mono">
            {quality?.duplicate_rows?.toLocaleString() ?? '0'}
          </p>
          <p className="text-xs text-slate-500 mt-0.5 font-medium">{quality?.duplicate_pct ?? 0}% duplicate rate</p>
        </div>

        <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 p-5 shadow-sm">
          <p className="text-xs uppercase font-bold tracking-wider text-slate-400">Active Snapshot</p>
          <p className="text-2xl sm:text-3xl font-bold mt-1 text-blue-600 dark:text-blue-400 font-mono">
            v{activeDataset?.version}
          </p>
          <p className="text-xs text-slate-500 mt-0.5 font-medium">{versions.length} version snapshots saved</p>
        </div>
      </div>

      {/* Tabs Navigation */}
      <div className="flex items-center gap-2 border-b border-slate-200 dark:border-slate-800 pb-2">
        <button
          onClick={() => setActiveTab('pipeline')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs sm:text-sm font-bold transition-all ${
            activeTab === 'pipeline'
              ? 'bg-blue-600 text-white shadow-md shadow-blue-600/20'
              : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-800'
          }`}
        >
          <Sparkles className="h-4 w-4" />
          Automated Pipeline
        </button>

        <button
          onClick={() => setActiveTab('columns')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs sm:text-sm font-bold transition-all ${
            activeTab === 'columns'
              ? 'bg-blue-600 text-white shadow-md shadow-blue-600/20'
              : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-800'
          }`}
        >
          <Wrench className="h-4 w-4" />
          Column Engineering Studio
        </button>

        <button
          onClick={() => setActiveTab('history')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs sm:text-sm font-bold transition-all ${
            activeTab === 'history'
              ? 'bg-blue-600 text-white shadow-md shadow-blue-600/20'
              : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-800'
          }`}
        >
          <History className="h-4 w-4" />
          Version History ({versions.length})
        </button>
      </div>

      {/* Tab 1: Automated Pipeline */}
      {activeTab === 'pipeline' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 p-5 space-y-4 shadow-sm">
            <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-blue-500" />
              Automated Cleaning Rules
            </h3>

            <div className="space-y-4 text-xs sm:text-sm">
              <div>
                <label className="block text-slate-700 dark:text-slate-300 font-semibold mb-1">Missing Value Imputation</label>
                <select
                  value={missingStrategy}
                  onChange={(e) => setMissingStrategy(e.target.value)}
                  className="w-full bg-slate-100 dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 text-slate-900 dark:text-slate-200 focus:outline-none focus:border-blue-500"
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
                <label className="block text-slate-700 dark:text-slate-300 font-semibold mb-1">Outlier Treatment</label>
                <select
                  value={outlierMethod}
                  onChange={(e) => setOutlierMethod(e.target.value)}
                  className="w-full bg-slate-100 dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 text-slate-900 dark:text-slate-200 focus:outline-none focus:border-blue-500"
                >
                  <option value="none">None (keep outliers)</option>
                  <option value="zscore">Z-Score Filtering</option>
                  <option value="iqr">IQR (Interquartile Range) Filtering</option>
                </select>
              </div>

              <div className="space-y-2.5 pt-2 border-t border-slate-200 dark:border-slate-800">
                <label className="flex items-center gap-2 text-slate-700 dark:text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={removeDuplicates}
                    onChange={(e) => setRemoveDuplicates(e.target.checked)}
                    className="rounded text-blue-600 focus:ring-blue-500"
                  />
                  <span>Remove exact duplicate rows</span>
                </label>

                <label className="flex items-center gap-2 text-slate-700 dark:text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={fixDtypes}
                    onChange={(e) => setFixDtypes(e.target.checked)}
                    className="rounded text-blue-600 focus:ring-blue-500"
                  />
                  <span>Infer & fix mismatched datatypes</span>
                </label>

                <label className="flex items-center gap-2 text-slate-700 dark:text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={normalizeNames}
                    onChange={(e) => setNormalizeNames(e.target.checked)}
                    className="rounded text-blue-600 focus:ring-blue-500"
                  />
                  <span>Normalize column headers (snake_case)</span>
                </label>
              </div>

              <div className="flex gap-2.5 pt-3">
                <button
                  onClick={handlePreview}
                  disabled={previewing || applying}
                  className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2.5 rounded-xl border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-800 dark:text-slate-200 font-semibold transition-colors"
                >
                  <Play className="h-4 w-4" />
                  {previewing ? 'Previewing...' : 'Dry Run Preview'}
                </button>
                <button
                  onClick={handleApply}
                  disabled={applying || previewing}
                  className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-bold shadow-md shadow-blue-600/20 transition-all"
                >
                  <Check className="h-4 w-4" />
                  {applying ? 'Applying...' : 'Apply & Save'}
                </button>
              </div>
            </div>
          </div>

          <div className="lg:col-span-2 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 p-5 shadow-sm space-y-4">
            <h3 className="text-sm font-bold text-slate-900 dark:text-white">
              {previewReport ? 'Cleaning Report & Transformed Sample' : 'Current Clean Preview'}
            </h3>

            {previewReport ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                  <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-800">
                    <p className="text-slate-400">Rows Removed</p>
                    <p className="text-lg font-bold text-slate-900 dark:text-white font-mono mt-0.5">
                      {(previewReport.rows_before - previewReport.rows_after).toLocaleString()}
                    </p>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-800">
                    <p className="text-slate-400">Missing Imputed</p>
                    <p className="text-lg font-bold text-slate-900 dark:text-white font-mono mt-0.5">
                      {Object.values(previewReport.missing_filled || {}).reduce((a, b) => a + (Number(b) || 0), 0)}
                    </p>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-800">
                    <p className="text-slate-400">Duplicates Dropped</p>
                    <p className="text-lg font-bold text-slate-900 dark:text-white font-mono mt-0.5">
                      {previewReport.duplicates_removed.toLocaleString()}
                    </p>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-800">
                    <p className="text-slate-400">Score Impact</p>
                    <p className="text-lg font-bold text-emerald-600 dark:text-emerald-400 font-mono mt-0.5">
                      {previewReport.quality_score_before} ➔ {previewReport.quality_score_after}
                    </p>
                  </div>
                </div>

                <DataTable data={previewReport.preview_data || []} pageSize={6} />
              </div>
            ) : (
              <DataTable data={preview?.data || []} pageSize={6} />
            )}
          </div>
        </div>
      )}

      {/* Tab 2: Column Engineering Studio */}
      {activeTab === 'columns' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 p-5 space-y-4 shadow-sm">
            <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <Wrench className="h-4 w-4 text-emerald-500" />
              Column Transformation Setup
            </h3>

            <div className="space-y-4 text-xs sm:text-sm">
              {/* Target Column Selector */}
              {colOperation !== 'create_column' && (
                <div>
                  <label className="block text-slate-700 dark:text-slate-300 font-semibold mb-1">Select Target Column</label>
                  <select
                    value={selectedColumn}
                    onChange={(e) => setSelectedColumn(e.target.value)}
                    className="w-full bg-slate-100 dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 font-mono text-slate-900 dark:text-slate-200 focus:outline-none focus:border-emerald-500"
                  >
                    {columns.map((col) => (
                      <option key={col} value={col}>
                        {col}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {/* Operation Selector */}
              <div>
                <label className="block text-slate-700 dark:text-slate-300 font-semibold mb-1">Transformation Operation</label>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    { id: 'rename', label: 'Rename', icon: Edit3 },
                    { id: 'cast', label: 'Cast Type', icon: Binary },
                    { id: 'string_case', label: 'String Case', icon: Type },
                    { id: 'math_expr', label: 'Math Expr', icon: Calculator },
                    { id: 'create_column', label: 'New Column', icon: PlusCircle },
                    { id: 'drop', label: 'Drop Column', icon: Trash2 },
                  ].map((op) => (
                    <button
                      key={op.id}
                      type="button"
                      onClick={() => setColOperation(op.id as any)}
                      className={`flex items-center gap-1.5 p-2.5 rounded-xl border text-xs font-semibold transition-all ${
                        colOperation === op.id
                          ? 'border-emerald-500 bg-emerald-50 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300 shadow-sm'
                          : 'border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/40 text-slate-700 dark:text-slate-300 hover:border-slate-300 dark:hover:border-slate-700'
                      }`}
                    >
                      <op.icon className="h-3.5 w-3.5" />
                      <span>{op.label}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Dynamic Inputs based on operation */}
              {colOperation === 'rename' && (
                <div>
                  <label className="block text-slate-700 dark:text-slate-300 font-semibold mb-1">New Column Name</label>
                  <input
                    type="text"
                    value={newName}
                    onChange={(e) => setNewName(e.target.value)}
                    placeholder="e.g. net_revenue"
                    className="w-full bg-slate-100 dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 font-mono text-slate-900 dark:text-slate-200 focus:outline-none focus:border-emerald-500"
                  />
                </div>
              )}

              {colOperation === 'cast' && (
                <div>
                  <label className="block text-slate-700 dark:text-slate-300 font-semibold mb-1">Target Data Type</label>
                  <select
                    value={targetType}
                    onChange={(e) => setTargetType(e.target.value)}
                    className="w-full bg-slate-100 dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 text-slate-900 dark:text-slate-200 focus:outline-none focus:border-emerald-500"
                  >
                    <option value="int">Integer (int64)</option>
                    <option value="float">Floating Point (float64)</option>
                    <option value="str">String / Text</option>
                    <option value="datetime">Datetime</option>
                    <option value="bool">Boolean</option>
                  </select>
                </div>
              )}

              {colOperation === 'string_case' && (
                <div>
                  <label className="block text-slate-700 dark:text-slate-300 font-semibold mb-1">Case Formatting Mode</label>
                  <select
                    value={caseMode}
                    onChange={(e) => setCaseMode(e.target.value)}
                    className="w-full bg-slate-100 dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 text-slate-900 dark:text-slate-200 focus:outline-none focus:border-emerald-500"
                  >
                    <option value="upper">UPPERCASE</option>
                    <option value="lower">lowercase</option>
                    <option value="title">Title Case</option>
                    <option value="strip">Trim Whitespace (strip)</option>
                  </select>
                </div>
              )}

              {colOperation === 'math_expr' && (
                <div>
                  <label className="block text-slate-700 dark:text-slate-300 font-semibold mb-1">Math Expression</label>
                  <input
                    type="text"
                    value={expression}
                    onChange={(e) => setExpression(e.target.value)}
                    placeholder={`e.g. ${selectedColumn} * 1.18 or np.log(${selectedColumn})`}
                    className="w-full bg-slate-100 dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 font-mono text-slate-900 dark:text-slate-200 focus:outline-none focus:border-emerald-500"
                  />
                  <p className="text-[11px] text-slate-400 mt-1">Available namespace: column names, <code>np</code>, <code>pd</code></p>
                </div>
              )}

              {colOperation === 'create_column' && (
                <div className="space-y-3">
                  <div>
                    <label className="block text-slate-700 dark:text-slate-300 font-semibold mb-1">New Column Name</label>
                    <input
                      type="text"
                      value={newName}
                      onChange={(e) => setNewName(e.target.value)}
                      placeholder="e.g. profit_margin"
                      className="w-full bg-slate-100 dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 font-mono text-slate-900 dark:text-slate-200 focus:outline-none focus:border-emerald-500"
                    />
                  </div>
                  <div>
                    <label className="block text-slate-700 dark:text-slate-300 font-semibold mb-1">Computation Expression</label>
                    <input
                      type="text"
                      value={expression}
                      onChange={(e) => setExpression(e.target.value)}
                      placeholder="e.g. profit / sales * 100"
                      className="w-full bg-slate-100 dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 font-mono text-slate-900 dark:text-slate-200 focus:outline-none focus:border-emerald-500"
                    />
                  </div>
                </div>
              )}

              <button
                onClick={handleTransformColumn}
                disabled={transforming}
                className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white font-bold text-xs sm:text-sm shadow-md shadow-emerald-600/20 transition-all mt-2"
              >
                <CheckCircle2 className="h-4 w-4" />
                {transforming ? 'Transforming...' : 'Execute Column Operation'}
              </button>
            </div>
          </div>

          <div className="lg:col-span-2 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 p-5 shadow-sm space-y-4">
            <h3 className="text-sm font-bold text-slate-900 dark:text-white">Active Table Schema & Preview</h3>
            <DataTable data={preview?.data || []} pageSize={8} />
          </div>
        </div>
      )}

      {/* Tab 3: Version History & Rollback Timeline */}
      {activeTab === 'history' && (
        <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 p-5 space-y-4 shadow-sm">
          <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <History className="h-4 w-4 text-blue-500" />
            Dataset Version History & Snapshot Rollback
          </h3>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs sm:text-sm text-slate-700 dark:text-slate-300 border-collapse">
              <thead className="bg-slate-50 dark:bg-slate-950/70 border-b border-slate-200 dark:border-slate-800 text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                <tr>
                  <th className="px-4 py-3">Version</th>
                  <th className="px-4 py-3">Timestamp</th>
                  <th className="px-4 py-3">Rows</th>
                  <th className="px-4 py-3">Cols</th>
                  <th className="px-4 py-3">Transformation Note</th>
                  <th className="px-4 py-3 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-800/80">
                {versions.map((v) => {
                  const isCurrent = v.version === activeDataset?.version;
                  return (
                    <tr key={v.version} className="hover:bg-slate-50 dark:hover:bg-slate-800/40 transition-colors">
                      <td className="px-4 py-3.5 font-mono font-bold text-slate-900 dark:text-white">v{v.version}</td>
                      <td className="px-4 py-3.5 text-slate-500 dark:text-slate-400">{new Date(v.timestamp).toLocaleString()}</td>
                      <td className="px-4 py-3.5 font-mono">{v.rows.toLocaleString()}</td>
                      <td className="px-4 py-3.5 font-mono">{v.cols}</td>
                      <td className="px-4 py-3.5 font-medium">{v.description || 'Initial Load'}</td>
                      <td className="px-4 py-3.5 text-right">
                        {isCurrent ? (
                          <span className="px-2.5 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-800 dark:bg-emerald-950/80 dark:text-emerald-300">
                            Active Version
                          </span>
                        ) : (
                          <button
                            onClick={() => handleRollback(v.version)}
                            className="inline-flex items-center gap-1 px-3 py-1 rounded-lg bg-blue-50 dark:bg-blue-950/60 hover:bg-blue-100 dark:hover:bg-blue-900/60 text-blue-600 dark:text-blue-400 text-xs font-bold transition-colors"
                          >
                            <RotateCcw className="h-3.5 w-3.5" />
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
      )}
    </div>
  );
};
