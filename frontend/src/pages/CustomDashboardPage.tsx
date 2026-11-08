import React, { useState } from 'react';
import { PinnedChartItem } from '../types';
import { NavLink } from 'react-router-dom';
import {
  Layers,
  Trash2,
  BarChart3,
  Calendar,
  Sparkles,
  ExternalLink,
  Plus,
} from 'lucide-react';
import { useDataset } from '../context/DatasetContext';
import { PlotlyChart } from '../components/PlotlyChart';
import { useToast } from '../components/Toast';
import { DatasetEmptyState } from '../components/DatasetEmptyState';

export const CustomDashboardPage: React.FC = () => {
  const { activeDataset, pinnedCharts, unpinChart, pinChart } = useDataset();
  const { success, error } = useToast();

  const [removed, setRemoved] = useState<PinnedChartItem | null>(null);
  const [pending, setPending] = useState(false);
  const handleUnpin = async (item: PinnedChartItem) => {
    setPending(true);
    try {
      await unpinChart(item.id);
      setRemoved(item);
    } catch (err: any) {
      error('Error', err.message || 'Failed to remove pinned chart.');
    } finally {
      setPending(false);
    }
  };

  const restoreChart = async () => {
    if (!removed || pending) return;
    setPending(true);
    try {
      const { title, chart_type, figure_spec, source_page, notes, dataset_name, dataset_version } = removed;
      await pinChart({ title, chart_type, figure_spec, source_page, notes, dataset_name, dataset_version });
      setRemoved(null);
      success('Chart restored');
    } catch {
      error('Could not restore chart', 'Try Undo again.');
    } finally {
      setPending(false);
    }
  };

  if (!pinnedCharts.length && !removed && (!activeDataset || activeDataset.is_text)) {
    return (
      <DatasetEmptyState
        title="BI Canvas"
        description="Create a chart from your dataset and pin it here to keep important views together."
      />
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white tracking-tight flex items-center gap-2.5">
            <Layers className="h-6 w-6 text-blue-500" />
            Custom BI Multi-Chart Canvas
          </h2>
          <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-1">
            Curate and monitor key interactive visualizations pinned from across your analysis workflows
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <NavLink
            to="/visualizations"
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs sm:text-sm font-bold shadow-md shadow-blue-600/20 transition-all"
          >
            <Plus className="h-4 w-4" />
            Explore Visualizations
          </NavLink>
        </div>
      </div>

      {removed && <div className="flex flex-wrap items-center gap-3 rounded-xl border border-slate-200 p-3 dark:border-slate-700"><p role="status" className="break-words">Removed {removed.title}.</p><button type="button" disabled={pending} onClick={restoreChart} className="min-h-11 px-3 font-semibold text-blue-700 underline dark:text-blue-300">Undo</button></div>}
      {/* Grid of Pinned Charts */}
      {pinnedCharts.length === 0 ? (
        <div className="p-12 text-center rounded-2xl border border-dashed border-slate-300 dark:border-slate-800 bg-white/40 dark:bg-slate-900/20 space-y-4">
          <BarChart3 className="h-12 w-12 text-slate-400 mx-auto opacity-50" />
          <div className="space-y-1">
            <h3 className="text-base font-bold text-slate-800 dark:text-slate-200">No pinned charts yet</h3>
            <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 max-w-md mx-auto">
              You can pin any interactive chart from Visualizations, AI Copilot, or Explorers using the <span className="font-semibold text-blue-500">Pin icon (📌)</span>.
            </p>
          </div>
          <div className="flex flex-wrap justify-center gap-3 pt-2">
            <NavLink
              to="/visualizations"
              className="px-4 py-2 rounded-xl bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 text-xs font-semibold"
            >
              Go to Visualizations
            </NavLink>
            <NavLink
              to="/query"
              className="px-4 py-2 rounded-xl bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 text-xs font-semibold"
            >
              Ask AI Assistant
            </NavLink>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {pinnedCharts.map((item) => (
            <div
              key={item.id}
              className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 p-5 shadow-sm space-y-4 flex flex-col justify-between transition-all hover:shadow-md"
            >
              {/* Card Top Header */}
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h3 className="text-base font-bold text-slate-900 dark:text-white leading-tight">
                    {item.title}
                  </h3>
                  <div className="flex flex-wrap items-center gap-2 mt-1 text-xs text-slate-500 dark:text-slate-400">
                    <span className="px-2 py-0.5 rounded-md bg-blue-50 dark:bg-blue-950/80 text-blue-600 dark:text-blue-400 font-medium">
                      {item.chart_type}
                    </span>
                    <span>•</span>
                    <span className="flex items-center gap-1 font-mono text-[11px]">
                      <Calendar className="h-3 w-3" />
                      {new Date(item.pinned_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                    <span>•</span>
                    <span>{item.source_page}</span>
                    <span className="basis-full break-words">{item.dataset_name || 'Dataset source unavailable'}{item.dataset_version != null ? ` · Version ${item.dataset_version}` : ''}</span>
                  </div>
                </div>

                <button
                  type="button"
                  disabled={pending}
                  onClick={() => handleUnpin(item)}
                  aria-label={`Remove ${item.title} from dashboard`}
                  className="min-h-11 min-w-11 shrink-0 rounded-lg text-slate-600 hover:text-rose-700 hover:bg-rose-50 dark:text-slate-300 dark:hover:bg-rose-950/50 transition-colors"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>

              {/* Interactive Plotly Chart */}
              <div className="rounded-xl border border-slate-100 dark:border-slate-800/80 bg-slate-50/50 dark:bg-slate-950/50 p-2">
                <PlotlyChart
                  spec={item.figure_spec}
                  height={380}
                  title={item.title}
                  chartType={item.chart_type}
                  sourcePage={item.source_page}
                  source={{ name: item.dataset_name, version: item.dataset_version }}
                />
              </div>

              {/* Card Footer / Notes */}
              {item.notes && (
                <div className="text-xs text-slate-600 dark:text-slate-400 italic bg-slate-50 dark:bg-slate-900 p-2.5 rounded-lg border border-slate-200 dark:border-slate-800">
                  "{item.notes}"
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
