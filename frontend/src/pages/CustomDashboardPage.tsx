import React from 'react';
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

export const CustomDashboardPage: React.FC = () => {
  const { pinnedCharts, unpinChart, hasDataset, activeDataset } = useDataset();
  const { success, error } = useToast();

  const handleUnpin = async (id: string, title: string) => {
    try {
      await unpinChart(id);
      success('Chart Removed', `Unpinned "${title}" from dashboard.`);
    } catch (err: any) {
      error('Error', err.message || 'Failed to remove pinned chart.');
    }
  };

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
          <div className="flex justify-center gap-3 pt-2">
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
                  <div className="flex items-center gap-2 mt-1 text-xs text-slate-500 dark:text-slate-400">
                    <span className="px-2 py-0.5 rounded-md bg-blue-50 dark:bg-blue-950/80 text-blue-600 dark:text-blue-400 font-medium">
                      {item.chart_type}
                    </span>
                    <span>•</span>
                    <span className="flex items-center gap-1 font-mono text-[11px]">
                      <Calendar className="h-3 w-3" />
                      {new Date(item.pinned_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                    <span>•</span>
                    <span>Source: {item.source_page}</span>
                  </div>
                </div>

                <button
                  onClick={() => handleUnpin(item.id, item.title)}
                  title="Remove from Dashboard"
                  className="p-1.5 rounded-lg text-slate-400 hover:text-rose-500 hover:bg-rose-50 dark:hover:bg-rose-950/50 transition-colors"
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
