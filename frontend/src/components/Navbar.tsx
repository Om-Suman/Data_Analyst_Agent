import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  Database,
  Key,
  Sun,
  Moon,
  Sparkles,
  Bot,
  Terminal,
  Layers,
  RefreshCw,
} from 'lucide-react';
import { useDataset } from '../context/DatasetContext';
import { useTheme } from '../context/ThemeContext';
import { datasetApi } from '../api/client';
import { useToast } from './Toast';

interface NavbarProps {
  onToggleCopilot?: () => void;
  isCopilotOpen?: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({ onToggleCopilot, isCopilotOpen }) => {
  const { datasets, activeDatasetName, setActiveDataset, refreshDatasets, config, pinnedCharts } = useDataset();
  const { theme, toggleTheme } = useTheme();
  const { success } = useToast();

  const handleSample = async (name: string) => {
    try {
      await datasetApi.loadSample(name);
      await refreshDatasets();
      success('Dataset Loaded', `Switched to sample: ${name}`);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <header className="h-16 border-b border-slate-200 dark:border-slate-800/80 bg-white/80 dark:bg-[#0d1322]/80 backdrop-blur-md px-4 sm:px-6 flex items-center justify-between sticky top-0 z-20 transition-colors">
      {/* Left: Dataset Selector & Demo Loaders */}
      <div className="flex items-center gap-3 sm:gap-4 overflow-hidden">
        <div className="flex items-center gap-2">
          <Database className="h-4 w-4 text-blue-500 flex-shrink-0" />
          <select
            value={activeDatasetName || ''}
            onChange={(e) => {
              if (e.target.value) setActiveDataset(e.target.value);
            }}
            className="bg-slate-100 dark:bg-slate-900 border border-slate-300 dark:border-slate-700 text-xs sm:text-sm font-semibold text-slate-800 dark:text-slate-200 rounded-xl px-3 py-1.5 focus:outline-none focus:border-blue-500 max-w-[200px] sm:max-w-xs truncate"
          >
            {datasets.length === 0 ? (
              <option value="">No datasets loaded</option>
            ) : (
              datasets.map((d) => (
                <option key={d.name} value={d.name}>
                  {d.name} ({d.rows > 0 ? `${d.rows.toLocaleString()} rows` : 'text'})
                </option>
              ))
            )}
          </select>
        </div>

        {/* Quick Demo Pill Buttons */}
        <div className="hidden xl:flex items-center gap-1.5 pl-3 border-l border-slate-200 dark:border-slate-800">
          <span className="text-xs text-slate-400 font-medium">Quick Demo:</span>
          {['Sales Data', 'Employee Data', 'Finance Data'].map((sName) => (
            <button
              key={sName}
              onClick={() => handleSample(sName)}
              className="text-xs px-2.5 py-1 rounded-lg bg-slate-100 dark:bg-slate-800/80 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 border border-slate-300 dark:border-slate-700/60 font-medium transition-colors"
            >
              {sName.split(' ')[0]}
            </button>
          ))}
        </div>
      </div>

      {/* Right: Quick Tools, Theme Switcher & AI Copilot Trigger */}
      <div className="flex items-center gap-2 sm:gap-3">
        {/* Custom Dashboard Pill Shortcut */}
        <NavLink
          to="/custom-dashboard"
          className="hidden md:flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-100/80 dark:bg-slate-900/60 hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300 text-xs font-semibold transition-colors"
        >
          <Layers className="h-3.5 w-3.5 text-blue-500" />
          <span>BI Canvas</span>
          {pinnedCharts.length > 0 && (
            <span className="px-1.5 py-0.2 rounded-full bg-blue-600 text-white text-[10px] font-bold">
              {pinnedCharts.length}
            </span>
          )}
        </NavLink>

        {/* SQL Studio Pill Shortcut */}
        <NavLink
          to="/sql"
          className="hidden md:flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-100/80 dark:bg-slate-900/60 hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300 text-xs font-semibold transition-colors"
        >
          <Terminal className="h-3.5 w-3.5 text-emerald-500" />
          <span>SQL Studio</span>
        </NavLink>

        {/* Theme Switcher Toggle Button */}
        <button
          onClick={toggleTheme}
          title={theme === 'dark' ? 'Switch to Light Theme' : 'Switch to Dark Theme'}
          className="p-2 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-100 dark:bg-slate-900/60 hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300 transition-colors"
        >
          {theme === 'dark' ? <Sun className="h-4 w-4 text-amber-400" /> : <Moon className="h-4 w-4 text-indigo-600" />}
        </button>

        {/* Global AI Copilot Toggle Button */}
        <button
          onClick={onToggleCopilot}
          className={`flex items-center gap-2 px-3.5 py-1.5 rounded-xl text-xs sm:text-sm font-bold shadow-sm transition-all ${
            isCopilotOpen
              ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-blue-600/30 ring-2 ring-blue-400'
              : 'bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white shadow-blue-600/20'
          }`}
        >
          <Sparkles className="h-4 w-4 animate-spin-slow" />
          <span>AI Copilot</span>
        </button>
      </div>
    </header>
  );
};
