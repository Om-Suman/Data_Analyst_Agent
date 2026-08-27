import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Upload,
  Sparkles,
  Compass,
  MessageSquareCode,
  FileText,
  BarChart3,
  Lightbulb,
  TrendingUp,
  AlertTriangle,
  FileSpreadsheet,
  Settings,
  Database,
  Terminal,
  Layers,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { useDataset } from '../context/DatasetContext';

interface SidebarProps {
  collapsed?: boolean;
  onToggleCollapse?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  collapsed: externalCollapsed,
  onToggleCollapse,
}) => {
  const [internalCollapsed, setInternalCollapsed] = useState(false);
  const isCollapsed = externalCollapsed !== undefined ? externalCollapsed : internalCollapsed;

  const toggle = () => {
    if (onToggleCollapse) {
      onToggleCollapse();
    } else {
      setInternalCollapsed(!internalCollapsed);
    }
  };

  const { activeDataset, hasDataset, pinnedCharts } = useDataset();

  const navItems = [
    { to: '/', label: 'Overview', icon: LayoutDashboard, badge: null },
    { to: '/custom-dashboard', label: 'Custom BI Canvas', icon: Layers, badge: pinnedCharts.length > 0 ? String(pinnedCharts.length) : null },
    { to: '/upload', label: 'Upload & Ingest', icon: Upload, badge: null },
    { to: '/cleaning', label: 'Data Cleaning', icon: Sparkles, badge: hasDataset ? 'v' + activeDataset?.version : null },
    { to: '/explorer', label: 'Data Explorer', icon: Compass, badge: null },
    { to: '/sql', label: 'SQL Studio', icon: Terminal, badge: 'SQL' },
    { to: '/query', label: 'AI Data Query', icon: MessageSquareCode, badge: 'AI' },
    { to: '/document', label: 'Document QA', icon: FileText, badge: 'RAG' },
    { to: '/visualizations', label: 'Visualizations', icon: BarChart3, badge: '14+' },
    { to: '/insights', label: 'Insights & Stories', icon: Lightbulb, badge: null },
    { to: '/forecasting', label: 'Forecasting', icon: TrendingUp, badge: null },
    { to: '/anomalies', label: 'Anomaly Detection', icon: AlertTriangle, badge: null },
    { to: '/reports', label: 'Export Reports', icon: FileSpreadsheet, badge: null },
    { to: '/settings', label: 'HF Config & Settings', icon: Settings, badge: null },
  ];

  return (
    <aside
      className={`flex-shrink-0 border-r border-slate-200 dark:border-slate-800/80 bg-white dark:bg-[#0d1322] flex flex-col justify-between h-screen sticky top-0 transition-all duration-300 z-30 ${
        isCollapsed ? 'w-20' : 'w-64'
      }`}
    >
      <div className="flex-1 flex flex-col min-h-0">
        {/* Logo Header */}
        <div className="p-4 border-b border-slate-200 dark:border-slate-800/80 flex items-center justify-between">
          <div className="flex items-center gap-3 overflow-hidden">
            <div className="h-10 w-10 min-w-[2.5rem] rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/20 text-white font-bold text-lg">
              ⚡
            </div>
            {!isCollapsed && (
              <div className="overflow-hidden">
                <h1 className="font-bold text-sm text-slate-900 dark:text-white tracking-tight truncate leading-tight">
                  Data Analyst Agent
                </h1>
                <p className="text-xs text-blue-600 dark:text-blue-400 font-semibold flex items-center gap-1.5 mt-0.5">
                  <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
                  Enterprise 2.0
                </p>
              </div>
            )}
          </div>

          <button
            onClick={toggle}
            className="hidden sm:flex p-1.5 rounded-lg border border-slate-200 dark:border-slate-800 hover:bg-slate-100 dark:hover:bg-slate-800/60 text-slate-500 dark:text-slate-400 transition-colors"
            title={isCollapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
          >
            {isCollapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
          </button>
        </div>

        {/* Active Dataset Status (Expanded only) */}
        {!isCollapsed && (
          <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-800/60 bg-slate-50 dark:bg-slate-900/40">
            <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 mb-1">
              <span className="flex items-center gap-1.5 font-semibold">
                <Database className="h-3.5 w-3.5 text-blue-500" />
                Active Dataset
              </span>
              {hasDataset && (
                <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-blue-100 dark:bg-blue-950/80 text-blue-600 dark:text-blue-400">
                  v{activeDataset?.version}
                </span>
              )}
            </div>
            <p className="text-xs font-bold text-slate-800 dark:text-slate-200 truncate font-mono">
              {activeDataset ? activeDataset.name : 'No dataset loaded'}
            </p>
          </div>
        )}

        {/* Navigation Link List */}
        <div className="flex-1 overflow-y-auto p-3 space-y-1">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-semibold transition-all group relative ${
                  isActive
                    ? 'bg-blue-600 text-white shadow-md shadow-blue-600/20'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-800/50'
                } ${isCollapsed ? 'justify-center' : ''}`
              }
              title={isCollapsed ? item.label : undefined}
            >
              <item.icon className="h-5 w-5 flex-shrink-0" />

              {!isCollapsed && (
                <div className="flex-1 flex items-center justify-between overflow-hidden">
                  <span className="truncate">{item.label}</span>
                  {item.badge && (
                    <span className="ml-2 text-[10px] font-extrabold uppercase px-2 py-0.5 rounded-full bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300">
                      {item.badge}
                    </span>
                  )}
                </div>
              )}
            </NavLink>
          ))}
        </div>
      </div>

      {/* Sidebar Footer */}
      <div className="p-3 border-t border-slate-200 dark:border-slate-800/80 bg-slate-50 dark:bg-slate-900/30">
        {!isCollapsed ? (
          <div className="text-center text-xs text-slate-400 dark:text-slate-500 font-mono">
            FastAPI + React • v2.0 Enterprise
          </div>
        ) : (
          <div className="flex justify-center text-xs font-mono text-slate-400">2.0</div>
        )}
      </div>
    </aside>
  );
};
