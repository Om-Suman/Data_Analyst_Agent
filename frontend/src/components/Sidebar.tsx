import React from 'react';
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
} from 'lucide-react';
import { useDataset } from '../context/DatasetContext';

export const Sidebar: React.FC = () => {
  const { activeDataset, hasDataset } = useDataset();

  const navItems = [
    { to: '/', label: 'Dashboard', icon: LayoutDashboard, badge: null },
    { to: '/upload', label: 'Upload & Ingest', icon: Upload, badge: null },
    { to: '/cleaning', label: 'Data Cleaning', icon: Sparkles, badge: hasDataset ? 'v' + activeDataset?.version : null },
    { to: '/explorer', label: 'Data Explorer', icon: Compass, badge: null },
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
    <aside className="w-64 flex-shrink-0 border-r border-slate-800 bg-[#0d1322]/90 flex flex-col justify-between h-screen sticky top-0">
      <div>
        {/* Logo and Brand */}
        <div className="p-5 border-b border-slate-800 flex items-center gap-3">
          <div className="h-9 w-9 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/20 text-white font-bold text-lg">
            ⚡
          </div>
          <div>
            <h1 className="font-bold text-sm text-white tracking-tight leading-none">
              Data Analyst Agent
            </h1>
            <p className="text-[11px] text-blue-400 font-medium mt-1 flex items-center gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-blue-400 animate-pulse"></span>
              FastAPI + React 2.0
            </p>
          </div>
        </div>

        {/* Active Dataset Status Widget */}
        <div className="px-4 py-3 border-b border-slate-800/80 bg-slate-900/40">
          <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1">
            <span className="flex items-center gap-1">
              <Database className="h-3 w-3 text-blue-400" />
              Active Target
            </span>
            {hasDataset && (
              <span className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-blue-500/10 text-blue-400 border border-blue-500/20">
                {activeDataset?.rows.toLocaleString()} rows
              </span>
            )}
          </div>
          <div className="text-xs font-semibold text-slate-200 truncate">
            {hasDataset ? activeDataset?.name : <span className="text-slate-500 italic">No dataset loaded</span>}
          </div>
        </div>

        {/* Navigation Links */}
        <nav className="p-3 space-y-1 overflow-y-auto max-h-[calc(100vh-230px)]">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                  isActive
                    ? 'bg-blue-600/15 text-blue-400 border border-blue-500/30 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`
              }
            >
              <div className="flex items-center gap-3">
                <item.icon className="h-4 w-4 flex-shrink-0" />
                <span>{item.label}</span>
              </div>
              {item.badge && (
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                  {item.badge}
                </span>
              )}
            </NavLink>
          ))}
        </nav>
      </div>

      {/* Footer Info */}
      <div className="p-4 border-t border-slate-800 bg-slate-950/40 text-[11px] text-slate-500">
        <div className="flex items-center justify-between">
          <span>Backend API</span>
          <span className="text-emerald-400 flex items-center gap-1 font-mono">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400"></span>
            Online: 8000
          </span>
        </div>
      </div>
    </aside>
  );
};
