import React from "react";
import { NavLink, useLocation } from "react-router-dom";
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
  X,
} from "lucide-react";
import { useDataset } from "../context/DatasetContext";

interface SidebarProps {
  collapsed?: boolean;
  onToggleCollapse?: () => void;
  mobile?: boolean;
  onNavigate?: () => void;
}
export const Sidebar: React.FC<SidebarProps> = ({
  collapsed = false,
  onToggleCollapse,
  mobile = false,
  onNavigate,
}) => {
  const { activeDataset, pinnedCharts } = useDataset();
  const { pathname } = useLocation();
  const compact = collapsed && !mobile;
  const groups = [
    {
      name: "Data",
      items: [
        { to: "/", label: "Overview", icon: LayoutDashboard },
        { to: "/upload", label: "Upload data", icon: Upload },
        { to: "/cleaning", label: "Data cleaning", icon: Sparkles },
        { to: "/document", label: "Document questions", icon: FileText },
      ],
    },
    {
      name: "Analyze",
      items: [
        { to: "/query", label: "AI data query", icon: MessageSquareCode },
        { to: "/explorer", label: "Data explorer", icon: Compass },
        { to: "/sql", label: "SQL Studio", icon: Terminal },
        { to: "/visualizations", label: "Visualizations", icon: BarChart3 },
      ],
    },
    {
      name: "Share",
      items: [
        { to: "/custom-dashboard", label: "BI canvas", icon: Layers },
        { to: "/reports", label: "Export reports", icon: FileSpreadsheet },
      ],
    },
  ];
  const advanced = [
    { to: "/insights", label: "Insights and stories", icon: Lightbulb },
    { to: "/forecasting", label: "Forecasting", icon: TrendingUp },
    { to: "/anomalies", label: "Anomaly detection", icon: AlertTriangle },
  ];
  const link = (item: (typeof advanced)[number]) => (
    <NavLink
      key={item.to}
      to={item.to}
      end={item.to === "/"}
      onClick={onNavigate}
      aria-label={compact ? item.label : undefined}
      title={compact ? item.label : undefined}
      className={({ isActive }) =>
        `flex min-h-11 items-center gap-3 rounded-xl px-3 py-2 text-sm font-medium ${compact ? "justify-center" : ""} ${isActive ? "bg-blue-600 text-white" : "text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"}`
      }
    >
      <item.icon className="h-5 w-5 shrink-0" aria-hidden="true" />
      {!compact && <span className="flex-1">{item.label}</span>}
      {!compact &&
        item.to === "/custom-dashboard" &&
        pinnedCharts.length > 0 && (
          <span className="text-xs tabular-nums">{pinnedCharts.length}</span>
        )}
    </NavLink>
  );
  return (
    <aside
      aria-label="Workspace navigation"
      className={`flex shrink-0 flex-col border-r border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0d1322] ${mobile ? "h-full w-full" : `sticky top-0 h-dvh ${compact ? "w-20" : "w-64"}`}`}
    >
      <div className="flex min-h-16 items-center justify-between gap-2 border-b border-slate-200 dark:border-slate-800 px-3">
        {!compact && (
          <span className="font-semibold text-sm">Data Analyst Agent</span>
        )}
        <button
          type="button"
          autoFocus={mobile}
          onClick={mobile ? onNavigate : onToggleCollapse}
          aria-label={
            mobile
              ? "Close navigation"
              : compact
                ? "Expand navigation"
                : "Collapse navigation"
          }
          className="min-h-11 min-w-11 inline-flex items-center justify-center rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800"
        >
          {mobile ? (
            <X className="h-5 w-5" />
          ) : compact ? (
            <ChevronRight className="h-5 w-5" />
          ) : (
            <ChevronLeft className="h-5 w-5" />
          )}
        </button>
      </div>
      {!compact && (
        <div className="px-4 py-4 border-b border-slate-200 dark:border-slate-800">
          <p className="text-xs text-slate-600 dark:text-slate-400 flex items-center gap-2">
            <Database className="h-4 w-4" />
            Active dataset
          </p>
          <p className="text-sm mt-1 truncate" title={activeDataset?.name}>
            {activeDataset?.name || "No dataset loaded"}
          </p>
        </div>
      )}
      <nav aria-label="Main" className="flex-1 overflow-y-auto p-3 space-y-5">
        {groups.map((group) => (
          <div key={group.name}>
            {!compact && (
              <h2 className="px-3 mb-1 text-sm font-semibold text-slate-600 dark:text-slate-400">
                {group.name}
              </h2>
            )}
            {group.items.map(link)}
            {group.name === "Analyze" &&
              (compact ? (
                advanced.map(link)
              ) : (
                <details
                  open={advanced.some((item) => item.to === pathname)}
                  className="mt-1"
                >
                  <summary className="cursor-pointer px-3 py-3 text-sm text-slate-700 dark:text-slate-300 rounded-lg">
                    More analysis tools
                  </summary>
                  <div>{advanced.map(link)}</div>
                </details>
              ))}
          </div>
        ))}
      </nav>
      <div className="p-3 border-t border-slate-200 dark:border-slate-800">
        {link({ to: "/settings", label: "Settings", icon: Settings })}
        {!compact && (
          <p className="px-3 mt-2 text-xs text-slate-600 dark:text-slate-400">
            Open Copilot with Ctrl / ⌘ + K
          </p>
        )}
      </div>
    </aside>
  );
};
