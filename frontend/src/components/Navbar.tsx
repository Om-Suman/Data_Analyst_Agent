import React from "react";
import { Menu, Sun, Moon, Sparkles } from "lucide-react";
import { useDataset } from "../context/DatasetContext";
import { useTheme } from "../context/ThemeContext";
interface NavbarProps {
  onToggleCopilot?: () => void;
  isCopilotOpen?: boolean;
  onOpenNavigation?: () => void;
  menuRef?: React.RefObject<HTMLButtonElement>;
  mobileOpen?: boolean;
}
export const Navbar: React.FC<NavbarProps> = ({
  onToggleCopilot,
  isCopilotOpen,
  onOpenNavigation,
  menuRef,
  mobileOpen,
}) => {
  const { datasets, activeDatasetName, setActiveDataset } = useDataset();
  const { theme, toggleTheme } = useTheme();
  return (
    <header className="sticky top-0 z-20 flex min-h-16 items-center justify-between gap-2 border-b border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0d1322] px-3 sm:px-6 py-2">
      <div className="flex flex-1 min-w-0 items-center gap-2">
        <button
          ref={menuRef}
          type="button"
          aria-label="Open navigation"
          aria-expanded={mobileOpen}
          aria-controls="mobile-navigation"
          onClick={onOpenNavigation}
          className="lg:hidden shrink-0 min-h-11 min-w-11 flex items-center justify-center rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800"
        >
          <Menu className="h-5 w-5" />
        </button>
        <div className="min-w-0 max-w-xs flex-1">
          <label htmlFor="active-dataset" className="sr-only">
            Active dataset
          </label>
          <select
            id="active-dataset"
            value={activeDatasetName || ""}
            disabled={datasets.length === 0}
            onChange={(e) => {
              if (e.target.value) setActiveDataset(e.target.value);
            }}
            className="w-full min-w-0 min-h-11 rounded-xl border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 text-slate-900 dark:text-slate-100 px-2 text-base sm:text-sm truncate"
          >
            {datasets.length === 0 ? (
              <option value="">No datasets loaded</option>
            ) : (
              datasets.map((d) => (
                <option key={d.name} value={d.name}>
                  {d.name}
                </option>
              ))
            )}
          </select>
        </div>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        <button
          type="button"
          onClick={toggleTheme}
          aria-label={
            theme === "dark" ? "Switch to light theme" : "Switch to dark theme"
          }
          className="min-h-11 min-w-11 inline-flex items-center justify-center rounded-xl border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800"
        >
          {theme === "dark" ? (
            <Sun className="h-4 w-4" />
          ) : (
            <Moon className="h-4 w-4" />
          )}
        </button>
        <button
          type="button"
          onClick={onToggleCopilot}
          aria-label="AI Copilot"
          aria-expanded={isCopilotOpen}
          aria-keyshortcuts="Control+k Meta+k"
          title="AI Copilot (Ctrl / ⌘ + K)"
          className="min-h-11 min-w-11 flex items-center justify-center gap-2 px-3 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold"
        >
          <Sparkles className="h-4 w-4" />
          <span className="hidden sm:inline">AI Copilot</span>
        </button>
      </div>
    </header>
  );
};
