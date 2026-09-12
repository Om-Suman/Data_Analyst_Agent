import React, { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import { Maximize2, Minimize2, Pin, Download, Check } from "lucide-react";
import { useDataset } from "../context/DatasetContext";
import { useToast } from "./Toast";

interface PlotlyChartProps {
  spec: any;
  className?: string;
  height?: number | string;
  title?: string;
  chartType?: string;
  sourcePage?: string;
  showActions?: boolean;
}

export const PlotlyChart: React.FC<PlotlyChartProps> = ({
  spec,
  className = "",
  height = 450,
  title = "Analytics Chart",
  chartType = "Chart",
  sourcePage = "Visualizations",
  showActions = true,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const fullscreenRef = useRef<HTMLDivElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [pinned, setPinned] = useState(false);

  const { pinChart } = useDataset();
  const { success, error } = useToast();

  useEffect(() => {
    if (!containerRef.current || !spec) return;

    const data = spec.data || [];
    const isDark = document.documentElement.classList.contains("dark");
    const layout = {
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: {
        color: isDark ? "#94a3b8" : "#475569",
        family: "Inter, system-ui, sans-serif",
        size: 13,
      },
      margin: { l: 45, r: 25, t: 45, b: 45 },
      autosize: true,
      ...(spec.layout || {}),
    };

    const config = {
      responsive: true,
      displayModeBar: showActions ? "hover" : false,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
      toImageButtonOptions: {
        format: "png",
        filename: `${title.toLowerCase().replace(/\s+/g, "_")}`,
        height: 700,
        width: 1200,
        scale: 2,
      },
      ...(spec.config || {}),
    };

    Plotly.newPlot(containerRef.current, data, layout, config);

    const handleResize = () => {
      if (containerRef.current) {
        Plotly.Plots.resize(containerRef.current);
      }
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      if (containerRef.current) {
        Plotly.purge(containerRef.current);
      }
    };
  }, [spec, title]);

  // Fullscreen plot effect
  useEffect(() => {
    if (!isFullscreen || !fullscreenRef.current || !spec) return;

    const isDark = document.documentElement.classList.contains("dark");
    const layout = {
      paper_bgcolor: isDark ? "#0f172a" : "#ffffff",
      plot_bgcolor: "transparent",
      font: {
        color: isDark ? "#cbd5e1" : "#1e293b",
        family: "Inter, system-ui, sans-serif",
        size: 14,
      },
      margin: { l: 60, r: 40, t: 60, b: 60 },
      autosize: true,
      ...(spec.layout || {}),
    };

    Plotly.newPlot(fullscreenRef.current, spec.data || [], layout, {
      responsive: true,
      displaylogo: false,
    });

    return () => {
      if (fullscreenRef.current) {
        Plotly.purge(fullscreenRef.current);
      }
    };
  }, [isFullscreen, spec]);

  const handlePin = async () => {
    if (!spec) return;
    try {
      await pinChart({
        title: title || "Custom Chart",
        chart_type: chartType,
        figure_spec: spec,
        source_page: sourcePage,
      });
      setPinned(true);
      success("Chart Pinned", "Saved to your Custom BI Dashboard.");
      setTimeout(() => setPinned(false), 3000);
    } catch (err: any) {
      error("Pin Failed", err.message || "Could not pin chart.");
    }
  };

  const handleExportPNG = () => {
    if (!containerRef.current) return;
    Plotly.downloadImage(containerRef.current, {
      format: "png",
      filename: `${title.toLowerCase().replace(/\s+/g, "_")}`,
      width: 1200,
      height: 700,
      scale: 2,
    });
    success("Exporting Image", "High-res chart download started.");
  };

  if (!spec) {
    return (
      <div
        className={`flex items-center justify-center rounded-xl border border-slate-800 bg-slate-900/30 text-slate-500 text-sm ${className}`}
        style={{ height }}
      >
        No chart specification available
      </div>
    );
  }

  return (
    <div className={`relative group ${className}`}>
      {/* Action Toolbar */}
      {showActions && (
        <div className="absolute top-2 right-2 z-10 flex items-center gap-1.5 opacity-80 group-hover:opacity-100 transition-opacity bg-slate-900/80 dark:bg-slate-950/80 backdrop-blur-md p-1 rounded-lg border border-slate-700/60 shadow-md">
          <button
            onClick={handlePin}
            title="Pin to Custom Dashboard"
            className="p-1.5 rounded-md text-slate-400 hover:text-amber-300 hover:bg-slate-800 transition-colors"
          >
            {pinned ? (
              <Check className="h-4 w-4 text-emerald-400" />
            ) : (
              <Pin className="h-4 w-4" />
            )}
          </button>
          <button
            onClick={handleExportPNG}
            title="Download PNG"
            className="p-1.5 rounded-md text-slate-400 hover:text-blue-300 hover:bg-slate-800 transition-colors"
          >
            <Download className="h-4 w-4" />
          </button>
          <button
            onClick={() => setIsFullscreen(true)}
            title="Expand Fullscreen"
            className="p-1.5 rounded-md text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            <Maximize2 className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* Chart Canvas */}
      <div ref={containerRef} style={{ height, width: "100%" }} />

      {/* Fullscreen Expansion Modal */}
      {isFullscreen && (
        <div className="fixed inset-0 z-50 bg-slate-950/90 backdrop-blur-lg flex flex-col p-6 animate-in fade-in">
          <div className="flex items-center justify-between pb-4 border-b border-slate-800">
            <div>
              <h3 className="text-lg font-bold text-white">{title}</h3>
              <p className="text-xs text-slate-400">
                {chartType} • Source: {sourcePage}
              </p>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={handlePin}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold"
              >
                <Pin className="h-4 w-4" />
                {pinned ? "Pinned!" : "Pin to Dashboard"}
              </button>
              <button
                onClick={() => setIsFullscreen(false)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold"
              >
                <Minimize2 className="h-4 w-4" />
                Exit Fullscreen
              </button>
            </div>
          </div>
          <div className="flex-1 w-full mt-4" ref={fullscreenRef} />
        </div>
      )}
    </div>
  );
};
