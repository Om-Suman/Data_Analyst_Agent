import React, { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';

interface PlotlyChartProps {
  spec: any;
  className?: string;
  height?: number | string;
}

export const PlotlyChart: React.FC<PlotlyChartProps> = ({ spec, className = '', height = 400 }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current || !spec) return;

    const data = spec.data || [];
    const layout = {
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { color: '#94a3b8', family: 'Inter, system-ui, sans-serif' },
      margin: { l: 40, r: 20, t: 40, b: 40 },
      autosize: true,
      ...(spec.layout || {}),
    };
    const config = {
      responsive: true,
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      toImageButtonOptions: {
        format: 'png',
        filename: 'data_analyst_agent_chart',
        height: 600,
        width: 1000,
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

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (containerRef.current) {
        Plotly.purge(containerRef.current);
      }
    };
  }, [spec]);

  if (!spec) {
    return (
      <div className={`flex items-center justify-center bg-slate-900/50 rounded-lg border border-slate-800 text-slate-500 ${className}`} style={{ height }}>
        No visualization data available
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={`w-full overflow-hidden rounded-lg bg-slate-900/40 p-2 border border-slate-800/80 ${className}`}
      style={{ minHeight: typeof height === 'number' ? `${height}px` : height }}
    />
  );
};
