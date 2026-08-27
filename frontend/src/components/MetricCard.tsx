import React from 'react';
import { LucideIcon } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: LucideIcon;
  trend?: {
    value: string | number;
    positive: boolean;
  };
  color?: 'blue' | 'green' | 'amber' | 'purple' | 'red';
  className?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  color = 'blue',
  className = '',
}) => {
  const colorMap = {
    blue: 'border-blue-500/30 bg-gradient-to-br from-blue-950/20 to-slate-900/40 text-blue-400',
    green: 'border-emerald-500/30 bg-gradient-to-br from-emerald-950/20 to-slate-900/40 text-emerald-400',
    amber: 'border-amber-500/30 bg-gradient-to-br from-amber-950/20 to-slate-900/40 text-amber-400',
    purple: 'border-purple-500/30 bg-gradient-to-br from-purple-950/20 to-slate-900/40 text-purple-400',
    red: 'border-rose-500/30 bg-gradient-to-br from-rose-950/20 to-slate-900/40 text-rose-400',
  };

  const iconBgMap = {
    blue: 'bg-blue-500/10 text-blue-400 border border-blue-500/20',
    green: 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20',
    amber: 'bg-amber-500/10 text-amber-400 border border-amber-500/20',
    purple: 'bg-purple-500/10 text-purple-400 border border-purple-500/20',
    red: 'bg-rose-500/10 text-rose-400 border border-rose-500/20',
  };

  return (
    <div
      className={`relative overflow-hidden rounded-xl border p-5 transition-all duration-200 hover:border-slate-700 hover:shadow-lg ${colorMap[color]} ${className}`}
    >
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">{title}</p>
          <p className="text-2xl sm:text-3xl font-bold tracking-tight text-white">{value}</p>
          {subtitle && <p className="text-xs text-slate-400">{subtitle}</p>}
        </div>

        {Icon && (
          <div className={`p-2.5 rounded-lg ${iconBgMap[color]}`}>
            <Icon className="h-5 w-5" />
          </div>
        )}
      </div>

      {trend && (
        <div className="mt-3 flex items-center gap-1.5 text-xs">
          <span className={`font-semibold ${trend.positive ? 'text-emerald-400' : 'text-rose-400'}`}>
            {trend.positive ? '↑' : '↓'} {trend.value}
          </span>
          <span className="text-slate-500">vs baseline</span>
        </div>
      )}
    </div>
  );
};
