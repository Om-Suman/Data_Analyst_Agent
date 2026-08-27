import React from 'react';
import { LucideIcon, TrendingUp, TrendingDown } from 'lucide-react';

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
  const cardColorMap = {
    blue: 'border-blue-200 dark:border-blue-500/30 bg-blue-50/50 dark:bg-gradient-to-br dark:from-blue-950/20 dark:to-slate-900/40 text-blue-900 dark:text-blue-400',
    green: 'border-emerald-200 dark:border-emerald-500/30 bg-emerald-50/50 dark:bg-gradient-to-br dark:from-emerald-950/20 dark:to-slate-900/40 text-emerald-900 dark:text-emerald-400',
    amber: 'border-amber-200 dark:border-amber-500/30 bg-amber-50/50 dark:bg-gradient-to-br dark:from-amber-950/20 dark:to-slate-900/40 text-amber-900 dark:text-amber-400',
    purple: 'border-purple-200 dark:border-purple-500/30 bg-purple-50/50 dark:bg-gradient-to-br dark:from-purple-950/20 dark:to-slate-900/40 text-purple-900 dark:text-purple-400',
    red: 'border-rose-200 dark:border-rose-500/30 bg-rose-50/50 dark:bg-gradient-to-br dark:from-rose-950/20 dark:to-slate-900/40 text-rose-900 dark:text-rose-400',
  };

  const iconBgMap = {
    blue: 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20',
    green: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20',
    amber: 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20',
    purple: 'bg-purple-500/10 text-purple-600 dark:text-purple-400 border border-purple-500/20',
    red: 'bg-rose-500/10 text-rose-600 dark:text-rose-400 border border-rose-500/20',
  };

  return (
    <div
      className={`relative overflow-hidden rounded-2xl border p-5 sm:p-6 transition-all duration-200 hover:shadow-lg dark:hover:border-slate-700 ${cardColorMap[color]} ${className}`}
    >
      <div className="flex items-start justify-between">
        <div className="space-y-1.5">
          <p className="text-xs sm:text-sm font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
            {title}
          </p>
          <p className="text-2xl sm:text-4xl font-extrabold tracking-tight text-slate-900 dark:text-white font-mono">
            {value}
          </p>
          {subtitle && (
            <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-400 font-medium">
              {subtitle}
            </p>
          )}
        </div>

        {Icon && (
          <div className={`p-3 rounded-xl ${iconBgMap[color]} shadow-sm`}>
            <Icon className="h-6 w-6" />
          </div>
        )}
      </div>

      {trend && (
        <div className="mt-4 flex items-center gap-1.5 text-xs sm:text-sm font-semibold">
          {trend.positive ? (
            <span className="flex items-center gap-1 text-emerald-600 dark:text-emerald-400 bg-emerald-100 dark:bg-emerald-950/60 px-2 py-0.5 rounded-md">
              <TrendingUp className="h-3.5 w-3.5" />
              +{trend.value}
            </span>
          ) : (
            <span className="flex items-center gap-1 text-rose-600 dark:text-rose-400 bg-rose-100 dark:bg-rose-950/60 px-2 py-0.5 rounded-md">
              <TrendingDown className="h-3.5 w-3.5" />
              -{trend.value}
            </span>
          )}
          <span className="text-slate-500 text-xs">vs baseline</span>
        </div>
      )}
    </div>
  );
};
