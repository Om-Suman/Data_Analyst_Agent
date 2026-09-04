import React, { useState, useEffect } from "react";
import {
  Lightbulb,
  Sparkles,
  TrendingUp,
  AlertOctagon,
  ShieldCheck,
  BookOpen,
  CheckCircle2,
  AlertCircle,
  Copy,
} from "lucide-react";
import { useDataset } from "../context/DatasetContext";
import { insightsApi } from "../api/client";
import { AIInsightsResponse } from "../types";

export const InsightsPage: React.FC = () => {
  const { hasDataset, config } = useDataset();
  const [quickInsights, setQuickInsights] = useState<string[]>([]);
  const [aiInsights, setAiInsights] = useState<AIInsightsResponse | null>(null);
  const [loadingAi, setLoadingAi] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!hasDataset) return;
    insightsApi
      .getQuickInsights()
      .then((res) => setQuickInsights(res.insights))
      .catch((err) => console.error(err));

    insightsApi
      .getCachedAIInsights()
      .then((res) => {
        if (res.executive_summary) setAiInsights(res);
      })
      .catch((err) => console.error(err));
  }, [hasDataset]);

  const handleGenerateAI = async () => {
    if (!hasDataset) return;
    setLoadingAi(true);
    setError(null);
    try {
      const res = await insightsApi.getAIInsights(1500);
      setAiInsights(res);
    } catch (err: any) {
      setError(
        err.response?.data?.detail ||
          "Failed to generate AI Business Insights.",
      );
    } finally {
      setLoadingAi(false);
    }
  };

  const handleCopy = () => {
    if (!aiInsights) return;
    const text = `Executive Summary:
${aiInsights.executive_summary}

Key Findings:
${aiInsights.key_findings.join("\n")}

Recommendations:
${aiInsights.recommendations.join("\n")}`;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!hasDataset) {
    return (
      <div className="p-8 text-center text-slate-500 dark:text-slate-400 border border-slate-200 dark:border-slate-800 rounded-2xl bg-white dark:bg-slate-900/30 shadow-sm">
        Please load or select a dataset first to generate statistical and AI
        business insights.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white tracking-tight flex items-center gap-2">
            <Lightbulb className="h-5 w-5 text-amber-500 dark:text-amber-400" />
            Insights & Data Stories
          </h2>
          <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-1">
            Automated statistical patterns and deep AI executive business
            intelligence
          </p>
        </div>

        <button
          onClick={handleGenerateAI}
          disabled={loadingAi}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white text-xs sm:text-sm font-semibold shadow-md shadow-blue-600/20 transition-all"
        >
          <Sparkles className="h-4 w-4" />
          {loadingAi ? "Generating AI Report..." : "Generate Full AI Story"}
        </button>
      </div>

      {error && (
        <div className="p-4 rounded-xl flex items-center gap-3 text-xs font-medium border bg-rose-50 dark:bg-rose-950/40 border-rose-200 dark:border-rose-500/30 text-rose-800 dark:text-rose-300">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Quick Statistical Facts */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 p-5 sm:p-6 space-y-3 shadow-sm">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-400 flex items-center gap-2">
          <TrendingUp className="h-4 w-4 text-emerald-500 dark:text-emerald-400" />
          Instant Statistical Observations
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
          {quickInsights.map((insight, idx) => (
            <div
              key={idx}
              className="flex items-start gap-2.5 p-3 rounded-xl bg-slate-50 dark:bg-slate-800/30 border border-slate-200 dark:border-slate-700/50 text-xs text-slate-700 dark:text-slate-300 shadow-sm"
            >
              <span className="text-blue-500 dark:text-blue-400 font-bold">
                •
              </span>
              <span className="leading-relaxed">{insight}</span>
            </div>
          ))}
        </div>
      </div>

      {/* AI Business Intelligence Section */}
      {aiInsights && (
        <div className="space-y-6">
          {/* Executive Summary */}
          <div className="rounded-2xl border border-blue-200 dark:border-blue-500/30 bg-gradient-to-br from-blue-50/70 via-white to-white dark:from-blue-950/20 dark:via-slate-900/60 dark:to-slate-900/40 p-6 space-y-3 shadow-sm dark:shadow-xl">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-blue-600 dark:text-blue-400">
                Executive Summary
              </span>
              <button
                onClick={handleCopy}
                className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200 font-medium transition-colors"
              >
                <Copy className="h-3.5 w-3.5" />
                {copied ? "Copied!" : "Copy Summary"}
              </button>
            </div>
            <p className="text-sm text-slate-800 dark:text-slate-200 leading-relaxed font-medium">
              {aiInsights.executive_summary}
            </p>
          </div>

          {/* Key Findings & Recommendations Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Key Findings */}
            <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 p-5 sm:p-6 space-y-3 shadow-sm">
              <h4 className="text-xs font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400 flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4" />
                Key Findings ({aiInsights.key_findings.length})
              </h4>
              <ul className="space-y-2">
                {aiInsights.key_findings.map((f, idx) => (
                  <li
                    key={idx}
                    className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700/60 text-xs text-slate-800 dark:text-slate-300 leading-relaxed"
                  >
                    {f}
                  </li>
                ))}
              </ul>
            </div>

            {/* Recommendations */}
            <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 p-5 sm:p-6 space-y-3 shadow-sm">
              <h4 className="text-xs font-bold uppercase tracking-wider text-blue-600 dark:text-blue-400 flex items-center gap-2">
                <ShieldCheck className="h-4 w-4" />
                Strategic Recommendations ({aiInsights.recommendations.length})
              </h4>
              <ul className="space-y-2">
                {aiInsights.recommendations.map((r, idx) => (
                  <li
                    key={idx}
                    className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700/60 text-xs text-slate-800 dark:text-slate-300 leading-relaxed"
                  >
                    {r}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Opportunities & Risks Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Opportunities */}
            <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 p-5 sm:p-6 space-y-3 shadow-sm">
              <h4 className="text-xs font-bold uppercase tracking-wider text-amber-600 dark:text-amber-400 flex items-center gap-2">
                <Sparkles className="h-4 w-4" />
                Growth Opportunities
              </h4>
              <ul className="space-y-2">
                {aiInsights.opportunities?.map((op, idx) => (
                  <li
                    key={idx}
                    className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700/60 text-xs text-slate-800 dark:text-slate-300 leading-relaxed"
                  >
                    {op}
                  </li>
                ))}
              </ul>
            </div>

            {/* Risks */}
            <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 p-5 sm:p-6 space-y-3 shadow-sm">
              <h4 className="text-xs font-bold uppercase tracking-wider text-rose-600 dark:text-rose-400 flex items-center gap-2">
                <AlertOctagon className="h-4 w-4" />
                Identified Risks & Caveats
              </h4>
              <ul className="space-y-2">
                {aiInsights.risks?.map((risk, idx) => (
                  <li
                    key={idx}
                    className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700/60 text-xs text-slate-800 dark:text-slate-300 leading-relaxed"
                  >
                    {risk}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Narrative Data Story */}
          {aiInsights.data_story && (
            <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 p-6 space-y-3 shadow-sm">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-400 flex items-center gap-2">
                <BookOpen className="h-4 w-4 text-purple-500 dark:text-purple-400" />
                Narrative Data Story
              </h4>
              <p className="text-xs text-slate-800 dark:text-slate-300 leading-relaxed whitespace-pre-wrap">
                {aiInsights.data_story}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
