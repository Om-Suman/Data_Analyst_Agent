import { DatasetEmptyState } from "../components/DatasetEmptyState";
import React, { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import {
  Sparkles,
  Send,
  Code2,
  CheckCircle,
  AlertCircle,
  Clock,
  Download,
  Trash2,
  Bot,
  Terminal,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  Table as TableIcon,
  BarChart3,
  RotateCw,
} from "lucide-react";
import { useDataset } from "../context/DatasetContext";
import { queryApi } from "../api/client";
import { QueryHistoryItem, QueryResponse } from "../types";
import { PlotlyChart } from "../components/PlotlyChart";
import { DataTable } from "../components/DataTable";
import { MarkdownRenderer } from "../components/MarkdownRenderer";

export const AIQueryPage: React.FC = () => {
  const { activeDataset, hasDataset: hasActiveDataset } = useDataset();
  const hasDataset = hasActiveDataset && !activeDataset?.is_text;
  const [searchParams] = useSearchParams();
  const incomingPrompt = searchParams.get("prompt") || "";
  const sourceDataset = searchParams.get("dataset");
  const [question, setQuestion] = useState(incomingPrompt);
  useEffect(() => {
    setQuestion(incomingPrompt);
  }, [incomingPrompt]);
  const [loading, setLoading] = useState(false);
  const [currentResponse, setCurrentResponse] = useState<QueryResponse | null>(
    null,
  );
  const [history, setHistory] = useState<QueryHistoryItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [copiedAnswer, setCopiedAnswer] = useState(false);
  const [showCode, setShowCode] = useState(true);

  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(
    null,
  );
  const historySource = history.find(item => item.id === selectedHistoryId);
  const chartSource = selectedHistoryId
    ? { name: historySource?.dataset, version: historySource?.dataset_version }
    : undefined;

  const promptSuggestions = [
    "What are the top 5 highest sales categories?",
    "Plot a correlation heatmap between all numeric metrics",
    "Calculate the average and median values by category",
    "Forecast sales for the next 30 days",
    "Detect anomalies and outliers in our data",
  ];

  const fetchHistory = async () => {
    try {
      const res = await queryApi.getHistory();
      setHistory(res.history);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleSelectHistoryItem = (item: QueryHistoryItem) => {
    setSelectedHistoryId(item.id);
    setQuestion(item.question);
    setError(null);

    const codeBlocks =
      item.code_blocks && item.code_blocks.length > 0
        ? item.code_blocks
        : item.code
          ? [item.code]
          : [];

    setCurrentResponse({
      question: item.question,
      route: item.route || "dataframe_analysis",
      route_reason: "",
      routing_source: "history",
      model_used: item.model_used || "",
      insights: item.insights || item.result_summary,
      code_blocks: codeBlocks,
      execution_results:
        item.execution_results && item.execution_results.length > 0
          ? item.execution_results
          : item.code
            ? [
                {
                  code: item.code,
                  success: true,
                  execution_time: 0,
                  stdout: "",
                  error: null,
                  figures: [],
                  dataframes: {},
                },
              ]
            : [],
      tool_result: item.tool_result,
      error: null,
    });

    const mainEl = document.querySelector("main");
    if (mainEl) {
      mainEl.scrollTo({ top: 0, behavior: "smooth" });
    }
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleAsk = async (qText?: string) => {
    const q = qText || question;
    if (!q.trim() || !hasDataset || loading) return;
    setLoading(true);
    setError(null);
    setSelectedHistoryId(null);
    try {
      const res = await queryApi.ask(q, undefined, activeDataset?.name);
      setCurrentResponse(res);
      await fetchHistory();
      if (!qText) setQuestion("");
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to process AI query.");
    } finally {
      setLoading(false);
    }
  };

  const handleCopyAnswer = () => {
    if (!currentResponse?.insights) return;
    navigator.clipboard.writeText(currentResponse.insights);
    setCopiedAnswer(true);
    setTimeout(() => setCopiedAnswer(false), 2000);
  };

  const handleClearHistory = async () => {
    if (!window.confirm("Clear all query history?")) return;
    try {
      await queryApi.clearHistory();
      setHistory([]);
    } catch (err) {
      console.error(err);
    }
  };

  if (!hasDataset) {
    return (
      <DatasetEmptyState
        title="AI Data Query"
        description="Ask questions in plain language and explore the answers with your data."
      />
    );
  }

  // Filter out redundant intermediate dataframes
  const getCleanDataframes = () => {
    if (!currentResponse?.execution_results) return [];
    const allTables: { name: string; rows: any[] }[] = [];

    currentResponse.execution_results.forEach((res) => {
      if (!res.dataframes) return;
      const entries = Object.entries(res.dataframes);
      const hasSortedOrResult = entries.some(([k]) =>
        ["result_df", "sorted_df", "agg_stats"].includes(k),
      );

      entries.forEach(([name, rows]) => {
        if (!Array.isArray(rows) || rows.length === 0) return;
        // If result_df or sorted_df exists, skip intermediate 'grouped'
        if (hasSortedOrResult && name === "grouped") return;
        allTables.push({ name, rows });
      });
    });

    return allTables;
  };

  const cleanDataframes = getCleanDataframes();

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white tracking-tight flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-blue-500 dark:text-blue-400" />
          Natural Language AI Data Query
        </h2>
        <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-1">
          Ask any analytical question in plain English. The agent routes
          queries, runs safe pandas/plotly analytics, and synthesizes executive
          business intelligence.
        </p>
      </div>

      {incomingPrompt && (
        <p role="status" className="text-sm text-slate-700 dark:text-slate-300">
          {sourceDataset && sourceDataset !== activeDataset?.name
            ? `This prompt was selected for ${sourceDataset}. The active dataset is ${activeDataset?.name}; check the dataset before running.`
            : `Question prepared for ${activeDataset?.name}. Edit it, then choose Query to run.`}
        </p>
      )}
      {/* Query Input Box */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/50 p-4 sm:p-5 space-y-3 shadow-sm">
        <label htmlFor="analytical-question" className="block text-sm font-semibold">Your analytical question</label>
        <form className="flex flex-col sm:flex-row gap-2" onSubmit={(event) => { event.preventDefault(); void handleAsk(); }}>
          <input
            type="text"
            id="analytical-question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="What are the top five sales categories?"
            className="flex-1 min-w-0 min-h-11 bg-slate-50 dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded-xl px-4 py-2 text-base sm:text-sm text-slate-900 dark:text-slate-100 placeholder-slate-600 dark:placeholder-slate-400 focus:border-blue-500"
          />
          <button
            type="submit"
            disabled={loading || !question.trim()}
            className="min-h-11 shrink-0 px-5 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:opacity-40 text-white font-semibold flex items-center justify-center gap-2 text-sm"
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <span className="h-4 w-4 rounded-full border-2 border-white/30 border-t-white animate-spin"></span>
                Analyzing...
              </span>
            ) : (
              <>
                <Send className="h-4 w-4" />
                Query
              </>
            )}
          </button>
        </form>
        {loading && <span role="status" className="sr-only">Analyzing your question</span>}

        {/* Suggestion Pills */}
        <div className="flex flex-wrap items-center gap-2 pt-1">
          <span className="text-[11px] font-semibold text-slate-500 dark:text-slate-400">
            Suggestions:
          </span>
          {promptSuggestions.map((s, idx) => (
            <button
              key={idx}
              disabled={loading}
              type="button"
              onClick={() => setQuestion(s)}
              className="text-[11px] px-2.5 py-1 rounded-full bg-slate-100 dark:bg-slate-800/80 hover:bg-slate-200 dark:hover:bg-slate-700/80 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700/60 transition-colors"
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div role="alert" className="p-4 rounded-xl flex items-center gap-3 text-xs font-medium border bg-rose-50 dark:bg-rose-950/40 border-rose-200 dark:border-rose-500/30 text-rose-800 dark:text-rose-300">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          <span className="min-w-0 break-words">{error}</span>
        </div>
      )}

      {selectedHistoryId && <p className="break-words text-sm">History result from <strong>{historySource?.dataset || "an earlier dataset"}</strong>{historySource?.dataset_version != null ? ` · Version ${historySource.dataset_version}` : ''}</p>}
      {/* Query Result Section */}
      {currentResponse && (
        <div className="space-y-6">
          {/* Answer Card */}
          <div className="rounded-2xl border border-blue-200 dark:border-blue-500/30 bg-gradient-to-b from-blue-50/40 via-white to-white dark:from-slate-900/90 dark:to-slate-900/60 p-6 space-y-4 shadow-sm dark:shadow-xl">
            {/* Card Header & Toolbar */}
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 dark:border-slate-800 pb-3">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-lg bg-blue-50 dark:bg-blue-500/10 border border-blue-200 dark:border-blue-500/20 text-blue-600 dark:text-blue-400">
                  <Bot className="h-5 w-5" />
                </div>
                <div>
                  <span className="text-xs font-bold text-slate-900 dark:text-white uppercase tracking-wider block">
                    Analysis Findings & Intelligence
                  </span>
                  <span className="text-[11px] text-slate-500 dark:text-slate-400">
                    Question: "{currentResponse.question}"
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <span className="px-2.5 py-1 rounded bg-blue-50 dark:bg-blue-500/10 text-blue-700 dark:text-blue-400 border border-blue-200 dark:border-blue-500/20 text-xs font-mono">
                  Route: {currentResponse.route}
                </span>

                {currentResponse.model_used &&
                  currentResponse.model_used !== "none" && (
                    <span className="px-2.5 py-1 rounded text-xs font-medium border bg-purple-50 dark:bg-purple-500/10 text-purple-700 dark:text-purple-400 border-purple-200 dark:border-purple-500/20">
                      Model: {currentResponse.model_used}
                    </span>
                  )}
                {(!currentResponse.model_used ||
                  currentResponse.model_used === "none") && (
                  <span className="px-2.5 py-1 rounded text-xs font-medium border bg-rose-50 dark:bg-rose-500/10 text-rose-700 dark:text-rose-400 border-rose-200 dark:border-rose-500/20">
                    LLM Unavailable
                  </span>
                )}

                <button
                  onClick={handleCopyAnswer}
                  className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 text-xs font-medium border border-slate-200 dark:border-slate-700 transition-colors"
                  title="Copy formatted insights"
                >
                  {copiedAnswer ? (
                    <>
                      <Check className="h-3.5 w-3.5 text-emerald-500 dark:text-emerald-400" />
                      <span className="text-emerald-600 dark:text-emerald-400">
                        Copied
                      </span>
                    </>
                  ) : (
                    <>
                      <Copy className="h-3.5 w-3.5 text-slate-500 dark:text-slate-400" />
                      <span>Copy Insights</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Rich Formatted Markdown Answer & Executive KPI Cards */}
            <div className="pt-2">
              <MarkdownRenderer
                content={
                  currentResponse.insights ||
                  (currentResponse.execution_results?.[0]?.stdout
                    ? `### 📊 Execution Output\n\`\`\`text\n${currentResponse.execution_results[0].stdout}\n\`\`\``
                    : "### 🎯 Executive Summary\nAnalysis completed successfully.")
                }
                showKpiCards={true}
              />
            </div>

            {/* Render any generated Plotly figures from tool result */}
            {currentResponse.tool_result?.figure_spec && (
              <div className="pt-4 border-t border-slate-200 dark:border-slate-800/80">
                <div className="flex items-center gap-2 mb-2">
                  <BarChart3 className="h-4 w-4 text-blue-500 dark:text-blue-400" />
                  <span className="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider">
                    Interactive Projection Visual
                  </span>
                </div>
                <PlotlyChart
                  spec={currentResponse.tool_result.figure_spec}
                  height={420}
                  title={currentResponse.question}
                  sourcePage="AI Query"
                  source={chartSource}
                />
              </div>
            )}

            {/* Render figures from code execution */}
            {currentResponse.execution_results?.map((res, i) => (
              <div key={i} className="space-y-4 pt-2">
                {res.figures?.map((fig, figIdx) => (
                  <div
                    key={figIdx}
                    className="pt-4 border-t border-slate-200 dark:border-slate-800/80"
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <BarChart3 className="h-4 w-4 text-blue-500 dark:text-blue-400" />
                      <span className="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider">
                        Interactive Data Visualization
                      </span>
                    </div>
                    <PlotlyChart
                      spec={fig}
                      height={420}
                      title={currentResponse.question}
                      sourcePage="AI Query"
                      source={chartSource}
                    />
                  </div>
                ))}
              </div>
            ))}

            {/* Render Clean Deduplicated Generated Tables */}
            {cleanDataframes.length > 0 && (
              <div className="space-y-4 pt-4 border-t border-slate-200 dark:border-slate-800/80">
                {cleanDataframes.map((table, idx) => (
                  <div key={idx} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <p className="text-xs font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-2">
                        <TableIcon className="h-4 w-4 text-emerald-500 dark:text-emerald-400" />
                        {table.name === "result_df" ||
                        table.name === "sorted_df"
                          ? `Ranked Output Table (${table.rows.length} records)`
                          : table.name === "agg_stats"
                            ? `Aggregated Metrics Summary (${table.rows.length} records)`
                            : `Data Table: ${table.name} (${table.rows.length} records)`}
                      </p>
                    </div>
                    <DataTable data={table.rows} pageSize={10} />
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Generated Code Sandbox Blocks */}
          {currentResponse.code_blocks?.length > 0 && (
            <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 p-4 space-y-3 shadow-sm">
              <button
                onClick={() => setShowCode(!showCode)}
                className="w-full flex items-center justify-between text-xs text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 transition-colors"
              >
                <span className="flex items-center gap-2 font-bold text-slate-800 dark:text-slate-300">
                  <Terminal className="h-4 w-4 text-emerald-500 dark:text-emerald-400" />
                  Executed Python Analytics Code
                </span>
                <div className="flex items-center gap-3">
                  {currentResponse.execution_results?.[0]?.execution_time && (
                    <span className="flex items-center gap-1 font-mono text-[11px] text-slate-500 dark:text-slate-400">
                      <Clock className="h-3 w-3" />
                      {currentResponse.execution_results[0].execution_time}s
                    </span>
                  )}
                  {showCode ? (
                    <ChevronUp className="h-4 w-4" />
                  ) : (
                    <ChevronDown className="h-4 w-4" />
                  )}
                </div>
              </button>

              {showCode && (
                <div className="space-y-3 pt-2">
                  {currentResponse.code_blocks.map((code, idx) => (
                    <pre
                      key={idx}
                      className="p-4 rounded-xl bg-slate-900 dark:bg-[#080d1a] border border-slate-800 text-xs font-mono text-emerald-400 dark:text-emerald-300 overflow-x-auto leading-relaxed shadow-inner"
                    >
                      <code>{code}</code>
                    </pre>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Query History Drawer */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 p-5 space-y-4 shadow-sm">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold text-slate-900 dark:text-slate-200 flex items-center gap-2">
            <Clock className="h-4 w-4 text-blue-500 dark:text-blue-400" />
            Query History ({history.length})
          </h3>
          <div className="flex items-center gap-2">
            <a
              href={queryApi.getExportUrl()}
              download="query_history.csv"
              className="flex items-center gap-1 text-xs px-2.5 py-1 rounded-lg bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700 transition-colors font-medium"
            >
              <Download className="h-3 w-3" />
              Export CSV
            </a>
            {history.length > 0 && (
              <button
                onClick={handleClearHistory}
                className="p-2 rounded-lg hover:bg-rose-50 dark:hover:bg-rose-500/20 text-rose-700 dark:text-rose-300 transition-colors"
                title="Clear history"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        </div>

        {history.length === 0 ? (
          <div className="text-center py-6 text-xs text-slate-500 dark:text-slate-400">
            No query history yet.
          </div>
        ) : (
          <div className="space-y-2 max-h-80 overflow-y-auto">
            {history.map((item) => {
              const isSelected = selectedHistoryId === item.id;
              return (
                <div
                  key={item.id}
                  className={`p-3 rounded-xl border transition-all space-y-1 shadow-sm ${
                    isSelected
                      ? "bg-blue-50/80 dark:bg-blue-950/40 border-blue-400 dark:border-blue-500/50 ring-1 ring-blue-400/30"
                      : "bg-slate-50 dark:bg-slate-800/30 hover:bg-blue-50/50 dark:hover:bg-slate-800/70 border-slate-200 dark:border-slate-700/50"
                  }`}
                >
                  <div className="flex items-center justify-between text-xs">
                    <button type="button" aria-pressed={isSelected} onClick={() => handleSelectHistoryItem(item)} className="min-w-0 break-words text-left font-semibold text-slate-900 dark:text-white">
                      {isSelected && (
                        <span className="w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0" />
                      )}
                      "{item.question}"
                    </button>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      <span className="text-[10px] text-slate-500 dark:text-slate-400 font-mono">
                        {new Date(item.timestamp).toLocaleTimeString()}
                      </span>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleAsk(item.question);
                        }}
                        className="p-1 rounded-md hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                        title="Re-run this query"
                      >
                        <RotateCw className="h-3 w-3" />
                      </button>
                    </div>
                  </div>
                  <p className="text-[11px] text-slate-600 dark:text-slate-400 line-clamp-1">
                    {item.dataset} · {item.result_summary}
                  </p>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
