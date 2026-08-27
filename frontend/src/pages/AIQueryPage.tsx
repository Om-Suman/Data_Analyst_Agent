import React, { useState, useEffect } from 'react';
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
} from 'lucide-react';
import { useDataset } from '../context/DatasetContext';
import { queryApi } from '../api/client';
import { QueryHistoryItem, QueryResponse } from '../types';
import { PlotlyChart } from '../components/PlotlyChart';
import { DataTable } from '../components/DataTable';

export const AIQueryPage: React.FC = () => {
  const { activeDataset, hasDataset } = useDataset();
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentResponse, setCurrentResponse] = useState<QueryResponse | null>(null);
  const [history, setHistory] = useState<QueryHistoryItem[]>([]);
  const [error, setError] = useState<string | null>(null);

  const promptSuggestions = [
    'What are the top 5 highest sales days?',
    'Plot a correlation heatmap between all numeric metrics',
    'Calculate the average and median values by category',
    'Forecast sales for the next 30 days',
    'Detect anomalies and outliers in our data',
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

  const handleAsk = async (qText?: string) => {
    const q = qText || question;
    if (!q.trim() || !hasDataset) return;
    setLoading(true);
    setError(null);
    try {
      const res = await queryApi.ask(q);
      setCurrentResponse(res);
      await fetchHistory();
      if (!qText) setQuestion('');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to process AI query.');
    } finally {
      setLoading(false);
    }
  };

  const handleClearHistory = async () => {
    if (!window.confirm('Clear all query history?')) return;
    try {
      await queryApi.clearHistory();
      setHistory([]);
    } catch (err) {
      console.error(err);
    }
  };

  if (!hasDataset) {
    return (
      <div className="p-8 text-center text-slate-500 border border-slate-800 rounded-xl bg-slate-900/30">
        Please load or select a dataset first to execute AI Data Queries.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-blue-400" />
          Natural Language AI Data Query
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Ask any question about your data in plain English. The agent routes questions, writes pandas/plotly code in a safe sandbox, and renders insights.
        </p>
      </div>

      {/* Query Input Box */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4 space-y-3 shadow-lg">
        <div className="flex gap-2">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleAsk();
            }}
            placeholder="Ask anything (e.g. 'Show total revenue by region and plot a bar chart')..."
            className="flex-1 bg-slate-950/80 border border-slate-700/80 rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-blue-500 shadow-inner"
          />
          <button
            onClick={() => handleAsk()}
            disabled={loading || !question.trim()}
            className="px-5 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold flex items-center gap-2 shadow-md shadow-blue-600/20 transition-all text-sm"
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
        </div>

        {/* Suggestion Pills */}
        <div className="flex flex-wrap items-center gap-2 pt-1">
          <span className="text-[11px] text-slate-500">Suggestions:</span>
          {promptSuggestions.map((s, idx) => (
            <button
              key={idx}
              onClick={() => handleAsk(s)}
              className="text-[11px] px-2.5 py-1 rounded-full bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 border border-slate-700/60 transition-colors"
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl flex items-center gap-3 text-xs font-medium border bg-rose-950/40 border-rose-500/30 text-rose-300">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Query Result Section */}
      {currentResponse && (
        <div className="space-y-6">
          {/* Answer Card */}
          <div className="rounded-xl border border-blue-500/30 bg-slate-900/60 p-6 space-y-4 shadow-xl">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <Bot className="h-5 w-5 text-blue-400" />
                <span className="text-xs font-bold text-white uppercase tracking-wider">Analysis Result</span>
              </div>
              <div className="flex items-center gap-2 text-xs font-mono">
                <span className="px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
                  Route: {currentResponse.route}
                </span>
                {currentResponse.model_used && (
                  <span className="px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20">
                    Model: {currentResponse.model_used}
                  </span>
                )}
              </div>
            </div>

            {/* Answer text */}
            <div className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
              {currentResponse.insights || 'Analysis completed successfully.'}
            </div>

            {/* Render any generated Plotly figures from tool result or execution */}
            {currentResponse.tool_result?.figure_spec && (
              <div className="pt-3">
                <PlotlyChart spec={currentResponse.tool_result.figure_spec} height={400} />
              </div>
            )}

            {/* Render figures and dataframes from code execution */}
            {currentResponse.execution_results?.map((res, i) => (
              <div key={i} className="space-y-4 pt-2">
                {res.figures?.map((fig, figIdx) => (
                  <PlotlyChart key={figIdx} spec={fig} height={400} />
                ))}

                {Object.entries(res.dataframes || {}).map(([name, rows]) => (
                  <div key={name} className="space-y-2">
                    <p className="text-xs font-semibold text-slate-300">Generated Table: {name}</p>
                    <DataTable data={rows} pageSize={10} />
                  </div>
                ))}
              </div>
            ))}
          </div>

          {/* Generated Code Sandbox Blocks */}
          {currentResponse.code_blocks?.length > 0 && (
            <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-5 space-y-3">
              <div className="flex items-center justify-between text-xs text-slate-400 border-b border-slate-800 pb-2">
                <span className="flex items-center gap-1.5 font-semibold text-slate-300">
                  <Terminal className="h-4 w-4 text-emerald-400" />
                  Executed Python Sandbox Code
                </span>
                {currentResponse.execution_results?.[0]?.execution_time && (
                  <span className="flex items-center gap-1 font-mono text-[11px] text-slate-400">
                    <Clock className="h-3 w-3" />
                    {currentResponse.execution_results[0].execution_time}s
                  </span>
                )}
              </div>
              <div className="space-y-3">
                {currentResponse.code_blocks.map((code, idx) => (
                  <pre
                    key={idx}
                    className="p-4 rounded-lg bg-[#0b0f19] border border-slate-800 text-xs font-mono text-emerald-300 overflow-x-auto"
                  >
                    <code>{code}</code>
                  </pre>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Query History Drawer */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
            <Clock className="h-4 w-4 text-blue-400" />
            Query History ({history.length})
          </h3>
          <div className="flex items-center gap-2">
            <a
              href={queryApi.getExportUrl()}
              download="query_history.csv"
              className="flex items-center gap-1 text-xs px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-colors"
            >
              <Download className="h-3 w-3" />
              Export CSV
            </a>
            {history.length > 0 && (
              <button
                onClick={handleClearHistory}
                className="p-1 rounded hover:bg-rose-500/20 text-slate-400 hover:text-rose-400 transition-colors"
                title="Clear history"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        </div>

        {history.length === 0 ? (
          <div className="text-center py-6 text-xs text-slate-500">No query history yet.</div>
        ) : (
          <div className="space-y-2 max-h-80 overflow-y-auto">
            {history.map((item) => (
              <div
                key={item.id}
                onClick={() => handleAsk(item.question)}
                className="p-3 rounded-lg bg-slate-800/30 hover:bg-slate-800/70 border border-slate-700/50 cursor-pointer transition-all space-y-1"
              >
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-white truncate max-w-md">"{item.question}"</span>
                  <span className="text-[10px] text-slate-500 font-mono">
                    {new Date(item.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 line-clamp-1">{item.result_summary}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
