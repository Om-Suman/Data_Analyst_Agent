import React, { useState } from 'react';
import {
  Sparkles,
  X,
  Send,
  Code2,
  Table,
  BarChart3,
  Bot,
  Compass,
  AlertCircle,
  Copy,
  Check,
} from 'lucide-react';
import { useDataset } from '../context/DatasetContext';
import { queryApi } from '../api/client';
import { QueryResponse } from '../types';
import { PlotlyChart } from './PlotlyChart';
import { DataTable } from './DataTable';
import { useToast } from './Toast';

interface AICopilotDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

export const AICopilotDrawer: React.FC<AICopilotDrawerProps> = ({ isOpen, onClose }) => {
  const { hasDataset, activeDataset } = useDataset();
  const { success } = useToast();

  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [conversation, setConversation] = useState<{ q: string; res: QueryResponse }[]>([]);
  const [copiedCodeIdx, setCopiedCodeIdx] = useState<number | null>(null);

  const samplePrompts = [
    'Summary statistics and key drivers',
    'Which categories generate the highest revenue?',
    'Detect top 5 anomalies in the dataset',
    'Forecast the primary metric for the next 30 periods',
  ];

  const handleSend = async (textToSend?: string) => {
    const q = (textToSend || question).trim();
    if (!q || !hasDataset || loading) return;

    setLoading(true);
    try {
      const res = await queryApi.ask(q);
      setConversation((prev) => [...prev, { q, res }]);
      setQuestion('');
    } catch (err: any) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleCopyCode = (code: string, idx: number) => {
    navigator.clipboard.writeText(code);
    setCopiedCodeIdx(idx);
    success('Code Copied', 'Python code copied to clipboard.');
    setTimeout(() => setCopiedCodeIdx(null), 2000);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-y-0 right-0 z-40 w-full sm:w-[480px] lg:w-[540px] bg-white dark:bg-[#0f172a] shadow-2xl border-l border-slate-200 dark:border-slate-800 flex flex-col animate-in slide-in-from-right duration-300">
      {/* Drawer Header */}
      <div className="p-4 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between bg-slate-50/80 dark:bg-slate-900/60 backdrop-blur-md">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-600 text-white shadow-md shadow-blue-500/20">
            <Sparkles className="h-5 w-5" />
          </div>
          <div>
            <h3 className="font-bold text-sm text-slate-900 dark:text-white">AI Data Copilot</h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              {hasDataset ? `Context: ${activeDataset?.name}` : 'No dataset active'}
            </p>
          </div>
        </div>

        <button
          onClick={onClose}
          className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-800 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-500 transition-colors"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Messages List Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {conversation.length === 0 ? (
          <div className="h-full flex flex-col justify-center items-center text-center p-6 space-y-4 text-slate-500 dark:text-slate-400">
            <Bot className="h-12 w-12 text-blue-500 opacity-60" />
            <div className="space-y-1">
              <p className="font-bold text-slate-800 dark:text-slate-200 text-sm">How can I assist your analysis?</p>
              <p className="text-xs leading-relaxed max-w-xs">
                Ask questions in plain English. I'll route queries, execute Pandas/Plotly code, and explain statistical insights.
              </p>
            </div>

            {hasDataset && (
              <div className="w-full space-y-2 pt-2 text-left">
                <p className="text-[11px] font-bold uppercase tracking-wider text-slate-400">Try asking:</p>
                {samplePrompts.map((prompt, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSend(prompt)}
                    className="w-full text-left p-2.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/40 hover:border-blue-500 text-xs text-slate-700 dark:text-slate-300 font-medium transition-all"
                  >
                    "{prompt}"
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          conversation.map((item, idx) => (
            <div key={idx} className="space-y-3">
              {/* User Bubble */}
              <div className="flex justify-end">
                <div className="max-w-[85%] rounded-2xl rounded-tr-sm bg-blue-600 text-white px-4 py-2.5 text-xs sm:text-sm font-medium shadow-md shadow-blue-600/10">
                  {item.q}
                </div>
              </div>

              {/* Copilot Response Bubble */}
              <div className="rounded-2xl rounded-tl-sm border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/60 p-4 space-y-3 text-xs sm:text-sm">
                {/* Route Pill */}
                <div className="flex items-center gap-2 text-xs">
                  <span className="px-2 py-0.5 rounded-md bg-blue-100 dark:bg-blue-950/80 text-blue-700 dark:text-blue-300 font-mono font-bold text-[11px]">
                    {item.res.route}
                  </span>
                  <span className="text-slate-400 text-[11px]">
                    via {item.res.routing_source}
                  </span>
                </div>

                {/* Explanation text */}
                {item.res.insights && (
                  <p className="text-slate-800 dark:text-slate-200 leading-relaxed whitespace-pre-wrap">
                    {item.res.insights}
                  </p>
                )}

                {/* Plotly Chart Spec if present */}
                {item.res.execution_results?.some((e) => e.figures && e.figures.length > 0) && (
                  <div className="mt-2 rounded-xl overflow-hidden border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-2">
                    {item.res.execution_results.map((e, eIdx) =>
                      e.figures.map((fig, fIdx) => (
                        <PlotlyChart key={`${eIdx}-${fIdx}`} spec={fig} height={280} title={item.q} sourcePage="AI Copilot" />
                      ))
                    )}
                  </div>
                )}

                {/* Code Block Accordion */}
                {item.res.code_blocks && item.res.code_blocks.length > 0 && (
                  <div className="mt-2 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-900 overflow-hidden text-xs">
                    <div className="px-3 py-1.5 bg-slate-950 border-b border-slate-800 flex items-center justify-between text-slate-400">
                      <span className="font-mono text-[11px] flex items-center gap-1.5">
                        <Code2 className="h-3.5 w-3.5 text-blue-400" />
                        Executed Python Sandbox
                      </span>
                      <button
                        onClick={() => handleCopyCode(item.res.code_blocks.join('\n\n'), idx)}
                        className="flex items-center gap-1 hover:text-white transition-colors"
                      >
                        {copiedCodeIdx === idx ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
                        {copiedCodeIdx === idx ? 'Copied' : 'Copy'}
                      </button>
                    </div>
                    <pre className="p-3 text-slate-200 font-mono text-[11px] overflow-x-auto max-h-40">
                      {item.res.code_blocks.join('\n\n')}
                    </pre>
                  </div>
                )}
              </div>
            </div>
          ))
        )}

        {loading && (
          <div className="flex items-center gap-3 p-4 rounded-2xl bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 text-xs text-slate-500 animate-pulse">
            <Sparkles className="h-4 w-4 text-blue-500 animate-spin" />
            <span>Analyzing dataset and computing response...</span>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-3 sm:p-4 border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/90 backdrop-blur-md">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
          className="flex items-center gap-2"
        >
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={!hasDataset || loading}
            placeholder={hasDataset ? 'Ask a question or request a chart...' : 'Load a dataset first...'}
            className="flex-1 bg-slate-100 dark:bg-slate-950 border border-slate-300 dark:border-slate-700/80 rounded-xl px-4 py-2.5 text-xs sm:text-sm text-slate-900 dark:text-slate-100 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
          />
          <button
            type="submit"
            disabled={!question.trim() || !hasDataset || loading}
            className="p-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white font-semibold transition-all shadow-md shadow-blue-600/20"
          >
            <Send className="h-4 w-4" />
          </button>
        </form>
      </div>
    </div>
  );
};
