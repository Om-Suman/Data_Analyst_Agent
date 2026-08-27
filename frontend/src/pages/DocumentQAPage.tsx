import React, { useState } from 'react';
import { FileText, Send, BookOpen, Layers, AlertCircle, Bot } from 'lucide-react';
import { useDataset } from '../context/DatasetContext';
import { documentApi } from '../api/client';
import { DocumentQAResponse } from '../types';

export const DocumentQAPage: React.FC = () => {
  const { activeDataset, hasDataset } = useDataset();
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DocumentQAResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAsk = async () => {
    if (!question.trim() || !hasDataset) return;
    setLoading(true);
    setError(null);
    try {
      const res = await documentApi.askQA(question, activeDataset?.name);
      setResult(res);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to query document.');
    } finally {
      setLoading(false);
    }
  };

  if (!hasDataset || !activeDataset?.is_text) {
    return (
      <div className="p-8 text-center text-slate-500 border border-slate-800 rounded-xl bg-slate-900/30 space-y-3">
        <FileText className="h-10 w-10 mx-auto text-slate-600" />
        <p className="text-sm text-slate-300 font-semibold">No Document Loaded in Active Workspace</p>
        <p className="text-xs text-slate-500 max-w-md mx-auto">
          Please upload a document (.pdf, .docx, .txt, or OCR image) on the Upload page, or set a document as the active workspace.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
          <FileText className="h-5 w-5 text-amber-400" />
          Document Question Answering (RAG)
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Perform Retrieval-Augmented Generation (RAG) over '{activeDataset.name}' using vector embeddings and similarity indexing.
        </p>
      </div>

      {/* Question Input */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4 space-y-3">
        <div className="flex gap-2">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleAsk();
            }}
            placeholder="Ask a question about this document..."
            className="flex-1 bg-slate-950/80 border border-slate-700/80 rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-amber-500"
          />
          <button
            onClick={handleAsk}
            disabled={loading || !question.trim()}
            className="px-5 py-3 rounded-xl bg-amber-600 hover:bg-amber-500 disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold flex items-center gap-2 shadow-md shadow-amber-600/20 transition-all text-sm"
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <span className="h-4 w-4 rounded-full border-2 border-white/30 border-t-white animate-spin"></span>
                Searching...
              </span>
            ) : (
              <>
                <Send className="h-4 w-4" />
                Ask RAG
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl flex items-center gap-3 text-xs font-medium border bg-rose-950/40 border-rose-500/30 text-rose-300">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* RAG Answer */}
      {result && (
        <div className="space-y-6">
          <div className="rounded-xl border border-amber-500/30 bg-slate-900/60 p-6 space-y-4 shadow-xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <Bot className="h-5 w-5 text-amber-400" />
                <span className="text-xs font-bold text-white uppercase tracking-wider">Answer</span>
              </div>
              <span className="px-2.5 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20 text-xs font-mono">
                Engine: {result.engine}
              </span>
            </div>

            <div className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">{result.answer}</div>
          </div>

          {/* Retrieved Sources */}
          {result.sources?.length > 0 && (
            <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-3">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                <BookOpen className="h-4 w-4 text-blue-400" />
                Retrieved Context Chunks ({result.sources.length})
              </h3>
              <div className="space-y-2">
                {result.sources.map((src, i) => (
                  <div key={i} className="p-3 rounded-lg bg-slate-950/60 border border-slate-800 space-y-1">
                    <div className="flex items-center justify-between text-[11px] text-slate-400">
                      <span className="font-semibold text-slate-300">Source Chunk #{i + 1}</span>
                      <span className="font-mono text-emerald-400">Score: {(src.score * 100).toFixed(1)}%</span>
                    </div>
                    <p className="text-xs text-slate-300 font-mono leading-relaxed">{src.text}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
