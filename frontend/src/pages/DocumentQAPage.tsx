import React, { useState } from "react";
import {
  FileText,
  Send,
  BookOpen,
  Layers,
  AlertCircle,
  Bot,
  Copy,
  Check,
} from "lucide-react";
import { useDataset } from "../context/DatasetContext";
import { documentApi } from "../api/client";
import { DocumentQAResponse } from "../types";
import { MarkdownRenderer } from "../components/MarkdownRenderer";

export const DocumentQAPage: React.FC = () => {
  const { activeDataset, hasDataset } = useDataset();
  const [question, setQuestion] = useState("");
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
      setError(err.response?.data?.detail || "Failed to query document.");
    } finally {
      setLoading(false);
    }
  };

  if (!hasDataset || !activeDataset?.is_text) {
    return (
      <div className="p-8 text-center text-slate-500 dark:text-slate-400 border border-slate-200 dark:border-slate-800 rounded-2xl bg-white dark:bg-slate-900/30 space-y-3 shadow-sm">
        <FileText className="h-10 w-10 mx-auto text-amber-500/70 dark:text-slate-600" />
        <p className="text-sm text-slate-800 dark:text-slate-300 font-bold">
          No Document Loaded in Active Workspace
        </p>
        <p className="text-xs text-slate-500 dark:text-slate-400 max-w-md mx-auto">
          Please upload a document (.pdf, .docx, .txt, or OCR image) on the
          Upload page, or set a document as the active workspace.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white tracking-tight flex items-center gap-2">
          <FileText className="h-5 w-5 text-amber-500 dark:text-amber-400" />
          Document Question Answering (RAG)
        </h2>
        <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-1">
          Perform Retrieval-Augmented Generation (RAG) over '
          {activeDataset.name}' using vector embeddings and similarity indexing.
        </p>
      </div>

      {/* Question Input */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/50 p-4 sm:p-5 space-y-3 shadow-sm">
        <div className="flex gap-2">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleAsk();
            }}
            placeholder="Ask a question about this document..."
            className="flex-1 bg-slate-50 dark:bg-slate-950/80 border border-slate-300 dark:border-slate-700/80 rounded-xl px-4 py-3 text-sm text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:border-amber-500 focus:ring-2 focus:ring-amber-500/20 shadow-inner transition-all"
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
        <div className="p-4 rounded-xl flex items-center gap-3 text-xs font-medium border bg-rose-50 dark:bg-rose-950/40 border-rose-200 dark:border-rose-500/30 text-rose-800 dark:text-rose-300">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* RAG Answer */}
      {result && (
        <div className="space-y-6">
          <div className="rounded-2xl border border-amber-200 dark:border-amber-500/30 bg-gradient-to-b from-amber-50/40 via-white to-white dark:from-slate-900/90 dark:to-slate-900/60 p-6 space-y-4 shadow-sm dark:shadow-xl">
            <div className="flex items-center justify-between border-b border-slate-200 dark:border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <div className="p-2 rounded-lg bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/20 text-amber-600 dark:text-amber-400">
                  <Bot className="h-5 w-5" />
                </div>
                <span className="text-xs font-bold text-slate-900 dark:text-white uppercase tracking-wider">
                  Answer
                </span>
              </div>
              <span className="px-2.5 py-0.5 rounded bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border border-amber-200 dark:border-amber-500/20 text-xs font-mono">
                Engine: {result.engine}
              </span>
            </div>

            <div className="pt-1">
              <MarkdownRenderer content={result.answer} showKpiCards={false} />
            </div>
          </div>

          {/* Retrieved Sources */}
          {result.sources?.length > 0 && (
            <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 p-5 space-y-3 shadow-sm">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-400 flex items-center gap-2">
                <BookOpen className="h-4 w-4 text-blue-500 dark:text-blue-400" />
                Retrieved Context Chunks ({result.sources.length})
              </h3>
              <div className="space-y-2">
                {result.sources.map((src, i) => (
                  <div
                    key={i}
                    className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-950/60 border border-slate-200 dark:border-slate-800 space-y-1"
                  >
                    <div className="flex items-center justify-between text-[11px] text-slate-500 dark:text-slate-400">
                      <span className="font-semibold text-slate-800 dark:text-slate-300">
                        Source Chunk #{i + 1}
                      </span>
                      <span className="font-mono text-emerald-600 dark:text-emerald-400 font-medium">
                        Score: {(src.score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <p className="text-xs text-slate-700 dark:text-slate-300 font-mono leading-relaxed">
                      {src.text}
                    </p>
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
