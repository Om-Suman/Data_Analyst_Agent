import React, { useEffect, useState, useRef } from "react";
import { Link, useLocation } from "react-router-dom";
import {
  FileSpreadsheet,
  FileText,
  Database,
  Trash2,
  CheckCircle,
  AlertCircle,
  File,
  Layers,
} from "lucide-react";
import { useDataset } from "../context/DatasetContext";
import { datasetApi } from "../api/client";
import { uploadFeedback } from "../utils/uploadFeedback";

export const UploadPage: React.FC = () => {
  const { datasets, activeDatasetName, setActiveDataset, refreshDatasets } =
    useDataset();
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);
  const operationRef = useRef(false);
  const { hash } = useLocation();
  const samplesRef = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    if (hash !== "#samples") return;
    const frame = requestAnimationFrame(() => samplesRef.current?.focus());
    return () => cancelAnimationFrame(frame);
  }, [hash]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = async (files: FileList | File[]) => {
    if (!files || files.length === 0 || operationRef.current) return;
    const unsupported = Array.from(files).filter(
      (file) =>
        !/\.(csv|xlsx?|json|sqlite|db|pdf|docx|txt|png|jpe?g)$/i.test(
          file.name,
        ),
    );
    if (unsupported.length) {
      setMessage({
        type: "error",
        text: `Unsupported file type: ${unsupported.map((file) => file.name).join(", ")}. Choose one of the formats listed below.`,
      });
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }
    operationRef.current = true;
    setUploading(true);
    setMessage(null);
    try {
      const res = await datasetApi.uploadFiles(files);
      setMessage(uploadFeedback(res));
      await refreshDatasets();
    } catch (err: any) {
      setMessage({
        type: "error",
        text:
          err.response?.data?.detail ||
          "Could not upload the files. Check your connection and choose the files again to retry.",
      });
    } finally {
      operationRef.current = false;
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleSample = async (name: string) => {
    if (operationRef.current) return;
    operationRef.current = true;
    setUploading(true);
    setMessage(null);
    try {
      await datasetApi.loadSample(name);
      setMessage({ type: "success", text: `Loaded demo sample: ${name}!` });
      await refreshDatasets();
    } catch (err: any) {
      setMessage({
        type: "error",
        text:
          err.response?.data?.detail ||
          "Could not load the sample. Try again or choose a file to upload.",
      });
    } finally {
      operationRef.current = false;
      setUploading(false);
    }
  };

  const handleDelete = async (name: string) => {
    if (!window.confirm(`Are you sure you want to remove '${name}'?`)) return;
    try {
      await datasetApi.deleteDataset(name);
      await refreshDatasets();
    } catch (err: any) {
      alert(err.response?.data?.detail || "Failed to delete dataset.");
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white tracking-tight">
          Upload data
        </h1>
        <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-1">
          Add datasets to analyze or documents to ask questions about. For scanned PDFs, upload a searchable copy after OCR.
        </p>
      </div>

      {message && (
        <div
          role={message.type === "error" ? "alert" : "status"}
          className={`p-4 rounded-xl flex items-center gap-3 text-xs font-medium border ${
            message.type === "success"
              ? "bg-emerald-50 dark:bg-emerald-950/40 border-emerald-200 dark:border-emerald-500/30 text-emerald-800 dark:text-emerald-300"
              : "bg-rose-50 dark:bg-rose-950/40 border-rose-200 dark:border-rose-500/30 text-rose-800 dark:text-rose-300"
          }`}
        >
          {message.type === "success" ? (
            <CheckCircle className="h-4 w-4" />
          ) : (
            <AlertCircle className="h-4 w-4" />
          )}
          <span className="min-w-0 break-words">{message.text}</span>
        </div>
      )}

      {datasets.some((dataset) => dataset.name === activeDatasetName && dataset.is_text) && (
        <Link to="/document" className="inline-flex min-h-11 items-center gap-2 rounded-xl bg-blue-600 px-4 text-sm font-semibold text-white hover:bg-blue-700">
          <FileText className="h-4 w-4" aria-hidden="true" />
          Ask about this document
        </Link>
      )}

      {/* Drag and Drop Zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          if (e.dataTransfer.files) handleFiles(e.dataTransfer.files);
        }}
        className={`border-2 border-dashed rounded-2xl px-5 py-6 sm:py-8 text-center transition-colors ${
          isDragging
            ? "border-blue-500 bg-blue-50 dark:bg-blue-500/10"
            : "border-slate-300 dark:border-slate-800 bg-white dark:bg-slate-900/30 hover:border-blue-400 dark:hover:border-slate-700 hover:bg-blue-50/20 dark:hover:bg-slate-900/50 shadow-sm"
        }`}
      >
        <input
          aria-label="Choose data files"
          disabled={uploading}
          ref={fileInputRef}
          type="file"
          multiple
          accept=".csv,.xlsx,.xls,.json,.sqlite,.db,.pdf,.docx,.txt,.png,.jpg,.jpeg"
          className="hidden"
          onChange={(e) => {
            if (e.target.files) handleFiles(e.target.files);
          }}
        />
        <div className="flex flex-col items-center justify-center space-y-3">
          <button
            type="button"
            disabled={uploading}
            onClick={() => fileInputRef.current?.click()}
            className="min-h-11 px-5 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold disabled:opacity-50"
          >
            Choose files
          </button>
          <div className="space-y-1">
            <p className="text-sm sm:text-base font-semibold text-slate-900 dark:text-white">
              <span role="status">
                {uploading ? "Processing files…" : "Drag and drop files here"}
              </span>
            </p>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Supports CSV, Excel, JSON, SQLite, PDF, DOCX, TXT, and Images
            </p>
          </div>
        </div>
      </div>

      {/* Demo Sample Cards */}
      <div className="space-y-3">
        <h2 id="samples" ref={samplesRef} tabIndex={-1} className="scroll-mt-24 text-base font-semibold text-slate-900 dark:text-slate-200">
          Try a sample dataset
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <button
            type="button"
            disabled={uploading}
            onClick={() => handleSample("Sales Data")}
            className="text-left disabled:opacity-50 p-4 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 hover:border-blue-400 dark:hover:border-blue-500/40 hover:bg-blue-50/30 dark:hover:bg-slate-800/50 shadow-sm hover:shadow-md cursor-pointer transition-all space-y-2"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs sm:text-sm font-bold text-slate-900 dark:text-white">
                Sales & Revenue
              </span>
              <FileSpreadsheet className="h-4 w-4 text-blue-500 dark:text-blue-400" />
            </div>
            <p className="text-[11px] sm:text-xs text-slate-600 dark:text-slate-400">
              500 daily sales transactions across products and regions with
              profit margins.
            </p>
          </button>

          <button
            type="button"
            disabled={uploading}
            onClick={() => handleSample("Employee Data")}
            className="text-left disabled:opacity-50 p-4 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 hover:border-emerald-400 dark:hover:border-emerald-500/40 hover:bg-emerald-50/30 dark:hover:bg-slate-800/50 shadow-sm hover:shadow-md cursor-pointer transition-all space-y-2"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs sm:text-sm font-bold text-slate-900 dark:text-white">
                Employee HR
              </span>
              <Database className="h-4 w-4 text-emerald-500 dark:text-emerald-400" />
            </div>
            <p className="text-[11px] sm:text-xs text-slate-600 dark:text-slate-400">
              300 employee records with salaries, departments, experience, and
              remote status.
            </p>
          </button>

          <button
            type="button"
            disabled={uploading}
            onClick={() => handleSample("Finance Data")}
            className="text-left disabled:opacity-50 p-4 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 hover:border-amber-400 dark:hover:border-amber-500/40 hover:bg-amber-50/30 dark:hover:bg-slate-800/50 shadow-sm hover:shadow-md cursor-pointer transition-all space-y-2"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs sm:text-sm font-bold text-slate-900 dark:text-white">
                Finance & Stock
              </span>
              <FileSpreadsheet className="h-4 w-4 text-amber-500 dark:text-amber-400" />
            </div>
            <p className="text-[11px] sm:text-xs text-slate-600 dark:text-slate-400">
              365 daily stock price movements with volume, high, low, and
              closing values.
            </p>
          </button>
        </div>
      </div>

      {/* Dataset Management Table */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 p-5 sm:p-6 space-y-4 shadow-sm">
        <h2 className="text-sm font-bold text-slate-900 dark:text-slate-200 flex items-center gap-2">
          <Layers className="h-4 w-4 text-blue-500 dark:text-blue-400" />
          Workspace files ({datasets.length})
        </h2>

        {datasets.length === 0 ? (
          <div className="text-center py-8 text-xs text-slate-500 dark:text-slate-400">
            Your uploaded files and sample datasets will appear here.
          </div>
        ) : (
          <div role="region" aria-label="Loaded datasets table" tabIndex={0} className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50 dark:bg-slate-800/60 text-[11px] uppercase tracking-wider text-slate-600 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
                <tr>
                  <th className="px-4 py-3">Dataset Name</th>
                  <th className="px-4 py-3">Type</th>
                  <th className="px-4 py-3">Rows</th>
                  <th className="px-4 py-3">Cols</th>
                  <th className="px-4 py-3">Version</th>
                  <th className="px-4 py-3">Status</th>
                  <th className="px-4 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-800 text-slate-700 dark:text-slate-300">
                {datasets.map((d) => {
                  const isActive = d.name === activeDatasetName;
                  return (
                    <tr
                      key={d.name}
                      className="hover:bg-slate-50 dark:hover:bg-slate-800/30 transition-colors"
                    >
                      <td className="px-4 py-3 font-medium text-slate-900 dark:text-white flex items-center gap-2">
                        {d.is_text ? (
                          <FileText className="h-4 w-4 text-amber-500 dark:text-amber-400" />
                        ) : (
                          <File className="h-4 w-4 text-blue-500 dark:text-blue-400" />
                        )}
                        {d.name}
                      </td>
                      <td className="px-4 py-3 text-slate-500 dark:text-slate-400">
                        {d.is_text ? "Document / Text" : "Tabular Data"}
                      </td>
                      <td className="px-4 py-3 font-mono">
                        {d.rows > 0 ? d.rows.toLocaleString() : "—"}
                      </td>
                      <td className="px-4 py-3 font-mono">
                        {d.cols > 0 ? d.cols : "—"}
                      </td>
                      <td className="px-4 py-3 font-mono">v{d.version}</td>
                      <td className="px-4 py-3">
                        {isActive ? (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-500/20">
                            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500"></span>{" "}
                            Active
                          </span>
                        ) : (
                          <button
                            onClick={() => setActiveDataset(d.name)}
                            className="text-[11px] text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 font-semibold"
                          >
                            Set Active
                          </button>
                        )}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <button
                          onClick={() => handleDelete(d.name)}
                          className="p-2 rounded-lg hover:bg-rose-50 dark:hover:bg-rose-500/20 text-rose-700 dark:text-rose-300 transition-colors"
                          aria-label={`Delete ${d.name}`}
                          disabled={uploading}
                          title="Delete dataset"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
