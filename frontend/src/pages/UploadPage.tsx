import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  FileSpreadsheet,
  FileText,
  Database,
  Trash2,
  CheckCircle,
  AlertCircle,
  File,
  Layers,
} from 'lucide-react';
import { useDataset } from '../context/DatasetContext';
import { datasetApi } from '../api/client';

export const UploadPage: React.FC = () => {
  const { datasets, activeDatasetName, setActiveDataset, refreshDatasets } = useDataset();
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = async (files: FileList | File[]) => {
    if (!files || files.length === 0) return;
    setUploading(true);
    setMessage(null);
    try {
      const res = await datasetApi.uploadFiles(files);
      setMessage({
        type: 'success',
        text: `Successfully uploaded ${res.processed?.length || 1} file(s)!`,
      });
      await refreshDatasets();
    } catch (err: any) {
      setMessage({
        type: 'error',
        text: err.response?.data?.detail || 'Failed to upload files.',
      });
    } finally {
      setUploading(false);
    }
  };

  const handleSample = async (name: string) => {
    setMessage(null);
    try {
      await datasetApi.loadSample(name);
      setMessage({ type: 'success', text: `Loaded demo sample: ${name}!` });
      await refreshDatasets();
    } catch (err: any) {
      setMessage({ type: 'error', text: err.response?.data?.detail || 'Failed to load sample.' });
    }
  };

  const handleDelete = async (name: string) => {
    if (!window.confirm(`Are you sure you want to remove '${name}'?`)) return;
    try {
      await datasetApi.deleteDataset(name);
      await refreshDatasets();
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to delete dataset.');
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight">Upload & Ingest Data</h2>
        <p className="text-xs text-slate-400 mt-1">
          Upload tabular datasets (CSV, Excel, JSON, SQLite) or unstructured documents (PDF, DOCX, TXT, OCR Images)
        </p>
      </div>

      {message && (
        <div
          className={`p-4 rounded-xl flex items-center gap-3 text-xs font-medium border ${
            message.type === 'success'
              ? 'bg-emerald-950/40 border-emerald-500/30 text-emerald-300'
              : 'bg-rose-950/40 border-rose-500/30 text-rose-300'
          }`}
        >
          {message.type === 'success' ? <CheckCircle className="h-4 w-4" /> : <AlertCircle className="h-4 w-4" />}
          <span>{message.text}</span>
        </div>
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
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all ${
          isDragging
            ? 'border-blue-500 bg-blue-500/10'
            : 'border-slate-800 bg-slate-900/30 hover:border-slate-700 hover:bg-slate-900/50'
        }`}
      >
        <input
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
          <div className="h-14 w-14 rounded-2xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400">
            <UploadCloud className="h-7 w-7" />
          </div>
          <div className="space-y-1">
            <p className="text-sm font-semibold text-white">
              {uploading ? 'Processing uploaded files...' : 'Click to upload or drag & drop files here'}
            </p>
            <p className="text-xs text-slate-400">Supports CSV, Excel, JSON, SQLite, PDF, DOCX, TXT, and Images</p>
          </div>
        </div>
      </div>

      {/* Demo Sample Cards */}
      <div className="space-y-3">
        <h3 className="text-sm font-semibold text-slate-300">Or Start with Pre-Loaded Demo Datasets</h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div
            onClick={() => handleSample('Sales Data')}
            className="p-4 rounded-xl border border-slate-800 bg-slate-900/40 hover:border-blue-500/40 hover:bg-slate-800/50 cursor-pointer transition-all space-y-2"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-white">Sales & Revenue</span>
              <FileSpreadsheet className="h-4 w-4 text-blue-400" />
            </div>
            <p className="text-[11px] text-slate-400">500 daily sales transactions across products and regions with profit margins.</p>
          </div>

          <div
            onClick={() => handleSample('Employee Data')}
            className="p-4 rounded-xl border border-slate-800 bg-slate-900/40 hover:border-emerald-500/40 hover:bg-slate-800/50 cursor-pointer transition-all space-y-2"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-white">Employee HR</span>
              <Database className="h-4 w-4 text-emerald-400" />
            </div>
            <p className="text-[11px] text-slate-400">300 employee records with salaries, departments, experience, and remote status.</p>
          </div>

          <div
            onClick={() => handleSample('Finance Data')}
            className="p-4 rounded-xl border border-slate-800 bg-slate-900/40 hover:border-amber-500/40 hover:bg-slate-800/50 cursor-pointer transition-all space-y-2"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-white">Finance & Stock</span>
              <FileSpreadsheet className="h-4 w-4 text-amber-400" />
            </div>
            <p className="text-[11px] text-slate-400">365 daily stock price movements with volume, high, low, and closing values.</p>
          </div>
        </div>
      </div>

      {/* Dataset Management Table */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
        <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
          <Layers className="h-4 w-4 text-blue-400" />
          Loaded Datasets in Workspace ({datasets.length})
        </h3>

        {datasets.length === 0 ? (
          <div className="text-center py-8 text-xs text-slate-500">No datasets currently loaded in memory.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-slate-800/60 text-[11px] uppercase tracking-wider text-slate-400">
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
              <tbody className="divide-y divide-slate-800">
                {datasets.map((d) => {
                  const isActive = d.name === activeDatasetName;
                  return (
                    <tr key={d.name} className="hover:bg-slate-800/30 transition-colors">
                      <td className="px-4 py-3 font-medium text-white flex items-center gap-2">
                        {d.is_text ? <FileText className="h-4 w-4 text-amber-400" /> : <File className="h-4 w-4 text-blue-400" />}
                        {d.name}
                      </td>
                      <td className="px-4 py-3 text-slate-400">{d.is_text ? 'Document / Text' : 'Tabular Data'}</td>
                      <td className="px-4 py-3 font-mono">{d.rows > 0 ? d.rows.toLocaleString() : '—'}</td>
                      <td className="px-4 py-3 font-mono">{d.cols > 0 ? d.cols : '—'}</td>
                      <td className="px-4 py-3 font-mono">v{d.version}</td>
                      <td className="px-4 py-3">
                        {isActive ? (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400"></span> Active
                          </span>
                        ) : (
                          <button
                            onClick={() => setActiveDataset(d.name)}
                            className="text-[11px] text-blue-400 hover:text-blue-300 font-medium"
                          >
                            Set Active
                          </button>
                        )}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <button
                          onClick={() => handleDelete(d.name)}
                          className="p-1.5 rounded hover:bg-rose-500/20 text-slate-400 hover:text-rose-400 transition-colors"
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
