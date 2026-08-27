import React from 'react';
import { Database, Key, Plus, RefreshCw } from 'lucide-react';
import { useDataset } from '../context/DatasetContext';
import { datasetApi } from '../api/client';

export const Navbar: React.FC = () => {
  const { datasets, activeDatasetName, setActiveDataset, refreshDatasets, config } = useDataset();

  const handleSample = async (name: string) => {
    try {
      await datasetApi.loadSample(name);
      await refreshDatasets();
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <header className="h-16 border-b border-slate-800 bg-[#0d1322]/80 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-20">
      <div className="flex items-center gap-4">
        {/* Dataset Dropdown */}
        <div className="flex items-center gap-2">
          <Database className="h-4 w-4 text-blue-400" />
          <select
            value={activeDatasetName || ''}
            onChange={(e) => {
              if (e.target.value) setActiveDataset(e.target.value);
            }}
            className="bg-slate-900 border border-slate-700 text-xs font-medium text-slate-200 rounded-lg px-3 py-1.5 focus:outline-none focus:border-blue-500 max-w-xs"
          >
            {datasets.length === 0 ? (
              <option value="">No datasets available</option>
            ) : (
              datasets.map((d) => (
                <option key={d.name} value={d.name}>
                  {d.name} ({d.rows > 0 ? `${d.rows.toLocaleString()} rows` : 'text'})
                </option>
              ))
            )}
          </select>
        </div>

        {/* Quick Sample Buttons */}
        <div className="hidden lg:flex items-center gap-1.5 pl-3 border-l border-slate-800">
          <span className="text-[11px] text-slate-500 mr-1">Load Demo:</span>
          {['Sales Data', 'Employee Data', 'Finance Data'].map((sName) => (
            <button
              key={sName}
              onClick={() => handleSample(sName)}
              className="text-[11px] px-2.5 py-1 rounded bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-colors"
            >
              {sName.split(' ')[0]}
            </button>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-3">
        {/* API Key Status Pill */}
        <div className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border bg-slate-900/60 border-slate-800">
          <Key className="h-3 w-3 text-amber-400" />
          <span className="text-slate-400">HF API:</span>
          {config?.has_api_key ? (
            <span className="text-emerald-400 flex items-center gap-1 font-mono text-[11px]">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400"></span>
              {config.api_key_masked}
            </span>
          ) : (
            <span className="text-amber-400 font-mono text-[11px]">Not configured</span>
          )}
        </div>

        {/* Refresh Button */}
        <button
          onClick={() => refreshDatasets()}
          title="Refresh datasets"
          className="p-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
        >
          <RefreshCw className="h-4 w-4" />
        </button>
      </div>
    </header>
  );
};
