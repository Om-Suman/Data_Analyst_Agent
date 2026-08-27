import React, { useState, useEffect } from 'react';
import {
  Settings,
  Key,
  Cpu,
  Sliders,
  RotateCcw,
  CheckCircle2,
  AlertCircle,
  Eye,
  EyeOff,
  Save,
  Shield,
} from 'lucide-react';
import { useDataset } from '../context/DatasetContext';
import { configApi } from '../api/client';

export const SettingsPage: React.FC = () => {
  const { config, refreshConfig, refreshDatasets } = useDataset();

  const [apiKey, setApiKey] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);
  const [primaryModel, setPrimaryModel] = useState('');
  const [fallbackModel, setFallbackModel] = useState('');
  const [maxTokens, setMaxTokens] = useState(2048);
  const [temperature, setTemperature] = useState(0.3);
  const [defaultOutlier, setDefaultOutlier] = useState('none');
  const [defaultMissingThreshold, setDefaultMissingThreshold] = useState(50);

  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const modelOptions = [
    { label: 'DeepSeek R1 (Recommended)', value: 'deepseek-ai/DeepSeek-R1' },
    { label: 'Qwen 2.5 Coder 32B', value: 'Qwen/Qwen2.5-Coder-32B-Instruct' },
    { label: 'Llama 3.3 70B Instruct', value: 'meta-llama/Llama-3.3-70B-Instruct' },
    { label: 'Mistral 7B Instruct', value: 'mistralai/Mistral-7B-Instruct-v0.3' },
  ];

  useEffect(() => {
    if (config) {
      setPrimaryModel(config.primary_model || 'deepseek-ai/DeepSeek-R1');
      setFallbackModel(config.fallback_model || '');
      setMaxTokens(config.max_tokens || 2048);
      setTemperature(config.temperature ?? 0.3);
      setDefaultOutlier(config.default_outlier || 'none');
      setDefaultMissingThreshold(config.default_missing_threshold || 50);
    }
  }, [config]);

  const handleSave = async () => {
    setSaving(true);
    setMessage(null);
    try {
      await configApi.updateConfig({
        hf_api_key: apiKey.trim() ? apiKey.trim() : undefined,
        primary_model: primaryModel,
        fallback_model: fallbackModel,
        max_tokens: maxTokens,
        temperature: temperature,
        default_outlier: defaultOutlier,
        default_missing_threshold: defaultMissingThreshold,
      });
      setMessage({ type: 'success', text: 'Configuration and API settings saved successfully!' });
      setApiKey('');
      await refreshConfig();
    } catch (err: any) {
      setMessage({ type: 'error', text: err.response?.data?.detail || 'Failed to update configuration.' });
    } finally {
      setSaving(false);
    }
  };

  const handleResetSession = async () => {
    if (!window.confirm('Reset session? All active datasets, query history, and state in this browser will be cleared.')) return;
    try {
      await configApi.resetSession();
      await refreshDatasets();
      setMessage({ type: 'success', text: 'Session reset successfully.' });
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
          <Settings className="h-5 w-5 text-blue-400" />
          Hugging Face & System Settings
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Manage Hugging Face inference tokens, LLM model routing, inference parameters, and session state
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
          {message.type === 'success' ? <CheckCircle2 className="h-4 w-4" /> : <AlertCircle className="h-4 w-4" />}
          <span>{message.text}</span>
        </div>
      )}

      {/* Hugging Face API Key Card */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Key className="h-4 w-4 text-amber-400" />
            <h3 className="text-sm font-semibold text-slate-200">Hugging Face API Token</h3>
          </div>
          <span className="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700 font-mono">
            {config?.has_api_key ? `Configured (${config.api_key_masked})` : 'Not Configured'}
          </span>
        </div>

        <div className="space-y-2 text-xs">
          <label className="block text-slate-300 font-medium">Update API Token</label>
          <div className="relative">
            <input
              type={showApiKey ? 'text' : 'password'}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={config?.has_api_key ? 'Enter new key to overwrite...' : 'hf_...'}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg pl-3 pr-10 py-2.5 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-blue-500 font-mono"
            />
            <button
              type="button"
              onClick={() => setShowApiKey(!showApiKey)}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
            >
              {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </button>
          </div>
          <p className="text-[11px] text-slate-500 flex items-center gap-1.5 pt-1">
            <Shield className="h-3.5 w-3.5 text-emerald-400 flex-shrink-0" />
            API tokens are stored securely in backend session memory and are never exposed to the frontend.
          </p>
        </div>
      </div>

      {/* Model Selection Card */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
        <div className="flex items-center gap-2 pb-2 border-b border-slate-800">
          <Cpu className="h-4 w-4 text-purple-400" />
          <h3 className="text-sm font-semibold text-slate-200">LLM Model Configuration</h3>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
          <div>
            <label className="block text-slate-300 font-medium mb-1">Primary LLM Model</label>
            <select
              value={primaryModel}
              onChange={(e) => setPrimaryModel(e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-200 mb-2"
            >
              {modelOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            <input
              type="text"
              value={primaryModel}
              onChange={(e) => setPrimaryModel(e.target.value)}
              placeholder="Or enter custom HF model identifier..."
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-300 font-mono"
            />
          </div>

          <div>
            <label className="block text-slate-300 font-medium mb-1">Fallback LLM Model (Optional)</label>
            <select
              value={fallbackModel}
              onChange={(e) => setFallbackModel(e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-200 mb-2"
            >
              <option value="">None</option>
              {modelOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            <input
              type="text"
              value={fallbackModel}
              onChange={(e) => setFallbackModel(e.target.value)}
              placeholder="Or custom fallback model..."
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-300 font-mono"
            />
          </div>
        </div>
      </div>

      {/* Inference Parameters */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
        <div className="flex items-center gap-2 pb-2 border-b border-slate-800">
          <Sliders className="h-4 w-4 text-blue-400" />
          <h3 className="text-sm font-semibold text-slate-200">Inference Hyperparameters</h3>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
          <div>
            <label className="block text-slate-300 font-medium mb-1">Max Output Tokens: {maxTokens}</label>
            <input
              type="range"
              min="512"
              max="4096"
              step="256"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          <div>
            <label className="block text-slate-300 font-medium mb-1">Temperature: {temperature}</label>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.05"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex items-center justify-between pt-2">
        <button
          onClick={handleResetSession}
          className="px-4 py-2.5 rounded-xl bg-slate-900 hover:bg-rose-950/40 text-slate-400 hover:text-rose-300 border border-slate-800 hover:border-rose-500/30 text-xs font-semibold flex items-center gap-2 transition-all"
        >
          <RotateCcw className="h-4 w-4" />
          Reset Workspace Session
        </button>

        <button
          onClick={handleSave}
          disabled={saving}
          className="px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold flex items-center gap-2 shadow-md shadow-blue-600/20 transition-all"
        >
          <Save className="h-4 w-4" />
          {saving ? 'Saving...' : 'Save Settings'}
        </button>
      </div>
    </div>
  );
};
