import React, { useState, useEffect } from "react";
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
} from "lucide-react";
import { useDataset } from "../context/DatasetContext";
import { configApi } from "../api/client";

export const SettingsPage: React.FC = () => {
  const { config, refreshConfig, refreshDatasets, refreshPinnedCharts } = useDataset();

  const [apiKey, setApiKey] = useState("");
  const [showApiKey, setShowApiKey] = useState(false);
  const [primaryModel, setPrimaryModel] = useState("");
  const [fallbackModel, setFallbackModel] = useState("");
  const [maxTokens, setMaxTokens] = useState(2048);
  const [temperature, setTemperature] = useState(0.3);
  const [defaultOutlier, setDefaultOutlier] = useState("none");
  const [defaultMissingThreshold, setDefaultMissingThreshold] = useState(50);

  const [saving, setSaving] = useState(false);
  const [checking, setChecking] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [message, setMessage] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);

  const modelOptions = [
    { label: "Qwen 3.8 27B (Recommended)", value: "qwen/qwen3.8-27b" },
    { label: "GPT-OSS 120B", value: "openai/gpt-oss-120b" },
    { label: "GPT-OSS 20B", value: "openai/gpt-oss-20b" },
    { label: "Allam 2 7B", value: "allam-2-7b" },
  ];

  useEffect(() => {
    if (config) {
      setPrimaryModel(config.primary_model || "qwen/qwen3.8-27b");
      setFallbackModel(config.fallback_model || "openai/gpt-oss-120b");
      setMaxTokens(config.max_tokens || 2048);
      setTemperature(config.temperature ?? 0.3);
      setDefaultOutlier(config.default_outlier || "none");
      setDefaultMissingThreshold(config.default_missing_threshold || 50);
    }
  }, [config]);

  const handleSave = async () => {
    setSaving(true);
    setMessage(null);
    try {
      await configApi.updateConfig({
        groq_api_key: apiKey.trim() ? apiKey.trim() : undefined,
        primary_model: primaryModel,
        fallback_model: fallbackModel,
        max_tokens: maxTokens,
        temperature: temperature,
        default_outlier: defaultOutlier,
        default_missing_threshold: defaultMissingThreshold,
      });
      setMessage({
        type: "success",
        text: "Configuration and API settings saved successfully!",
      });
      setApiKey("");
      await refreshConfig();
    } catch (err: any) {
      setMessage({
        type: "error",
        text: err.response?.data?.detail || "Failed to update configuration.",
      });
    } finally {
      setSaving(false);
    }
  };

  const handleResetSession = async () => {
    if (
      !window.confirm(
        "Reset workspace? This clears uploaded datasets, query history, and pinned charts from this session.",
      )
    )
      return;
    setResetting(true);
    try {
      await configApi.resetSession();
      await Promise.all([refreshDatasets(), refreshConfig(), refreshPinnedCharts()]);
      setMessage({ type: "success", text: "Session reset successfully." });
    } catch (err) {
      setMessage({ type: "error", text: "Could not reset the workspace. Try again." });
    } finally {
      setResetting(false);
    }
  };

  const handleCheckConnection = async () => {
    setChecking(true);
    setMessage(null);
    try {
      const { unavailable_models: unavailable } = await configApi.checkConnection();
      setMessage(unavailable.length
        ? { type: "error", text: `Connected to Groq. Unavailable saved models: ${unavailable.join(", ")}.` }
        : { type: "success", text: "Connected to Groq. Your saved models are available." });
    } catch (err: any) {
      setMessage({ type: "error", text: err.response?.data?.detail || "Could not check the connection." });
    } finally {
      setChecking(false);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h2 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white tracking-tight flex items-center gap-2">
          <Settings className="h-5 w-5 text-blue-500 dark:text-blue-400" />
          Groq & System Settings
        </h2>
        <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-1">
          Manage Groq API keys, LLM model routing, inference parameters, and session state
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
            <CheckCircle2 className="h-4 w-4" />
          ) : (
            <AlertCircle className="h-4 w-4" />
          )}
          <span className="min-w-0 break-words">{message.text}</span>
        </div>
      )}

      {/* Groq API Key Card */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 p-5 sm:p-6 space-y-4 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-200 dark:border-slate-800">
          <div className="flex items-center gap-2">
            <Key className="h-4 w-4 text-amber-500 dark:text-amber-400" />
            <h3 className="text-sm font-bold text-slate-900 dark:text-slate-200">
              Groq API Key
            </h3>
          </div>
          <span className="text-[11px] px-2.5 py-0.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700 font-mono font-medium">
            {config?.has_api_key
              ? "Key saved"
              : "No key saved"}
          </span>
        </div>

        <div className="space-y-2 text-xs">
          <label htmlFor="groq-key" className="block text-slate-700 dark:text-slate-300 font-medium">
            Update API Key
          </label>
          <div className="relative">
            <input
              id="groq-key"
              type={showApiKey ? "text" : "password"}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={
                config?.has_api_key ? "Enter new key to overwrite..." : "gsk_..."
              }
              className="w-full bg-slate-50 dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded-xl pl-3.5 pr-10 py-2.5 text-xs text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 font-mono shadow-inner transition-all"
            />
            <button
              type="button"
              aria-label={showApiKey ? "Hide API key" : "Show API key"}
              onClick={() => setShowApiKey(!showApiKey)}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors"
            >
              {showApiKey ? (
                <EyeOff className="h-4 w-4" />
              ) : (
                <Eye className="h-4 w-4" />
              )}
            </button>
          </div>
          <p className="text-[11px] text-slate-500 dark:text-slate-400 flex items-center gap-1.5 pt-1 font-medium">
            <Shield className="h-3.5 w-3.5 text-emerald-500 dark:text-emerald-400 flex-shrink-0" />
            API tokens are stored securely in backend session memory and are
            never exposed to the frontend.
          </p>
          <button type="button" disabled={saving || checking || resetting || !config?.has_api_key} onClick={handleCheckConnection} className="min-h-11 rounded-lg border px-3 font-semibold disabled:opacity-50">
            {checking ? "Checking connection…" : "Test saved connection"}
          </button>
        </div>
      </div>

      {/* Model Selection Card */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 p-5 sm:p-6 space-y-4 shadow-sm">
        <div className="flex items-center gap-2 pb-3 border-b border-slate-200 dark:border-slate-800">
          <Cpu className="h-4 w-4 text-purple-500 dark:text-purple-400" />
          <h3 className="text-sm font-bold text-slate-900 dark:text-slate-200">
            LLM Model Configuration
          </h3>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
          <div>
            <label className="block text-slate-700 dark:text-slate-300 font-medium mb-1.5">
              Primary LLM Model
            </label>
            <select
              aria-label="Primary model preset"
              value={primaryModel}
              onChange={(e) => setPrimaryModel(e.target.value)}
              className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 text-slate-900 dark:text-slate-200 mb-2 shadow-sm focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all"
            >
              {modelOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            <input
              aria-label="Primary model ID"
              type="text"
              value={primaryModel}
              onChange={(e) => setPrimaryModel(e.target.value)}
              placeholder="Or enter custom Groq model identifier..."
              className="w-full bg-slate-50 dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 text-xs text-slate-900 dark:text-slate-200 font-mono shadow-inner focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all placeholder-slate-400 dark:placeholder-slate-500"
            />
          </div>

          <div>
            <label className="block text-slate-700 dark:text-slate-300 font-medium mb-1.5">
              Fallback LLM Model (Optional)
            </label>
            <select
              aria-label="Fallback model preset"
              value={fallbackModel}
              onChange={(e) => setFallbackModel(e.target.value)}
              className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 text-slate-900 dark:text-slate-200 mb-2 shadow-sm focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all"
            >
              <option value="">None</option>
              {modelOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            <input
              aria-label="Fallback model ID"
              type="text"
              value={fallbackModel}
              onChange={(e) => setFallbackModel(e.target.value)}
              placeholder="Or custom fallback model..."
              className="w-full bg-slate-50 dark:bg-slate-950 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 text-xs text-slate-900 dark:text-slate-200 font-mono shadow-inner focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all placeholder-slate-400 dark:placeholder-slate-500"
            />
          </div>
        </div>
      </div>

      {/* Inference Parameters */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40 p-5 sm:p-6 space-y-4 shadow-sm">
        <div className="flex items-center gap-2 pb-3 border-b border-slate-200 dark:border-slate-800">
          <Sliders className="h-4 w-4 text-blue-500 dark:text-blue-400" />
          <h3 className="text-sm font-bold text-slate-900 dark:text-slate-200">
            Inference Hyperparameters
          </h3>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
          <div>
            <label className="block text-slate-700 dark:text-slate-300 font-medium mb-1.5">
              Max Output Tokens: {maxTokens}
            </label>
            <input
              aria-label="Maximum output tokens"
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
            <label className="block text-slate-700 dark:text-slate-300 font-medium mb-1.5">
              Temperature: {temperature}
            </label>
            <input
              aria-label="Temperature"
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

      <div className="flex justify-end pt-2">
        <button
          onClick={handleSave}
          disabled={saving || checking || resetting}
          className="px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold flex items-center gap-2 shadow-md shadow-blue-600/20 transition-all"
        >
          <Save className="h-4 w-4" />
          {saving ? "Saving..." : "Save Settings"}
        </button>
      </div>
      <section className="space-y-3 border-t border-slate-200 pt-5 dark:border-slate-700" aria-labelledby="workspace-reset">
        <h3 id="workspace-reset" className="font-semibold">Reset workspace</h3>
        <p className="text-sm text-slate-600 dark:text-slate-300">This clears uploaded datasets, query history, and pinned charts from this session.</p>
        <button type="button" disabled={saving || checking || resetting} onClick={handleResetSession} className="flex min-h-11 items-center gap-2 rounded-xl border border-rose-300 px-4 text-sm font-semibold text-rose-700 hover:bg-rose-50 disabled:opacity-50 dark:border-rose-800 dark:text-rose-300 dark:hover:bg-rose-950">
          <RotateCcw className="h-4 w-4" />
          {resetting ? "Resetting workspace…" : "Reset workspace session"}
        </button>
      </section>
    </div>
  );
};
