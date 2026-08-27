import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { ConfigState, DatasetItem, DatasetPreviewResponse, PinnedChartItem, PinChartRequest } from '../types';
import { configApi, datasetApi } from '../api/client';

interface DatasetContextType {
  datasets: DatasetItem[];
  activeDataset: DatasetItem | null;
  activeDatasetName: string | null;
  preview: DatasetPreviewResponse | null;
  config: ConfigState | null;
  pinnedCharts: PinnedChartItem[];
  loading: boolean;
  previewLoading: boolean;
  refreshDatasets: () => Promise<void>;
  refreshPreview: () => Promise<void>;
  refreshConfig: () => Promise<void>;
  refreshPinnedCharts: () => Promise<void>;
  setActiveDataset: (name: string) => Promise<void>;
  pinChart: (req: PinChartRequest) => Promise<PinnedChartItem>;
  unpinChart: (pinId: string) => Promise<void>;
  hasDataset: boolean;
}

const DatasetContext = createContext<DatasetContextType | undefined>(undefined);

export const DatasetProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [activeDatasetName, setActiveDatasetNameState] = useState<string | null>(null);
  const [preview, setPreview] = useState<DatasetPreviewResponse | null>(null);
  const [config, setConfig] = useState<ConfigState | null>(null);
  const [pinnedCharts, setPinnedCharts] = useState<PinnedChartItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [previewLoading, setPreviewLoading] = useState<boolean>(false);

  const refreshConfig = useCallback(async () => {
    try {
      const cfg = await configApi.getConfig();
      setConfig(cfg);
    } catch (err) {
      console.error('Failed to load config', err);
    }
  }, []);

  const refreshPinnedCharts = useCallback(async () => {
    try {
      const res = await datasetApi.getPinnedCharts();
      setPinnedCharts(res.pinned_charts || []);
    } catch (err) {
      console.error('Failed to load pinned charts', err);
    }
  }, []);

  const pinChart = useCallback(async (req: PinChartRequest) => {
    const item = await datasetApi.pinChart(req);
    await refreshPinnedCharts();
    return item;
  }, [refreshPinnedCharts]);

  const unpinChart = useCallback(async (pinId: string) => {
    await datasetApi.unpinChart(pinId);
    await refreshPinnedCharts();
  }, [refreshPinnedCharts]);

  const refreshPreview = useCallback(async () => {
    if (!activeDatasetName) {
      setPreview(null);
      return;
    }
    setPreviewLoading(true);
    try {
      const data = await datasetApi.getPreview(50);
      setPreview(data);
    } catch (err) {
      console.error('Failed to load dataset preview', err);
      setPreview(null);
    } finally {
      setPreviewLoading(false);
    }
  }, [activeDatasetName]);

  const refreshDatasets = useCallback(async () => {
    setLoading(true);
    try {
      const res = await datasetApi.listDatasets();
      setDatasets(res.datasets);
      setActiveDatasetNameState(res.active_dataset);
    } catch (err) {
      console.error('Failed to list datasets', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const setActiveDataset = useCallback(
    async (name: string) => {
      try {
        await datasetApi.setActive(name);
        setActiveDatasetNameState(name);
        await refreshDatasets();
      } catch (err) {
        console.error('Failed to set active dataset', err);
      }
    },
    [refreshDatasets]
  );

  useEffect(() => {
    refreshDatasets();
    refreshConfig();
    refreshPinnedCharts();
  }, [refreshDatasets, refreshConfig, refreshPinnedCharts]);

  useEffect(() => {
    refreshPreview();
  }, [activeDatasetName, refreshPreview]);

  const activeDataset = datasets.find((d) => d.name === activeDatasetName) || null;
  const hasDataset = Boolean(activeDataset);

  return (
    <DatasetContext.Provider
      value={{
        datasets,
        activeDataset,
        activeDatasetName,
        preview,
        config,
        pinnedCharts,
        loading,
        previewLoading,
        refreshDatasets,
        refreshPreview,
        refreshConfig,
        refreshPinnedCharts,
        setActiveDataset,
        pinChart,
        unpinChart,
        hasDataset,
      }}
    >
      {children}
    </DatasetContext.Provider>
  );
};

export const useDataset = () => {
  const context = useContext(DatasetContext);
  if (!context) {
    throw new Error('useDataset must be used within a DatasetProvider');
  }
  return context;
};
