import React, {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
  useRef,
} from "react";
import {
  ConfigState,
  DatasetItem,
  DatasetPreviewResponse,
  PinnedChartItem,
  PinChartRequest,
} from "../types";
import { configApi, datasetApi } from "../api/client";
import { useToast } from "../components/Toast";

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

export const DatasetProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [activeDatasetName, setActiveDatasetNameState] = useState<
    string | null
  >(null);
  const [preview, setPreview] = useState<DatasetPreviewResponse | null>(null);
  const [config, setConfig] = useState<ConfigState | null>(null);
  const [pinnedCharts, setPinnedCharts] = useState<PinnedChartItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [previewLoading, setPreviewLoading] = useState<boolean>(false);
  const previewRequest = useRef(0);
  const changingDataset = useRef(false);
  const { error: toastError } = useToast();

  const refreshConfig = useCallback(async () => {
    try {
      const cfg = await configApi.getConfig();
      setConfig(cfg);
    } catch (err) {
      console.error("Failed to load config", err);
    }
  }, []);

  const refreshPinnedCharts = useCallback(async () => {
    try {
      const res = await datasetApi.getPinnedCharts();
      setPinnedCharts(res.pinned_charts || []);
    } catch (err) {
      console.error("Failed to load pinned charts", err);
    }
  }, []);

  const pinChart = useCallback(
    async (req: PinChartRequest) => {
      const item = await datasetApi.pinChart(req);
      await refreshPinnedCharts();
      return item;
    },
    [refreshPinnedCharts],
  );

  const unpinChart = useCallback(
    async (pinId: string) => {
      await datasetApi.unpinChart(pinId);
      await refreshPinnedCharts();
    },
    [refreshPinnedCharts],
  );

  const fetchPreview = useCallback(async (name: string | null) => {
    const request = ++previewRequest.current;
    if (!name) {
      setPreview(null);
      setPreviewLoading(false);
      return;
    }
    setPreviewLoading(true);
    try {
      const data = await datasetApi.getPreview(50);
      if (request === previewRequest.current) setPreview(data.name === name ? data : null);
    } catch (err) {
      console.error("Failed to load dataset preview", err);
      if (request === previewRequest.current) setPreview(null);
    } finally {
      if (request === previewRequest.current) setPreviewLoading(false);
    }
  }, []);

  const refreshPreview = useCallback(() => fetchPreview(activeDatasetName), [activeDatasetName, fetchPreview]);

  const refreshDatasets = useCallback(async () => {
    setLoading(true);
    try {
      const res = await datasetApi.listDatasets();
      setDatasets(res.datasets);
      setActiveDatasetNameState(res.active_dataset);
      await fetchPreview(res.active_dataset);
    } catch (err) {
      console.error("Failed to list datasets", err);
    } finally {
      setLoading(false);
    }
  }, [fetchPreview]);

  const setActiveDataset = useCallback(
    async (name: string) => {
      if (changingDataset.current) return;
      changingDataset.current = true;
      setLoading(true);
      try {
        await datasetApi.setActive(name);
        await refreshDatasets();
      } catch (err) {
        toastError("Could not switch datasets", "Try selecting the dataset again.");
      } finally {
        changingDataset.current = false;
        setLoading(false);
      }
    },
    [refreshDatasets, toastError],
  );

  useEffect(() => {
    refreshDatasets();
    refreshConfig();
    refreshPinnedCharts();
  }, [refreshDatasets, refreshConfig, refreshPinnedCharts]);

  const activeDataset = datasets.find((d) => d.name === activeDatasetName) || null;
  const currentPreview = preview?.name === activeDataset?.name &&
    (preview?.version == null || preview.version === activeDataset?.version) ? preview : null;
  const hasDataset = Boolean(activeDataset);

  return (
    <DatasetContext.Provider
      value={{
        datasets,
        activeDataset,
        activeDatasetName,
        preview: currentPreview,
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
    throw new Error("useDataset must be used within a DatasetProvider");
  }
  return context;
};
