import axios from 'axios';
import {
  AIInsightsResponse,
  AnomalyResponse,
  CleaningReportResponse,
  ColumnProfileResponse,
  ConfigState,
  CorrelationsResponse,
  DatasetListResponse,
  DatasetPreviewResponse,
  DocumentQAResponse,
  ExplorerBrowseResponse,
  ForecastingResponse,
  ProfileResponse,
  QualityScoreResponse,
  QueryHistoryResponse,
  QueryResponse,
  VersionHistoryResponse,
  VisualizationResponse,
  SQLQueryRequest,
  SQLQueryResponse,
  ColumnTransformRequest,
  ColumnTransformResponse,
  PinChartRequest,
  PinnedChartItem,
  PinnedDashboardResponse,
} from '../types';

// Generate or retrieve persistent browser session ID
function getSessionId(): string {
  let sid = localStorage.getItem('data_agent_session_id');
  if (!sid) {
    sid = 'sess_' + Math.random().toString(36).substring(2, 11);
    localStorage.setItem('data_agent_session_id', sid);
  }
  return sid;
}

export const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
    'X-Session-ID': getSessionId(),
  },
});

// Config & Settings API
export const configApi = {
  getConfig: async (): Promise<ConfigState> => {
    const res = await api.get<ConfigState>('/config');
    return res.data;
  },
  updateConfig: async (data: Partial<ConfigState> & { hf_api_key?: string }): Promise<ConfigState> => {
    const res = await api.post<ConfigState>('/config', data);
    return res.data;
  },
  resetSession: async () => {
    const res = await api.post('/session/reset');
    return res.data;
  },
};

// Datasets API
export const datasetApi = {
  listDatasets: async (): Promise<DatasetListResponse> => {
    const res = await api.get<DatasetListResponse>('/datasets');
    return res.data;
  },
  uploadFiles: async (files: FileList | File[]): Promise<any> => {
    const formData = new FormData();
    Array.from(files).forEach((file) => {
      formData.append('files', file);
    });
    const res = await api.post('/datasets/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },
  loadSample: async (sampleName: string) => {
    const res = await api.post('/datasets/sample', { sample_name: sampleName });
    return res.data;
  },
  setActive: async (name: string) => {
    const res = await api.post('/datasets/active', { name });
    return res.data;
  },
  deleteDataset: async (name: string) => {
    const res = await api.delete(`/datasets/${encodeURIComponent(name)}`);
    return res.data;
  },
  getPreview: async (limit: number = 50): Promise<DatasetPreviewResponse> => {
    const res = await api.get<DatasetPreviewResponse>(`/datasets/preview?limit=${limit}`);
    return res.data;
  },
  getDownloadCsvUrl: () => `/api/datasets/download/csv`,
  getDownloadExcelUrl: () => `/api/datasets/download/excel`,
  getPinnedCharts: async (): Promise<PinnedDashboardResponse> => {
    const res = await api.get<PinnedDashboardResponse>('/datasets/dashboard/pins');
    return res.data;
  },
  pinChart: async (req: PinChartRequest): Promise<PinnedChartItem> => {
    const res = await api.post<PinnedChartItem>('/datasets/dashboard/pins', req);
    return res.data;
  },
  unpinChart: async (pinId: string) => {
    const res = await api.delete(`/datasets/dashboard/pins/${encodeURIComponent(pinId)}`);
    return res.data;
  },
};

// Data Cleaning API
export const cleaningApi = {
  getQuality: async (): Promise<QualityScoreResponse> => {
    const res = await api.get<QualityScoreResponse>('/cleaning/quality');
    return res.data;
  },
  previewCleaning: async (config: any): Promise<CleaningReportResponse> => {
    const res = await api.post<CleaningReportResponse>('/cleaning/preview', config);
    return res.data;
  },
  applyCleaning: async (config: any): Promise<CleaningReportResponse> => {
    const res = await api.post<CleaningReportResponse>('/cleaning/apply', config);
    return res.data;
  },
  transformColumn: async (req: ColumnTransformRequest): Promise<ColumnTransformResponse> => {
    const res = await api.post<ColumnTransformResponse>('/cleaning/transform-column', req);
    return res.data;
  },
  getVersions: async (): Promise<VersionHistoryResponse> => {
    const res = await api.get<VersionHistoryResponse>('/cleaning/versions');
    return res.data;
  },
  rollbackVersion: async (version: number) => {
    const res = await api.post('/cleaning/rollback', { version });
    return res.data;
  },
};

// Explorer API
export const explorerApi = {
  browse: async (req: {
    columns?: string[];
    page?: number;
    page_size?: number;
    sort_by?: string;
    sort_dir?: string;
    filters?: any[];
  }): Promise<ExplorerBrowseResponse> => {
    const res = await api.post<ExplorerBrowseResponse>('/explorer/browse', req);
    return res.data;
  },
  getCorrelations: async (columns: string[], method: string = 'pearson'): Promise<CorrelationsResponse> => {
    const res = await api.post<CorrelationsResponse>('/explorer/correlations', { columns, method });
    return res.data;
  },
  getDistribution: async (req: {
    column: string;
    chart_type: string;
    group_by?: string;
    nbins?: number;
    top_n?: number;
  }) => {
    const res = await api.post('/explorer/distribution', req);
    return res.data;
  },
  getColumnProfile: async (column: string): Promise<ColumnProfileResponse> => {
    const res = await api.get<ColumnProfileResponse>(`/explorer/profile/${encodeURIComponent(column)}`);
    return res.data;
  },
};

// Natural Language AI Query & SQL Studio API
export const queryApi = {
  ask: async (question: string, maxTokens?: number, datasetName?: string): Promise<QueryResponse> => {
    const res = await api.post<QueryResponse>('/query', {
      question,
      max_tokens: maxTokens,
      dataset_name: datasetName,
    });
    return res.data;
  },
  runSQL: async (req: SQLQueryRequest): Promise<SQLQueryResponse> => {
    const res = await api.post<SQLQueryResponse>('/query/sql', req);
    return res.data;
  },
  getHistory: async (): Promise<QueryHistoryResponse> => {
    const res = await api.get<QueryHistoryResponse>('/query/history');
    return res.data;
  },
  clearHistory: async () => {
    const res = await api.delete('/query/history');
    return res.data;
  },
  getExportUrl: () => `/api/query/history/export`,
};

// Document QA API
export const documentApi = {
  askQA: async (question: string, documentName?: string, maxTokens?: number): Promise<DocumentQAResponse> => {
    const res = await api.post<DocumentQAResponse>('/document/qa', {
      question,
      document_name: documentName,
      max_tokens: maxTokens,
    });
    return res.data;
  },
};

// Visualizations API
export const visualizationApi = {
  generateChart: async (params: any): Promise<VisualizationResponse> => {
    const res = await api.post<VisualizationResponse>('/visualizations/generate', params);
    return res.data;
  },
};

// Insights API
export const insightsApi = {
  getQuickInsights: async (): Promise<{ insights: string[] }> => {
    const res = await api.get<{ insights: string[] }>('/insights/quick');
    return res.data;
  },
  getAIInsights: async (maxTokens: number = 1500): Promise<AIInsightsResponse> => {
    const res = await api.post<AIInsightsResponse>('/insights/ai', { max_tokens: maxTokens });
    return res.data;
  },
  getCachedAIInsights: async (): Promise<AIInsightsResponse> => {
    const res = await api.get<AIInsightsResponse>('/insights/ai/cached');
    return res.data;
  },
};

// Forecasting API
export const forecastingApi = {
  run: async (req: {
    target_col: string;
    method?: string;
    horizon?: number;
    window?: number;
    alpha?: number;
  }): Promise<ForecastingResponse> => {
    const res = await api.post<ForecastingResponse>('/forecasting/run', req);
    return res.data;
  },
  getDownloadUrl: () => `/api/forecasting/download`,
};

// Anomaly Detection API
export const anomalyApi = {
  run: async (req: {
    method?: string;
    contamination?: number;
    threshold?: number;
    factor?: number;
  }): Promise<AnomalyResponse> => {
    const res = await api.post<AnomalyResponse>('/anomalies/run', req);
    return res.data;
  },
  getDownloadUrl: () => `/api/anomalies/download`,
};

// Reports API
export const reportsApi = {
  downloadHtmlReport: async (options: { include_sample: boolean; include_stats: boolean; include_insights: boolean }) => {
    const res = await api.post('/reports/html', options, { responseType: 'blob' });
    const url = window.URL.createObjectURL(new Blob([res.data], { type: 'text/html' }));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'analysis_report.html');
    document.body.appendChild(link);
    link.click();
    link.remove();
  },
  downloadExcelReport: async () => {
    const res = await api.get('/reports/excel', { responseType: 'blob' });
    const url = window.URL.createObjectURL(new Blob([res.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'analysis_report.xlsx');
    document.body.appendChild(link);
    link.click();
    link.remove();
  },
  getProfile: async (): Promise<ProfileResponse> => {
    const res = await api.get<ProfileResponse>('/reports/profile');
    return res.data;
  },
};
