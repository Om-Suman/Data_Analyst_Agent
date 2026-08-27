export interface ConfigState {
  has_api_key: boolean;
  api_key_masked: string;
  primary_model: string;
  fallback_model: string;
  theme: string;
  max_tokens: number;
  temperature: number;
  default_outlier: string;
  default_missing_threshold: number;
  auto_profile: boolean;
  auto_insights: boolean;
}

export interface DatasetItem {
  name: string;
  source: string;
  uploaded_at: string;
  rows: number;
  cols: number;
  version: number;
  is_active: boolean;
  is_text: boolean;
}

export interface DatasetListResponse {
  datasets: DatasetItem[];
  active_dataset: string | null;
}

export interface DatasetPreviewResponse {
  name: string;
  rows: number;
  cols: number;
  columns: string[];
  data: Record<string, any>[];
  metadata: Record<string, any>;
  numeric_cols: string[];
  categorical_cols: string[];
  date_cols: string[];
  is_text: boolean;
  text_content: string | null;
}

export interface QualityScoreResponse {
  quality_score: number;
  quality_grade: string;
  missing_total: number;
  missing_pct: number;
  duplicate_rows: number;
  duplicate_pct: number;
  missing_by_column: { column: string; missing_count: number; missing_pct: number }[];
  dtypes_by_column: { column: string; dtype: string }[];
}

export interface CleaningReportResponse {
  rows_before: number;
  rows_after: number;
  cols_before: number;
  cols_after: number;
  duplicates_removed: number;
  missing_filled: Record<string, any>;
  cols_dropped: string[];
  outliers_removed: number;
  dtype_changes: Record<string, string>;
  col_renames: Record<string, string>;
  quality_score_before: number;
  quality_score_after: number;
  quality_grade_before: string;
  quality_grade_after: string;
  recommendations: string[];
  preview_data?: Record<string, any>[];
}

export interface VersionItem {
  version: number;
  timestamp: string;
  rows: number;
  cols: number;
  description: string;
}

export interface VersionHistoryResponse {
  dataset_name: string;
  current_version: number;
  versions: VersionItem[];
}

export interface ExplorerBrowseResponse {
  total_rows: number;
  total_unfiltered_rows: number;
  page: number;
  page_size: number;
  columns: string[];
  data: Record<string, any>[];
  filters_applied: string[];
}

export interface CorrelationPair {
  col_a: string;
  col_b: string;
  correlation: number;
}

export interface CorrelationsResponse {
  matrix: Record<string, Record<string, number>>;
  columns: string[];
  top_pairs: CorrelationPair[];
  figure_spec?: any;
}

export interface ColumnProfileResponse {
  column: string;
  dtype: string;
  total_count: number;
  missing_count: number;
  missing_pct: number;
  unique_count: number;
  is_numeric: boolean;
  numeric_stats?: {
    min: number | null;
    max: number | null;
    mean: number | null;
    std: number | null;
    median: number | null;
    skew: number | null;
    kurtosis: number | null;
    zeros: number;
  };
  top_values?: { value: string; count: number; percent: number }[];
  figure_spec?: any;
}

export interface CodeExecutionResult {
  code: string;
  success: boolean;
  execution_time: number;
  stdout: string;
  error?: string | null;
  figures: any[];
  dataframes: Record<string, Record<string, any>[]>;
}

export interface QueryResponse {
  question: string;
  route: string;
  route_reason: string;
  routing_source: string;
  model_used: string;
  insights: string;
  code_blocks: string[];
  execution_results: CodeExecutionResult[];
  tool_result?: any;
  error?: string | null;
}

export interface QueryHistoryItem {
  id: string;
  timestamp: string;
  question: string;
  code: string;
  result_summary: string;
  dataset: string;
  route?: string;
  model_used?: string;
}

export interface QueryHistoryResponse {
  history: QueryHistoryItem[];
}

export interface DocumentQAResponse {
  answer: string;
  sources: { text: string; score: number; metadata: Record<string, any> }[];
  engine: string;
  model_used?: string | null;
  error?: string | null;
}

export interface VisualizationResponse {
  figure_spec: any;
  chart_type: string;
  title: string;
}

export interface AIInsightsResponse {
  executive_summary: string;
  key_findings: string[];
  trends: string[];
  opportunities: string[];
  risks: string[];
  recommendations: string[];
  data_story: string;
  cached: boolean;
}

export interface ForecastingResponse {
  method: string;
  column: string;
  horizon: number;
  forecast_index: (number | string)[];
  forecast_values: (number | null)[];
  confidence_lower: (number | null)[];
  confidence_upper: (number | null)[];
  metrics: Record<string, any>;
  interpretation: string;
  figure_spec?: any;
  historical_points: number;
}

export interface AnomalyResponse {
  method: string;
  n_anomalies: number;
  anomaly_rate: number;
  anomaly_indices: number[];
  anomalous_rows: Record<string, any>[];
  scores: number[];
  columns_used: string[];
  figure_spec?: any;
  recommendations: string[];
}

export interface ProfileResponse {
  dataset_name: string;
  rows: number;
  cols: number;
  numeric_columns: {
    column: string;
    count: number;
    mean: number | null;
    std: number | null;
    min: number | null;
    "25%": number | null;
    "50%": number | null;
    "75%": number | null;
    max: number | null;
    skew: number | null;
    kurtosis: number | null;
    missing: number;
    missing_pct: number;
  }[];
  categorical_columns: {
    column: string;
    unique_values: number;
    most_common: string;
    most_common_count: number;
    missing: number;
    missing_pct: number;
  }[];
}
