import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import { ToastProvider } from './components/Toast';
import { DatasetProvider } from './context/DatasetContext';
import { AppLayout } from './layouts/AppLayout';
import { DashboardPage } from './pages/DashboardPage';
import { CustomDashboardPage } from './pages/CustomDashboardPage';
import { SQLStudioPage } from './pages/SQLStudioPage';
import { UploadPage } from './pages/UploadPage';
import { CleaningPage } from './pages/CleaningPage';
import { ExplorerPage } from './pages/ExplorerPage';
import { AIQueryPage } from './pages/AIQueryPage';
import { DocumentQAPage } from './pages/DocumentQAPage';
import { VisualizationsPage } from './pages/VisualizationsPage';
import { InsightsPage } from './pages/InsightsPage';
import { ForecastingPage } from './pages/ForecastingPage';
import { AnomaliesPage } from './pages/AnomaliesPage';
import { ReportsPage } from './pages/ReportsPage';
import { SettingsPage } from './pages/SettingsPage';

export const App: React.FC = () => {
  return (
    <ThemeProvider>
      <ToastProvider>
        <DatasetProvider>
          <BrowserRouter>
            <Routes>
              <Route element={<AppLayout />}>
                <Route path="/" element={<DashboardPage />} />
                <Route path="/custom-dashboard" element={<CustomDashboardPage />} />
                <Route path="/sql" element={<SQLStudioPage />} />
                <Route path="/upload" element={<UploadPage />} />
                <Route path="/cleaning" element={<CleaningPage />} />
                <Route path="/explorer" element={<ExplorerPage />} />
                <Route path="/query" element={<AIQueryPage />} />
                <Route path="/document" element={<DocumentQAPage />} />
                <Route path="/visualizations" element={<VisualizationsPage />} />
                <Route path="/insights" element={<InsightsPage />} />
                <Route path="/forecasting" element={<ForecastingPage />} />
                <Route path="/anomalies" element={<AnomaliesPage />} />
                <Route path="/reports" element={<ReportsPage />} />
                <Route path="/settings" element={<SettingsPage />} />
              </Route>
            </Routes>
          </BrowserRouter>
        </DatasetProvider>
      </ToastProvider>
    </ThemeProvider>
  );
};

export default App;
