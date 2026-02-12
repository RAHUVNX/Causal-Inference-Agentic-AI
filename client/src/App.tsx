import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import UploadDataPage from './pages/UploadDataPage';
import TrainModelsPage from './pages/TrainModelsPage';
import RunLiftPage from './pages/RunLiftPage';
import DashboardPage from './pages/DashboardPage';
import ModelMetricsPage from './pages/ModelMetricsPage';

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-layout">
        <Sidebar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/upload" element={<UploadDataPage />} />
            <Route path="/train" element={<TrainModelsPage />} />
            <Route path="/lift" element={<RunLiftPage />} />
            <Route path="/metrics" element={<ModelMetricsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
