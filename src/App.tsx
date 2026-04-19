import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AppProvider } from './context/AppContext';
import { Sidebar } from './components/layout/Sidebar';
import { DatasetPage } from './pages/DatasetPage';
import { ConfigPage } from './pages/ConfigPage';
import { ModelsPage } from './pages/ModelsPage';
import { TrainingPage } from './pages/TrainingPage';
import { GalleryPage } from './pages/GalleryPage';
import { PlaygroundPage } from './pages/PlaygroundPage';
import { TitleBar } from './components/layout/TitleBar';
import './App.css';

export function App() {
  return (
    <AppProvider>
      <Router>
        <div className="window-layout">
          <TitleBar />
          <div className="app-layout">
            <Sidebar />
            <main className="app-main">
              <div className="app-content-wrapper">
                <Routes>
                  <Route path="/" element={<DatasetPage />} />
                  <Route path="/config" element={<ConfigPage />} />
                  <Route path="/models" element={<ModelsPage />} />
                  <Route path="/training" element={<TrainingPage />} />
                  <Route path="/playground" element={<PlaygroundPage />} />
                  <Route path="/gallery" element={<GalleryPage />} />
                </Routes>
              </div>
            </main>
          </div>
        </div>
      </Router>
    </AppProvider>
  );
}
