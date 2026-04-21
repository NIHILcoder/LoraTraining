import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AppProvider } from './context/AppContext';
import { Sidebar } from './components/layout/Sidebar';
import { TrainingWorkspacePage } from './pages/TrainingWorkspacePage';
import { ModelsPage } from './pages/ModelsPage';
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
                  <Route path="/" element={<TrainingWorkspacePage />} />
                  {/* Legacy redirects */}
                  <Route path="/config" element={<Navigate to="/" replace />} />
                  <Route path="/training" element={<Navigate to="/" replace />} />
                  <Route path="/models" element={<div className="page-wrapper"><ModelsPage /></div>} />
                  <Route path="/playground" element={<PlaygroundPage />} />
                  <Route path="/gallery" element={<div className="page-wrapper"><GalleryPage /></div>} />
                </Routes>
              </div>
            </main>
          </div>
        </div>
      </Router>
    </AppProvider>
  );
}
