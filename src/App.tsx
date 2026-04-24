import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AppProvider } from './context/AppContext';
import { Sidebar } from './components/layout/Sidebar';
import { TrainingWorkspacePage } from './pages/TrainingWorkspacePage';
import { ModelsPage } from './pages/ModelsPage';
import { GalleryPage } from './pages/GalleryPage';
import { PlaygroundPage } from './pages/PlaygroundPage';
import { TitleBar } from './components/layout/TitleBar';
import { SetupScreen } from './components/setup/SetupScreen';
import { Card } from './components/ui/Card';
import { Cpu, AlertTriangle } from 'lucide-react';
import './App.css';

type AppState = 'checking' | 'needs_setup' | 'starting' | 'ready' | 'error';

export function App() {
  const [appState, setAppState] = useState<AppState>('checking');
  const [errorMsg, setErrorMsg] = useState('');

  useEffect(() => {
    const ipcRenderer = (window as any).require?.('electron')?.ipcRenderer;
    if (!ipcRenderer) {
      // Running in browser mode (dev without electron)
      setAppState('ready');
      return;
    }

    const init = async () => {
      try {
        const envExists = await ipcRenderer.invoke('check-env');
        if (!envExists) {
          setAppState('needs_setup');
        } else {
          setAppState('starting');
          ipcRenderer.once('backend-started', (_e: any, res: any) => {
            if (res.success) setAppState('ready');
            else {
              setAppState('error');
              setErrorMsg(res.error || 'Failed to start AI Server');
            }
          });
          ipcRenderer.send('start-backend');
        }
      } catch (err: any) {
        setAppState('error');
        setErrorMsg(err.message);
      }
    };
    init();
  }, []);

  if (appState === 'checking' || appState === 'starting') {
    return (
      <div className="window-layout">
        <TitleBar />
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', background: 'var(--color-bg-primary)' }}>
          <Card padding="lg" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 'var(--space-3)' }}>
            <Cpu size={32} className={appState === 'starting' ? 'animate-pulse' : ''} color="var(--color-accent)" />
            <span style={{ color: 'var(--color-text-secondary)', fontSize: 'var(--text-lg)' }}>
              {appState === 'checking' ? 'Checking environment...' : 'Starting AI Server...'}
            </span>
          </Card>
        </div>
      </div>
    );
  }

  if (appState === 'needs_setup') {
    return (
      <div className="window-layout">
        <TitleBar />
        <SetupScreen onComplete={() => setAppState('ready')} />
      </div>
    );
  }

  if (appState === 'error') {
    return (
      <div className="window-layout">
        <TitleBar />
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', background: 'var(--color-bg-primary)' }}>
          <Card padding="lg" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 'var(--space-3)' }}>
            <AlertTriangle size={32} color="var(--color-error)" />
            <span style={{ color: 'var(--color-error)', fontSize: 'var(--text-lg)', fontWeight: 'bold' }}>Startup Error</span>
            <span style={{ color: 'var(--color-text-secondary)' }}>{errorMsg}</span>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <AppProvider>
      <Router>
        <div className="window-layout">
          <TitleBar />
          <div className="app-layout">
            <Sidebar />
            <main className="app-main">
              <div className="app-content-wrapper" style={{ height: '100%' }}>
                <Routes>
                  <Route path="/" element={<TrainingWorkspacePage />} />
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
