import React, { useState, useEffect, useRef } from 'react';
import {
  Play,
  Square,
  Terminal,
  Activity,
  Clock,
  Zap,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { Header } from '../components/layout/Header';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { ProgressBar } from '../components/ui/ProgressBar';
import { Modal } from '../components/ui/Modal';
import { useApp } from '../context/AppContext';
import { useWebSocket } from '../hooks/useWebSocket';
import { startTraining, stopTraining, getWsUrl } from '../services/api';
import './TrainingPage.css';

export function TrainingPage() {
  const { state, dispatch } = useApp();
  const { status, config, dataset } = {
    status: state.trainingStatus,
    config: state.trainingConfig,
    dataset: state.currentDataset,
  };
  
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isStopModalOpen, setIsStopModalOpen] = useState(false);
  const logsEndRef = useRef<HTMLDivElement>(null);

  // WebSocket for live updates
  const { isConnected, send } = useWebSocket({
    url: getWsUrl('/ws/training'),
    onMessage: (msg) => {
      switch (msg.type) {
        case 'training_update':
          dispatch({ type: 'SET_TRAINING_STATUS', payload: msg.data });
          break;
        case 'training_step':
          dispatch({ type: 'ADD_TRAINING_STEP', payload: msg.data });
          break;
        case 'training_log':
          dispatch({ type: 'ADD_LOG', payload: msg.data });
          break;
      }
    },
  });

  // Auto-scroll logs
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [status.logs]);

  // Sync OS level Progress Bar and Notifications via preload API
  useEffect(() => {
    const api = window.loraStudio;
    if (!api) return;

    if (status.phase === 'training' && status.totalSteps > 0) {
      api.setProgressBar(status.currentStep / status.totalSteps);
    } else if (status.phase === 'completed') {
      api.setProgressBar(-1);
      api.showNotification('Training Complete', 'Your LoRA model has successfully finished training!');
    } else if (status.phase === 'error') {
      api.setProgressBar(-1);
      api.showNotification('Training Error', 'An error interrupted your training process.');
    } else if (status.phase === 'idle') {
      api.setProgressBar(-1);
    }
  }, [status.phase, status.currentStep, status.totalSteps]);

  // Sync WS connection status
  useEffect(() => {
    dispatch({ type: 'SET_WS_CONNECTED', payload: isConnected });
  }, [isConnected, dispatch]);

  const handleStart = async () => {
    if (!dataset) {
      alert('Please select or upload a dataset first.');
      return;
    }
    if (dataset.images.length === 0) {
      alert('Dataset is empty. Please add images before training.');
      return;
    }
    // Validate images have file paths
    const imagesWithPaths = dataset.images.filter(img => img.filePath);
    if (imagesWithPaths.length === 0) {
      alert('No images with local file paths found. Please re-upload your images.');
      return;
    }
    try {
      dispatch({
        type: 'SET_TRAINING_STATUS',
        payload: { phase: 'preparing', totalSteps: config.trainingSteps, lossHistory: [], logs: [] },
      });
      
      // Send images data (filePaths + captions) for dataset preparation
      const imageData = imagesWithPaths.map(img => ({
        filePath: img.filePath,
        captions: img.captions || [],
      }));
      
      const res = await startTraining(config, imageData);
      
      if (res.error) {
        alert(res.error);
        dispatch({
          type: 'SET_TRAINING_STATUS',
          payload: { phase: 'idle' },
        });
        return;
      }
      
      setSessionId(res.sessionId || null);
      
      // Let the python backend know that it should start the real training for this session
      if (isConnected && res.sessionId) {
         send({ type: 'start_training', payload: { sessionId: res.sessionId } });
      }
      
    } catch (err) {
      dispatch({
        type: 'SET_TRAINING_STATUS',
        payload: { phase: 'error' },
      });
      console.error(err);
    }
  };

  // P1-04: Pause/resume removed — no real backend support exists.
  // Will be re-added when the trainer implements thread-safe pause.

  const handleStopClick = () => {
    if (!sessionId) return;
    setIsStopModalOpen(true);
  };

  const confirmStop = async () => {
    if (!sessionId) return;
    try {
      await stopTraining(sessionId);
      dispatch({
        type: 'SET_TRAINING_STATUS',
        payload: { phase: 'idle' },
      });
      setSessionId(null);
    } catch (err) {
      console.error(err);
    } finally {
      setIsStopModalOpen(false);
    }
  };

  const formatTime = (seconds: number) => {
    if (seconds <= 0) return '00:00';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  const isTraining = status.phase === 'training' || status.phase === 'preparing';
  const progressPercent = status.totalSteps > 0 ? (status.currentStep / status.totalSteps) * 100 : 0;

  return (
    <div className="training-page animate-fade-in-up">
      <Header
        title="Training"
        subtitle="Monitor training progress in real-time"
        actions={
          <>
            {!isTraining ? (
              <Button
                variant="primary"
                icon={<Play size={16} />}
                onClick={handleStart}
                disabled={!dataset}
              >
                Start Training
              </Button>
            ) : (
              <Button
                variant="danger"
                icon={<Square size={16} />}
                onClick={handleStopClick}
              >
                Stop
              </Button>
            )}
          </>
        }
      />

      <div className="training-layout stagger-children">
        {/* Main Stats Column */}
        <div className="training-main">
          {/* Progress Card */}
          <Card className="training-progress-card">
            <div className="training-progress-header">
              <div className="training-status-indicator">
                <Badge 
                  variant={
                    status.phase === 'training' ? 'accent' :
                    status.phase === 'completed' ? 'success' :
                    status.phase === 'error' ? 'error' : 'default'
                  }
                  dot={status.phase === 'training'}
                >
                  {status.phase.toUpperCase()}
                </Badge>
                <span className="training-step-text">
                  Step {status.currentStep.toLocaleString()} of {status.totalSteps.toLocaleString()}
                </span>
              </div>
              <div className="training-eta">
                <Clock size={14} />
                <span>ETA: {formatTime(status.eta)}</span>
              </div>
            </div>
            <ProgressBar 
              value={progressPercent} 
              max={100} 
              size="lg" 
              variant={status.phase === 'error' ? 'error' : 'accent'}
              animated={status.phase === 'training'}
            />
          </Card>

          {/* Metrics Grid */}
          <div className="metrics-grid">
            <Card className="metric-card" padding="sm">
              <div className="metric-icon"><Activity size={20} /></div>
              <div className="metric-info">
                <span className="metric-label">Current Loss</span>
                <span className="metric-value">{status.currentLoss.toFixed(4)}</span>
              </div>
            </Card>
            <Card className="metric-card" padding="sm">
              <div className="metric-icon"><Activity size={20} /></div>
              <div className="metric-info">
                <span className="metric-label">Avg Loss</span>
                <span className="metric-value">{status.avgLoss > 0 ? status.avgLoss.toFixed(4) : '--'}</span>
              </div>
            </Card>
            <Card className="metric-card" padding="sm">
              <div className="metric-icon"><Zap size={20} /></div>
              <div className="metric-info">
                <span className="metric-label">Learning Rate</span>
                <span className="metric-value">{status.learningRate > 0 ? status.learningRate.toExponential(2) : '--'}</span>
              </div>
            </Card>
          </div>

          {/* Loss Chart */}
          <Card className="chart-card">
            <h3 className="chart-title">Loss Curve</h3>
            <div className="chart-container">
              {status.lossHistory.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={status.lossHistory} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-surface-border)" />
                    <XAxis 
                      dataKey="step" 
                      stroke="var(--color-text-tertiary)"
                      tick={{ fill: 'var(--color-text-tertiary)', fontSize: 12 }}
                    />
                    <YAxis 
                      stroke="var(--color-text-tertiary)"
                      tick={{ fill: 'var(--color-text-tertiary)', fontSize: 12 }}
                      domain={['auto', 'auto']}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'var(--color-bg-elevated)',
                        borderColor: 'var(--color-surface-border)',
                        borderRadius: 'var(--radius-md)'
                      }}
                      itemStyle={{ color: 'var(--color-text-primary)' }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="loss" 
                      stroke="var(--color-accent)" 
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="chart-empty">No data yet</div>
              )}
            </div>
          </Card>
        </div>

        {/* Sidebar / Logs */}
        <div className="training-sidebar">
          <Card className="logs-card" padding="none">
            <div className="logs-header">
              <Terminal size={16} />
              <span>Training Logs</span>
            </div>
            <div className="logs-content">
              {status.logs.length === 0 ? (
                <div className="logs-empty">Awaiting training output...</div>
              ) : (
                status.logs.map((log) => (
                  <div key={log.id} className={`log-entry log-entry--${log.level}`}>
                    <span className="log-time">
                      {new Date(log.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                    </span>
                    <span className="log-msg">{log.message}</span>
                  </div>
                ))
              )}
              <div ref={logsEndRef} />
            </div>
          </Card>
        </div>
      </div>

      <Modal isOpen={isStopModalOpen} onClose={() => setIsStopModalOpen(false)} title="Stop Training">
        <p style={{ marginBottom: '1.5rem', color: 'var(--color-text-secondary)' }}>
          Are you sure you want to stop training? Progress will be lost.
        </p>
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
          <Button variant="secondary" onClick={() => setIsStopModalOpen(false)}>
            Cancel
          </Button>
          <Button variant="danger" onClick={confirmStop}>
            Stop
          </Button>
        </div>
      </Modal>
    </div>
  );
}
