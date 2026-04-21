import React, { useState, useEffect, useRef } from 'react';
import {
  Play,
  Square,
  Pause,
  Terminal,
  Activity,
  Clock,
  Zap,
  Rocket,
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import { ProgressBar } from '../ui/ProgressBar';
import { Modal } from '../ui/Modal';
import { useApp } from '../../context/AppContext';
import { useWebSocket } from '../../hooks/useWebSocket';
import { startTraining, stopTraining, pauseTraining } from '../../services/api';
import { HardwarePanel } from './HardwarePanel';

interface TrainingMonitorProps {
  onTrainingStateChange?: (isTraining: boolean) => void;
}

export function TrainingMonitor({ onTrainingStateChange }: TrainingMonitorProps) {
  const { state, dispatch } = useApp();
  const { status, config, dataset } = {
    status: state.trainingStatus,
    config: state.trainingConfig,
    dataset: state.currentDataset,
  };

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isStopModalOpen, setIsStopModalOpen] = useState(false);
  const logsEndRef = useRef<HTMLDivElement>(null);

  const { isConnected, send } = useWebSocket({
    url: 'ws://localhost:8000/ws/training',
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

  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [status.logs]);

  // OS-level progress bar
  useEffect(() => {
    try {
      const ipcRenderer = (window as any).require?.('electron')?.ipcRenderer;
      if (!ipcRenderer) return;
      if (status.phase === 'training' && status.totalSteps > 0) {
        ipcRenderer.send('set-progress-bar', status.currentStep / status.totalSteps);
      } else if (status.phase === 'completed') {
        ipcRenderer.send('set-progress-bar', -1);
        ipcRenderer.send('show-notification', 'Training Complete', 'Your LoRA model has finished training!');
      } else if (status.phase === 'error') {
        ipcRenderer.send('set-progress-bar', -1);
        ipcRenderer.send('show-notification', 'Training Error', 'An error interrupted training.');
      } else if (status.phase === 'idle' || status.phase === 'paused') {
        ipcRenderer.send('set-progress-bar', -1);
      }
    } catch (e) { /* not electron */ }
  }, [status.phase, status.currentStep, status.totalSteps]);

  useEffect(() => {
    dispatch({ type: 'SET_WS_CONNECTED', payload: isConnected });
  }, [isConnected, dispatch]);

  const isTraining = status.phase === 'training' || status.phase === 'preparing';
  const isPaused = status.phase === 'paused';
  const isActive = isTraining || isPaused || status.phase === 'completed' || status.phase === 'error';

  // Notify parent of training state
  useEffect(() => {
    onTrainingStateChange?.(isTraining);
  }, [isTraining, onTrainingStateChange]);

  const handleStart = async () => {
    if (!dataset) { alert('Please upload a dataset first.'); return; }
    if (dataset.images.length === 0) { alert('Dataset is empty.'); return; }
    const imagesWithPaths = dataset.images.filter(img => img.filePath);
    if (imagesWithPaths.length === 0) { alert('No images with local file paths.'); return; }
    try {
      dispatch({ type: 'SET_TRAINING_STATUS', payload: { phase: 'preparing', totalSteps: config.trainingSteps, lossHistory: [], logs: [] } });
      const imageData = imagesWithPaths.map(img => ({ filePath: img.filePath, captions: img.captions || [] }));
      const res = await startTraining(config, imageData);
      if (res.error) {
        alert(res.error);
        dispatch({ type: 'SET_TRAINING_STATUS', payload: { phase: 'idle' } });
        return;
      }
      setSessionId(res.sessionId || null);
      if (isConnected && res.sessionId) {
        send({ type: 'start_training', payload: { sessionId: res.sessionId } });
      }
    } catch (err) {
      dispatch({ type: 'SET_TRAINING_STATUS', payload: { phase: 'error' } });
      console.error(err);
    }
  };

  const handlePause = async () => {
    if (!sessionId) return;
    try {
      await pauseTraining(sessionId);
      dispatch({ type: 'SET_TRAINING_STATUS', payload: { phase: 'paused' } });
    } catch (err) { console.error(err); }
  };

  const handleStopClick = () => {
    if (!sessionId) return;
    setIsStopModalOpen(true);
  };

  const confirmStop = async () => {
    if (!sessionId) return;
    try {
      await stopTraining(sessionId);
      dispatch({ type: 'SET_TRAINING_STATUS', payload: { phase: 'idle' } });
      setSessionId(null);
    } catch (err) { console.error(err); }
    finally { setIsStopModalOpen(false); }
  };

  const formatTime = (seconds: number) => {
    if (seconds <= 0) return '00:00';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  const progressPercent = status.totalSteps > 0 ? (status.currentStep / status.totalSteps) * 100 : 0;

  return (
    <div className="workspace__right-inner">
      {/* Controls in header area */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h2 style={{ fontSize: 'var(--text-lg)', fontWeight: 'var(--weight-semibold)', color: 'var(--color-text-primary)' }}>
          Training Monitor
        </h2>
        <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
          {(!isTraining && !isPaused) ? (
            <Button variant="primary" icon={<Play size={16} />} onClick={handleStart} disabled={!dataset}>
              Start Training
            </Button>
          ) : (
            <>
              <Button
                variant={isPaused ? 'primary' : 'secondary'}
                icon={isPaused ? <Play size={16} /> : <Pause size={16} />}
                onClick={isPaused ? handleStart : handlePause}
              >
                {isPaused ? 'Resume' : 'Pause'}
              </Button>
              <Button variant="danger" icon={<Square size={16} />} onClick={handleStopClick}>
                Stop
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Idle state — show hardware info + hint */}
      {!isActive && (
        <div className="monitor__idle-container">
          <HardwarePanel />
          <div className="monitor__idle">
            <div className="monitor__idle-icon"><Rocket size={28} /></div>
            <p className="monitor__idle-text">
              Configure your dataset and parameters on the left, then hit Start Training.
            </p>
          </div>
        </div>
      )}

      {/* Active training content */}
      {isActive && (
        <>
          {/* Progress */}
          <Card className="monitor__progress-card">
            <div className="training-progress-header">
              <div className="training-status-indicator">
                <Badge
                  variant={
                    status.phase === 'training' ? 'accent' :
                    status.phase === 'completed' ? 'success' :
                    status.phase === 'error' ? 'error' :
                    status.phase === 'paused' ? 'warning' : 'default'
                  }
                  dot={status.phase === 'training'}
                >
                  {status.phase.toUpperCase()}
                </Badge>
                <span className="training-step-text">
                  Step {status.currentStep.toLocaleString()} / {status.totalSteps.toLocaleString()}
                </span>
              </div>
              <div className="training-eta">
                <Clock size={14} />
                <span>ETA: {formatTime(status.eta)}</span>
              </div>
            </div>
            <ProgressBar
              value={progressPercent} max={100} size="lg"
              variant={status.phase === 'error' ? 'error' : 'accent'}
              animated={status.phase === 'training'}
            />
          </Card>

          {/* Metrics */}
          <div className="monitor__metrics">
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
          <Card className="monitor__chart">
            <h3 className="chart-title">Loss Curve</h3>
            <div className="chart-container">
              {status.lossHistory.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={status.lossHistory} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-surface-border)" />
                    <XAxis dataKey="step" stroke="var(--color-text-tertiary)" tick={{ fill: 'var(--color-text-tertiary)', fontSize: 12 }} />
                    <YAxis stroke="var(--color-text-tertiary)" tick={{ fill: 'var(--color-text-tertiary)', fontSize: 12 }} domain={['auto', 'auto']} />
                    <Tooltip
                      contentStyle={{ backgroundColor: 'var(--color-bg-elevated)', borderColor: 'var(--color-surface-border)', borderRadius: 'var(--radius-md)' }}
                      itemStyle={{ color: 'var(--color-text-primary)' }}
                    />
                    <Line type="monotone" dataKey="loss" stroke="var(--color-accent)" strokeWidth={2} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="chart-empty">No data yet</div>
              )}
            </div>
          </Card>

          {/* Logs */}
          <div className="monitor__logs">
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
        </>
      )}

      <Modal isOpen={isStopModalOpen} onClose={() => setIsStopModalOpen(false)} title="Stop Training">
        <p style={{ marginBottom: '1.5rem', color: 'var(--color-text-secondary)' }}>
          Are you sure you want to stop training? Progress will be lost.
        </p>
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
          <Button variant="secondary" onClick={() => setIsStopModalOpen(false)}>Cancel</Button>
          <Button variant="danger" onClick={confirmStop}>Stop</Button>
        </div>
      </Modal>
    </div>
  );
}
