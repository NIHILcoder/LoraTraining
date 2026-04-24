import React, { useEffect, useRef, useState } from 'react';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { Terminal, Download, Cpu, Activity, CheckCircle2, AlertTriangle, Zap } from 'lucide-react';
import { ProgressBar } from '../ui/ProgressBar';

export function SetupScreen({ onComplete }: { onComplete: () => void }) {
  const [logs, setLogs] = useState<string[]>([]);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [status, setStatus] = useState<'idle' | 'installing' | 'starting' | 'error'>('idle');
  const [errorMsg, setErrorMsg] = useState('');
  
  const logsEndRef = useRef<HTMLDivElement>(null);

  const addLog = (msg: string) => {
    setLogs((prev) => [...prev, msg]);
  };

  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  useEffect(() => {
    const ipcRenderer = (window as any).require?.('electron')?.ipcRenderer;
    if (!ipcRenderer) return;

    ipcRenderer.on('install-log', (_e: any, msg: string) => addLog(msg.trim()));
    ipcRenderer.on('install-progress', (_e: any, pct: number) => setProgress(pct));
    ipcRenderer.on('install-step', (_e: any, stepName: string) => setCurrentStep(stepName));
    ipcRenderer.on('backend-log', (_e: any, msg: string) => addLog(msg.trim()));

    return () => {
      ipcRenderer.removeAllListeners('install-log');
      ipcRenderer.removeAllListeners('install-progress');
      ipcRenderer.removeAllListeners('install-step');
      ipcRenderer.removeAllListeners('backend-log');
    };
  }, []);

  const handleInstall = async () => {
    const ipcRenderer = (window as any).require?.('electron')?.ipcRenderer;
    if (!ipcRenderer) {
      addLog('Error: Electron IPC not found.');
      setStatus('error');
      return;
    }

    setStatus('installing');
    setProgress(0);
    setCurrentStep('Initializing Setup...');
    setLogs(['Starting automatic environment setup...']);

    ipcRenderer.once('install-complete', (_e: any, result: any) => {
      if (result.success) {
        addLog('Environment installed successfully. Starting backend server...');
        setStatus('starting');
        setCurrentStep('Booting up AI Server...');
        setProgress(100);
        ipcRenderer.send('start-backend');
      } else {
        setStatus('error');
        setErrorMsg(result.error || 'Unknown installation error');
        addLog(`Error: ${result.error}`);
      }
    });

    ipcRenderer.send('install-env');
  };

  useEffect(() => {
    const ipcRenderer = (window as any).require?.('electron')?.ipcRenderer;
    if (!ipcRenderer) return;

    const onBackendStarted = (_e: any, result: any) => {
      if (result.success) {
        onComplete();
      } else {
        setStatus('error');
        setErrorMsg(result.error || 'Failed to start backend');
        addLog(`Error: ${result.error}`);
      }
    };

    ipcRenderer.on('backend-started', onBackendStarted);
    return () => {
      ipcRenderer.removeAllListeners('backend-started');
    };
  }, [onComplete]);

  return (
    <div style={styles.container}>
      <Card style={styles.card} padding="lg">
        <div style={styles.header}>
          <div style={styles.iconContainer}>
            <Zap size={32} color="var(--color-accent)" />
          </div>
          <h1 style={styles.title}>Welcome to LoRA Training Dashboard</h1>
          <p style={styles.subtitle}>
            Before we begin, we need to set up an isolated Python environment. 
            This ensures your global Python is untouched and guarantees PyTorch runs perfectly on your hardware.
          </p>
        </div>

        {status === 'idle' ? (
          <div style={styles.actionArea}>
            <div style={styles.featureList}>
              <div style={styles.feature}><Download size={16} /> Downloads portable Python 3.12</div>
              <div style={styles.feature}><Cpu size={16} /> Installs PyTorch with CUDA 12.1 support</div>
              <div style={styles.feature}><Activity size={16} /> Fully self-contained, no global changes</div>
            </div>
            <Button size="lg" variant="primary" onClick={handleInstall}>
              Install Environment (~2.5 GB)
            </Button>
          </div>
        ) : (
          <div style={styles.progressArea}>
            <div style={styles.statusHeader}>
              <span style={styles.statusText}>
                {status === 'installing' && currentStep}
                {status === 'starting' && 'Starting AI Server...'}
                {status === 'error' && 'Installation Failed'}
              </span>
              {status === 'installing' && <span style={styles.pctText}>{Math.round(progress)}%</span>}
            </div>
            
            <ProgressBar 
              value={progress} 
              max={100} 
              variant={status === 'error' ? 'error' : 'accent'} 
              animated={status === 'installing' || status === 'starting'}
              size="lg"
            />
            
            {status === 'error' && (
              <div style={styles.errorBox}>
                <AlertTriangle size={16} />
                <span>{errorMsg}</span>
              </div>
            )}
            
            {status === 'error' && (
              <Button style={{ marginTop: '1rem' }} variant="secondary" onClick={() => setStatus('idle')}>
                Try Again
              </Button>
            )}

            <div style={styles.terminal}>
              <div style={styles.terminalHeader}>
                <Terminal size={14} /> <span>Setup Logs</span>
              </div>
              <div style={styles.logs}>
                {logs.map((log, i) => (
                  <div key={i} style={styles.logLine}>{log}</div>
                ))}
                <div ref={logsEndRef} />
              </div>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
    height: '100vh',
    padding: 'var(--space-6)',
    backgroundColor: 'var(--color-bg-primary)',
  },
  card: {
    width: '100%',
    maxWidth: '680px',
    display: 'flex',
    flexDirection: 'column',
    gap: 'var(--space-6)',
  },
  header: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    textAlign: 'center',
    gap: 'var(--space-3)',
  },
  iconContainer: {
    width: '64px',
    height: '64px',
    borderRadius: 'var(--radius-2xl)',
    backgroundColor: 'var(--color-accent-muted)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 'var(--space-2)',
  },
  title: {
    fontSize: 'var(--text-2xl)',
    fontWeight: 'var(--weight-bold)',
    color: 'var(--color-text-primary)',
    margin: 0,
  },
  subtitle: {
    fontSize: 'var(--text-sm)',
    color: 'var(--color-text-tertiary)',
    lineHeight: '1.6',
    maxWidth: '520px',
  },
  actionArea: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 'var(--space-6)',
  },
  featureList: {
    display: 'flex',
    flexDirection: 'column',
    gap: 'var(--space-3)',
    backgroundColor: 'var(--color-bg-secondary)',
    padding: 'var(--space-4)',
    borderRadius: 'var(--radius-lg)',
    width: '100%',
    maxWidth: '420px',
  },
  feature: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-3)',
    fontSize: 'var(--text-sm)',
    color: 'var(--color-text-secondary)',
  },
  progressArea: {
    display: 'flex',
    flexDirection: 'column',
    gap: 'var(--space-4)',
  },
  statusHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  statusText: {
    fontSize: 'var(--text-sm)',
    fontWeight: 'var(--weight-semibold)',
    color: 'var(--color-text-primary)',
  },
  pctText: {
    fontSize: 'var(--text-sm)',
    color: 'var(--color-text-tertiary)',
  },
  terminal: {
    marginTop: 'var(--space-2)',
    backgroundColor: '#000',
    borderRadius: 'var(--radius-md)',
    overflow: 'hidden',
    border: '1px solid var(--color-surface-border)',
  },
  terminalHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
    padding: 'var(--space-2) var(--space-3)',
    backgroundColor: 'var(--color-bg-tertiary)',
    borderBottom: '1px solid var(--color-surface-border)',
    fontSize: 'var(--text-xs)',
    color: 'var(--color-text-tertiary)',
    fontWeight: 'var(--weight-medium)',
  },
  logs: {
    height: '220px',
    overflowY: 'auto',
    padding: 'var(--space-3)',
    fontFamily: 'monospace',
    fontSize: '12px',
    color: '#a1a1aa',
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  logLine: {
    wordBreak: 'break-all',
  },
  errorBox: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
    padding: 'var(--space-3)',
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    border: '1px solid rgba(239, 68, 68, 0.2)',
    borderRadius: 'var(--radius-md)',
    color: '#ef4444',
    fontSize: 'var(--text-sm)',
  }
};
