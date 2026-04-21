import React, { useEffect, useState, useCallback } from 'react';
import {
  Cpu, MemoryStick, Gauge, Clock, AlertTriangle,
  CheckCircle2, XCircle, ChevronDown, ChevronUp, Zap,
} from 'lucide-react';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { useApp } from '../../context/AppContext';
import { fetchGpuInfo, estimateTrainingTime } from '../../services/api';

// Architecture display names
const ARCH_LABELS: Record<string, string> = {
  sd15: 'SD 1.5', sd21: 'SD 2.1', sdxl: 'SDXL', sd3: 'SD3',
  flux: 'Flux', cascade: 'Cascade', hunyuan: 'Hunyuan',
  pixart: 'PixArt', kolors: 'Kolors', auraflow: 'AuraFlow',
};

interface GpuData {
  available: boolean;
  name: string;
  vram_gb: number;
  bf16_supported: boolean;
  cuda_version: string;
  ram_total_gb: number;
  ram_available_gb: number;
}

interface ProfileData {
  feasible: boolean;
  warning: string | null;
  min_vram: number;
  recommended_resolution: number;
}

interface EstimateData {
  eta_seconds: number;
  time_per_step: number;
  feasible: boolean;
}

export function HardwarePanel() {
  const { state } = useApp();
  const config = state.trainingConfig;

  const [gpu, setGpu] = useState<GpuData | null>(null);
  const [profiles, setProfiles] = useState<Record<string, ProfileData>>({});
  const [estimates, setEstimates] = useState<Record<string, EstimateData>>({});
  const [liveEta, setLiveEta] = useState<{ eta_seconds: number; time_per_step: number; feasible: boolean } | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch hardware info on mount
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const data = await fetchGpuInfo();
        if (cancelled) return;
        setGpu(data.gpu);
        setProfiles(data.profiles || {});
        setEstimates(data.estimates || {});
        setError(null);
      } catch (err) {
        if (!cancelled) setError('Could not connect to backend');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, []);

  // Live ETA estimation when config changes
  useEffect(() => {
    if (!gpu?.available) return;

    const timer = setTimeout(async () => {
      try {
        const est = await estimateTrainingTime({
          architecture: config.baseModel,
          steps: config.trainingSteps,
          rank: config.loraRank,
          resolution: config.resolution,
          batchSize: config.batchSize,
        });
        setLiveEta(est);
      } catch {
        // Silently fail — non-critical
      }
    }, 300); // Debounce

    return () => clearTimeout(timer);
  }, [gpu, config.baseModel, config.trainingSteps, config.loraRank, config.resolution, config.batchSize]);

  const formatTime = (seconds: number): string => {
    if (seconds <= 0) return '--';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}h ${m}m`;
  };

  const vramPercent = gpu ? Math.min(100, (gpu.vram_gb / 24) * 100) : 0;
  const ramPercent = gpu ? Math.min(100, ((gpu.ram_total_gb - gpu.ram_available_gb) / gpu.ram_total_gb) * 100) : 0;

  const currentProfile = profiles[config.baseModel];
  const currentEstimate = liveEta || estimates[config.baseModel];

  if (loading) {
    return (
      <Card className="hw-panel hw-panel--loading" padding="sm">
        <div className="hw-panel__loading">
          <Cpu size={16} className="animate-spin" />
          <span>Detecting hardware...</span>
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="hw-panel hw-panel--error" padding="sm">
        <div className="hw-panel__error">
          <AlertTriangle size={16} />
          <span>{error}</span>
        </div>
      </Card>
    );
  }

  if (!gpu) return null;

  const feasibleCount = Object.values(profiles).filter(p => p.feasible).length;
  const totalArchs = Object.keys(profiles).length;

  return (
    <Card className="hw-panel" padding="none">
      {/* Compact summary — always visible */}
      <button
        className="hw-panel__header"
        onClick={() => setExpanded(!expanded)}
        type="button"
      >
        <div className="hw-panel__header-left">
          <div className="hw-panel__gpu-icon">
            <Cpu size={16} />
          </div>
          <div className="hw-panel__gpu-info">
            <span className="hw-panel__gpu-name">{gpu.available ? gpu.name : 'No GPU'}</span>
            <span className="hw-panel__gpu-vram">
              {gpu.available ? `${gpu.vram_gb} GB VRAM` : 'CPU Mode'}
              {gpu.bf16_supported && <Badge variant="accent" size="sm" style={{ marginLeft: 6 }}>BF16</Badge>}
            </span>
          </div>
        </div>

        <div className="hw-panel__header-right">
          {/* Live ETA for current config */}
          {currentEstimate && currentEstimate.feasible && (
            <div className="hw-panel__eta">
              <Clock size={13} />
              <span>~{formatTime(currentEstimate.eta_seconds)}</span>
            </div>
          )}
          {currentProfile && !currentProfile.feasible && (
            <Badge variant="error" size="sm">Not Feasible</Badge>
          )}
          <span className="hw-panel__expand-icon">
            {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </span>
        </div>
      </button>

      {/* Expanded details */}
      {expanded && (
        <div className="hw-panel__details">
          {/* VRAM & RAM bars */}
          <div className="hw-panel__bars">
            <div className="hw-bar">
              <div className="hw-bar__label">
                <Cpu size={12} />
                <span>VRAM</span>
                <span className="hw-bar__value">{gpu.vram_gb} GB</span>
              </div>
              <div className="hw-bar__track">
                <div className="hw-bar__fill hw-bar__fill--vram" style={{ width: `${vramPercent}%` }} />
              </div>
            </div>
            <div className="hw-bar">
              <div className="hw-bar__label">
                <MemoryStick size={12} />
                <span>RAM</span>
                <span className="hw-bar__value">
                  {(gpu.ram_total_gb - gpu.ram_available_gb).toFixed(1)} / {gpu.ram_total_gb} GB
                </span>
              </div>
              <div className="hw-bar__track">
                <div className="hw-bar__fill hw-bar__fill--ram" style={{ width: `${ramPercent}%` }} />
              </div>
            </div>
          </div>

          {/* System info row */}
          {gpu.cuda_version && (
            <div className="hw-panel__sysinfo">
              <span>CUDA {gpu.cuda_version}</span>
              <span>·</span>
              <span>{feasibleCount}/{totalArchs} architectures supported</span>
            </div>
          )}

          {/* Architecture compatibility matrix */}
          <div className="hw-panel__compat">
            <div className="hw-panel__compat-title">
              <Gauge size={13} />
              <span>Architecture Compatibility</span>
            </div>
            <div className="hw-panel__compat-grid">
              {Object.entries(profiles).map(([arch, profile]) => {
                const est = estimates[arch];
                const isSelected = arch === config.baseModel;
                return (
                  <div
                    key={arch}
                    className={`hw-compat-item ${isSelected ? 'hw-compat-item--selected' : ''} ${!profile.feasible ? 'hw-compat-item--disabled' : ''}`}
                  >
                    <div className="hw-compat-item__top">
                      {profile.feasible ? (
                        <CheckCircle2 size={13} className="hw-compat-icon--ok" />
                      ) : (
                        <XCircle size={13} className="hw-compat-icon--no" />
                      )}
                      <span className="hw-compat-item__name">{ARCH_LABELS[arch] || arch}</span>
                    </div>
                    <div className="hw-compat-item__bottom">
                      {profile.feasible && est?.feasible ? (
                        <span className="hw-compat-item__eta">~{formatTime(est.eta_seconds)}</span>
                      ) : (
                        <span className="hw-compat-item__req">{profile.min_vram}GB+</span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Warning for current selection */}
          {currentProfile?.warning && (
            <div className="hw-panel__warning">
              <AlertTriangle size={13} />
              <span>{currentProfile.warning}</span>
            </div>
          )}

          {/* Detailed ETA for current config */}
          {currentEstimate?.feasible && (
            <div className="hw-panel__estimate">
              <Zap size={13} />
              <span>
                {config.trainingSteps.toLocaleString()} steps × {currentEstimate.time_per_step}s/step
                ≈ <strong>{formatTime(currentEstimate.eta_seconds)}</strong>
              </span>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
