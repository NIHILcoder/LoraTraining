import React, { useState } from 'react';
import {
  Zap,
  Shield,
  Gauge,
  Save,
  RotateCcw,
  ChevronDown,
  ChevronUp,
  Info,
} from 'lucide-react';
import { Header } from '../components/layout/Header';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { useApp } from '../context/AppContext';
import { defaultConfig } from '../context/AppContext';
import type { TrainingConfig, OptimizerType, SchedulerType, ResolutionType, BaseModelType, ConfigPreset } from '../types';
import './ConfigPage.css';

const presets: ConfigPreset[] = [
  {
    id: 'quality',
    name: 'Quality',
    description: 'Higher rank, more steps — best results',
    icon: 'shield',
    config: {
      learningRate: 5e-5,
      trainingSteps: 3000,
      loraRank: 32,
      networkAlpha: 16,
      batchSize: 1,
      optimizer: 'AdamW',
      scheduler: 'cosine',
    },
  },
  {
    id: 'balanced',
    name: 'Balanced',
    description: 'Good quality with reasonable speed',
    icon: 'gauge',
    config: {
      learningRate: 1e-4,
      trainingSteps: 1500,
      loraRank: 16,
      networkAlpha: 8,
      batchSize: 1,
      optimizer: 'AdamW',
      scheduler: 'cosine',
    },
  },
  {
    id: 'fast',
    name: 'Fast',
    description: 'Quick training for iteration',
    icon: 'zap',
    config: {
      learningRate: 2e-4,
      trainingSteps: 500,
      loraRank: 8,
      networkAlpha: 4,
      batchSize: 2,
      optimizer: 'AdamW',
      scheduler: 'constant',
    },
  },
];

const presetIcons: Record<string, React.ReactNode> = {
  shield: <Shield size={20} />,
  gauge: <Gauge size={20} />,
  zap: <Zap size={20} />,
};

interface ParamSliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  tooltip?: string;
  logarithmic?: boolean;
  onChange: (value: number) => void;
}

function ParamSlider({
  label,
  value,
  min,
  max,
  step,
  unit = '',
  tooltip,
  logarithmic = false,
  onChange,
}: ParamSliderProps) {
  const displayValue = logarithmic ? value.toExponential(1) : value;

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (logarithmic) {
      const logMin = Math.log10(min);
      const logMax = Math.log10(max);
      const logValue = logMin + (parseFloat(e.target.value) / 100) * (logMax - logMin);
      // Round to nearest step in log space
      const rawValue = Math.pow(10, logValue);
      onChange(parseFloat(rawValue.toExponential(1)));
    } else {
      onChange(parseFloat(e.target.value));
    }
  };

  const getSliderValue = () => {
    if (logarithmic) {
      const logMin = Math.log10(min);
      const logMax = Math.log10(max);
      const logVal = Math.log10(value);
      return ((logVal - logMin) / (logMax - logMin)) * 100;
    }
    return value;
  };

  const sliderMin = logarithmic ? 0 : min;
  const sliderMax = logarithmic ? 100 : max;
  const sliderStep = logarithmic ? 0.1 : step;

  const percentage = logarithmic
    ? getSliderValue()
    : ((value - min) / (max - min)) * 100;

  return (
    <div className="param-slider">
      <div className="param-slider__header">
        <label className="param-slider__label">
          {label}
          {tooltip && (
            <span className="param-slider__tooltip" title={tooltip}>
              <Info size={12} />
            </span>
          )}
        </label>
        <span className="param-slider__value">
          {displayValue}
          {unit && <span className="param-slider__unit">{unit}</span>}
        </span>
      </div>
      <div className="param-slider__track-wrapper">
        <input
          type="range"
          className="param-slider__input"
          min={sliderMin}
          max={sliderMax}
          step={sliderStep}
          value={getSliderValue()}
          onChange={handleSliderChange}
          style={{ '--fill-percent': `${percentage}%` } as React.CSSProperties}
        />
      </div>
    </div>
  );
}

interface ParamSelectProps {
  label: string;
  value: string | number;
  options: { value: string | number; label: string }[];
  tooltip?: string;
  onChange: (value: string) => void;
}

function ParamSelect({ label, value, options, tooltip, onChange }: ParamSelectProps) {
  return (
    <div className="param-select">
      <label className="param-select__label">
        {label}
        {tooltip && (
          <span className="param-slider__tooltip" title={tooltip}>
            <Info size={12} />
          </span>
        )}
      </label>
      <div className="param-select__wrapper">
        <select
          className="param-select__input"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        >
          {options.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
        <ChevronDown size={14} className="param-select__chevron" />
      </div>
    </div>
  );
}

export function ConfigPage() {
  const { state, dispatch } = useApp();
  const config = state.trainingConfig;
  const [expandedSections, setExpandedSections] = useState({
    basic: true,
    optimizer: true,
    advanced: false,
  });
  const [activePreset, setActivePreset] = useState<string | null>('balanced');

  const updateConfig = (updates: Partial<TrainingConfig>) => {
    dispatch({ type: 'UPDATE_CONFIG', payload: updates });
    setActivePreset(null);
  };

  const applyPreset = (preset: ConfigPreset) => {
    dispatch({ type: 'UPDATE_CONFIG', payload: preset.config });
    setActivePreset(preset.id);
  };

  const resetConfig = () => {
    dispatch({ type: 'UPDATE_CONFIG', payload: defaultConfig });
    setActivePreset(null);
  };

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections((prev) => ({ ...prev, [section]: !prev[section] }));
  };

  return (
    <div className="config-page animate-fade-in-up">
      <Header
        title="Configuration"
        subtitle="Set hyperparameters for your LoRA training"
        actions={
          <>
            <Button
              variant="ghost"
              size="sm"
              icon={<RotateCcw size={14} />}
              onClick={resetConfig}
            >
              Reset
            </Button>
            <Button
              variant="primary"
              size="sm"
              icon={<Save size={14} />}
              onClick={() => {}}
            >
              Save Preset
            </Button>
          </>
        }
      />

      {/* Presets */}
      <div className="config-page__presets stagger-children">
        {presets.map((preset) => (
          <Card
            key={preset.id}
            hover
            glow={activePreset === preset.id}
            className={`preset-card ${activePreset === preset.id ? 'preset-card--active' : ''}`}
            onClick={() => applyPreset(preset)}
          >
            <div className="preset-card__icon">
              {presetIcons[preset.icon]}
            </div>
            <div className="preset-card__info">
              <span className="preset-card__name">{preset.name}</span>
              <span className="preset-card__desc">{preset.description}</span>
            </div>
            {activePreset === preset.id && (
              <Badge variant="accent" size="sm">Active</Badge>
            )}
          </Card>
        ))}
      </div>

      {/* Basic Parameters */}
      <div className="config-section">
        <button
          className="config-section__header"
          onClick={() => toggleSection('basic')}
        >
          <span className="config-section__title">Basic Parameters</span>
          {expandedSections.basic ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
        {expandedSections.basic && (
          <Card className="config-section__body animate-fade-in">
            <div className="config-grid">
              <ParamSelect
                label="Base Model Architecture"
                value={config.baseModel}
                options={[
                  { value: 'sd15', label: 'Stable Diffusion 1.5' },
                  { value: 'sdxl', label: 'Stable Diffusion XL' },
                  { value: 'flux', label: 'Flux.1 (Dev)' },
                ]}
                tooltip="The foundation model you are training on. This affects optimal parameters."
                onChange={(v) => updateConfig({ baseModel: v as BaseModelType })}
              />
              <ParamSlider
                label="Learning Rate"
                value={config.learningRate}
                min={1e-6}
                max={1e-2}
                step={1e-6}
                logarithmic
                tooltip="Controls how fast the model learns. Lower = more stable, higher = faster convergence."
                onChange={(v) => updateConfig({ learningRate: v })}
              />
              <ParamSlider
                label="Training Steps"
                value={config.trainingSteps}
                min={100}
                max={10000}
                step={100}
                tooltip="Total number of optimizer steps. More steps = better quality but longer training."
                onChange={(v) => updateConfig({ trainingSteps: v })}
              />
              <ParamSlider
                label="LoRA Rank"
                value={config.loraRank}
                min={4}
                max={128}
                step={4}
                tooltip="Dimensionality of LoRA matrices. Higher = more capacity but larger file size."
                onChange={(v) => updateConfig({ loraRank: v })}
              />
              <ParamSlider
                label="Network Alpha"
                value={config.networkAlpha}
                min={1}
                max={128}
                step={1}
                tooltip="Scaling factor for LoRA weights. Usually set to rank/2."
                onChange={(v) => updateConfig({ networkAlpha: v })}
              />
              <ParamSlider
                label="Batch Size"
                value={config.batchSize}
                min={1}
                max={8}
                step={1}
                tooltip="Number of images processed per step. Higher = more VRAM needed."
                onChange={(v) => updateConfig({ batchSize: v })}
              />
              <ParamSelect
                label="Resolution"
                value={config.resolution}
                options={[
                  { value: 512, label: '512 × 512' },
                  { value: 768, label: '768 × 768' },
                  { value: 1024, label: '1024 × 1024' },
                ]}
                tooltip="Training resolution. Match to your target generation size."
                onChange={(v) => updateConfig({ resolution: parseInt(v) as ResolutionType })}
              />
            </div>
          </Card>
        )}
      </div>

      {/* Optimizer Settings */}
      <div className="config-section">
        <button
          className="config-section__header"
          onClick={() => toggleSection('optimizer')}
        >
          <span className="config-section__title">Optimizer & Scheduler</span>
          {expandedSections.optimizer ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
        {expandedSections.optimizer && (
          <Card className="config-section__body animate-fade-in">
            <div className="config-grid">
              <ParamSelect
                label="Optimizer"
                value={config.optimizer}
                options={[
                  { value: 'AdamW', label: 'AdamW' },
                ]}
                tooltip="Optimizer algorithm."
                onChange={(v) => updateConfig({ optimizer: v as OptimizerType })}
              />
              <ParamSelect
                label="LR Scheduler"
                value={config.scheduler}
                options={[
                  { value: 'cosine', label: 'Cosine Annealing' },
                  { value: 'constant', label: 'Constant' },
                ]}
                tooltip="How learning rate changes during training."
                onChange={(v) => updateConfig({ scheduler: v as SchedulerType })}
              />
              <ParamSlider
                label="Warmup Steps"
                value={config.warmupSteps}
                min={0}
                max={500}
                step={10}
                tooltip="Steps to gradually increase LR from 0 to target."
                onChange={(v) => updateConfig({ warmupSteps: v })}
              />
            </div>
          </Card>
        )}
      </div>

      {/* Advanced */}
      <div className="config-section">
        <button
          className="config-section__header"
          onClick={() => toggleSection('advanced')}
        >
          <span className="config-section__title">Advanced Settings</span>
          <Badge variant="default" size="sm">Optional</Badge>
          {expandedSections.advanced ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
        {expandedSections.advanced && (
          <Card className="config-section__body animate-fade-in">
            <div className="config-grid">
              <ParamSlider
                label="Seed"
                value={config.seed}
                min={0}
                max={99999}
                step={1}
                tooltip="Random seed for reproducibility."
                onChange={(v) => updateConfig({ seed: v })}
              />
              <ParamSelect
                label="Mixed Precision"
                value={config.mixedPrecision}
                options={[
                  { value: 'bf16', label: 'BFloat16 (recommended)' },
                  { value: 'fp16', label: 'Float16' },
                  { value: 'fp32', label: 'Float32 (slow)' },
                ]}
                tooltip="Precision for training. bf16 is best for modern GPUs."
                onChange={(v) => updateConfig({ mixedPrecision: v as 'fp16' | 'bf16' | 'fp32' })}
              />
              <ParamSlider
                label="Gradient Accumulation"
                value={config.gradientAccumulation}
                min={1}
                max={8}
                step={1}
                tooltip="Simulate larger batch sizes without extra VRAM."
                onChange={(v) => updateConfig({ gradientAccumulation: v })}
              />
              <ParamSlider
                label="CLIP Skip"
                value={config.clipSkip}
                min={1}
                max={4}
                step={1}
                tooltip="Skip last N layers of CLIP. 1 = no skip."
                onChange={(v) => updateConfig({ clipSkip: v })}
              />
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}
