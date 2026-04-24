import React, { useState } from 'react';
import {
  Zap,
  Shield,
  Gauge,
  ChevronDown,
  Info,
  SlidersHorizontal,
} from 'lucide-react';
import { Badge } from '../ui/Badge';
import { useApp } from '../../context/AppContext';
import { defaultConfig } from '../../context/AppContext';
import type { TrainingConfig, OptimizerType, SchedulerType, ResolutionType, BaseModelType, ConfigPreset } from '../../types';

const presets: ConfigPreset[] = [
  {
    id: 'quality', name: 'Quality', description: 'Higher rank, more steps', icon: 'shield',
    config: { learningRate: 5e-5, trainingSteps: 3000, loraRank: 32, networkAlpha: 16, batchSize: 1, optimizer: 'AdamW', scheduler: 'cosine' },
  },
  {
    id: 'balanced', name: 'Balanced', description: 'Good quality & speed', icon: 'gauge',
    config: { learningRate: 1e-4, trainingSteps: 1500, loraRank: 16, networkAlpha: 8, batchSize: 1, optimizer: 'AdamW', scheduler: 'cosine' },
  },
  {
    id: 'fast', name: 'Fast', description: 'Quick iteration', icon: 'zap',
    config: { learningRate: 2e-4, trainingSteps: 500, loraRank: 8, networkAlpha: 4, batchSize: 2, optimizer: 'Prodigy', scheduler: 'constant' },
  },
];

const presetIcons: Record<string, React.ReactNode> = {
  shield: <Shield size={14} />,
  gauge: <Gauge size={14} />,
  zap: <Zap size={14} />,
};

interface ParamSliderProps {
  label: string; value: number; min: number; max: number; step: number;
  unit?: string; tooltip?: string; logarithmic?: boolean;
  onChange: (value: number) => void;
}

function ParamSlider({ label, value, min, max, step, unit = '', tooltip, logarithmic = false, onChange }: ParamSliderProps) {
  const displayValue = logarithmic ? value.toExponential(1) : value;
  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (logarithmic) {
      const logMin = Math.log10(min);
      const logMax = Math.log10(max);
      const logValue = logMin + (parseFloat(e.target.value) / 100) * (logMax - logMin);
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
  const percentage = logarithmic ? getSliderValue() : ((value - min) / (max - min)) * 100;

  return (
    <div className="param-slider">
      <div className="param-slider__header">
        <label className="param-slider__label">
          {label}
          {tooltip && <span className="param-slider__tooltip" title={tooltip}><Info size={12} /></span>}
        </label>
        <span className="param-slider__value">
          {displayValue}
          {unit && <span className="param-slider__unit">{unit}</span>}
        </span>
      </div>
      <div className="param-slider__track-wrapper">
        <input
          type="range" className="param-slider__input"
          min={sliderMin} max={sliderMax} step={sliderStep}
          value={getSliderValue()} onChange={handleSliderChange}
          style={{ '--fill-percent': `${percentage}%` } as React.CSSProperties}
        />
      </div>
    </div>
  );
}

interface ParamSelectProps {
  label: string; value: string | number;
  options: { value: string | number; label: string }[];
  tooltip?: string; onChange: (value: string) => void;
}

function ParamSelect({ label, value, options, tooltip, onChange }: ParamSelectProps) {
  return (
    <div className="param-select">
      <label className="param-select__label">
        {label}
        {tooltip && <span className="param-slider__tooltip" title={tooltip}><Info size={12} /></span>}
      </label>
      <div className="param-select__wrapper">
        <select className="param-select__input" value={value} onChange={(e) => onChange(e.target.value)}>
          {options.map((opt) => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
        </select>
        <ChevronDown size={14} className="param-select__chevron" />
      </div>
    </div>
  );
}

interface ConfigSectionProps {
  disabled?: boolean;
}

export function ConfigSection({ disabled = false }: ConfigSectionProps) {
  const { state, dispatch } = useApp();
  const config = state.trainingConfig;
  const [isOpen, setIsOpen] = useState(true);
  const [activePreset, setActivePreset] = useState<string | null>('balanced');
  const [showAdvanced, setShowAdvanced] = useState(false);

  const updateConfig = (updates: Partial<TrainingConfig>) => {
    dispatch({ type: 'UPDATE_CONFIG', payload: updates });
    setActivePreset(null);
  };

  const applyPreset = (preset: ConfigPreset) => {
    dispatch({ type: 'UPDATE_CONFIG', payload: preset.config });
    setActivePreset(preset.id);
  };

  return (
    <div className={`ws-section ${disabled ? 'ws-section--disabled' : ''}`}>
      <button className="ws-section__toggle" onClick={() => setIsOpen(!isOpen)}>
        <span className="ws-section__toggle-icon"><SlidersHorizontal size={16} /></span>
        <span className="ws-section__toggle-title">Configuration</span>
        <span className={`ws-section__toggle-chevron ${isOpen ? 'ws-section__toggle-chevron--open' : ''}`}>
          <ChevronDown size={14} />
        </span>
      </button>

      {isOpen && (
        <div className="ws-section__body">
          {/* Presets */}
          <div className="ws-config-presets">
            {presets.map((preset) => (
              <button
                key={preset.id}
                className={`ws-preset-btn ${activePreset === preset.id ? 'ws-preset-btn--active' : ''}`}
                onClick={() => applyPreset(preset)}
              >
                {presetIcons[preset.icon]}
                {preset.name}
              </button>
            ))}
          </div>

          <div className="ws-config-grid">
            {/* Basic */}
            <div className="ws-config-subsection">Basic</div>
            <ParamSelect label="Base Model" value={config.baseModel}
              options={[
                { value: 'sd15', label: 'SD 1.5 — 512px' },
                { value: 'sd21', label: 'SD 2.1 — 768px' },
                { value: 'sdxl', label: 'SDXL — 1024px' },
                { value: 'sd3', label: 'SD3 — 1024px' },
                { value: 'flux', label: 'Flux.1 — 1024px' },
                { value: 'cascade', label: 'Cascade — 1024px' },
                { value: 'hunyuan', label: 'HunyuanDiT — 1024px' },
                { value: 'pixart', label: 'PixArt-Σ — 1024px' },
                { value: 'kolors', label: 'Kolors — 1024px' },
                { value: 'auraflow', label: 'AuraFlow — 1024px' },
              ]}
              tooltip="Foundation model architecture"
              onChange={(v) => updateConfig({ baseModel: v as BaseModelType })}
            />
            <ParamSlider label="Learning Rate" value={config.learningRate}
              min={1e-6} max={1e-2} step={1e-6} logarithmic
              tooltip="Controls how fast the model learns"
              onChange={(v) => updateConfig({ learningRate: v })}
            />
            <ParamSlider label="Training Steps" value={config.trainingSteps}
              min={100} max={10000} step={100}
              tooltip="Total optimizer steps"
              onChange={(v) => updateConfig({ trainingSteps: v })}
            />
            <ParamSlider label="LoRA Rank" value={config.loraRank}
              min={4} max={128} step={4}
              tooltip="Dimensionality of LoRA matrices"
              onChange={(v) => updateConfig({ loraRank: v })}
            />
            <ParamSlider label="Network Alpha" value={config.networkAlpha}
              min={1} max={128} step={1}
              tooltip="Scaling factor, usually rank/2"
              onChange={(v) => updateConfig({ networkAlpha: v })}
            />
            <ParamSlider label="Batch Size" value={config.batchSize}
              min={1} max={8} step={1}
              tooltip="Images per step"
              onChange={(v) => updateConfig({ batchSize: v })}
            />
            <ParamSelect label="Resolution" value={config.resolution}
              options={[
                { value: 512, label: '512 × 512' },
                { value: 768, label: '768 × 768' },
                { value: 1024, label: '1024 × 1024' },
                { value: 1536, label: '1536 × 1536' },
              ]}
              tooltip="Training resolution"
              onChange={(v) => updateConfig({ resolution: parseInt(v) as ResolutionType })}
            />

            {/* Optimizer */}
            <div className="ws-config-subsection">Optimizer</div>
            <ParamSelect label="Optimizer" value={config.optimizer}
              options={[
                { value: 'AdamW', label: 'AdamW' },
                { value: 'Prodigy', label: 'Prodigy' },
                { value: 'DAdaptAdam', label: 'D-Adapt Adam' },
              ]}
              tooltip="Optimizer algorithm"
              onChange={(v) => updateConfig({ optimizer: v as OptimizerType })}
            />
            <ParamSelect label="LR Scheduler" value={config.scheduler}
              options={[
                { value: 'cosine', label: 'Cosine' },
                { value: 'linear', label: 'Linear' },
                { value: 'constant', label: 'Constant' },
                { value: 'cosine_with_restarts', label: 'Cosine w/ Restarts' },
              ]}
              tooltip="How LR changes during training"
              onChange={(v) => updateConfig({ scheduler: v as SchedulerType })}
            />
            <ParamSlider label="Warmup Steps" value={config.warmupSteps}
              min={0} max={500} step={10}
              tooltip="Steps to gradually increase LR"
              onChange={(v) => updateConfig({ warmupSteps: v })}
            />

            {/* Advanced toggle */}
            <button
              className="ws-config-subsection"
              style={{ cursor: 'pointer', border: 'none', background: 'none', width: '100%' }}
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              Advanced
              <Badge variant="default" size="sm">{showAdvanced ? 'Hide' : 'Show'}</Badge>
            </button>

            {showAdvanced && (
              <>
                <ParamSelect label="AR Bucketing" value={config.enableBucketing ? 'true' : 'false'}
                  options={[
                    { value: 'true', label: 'Enabled' },
                    { value: 'false', label: 'Disabled' },
                  ]}
                  tooltip="Aspect Ratio Bucketing. Groups images by ratio to prevent distortion."
                  onChange={(v) => updateConfig({ enableBucketing: v === 'true' })}
                />
                <ParamSlider label="Caption Dropout" value={config.captionDropout ?? 0.1}
                  min={0} max={1} step={0.05}
                  tooltip="Probability to ignore captions. Forces the model to learn the subject independently of text."
                  onChange={(v) => updateConfig({ captionDropout: v })}
                />
                <ParamSlider label="Noise Offset" value={config.noiseOffset ?? 0.05}
                  min={0} max={0.2} step={0.01}
                  tooltip="Adds noise offset to improve the model's ability to generate very dark or bright images."
                  onChange={(v) => updateConfig({ noiseOffset: v })}
                />
                <ParamSlider label="Seed" value={config.seed}
                  min={0} max={99999} step={1}
                  tooltip="Random seed for reproducibility"
                  onChange={(v) => updateConfig({ seed: v })}
                />
                <ParamSelect label="Mixed Precision" value={config.mixedPrecision}
                  options={[
                    { value: 'bf16', label: 'BFloat16' },
                    { value: 'fp16', label: 'Float16' },
                    { value: 'fp32', label: 'Float32' },
                  ]}
                  tooltip="Training precision"
                  onChange={(v) => updateConfig({ mixedPrecision: v as 'fp16' | 'bf16' | 'fp32' })}
                />
                <ParamSlider label="Gradient Accum." value={config.gradientAccumulation}
                  min={1} max={8} step={1}
                  tooltip="Simulate larger batch sizes"
                  onChange={(v) => updateConfig({ gradientAccumulation: v })}
                />
                <ParamSlider label="CLIP Skip" value={config.clipSkip}
                  min={1} max={4} step={1}
                  tooltip="Skip last N CLIP layers"
                  onChange={(v) => updateConfig({ clipSkip: v })}
                />
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
