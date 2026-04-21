import React, { useState, useEffect } from 'react';
import {
  Play,
  Image as ImageIcon,
  SlidersHorizontal,
  Loader2,
  Sparkles,
  Wand2,
  Copy,
  Shuffle,
  Info,
  Clock,
  Layers,
  Check,
  X,
} from 'lucide-react';
import { useLocation } from 'react-router-dom';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { generateImage, fetchModels, fetchAvailableBaseModels } from '../services/api';
import './PlaygroundPage.css';

interface GenHistory {
  id: string;
  url: string;
  seed: number;
  prompt: string;
  negativePrompt: string;
  width: number;
  height: number;
  cfgScale: number;
  steps: number;
  sampler: string;
  loraWeight: number;
  loraName: string;
  baseModelName: string;
  generatedAt: string;
  isMock?: boolean;
  mockReason?: string;
}

interface LoRAOption {
  id: string;
  name: string;
  filename: string;
  path: string;
  rank: number;
  architecture: string;
}

interface BaseModelOption {
  id: string;
  name: string;
  architecture: string;
}

export function PlaygroundPage() {
  const location = useLocation();

  // Generation params
  const [prompt, setPrompt] = useState('masterpiece, highly detailed, cyberpunk neon street, reflection in puddle');
  const [negativePrompt, setNegativePrompt] = useState('lowres, bad anatomy, text, error, worst quality, low quality');
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [cfgScale, setCfgScale] = useState(7.0);
  const [steps, setSteps] = useState(25);
  const [seed, setSeed] = useState(-1);
  const [loraWeight, setLoraWeight] = useState(1.0);
  const [sampler, setSampler] = useState('Euler a');

  // LoRA model selection
  const [availableModels, setAvailableModels] = useState<LoRAOption[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('none');
  const [availableBaseModels, setAvailableBaseModels] = useState<BaseModelOption[]>([]);
  const [selectedBaseModel, setSelectedBaseModel] = useState<string>('auto');
  const [modelsLoading, setModelsLoading] = useState(true);

  // Generation state
  const [isGenerating, setIsGenerating] = useState(false);
  const [history, setHistory] = useState<GenHistory[]>([]);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const [copiedSeed, setCopiedSeed] = useState(false);

  // Load available LoRA models
  useEffect(() => {
    const loadModels = async () => {
      try {
        setModelsLoading(true);
        const [loraModels, baseModels] = await Promise.all([
          fetchModels(),
          fetchAvailableBaseModels(),
        ]);
        setAvailableModels(loraModels.map((m: any) => ({
          id: m.id,
          name: m.name,
          filename: m.filename,
          path: m.path,
          rank: m.rank,
          architecture: m.architecture || '',
        })));
        setAvailableBaseModels(baseModels.map((m: any) => ({
          id: m.id,
          name: m.name || m.shortName,
          architecture: m.architecture,
        })));
        if (baseModels.length > 0) setSelectedBaseModel(baseModels[0].id);
      } catch (err) {
        console.error('Failed to load models:', err);
      } finally {
        setModelsLoading(false);
      }
    };
    loadModels();
  }, []);

  // Handle incoming model from Gallery "Test" button
  useEffect(() => {
    const state = location.state as any;
    if (state?.loraModel) {
      setSelectedModel(state.loraModel.id);
      // Clear the state to prevent re-selecting on navigation
      window.history.replaceState({}, document.title);
    }
  }, [location.state]);

  const handleGenerate = async () => {
    setIsGenerating(true);
    try {
      const actualSeed = seed === -1 ? Math.floor(Math.random() * 2147483647) : seed;
      const loraModelName = availableModels.find(m => m.id === selectedModel)?.name || 'None';
      const baseModelName = availableBaseModels.find(m => m.id === selectedBaseModel)?.name || 'Auto';
      const res = await generateImage({
        prompt,
        negativePrompt,
        width,
        height,
        seed: actualSeed,
        cfgScale,
        steps,
        loraWeight,
        sampler,
        loraModelId: selectedModel !== 'none' ? selectedModel : null,
        baseModelId: selectedBaseModel !== 'auto' ? selectedBaseModel : null,
      });

      const newGen: GenHistory = {
        id: Math.random().toString(36).substr(2, 9),
        url: res.url,
        seed: res.seed,
        prompt,
        negativePrompt,
        width,
        height,
        cfgScale,
        steps,
        sampler,
        loraWeight,
        loraName: loraModelName,
        baseModelName,
        generatedAt: new Date().toISOString(),
        isMock: res.mock,
        mockReason: res.reason,
      };
      setHistory(prev => [newGen, ...prev].slice(0, 30));
      setActiveIndex(0);
    } catch (err) {
      console.error('Generation failed:', err);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleRandomSeed = () => {
    setSeed(Math.floor(Math.random() * 2147483647));
  };

  const handleCopySeed = (s: number) => {
    navigator.clipboard.writeText(String(s));
    setCopiedSeed(true);
    setTimeout(() => setCopiedSeed(false), 1500);
  };

  const handleReuseSeed = (s: number) => {
    setSeed(s);
  };

  const handleReuseParams = (item: GenHistory) => {
    setPrompt(item.prompt);
    setNegativePrompt(item.negativePrompt);
    setWidth(item.width);
    setHeight(item.height);
    setCfgScale(item.cfgScale);
    setSteps(item.steps);
    setSampler(item.sampler);
    setLoraWeight(item.loraWeight);
    setSeed(item.seed);
  };

  const activeImage = activeIndex !== null ? history[activeIndex] : null;

  const resPresets = [
    { label: '512²', w: 512, h: 512 },
    { label: '768²', w: 768, h: 768 },
    { label: '1024²', w: 1024, h: 1024 },
    { label: '768×1024', w: 768, h: 1024 },
    { label: '1024×768', w: 1024, h: 768 },
  ];

  return (
    <div className="playground-page animate-fade-in">
      {/* Left Sidebar */}
      <div className="playground-sidebar">
        <div className="playground-sidebar__title">
          <Wand2 size={20} className="playground-sidebar__title-icon" />
          <h2>Inference Lab</h2>
        </div>

        <div className="playground-sidebar__inner">
          {/* Prompt */}
          <div className="pg-field">
            <label className="pg-label">Prompt</label>
            <textarea
              className="pg-textarea"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={3}
            />
          </div>

          {/* Negative Prompt */}
          <div className="pg-field">
            <label className="pg-label">Negative Prompt</label>
            <textarea
              className="pg-textarea pg-textarea--small"
              value={negativePrompt}
              onChange={(e) => setNegativePrompt(e.target.value)}
              rows={2}
            />
          </div>

          {/* LoRA Model Selector */}
          <Card padding="sm" className="pg-section-card">
            <div className="pg-section-header">
              <Layers size={14} />
              <span>LoRA Model</span>
            </div>
            {modelsLoading ? (
              <div className="pg-model-loading">
                <Loader2 size={14} className="animate-spin" /> Loading models...
              </div>
            ) : (
              <>
                {/* Base Model */}
                <div className="pg-field">
                  <label className="pg-label" style={{ fontSize: '10px' }}>Base Model</label>
                  <select
                    className="pg-select"
                    value={selectedBaseModel}
                    onChange={(e) => setSelectedBaseModel(e.target.value)}
                  >
                    <option value="auto">Auto-detect (any downloaded)</option>
                    {availableBaseModels.map(m => (
                      <option key={m.id} value={m.id}>{m.name} ({m.architecture.toUpperCase()})</option>
                    ))}
                  </select>
                </div>
                {/* LoRA */}
                <div className="pg-field">
                  <label className="pg-label" style={{ fontSize: '10px' }}>LoRA</label>
                  <select
                    className="pg-select"
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                  >
                    <option value="none">No LoRA (base model only)</option>
                    {availableModels.map(m => (
                      <option key={m.id} value={m.id}>
                        {m.name} {m.architecture ? `(${m.architecture.toUpperCase()})` : ''} — R{m.rank}
                      </option>
                    ))}
                  </select>
                </div>
              </>
            )}
            {selectedModel !== 'none' && (
              <div className="pg-lora-weight">
                <div className="pg-param-header">
                  <span>Weight</span>
                  <span className="pg-param-value">{loraWeight.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  className="pg-slider"
                  min={-1.0}
                  max={2.0}
                  step={0.05}
                  value={loraWeight}
                  onChange={(e) => setLoraWeight(Number(e.target.value))}
                />
              </div>
            )}
          </Card>

          {/* Resolution presets */}
          <Card padding="sm" className="pg-section-card">
            <div className="pg-section-header">
              <ImageIcon size={14} />
              <span>Resolution</span>
            </div>
            <div className="pg-res-presets">
              {resPresets.map(p => (
                <button
                  key={p.label}
                  className={`pg-res-btn ${width === p.w && height === p.h ? 'pg-res-btn--active' : ''}`}
                  onClick={() => { setWidth(p.w); setHeight(p.h); }}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </Card>

          {/* Advanced Engine */}
          <Card padding="sm" className="pg-section-card">
            <div className="pg-section-header">
              <SlidersHorizontal size={14} />
              <span>Advanced</span>
            </div>

            <div className="pg-params-grid">
              <div className="pg-param">
                <div className="pg-param-header">
                  <span>CFG Scale</span>
                  <span className="pg-param-value">{cfgScale.toFixed(1)}</span>
                </div>
                <input type="range" className="pg-slider" min={1} max={20} step={0.5} value={cfgScale} onChange={(e) => setCfgScale(Number(e.target.value))} />
              </div>

              <div className="pg-param">
                <div className="pg-param-header">
                  <span>Steps</span>
                  <span className="pg-param-value">{steps}</span>
                </div>
                <input type="range" className="pg-slider" min={10} max={100} step={1} value={steps} onChange={(e) => setSteps(Number(e.target.value))} />
              </div>
            </div>

            <div className="pg-param" style={{ marginTop: '12px' }}>
              <label className="pg-label" style={{ fontSize: '11px' }}>Sampler</label>
              <select className="pg-select" value={sampler} onChange={(e) => setSampler(e.target.value)}>
                <option>Euler a</option>
                <option>DPM++ 2M Karras</option>
                <option>DPM++ SDE Karras</option>
              </select>
            </div>

            <div className="pg-seed-row" style={{ marginTop: '12px' }}>
              <div style={{ flex: 1 }}>
                <label className="pg-label" style={{ fontSize: '11px' }}>Seed</label>
                <input
                  type="number"
                  className="pg-select"
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                  placeholder="-1 for random"
                />
              </div>
              <button className="pg-seed-btn" onClick={handleRandomSeed} title="Random seed">
                <Shuffle size={14} />
              </button>
              <button className="pg-seed-btn" onClick={() => setSeed(-1)} title="Reset to random">
                <X size={14} />
              </button>
            </div>
          </Card>
        </div>

        {/* Generate button */}
        <Button
          variant="primary"
          size="lg"
          icon={isGenerating ? <Loader2 size={18} className="animate-spin" /> : <Play size={18} />}
          onClick={handleGenerate}
          disabled={isGenerating || !prompt}
          style={{ flexShrink: 0, marginTop: '8px' }}
        >
          {isGenerating ? 'Generating...' : 'Generate'}
        </Button>
      </div>

      {/* Main Canvas Area */}
      <div className="playground-main">
        <div className="playground-canvas">
          {isGenerating ? (
            <div className="playground-empty">
              <Loader2 size={48} className="animate-spin" style={{ color: 'var(--color-accent)' }} />
              <p>Warming up Neural Engine...</p>
            </div>
          ) : activeImage ? (
            <>
              <img src={activeImage.url} alt="Generated" className="playground-result" />
              {activeImage.isMock && (
                <div className="playground-mock-badge">
                  <span>⚠ Simulated — {activeImage.mockReason}</span>
                </div>
              )}
            </>
          ) : (
            <div className="playground-empty">
              <div className="playground-empty__icon">
                <ImageIcon size={48} />
              </div>
              <p className="playground-empty__title">Enter a prompt and generate</p>
              <p className="playground-empty__subtitle">Results will appear here</p>
            </div>
          )}
        </div>

        {/* Metadata panel under image */}
        {activeImage && !isGenerating && (
          <div className="playground-meta">
            <div className="playground-meta__left">
              <div className="playground-meta__item">
                <span className="playground-meta__label">Seed</span>
                <button className="playground-meta__seed" onClick={() => handleCopySeed(activeImage.seed)}>
                  {activeImage.seed}
                  {copiedSeed ? <Check size={12} /> : <Copy size={12} />}
                </button>
              </div>
              <div className="playground-meta__item">
                <span className="playground-meta__label">Size</span>
                <span className="playground-meta__value">{activeImage.width}×{activeImage.height}</span>
              </div>
              <div className="playground-meta__item">
                <span className="playground-meta__label">CFG</span>
                <span className="playground-meta__value">{activeImage.cfgScale}</span>
              </div>
              <div className="playground-meta__item">
                <span className="playground-meta__label">Steps</span>
                <span className="playground-meta__value">{activeImage.steps}</span>
              </div>
              <div className="playground-meta__item">
                <span className="playground-meta__label">Sampler</span>
                <span className="playground-meta__value">{activeImage.sampler}</span>
              </div>
              {activeImage.loraName !== 'None' && (
                <div className="playground-meta__item">
                  <span className="playground-meta__label">LoRA</span>
                  <span className="playground-meta__value">{activeImage.loraName} ({activeImage.loraWeight.toFixed(2)})</span>
                </div>
              )}
              <div className="playground-meta__item">
                <span className="playground-meta__label">Base</span>
                <span className="playground-meta__value">{activeImage.baseModelName}</span>
              </div>
            </div>
            <div className="playground-meta__right">
              <button className="pg-meta-action" onClick={() => handleReuseSeed(activeImage.seed)} title="Reuse this seed">
                <Shuffle size={13} /> Reuse Seed
              </button>
              <button className="pg-meta-action" onClick={() => handleReuseParams(activeImage)} title="Reuse all parameters">
                <SlidersHorizontal size={13} /> Reuse All
              </button>
            </div>
          </div>
        )}

        {/* History Strip */}
        {history.length > 0 && (
          <div className="playground-history">
            {history.map((item, idx) => (
              <div
                key={item.id}
                className={`history-thumb-wrap ${activeIndex === idx ? 'history-thumb-wrap--active' : ''}`}
                onClick={() => setActiveIndex(idx)}
              >
                <img
                  src={item.url}
                  className="history-thumbnail"
                  alt={`Gen ${idx + 1}`}
                />
                <span className="history-thumb-seed">#{item.seed}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
