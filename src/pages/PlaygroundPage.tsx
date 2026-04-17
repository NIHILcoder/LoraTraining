import React, { useState } from 'react';
import { Play, Image as ImageIcon, SlidersHorizontal, Loader2, Sparkles, Wand2 } from 'lucide-react';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { generateImage } from '../services/api';
import './PlaygroundPage.css';

interface GenHistory {
  id: string;
  url: string;
  seed: number;
  prompt: string;
}

export function PlaygroundPage() {
  const [prompt, setPrompt] = useState('masterpiece, highly detailed, cyberpunk neon street, reflection in puddle');
  const [negativePrompt, setNegativePrompt] = useState('lowres, bad anatomy, text, error, worst quality, low quality');
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [cfgScale, setCfgScale] = useState(7.0);
  const [steps, setSteps] = useState(25);
  const [seed, setSeed] = useState(-1);
  const [loraWeight, setLoraWeight] = useState(1.0);
  const [sampler, setSampler] = useState('Euler a');
  
  const [isGenerating, setIsGenerating] = useState(false);
  const [history, setHistory] = useState<GenHistory[]>([]);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);

  const handleGenerate = async () => {
    setIsGenerating(true);
    try {
      const actualSeed = seed === -1 ? Math.floor(Math.random() * 2147483647) : seed;
      const res = await generateImage({
        prompt,
        negativePrompt,
        width,
        height,
        seed: actualSeed,
        cfgScale,
        steps,
        loraWeight,
        sampler
      });
      
      const newGen = { id: Math.random().toString(), url: res.url, seed: res.seed, prompt };
      setHistory(prev => [newGen, ...prev].slice(0, 20)); // Keep last 20
      setActiveIndex(0);
      
    } catch (err) {
      console.error('Generation failed:', err);
    } finally {
      setIsGenerating(false);
    }
  };

  const activeImage = activeIndex !== null ? history[activeIndex] : null;

  return (
    <div className="playground-page animate-fade-in">
      {/* Left Sidebar for Constraints */}
      <div className="playground-sidebar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
          <Wand2 size={20} color="var(--color-accent)" />
          <h2 style={{ fontSize: 'var(--text-lg)', fontWeight: 600 }}>Inference Lab</h2>
        </div>
        
        <div className="playground-sidebar__inner">
          <div>
            <label className="playground-label">Prompt</label>
            <textarea 
              className="playground-textarea" 
              value={prompt} 
              onChange={(e) => setPrompt(e.target.value)}
            />
          </div>

          <div>
            <label className="playground-label">Negative Prompt</label>
            <textarea 
              className="playground-textarea" 
              style={{ minHeight: '40px' }}
              value={negativePrompt} 
              onChange={(e) => setNegativePrompt(e.target.value)}
            />
          </div>

          <Card padding="sm" style={{ padding: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px', color: 'var(--color-text-secondary)', fontWeight: 600, fontSize: '13px' }}>
              <Sparkles size={14} /> LoRA Injection
            </div>
            <div className="playground-param-compact">
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                <span style={{ color: 'var(--color-text-secondary)' }}>Model Weight</span>
                <span style={{ color: 'var(--color-accent)', fontWeight: 600 }}>{loraWeight.toFixed(2)}</span>
              </div>
              <input type="range" className="param-slider__input" min={-1.0} max={2.0} step={0.05} value={loraWeight} onChange={(e) => setLoraWeight(Number(e.target.value))} />
            </div>
          </Card>

          <Card padding="sm" style={{ padding: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', color: 'var(--color-text-secondary)', fontWeight: 600, fontSize: '13px' }}>
              <SlidersHorizontal size={14} /> Advanced Engine
            </div>
            
            <div className="playground-params-grid">
              <div className="playground-param-compact">
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px' }}>
                  <span style={{ color: 'var(--color-text-secondary)' }}>Width</span>
                  <span style={{ color: 'var(--color-text-primary)' }}>{width}</span>
                </div>
                <input type="range" className="param-slider__input" min={512} max={1536} step={64} value={width} onChange={(e) => setWidth(Number(e.target.value))} />
              </div>

              <div className="playground-param-compact">
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px' }}>
                  <span style={{ color: 'var(--color-text-secondary)' }}>Height</span>
                  <span style={{ color: 'var(--color-text-primary)' }}>{height}</span>
                </div>
                <input type="range" className="param-slider__input" min={512} max={1536} step={64} value={height} onChange={(e) => setHeight(Number(e.target.value))} />
              </div>

              <div className="playground-param-compact">
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px' }}>
                  <span style={{ color: 'var(--color-text-secondary)' }}>CFG</span>
                  <span style={{ color: 'var(--color-text-primary)' }}>{cfgScale.toFixed(1)}</span>
                </div>
                <input type="range" className="param-slider__input" min={1} max={20} step={0.5} value={cfgScale} onChange={(e) => setCfgScale(Number(e.target.value))} />
              </div>

              <div className="playground-param-compact">
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px' }}>
                  <span style={{ color: 'var(--color-text-secondary)' }}>Steps</span>
                  <span style={{ color: 'var(--color-text-primary)' }}>{steps}</span>
                </div>
                <input type="range" className="param-slider__input" min={10} max={100} step={1} value={steps} onChange={(e) => setSteps(Number(e.target.value))} />
              </div>
            </div>

            <div style={{ marginTop: '12px' }}>
              <label className="playground-label" style={{ fontSize: '11px' }}>Sampler</label>
              <select className="param-select__input" value={sampler} onChange={(e) => setSampler(e.target.value)}>
                <option>Euler a</option>
                <option>DPM++ 2M Karras</option>
                <option>DPM++ SDE Karras</option>
              </select>
            </div>
            
            <div style={{ marginTop: '12px' }}>
              <label className="playground-label" style={{ fontSize: '11px' }}>Seed (-1 for random)</label>
              <input type="number" className="param-select__input" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
            </div>
          </Card>
        </div>

        <Button 
          variant="primary" 
          size="lg" 
          icon={isGenerating ? <Loader2 size={18} className="animate-spin" /> : <Play size={18} />}
          onClick={handleGenerate}
          disabled={isGenerating || !prompt}
          style={{ flexShrink: 0, marginTop: '8px' }}
        >
          {isGenerating ? 'Generating...' : 'Generate Output'}
        </Button>
      </div>

      {/* Main Canvas Area */}
      <div className="playground-main" style={{ backgroundImage: 'radial-gradient(var(--color-surface-border) 1px, transparent 1px)', backgroundSize: '16px 16px' }}>
        <div className="playground-canvas">
          {isGenerating ? (
            <div className="playground-empty">
              <Loader2 size={48} className="animate-spin" style={{ color: 'var(--color-accent)' }} />
              <p>Warming up Neural Engine...</p>
            </div>
          ) : activeImage ? (
            <img src={activeImage.url} alt="Generated" className="playground-result" />
          ) : (
            <div className="playground-empty">
              <ImageIcon size={64} style={{ opacity: 0.2 }} />
              <p>Enter a prompt and generate to see results</p>
            </div>
          )}
        </div>
        
        {/* Gallery Filmstrip */}
        {history.length > 0 && (
          <div className="playground-history">
            {history.map((item, idx) => (
              <img 
                key={item.id}
                src={item.url} 
                className={`history-thumbnail ${activeIndex === idx ? 'history-thumbnail--active' : ''}`}
                onClick={() => setActiveIndex(idx)}
                alt="History thumbnail"
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
