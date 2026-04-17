import React, { useEffect, useState } from 'react';
import { Download, Trash2, Search, Filter } from 'lucide-react';
import { Header } from '../components/layout/Header';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { useApp } from '../context/AppContext';
import { fetchModels, deleteModel } from '../services/api';
import type { LoraModel } from '../types';
import './GalleryPage.css';

export function GalleryPage() {
  const { state, dispatch } = useApp();
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    const loadModels = async () => {
      try {
        setLoading(true);
        const models = await fetchModels();
        dispatch({ type: 'SET_MODELS', payload: models });
      } catch (err) {
        console.error('Failed to load models:', err);
      } finally {
        setLoading(false);
      }
    };
    
    if (state.models.length === 0) {
      loadModels();
    } else {
      setLoading(false);
    }
  }, [dispatch, state.models.length]);

  const handleDelete = async (id: string) => {
    if (confirm('Are you sure you want to delete this model?')) {
      try {
        await deleteModel(id);
        const newModels = state.models.filter(m => m.id !== id);
        dispatch({ type: 'SET_MODELS', payload: newModels });
      } catch (err) {
        console.error('Failed to delete model:', err);
      }
    }
  };

  const filteredModels = state.models.filter(m => 
    m.name.toLowerCase().includes(searchQuery.toLowerCase()) || 
    m.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  const formatSize = (bytes: number) => {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="gallery-page animate-fade-in-up">
      <Header
        title="Gallery"
        subtitle="View and manage your trained LoRA models"
        actions={
          <div className="gallery-search">
            <Search size={16} className="gallery-search__icon" />
            <input 
              type="text" 
              placeholder="Search models..." 
              className="gallery-search__input"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
        }
      />

      {loading ? (
        <div className="gallery-loading">
          <div className="animate-spin text-accent">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeDasharray="42" strokeDashoffset="12" strokeLinecap="round" />
            </svg>
          </div>
          <p>Loading models...</p>
        </div>
      ) : filteredModels.length === 0 ? (
        <div className="gallery-empty animate-fade-in">
           <Filter size={48} className="gallery-empty__icon" />
           <p className="gallery-empty__text">No models found. Try adjusting your search or train a new model.</p>
        </div>
      ) : (
        <div className="gallery-grid stagger-children">
          {filteredModels.map((model) => (
            <Card key={model.id} padding="none" className="model-card">
              <div className="model-card__preview-area">
                 {model.sampleImages && model.sampleImages.length > 0 ? (
                   <img 
                     src={model.sampleImages[0].url} 
                     alt={`Sample from ${model.name}`} 
                     className="model-card__image"
                   />
                 ) : (
                   <div className="model-card__placeholder">No Sample</div>
                 )}
                 <div className="model-card__badges">
                    <Badge variant="accent" size="sm">Rank {model.rank}</Badge>
                 </div>
              </div>
              <div className="model-card__content">
                <div className="model-card__header">
                  <h3 className="model-card__title truncate">{model.name}</h3>
                  <span className="model-card__size">{formatSize(model.fileSize)}</span>
                </div>
                <p className="model-card__desc truncate">{model.description}</p>
                <div className="model-card__tags scrollbar-hide">
                  {model.tags.map(tag => (
                    <span key={tag} className="model-tag">#{tag}</span>
                  ))}
                </div>
                <div className="model-card__actions">
                  <Button variant="secondary" size="sm" icon={<Download size={14} />}>
                    Download
                  </Button>
                  <Button variant="ghost" size="sm" icon={<Trash2 size={14} />} onClick={() => handleDelete(model.id)}>
                    Delete
                  </Button>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
