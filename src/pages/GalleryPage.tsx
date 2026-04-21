import React, { useEffect, useState, useMemo } from 'react';
import {
  Trash2,
  Search,
  FolderOpen,
  Zap,
  LayoutGrid,
  ArrowUpDown,
  Activity,
  Layers,
  Clock,
  HardDrive,
  AlertTriangle,
  Sparkles,
  FileBox,
  ChevronDown,
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { Header } from '../components/layout/Header';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { Modal } from '../components/ui/Modal';
import { fetchModels, deleteModel, openModelFolder } from '../services/api';
import './GalleryPage.css';

interface TrainedModel {
  id: string;
  name: string;
  filename: string;
  fileSize: number;
  path: string;
  directory: string;
  createdAt: string;
  rank: number;
  alpha: number;
  targetModules: string[];
  finalLoss: number;
  avgLoss: number;
  totalSteps: number;
  stoppedEarly: boolean;
  architecture: string;
  baseModelName: string;
}

type SortKey = 'date' | 'name' | 'loss' | 'size';

export function GalleryPage() {
  const navigate = useNavigate();
  const [models, setModels] = useState<TrainedModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<SortKey>('date');
  const [sortDropdownOpen, setSortDropdownOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<TrainedModel | null>(null);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      setLoading(true);
      const data = await fetchModels();
      setModels(data);
    } catch (err) {
      console.error('Failed to load models:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    try {
      await deleteModel(deleteTarget.id);
      setModels(prev => prev.filter(m => m.id !== deleteTarget.id));
    } catch (err) {
      console.error('Failed to delete model:', err);
    } finally {
      setDeleteTarget(null);
    }
  };

  const handleOpenFolder = async (id: string) => {
    try {
      await openModelFolder(id);
    } catch (err) {
      console.error('Failed to open folder:', err);
    }
  };

  const handleTestInPlayground = (model: TrainedModel) => {
    // Navigate to playground with model info in state
    navigate('/playground', { state: { loraModel: model } });
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  };

  const formatDate = (iso: string) => {
    const d = new Date(iso);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };

  const formatTime = (iso: string) => {
    const d = new Date(iso);
    return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
  };

  const sortOptions: { key: SortKey; label: string }[] = [
    { key: 'date', label: 'Newest first' },
    { key: 'name', label: 'Name A–Z' },
    { key: 'loss', label: 'Best loss' },
    { key: 'size', label: 'Largest' },
  ];

  const filteredAndSorted = useMemo(() => {
    let result = models.filter(m =>
      m.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      m.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
      m.architecture.toLowerCase().includes(searchQuery.toLowerCase())
    );

    switch (sortBy) {
      case 'date':
        result.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
        break;
      case 'name':
        result.sort((a, b) => a.name.localeCompare(b.name));
        break;
      case 'loss':
        result.sort((a, b) => (a.finalLoss || Infinity) - (b.finalLoss || Infinity));
        break;
      case 'size':
        result.sort((a, b) => b.fileSize - a.fileSize);
        break;
    }

    return result;
  }, [models, searchQuery, sortBy]);

  return (
    <div className="gallery-page animate-fade-in-up">
      <Header
        title="Gallery"
        subtitle={`${models.length} trained LoRA model${models.length !== 1 ? 's' : ''}`}
        actions={
          <div className="gallery-header-actions">
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
            <div className="gallery-sort" onClick={() => setSortDropdownOpen(!sortDropdownOpen)}>
              <ArrowUpDown size={14} />
              <span>{sortOptions.find(s => s.key === sortBy)?.label}</span>
              <ChevronDown size={14} className={sortDropdownOpen ? 'gallery-sort__chevron--open' : ''} />
              {sortDropdownOpen && (
                <div className="gallery-sort__dropdown" onClick={(e) => e.stopPropagation()}>
                  {sortOptions.map(opt => (
                    <button
                      key={opt.key}
                      className={`gallery-sort__option ${sortBy === opt.key ? 'gallery-sort__option--active' : ''}`}
                      onClick={() => { setSortBy(opt.key); setSortDropdownOpen(false); }}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
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
          <p>Scanning trained models...</p>
        </div>
      ) : filteredAndSorted.length === 0 ? (
        <div className="gallery-empty animate-fade-in">
          <div className="gallery-empty__icon-wrap">
            <FileBox size={48} />
          </div>
          <h3 className="gallery-empty__title">
            {models.length === 0 ? 'No trained models yet' : 'No matches found'}
          </h3>
          <p className="gallery-empty__text">
            {models.length === 0
              ? 'Train your first LoRA model from the Training workspace. Completed models will appear here.'
              : 'Try adjusting your search query.'}
          </p>
          {models.length === 0 && (
            <Button variant="primary" icon={<Sparkles size={16} />} onClick={() => navigate('/')}>
              Start Training
            </Button>
          )}
        </div>
      ) : (
        <div className="gallery-grid stagger-children">
          {filteredAndSorted.map((model) => (
            <Card key={model.id} padding="none" className="gallery-model-card">
              {/* Top accent bar */}
              <div className="gallery-model-card__accent" />

              {/* Header */}
              <div className="gallery-model-card__header">
                <div className="gallery-model-card__title-row">
                  <Layers size={18} className="gallery-model-card__icon" />
                  <div className="gallery-model-card__title-info">
                    <h3 className="gallery-model-card__name truncate">{model.name}</h3>
                    <span className="gallery-model-card__filename">{model.filename}</span>
                  </div>
                </div>
                <Badge variant={model.stoppedEarly ? 'warning' : 'success'} size="sm">
                  {model.stoppedEarly ? 'Stopped' : 'Complete'}
                </Badge>
              </div>

              {/* Stats grid */}
              <div className="gallery-model-card__stats">
                <div className="gallery-stat">
                  <Activity size={14} />
                  <div className="gallery-stat__info">
                    <span className="gallery-stat__label">Final Loss</span>
                    <span className="gallery-stat__value">
                      {model.finalLoss > 0 ? model.finalLoss.toFixed(4) : '—'}
                    </span>
                  </div>
                </div>
                <div className="gallery-stat">
                  <Zap size={14} />
                  <div className="gallery-stat__info">
                    <span className="gallery-stat__label">Steps</span>
                    <span className="gallery-stat__value">
                      {model.totalSteps > 0 ? model.totalSteps.toLocaleString() : '—'}
                    </span>
                  </div>
                </div>
                <div className="gallery-stat">
                  <LayoutGrid size={14} />
                  <div className="gallery-stat__info">
                    <span className="gallery-stat__label">Rank</span>
                    <span className="gallery-stat__value">{model.rank || '—'}</span>
                  </div>
                </div>
                <div className="gallery-stat">
                  <HardDrive size={14} />
                  <div className="gallery-stat__info">
                    <span className="gallery-stat__label">Size</span>
                    <span className="gallery-stat__value">{formatSize(model.fileSize)}</span>
                  </div>
                </div>
              </div>

              {/* Meta info */}
              <div className="gallery-model-card__meta">
                {model.architecture && (
                  <span className="gallery-meta-tag">{model.architecture.toUpperCase()}</span>
                )}
                {model.alpha > 0 && (
                  <span className="gallery-meta-tag">α {model.alpha}</span>
                )}
                <span className="gallery-meta-date">
                  <Clock size={12} />
                  {formatDate(model.createdAt)} · {formatTime(model.createdAt)}
                </span>
              </div>

              {/* Actions */}
              <div className="gallery-model-card__actions">
                <Button variant="secondary" size="sm" icon={<FolderOpen size={14} />} onClick={() => handleOpenFolder(model.id)}>
                  Open Folder
                </Button>
                <Button variant="ghost" size="sm" icon={<Zap size={14} />} onClick={() => handleTestInPlayground(model)}>
                  Test
                </Button>
                <Button variant="ghost" size="sm" icon={<Trash2 size={14} />} onClick={() => setDeleteTarget(model)} className="gallery-delete-btn">
                  Delete
                </Button>
              </div>
            </Card>
          ))}
        </div>
      )}

      <Modal isOpen={!!deleteTarget} onClose={() => setDeleteTarget(null)} title="Delete Model">
        <p style={{ marginBottom: '0.5rem', color: 'var(--color-text-secondary)' }}>
          Are you sure you want to permanently delete this model?
        </p>
        {deleteTarget && (
          <div style={{ padding: '12px', background: 'var(--color-bg-tertiary)', borderRadius: 'var(--radius-md)', marginBottom: '1.5rem', fontSize: 'var(--text-sm)' }}>
            <strong style={{ color: 'var(--color-text-primary)' }}>{deleteTarget.name}</strong>
            <br />
            <span style={{ color: 'var(--color-text-tertiary)' }}>{deleteTarget.filename} · {formatSize(deleteTarget.fileSize)}</span>
          </div>
        )}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 12px', background: 'var(--color-error-muted)', borderRadius: 'var(--radius-md)', marginBottom: '1.5rem', fontSize: 'var(--text-xs)', color: 'var(--color-error)' }}>
          <AlertTriangle size={14} />
          <span>This will delete the .safetensors file and all training metadata.</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
          <Button variant="secondary" onClick={() => setDeleteTarget(null)}>Cancel</Button>
          <Button variant="danger" onClick={handleDelete}>Delete Permanently</Button>
        </div>
      </Modal>
    </div>
  );
}
