import React, { useState, useEffect, useCallback } from 'react';
import {
  Download,
  CheckCircle2,
  HardDrive,
  Trash2,
  XCircle,
  FolderOpen,
  Plus,
  Link,
  AlertTriangle,
  Loader,
  X,
} from 'lucide-react';
import { Header } from '../components/layout/Header';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { ProgressBar } from '../components/ui/ProgressBar';
import { Modal } from '../components/ui/Modal';
import { useApp } from '../context/AppContext';
import { useWebSocket } from '../hooks/useWebSocket';
import {
  fetchBaseModels,
  downloadBaseModel,
  cancelBaseModelDownload,
  deleteBaseModel,
  addCustomModel,
  setModelsDirectory,
} from '../services/api';
import type { BaseModel } from '../types';
import './ModelsPage.css';

export function ModelsPage() {
  const { state, dispatch } = useApp();
  const [loading, setLoading] = useState(true);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [showCustomForm, setShowCustomForm] = useState(false);
  const [customUrl, setCustomUrl] = useState('');
  const [customName, setCustomName] = useState('');
  const [customArch, setCustomArch] = useState('sd15');
  const [addingCustom, setAddingCustom] = useState(false);
  const [showDirEdit, setShowDirEdit] = useState(false);
  const [dirInput, setDirInput] = useState('');
  const [savingDir, setSavingDir] = useState(false);

  // WebSocket for download progress
  useWebSocket({
    url: 'ws://localhost:8000/ws/training',
    onMessage: (msg) => {
      switch (msg.type) {
        case 'download_progress':
          dispatch({
            type: 'UPDATE_BASE_MODEL',
            payload: {
              id: msg.data.modelId,
              status: 'downloading',
              downloadProgress: msg.data.progress,
              downloadSpeed: msg.data.speed,
              downloadedBytes: msg.data.downloadedBytes,
              fileSize: msg.data.totalBytes || state.baseModels.find(m => m.id === msg.data.modelId)?.fileSize || 0,
            },
          });
          break;
        case 'download_complete':
          dispatch({
            type: 'UPDATE_BASE_MODEL',
            payload: {
              id: msg.data.modelId,
              status: 'downloaded',
              localPath: msg.data.localPath,
              downloadProgress: 100,
              downloadSpeed: undefined,
            },
          });
          break;
        case 'download_error':
          dispatch({
            type: 'UPDATE_BASE_MODEL',
            payload: {
              id: msg.data.modelId,
              status: 'error',
              error: msg.data.message,
              downloadSpeed: undefined,
            },
          });
          break;
      }
    },
  });

  // Load models on mount
  const loadModels = useCallback(async () => {
    try {
      const res = await fetchBaseModels();
      dispatch({ type: 'SET_BASE_MODELS', payload: res.models as BaseModel[] });
      dispatch({ type: 'SET_MODELS_DIRECTORY', payload: res.modelsDirectory });
      setDirInput(res.modelsDirectory);
    } catch (err) {
      console.error('Failed to load base models:', err);
    } finally {
      setLoading(false);
    }
  }, [dispatch]);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  const handleDownload = async (modelId: string) => {
    try {
      dispatch({
        type: 'UPDATE_BASE_MODEL',
        payload: { id: modelId, status: 'downloading', downloadProgress: 0, error: undefined },
      });
      await downloadBaseModel(modelId);
    } catch (err) {
      console.error('Download failed:', err);
    }
  };

  const handleCancelDownload = async (modelId: string) => {
    try {
      await cancelBaseModelDownload(modelId);
      dispatch({
        type: 'UPDATE_BASE_MODEL',
        payload: { id: modelId, status: 'not_downloaded', downloadProgress: undefined, downloadSpeed: undefined },
      });
    } catch (err) {
      console.error('Cancel failed:', err);
    }
  };

  const handleDelete = async (modelId: string) => {
    try {
      await deleteBaseModel(modelId);
      // If custom model, remove from state entirely
      const model = state.baseModels.find(m => m.id === modelId);
      if (model?.isCustom) {
        dispatch({ type: 'REMOVE_BASE_MODEL', payload: modelId });
      } else {
        dispatch({
          type: 'UPDATE_BASE_MODEL',
          payload: { id: modelId, status: 'not_downloaded', localPath: undefined },
        });
      }
      setDeleteConfirm(null);
    } catch (err) {
      console.error('Delete failed:', err);
    }
  };

  const handleAddCustom = async () => {
    if (!customUrl.trim()) return;
    setAddingCustom(true);
    try {
      const result = await addCustomModel(customUrl, customName || undefined, customArch);
      dispatch({ type: 'ADD_BASE_MODEL', payload: result as BaseModel });
      setCustomUrl('');
      setCustomName('');
      setCustomArch('sd15');
      setShowCustomForm(false);
    } catch (err) {
      console.error('Add custom model failed:', err);
    } finally {
      setAddingCustom(false);
    }
  };

  const handleSaveDir = async () => {
    if (!dirInput.trim()) return;
    setSavingDir(true);
    try {
      const res = await setModelsDirectory(dirInput);
      dispatch({ type: 'SET_MODELS_DIRECTORY', payload: res.modelsDirectory });
      setShowDirEdit(false);
      // Reload models to check what's available in new dir
      await loadModels();
    } catch (err) {
      console.error('Set directory failed:', err);
    } finally {
      setSavingDir(false);
    }
  };

  const formatSize = (bytes: number) => {
    if (bytes <= 0) return 'Unknown';
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  };

  const catalogModels = state.baseModels.filter(m => !m.isCustom);
  const customModels = state.baseModels.filter(m => m.isCustom);
  const downloadedCount = state.baseModels.filter(m => m.status === 'downloaded').length;
  const totalDownloadedSize = state.baseModels
    .filter(m => m.status === 'downloaded')
    .reduce((acc, m) => acc + (m.fileSize || 0), 0);

  const renderModelCard = (model: BaseModel) => (
    <div
      key={model.id}
      className={`model-card ${model.status === 'downloaded' ? 'model-card--downloaded' : ''} ${model.status === 'downloading' ? 'model-card--downloading' : ''}`}
    >
      {model.isCustom && <div className="model-card__custom-tag">Custom</div>}

      <div className="model-card__header">
        <div className="model-card__title-group">
          <span className="model-card__name">{model.name}</span>
          <span className={`model-card__arch-badge model-card__arch-badge--${model.architecture}`}>
            {model.architecture === 'sd15' && 'SD 1.5'}
            {model.architecture === 'sdxl' && 'SDXL'}
            {model.architecture === 'flux' && 'FLUX'}
          </span>
        </div>
        <div className={`model-card__status-icon model-card__status-icon--${model.status}`}>
          {model.status === 'downloaded' && <CheckCircle2 size={18} />}
          {model.status === 'not_downloaded' && <Download size={18} />}
          {model.status === 'downloading' && <Loader size={18} className="animate-spin" />}
          {model.status === 'error' && <AlertTriangle size={18} />}
        </div>
      </div>

      <p className="model-card__description">{model.description}</p>

      <div className="model-card__meta">
        <span className="model-card__meta-item">
          <HardDrive size={12} />
          {formatSize(model.fileSize)}
        </span>
        <span className="model-card__meta-item">{model.filename}</span>
      </div>

      {model.error && (
        <div className="model-card__error">{model.error}</div>
      )}

      {model.status === 'downloading' && (
        <div className="model-card__progress">
          <div className="model-card__progress-info">
            <span>{formatSize(model.downloadedBytes || 0)} / {formatSize(model.fileSize)}</span>
            <span className="model-card__speed">{model.downloadSpeed || '...'}</span>
          </div>
          <ProgressBar
            value={model.downloadProgress || 0}
            max={100}
            size="sm"
            variant="accent"
            animated
          />
        </div>
      )}

      <div className="model-card__actions">
        {model.status === 'not_downloaded' && (
          <Button
            variant="primary"
            size="sm"
            icon={<Download size={14} />}
            onClick={() => handleDownload(model.id)}
          >
            Download
          </Button>
        )}
        {model.status === 'error' && (
          <Button
            variant="primary"
            size="sm"
            icon={<Download size={14} />}
            onClick={() => handleDownload(model.id)}
          >
            Retry
          </Button>
        )}
        {model.status === 'downloading' && (
          <Button
            variant="danger"
            size="sm"
            icon={<X size={14} />}
            onClick={() => handleCancelDownload(model.id)}
          >
            Cancel
          </Button>
        )}
        {model.status === 'downloaded' && (
          <>
            {deleteConfirm === model.id ? (
              <div className="model-card__delete-confirm">
                <span>Delete from disk?</span>
                <Button variant="danger" size="sm" onClick={() => handleDelete(model.id)}>
                  Yes
                </Button>
                <Button variant="ghost" size="sm" onClick={() => setDeleteConfirm(null)}>
                  No
                </Button>
              </div>
            ) : (
              <Button
                variant="ghost"
                size="sm"
                icon={<Trash2 size={14} />}
                onClick={() => setDeleteConfirm(model.id)}
              >
                Delete
              </Button>
            )}
          </>
        )}
      </div>
    </div>
  );

  if (loading) {
    return (
      <div className="models-page animate-fade-in-up">
        <Header title="Models Hub" subtitle="Loading models..." />
        <div style={{ display: 'flex', justifyContent: 'center', padding: 'var(--space-16)' }}>
          <Loader size={32} className="animate-spin" style={{ color: 'var(--color-text-disabled)' }} />
        </div>
      </div>
    );
  }

  return (
    <div className="models-page animate-fade-in-up">
      <Header
        title="Models Hub"
        subtitle="Download and manage base models for LoRA training"
        actions={
          <Button
            variant="outline"
            size="sm"
            icon={<Plus size={14} />}
            onClick={() => setShowCustomForm(true)}
          >
            Add Custom Model
          </Button>
        }
      />

      {/* Storage Info */}
      <Card className="models-storage-bar" padding="sm">
        <div className="models-storage-info">
          <HardDrive size={16} />
          <span>{downloadedCount} model{downloadedCount !== 1 ? 's' : ''} downloaded</span>
          <Badge variant="default">{formatSize(totalDownloadedSize)}</Badge>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
          {showDirEdit ? (
            <div className="dir-settings">
              <input
                className="dir-settings__input"
                value={dirInput}
                onChange={(e) => setDirInput(e.target.value)}
                placeholder="Path to models directory..."
                onKeyDown={(e) => e.key === 'Enter' && handleSaveDir()}
              />
              <Button variant="primary" size="sm" loading={savingDir} onClick={handleSaveDir}>
                Save
              </Button>
              <Button variant="ghost" size="sm" onClick={() => { setShowDirEdit(false); setDirInput(state.modelsDirectory); }}>
                Cancel
              </Button>
            </div>
          ) : (
            <>
              <span className="models-storage-path" title={state.modelsDirectory}>
                {state.modelsDirectory}
              </span>
              <Button variant="ghost" size="sm" icon={<FolderOpen size={14} />} onClick={() => setShowDirEdit(true)}>
                Change
              </Button>
            </>
          )}
        </div>
      </Card>

      {/* Catalog Models */}
      <div>
        <div className="models-section-header">
          <div>
            <h2 className="models-section-title">Base Models</h2>
            <p className="models-section-subtitle">Popular foundation models for LoRA training</p>
          </div>
        </div>
        <div className="models-grid stagger-children">
          {catalogModels.map(renderModelCard)}
        </div>
      </div>

      {/* Custom Models */}
      {customModels.length > 0 && (
        <div>
          <div className="models-section-header">
            <div>
              <h2 className="models-section-title">Custom Models</h2>
              <p className="models-section-subtitle">Models added by URL</p>
            </div>
          </div>
          <div className="models-grid stagger-children">
            {customModels.map(renderModelCard)}
          </div>
        </div>
      )}

      {/* Add Custom Model Modal */}
      <Modal isOpen={showCustomForm} onClose={() => setShowCustomForm(false)} title="Add Custom Model" size="md">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
          <div className="custom-model-form__field">
            <label className="custom-model-form__label">Download URL</label>
            <input
              className="custom-model-form__input"
              value={customUrl}
              onChange={(e) => setCustomUrl(e.target.value)}
              placeholder="https://huggingface.co/... or direct .safetensors link"
            />
          </div>
          <div style={{ display: 'flex', gap: 'var(--space-3)' }}>
            <div className="custom-model-form__field">
              <label className="custom-model-form__label">Name (optional)</label>
              <input
                className="custom-model-form__input"
                value={customName}
                onChange={(e) => setCustomName(e.target.value)}
                placeholder="My Custom Model"
              />
            </div>
            <div className="custom-model-form__field custom-model-form__field--small">
              <label className="custom-model-form__label">Architecture</label>
              <select
                className="custom-model-form__select"
                value={customArch}
                onChange={(e) => setCustomArch(e.target.value)}
              >
                <option value="sd15">SD 1.5</option>
                <option value="sdxl">SDXL</option>
                <option value="flux">Flux</option>
              </select>
            </div>
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 'var(--space-3)', marginTop: 'var(--space-2)' }}>
            <Button variant="secondary" onClick={() => setShowCustomForm(false)}>
              Cancel
            </Button>
            <Button
              variant="primary"
              icon={<Plus size={14} />}
              loading={addingCustom}
              onClick={handleAddCustom}
              disabled={!customUrl.trim()}
            >
              Add Model
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}
