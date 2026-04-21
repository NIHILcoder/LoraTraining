import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Upload,
  ImagePlus,
  Trash2,
  Eye,
  Grid3X3,
  X,
  FileImage,
  Images,
  ChevronDown,
} from 'lucide-react';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import { Modal } from '../ui/Modal';
import { ProgressBar } from '../ui/ProgressBar';
import { useApp } from '../../context/AppContext';
import { uploadImage, createDataset, autoCaptionImage } from '../../services/api';
import type { DatasetImage } from '../../types';

export function DatasetSection() {
  const { state, dispatch } = useApp();
  const [isOpen, setIsOpen] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadTotal, setUploadTotal] = useState(0);
  const [previewImage, setPreviewImage] = useState<DatasetImage | null>(null);
  const [selectedImages, setSelectedImages] = useState<Set<string>>(new Set());
  const [captioningMap, setCaptioningMap] = useState<Record<string, boolean>>({});
  const [tagInput, setTagInput] = useState('');

  const images = state.currentDataset?.images ?? [];

  const handleAddTag = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && tagInput.trim()) {
      if (previewImage && state.currentDataset) {
        const currentTags = previewImage.captions || [];
        if (!currentTags.includes(tagInput.trim())) {
          const newTags = [...currentTags, tagInput.trim()];
          dispatch({
            type: 'UPDATE_DATASET_IMAGE_CAPTIONS',
            payload: { datasetId: state.currentDataset.id, imageId: previewImage.id, captions: newTags }
          });
          setPreviewImage({ ...previewImage, captions: newTags });
        }
        setTagInput('');
      }
    }
  };

  const handleRemoveTag = (tagToRemove: string) => {
    if (previewImage && state.currentDataset) {
      const newTags = (previewImage.captions || []).filter(t => t !== tagToRemove);
      dispatch({
        type: 'UPDATE_DATASET_IMAGE_CAPTIONS',
        payload: { datasetId: state.currentDataset.id, imageId: previewImage.id, captions: newTags }
      });
      setPreviewImage({ ...previewImage, captions: newTags });
    }
  };

  const handleAutoCaptionAll = async () => {
    if (!state.currentDataset) return;
    const ds = state.currentDataset;
    const toCaption = ds.images.filter(img => !img.captions || img.captions.length === 0);
    if (toCaption.length === 0) return;

    const newMap: Record<string, boolean> = {};
    toCaption.forEach(img => newMap[img.id] = true);
    setCaptioningMap(prev => ({ ...prev, ...newMap }));

    for (const img of toCaption) {
      try {
        const tags = await autoCaptionImage(img.id, img.filePath || img.url);
        dispatch({
          type: 'UPDATE_DATASET_IMAGE_CAPTIONS',
          payload: { datasetId: ds.id, imageId: img.id, captions: tags }
        });
      } catch (err) {
        console.error('Captioning failed', err);
      } finally {
        setCaptioningMap(prev => ({ ...prev, [img.id]: false }));
      }
    }
  };

  const ensureDataset = useCallback(async () => {
    if (!state.currentDataset) {
      const ds = await createDataset('Default Dataset');
      dispatch({ type: 'SET_DATASETS', payload: [ds] });
      dispatch({ type: 'SET_CURRENT_DATASET', payload: ds });
      return ds;
    }
    return state.currentDataset;
  }, [state.currentDataset, dispatch]);

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;
      setUploading(true);
      setUploadProgress(0);
      setUploadTotal(acceptedFiles.length);
      const dataset = await ensureDataset();
      for (let i = 0; i < acceptedFiles.length; i++) {
        try {
          const image = await uploadImage(dataset.id, acceptedFiles[i]);
          dispatch({ type: 'ADD_DATASET_IMAGE', payload: { datasetId: dataset.id, image } });
          setUploadProgress(i + 1);
        } catch (err) {
          console.error('Upload failed:', err);
        }
      }
      setUploading(false);
    },
    [ensureDataset, dispatch]
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: { 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg'], 'image/webp': ['.webp'] },
    disabled: uploading,
  });

  const handleDelete = (imageId: string) => {
    if (state.currentDataset) {
      dispatch({ type: 'REMOVE_DATASET_IMAGE', payload: { datasetId: state.currentDataset.id, imageId } });
    }
  };

  const handleDeleteSelected = () => {
    selectedImages.forEach((id) => handleDelete(id));
    setSelectedImages(new Set());
  };

  const toggleSelect = (id: string) => {
    setSelectedImages((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <>
      <div className="ws-section">
        <button className="ws-section__toggle" onClick={() => setIsOpen(!isOpen)}>
          <span className="ws-section__toggle-icon"><Images size={16} /></span>
          <span className="ws-section__toggle-title">
            Dataset
            {images.length > 0 && (
              <span className="ws-section__toggle-count">{images.length} images</span>
            )}
          </span>
          <span className={`ws-section__toggle-chevron ${isOpen ? 'ws-section__toggle-chevron--open' : ''}`}>
            <ChevronDown size={14} />
          </span>
        </button>

        {isOpen && (
          <div className="ws-section__body">
            {/* Dropzone */}
            <div
              {...getRootProps()}
              className={`dropzone ws-dropzone ${isDragActive ? 'dropzone--active' : ''} ${isDragReject ? 'dropzone--reject' : ''} ${uploading ? 'dropzone--disabled' : ''}`}
            >
              <input {...getInputProps()} />
              <div className="dropzone__content">
                <div className={`dropzone__icon ${isDragActive ? 'dropzone__icon--active' : ''}`}>
                  {isDragActive ? <ImagePlus size={24} /> : <Upload size={24} />}
                </div>
                <div className="dropzone__text">
                  <span className="dropzone__title">
                    {isDragActive ? 'Drop images here' : 'Drag & drop images'}
                  </span>
                  <span className="dropzone__subtitle">PNG, JPG, WEBP</span>
                </div>
              </div>
            </div>

            {/* Upload Progress */}
            {uploading && (
              <div style={{ marginTop: 'var(--space-3)' }}>
                <ProgressBar
                  value={uploadProgress}
                  max={uploadTotal}
                  showLabel
                  label={`Uploading ${uploadProgress}/${uploadTotal}...`}
                  variant="accent"
                />
              </div>
            )}

            {/* Actions bar */}
            {images.length > 0 && (
              <div className="ws-dataset-actions">
                <div className="ws-dataset-actions-left">
                  {selectedImages.size > 0 && (
                    <Button variant="danger" size="sm" icon={<Trash2 size={12} />} onClick={handleDeleteSelected}>
                      Delete ({selectedImages.size})
                    </Button>
                  )}
                </div>
                <Button variant="outline" size="sm" onClick={handleAutoCaptionAll}>
                  Auto Caption
                </Button>
              </div>
            )}

            {/* Image Grid */}
            {images.length > 0 && (
              <div className="ws-image-grid">
                {images.map((img) => (
                  <div
                    key={img.id}
                    className={`image-card ${selectedImages.has(img.id) ? 'image-card--selected' : ''}`}
                    onClick={() => toggleSelect(img.id)}
                  >
                    <div className="image-card__preview">
                      <img src={img.url} alt={img.filename} loading="lazy" />
                      <div className="image-card__overlay">
                        <button
                          className="image-card__action"
                          onClick={(e) => { e.stopPropagation(); setPreviewImage(img); }}
                          title="Preview"
                        >
                          <Eye size={14} />
                        </button>
                        <button
                          className="image-card__action image-card__action--danger"
                          onClick={(e) => { e.stopPropagation(); handleDelete(img.id); }}
                          title="Delete"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                      {selectedImages.has(img.id) && <div className="image-card__check">✓</div>}
                    </div>
                    <div className="image-card__info">
                      <span className="image-card__name truncate">{img.filename}</span>
                      <span className="image-card__size">{formatSize(img.size)}</span>
                    </div>
                    {(captioningMap[img.id] || (img.captions && img.captions.length > 0)) && (
                      <div className="image-card__captions" style={{ padding: '0 4px 4px', display: 'flex', flexWrap: 'wrap', gap: '2px' }}>
                        {captioningMap[img.id] ? (
                          <span style={{ fontSize: '9px', color: 'var(--color-text-secondary)' }}>Captioning...</span>
                        ) : (
                          img.captions!.slice(0, 2).map(tag => (
                            <span key={tag} style={{ background: 'var(--color-bg-hover)', border: '1px solid var(--color-surface-border)', padding: '1px 3px', borderRadius: '3px', fontSize: '9px', color: 'var(--color-text-secondary)' }}>
                              {tag}
                            </span>
                          ))
                        )}
                        {img.captions && img.captions.length > 2 && (
                          <span style={{ fontSize: '9px', color: 'var(--color-text-disabled)' }}>+{img.captions.length - 2}</span>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* Empty */}
            {images.length === 0 && !uploading && (
              <div className="ws-dataset-empty">
                <Grid3X3 size={32} className="ws-dataset-empty-icon" />
                <p className="ws-dataset-empty-text">No images yet. Upload your training dataset.</p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Preview Modal */}
      <Modal isOpen={!!previewImage} onClose={() => setPreviewImage(null)} title={previewImage?.filename} size="lg">
        {previewImage && (
          <div className="image-preview">
            <img src={previewImage.url} alt={previewImage.filename} className="image-preview__img" />
            <div className="image-preview__meta">
              <span>Size: {formatSize(previewImage.size)}</span>
              <span>Dimensions: {previewImage.width}×{previewImage.height}</span>
            </div>
            <div style={{ marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <div style={{ fontSize: '12px', color: 'var(--color-text-secondary)', fontWeight: 600 }}>Captions / Tags:</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {(previewImage.captions || []).map(tag => (
                  <span key={tag} style={{ background: 'var(--color-bg-elevated)', border: '1px solid var(--color-surface-border)', padding: '4px 8px', borderRadius: '4px', fontSize: '12px', color: 'var(--color-text-primary)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                    {tag}
                    <button onClick={() => handleRemoveTag(tag)} style={{ background: 'none', border: 'none', color: 'var(--color-text-tertiary)', cursor: 'pointer', padding: 0, display: 'flex' }}>
                      <X size={12} />
                    </button>
                  </span>
                ))}
              </div>
              <input
                type="text"
                placeholder="Add new tag and press Enter..."
                value={tagInput}
                onChange={(e) => setTagInput(e.target.value)}
                onKeyDown={handleAddTag}
                style={{
                  marginTop: '4px', padding: '8px 12px', background: 'var(--color-bg-primary)',
                  border: '1px solid var(--color-surface-border)', borderRadius: 'var(--radius-md)',
                  color: 'var(--color-text-primary)', outline: 'none', fontSize: '12px', width: '100%'
                }}
              />
            </div>
          </div>
        )}
      </Modal>
    </>
  );
}
