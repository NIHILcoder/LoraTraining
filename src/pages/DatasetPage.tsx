import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Upload,
  ImagePlus,
  Trash2,
  Eye,
  FolderPlus,
  Grid3X3,
  X,
  FileImage,
} from 'lucide-react';
import { Header } from '../components/layout/Header';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { Modal } from '../components/ui/Modal';
import { ProgressBar } from '../components/ui/ProgressBar';
import { useApp } from '../context/AppContext';
import { uploadImage, createDataset, autoCaptionImage, generateId } from '../services/api';
import type { DatasetImage } from '../types';
import './DatasetPage.css';

export function DatasetPage() {
  const { state, dispatch } = useApp();
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadTotal, setUploadTotal] = useState(0);
  const [previewImage, setPreviewImage] = useState<DatasetImage | null>(null);
  const [selectedImages, setSelectedImages] = useState<Set<string>>(new Set());
  const [captioningMap, setCaptioningMap] = useState<Record<string, boolean>>({});
  const [tagInput, setTagInput] = useState('');

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

    // Run sequentially to simulate real ML load and save RAM
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

  // Initialize default dataset if none exists
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
          dispatch({
            type: 'ADD_DATASET_IMAGE',
            payload: { datasetId: dataset.id, image },
          });
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
    accept: {
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/webp': ['.webp'],
    },
    disabled: uploading,
  });

  const images = state.currentDataset?.images ?? [];

  const handleDelete = (imageId: string) => {
    if (state.currentDataset) {
      dispatch({
        type: 'REMOVE_DATASET_IMAGE',
        payload: { datasetId: state.currentDataset.id, imageId },
      });
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
    <div className="dataset-page animate-fade-in-up">
      <Header
        title="Dataset"
        subtitle="Upload and manage your training images"
        actions={
          images.length > 0 ? (
            <>
              <Badge variant="accent">
                <FileImage size={12} />
                {images.length} images
              </Badge>
              {selectedImages.size > 0 && (
                <Button
                  variant="danger"
                  size="sm"
                  icon={<Trash2 size={14} />}
                  onClick={handleDeleteSelected}
                >
                  Delete ({selectedImages.size})
                </Button>
              )}
              <Button
                variant="outline"
                size="sm"
                onClick={handleAutoCaptionAll}
              >
                Auto Caption All
              </Button>
            </>
          ) : undefined
        }
      />

      {/* Dropzone */}
      <Card padding="none" className="dataset-page__dropzone-card">
        <div
          {...getRootProps()}
          className={`dropzone ${isDragActive ? 'dropzone--active' : ''} ${isDragReject ? 'dropzone--reject' : ''} ${uploading ? 'dropzone--disabled' : ''}`}
          id="dataset-dropzone"
        >
          <input {...getInputProps()} />
          <div className="dropzone__content">
            <div className={`dropzone__icon ${isDragActive ? 'dropzone__icon--active' : ''}`}>
              {isDragActive ? <ImagePlus size={40} /> : <Upload size={40} />}
            </div>
            <div className="dropzone__text">
              <span className="dropzone__title">
                {isDragActive
                  ? 'Drop images here'
                  : 'Drag & drop images here'}
              </span>
              <span className="dropzone__subtitle">
                or click to browse • PNG, JPG, WEBP supported
              </span>
            </div>
          </div>
        </div>
      </Card>

      {/* Upload Progress */}
      {uploading && (
        <Card className="dataset-page__progress animate-fade-in">
          <ProgressBar
            value={uploadProgress}
            max={uploadTotal}
            showLabel
            label={`Uploading ${uploadProgress} of ${uploadTotal} images...`}
            variant="accent"
          />
        </Card>
      )}

      {/* Image Grid */}
      {images.length > 0 && (
        <div className="dataset-page__grid stagger-children">
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
                    onClick={(e) => {
                      e.stopPropagation();
                      setPreviewImage(img);
                    }}
                    title="Preview"
                  >
                    <Eye size={16} />
                  </button>
                  <button
                    className="image-card__action image-card__action--danger"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(img.id);
                    }}
                    title="Delete"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
                {selectedImages.has(img.id) && (
                  <div className="image-card__check">✓</div>
                )}
              </div>
              <div className="image-card__info">
                <span className="image-card__name truncate">{img.filename}</span>
                <span className="image-card__size">{formatSize(img.size)}</span>
              </div>
              {/* Display Captions or Loading */}
              {(captioningMap[img.id] || (img.captions && img.captions.length > 0)) && (
                <div className="image-card__captions" style={{ padding: '0 8px 8px', display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                  {captioningMap[img.id] ? (
                    <span style={{ fontSize: '10px', color: 'var(--color-text-secondary)' }}>Captioning...</span>
                  ) : (
                    img.captions!.slice(0, 3).map(tag => (
                      <span key={tag} style={{ background: 'var(--color-bg-hover)', border: '1px solid var(--color-surface-border)', padding: '2px 4px', borderRadius: '4px', fontSize: '10px', color: 'var(--color-text-secondary)' }}>
                        {tag}
                      </span>
                    ))
                  )}
                  {img.captions && img.captions.length > 3 && (
                    <span style={{ fontSize: '10px', color: 'var(--color-text-disabled)' }}>+{img.captions.length - 3}</span>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Empty State */}
      {images.length === 0 && !uploading && (
        <div className="dataset-page__empty animate-fade-in">
          <Grid3X3 size={48} className="dataset-page__empty-icon" />
          <p className="dataset-page__empty-text">
            No images yet. Upload your training dataset to get started.
          </p>
        </div>
      )}

      {/* Image Preview Modal */}
      <Modal
        isOpen={!!previewImage}
        onClose={() => setPreviewImage(null)}
        title={previewImage?.filename}
        size="lg"
      >
        {previewImage && (
          <div className="image-preview">
            <img
              src={previewImage.url}
              alt={previewImage.filename}
              className="image-preview__img"
            />
            <div className="image-preview__meta">
              <span>Size: {formatSize(previewImage.size)}</span>
              <span>
                Dimensions: {previewImage.width}×{previewImage.height}
              </span>
            </div>
            {/* Editable Tags Section */}
            <div className="image-preview__captions" style={{ marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <div style={{ fontSize: '12px', color: 'var(--color-text-secondary)', fontWeight: 600 }}>Captions / Tags:</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {(previewImage.captions || []).map(tag => (
                  <span key={tag} style={{ background: 'var(--color-bg-elevated)', border: '1px solid var(--color-surface-border)', padding: '4px 8px', borderRadius: '4px', fontSize: '12px', color: 'var(--color-text-primary)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                    {tag}
                    <button 
                      onClick={() => handleRemoveTag(tag)}
                      style={{ background: 'none', border: 'none', color: 'var(--color-text-tertiary)', cursor: 'pointer', padding: 0, display: 'flex' }}
                    >
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
    </div>
  );
}
