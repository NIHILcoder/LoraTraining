// ============================================
// Mock API Service
// Will be replaced with real backend endpoints
// ============================================

import type {
  Dataset,
  DatasetImage,
  TrainingConfig,
} from '../types';

const API_BASE = 'http://localhost:8000/api';

// --- Helpers ---
function generateId(): string {
  return Math.random().toString(36).substring(2, 15);
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// --- API Functions ---

export async function fetchDatasets(): Promise<Dataset[]> {
  await delay(300);
  return [];
}

export async function createDataset(name: string): Promise<Dataset> {
  await delay(200);
  return {
    id: generateId(),
    name,
    images: [],
    totalSize: 0,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
}

export async function uploadImage(
  _datasetId: string,
  file: File & { path?: string }
): Promise<DatasetImage> {
  // Simulate minor upload delay
  await delay(100 + Math.random() * 200);
  
  // Generate a lightweight thumbnail using Canvas to prevent UI freezing
  const thumbUrl = await new Promise<string>((resolve) => {
    const img = new Image();
    const objectUrl = URL.createObjectURL(file);
    
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const MAX_SIZE = 256;
      let width = img.width;
      let height = img.height;

      if (width > height) {
        if (width > MAX_SIZE) {
          height *= MAX_SIZE / width;
          width = MAX_SIZE;
        }
      } else {
        if (height > MAX_SIZE) {
          width *= MAX_SIZE / height;
          height = MAX_SIZE;
        }
      }
      
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(img, 0, 0, width, height);
      }
      
      resolve(canvas.toDataURL('image/jpeg', 0.7));
      URL.revokeObjectURL(objectUrl); // Free up raw image memory
    };
    img.src = objectUrl;
  });

  return {
    id: generateId(),
    filename: file.name,
    url: thumbUrl,
    filePath: file.path, 
    size: file.size,
    width: 1024, // Mocked original width
    height: 1024, // Mocked original height
    uploadedAt: new Date().toISOString(),
  };
}

export async function deleteImage(
  _datasetId: string,
  _imageId: string
): Promise<void> {
  await delay(200);
}

export async function autoCaptionImage(imageId: string, imageUrl: string): Promise<string[]> {
  const response = await fetch(`${API_BASE}/dataset/caption`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageId, imageUrl }),
  });
  if (!response.ok) throw new Error('Failed to auto-caption image');
  const data = await response.json();
  return data.tags;
}

export async function generateImage(params: {
  prompt: string;
  negativePrompt?: string;
  width?: number;
  height?: number;
  cfgScale?: number;
  steps?: number;
  seed?: number;
  loraWeight?: number;
  sampler?: string;
  loraModelId?: string | null;
  baseModelId?: string | null;
}): Promise<{url: string, seed: number, mock?: boolean, reason?: string}> {
  const response = await fetch(`${API_BASE}/playground/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!response.ok) throw new Error('Failed to generate image');
  return response.json();
}

export async function fetchAvailableBaseModels(): Promise<{ id: string; name: string; architecture: string; filename: string }[]> {
  // Returns only models that are actually downloaded (have a local file)
  const res = await fetch(`${API_BASE}/models/base`);
  if (!res.ok) return [];
  const data = await res.json();
  return (data.models || []).filter((m: any) => m.status === 'downloaded');
}

export async function fetchModels(): Promise<any[]> {
  const response = await fetch(`${API_BASE}/gallery/models`);
  if (!response.ok) throw new Error('Failed to fetch trained models');
  const data = await response.json();
  return data.models;
}

export async function deleteModel(modelId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/gallery/models/${modelId}`, {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error('Failed to delete model');
}

export async function openModelFolder(modelId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/gallery/models/${modelId}/open`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error('Failed to open folder');
}

export async function startTraining(
  config: TrainingConfig,
  images: { filePath?: string; captions?: string[] }[]
): Promise<{ sessionId?: string; error?: string }> {
  const response = await fetch(`${API_BASE}/training/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config, images }),
  });
  if (!response.ok) throw new Error('Failed to start training');
  return response.json();
}

export async function fetchGpuInfo(): Promise<any> {
  const response = await fetch(`${API_BASE}/gpu/info`);
  if (!response.ok) throw new Error('Failed to fetch GPU info');
  return response.json();
}

export async function estimateTrainingTime(params: {
  architecture: string;
  steps: number;
  rank: number;
  resolution: number;
  batchSize: number;
}): Promise<{ eta_seconds: number; time_per_step: number; feasible: boolean; reason: string | null }> {
  const response = await fetch(`${API_BASE}/gpu/estimate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!response.ok) throw new Error('Failed to estimate training time');
  return response.json();
}

export async function stopTraining(sessionId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/training/stop/${sessionId}`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error('Failed to stop training');
}

export async function pauseTraining(_sessionId: string): Promise<void> {
  await delay(200);
}

// --- Base Model Management ---

export async function fetchBaseModels(): Promise<{ models: any[]; modelsDirectory: string }> {
  const response = await fetch(`${API_BASE}/models/base`);
  if (!response.ok) throw new Error('Failed to fetch base models');
  return response.json();
}

export async function downloadBaseModel(modelId: string): Promise<{ status: string }> {
  const response = await fetch(`${API_BASE}/models/base/${modelId}/download`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error('Failed to start download');
  return response.json();
}

export async function cancelBaseModelDownload(modelId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/models/base/${modelId}/cancel`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error('Failed to cancel download');
}

export async function deleteBaseModel(modelId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/models/base/${modelId}`, {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error('Failed to delete model');
}

export async function addCustomModel(
  url: string,
  name?: string,
  architecture?: string
): Promise<any> {
  const response = await fetch(`${API_BASE}/models/base/custom`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, name, architecture }),
  });
  if (!response.ok) throw new Error('Failed to add custom model');
  return response.json();
}

export async function setModelsDirectory(path: string): Promise<{ modelsDirectory: string }> {
  const response = await fetch(`${API_BASE}/models/directory`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  });
  if (!response.ok) throw new Error('Failed to set models directory');
  return response.json();
}

export async function getModelsDirectory(): Promise<{ modelsDirectory: string }> {
  const response = await fetch(`${API_BASE}/models/directory`);
  if (!response.ok) throw new Error('Failed to get models directory');
  return response.json();
}

// --- Output Directory Management ---

export async function fetchOutputDirectory(): Promise<{ outputDirectory: string }> {
  const response = await fetch(`${API_BASE}/output/directory`);
  if (!response.ok) throw new Error('Failed to get output directory');
  return response.json();
}

export async function setOutputDirectory(path: string): Promise<{ outputDirectory: string }> {
  const response = await fetch(`${API_BASE}/output/directory`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  });
  if (!response.ok) throw new Error('Failed to set output directory');
  return response.json();
}

export async function openOutputDirectory(): Promise<void> {
  const response = await fetch(`${API_BASE}/output/open`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error('Failed to open output directory');
}

export async function fetchGeneratedImages(): Promise<any[]> {
  const response = await fetch(`${API_BASE}/gallery/images`);
  if (!response.ok) throw new Error('Failed to fetch generated images');
  const data = await response.json();
  return data.images;
}

export async function deleteGeneratedImage(id: string): Promise<void> {
  const response = await fetch(`${API_BASE}/gallery/images/${id}`, {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error('Failed to delete generated image');
}

export { generateId };
