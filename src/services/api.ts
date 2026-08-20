// ============================================
// Mock API Service
// Will be replaced with real backend endpoints
// ============================================

import type {
  Dataset,
  DatasetImage,
  TrainingConfig,
} from '../types';

// P0-04: Dynamic backend port — fetched from preload API at startup.
// Falls back to 8000 for browser-only development mode.

let _backendPort = 8000;
let _backendToken = '';
let _portResolved = false;

async function resolvePort(): Promise<number> {
  if (_portResolved) return _backendPort;
  try {
    const port = await window.loraStudio?.getBackendPort();
    if (port) _backendPort = port;
    const token = await window.loraStudio?.getBackendToken();
    if (token) _backendToken = token;
  } catch (e) { /* browser-only mode — use default */ }
  _portResolved = true;
  return _backendPort;
}

/** Call once during app init to resolve the port early */
export async function initApiPort(): Promise<void> {
  await resolvePort();
}

export function getApiBase(): string {
  return `http://localhost:${_backendPort}/api`;
}

export function getWsUrl(path: string): string {
  return `ws://localhost:${_backendPort}${path}?token=${_backendToken}`;
}

async function apiFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const headers = new Headers(init?.headers);
  if (_backendToken) {
    headers.set('Authorization', `Bearer ${_backendToken}`);
  }
  const response = await fetch(input, { ...init, headers });
  
  // Intercept errors and throw with detailed message
  if (!response.ok) {
    let errorMsg = `HTTP Error ${response.status}`;
    try {
      const errData = await response.json();
      if (errData.detail) errorMsg = errData.detail;
      else if (errData.error) errorMsg = errData.error;
    } catch (e) {
      // Not JSON or empty body
    }
    throw new Error(errorMsg);
  }
  
  // Some endpoints still return {"error": "..."} with 200 OK
  // We can intercept that if we inspect the body, but it's better to just fix the backend.
  return response;
}

// --- Helpers ---
function generateId(): string {
  return Math.random().toString(36).substring(2, 15);
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// --- API Functions ---

export async function fetchDatasets(): Promise<Dataset[]> {
  const response = await apiFetch(`${getApiBase()}/datasets`);
  if (!response.ok) throw new Error('Failed to fetch datasets');
  const data = await response.json();
  // Prefix URLs with the dynamic getApiBase() minus /api since the backend returns /api/...
  // Actually, we can just replace /api with getApiBase()
  const base = getApiBase().replace(/\/api$/, '');
  return data.map((ds: any) => ({
    ...ds,
    images: ds.images.map((img: any) => ({
      ...img,
      url: `${base}${img.url}?token=${_backendToken}`
    }))
  }));
}

export async function createDataset(name: string): Promise<Dataset> {
  const response = await apiFetch(`${getApiBase()}/datasets`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
  if (!response.ok) throw new Error('Failed to create dataset');
  return response.json();
}

export async function deleteDataset(id: string): Promise<void> {
  const response = await apiFetch(`${getApiBase()}/datasets/${id}`, {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error('Failed to delete dataset');
}

export async function updateDataset(dataset: Dataset): Promise<Dataset> {
  // Strip the dynamic origin from image URLs before saving
  const dsToSave = {
    ...dataset,
    images: dataset.images.map(img => {
      let relativeUrl = img.url;
      try {
        const urlObj = new URL(img.url);
        relativeUrl = urlObj.pathname;
      } catch (e) {
        // Not an absolute URL, leave it
      }
      return { ...img, url: relativeUrl };
    })
  };
  
  const response = await apiFetch(`${getApiBase()}/datasets/${dataset.id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset: dsToSave }),
  });
  if (!response.ok) throw new Error('Failed to update dataset');
  const data = await response.json();
  
  const base = getApiBase().replace(/\/api$/, '');
  return {
    ...data,
    images: data.images.map((img: any) => ({
      ...img,
      url: `${base}${img.url}?token=${_backendToken}`
    }))
  };
}

export async function uploadImage(
  datasetId: string,
  file: File & { path?: string }
): Promise<DatasetImage> {
  if (file.path) {
    const response = await apiFetch(`${getApiBase()}/datasets/${datasetId}/images/local`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filePath: file.path, filename: file.name, size: file.size })
    });
    if (!response.ok) throw new Error('Failed to upload image');
    const data = await response.json();
    const base = getApiBase().replace(/\/api$/, '');
    return {
      ...data,
      url: `${base}${data.url}?token=${_backendToken}`
    };
  }
  throw new Error("File path is required. Only local files are supported.");
}

export async function deleteImage(
  datasetId: string,
  imageId: string
): Promise<void> {
  const response = await apiFetch(`${getApiBase()}/datasets/${datasetId}/images/${imageId}`, {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error('Failed to delete image');
}

export async function updateImageCaptions(
  datasetId: string,
  imageId: string,
  captions: string[]
): Promise<void> {
  const response = await apiFetch(`${getApiBase()}/datasets/${datasetId}/images/${imageId}/captions`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ captions }),
  });
  if (!response.ok) throw new Error('Failed to update captions');
}

export async function updateManyImageCaptions(
  datasetId: string,
  updates: { imageId: string; captions: string[] }[]
): Promise<void> {
  if (updates.length === 0) return;
  const response = await apiFetch(`${getApiBase()}/datasets/${datasetId}/captions`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ updates }),
  });
  if (!response.ok) throw new Error('Failed to update captions');
}

export async function autoCaptionImage(imageId: string, imageUrl: string): Promise<string[]> {
  const response = await apiFetch(`${getApiBase()}/dataset/caption`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageId, imageUrl }),
  });
  if (!response.ok) throw new Error('Failed to auto-caption image');
  const data = await response.json();
  return data.tags;
}

export async function autoCaptionBatch(
  images: { imageId: string; imageUrl: string }[]
): Promise<{ imageId: string; tags?: string[]; error?: string }[]> {
  const response = await apiFetch(`${getApiBase()}/dataset/caption/batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ images }),
  });
  if (!response.ok) throw new Error('Failed to batch-caption');
  const data = await response.json();
  return data.results || [];
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
}): Promise<{url: string, seed: number, mock?: boolean, error?: boolean, reason?: string}> {
  const response = await apiFetch(`${getApiBase()}/playground/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!response.ok) throw new Error('Failed to generate image');
  const data = await response.json();
  // Real generations return a served path; prefix origin + token so <img> can load it.
  if (data.url && typeof data.url === 'string' && data.url.startsWith('/api/generated/')) {
    const base = getApiBase().replace(/\/api$/, '');
    data.url = `${base}${data.url}?token=${_backendToken}`;
  }
  return data;
}

export async function fetchAvailableBaseModels(): Promise<{ id: string; name: string; architecture: string; filename: string }[]> {
  // Returns only models that are actually downloaded (have a local file)
  const res = await apiFetch(`${getApiBase()}/models/base`);
  if (!res.ok) return [];
  const data = await res.json();
  return (data.models || []).filter(
    (m: any) => m.status === 'downloaded' && m.supportedInference !== false
  );
}

export async function fetchModels(): Promise<any[]> {
  const response = await apiFetch(`${getApiBase()}/gallery/models`);
  if (!response.ok) throw new Error('Failed to fetch trained models');
  const data = await response.json();
  return data.models;
}

export async function deleteModel(modelId: string): Promise<void> {
  const response = await apiFetch(`${getApiBase()}/gallery/models/${modelId}`, {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error('Failed to delete model');
}

export async function openModelFolder(modelId: string): Promise<void> {
  const response = await apiFetch(`${getApiBase()}/gallery/models/${modelId}/open`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error('Failed to open folder');
}

export async function startTraining(
  config: TrainingConfig,
  images: { filePath?: string; captions?: string[] }[]
): Promise<{ sessionId?: string; error?: string }> {
  const response = await apiFetch(`${getApiBase()}/training/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config, images }),
  });
  if (!response.ok) throw new Error('Failed to start training');
  return response.json();
}

export async function fetchGpuInfo(): Promise<any> {
  const response = await apiFetch(`${getApiBase()}/gpu/info`);
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
  const response = await apiFetch(`${getApiBase()}/gpu/estimate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!response.ok) throw new Error('Failed to estimate training time');
  return response.json();
}

export async function stopTraining(sessionId: string): Promise<void> {
  const response = await apiFetch(`${getApiBase()}/training/stop/${sessionId}`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error('Failed to stop training');
}

// P1-04: pauseTraining removed — backend has no real pause/resume support.
// Will be re-added when LoRATrainer implements thread-safe pause.

// --- Base Model Management ---

export async function fetchBaseModels(): Promise<{ models: any[]; modelsDirectory: string }> {
  const response = await apiFetch(`${getApiBase()}/models/base`);
  if (!response.ok) throw new Error('Failed to fetch base models');
  return response.json();
}

export async function downloadBaseModel(modelId: string): Promise<{ status: string }> {
  const response = await apiFetch(`${getApiBase()}/models/base/${modelId}/download`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error('Failed to start download');
  return response.json();
}

export async function cancelBaseModelDownload(modelId: string): Promise<void> {
  const response = await apiFetch(`${getApiBase()}/models/base/${modelId}/cancel`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error('Failed to cancel download');
}

export async function deleteBaseModel(modelId: string): Promise<void> {
  const response = await apiFetch(`${getApiBase()}/models/base/${modelId}`, {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error('Failed to delete model');
}

export async function addCustomModel(
  url: string,
  name?: string,
  architecture?: string
): Promise<any> {
  const response = await apiFetch(`${getApiBase()}/models/base/custom`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, name, architecture }),
  });
  if (!response.ok) throw new Error('Failed to add custom model');
  return response.json();
}

export async function importLocalModel(
  filePath: string,
  name?: string,
  architecture?: string
): Promise<any> {
  const response = await apiFetch(`${getApiBase()}/models/base/import`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filePath, name, architecture }),
  });
  if (!response.ok) throw new Error('Failed to import model');
  return response.json();
}

export async function setModelsDirectory(path: string): Promise<{ modelsDirectory: string }> {
  const response = await apiFetch(`${getApiBase()}/models/directory`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  });
  if (!response.ok) throw new Error('Failed to set models directory');
  return response.json();
}

export async function getModelsDirectory(): Promise<{ modelsDirectory: string }> {
  const response = await apiFetch(`${getApiBase()}/models/directory`);
  if (!response.ok) throw new Error('Failed to get models directory');
  return response.json();
}

export async function getHfToken(): Promise<{ token: string, hasToken: boolean }> {
  const response = await apiFetch(`${getApiBase()}/settings/token`);
  if (!response.ok) throw new Error('Failed to get HF token');
  return response.json();
}

export async function setHfToken(token: string): Promise<void> {
  const response = await apiFetch(`${getApiBase()}/settings/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token }),
  });
  if (!response.ok) throw new Error('Failed to set HF token');
}

// --- Output Directory Management ---

export async function fetchOutputDirectory(): Promise<{ outputDirectory: string }> {
  const response = await apiFetch(`${getApiBase()}/output/directory`);
  if (!response.ok) throw new Error('Failed to get output directory');
  return response.json();
}

export async function setOutputDirectory(path: string): Promise<{ outputDirectory: string }> {
  const response = await apiFetch(`${getApiBase()}/output/directory`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  });
  if (!response.ok) throw new Error('Failed to set output directory');
  return response.json();
}

export async function openOutputDirectory(): Promise<void> {
  const response = await apiFetch(`${getApiBase()}/output/open`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error('Failed to open output directory');
}

export async function fetchGeneratedImages(): Promise<any[]> {
  const response = await apiFetch(`${getApiBase()}/gallery/images`);
  if (!response.ok) throw new Error('Failed to fetch generated images');
  const data = await response.json();
  // Backend returns a relative /api/generated/... path (older records may be absolute).
  // Normalize to the dynamic origin and append the auth token so <img> requests pass the gate.
  const base = getApiBase().replace(/\/api$/, '');
  return (data.images || []).map((img: any) => {
    let u: string = img.url || '';
    const idx = u.indexOf('/api/generated/');
    if (idx >= 0) u = `${base}${u.slice(idx)}?token=${_backendToken}`;
    return { ...img, url: u };
  });
}

export async function deleteGeneratedImage(id: string): Promise<void> {
  const response = await apiFetch(`${getApiBase()}/gallery/images/${id}`, {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error('Failed to delete generated image');
}

export { generateId };
