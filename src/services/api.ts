// ============================================
// Mock API Service
// Will be replaced with real backend endpoints
// ============================================

import type {
  Dataset,
  DatasetImage,
  TrainingConfig,
  LoraModel,
  SampleImage,
} from '../types';

const API_BASE = 'http://localhost:8000/api';

// --- Helpers ---
function generateId(): string {
  return Math.random().toString(36).substring(2, 15);
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// --- Mock Data ---
const mockSampleImages: SampleImage[] = [
  {
    id: 's1',
    url: 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256"><rect width="256" height="256" fill="#1a1a2e"/><text x="128" y="128" text-anchor="middle" fill="#8B5CF6" font-size="14" font-family="sans-serif" dy="0.35em">Sample Gen 1</text></svg>'),
    prompt: 'a portrait in the style of <lora>',
    seed: 42,
    generatedAt: new Date().toISOString(),
  },
  {
    id: 's2',
    url: 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256"><rect width="256" height="256" fill="#1a1a2e"/><text x="128" y="128" text-anchor="middle" fill="#22c55e" font-size="14" font-family="sans-serif" dy="0.35em">Sample Gen 2</text></svg>'),
    prompt: 'a landscape in the style of <lora>',
    seed: 123,
    generatedAt: new Date().toISOString(),
  },
];

const mockModels: LoraModel[] = [
  {
    id: 'm1',
    name: 'Portrait Style v1',
    description: 'Fine-tuned on portrait photography dataset. Best for realistic face generation.',
    filename: 'portrait_style_v1.safetensors',
    fileSize: 52428800,
    rank: 16,
    alpha: 8,
    trainingSteps: 2000,
    finalLoss: 0.0823,
    config: {} as TrainingConfig,
    sampleImages: mockSampleImages,
    createdAt: '2026-04-05T10:30:00Z',
    tags: ['portrait', 'realistic', 'face'],
  },
  {
    id: 'm2',
    name: 'Anime Style v2',
    description: 'Trained on curated anime artwork. Produces consistent anime-style outputs.',
    filename: 'anime_style_v2.safetensors',
    fileSize: 38912000,
    rank: 32,
    alpha: 16,
    trainingSteps: 3000,
    finalLoss: 0.0654,
    config: {} as TrainingConfig,
    sampleImages: mockSampleImages,
    createdAt: '2026-04-07T14:15:00Z',
    tags: ['anime', 'illustration', 'style'],
  },
  {
    id: 'm3',
    name: 'Landscape Fine-tune',
    description: 'Landscape photography enhancement. Improves scenic compositions.',
    filename: 'landscape_ft.safetensors',
    fileSize: 41943040,
    rank: 8,
    alpha: 4,
    trainingSteps: 1500,
    finalLoss: 0.0912,
    config: {} as TrainingConfig,
    sampleImages: mockSampleImages,
    createdAt: '2026-04-08T09:45:00Z',
    tags: ['landscape', 'nature', 'scenic'],
  },
];

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

export async function generateImage(params: any): Promise<{url: string, seed: number}> {
  const response = await fetch(`${API_BASE}/playground/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!response.ok) throw new Error('Failed to generate image');
  return response.json();
}

export async function fetchModels(): Promise<LoraModel[]> {
  await delay(400);
  return mockModels;
}

export async function deleteModel(_modelId: string): Promise<void> {
  await delay(300);
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

export { generateId };
