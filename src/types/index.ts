// ============================================
// LoRA Training Dashboard — Type Definitions
// ============================================

// --- Base Model Types ---
export type ModelArchitecture = 'sd15' | 'sd21' | 'sdxl' | 'sd3' | 'flux' | 'cascade' | 'hunyuan' | 'pixart' | 'kolors' | 'auraflow';
export type ModelDownloadStatus = 'not_downloaded' | 'downloading' | 'downloaded' | 'error' | 'verifying';

export interface BaseModel {
  id: string;
  name: string;
  shortName: string;
  description: string;
  architecture: ModelArchitecture;
  fileSize: number;
  filename: string;
  downloadUrl: string;
  sha256?: string;
  status: ModelDownloadStatus;
  downloadProgress?: number;
  downloadSpeed?: string;
  downloadedBytes?: number;
  localPath?: string;
  error?: string;
  isCustom?: boolean;
}

// --- Dataset Types ---
export interface DatasetImage {
  id: string;
  filename: string;
  url: string;
  filePath?: string;
  size: number;
  width: number;
  height: number;
  caption?: string;
  captions?: string[];
  uploadedAt: string;
}

export interface Dataset {
  id: string;
  name: string;
  images: DatasetImage[];
  totalSize: number;
  createdAt: string;
  updatedAt: string;
}

// --- Training Configuration ---
export type OptimizerType = 'AdamW' | 'Prodigy' | 'DAdaptAdam';
export type SchedulerType = 'cosine' | 'linear' | 'constant' | 'cosine_with_restarts';
export type ResolutionType = 512 | 768 | 1024 | 1536;
export type BaseModelType = 'sd15' | 'sd21' | 'sdxl' | 'sd3' | 'flux' | 'cascade' | 'hunyuan' | 'pixart' | 'kolors' | 'auraflow';

export interface TrainingConfig {
  id: string;
  name: string;
  learningRate: number;
  trainingSteps: number;
  loraRank: number;
  networkAlpha: number;
  batchSize: number;
  resolution: ResolutionType;
  optimizer: OptimizerType;
  scheduler: SchedulerType;
  warmupSteps: number;
  seed: number;
  mixedPrecision: 'fp16' | 'bf16' | 'fp32';
  gradientAccumulation: number;
  clipSkip: number;
  datasetId?: string;
  baseModel: BaseModelType;
}

export interface ConfigPreset {
  id: string;
  name: string;
  description: string;
  icon: string;
  config: Partial<TrainingConfig>;
}

// --- Training Status ---
export type TrainingPhase = 'idle' | 'preparing' | 'training' | 'paused' | 'completed' | 'error';

export interface TrainingStep {
  step: number;
  loss: number;
  learningRate: number;
  timestamp: number;
}

export interface TrainingStatus {
  phase: TrainingPhase;
  currentStep: number;
  totalSteps: number;
  currentLoss: number;
  avgLoss: number;
  learningRate: number;
  elapsed: number;
  eta: number;
  epochsCompleted: number;
  lossHistory: TrainingStep[];
  logs: LogEntry[];
}

export interface LogEntry {
  id: string;
  timestamp: number;
  level: 'info' | 'warning' | 'error' | 'debug';
  message: string;
}

// --- LoRA Model ---
export interface LoraModel {
  id: string;
  name: string;
  description: string;
  filename: string;
  fileSize: number;
  rank: number;
  alpha: number;
  trainingSteps: number;
  finalLoss: number;
  config: TrainingConfig;
  sampleImages: SampleImage[];
  createdAt: string;
  tags: string[];
}

export interface SampleImage {
  id: string;
  url: string;
  prompt: string;
  seed: number;
  generatedAt: string;
}

// --- WebSocket Messages ---
export type WSMessage =
  | { type: 'training_update'; data: Partial<TrainingStatus> }
  | { type: 'training_step'; data: TrainingStep }
  | { type: 'training_log'; data: LogEntry }
  | { type: 'training_complete'; data: { modelId: string } }
  | { type: 'training_error'; data: { message: string } }
  | { type: 'download_progress'; data: { modelId: string; progress: number; speed: string; downloadedBytes: number; totalBytes: number } }
  | { type: 'download_complete'; data: { modelId: string; localPath: string } }
  | { type: 'download_error'; data: { modelId: string; message: string } };

// --- App State ---
export interface AppState {
  datasets: Dataset[];
  currentDataset: Dataset | null;
  trainingConfig: TrainingConfig;
  trainingStatus: TrainingStatus;
  models: LoraModel[];
  baseModels: BaseModel[];
  modelsDirectory: string;
  wsConnected: boolean;
}

export type AppAction =
  | { type: 'SET_DATASETS'; payload: Dataset[] }
  | { type: 'SET_CURRENT_DATASET'; payload: Dataset | null }
  | { type: 'ADD_DATASET_IMAGE'; payload: { datasetId: string; image: DatasetImage } }
  | { type: 'REMOVE_DATASET_IMAGE'; payload: { datasetId: string; imageId: string } }
  | { type: 'UPDATE_DATASET_IMAGE_CAPTIONS'; payload: { datasetId: string; imageId: string; captions: string[] } }
  | { type: 'UPDATE_CONFIG'; payload: Partial<TrainingConfig> }
  | { type: 'SET_TRAINING_STATUS'; payload: Partial<TrainingStatus> }
  | { type: 'ADD_TRAINING_STEP'; payload: TrainingStep }
  | { type: 'ADD_LOG'; payload: LogEntry }
  | { type: 'SET_MODELS'; payload: LoraModel[] }
  | { type: 'ADD_MODEL'; payload: LoraModel }
  | { type: 'SET_WS_CONNECTED'; payload: boolean }
  | { type: 'SET_BASE_MODELS'; payload: BaseModel[] }
  | { type: 'UPDATE_BASE_MODEL'; payload: { id: string } & Partial<BaseModel> }
  | { type: 'ADD_BASE_MODEL'; payload: BaseModel }
  | { type: 'REMOVE_BASE_MODEL'; payload: string }
  | { type: 'SET_MODELS_DIRECTORY'; payload: string };
