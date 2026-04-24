import React, { createContext, useContext, useReducer, ReactNode } from 'react';
import type { AppState, AppAction, TrainingConfig, TrainingStatus, BaseModel, DatasetImage } from '../types';

const defaultConfig: TrainingConfig = {
  id: 'default',
  name: 'Default Config',
  baseModel: 'sdxl',
  learningRate: 1e-4,
  trainingSteps: 1000,
  loraRank: 16,
  networkAlpha: 8,
  batchSize: 1,
  resolution: 1024,
  optimizer: 'AdamW',
  scheduler: 'cosine',
  warmupSteps: 100,
  seed: 42,
  mixedPrecision: 'bf16',
  gradientAccumulation: 1,
  clipSkip: 1,
  enableBucketing: true,
  captionDropout: 0.1,
  noiseOffset: 0.05,
};

const defaultTrainingStatus: TrainingStatus = {
  phase: 'idle',
  currentStep: 0,
  totalSteps: 0,
  currentLoss: 0,
  avgLoss: 0,
  learningRate: 0,
  elapsed: 0,
  eta: 0,
  epochsCompleted: 0,
  lossHistory: [],
  logs: [],
};

const initialState: AppState = {
  datasets: [],
  currentDataset: null,
  trainingConfig: defaultConfig,
  trainingStatus: defaultTrainingStatus,
  models: [],
  baseModels: [],
  modelsDirectory: '',
  wsConnected: false,
};

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_DATASETS':
      return { ...state, datasets: action.payload };

    case 'SET_CURRENT_DATASET':
      return { ...state, currentDataset: action.payload };

    case 'ADD_DATASET_IMAGE': {
      const datasets = state.datasets.map((ds) =>
        ds.id === action.payload.datasetId
          ? { ...ds, images: [...ds.images, action.payload.image] }
          : ds
      );
      const currentDataset =
        state.currentDataset?.id === action.payload.datasetId
          ? { ...state.currentDataset, images: [...state.currentDataset.images, action.payload.image] }
          : state.currentDataset;
      return { ...state, datasets, currentDataset };
    }

    case 'REMOVE_DATASET_IMAGE': {
      const datasets = state.datasets.map((ds) =>
        ds.id === action.payload.datasetId
          ? { ...ds, images: ds.images.filter((img) => img.id !== action.payload.imageId) }
          : ds
      );
      const currentDataset =
        state.currentDataset?.id === action.payload.datasetId
          ? {
            ...state.currentDataset,
            images: state.currentDataset.images.filter(
              (img) => img.id !== action.payload.imageId
            ),
          }
          : state.currentDataset;
      return { ...state, datasets, currentDataset };
    }

    case 'UPDATE_DATASET_IMAGE_CAPTIONS': {
      const updateImages = (images: DatasetImage[]) =>
        images.map(img => img.id === action.payload.imageId ? { ...img, captions: action.payload.captions } : img);

      const datasets = state.datasets.map(ds =>
        ds.id === action.payload.datasetId ? { ...ds, images: updateImages(ds.images) } : ds
      );

      const currentDataset = state.currentDataset?.id === action.payload.datasetId
        ? { ...state.currentDataset, images: updateImages(state.currentDataset.images) }
        : state.currentDataset;

      return { ...state, datasets, currentDataset };
    }

    case 'UPDATE_CONFIG':
      return {
        ...state,
        trainingConfig: { ...state.trainingConfig, ...action.payload },
      };

    case 'SET_TRAINING_STATUS':
      return {
        ...state,
        trainingStatus: { ...state.trainingStatus, ...action.payload },
      };

    case 'ADD_TRAINING_STEP':
      return {
        ...state,
        trainingStatus: {
          ...state.trainingStatus,
          lossHistory: [...state.trainingStatus.lossHistory, action.payload],
          currentStep: action.payload.step,
          currentLoss: action.payload.loss,
          learningRate: action.payload.learningRate,
        },
      };

    case 'ADD_LOG':
      return {
        ...state,
        trainingStatus: {
          ...state.trainingStatus,
          logs: [...state.trainingStatus.logs, action.payload],
        },
      };

    case 'SET_MODELS':
      return { ...state, models: action.payload };

    case 'ADD_MODEL':
      return { ...state, models: [...state.models, action.payload] };

    case 'SET_WS_CONNECTED':
      return { ...state, wsConnected: action.payload };

    case 'SET_BASE_MODELS':
      return { ...state, baseModels: action.payload };

    case 'UPDATE_BASE_MODEL':
      return {
        ...state,
        baseModels: state.baseModels.map((m) =>
          m.id === action.payload.id ? { ...m, ...action.payload } : m
        ),
      };

    case 'ADD_BASE_MODEL':
      return { ...state, baseModels: [...state.baseModels, action.payload] };

    case 'REMOVE_BASE_MODEL':
      return {
        ...state,
        baseModels: state.baseModels.filter((m) => m.id !== action.payload),
      };

    case 'SET_MODELS_DIRECTORY':
      return { ...state, modelsDirectory: action.payload };

    default:
      return state;
  }
}

interface AppContextType {
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
}

export function useApp(): AppContextType {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
}

export { defaultConfig, defaultTrainingStatus };
