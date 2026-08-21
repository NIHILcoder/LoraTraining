import { describe, it, expect } from 'vitest';
import { appReducer, initialState } from './AppContext';
import type { AppState, DatasetImage } from '../types';
import { TRAINING_BUSY_PHASES } from '../types';

function stateWithDataset(): AppState {
  const ds = { id: 'd1', name: 'DS', images: [], totalSize: 0, createdAt: '', updatedAt: '' };
  return { ...initialState, datasets: [ds], currentDataset: ds };
}

const sampleImage: DatasetImage = {
  id: 'img1', filename: 'a.png', url: '', size: 1, width: 10, height: 10, uploadedAt: '',
};

describe('appReducer', () => {
  it('caps lossHistory at 2000 entries and keeps the newest', () => {
    let state = initialState;
    for (let i = 0; i < 2100; i++) {
      state = appReducer(state, {
        type: 'ADD_TRAINING_STEP',
        payload: { step: i, loss: 1, learningRate: 1e-4, timestamp: i },
      });
    }
    const hist = state.trainingStatus.lossHistory;
    expect(hist.length).toBe(2000);
    expect(hist[hist.length - 1].step).toBe(2099);
    expect(state.trainingStatus.currentStep).toBe(2099);
  });

  it('caps logs at 1000 entries', () => {
    let state = initialState;
    for (let i = 0; i < 1100; i++) {
      state = appReducer(state, {
        type: 'ADD_LOG',
        payload: { id: String(i), timestamp: i, level: 'info', message: 'm' },
      });
    }
    expect(state.trainingStatus.logs.length).toBe(1000);
  });

  it('adds and removes a dataset image', () => {
    let state = stateWithDataset();
    state = appReducer(state, { type: 'ADD_DATASET_IMAGE', payload: { datasetId: 'd1', image: sampleImage } });
    expect(state.currentDataset?.images.length).toBe(1);
    expect(state.datasets[0].images.length).toBe(1);
    state = appReducer(state, { type: 'REMOVE_DATASET_IMAGE', payload: { datasetId: 'd1', imageId: 'img1' } });
    expect(state.currentDataset?.images.length).toBe(0);
  });

  it('updates image captions on both currentDataset and datasets', () => {
    let state = stateWithDataset();
    state = appReducer(state, { type: 'ADD_DATASET_IMAGE', payload: { datasetId: 'd1', image: sampleImage } });
    state = appReducer(state, {
      type: 'UPDATE_DATASET_IMAGE_CAPTIONS',
      payload: { datasetId: 'd1', imageId: 'img1', captions: ['cat', 'dog'] },
    });
    expect(state.currentDataset?.images[0].captions).toEqual(['cat', 'dog']);
    expect(state.datasets[0].images[0].captions).toEqual(['cat', 'dog']);
  });

  it('treats loading_model and saving as busy so Start stays disabled', () => {
    expect(TRAINING_BUSY_PHASES).toEqual(['preparing', 'loading_model', 'training', 'saving']);
    for (const phase of TRAINING_BUSY_PHASES) {
      const state = appReducer(initialState, { type: 'SET_TRAINING_STATUS', payload: { phase } });
      expect(TRAINING_BUSY_PHASES.includes(state.trainingStatus.phase)).toBe(true);
    }
    const idle = appReducer(initialState, { type: 'SET_TRAINING_STATUS', payload: { phase: 'idle' } });
    expect(TRAINING_BUSY_PHASES.includes(idle.trainingStatus.phase)).toBe(false);
  });
});
