import { describe, it, expect } from 'vitest';
import { appReducer, initialState } from './AppContext';
import type { AppState, DatasetImage } from '../types';

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

  it('replaces the dataset list and selects current for restore-on-boot', () => {
    const ds1 = { id: 'a', name: 'A', images: [], totalSize: 0, createdAt: '', updatedAt: '' };
    const ds2 = { id: 'b', name: 'B', images: [sampleImage], totalSize: 1, createdAt: '', updatedAt: '' };
    let state = appReducer(initialState, { type: 'SET_DATASETS', payload: [ds1, ds2] });
    expect(state.datasets.map(d => d.id)).toEqual(['a', 'b']);
    state = appReducer(state, { type: 'SET_CURRENT_DATASET', payload: ds2 });
    expect(state.currentDataset?.id).toBe('b');
    expect(state.currentDataset?.images.length).toBe(1);
  });
});
