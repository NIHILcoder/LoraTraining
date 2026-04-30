import { contextBridge, ipcRenderer } from 'electron';

/**
 * Preload script — exposes a narrow, typed API to the renderer process
 * via contextBridge. This is the ONLY way the renderer can interact with
 * Node/Electron. Direct access to require, ipcRenderer, etc. is blocked.
 *
 * P0-01: Harden Electron Security Boundary
 */

type IpcCallback = (...args: any[]) => void;

const loraStudioAPI = {
  // --- Window Controls ---
  windowMinimize: () => ipcRenderer.send('window-min'),
  windowMaximize: () => ipcRenderer.send('window-max'),
  windowClose: () => ipcRenderer.send('window-close'),

  // --- Progress Bar & Notifications ---
  setProgressBar: (progress: number) => ipcRenderer.send('set-progress-bar', progress),
  showNotification: (title: string, body: string) => ipcRenderer.send('show-notification', title, body),

  // --- Shell ---
  openExternal: (url: string) => ipcRenderer.invoke('open-external', url),

  // --- Environment Setup ---
  checkEnv: (): Promise<boolean> => ipcRenderer.invoke('check-env'),
  installEnv: () => ipcRenderer.send('install-env'),
  startBackend: () => ipcRenderer.send('start-backend'),

  // --- File Dialogs ---
  selectDirectory: (title?: string): Promise<string | null> => ipcRenderer.invoke('select-directory', title),

  // --- Backend Config (dynamic, injected at startup) ---
  getBackendPort: (): Promise<number> => ipcRenderer.invoke('get-backend-port'),
  getBackendToken: (): Promise<string> => ipcRenderer.invoke('get-backend-token'),

  // --- Event Listeners ---
  on: (channel: string, callback: IpcCallback) => {
    const validChannels = [
      'install-log',
      'install-progress',
      'install-step',
      'install-complete',
      'backend-log',
      'backend-started',
    ];
    if (validChannels.includes(channel)) {
      const wrappedCallback = (_event: Electron.IpcRendererEvent, ...args: any[]) => callback(...args);
      ipcRenderer.on(channel, wrappedCallback);
      return wrappedCallback;
    }
    return null;
  },

  once: (channel: string, callback: IpcCallback) => {
    const validChannels = [
      'install-complete',
      'backend-started',
    ];
    if (validChannels.includes(channel)) {
      ipcRenderer.once(channel, (_event, ...args) => callback(...args));
    }
  },

  removeAllListeners: (channel: string) => {
    const validChannels = [
      'install-log',
      'install-progress',
      'install-step',
      'install-complete',
      'backend-log',
      'backend-started',
    ];
    if (validChannels.includes(channel)) {
      ipcRenderer.removeAllListeners(channel);
    }
  },
};

contextBridge.exposeInMainWorld('loraStudio', loraStudioAPI);
