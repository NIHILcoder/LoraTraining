/**
 * Type declarations for the preload API exposed via contextBridge.
 * P0-01: Harden Electron Security Boundary
 */

type IpcCallback = (...args: any[]) => void;

interface LoraStudioAPI {
  // Window Controls
  windowMinimize: () => void;
  windowMaximize: () => void;
  windowClose: () => void;

  // Progress Bar & Notifications
  setProgressBar: (progress: number) => void;
  showNotification: (title: string, body: string) => void;

  // Shell
  openExternal: (url: string) => Promise<void>;

  // Environment Setup
  checkEnv: () => Promise<boolean>;
  installEnv: () => void;
  startBackend: () => void;

  // File Dialogs
  selectDirectory: (title?: string) => Promise<string | null>;

  // Backend Port
  getBackendPort: () => Promise<number>;

  // Event Listeners
  on: (channel: string, callback: IpcCallback) => any;
  once: (channel: string, callback: IpcCallback) => void;
  removeAllListeners: (channel: string) => void;
}

declare global {
  interface Window {
    loraStudio?: LoraStudioAPI;
  }
}

export {};
