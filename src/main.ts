import { app, BrowserWindow, ipcMain, Notification, dialog, shell } from 'electron';
import { autoUpdater } from 'electron-updater';
import * as path from 'path';
import * as net from 'net';
import * as crypto from 'crypto';

import { checkEnvExists, installEnvironment, startBackend, stopBackend, isBackendRunning } from './backend_manager';

// P0-02: Generate a strong random token per session to secure the local backend
const backendApiToken = crypto.randomBytes(32).toString('hex');


/**
 * Find a free TCP port starting from `preferred`.
 * Falls back to OS-assigned port if preferred is occupied.
 */
function findFreePort(preferred: number): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on('error', () => {
      // preferred port busy — let OS assign one
      const fallback = net.createServer();
      fallback.unref();
      fallback.on('error', reject);
      fallback.listen(0, '127.0.0.1', () => {
        const addr = fallback.address();
        fallback.close(() => resolve((addr as net.AddressInfo).port));
      });
    });
    server.listen(preferred, '127.0.0.1', () => {
      server.close(() => resolve(preferred));
    });
  });
}

/** The dynamically assigned backend port for this session */
let backendPort: number = 8000;

// --- Auto-update (electron-updater + GitHub Releases) ---
let autoUpdaterWired = false;
let rendererSend: ((channel: string, ...args: any[]) => void) | null = null;

function wireAutoUpdater() {
  if (autoUpdaterWired) return;
  autoUpdaterWired = true;

  autoUpdater.autoDownload = true;          // fetch the update in the background once found
  autoUpdater.autoInstallOnAppQuit = true;  // apply it on next quit if the user doesn't restart sooner

  const emit = (data: any) => { if (rendererSend) rendererSend('update-event', data); };

  autoUpdater.on('checking-for-update', () => emit({ type: 'checking' }));
  autoUpdater.on('update-available', (info) => emit({ type: 'available', version: info?.version }));
  autoUpdater.on('update-not-available', () => emit({ type: 'not-available' }));
  autoUpdater.on('download-progress', (p) => emit({
    type: 'progress',
    percent: Math.round(p.percent),
    bytesPerSecond: Math.round(p.bytesPerSecond),
    transferred: p.transferred,
    total: p.total,
  }));
  autoUpdater.on('update-downloaded', (info) => emit({ type: 'downloaded', version: info?.version }));
  autoUpdater.on('error', (err) => emit({ type: 'error', message: (err && (err as Error).message) || String(err) }));
}

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 1024,
    minHeight: 700,
    frame: false,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
    backgroundColor: '#0a0a0a',
  });

  const isDev = !app.isPackaged && process.env.NODE_ENV !== 'production';

  if (isDev) {
    mainWindow.loadURL('http://localhost:3005');
  } else {
    mainWindow.loadFile(path.join(__dirname, 'index.html'));
  }

  // Helper: safely send to renderer — no-op if window/webContents destroyed
  const safeSend = (channel: string, ...args: any[]) => {
    if (mainWindow && !mainWindow.isDestroyed() && mainWindow.webContents && !mainWindow.webContents.isDestroyed()) {
      mainWindow.webContents.send(channel, ...args);
    }
  };

  // Route auto-updater events to the current window; register updater listeners once
  rendererSend = safeSend;
  wireAutoUpdater();

  // Setup IPC for custom titlebar
  ipcMain.removeAllListeners('window-min');
  ipcMain.removeAllListeners('window-max');
  ipcMain.removeAllListeners('window-close');
  ipcMain.removeAllListeners('set-progress-bar');
  ipcMain.removeAllListeners('show-notification');
  ipcMain.removeAllListeners('install-env');
  ipcMain.removeAllListeners('start-backend');
  ipcMain.removeAllListeners('select-directory');
  ipcMain.removeAllListeners('install-update');

  ipcMain.on('window-min', () => mainWindow.minimize());
  ipcMain.on('window-max', () => {
    if (mainWindow.isMaximized()) mainWindow.unmaximize();
    else mainWindow.maximize();
  });
  ipcMain.on('window-close', () => mainWindow.close());

  // Taskbar Progress and Notifications
  ipcMain.on('set-progress-bar', (_event, progress: number) => {
    if (!mainWindow.isDestroyed()) {
      mainWindow.setProgressBar(progress);
    }
  });

  ipcMain.on('show-notification', (_event, title: string, body: string) => {
    if (Notification.isSupported()) {
      new Notification({
        title,
        body,
        icon: path.join(__dirname, 'icon.png')
      }).show();
    }
  });

  // Backend Setup IPC — handle must be removed before re-registering
  ipcMain.removeHandler('check-env');
  ipcMain.removeHandler('get-backend-port');
  ipcMain.removeHandler('get-backend-token');
  ipcMain.removeHandler('select-directory');
  ipcMain.removeHandler('select-file');
  ipcMain.removeHandler('open-external');
  ipcMain.removeHandler('check-for-updates');

  ipcMain.handle('check-env', () => checkEnvExists());

  // P0-04: Return dynamic backend port to renderer via preload
  ipcMain.handle('get-backend-port', () => backendPort);

  // P0-02: Return the session's API token to the renderer
  ipcMain.handle('get-backend-token', () => backendApiToken);

  // Shell — open external URLs in default browser
  ipcMain.handle('open-external', (_event, url: string) => {
    return shell.openExternal(url);
  });

  ipcMain.on('install-env', async (_event) => {
    try {
      await installEnvironment(
        (msg) => {
          console.log(`[Install] ${msg}`);
          safeSend('install-log', msg);
        },
        (pct) => safeSend('install-progress', pct),
        (stepName, pct) => {
          safeSend('install-step', stepName);
          safeSend('install-progress', pct);
        }
      );
      safeSend('install-complete', { success: true });
    } catch (err: any) {
      console.error(`[Install Error] ${err.message}`);
      safeSend('install-complete', { success: false, error: err.message });
    }
  });

  ipcMain.on('start-backend', async (_event) => {
    try {
      if (!isBackendRunning()) {
        // P0-04: Find a free port instead of force-killing port 8000
        backendPort = await findFreePort(8000);
        console.log(`[Backend] Using new port ${backendPort}`);
      } else {
        console.log(`[Backend] Already running on port ${backendPort}`);
      }

      await startBackend((msg) => {
        console.log(`[Backend] ${msg}`);
        safeSend('backend-log', msg);
      }, backendPort, backendApiToken);
      safeSend('backend-started', { success: true, port: backendPort });
    } catch (err: any) {
      console.error(`[Backend Error] ${err.message}`);
      safeSend('backend-started', { success: false, error: err.message });
    }
  });

  ipcMain.handle('select-directory', async (_event, title?: string) => {
    const result = await dialog.showOpenDialog(mainWindow, {
      title: title || 'Select Directory',
      properties: ['openDirectory', 'createDirectory'],
    });
    if (result.canceled || result.filePaths.length === 0) return null;
    return result.filePaths[0];
  });

  ipcMain.handle('select-file', async (_event, title?: string, filters?: Electron.FileFilter[]) => {
    const result = await dialog.showOpenDialog(mainWindow, {
      title: title || 'Select File',
      properties: ['openFile'],
      filters: filters && filters.length ? filters : [{ name: 'Models', extensions: ['safetensors'] }],
    });
    if (result.canceled || result.filePaths.length === 0) return null;
    return result.filePaths[0];
  });

  // --- Auto-update ---
  ipcMain.handle('check-for-updates', async () => {
    if (!app.isPackaged) {
      return { ok: false, error: 'Updates are only available in the installed app.' };
    }
    try {
      const result = await autoUpdater.checkForUpdates();
      return { ok: true, version: result?.updateInfo?.version };
    } catch (e: any) {
      return { ok: false, error: e?.message || String(e) };
    }
  });

  ipcMain.on('install-update', () => {
    if (app.isPackaged) autoUpdater.quitAndInstall();
  });
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('will-quit', () => {
  stopBackend();
});

app.on('window-all-closed', () => {
  stopBackend();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
