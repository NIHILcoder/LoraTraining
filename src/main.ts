import { app, BrowserWindow, ipcMain, Notification, dialog, shell } from 'electron';
import * as path from 'path';
import * as net from 'net';

import { checkEnvExists, installEnvironment, startBackend, stopBackend } from './backend_manager';

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

  // Setup IPC for custom titlebar
  ipcMain.removeAllListeners('window-min');
  ipcMain.removeAllListeners('window-max');
  ipcMain.removeAllListeners('window-close');
  ipcMain.removeAllListeners('set-progress-bar');
  ipcMain.removeAllListeners('show-notification');
  ipcMain.removeAllListeners('install-env');
  ipcMain.removeAllListeners('start-backend');
  ipcMain.removeAllListeners('select-directory');

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
  ipcMain.removeHandler('select-directory');
  ipcMain.removeHandler('open-external');

  ipcMain.handle('check-env', () => checkEnvExists());

  // P0-04: Return dynamic backend port to renderer via preload
  ipcMain.handle('get-backend-port', () => backendPort);

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
      // P0-04: Find a free port instead of force-killing port 8000
      backendPort = await findFreePort(8000);
      console.log(`[Backend] Using port ${backendPort}`);

      await startBackend((msg) => {
        console.log(`[Backend] ${msg}`);
        safeSend('backend-log', msg);
      }, backendPort);
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
