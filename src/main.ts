import { app, BrowserWindow, ipcMain, Notification } from 'electron';
import * as path from 'path';

import { checkEnvExists, installEnvironment, startBackend, stopBackend } from './backend_manager';

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 1024,
    minHeight: 700,
    frame: false, // Completely hide the standard titlebar
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
    backgroundColor: '#0a0a0a',
  });

  const isDev = !app.isPackaged && process.env.NODE_ENV !== 'production';

  if (isDev) {
    mainWindow.loadURL('http://localhost:3005');
  } else {
    mainWindow.loadFile(path.join(__dirname, 'index.html'));
  }

  // Setup IPC for custom titlebar
  ipcMain.removeAllListeners('window-min');
  ipcMain.removeAllListeners('window-max');
  ipcMain.removeAllListeners('window-close');

  ipcMain.on('window-min', () => mainWindow.minimize());
  ipcMain.on('window-max', () => {
    if (mainWindow.isMaximized()) mainWindow.unmaximize();
    else mainWindow.maximize();
  });
  ipcMain.on('window-close', () => mainWindow.close());

  // Taskbar Progress and Notifications
  ipcMain.on('set-progress-bar', (event, progress: number) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.setProgressBar(progress);
    }
  });

  ipcMain.on('show-notification', (event, title: string, body: string) => {
    if (Notification.isSupported()) {
      new Notification({
        title,
        body,
        icon: path.join(__dirname, 'icon.png')
      }).show();
    }
  });

  // Backend Setup IPC
  ipcMain.handle('check-env', () => checkEnvExists());

  ipcMain.on('install-env', async (event) => {
    try {
      await installEnvironment(
        (msg) => event.sender.send('install-log', msg),
        (pct) => event.sender.send('install-progress', pct),
        (stepName, pct) => {
          event.sender.send('install-step', stepName);
          event.sender.send('install-progress', pct);
        }
      );
      event.sender.send('install-complete', { success: true });
    } catch (err: any) {
      event.sender.send('install-complete', { success: false, error: err.message });
    }
  });

  ipcMain.on('start-backend', async (event) => {
    try {
      await startBackend((msg) => event.sender.send('backend-log', msg));
      event.sender.send('backend-started', { success: true });
    } catch (err: any) {
      event.sender.send('backend-started', { success: false, error: err.message });
    }
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


