import { app, BrowserWindow, ipcMain, Notification } from 'electron';
import * as path from 'path';

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
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
    // progress is a number between 0 and 1, or -1 to remove
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.setProgressBar(progress);
    }
  });

  ipcMain.on('show-notification', (event, title: string, body: string) => {
    if (Notification.isSupported()) {
      new Notification({
        title,
        body,
        icon: path.join(__dirname, 'icon.png') // Optional
      }).show();
    }
  });
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
