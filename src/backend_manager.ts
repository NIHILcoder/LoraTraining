import * as path from 'path';
import * as fs from 'fs';
import * as https from 'https';
import { spawn } from 'child_process';
import { app } from 'electron';

const BACKEND_DIR = path.join(__dirname, '..', 'backend');
const ENV_DIR = path.join(BACKEND_DIR, 'env');
const PYTHON_EXE = path.join(ENV_DIR, 'Scripts', 'python.exe');

const UV_URL = 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip';
const UV_ZIP = path.join(BACKEND_DIR, `uv-${Date.now()}.zip`);
const UV_EXE = path.join(BACKEND_DIR, 'uv.exe');

const SETUP_MARKER = path.join(ENV_DIR, '.setup_complete');

export function checkEnvExists(): boolean {
  return fs.existsSync(PYTHON_EXE) && fs.existsSync(SETUP_MARKER);
}

function downloadFile(url: string, dest: string, onProgress: (pct: number) => void): Promise<void> {
  return new Promise((resolve, reject) => {
    https.get(url, (response) => {
      // Handle redirects
      if (response.statusCode === 301 || response.statusCode === 302) {
        return downloadFile(response.headers.location as string, dest, onProgress).then(resolve).catch(reject);
      }

      const file = fs.createWriteStream(dest);
      const total = parseInt(response.headers['content-length'] || '0', 10);
      let downloaded = 0;

      response.on('data', (chunk) => {
        downloaded += chunk.length;
        if (total > 0) {
          onProgress((downloaded / total) * 100);
        }
      });

      response.pipe(file);
      file.on('finish', () => {
        file.close();
      });
      file.on('close', () => {
        resolve();
      });
    }).on('error', (err) => {
      fs.unlink(dest, () => {});
      reject(err);
    });
  });
}

let currentInstallerProcess: ReturnType<typeof spawn> | null = null;

function runCommand(command: string, args: string[], cwd: string, onLog: (msg: string) => void): Promise<void> {
  return new Promise((resolve, reject) => {
    const proc = spawn(command, args, { cwd, shell: false });
    currentInstallerProcess = proc;

    proc.stdout.on('data', (data) => {
      onLog(data.toString());
    });

    proc.stderr.on('data', (data) => {
      onLog(data.toString());
    });

    proc.on('close', (code) => {
      currentInstallerProcess = null;
      if (code === 0) resolve();
      else reject(new Error(`Command failed with code ${code}`));
    });
    
    proc.on('error', (err) => {
      currentInstallerProcess = null;
      reject(err);
    });
  });
}

export async function installEnvironment(
  onLog: (msg: string) => void,
  onProgress: (pct: number) => void,
  onStep: (stepName: string, baseProgress: number) => void
): Promise<void> {
  if (!fs.existsSync(BACKEND_DIR)) {
    fs.mkdirSync(BACKEND_DIR, { recursive: true });
  }

  // Force kill any orphaned uv.exe processes from previous interrupted installs
  try {
    await runCommand('taskkill', ['/F', '/IM', 'uv.exe', '/T'], BACKEND_DIR, () => {});
  } catch (e) {
    // Ignore error if process is not found
  }

  // Ensure clean slate if previous install was aborted
  if (fs.existsSync(ENV_DIR)) {
    onLog('Cleaning up broken environment from previous attempt...');
    try {
      fs.rmSync(ENV_DIR, { recursive: true, force: true, maxRetries: 5, retryDelay: 500 });
    } catch (e: any) {
      onLog(`Warning: Could not fully delete env folder: ${e.message}`);
    }
  }

  try {
    const files = fs.readdirSync(BACKEND_DIR);
    for (const f of files) {
      if (f.startsWith('uv-') && f.endsWith('.zip')) {
        try { fs.unlinkSync(path.join(BACKEND_DIR, f)); } catch (e) {}
      }
    }
    try { fs.unlinkSync(UV_EXE); } catch (e) {}
  } catch (e) {}

  // 1. Download uv
  onStep('Downloading Astral uv (Package Manager)...', 0);
  onLog('Downloading uv package manager...');
  await downloadFile(UV_URL, UV_ZIP, (pct) => onProgress(pct * 0.05)); // 0% to 5%
  onLog('Downloaded uv successfully.\n');

  // 2. Unzip uv
  onStep('Extracting Package Manager...', 5);
  onLog('Extracting uv...');
  await runCommand(
    'powershell',
    ['-Command', `Expand-Archive -Path '${UV_ZIP}' -DestinationPath '${BACKEND_DIR}' -Force`],
    BACKEND_DIR,
    onLog
  );
  onLog('Extracted successfully.\n');

  // 3. Create virtual environment
  onStep('Creating Python 3.12 Environment...', 10);
  onLog('Creating Python 3.12 environment...');
  await runCommand(
    UV_EXE,
    ['venv', '--python', '3.12', 'env', '--clear'],
    BACKEND_DIR,
    onLog
  );
  onLog('Environment created successfully.\n');

  // 4. Install PyTorch with CUDA 12.1
  onStep('Downloading PyTorch with CUDA 12.1 (~2.5 GB)...', 20);
  onLog('Installing PyTorch (CUDA 12.1)... This will take a few minutes depending on your internet speed...');
  await runCommand(
    UV_EXE,
    [
      'pip', 'install', 'torch', 'torchvision',
      '--index-url', 'https://download.pytorch.org/whl/cu121',
      '--python', PYTHON_EXE
    ],
    BACKEND_DIR,
    onLog
  );
  onLog('PyTorch installed successfully.\n');

  // 5. Install other requirements
  onStep('Installing AI Dependencies (Diffusers, Transformers)...', 80);
  onLog('Installing application dependencies...');
  await runCommand(
    UV_EXE,
    ['pip', 'install', '-r', 'requirements.txt', '--python', PYTHON_EXE],
    BACKEND_DIR,
    onLog
  );
  onLog('Dependencies installed successfully.\n');

  // 6. Cleanup
  onStep('Finalizing Setup...', 95);
  onLog('Cleaning up installer files...');
  try {
    fs.unlinkSync(UV_ZIP);
    fs.unlinkSync(UV_EXE);
  } catch (e) {
    onLog('Notice: Could not delete temp files, continuing anyway.\n');
  }

  // Mark as fully installed
  fs.writeFileSync(SETUP_MARKER, 'ready');

  onStep('Setup Complete!', 100);
  onLog('✅ Setup Complete! Application is ready.');
}

let backendProcess: ReturnType<typeof spawn> | null = null;

export function startBackend(onLog: (msg: string) => void): Promise<void> {
  return new Promise((resolve, reject) => {
    if (backendProcess) {
      resolve();
      return;
    }

    onLog('Starting Python backend server...');
    
    // Ensure dependencies are up to date (fast if already installed)
    try {
      const { execSync } = require('child_process');
      onLog('Checking dependencies...');
      execSync(`"${PYTHON_EXE}" -m pip install omegaconf`, { cwd: BACKEND_DIR });
    } catch (e) {
      onLog('Warning: Could not auto-update dependencies. Training might fail if omegaconf is missing.');
    }
    
    // Kill any process already using port 8000 (Windows)
    try {
      const { execSync } = require('child_process');
      const output = execSync('netstat -ano | findstr :8000').toString();
      const lines = output.trim().split('\n');
      for (const line of lines) {
        const parts = line.trim().split(/\s+/);
        const pid = parts[parts.length - 1];
        if (pid && !isNaN(parseInt(pid))) {
          onLog(`Cleaning up port 8000 (PID: ${pid})...`);
          execSync(`taskkill /F /PID ${pid} /T`);
        }
      }
    } catch (e) {
      // Port likely not in use, ignore
    }

    backendProcess = spawn(
      `"${PYTHON_EXE}"`,
      ['-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', '8000'],
      { cwd: BACKEND_DIR, shell: true }
    );

    let isStarted = false;

    backendProcess.stdout?.on('data', (data) => {
      const msg = data.toString();
      onLog(msg);
      if (msg.includes('Application startup complete.') && !isStarted) {
        isStarted = true;
        resolve();
      }
    });

    backendProcess.stderr?.on('data', (data) => {
      const msg = data.toString();
      onLog(msg);
      if (msg.includes('Application startup complete.') && !isStarted) {
        isStarted = true;
        resolve();
      }
    });

    backendProcess.on('close', (code) => {
      backendProcess = null;
      if (!isStarted) {
        reject(new Error(`Backend exited prematurely with code ${code}`));
      }
    });
  });
}

export function stopBackend() {
  if (backendProcess) {
    backendProcess.kill('SIGTERM');
    backendProcess = null;
  }
  if (currentInstallerProcess) {
    currentInstallerProcess.kill('SIGKILL');
    currentInstallerProcess = null;
  }
}
