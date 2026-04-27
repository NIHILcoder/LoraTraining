# LoRA Studio

LoRA Studio is a Windows-first Electron desktop application for preparing datasets, managing base diffusion models, training LoRA adapters, and testing generated outputs from one local interface.

The project is currently in `1.0.0-beta.1`. Treat it as an active beta: the UI and packaging flow are usable, but training stability still depends heavily on the installed Python environment, GPU driver, CUDA-compatible PyTorch build, available VRAM, and the selected base model.

## Current Scope

LoRA Studio provides:

- Dataset staging with local image selection, thumbnail generation, and caption/tag editing.
- Training configuration for SD 1.5, SD 2.1, SDXL, SD3, Flux, and Stable Cascade model families.
- Local base model management, including downloads and custom model registration.
- GPU and memory visibility with architecture-specific feasibility estimates.
- Training progress, logs, and loss history through a local WebSocket bridge.
- Playground generation using local base models and trained LoRA weights.
- Gallery views for trained LoRA outputs and generated images.
- Windows installer packaging through Electron Builder and NSIS.

The frontend still contains some mock dataset helpers while the training, model, gallery, output directory, GPU, and playground workflows talk to the local FastAPI backend.

## Requirements

### Runtime

- Windows 10 or Windows 11.
- NVIDIA GPU recommended.
- 8 GB VRAM minimum for smaller SD 1.5 workflows.
- 12 GB or more VRAM recommended for SDXL.
- Flux workflows can require substantially more VRAM and disk space.
- Stable internet connection for first-time Python dependency and model downloads.

### Development

- Node.js 18 or newer.
- npm 9 or newer.
- Python is installed by the app setup flow through `uv` when running the desktop app.
- A recent NVIDIA driver if GPU training or inference is expected.

## Project Layout

```text
.
|-- backend/                 # FastAPI backend, trainer, Python requirements
|-- installer-assets/        # NSIS installer branding assets
|-- public/                  # HTML template and application icons
|-- src/
|   |-- components/          # Reusable React UI components
|   |-- context/             # App state provider
|   |-- hooks/               # WebSocket and UI hooks
|   |-- pages/               # Main application screens
|   |-- services/            # Backend API client
|   |-- App.tsx              # Application shell and router
|   |-- backend_manager.ts   # Electron-side backend setup and process control
|   `-- main.ts              # Electron main process
|-- package.json             # npm scripts and Electron Builder config
|-- webpack.config.js        # Renderer and Electron main builds
`-- tsconfig.json
```

Generated folders such as `dist/`, `node_modules/`, `backend/env/`, model files, datasets, and training outputs are intentionally ignored by Git.

## Installation

Install JavaScript dependencies:

```bash
npm install
```

Start the full Electron development flow:

```bash
npm run electron:dev
```

This command:

1. Builds the Electron main process in development mode.
2. Starts the Webpack dev server on `http://localhost:3005`.
3. Launches Electron after the dev server is reachable.

On first desktop launch, the application checks `backend/env`. If the Python environment is missing, it opens the setup screen. The setup flow downloads `uv`, creates a Python 3.12 virtual environment, installs PyTorch with CUDA 12.1, then installs `backend/requirements.txt`.

## Development Commands

```bash
npm run dev
```

Starts only the renderer dev server on port `3005`. This is useful for frontend-only work, but it does not provide Electron IPC features.

```bash
npm run electron:build
```

Builds only the Electron main process into `dist/main.js`.

```bash
npm run electron:start
```

Starts Electron from the already built `dist/main.js`.

```bash
npm run type-check
```

Runs TypeScript validation without emitting files.

```bash
npm run build
```

Builds the production renderer and Electron main process into `dist/`.

```bash
npm run electron:dist
```

Builds the production app and creates a Windows NSIS installer in `dist/release/`.

## Backend

The backend is a FastAPI application served locally on port `8000`. Electron starts it automatically after the Python environment is ready.

Main backend responsibilities:

- Base model catalog and download management.
- Custom model registration.
- Training session orchestration.
- GPU and VRAM inspection.
- Output directory management.
- Gallery metadata and file management.
- Playground image generation.
- WebSocket progress updates.

Manual backend startup is only needed for debugging:

```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Use the Python interpreter from `backend/env/Scripts/python.exe` when debugging the environment created by the app.

## Packaging

The project uses Electron Builder with the NSIS target.

Current packaging behavior:

- Output directory: `dist/release/`
- Installer target: NSIS
- Installer mode: assisted wizard
- Installation path selection: enabled
- Desktop shortcut: enabled
- Start menu shortcut: enabled
- Custom installer and uninstaller sidebars: enabled
- Custom installer header: enabled

Build the installer:

```bash
npm run electron:dist
```

The generated installer is:

```text
dist/release/LoRA Studio Setup 1.0.0-beta.1.exe
```

Installer branding files live in `installer-assets/` and are referenced from the `build.nsis` section of `package.json`.

## Ports

LoRA Studio uses fixed local ports during development and runtime:

- `3005` for the Webpack renderer dev server.
- `8000` for the FastAPI backend.

If port `8000` is already occupied, the Electron backend manager attempts to clean up the conflicting process on Windows before starting the backend.

## Troubleshooting

### The packaged app opens to a black screen

Make sure production assets are referenced with relative paths in `webpack.config.js`:

```js
publicPath: isDev ? '/' : './'
```

The packaged app loads `dist/index.html` through Electron `loadFile()`. Absolute asset paths such as `/renderer.js` resolve incorrectly under `file://`.

### Routes do not work in the packaged app

The renderer must use `HashRouter` under `file://` and `BrowserRouter` in normal browser/dev-server mode.

### `electron-builder` fails with `spawn EPERM`

This is usually caused by Windows permissions, antivirus, or sandbox restrictions blocking `node_modules/app-builder-bin/win/x64/app-builder.exe`. Run the packaging command from a normal trusted terminal, or allow the binary in the security tool that blocked it.

### npm prints `Test-Path : Access is denied`

Some Windows npm installations can print this warning from `C:\Program Files\nodejs\npm.ps1`. If the command exits with code `0`, the project command still completed successfully.

### First setup is slow

The setup flow downloads a Python environment, CUDA PyTorch wheels, AI dependencies, and large base models. This can take a long time and requires significant disk space.

## Data and Model Storage

Large and generated artifacts must stay out of Git:

- `backend/env/`
- `backend/models/`
- `backend/output/`
- `backend/training_data/`
- `datasets/`
- `models/`
- `outputs/`
- `*.safetensors`
- `*.ckpt`
- `*.pt`
- `*.bin`

These paths are ignored to avoid committing local environments, model weights, datasets, and generated outputs.

## Security Notes

LoRA Studio is a local desktop tool. It starts a local HTTP API on `127.0.0.1:8000` and uses Electron IPC for setup, window controls, folder selection, and backend process management.

Do not expose the backend port to a public network. Do not install or run model files from untrusted sources.

## Release

Current release: `1.0.0-beta.1`

See `CHANGELOG.md` for version history.

## License

This project is distributed under the license in `LICENSE`.
