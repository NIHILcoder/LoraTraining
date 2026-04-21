<div align="center">

<br />

# LoRA Training Studio

**A professional desktop application for training, managing, and testing LoRA adapters.**  
Built on Electron · React · Python · HuggingFace Diffusers

<br />

[![Electron](https://img.shields.io/badge/Electron-30-47848F?style=flat-square&logo=electron&logoColor=white)](https://electronjs.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![CUDA](https://img.shields.io/badge/CUDA-required-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-6366f1?style=flat-square)](LICENSE)

<br />

</div>

---

LoRA Training Studio covers the complete fine-tuning workflow in a single application — from dataset preparation to model inference — without switching between separate tools. Training, monitoring, model management, and image generation are fully integrated.

<br />

## Overview

| Module | Description |
|---|---|
| **Training Workspace** | Configure and launch LoRA training with real-time loss charts and hardware monitoring |
| **Dataset Manager** | Import images, generate captions automatically via BLIP, edit per-image annotations |
| **Models Hub** | Download base models directly from HuggingFace, manage local storage |
| **Gallery** | Browse all trained LoRA adapters with metrics, open files, test in Playground |
| **Playground** | Run inference with any base model and LoRA combination, explore generation parameters |

<br />

## Features

### Training Engine

- Real GPU training powered by HuggingFace `diffusers` and `peft` — no stubs or simulations
- Automatic VRAM profiling: gradient checkpointing, latent caching, xformers, mixed precision
- Supports **SD 1.5 · SD 2.1 · SDXL · SD3 · Flux · Cascade · HunyuanDiT · PixArt · Kolors · AuraFlow**
- Real-time metrics streamed via WebSocket: loss curve, learning rate, ETA, elapsed time
- Graceful stop with instant UI reset — no zombie processes or stale state
- Training results saved as kohya-compatible `.safetensors` alongside full metadata JSON

### Dataset Manager

- Drag-and-drop or folder import with automatic thumbnail generation
- Local BLIP auto-captioning — no external API, no internet required
- Per-image caption editor with tag support
- Dataset statistics and image count tracking

### Gallery

- Scans the output directory and displays all trained adapters automatically
- Per-model metadata: final loss, average loss, total steps, LoRA rank, alpha, architecture, file size, creation date
- Sort by newest, name, best loss, or largest file
- Open containing folder in Explorer, send directly to Playground, or delete with confirmation

### Inference Playground

- Select any downloaded base model alongside any trained LoRA from Gallery
- Real `diffusers` pipeline with LoRA weight injection via `load_attn_procs`
- Pipeline caching in VRAM — subsequent generations load instantly
- Samplers: Euler a · DPM++ 2M Karras · DPM++ SDE Karras
- Resolution presets: 512² · 768² · 1024² · 768×1024 · 1024×768
- Generation metadata panel: seed, CFG scale, steps, sampler, LoRA weight, base model
- Copy seed, reuse seed, or restore all parameters from any history entry
- Scrollable generation history with per-entry seed labels
- Informative fallback response when GPU or model is unavailable

<br />

## Requirements

| Dependency | Version |
|---|---|
| Node.js | 18 or later |
| Python | 3.10 or later |
| NVIDIA GPU | 6 GB VRAM minimum (SD 1.5) · 8 GB (SDXL) |
| CUDA Toolkit | 11.8 or 12.x |

> The Gallery and Playground UI are fully functional without a GPU. Inference without a GPU returns a descriptive placeholder.

<br />

## Installation

**Clone the repository**

```bash
git clone https://github.com/NIHILcoder/LoraTraining.git
cd LoraTraining
```

**Install frontend dependencies**

```bash
npm install
```

**Start the Python backend**

```bash
.\start-backend.bat
```

The script creates a virtual environment, installs all Python dependencies from `requirements.txt`, and starts the FastAPI server on `http://localhost:8000`.

**Start the desktop application** *(separate terminal)*

```bash
npm run electron:dev
```

<br />

## Optional Dependencies

These packages improve performance but are not required to run the application.

```bash
# Memory-efficient attention — reduces VRAM usage by 20–30%
pip install xformers

# Adaptive optimizers — no learning rate tuning required
pip install prodigyopt dadaptation
```

<br />

## Tech Stack

**Frontend**

| Package | Role |
|---|---|
| Electron 30 | Desktop application shell, IPC, native window management |
| React 18 + TypeScript | Component-based UI with full type safety |
| React Router 6 | Client-side page navigation |
| Recharts | Real-time loss and metric charts |
| Webpack 5 | Module bundler for both renderer and main processes |

**Backend**

| Package | Role |
|---|---|
| FastAPI | Async REST API and WebSocket server |
| HuggingFace Diffusers ≥ 0.31 | Training and inference pipelines |
| PEFT ≥ 0.13 | LoRA adapter injection and extraction |
| Accelerate ≥ 0.33 | Distributed training utilities and mixed precision |
| Transformers | BLIP model for automatic image captioning |
| Safetensors | Efficient, safe model serialization |
| PyTorch + CUDA | GPU compute backend |

<br />

## Project Structure

```
LoraTraining/
├── src/
│   ├── components/
│   │   ├── layout/          # TitleBar, Sidebar, Header
│   │   ├── ui/              # Button, Card, Badge, Modal, ProgressBar
│   │   └── workspace/       # TrainingMonitor, DatasetSection, ConfigSection, HardwarePanel
│   ├── context/             # Global application state (AppContext)
│   ├── hooks/               # useWebSocket
│   ├── pages/               # TrainingWorkspace, Gallery, Playground, Models
│   ├── services/            # api.ts — all HTTP and WebSocket calls
│   ├── styles/              # Design tokens and global CSS
│   └── types/               # Shared TypeScript interfaces
│
├── backend/
│   ├── main.py              # FastAPI routes, WebSocket hub, inference engine
│   ├── trainer.py           # LoRA training loop, VRAM profiler, dataset loader
│   ├── requirements.txt
│   └── output/              # Trained LoRA models (auto-indexed by Gallery)
│
├── start-backend.bat        # Windows: venv setup + uvicorn start
├── package.json
└── webpack.config.js
```

<br />

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/gpu/info` | GPU info, VRAM tier, optimization profiles for all architectures |
| `POST` | `/api/training/start` | Begin training session, returns session ID |
| `POST` | `/api/training/stop/{id}` | Stop training, broadcast idle state via WebSocket |
| `WS` | `/ws/training` | Real-time training updates (loss, phase, logs) |
| `POST` | `/api/playground/generate` | Run inference, returns image as base64 data URL |
| `GET` | `/api/gallery/models` | Scan output directory, return all trained adapters |
| `DELETE` | `/api/gallery/models/{id}` | Delete model directory |
| `POST` | `/api/gallery/models/{id}/open` | Open folder in system file explorer |
| `GET` | `/api/models/base` | List base models with download status |
| `POST` | `/api/models/base/{id}/download` | Start HuggingFace download with progress streaming |
| `POST` | `/api/dataset/caption` | Auto-caption image via local BLIP model |

<br />

## Roadmap

- [ ] Aspect ratio bucketing (multi-resolution dataset support)
- [ ] Caption dropout for improved LoRA generalization
- [ ] Noise offset training parameter
- [ ] DoRA and LyCORIS training methods
- [ ] Img2Img in Playground
- [ ] LoRA strength grid testing (XYZ plot)
- [ ] Side-by-side model comparison in Gallery
- [ ] CIVITAI-compatible metadata export
- [ ] macOS and Linux support

<br />

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit using [Conventional Commits](https://www.conventionalcommits.org): `git commit -m 'feat: add bucket resolution'`
4. Push and open a Pull Request

Bug reports and feature requests are welcome via [GitHub Issues](https://github.com/NIHILcoder/LoraTraining/issues).

<br />

## License

[MIT](LICENSE) — free for personal and commercial use.
