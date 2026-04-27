# 🎨 LoRA Studio — Beta 1.0.0

**LoRA Studio** is a professional, minimalist desktop dashboard for training and managing Stable Diffusion LoRA adapters. Built for speed, efficiency, and aesthetics.

## 🚀 Key Features

- **Advanced Training Support**: Train LoRA for **SD1.5**, **SDXL**, and the cutting-edge **Flux.1 (Dev)**.
- **Hardware Optimized**: Low-VRAM mode, gradient checkpointing, BF16 support, and automatic VRAM leak management.
- **Intelligent Dataset Prep**: Auto-captioning using BLIP/LLM models, tag management, and local directory synchronization.
- **Inference Playground**: Test your trained models instantly with a built-in playground featuring seed reuse, LoRA weight control, and metadata tracking.
- **Integrated Gallery**: Manage your models and generations in a unified interface. Switch between trained weights and creative results with ease.
- **Modern UI**: A premium, monochrome-themed dashboard designed for focus and productivity.

## 🛠 Tech Stack

- **Frontend**: React, TypeScript, Vite/Webpack, Lucide Icons.
- **Shell**: Electron (Cross-platform desktop integration).
- **Backend**: FastAPI (Python), PyTorch, Diffusers, PEFT, Accelerate.

## 🏁 Getting Started

### Prerequisites

- **Python 3.10+** (Recommend using `uv` or `conda`)
- **Node.js 18+**
- **NVIDIA GPU** with 8GB+ VRAM (Recommended)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/NIHILcoder/LoraTraining.git
   cd LoraTraining
   ```

2. **Setup Backend**:
   ```bash
   cd backend
   uv venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   uv pip install -r requirements.txt
   ```

3. **Setup Frontend**:
   ```bash
   cd ..
   npm install
   ```

4. **Launch Application**:
   ```bash
   npm run electron:dev
   ```

## 📦 Release Notes — 1.0.0-beta.1

- ✨ **Flux.1 Support**: Native support for training Flux.1 LoRAs with VRAM optimizations.
- 🖼 **Image Gallery**: New dedicated tab for managing playground generations with metadata preservation.
- 🛠 **Stability Patches**: Resolved "Meta Tensor" loading issues and VRAM memory leaks.
- 📁 **Dynamic Output**: Configurable save locations with native folder picker integration.

---

*Made with ❤️ for the Generative AI Community.*

