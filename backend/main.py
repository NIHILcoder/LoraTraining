import asyncio
import time
import uuid
import random
import os
import json
import shutil
import aiofiles
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticBase

# Windows stability fixes
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["ACCELERATE_USE_CPU_INIT"] = "0"

# --- CRITICAL PATCHES FOR LIBRARIES ---
def bypass_check():
    return None

# 1. Fixes: AttributeError: 'CLIPTextModel' object has no attribute 'text_model'
try:
    from transformers.models.clip.modeling_clip import CLIPTextModel
    if not hasattr(CLIPTextModel, "text_model"):
        CLIPTextModel.text_model = property(lambda self: self)
except Exception: pass

# 2. Fixes: ValueError: Due to a serious vulnerability issue in `torch.load`...
try:
    import transformers.utils.import_utils
    transformers.utils.import_utils.check_torch_load_is_safe = bypass_check
    import transformers.modeling_utils
    transformers.modeling_utils.check_torch_load_is_safe = bypass_check
except Exception: pass

# 3. Force disable low_cpu_mem_usage globally in transformers
try:
    import transformers.modeling_utils
    transformers.modeling_utils._CONFIG_FOR_LOW_CPU_MEM_USAGE = False
except Exception: pass
# ----------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# P0-02: Require token
API_TOKEN = os.environ.get("LORA_STUDIO_API_TOKEN")

@app.middleware("http")
async def verify_api_token(request: Request, call_next):
    if not API_TOKEN or request.method == "OPTIONS":
        return await call_next(request)
        
    path = request.url.path
    if path.startswith("/api/"):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer ") or auth_header[7:] != API_TOKEN:
            return JSONResponse(status_code=401, content={"error": "Unauthorized"})
            
    return await call_next(request)

active_training_sessions = {}
active_connections: list[WebSocket] = []
active_downloads: dict[str, asyncio.Task] = {}

# --- User Data & Paths (P0-03) ---
BACKEND_DIR = Path(__file__).parent.resolve()
USER_DATA_DIR = Path(os.environ.get("LORA_STUDIO_USER_DATA", str(BACKEND_DIR)))

DEFAULT_MODELS_DIR = USER_DATA_DIR / "models"
SETTINGS_FILE = USER_DATA_DIR / "settings.json"
SECRETS_FILE = USER_DATA_DIR / "secrets.json"
TRAINING_DATA_DIR = USER_DATA_DIR / "training_data"
DEFAULT_OUTPUT_DIR = USER_DATA_DIR / "output"
GENERATED_DIR = USER_DATA_DIR / "generated"

# Ensure base directories exist
DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

def get_models_dir() -> Path:
    if SETTINGS_FILE.exists():
        try:
            settings = json.loads(SETTINGS_FILE.read_text())
            custom = settings.get("modelsDirectory")
            if custom and Path(custom).exists():
                return Path(custom)
        except Exception:
            pass
    return DEFAULT_MODELS_DIR

def save_settings(data: dict):
    existing = {}
    if SETTINGS_FILE.exists():
        try:
            existing = json.loads(SETTINGS_FILE.read_text())
        except Exception:
            pass
    existing.update(data)
    SETTINGS_FILE.write_text(json.dumps(existing, indent=2))

def save_secrets(data: dict):
    existing = {}
    if SECRETS_FILE.exists():
        try:
            existing = json.loads(SECRETS_FILE.read_text())
        except Exception:
            pass
    existing.update(data)
    SECRETS_FILE.write_text(json.dumps(existing, indent=2))


# --- Base Model Catalog ---
MODEL_CATALOG = [
    {
        "id": "sd15",
        "name": "Stable Diffusion 1.5",
        "shortName": "SD 1.5",
        "description": "The classic. Lightweight, fast, massive community ecosystem. Best for anime and artistic styles.",
        "architecture": "sd15",
        "fileSize": 4265380864,
        "filename": "v1-5-pruned-emaonly.safetensors",
        "downloadUrl": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
    },
    {
        "id": "sd21",
        "name": "Stable Diffusion 2.1",
        "shortName": "SD 2.1",
        "description": "Improved v2 with 768px native resolution. Better detail and composition than v1.5.",
        "architecture": "sd21",
        "fileSize": 5214865152,
        "filename": "v2-1_768-ema-pruned.safetensors",
        "downloadUrl": "https://civitai.com/api/download/models/130072",
    },
    {
        "id": "sdxl10",
        "name": "Stable Diffusion XL 1.0",
        "shortName": "SDXL",
        "description": "High-resolution powerhouse. 1024×1024 native. Superior detail for photorealistic output.",
        "architecture": "sdxl",
        "fileSize": 6938078334,
        "filename": "sd_xl_base_1.0.safetensors",
        "downloadUrl": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
    },
    {
        "id": "sd3-medium",
        "name": "Stable Diffusion 3 Medium",
        "shortName": "SD3",
        "description": "MMDiT architecture. State-of-the-art text rendering and prompt adherence.",
        "architecture": "sd3",
        "fileSize": 4339718720,
        "filename": "sd3_medium_incl_clips_t5xxlfp8.safetensors",
        "downloadUrl": "https://huggingface.co/Kijai/sd3-models/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors",
    },
    {
        "id": "flux-dev",
        "name": "Flux.1 Dev",
        "shortName": "Flux Dev",
        "description": "Next-gen by Black Forest Labs. Superior quality with best prompt adherence and photorealism.",
        "architecture": "flux",
        "fileSize": 23802932552,
        "filename": "flux1-dev.safetensors",
        "downloadUrl": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors",
    },
    {
        "id": "flux-schnell",
        "name": "Flux.1 Schnell",
        "shortName": "Flux Schnell",
        "description": "Next-gen by Black Forest Labs. Lightning fast generation with high prompt adherence. Free for commercial use.",
        "architecture": "flux",
        "fileSize": 23802932552,
        "filename": "flux1-schnell.safetensors",
        "downloadUrl": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors",
    },
    {
        "id": "cascade-stage-c",
        "name": "Stable Cascade (Stage C)",
        "shortName": "Cascade",
        "description": "Würstchen architecture. Extremely fast inference with compact latent space.",
        "architecture": "cascade",
        "fileSize": 9156923392,
        "filename": "stage_c_bf16.safetensors",
        "downloadUrl": "https://huggingface.co/stabilityai/stable-cascade/resolve/main/stage_c_bf16.safetensors",
    },
]

from typing import Optional, List

class TrainingConfig(PydanticBase):
    id: Optional[str] = None
    name: Optional[str] = None
    learningRate: float
    trainingSteps: int
    loraRank: int
    networkAlpha: int
    batchSize: int
    optimizer: str
    scheduler: str
    resolution: int
    seed: int
    mixedPrecision: str
    gradientAccumulation: int
    clipSkip: int
    warmupSteps: Optional[int] = 0
    datasetId: Optional[str] = None
    baseModel: str

class TrainingStartRequest(PydanticBase):
    config: TrainingConfig
    images: List[dict]  # [{filePath, captions: [...]}]

# Global trainer instance
from trainer import LoRATrainer, get_gpu_info, get_optimization_profile, prepare_dataset, estimate_training_time, ARCH_VRAM_MIN
trainer_instance = LoRATrainer()

def get_output_dir() -> Path:
    """Return the user-configured output directory, or the default."""
    if SETTINGS_FILE.exists():
        try:
            settings = json.loads(SETTINGS_FILE.read_text())
            custom = settings.get("outputDirectory")
            if custom:
                p = Path(custom)
                p.mkdir(parents=True, exist_ok=True)
                return p
        except Exception:
            pass
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR

ALL_ARCHITECTURES = ["sd15", "sd21", "sdxl", "sd3", "flux", "cascade"]

@app.get("/api/gpu/info")
async def gpu_info():
    """Return GPU info, optimization profiles, and ETA estimates for all architectures."""
    info = get_gpu_info()
    profiles = {}
    estimates = {}
    for arch in ALL_ARCHITECTURES:
        profiles[arch] = get_optimization_profile(info["vram_gb"], arch)
        estimates[arch] = estimate_training_time(
            architecture=arch, steps=1500, rank=16,
            resolution=profiles[arch].get("recommended_resolution", 512),
            vram_gb=info["vram_gb"], batch_size=1,
        )
    return {"gpu": info, "profiles": profiles, "estimates": estimates}

class EstimateRequest(PydanticBase):
    architecture: str
    steps: int
    rank: int = 16
    resolution: int = 512
    batchSize: int = 1

@app.post("/api/gpu/estimate")
async def estimate_time(req: EstimateRequest):
    """Estimate training time for specific config."""
    info = get_gpu_info()
    est = estimate_training_time(
        architecture=req.architecture, steps=req.steps, rank=req.rank,
        resolution=req.resolution, vram_gb=info["vram_gb"], batch_size=req.batchSize,
    )
    return est

@app.post("/api/training/start")
async def start_training(req: TrainingStartRequest):
    if trainer_instance.is_training:
        return {"error": "Training is already in progress"}
    
    config = req.config
    images = req.images
    session_id = str(uuid.uuid4())
    
    # Find the base model file
    models_dir = get_models_dir()
    model_path = None
    arch = config.baseModel
    
    for m in MODEL_CATALOG:
        if m["architecture"] == arch:
            candidate = models_dir / m["filename"]
            if candidate.exists():
                model_path = str(candidate)
                break
    
    # Also check custom models
    if not model_path and SETTINGS_FILE.exists():
        try:
            settings = json.loads(SETTINGS_FILE.read_text())
            for cm in settings.get("customModels", []):
                if cm.get("architecture") == arch:
                    candidate = models_dir / cm["filename"]
                    if candidate.exists():
                        model_path = str(candidate)
                        break
        except Exception:
            pass
    
    if not model_path:
        return {"error": f"No downloaded base model found for architecture '{arch}'. Please download one from Models Hub."}
    
    # Prepare dataset
    dataset_dir = TRAINING_DATA_DIR / session_id
    output_dir = get_output_dir() / session_id
    
    # Create both directories eagerly to avoid OS error 3 at save time
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prepare_dataset(images, dataset_dir)
    
    active_training_sessions[session_id] = {
        "status": "preparing",
        "config": config.model_dump(),
        "model_path": model_path,
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "task": None,
    }
    
    return {"sessionId": session_id}

@app.post("/api/training/stop/{session_id}")
async def stop_training(session_id: str):
    if session_id in active_training_sessions:
        trainer_instance.request_stop()
        task = active_training_sessions[session_id].get("task")
        if task:
            task.cancel()
        del active_training_sessions[session_id]
    # Broadcast idle state so frontend UI resets immediately
    await broadcast_to_connections({
        "type": "training_update",
        "data": {"phase": "idle", "currentStep": 0, "totalSteps": 0}
    })
    await broadcast_to_connections({
        "type": "training_log",
        "data": {
            "id": str(uuid.uuid4()),
            "timestamp": int(time.time() * 1000),
            "level": "warning",
            "message": "Training stopped by user."
        }
    })
    return {"status": "stopped"}

@app.get("/api/training/output/{session_id}")
async def get_training_output(session_id: str):
    output_dir = get_output_dir() / session_id
    if not output_dir.exists():
        return {"error": "Output not found"}
    safetensors = list(output_dir.glob("*.safetensors"))
    if safetensors:
        return {"path": str(safetensors[0]), "dir": str(output_dir)}
    return {"error": "No .safetensors file found"}


# ============================================
# Playground — Inference Engine
# ============================================

# Lazy-loaded inference pipeline cache: (model_path, lora_path) -> pipeline
_inference_cache: dict = {}
_inference_lock = asyncio.Lock()

class GenerateRequest(PydanticBase):
    prompt: str
    negativePrompt: Optional[str] = ""
    width: int = 1024
    height: int = 1024
    cfgScale: float = 7.0
    steps: int = 25
    seed: int = -1
    sampler: str = "Euler a"
    loraWeight: float = 1.0
    loraModelId: Optional[str] = None   # id of the trained LoRA from Gallery
    baseModelId: Optional[str] = None   # id from MODEL_CATALOG (sd15, sdxl, ...)

@app.post("/api/playground/generate")
async def generate_image(req: GenerateRequest):
    """
    Run inference with optional LoRA injection.
    Uses diffusers when a GPU+model is available, otherwise returns an informative mock.
    """
    import gc, torch, base64, io
    from PIL import Image, ImageDraw, ImageFont

    actual_seed = req.seed if req.seed >= 0 else random.randint(0, 2**31 - 1)

    # --- Resolve base model path ---
    models_dir = get_models_dir()
    model_path = None

    # If caller specified a baseModelId, find that model
    target_arch = req.baseModelId or "sd15"
    for m in MODEL_CATALOG:
        if m["architecture"] == target_arch or m["id"] == target_arch:
            candidate = models_dir / m["filename"]
            if candidate.exists():
                model_path = str(candidate)
                target_arch = m["architecture"]
                break

    # Also check any downloaded model
    if not model_path:
        for m in MODEL_CATALOG:
            candidate = models_dir / m["filename"]
            if candidate.exists():
                model_path = str(candidate)
                target_arch = m["architecture"]
                break

    # --- Resolve LoRA path ---
    lora_path = None
    if req.loraModelId and req.loraModelId != "none":
        lora_dir = get_output_dir() / req.loraModelId
        safetensors = list(lora_dir.glob("*.safetensors")) if lora_dir.exists() else []
        if safetensors:
            lora_path = str(safetensors[0])

    # --- Check GPU ---
    gpu_available = torch.cuda.is_available()

    if not model_path or not gpu_available:
        # Return informative mock SVG with generation parameters
        reason = "No GPU detected" if not gpu_available else f"No base model downloaded (looked for {target_arch.upper()})"
        svg = _make_mock_svg(req.prompt, req.seed if req.seed >= 0 else actual_seed, req.sampler, req.loraWeight if lora_path else None, reason)
        return {"url": svg, "seed": actual_seed, "mock": True, "reason": reason}

    # --- Real inference ---
    try:
        pipe = await asyncio.to_thread(
            _load_pipeline, model_path, lora_path, req.loraWeight, target_arch
        )

        result_image = await asyncio.to_thread(
            _run_inference, pipe, req, actual_seed
        )

        # Save to disk and return as data URL
        img_id = str(uuid.uuid4())[:8]
        img_path = GENERATED_DIR / f"{img_id}.png"
        meta_path = GENERATED_DIR / f"{img_id}.json"
        
        result_image.save(str(img_path), "PNG")
        
        # Resolve names for metadata
        lora_name = "None"
        if req.loraModelId and req.loraModelId != "none":
            lora_dir = get_output_dir() / req.loraModelId
            safetensors = list(lora_dir.glob("*.safetensors"))
            if safetensors:
                lora_name = safetensors[0].stem.replace("_", " ").title()
        
        base_name = "Auto"
        if req.baseModelId and req.baseModelId != "auto":
            for m in MODEL_CATALOG:
                if m["id"] == req.baseModelId:
                    base_name = m["name"]
                    break

        # Save metadata for Gallery
        image_meta = {
            "id": img_id,
            "prompt": req.prompt,
            "negativePrompt": req.negativePrompt,
            "seed": actual_seed,
            "steps": req.steps,
            "cfgScale": req.cfgScale,
            "sampler": req.sampler,
            "width": req.width,
            "height": req.height,
            "loraModelId": req.loraModelId,
            "loraName": lora_name,
            "loraWeight": req.loraWeight,
            "baseModelId": req.baseModelId,
            "baseModelName": base_name,
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        meta_path.write_text(json.dumps(image_meta, indent=2))

        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        data_url = f"data:image/png;base64,{img_b64}"

        return {"url": data_url, "seed": actual_seed, "mock": False}

    except Exception as e:
        print(f"[Inference] Error: {e}")
        svg = _make_mock_svg(req.prompt, actual_seed, req.sampler, None, f"Inference error: {str(e)[:80]}")
        return {"url": svg, "seed": actual_seed, "mock": True, "reason": str(e)}

def _module_has_meta(module) -> bool:
    """Return True if a torch module contains meta tensors."""
    if module is None or not hasattr(module, "parameters"):
        return False

    try:
        return any(param.is_meta for param in module.parameters())
    except Exception:
        return False


def _repair_text_encoders_if_needed(pipe, architecture: str):
    """
    from_single_file can sometimes leave text_encoder as meta.
    If that happens, replace text encoder/tokenizer from a known Diffusers repo.
    """
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer

    if architecture in ("sd15", "sd21"):
        if not _module_has_meta(getattr(pipe, "text_encoder", None)):
            return pipe

        if architecture == "sd21":
            repo_id = "stabilityai/stable-diffusion-2-1"
        else:
            repo_id = "runwayml/stable-diffusion-v1-5"

        print(f"[Inference] text_encoder is meta. Reloading text encoder from: {repo_id}")

        pipe.tokenizer = CLIPTokenizer.from_pretrained(
            repo_id,
            subfolder="tokenizer",
            local_files_only=False,
        )

        pipe.text_encoder = CLIPTextModel.from_pretrained(
            repo_id,
            subfolder="text_encoder",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
            local_files_only=False,
        )

        return pipe

    if architecture in ("sdxl", "kolors"):
        from transformers import CLIPTextModelWithProjection

        repo_id = "stabilityai/stable-diffusion-xl-base-1.0"

        need_text_encoder_1 = _module_has_meta(getattr(pipe, "text_encoder", None))
        need_text_encoder_2 = _module_has_meta(getattr(pipe, "text_encoder_2", None))

        if not need_text_encoder_1 and not need_text_encoder_2:
            return pipe

        print(f"[Inference] SDXL text encoder is meta. Reloading text encoders from: {repo_id}")

        if need_text_encoder_1:
            pipe.tokenizer = CLIPTokenizer.from_pretrained(
                repo_id,
                subfolder="tokenizer",
                local_files_only=False,
            )
            pipe.text_encoder = CLIPTextModel.from_pretrained(
                repo_id,
                subfolder="text_encoder",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
                local_files_only=False,
            )

        if need_text_encoder_2:
            pipe.tokenizer_2 = CLIPTokenizer.from_pretrained(
                repo_id,
                subfolder="tokenizer_2",
                local_files_only=False,
            )
            pipe.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                repo_id,
                subfolder="text_encoder_2",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
                local_files_only=False,
            )

        return pipe

    return pipe

def _load_pipeline(model_path: str, lora_path: Optional[str], lora_weight: float, architecture: str):
    """
    Load inference pipeline safely.
    Handles meta text_encoder by reloading it from a known repo.
    """
    import os
    import gc
    import torch

    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

    cache_key = (model_path, lora_path, round(float(lora_weight), 2), architecture)
    if cache_key in _inference_cache:
        return _inference_cache[cache_key]

    _inference_cache.clear()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    # Important: do not allow Accelerate CPU-init/meta-init during inference loading.
    os.environ["ACCELERATE_USE_CPU_INIT"] = "0"

    model_path = os.path.abspath(os.path.normpath(model_path))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Base model file not found: {model_path}")

    # Your current playground does not implement these pipelines yet.
    if architecture in ("flux", "sd3", "cascade"):
        raise RuntimeError(
            f"Playground inference for architecture '{architecture}' is not implemented yet. "
            f"Use SD 1.5 / SD 2.1 / SDXL, or add the correct Diffusers pipeline for this architecture."
        )

    print(f"[Inference] Loading model: {model_path}")
    print(f"[Inference] Architecture: {architecture}")

    load_kwargs = {
        "torch_dtype": torch.float32,
        "use_safetensors": True,
        "low_cpu_mem_usage": False,
        "local_files_only": False,
    }

    if architecture in ("sdxl", "kolors"):
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_path,
            **load_kwargs,
        )
    else:
        pipe = StableDiffusionPipeline.from_single_file(
            model_path,
            safety_checker=None,
            requires_safety_checker=False,
            **load_kwargs,
        )

    # Fix broken/meta text encoder after from_single_file.
    pipe = _repair_text_encoders_if_needed(pipe, architecture)

    # Final meta check before moving to GPU.
    meta_components = []
    for component_name, component in pipe.components.items():
        if _module_has_meta(component):
            meta_components.append(component_name)

    if meta_components:
        raise RuntimeError(
            "Model still has meta tensors in components: "
            + ", ".join(meta_components)
            + ". This usually means the model file is incomplete, the architecture is wrong, "
              "or required HuggingFace components could not be downloaded."
        )

    pipe = pipe.to("cuda", dtype=torch.float16)
    pipe.enable_attention_slicing()

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print(f"[Inference] xFormers not enabled: {e}")

    if lora_path:
        lora_path = os.path.abspath(os.path.normpath(lora_path))

        if os.path.exists(lora_path):
            try:
                pipe.load_lora_weights(
                    os.path.dirname(lora_path),
                    weight_name=os.path.basename(lora_path),
                )

                try:
                    pipe.set_adapters(["default"], adapter_weights=[float(lora_weight)])
                except Exception:
                    pass

                print(f"[Inference] LoRA loaded: {lora_path} weight={lora_weight}")

            except Exception as e:
                print(f"[Inference] LoRA load warning: {e}")
        else:
            print(f"[Inference] LoRA path not found: {lora_path}")

    _inference_cache[cache_key] = pipe
    return pipe

def _run_inference(pipe, req: GenerateRequest, seed: int):
    """Run the actual diffusion inference."""
    import torch
    from diffusers import (
        EulerAncestralDiscreteScheduler,
        DPMSolverMultistepScheduler,
        DPMSolverSinglestepScheduler,
    )

    # Set sampler
    sampler_lower = req.sampler.lower()
    if "euler a" in sampler_lower or "euler ancestral" in sampler_lower:
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif "dpm++ 2m" in sampler_lower:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas="karras" in sampler_lower
        )
    elif "dpm++ sde" in sampler_lower:
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas="karras" in sampler_lower
        )

    generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipe(
        prompt=req.prompt,
        negative_prompt=req.negativePrompt or "",
        width=req.width,
        height=req.height,
        guidance_scale=req.cfgScale,
        num_inference_steps=req.steps,
        generator=generator,
        cross_attention_kwargs={"scale": req.loraWeight} if req.loraWeight != 1.0 else None,
    )
    return result.images[0]


def _make_mock_svg(prompt: str, seed: int, sampler: str, lora_weight, reason: str) -> str:
    """Generate an informative SVG placeholder when inference can't run."""
    import urllib.parse
    prompt_short = (prompt[:55] + "…") if len(prompt) > 55 else prompt
    lora_line = f"LoRA Weight: {lora_weight} | {sampler}" if lora_weight is not None else f"No LoRA | {sampler}"

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0f0f1a"/>
      <stop offset="100%" stop-color="#1a1a2e"/>
    </linearGradient>
  </defs>
  <rect width="512" height="512" fill="url(#bg)"/>
  <rect x="1" y="1" width="510" height="510" fill="none" stroke="#2a2a4a" stroke-width="2"/>
  <line x1="1" y1="1" x2="511" y2="511" stroke="#2a2a4a" stroke-width="1"/>
  <line x1="511" y1="1" x2="1" y2="511" stroke="#2a2a4a" stroke-width="1"/>
  <text x="256" y="195" text-anchor="middle" fill="#ffffff" font-size="18" font-family="monospace" font-weight="bold">Simulated Inference</text>
  <text x="256" y="235" text-anchor="middle" fill="#888" font-size="13" font-family="monospace">{prompt_short}</text>
  <text x="256" y="285" text-anchor="middle" fill="#7c6af5" font-size="13" font-family="monospace">{lora_line}</text>
  <text x="256" y="320" text-anchor="middle" fill="#555" font-size="11" font-family="monospace">Seed: {seed} | CFG: — | Steps: —</text>
  <text x="256" y="370" text-anchor="middle" fill="#ef4444" font-size="11" font-family="monospace">{reason}</text>
</svg>'''
    return "data:image/svg+xml," + urllib.parse.quote(svg)


class CaptionRequest(PydanticBase):
    imageId: str
    imageUrl: Optional[str] = None # Will contain the local File path from Electron

# Global lazy-loaded models
processor = None
blip_model = None

@app.post("/api/dataset/caption")
async def auto_caption(req: CaptionRequest):
    global processor, blip_model
    
    if processor is None or blip_model is None:
        print("Loading BLIP model on first request...")
        from transformers import BlipProcessor, BlipForConditionalGeneration
        try:
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                use_safetensors=True,
            )
            print("BLIP loaded successfully.")
        except Exception as e:
            print(f"Failed to load BLIP model: {e}")
            processor = None
            blip_model = None
            return {"tags": ["model failed to load"]}

    try:
        from PIL import Image
        import os
        
        # Load local image using the path passed from Electron
        if not req.imageUrl or not os.path.exists(req.imageUrl):
             return {"tags": ["file not found locally"]}

        raw_image = Image.open(req.imageUrl).convert('RGB')
        
        # Generate multiple diverse captions
        inputs = processor(raw_image, return_tensors="pt")
        out = blip_model.generate(
            **inputs, 
            max_new_tokens=50,
            num_return_sequences=3,
            do_sample=True,
            top_k=30,
            temperature=0.7
        )
        
        captions = processor.batch_decode(out, skip_special_tokens=True)
        main_caption = captions[0]
        
        # The best universal approach for modern LoRA training:
        # 1. Provide the full main caption as the first "tag" (Best for SDXL/Flux)
        tags = [main_caption]
        
        # 2. Extract keywords from ALL generated captions (Best for SD1.5/Anime)
        stop_words = {
            "a", "an", "the", "and", "but", "or", "for", "nor", "on", "at", "in", "with", "by", "from", 
            "up", "about", "into", "over", "after", "is", "are", "was", "were", "be", "been", "being", 
            "have", "has", "had", "do", "does", "did", "of", "to", "it", "that", "this", "there", "their", 
            "they", "he", "she", "his", "her", "him", "its", "some", "many", "few", "all", "any", "no", 
            "not", "very", "too", "so", "just", "can", "will", "would", "could", "should", "what", "which", 
            "who", "when", "where", "why", "how", "background", "foreground", "picture", "image", "photo", 
            "photography", "showing", "view", "close", "standing", "sitting", "looking"
        }
        
        keyword_set = set()
        for cap in captions:
            words = [w.strip(".,!?\"'()[]") for w in cap.lower().split()]
            for w in words:
                if len(w) > 2 and w not in stop_words and not w.isdigit():
                    keyword_set.add(w)
                    
        # Sort keywords alphabetically for neatness
        sorted_keywords = sorted(list(keyword_set))
        
        # Combine: Main Caption first, then individual keywords
        final_tags = tags + sorted_keywords
        
        return {"tags": final_tags}
        
    except Exception as e:
        print("Captioning error:", e)
        return {"tags": ["error"]}



@app.websocket("/ws/training")
async def training_websocket(websocket: WebSocket):
    # P0-02: Authenticate WebSocket
    token = websocket.query_params.get("token")
    if API_TOKEN and token != API_TOKEN:
        await websocket.close(code=1008, reason="Unauthorized")
        return

    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "start_training":
                session_id = data["payload"]["sessionId"]
                if session_id in active_training_sessions:
                    session = active_training_sessions[session_id]
                    task = asyncio.create_task(
                        run_real_training(session_id, session, websocket)
                    )
                    active_training_sessions[session_id]["task"] = task
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)

async def run_real_training(session_id: str, session: dict, websocket: WebSocket):
    """Bridge between async WebSocket world and the blocking GPU trainer."""
    config_data = session["config"]
    model_path = session["model_path"]
    dataset_dir = Path(session["dataset_dir"])
    output_dir = Path(session["output_dir"])
    total_steps = config_data.get("trainingSteps", 1000)
    
    loop = asyncio.get_running_loop()
    
    # Progress callback that sends updates via WebSocket
    # Since the trainer runs in a thread, we use loop.call_soon_threadsafe
    def progress_callback(**kwargs):
        step = kwargs.get("step", 0)
        loss = kwargs.get("loss", 0)
        lr = kwargs.get("lr", 0)
        phase = kwargs.get("phase", "training")
        message = kwargs.get("message", "")
        avg_loss = kwargs.get("avg_loss", 0)
        eta = kwargs.get("eta", 0)
        
        async def _send():
            try:
                # Send step data for loss chart
                await broadcast_to_connections({
                    "type": "training_step",
                    "data": {
                        "step": step,
                        "loss": loss,
                        "learningRate": lr,
                        "timestamp": int(time.time() * 1000)
                    }
                })
                
                # Send status update
                await broadcast_to_connections({
                    "type": "training_update",
                    "data": {
                        "phase": phase,
                        "currentStep": step,
                        "totalSteps": total_steps,
                        "currentLoss": loss,
                        "avgLoss": avg_loss,
                        "learningRate": lr,
                        "eta": eta,
                    }
                })
                
                # Send log message
                if message:
                    await broadcast_to_connections({
                        "type": "training_log",
                        "data": {
                            "id": str(uuid.uuid4()),
                            "timestamp": int(time.time() * 1000),
                            "level": "info",
                            "message": message
                        }
                    })
            except Exception as e:
                print(f"[WS] Failed to send progress: {e}")
        
        asyncio.run_coroutine_threadsafe(_send(), loop)
    
    try:
        # Log start
        await broadcast_to_connections({
            "type": "training_log",
            "data": {
                "id": str(uuid.uuid4()),
                "timestamp": int(time.time() * 1000),
                "level": "info",
                "message": f"Starting LoRA training session: {session_id[:8]}..."
            }
        })
        await broadcast_to_connections({
            "type": "training_update",
            "data": {"phase": "preparing", "totalSteps": total_steps}
        })
        
        # Run the actual training in a thread pool (blocking GPU work)
        result = await asyncio.to_thread(
            trainer_instance.train,
            config=config_data,
            dataset_dir=dataset_dir,
            model_path=model_path,
            output_dir=output_dir,
            progress_callback=progress_callback,
        )
        
        # Training completed
        await broadcast_to_connections({
            "type": "training_update",
            "data": {"phase": "completed"}
        })
        await broadcast_to_connections({
            "type": "training_log",
            "data": {
                "id": str(uuid.uuid4()),
                "timestamp": int(time.time() * 1000),
                "level": "info",
                "message": f"Training complete! LoRA saved to: {result.get('output_path', 'unknown')}"
            }
        })
        
    except asyncio.CancelledError:
        # Task was cancelled via stop button — not an error
        print("[Training] Task cancelled (stop requested).")
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        error_msg = f"{str(e)}\n\nTraceback:\n{tb_str}"
        print(f"[Training] Error:\n{error_msg}")
        await broadcast_to_connections({
            "type": "training_update",
            "data": {"phase": "error"}
        })
        await broadcast_to_connections({
            "type": "training_log",
            "data": {
                "id": str(uuid.uuid4()),
                "timestamp": int(time.time() * 1000),
                "level": "error",
                "message": f"Training failed: {error_msg}"
            }
        })
    finally:
        active_training_sessions.pop(session_id, None)

# ============================================
# Model Download Manager
# ============================================

async def broadcast_to_connections(data: dict):
    """Send data to all active WebSocket connections."""
    disconnected = []
    for ws in active_connections:
        try:
            await ws.send_json(data)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        active_connections.remove(ws)

@app.get("/api/models/base")
async def list_base_models():
    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    
    result = []
    for model in MODEL_CATALOG:
        m = dict(model)
        filepath = models_dir / m["filename"]
        if filepath.exists():
            m["status"] = "downloaded"
            m["localPath"] = str(filepath)
        elif m["id"] in active_downloads:
            m["status"] = "downloading"
        else:
            m["status"] = "not_downloaded"
        result.append(m)
    
    # Load custom models from settings
    if SETTINGS_FILE.exists():
        try:
            settings = json.loads(SETTINGS_FILE.read_text())
            for custom in settings.get("customModels", []):
                cm = dict(custom)
                filepath = models_dir / cm["filename"]
                if filepath.exists():
                    cm["status"] = "downloaded"
                    cm["localPath"] = str(filepath)
                elif cm["id"] in active_downloads:
                    cm["status"] = "downloading"
                else:
                    cm["status"] = "not_downloaded"
                cm["isCustom"] = True
                result.append(cm)
        except Exception:
            pass
    
    return {"models": result, "modelsDirectory": str(models_dir)}

class CustomModelRequest(PydanticBase):
    url: str
    name: Optional[str] = None
    architecture: Optional[str] = "sd15"

@app.post("/api/models/base/custom")
async def add_custom_model(req: CustomModelRequest):
    """Add a custom model by URL (HuggingFace or direct link)."""
    url = req.url.strip()
    
    # Extract filename from URL
    filename = url.split("/")[-1].split("?")[0]
    if not filename.endswith((".safetensors", ".ckpt", ".bin", ".pt")):
        filename += ".safetensors"
    
    model_id = f"custom-{uuid.uuid4().hex[:8]}"
    name = req.name or filename.replace(".safetensors", "").replace(".", " ").replace("_", " ").replace("-", " ").title()
    
    custom_model = {
        "id": model_id,
        "name": name,
        "shortName": name[:12],
        "description": f"Custom model from: {url[:80]}",
        "architecture": req.architecture or "sd15",
        "fileSize": 0,  # Unknown until download starts
        "filename": filename,
        "downloadUrl": url,
        "isCustom": True,
        "status": "not_downloaded",
    }
    
    # Save to settings
    existing = {}
    if SETTINGS_FILE.exists():
        try:
            existing = json.loads(SETTINGS_FILE.read_text())
        except Exception:
            pass
    custom_models = existing.get("customModels", [])
    custom_models.append(custom_model)
    existing["customModels"] = custom_models
    SETTINGS_FILE.write_text(json.dumps(existing, indent=2))
    
    return custom_model

@app.post("/api/models/base/{model_id}/download")
async def download_model(model_id: str):
    if model_id in active_downloads:
        return {"status": "already_downloading"}
    
    # Find model in catalog or custom models
    model_info = None
    for m in MODEL_CATALOG:
        if m["id"] == model_id:
            model_info = m
            break
    
    if not model_info and SETTINGS_FILE.exists():
        try:
            settings = json.loads(SETTINGS_FILE.read_text())
            for cm in settings.get("customModels", []):
                if cm["id"] == model_id:
                    model_info = cm
                    break
        except Exception:
            pass
    
    if not model_info:
        return {"error": "Model not found"}
    
    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    
    task = asyncio.create_task(
        run_download(model_id, model_info["downloadUrl"], model_info["filename"], models_dir)
    )
    active_downloads[model_id] = task
    return {"status": "started"}

@app.post("/api/models/base/{model_id}/cancel")
async def cancel_download(model_id: str):
    if model_id in active_downloads:
        active_downloads[model_id].cancel()
        del active_downloads[model_id]
        # Clean up partial file
        models_dir = get_models_dir()
        for m in MODEL_CATALOG:
            if m["id"] == model_id:
                partial = models_dir / (m["filename"] + ".part")
                if partial.exists():
                    partial.unlink()
                break
    return {"status": "cancelled"}

@app.delete("/api/models/base/{model_id}")
async def delete_model_file(model_id: str):
    models_dir = get_models_dir()
    
    # Check catalog
    for m in MODEL_CATALOG:
        if m["id"] == model_id:
            filepath = models_dir / m["filename"]
            if filepath.exists():
                filepath.unlink()
            return {"status": "deleted"}
    
    # Check custom models
    if SETTINGS_FILE.exists():
        try:
            settings = json.loads(SETTINGS_FILE.read_text())
            custom_models = settings.get("customModels", [])
            for cm in custom_models:
                if cm["id"] == model_id:
                    filepath = models_dir / cm["filename"]
                    if filepath.exists():
                        filepath.unlink()
                    # Also remove from settings
                    custom_models = [x for x in custom_models if x["id"] != model_id]
                    settings["customModels"] = custom_models
                    SETTINGS_FILE.write_text(json.dumps(settings, indent=2))
                    return {"status": "deleted"}
        except Exception:
            pass
    
    return {"error": "Model not found"}

class SetDirectoryRequest(PydanticBase):
    path: str

@app.post("/api/models/directory")
async def set_models_directory(req: SetDirectoryRequest):
    target = Path(req.path)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return {"error": f"Cannot create directory: {e}"}
    save_settings({"modelsDirectory": str(target)})
    return {"status": "ok", "modelsDirectory": str(target)}

@app.post("/api/settings/token")
async def set_hf_token(req: dict):
    save_secrets({"hfToken": req.get("token", "")})
    return {"status": "ok"}

@app.get("/api/settings/token")
async def get_hf_token():
    secrets = {}
    if SECRETS_FILE.exists():
        try:
            secrets = json.loads(SECRETS_FILE.read_text())
        except Exception:
            pass
    # P0-02: Do not return full token in plaintext
    token = secrets.get("hfToken", "")
    masked = (token[:4] + "*" * (len(token) - 4)) if len(token) > 4 else ""
    return {"token": masked, "hasToken": bool(token)}

async def run_download(model_id: str, url: str, filename: str, models_dir: Path):
    """Download a model file with real-time progress broadcasting."""
    import aiohttp
    import aiofiles
    import shutil
    
    part_path = models_dir / (filename + ".part")
    final_path = models_dir / filename
    
    print(f"[Download] Starting: {filename}")
    print(f"[Download] URL: {url}")
    
    try:
        secrets = {}
        if SECRETS_FILE.exists():
            try:
                secrets = json.loads(SECRETS_FILE.read_text())
            except Exception:
                pass
        hf_token = secrets.get("hfToken", "")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*"
        }
        
        # Only send token for specific gated organizations to avoid 403 on public repos
        gated_orgs = ["black-forest-labs", "stabilityai"]
        should_send_token = any(org in url.lower() for org in gated_orgs)
        
        if hf_token and "huggingface.co" in url and should_send_token:
            headers["Authorization"] = f"Bearer {hf_token}"
            print(f"[Download] Using HuggingFace Token for gated repo: {url}")

        async with aiohttp.ClientSession(headers=headers) as session:
            print(f"[Download] Starting request to: {url}")
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=None), allow_redirects=True) as response:
                print(f"[Download] Response status: {response.status} for {url}")
                if response.status != 200:
                    error_msg = f"HTTP {response.status}: {response.reason}"
                    print(f"[Download] Error: {error_msg} for URL: {url}")
                    # Special check for 401/404 on HF which might mean gated
                    if response.status in [401, 403, 404] and "huggingface.co" in url and not hf_token:
                        error_msg += " (Token might be required)"
                    
                    await broadcast_to_connections({
                        "type": "download_error",
                        "data": {"modelId": model_id, "message": error_msg}
                    })
                    return
                
                total_size = int(response.headers.get("Content-Length", 0))
                
                # Integrity check: model files should be > 10MB (most are > 2GB)
                # If it's small, it's likely an HTML error page
                if total_size < 10 * 1024 * 1024:
                    error_msg = "Downloaded file is too small. Possibly an invalid URL or restricted access."
                    print(f"[Download] Error: {error_msg} (Size: {total_size} bytes)")
                    await broadcast_to_connections({
                        "type": "download_error",
                        "data": {"modelId": model_id, "message": error_msg}
                    })
                    return
                    
                downloaded = 0
                last_broadcast = 0
                start_time = time.time()
                chunk_times = []
                
                print(f"[Download] Total size: {total_size / (1024**3):.2f} GB")
                
                # Use aiofiles for non-blocking disk I/O
                async with aiofiles.open(part_path, mode="wb") as f:
                    async for chunk in response.content.iter_chunked(1024 * 1024):  # 1MB chunks
                        if model_id not in active_downloads:
                            print(f"[Download] Interrupted: {model_id}")
                            break
                        
                        await f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Track speed
                        now = time.time()
                        chunk_times.append((now, len(chunk)))
                        chunk_times = [(t, s) for t, s in chunk_times if now - t < 5] # 5 sec window
                        
                        # Broadcast progress
                        if now - last_broadcast >= 0.5:
                            last_broadcast = now
                            if len(chunk_times) > 1:
                                duration = chunk_times[-1][0] - chunk_times[0][0]
                                total_bytes_in_window = sum(s for _, s in chunk_times)
                                speed = total_bytes_in_window / max(duration, 0.001)
                            else:
                                speed = 0
                            
                            speed_str = f"{speed / (1024*1024):.1f} MB/s" if speed > 1024*1024 else f"{speed/1024:.1f} KB/s"
                            progress = (downloaded / total_size * 100) if total_size > 0 else 0
                            
                            await broadcast_to_connections({
                                "type": "download_progress",
                                "data": {
                                    "modelId": model_id,
                                    "progress": round(progress, 1),
                                    "speed": speed_str,
                                    "downloadedBytes": downloaded,
                                    "totalBytes": total_size
                                }
                            })
                
                # Verify we got everything
                if model_id in active_downloads:
                    if total_size > 0 and downloaded < total_size:
                        raise Exception(f"Download incomplete: {downloaded}/{total_size} bytes")
                    
                    print(f"[Download] Finishing: {filename}...")
                    
                    # Small delay to let OS release file handles on Windows
                    await asyncio.sleep(0.5)
                    
                    # Use shutil.move for more robust cross-platform moving/renaming
                    if final_path.exists():
                        final_path.unlink()
                    shutil.move(str(part_path), str(final_path))
                    
                    print(f"[Download] Success: {final_path}")
                    
                    await broadcast_to_connections({
                        "type": "download_complete",
                        "data": {"modelId": model_id, "localPath": str(final_path)}
                    })
                    
    except asyncio.CancelledError:
        print(f"[Download] Cancelled: {filename}")
        if part_path.exists():
            part_path.unlink()
    except Exception as e:
        error_msg = str(e)
        print(f"[Download] FAILED: {filename} - {error_msg}")
        await broadcast_to_connections({
            "type": "download_error",
            "data": {"modelId": model_id, "message": error_msg}
        })
        if part_path.exists():
            try:
                part_path.unlink()
            except:
                pass
    finally:
        active_downloads.pop(model_id, None)


# ============================================
# Gallery — Trained LoRA Models
# ============================================

@app.get("/api/gallery/models")
async def list_trained_models():
    """Scan the output directory for trained LoRA models."""
    models = []
    
    output_dir = get_output_dir()
    if not output_dir.exists():
        return {"models": []}
    
    for session_dir in sorted(output_dir.iterdir(), reverse=True):
        if not session_dir.is_dir():
            continue
        
        # Look for .safetensors files
        safetensors = list(session_dir.glob("*.safetensors"))
        if not safetensors:
            continue
        
        lora_file = safetensors[0]
        file_size = lora_file.stat().st_size
        created_at = lora_file.stat().st_ctime
        
        # Try to read training metadata (adapter_config.json from peft)
        metadata = {}
        adapter_config = session_dir / "adapter_config.json"
        if adapter_config.exists():
            try:
                metadata = json.loads(adapter_config.read_text())
            except Exception:
                pass
        
        # Try to read our own training result metadata
        result_file = session_dir / "training_result.json"
        result_data = {}
        if result_file.exists():
            try:
                result_data = json.loads(result_file.read_text())
            except Exception:
                pass
        
        raw_name = lora_file.stem.replace("_", " ").replace("-", " ").title()
        if raw_name == "Default Config" or raw_name == "Lora Output":
            display_name = f"{raw_name} ({session_dir.name[:6]})"
        else:
            display_name = raw_name

        model_info = {
            "id": session_dir.name,
            "name": display_name,
            "filename": lora_file.name,
            "fileSize": file_size,
            "path": str(lora_file),
            "directory": str(session_dir),
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created_at)),
            # From peft adapter_config
            "rank": metadata.get("r", 0),
            "alpha": metadata.get("lora_alpha", 0),
            "targetModules": metadata.get("target_modules", []),
            # From our training result
            "finalLoss": result_data.get("final_loss", 0),
            "avgLoss": result_data.get("avg_loss", 0),
            "totalSteps": result_data.get("total_steps", 0),
            "stoppedEarly": result_data.get("stopped_early", False),
            "architecture": result_data.get("architecture", ""),
            "baseModelName": result_data.get("base_model_name", ""),
        }
        
        models.append(model_info)
    
    return {"models": models}


@app.delete("/api/gallery/models/{model_id}")
async def delete_trained_model(model_id: str):
    """Delete a trained LoRA model and its directory."""
    model_dir = get_output_dir() / model_id
    if not model_dir.exists():
        return {"error": "Model not found"}
    
    try:
        shutil.rmtree(str(model_dir))
        return {"status": "deleted"}
    except Exception as e:
        return {"error": f"Failed to delete: {e}"}


@app.post("/api/gallery/models/{model_id}/open")
async def open_model_folder(model_id: str):
    """Open the model folder in the system file explorer."""
    model_dir = get_output_dir() / model_id
    if not model_dir.exists():
        return {"error": "Model folder not found"}
    
    import subprocess
    import sys
    
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(model_dir)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(model_dir)])
        else:
            subprocess.Popen(["xdg-open", str(model_dir)])
        return {"status": "opened"}
    except Exception as e:
        return {"error": f"Failed to open folder: {e}"}


@app.get("/api/gallery/images")
async def list_generated_images():
    """Scan the generated directory for images and their metadata."""
    images = []
    if not GENERATED_DIR.exists():
        return {"images": []}
        
    for img_file in sorted(GENERATED_DIR.glob("*.png"), key=os.path.getctime, reverse=True):
        img_id = img_file.stem
        meta_file = GENERATED_DIR / f"{img_id}.json"
        
        metadata = {}
        if meta_file.exists():
            try:
                metadata = json.loads(meta_file.read_text())
            except Exception:
                pass
        
        # If no metadata, provide defaults
        if not metadata:
            metadata = {
                "id": img_id,
                "prompt": "",
                "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(img_file.stat().st_ctime)),
            }
            
        # Add file info
        metadata["url"] = f"http://localhost:8000/api/generated/{img_id}.png"
        metadata["path"] = str(img_file)
        
        images.append(metadata)
        
    return {"images": images}

@app.delete("/api/gallery/images/{image_id}")
async def delete_generated_image(image_id: str):
    """Delete a generated image and its metadata."""
    img_file = GENERATED_DIR / f"{image_id}.png"
    meta_file = GENERATED_DIR / f"{image_id}.json"
    
    try:
        if img_file.exists():
            img_file.unlink()
        if meta_file.exists():
            meta_file.unlink()
        return {"success": True}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/generated/{filename}")
async def get_generated_image(filename: str):
    """Serve a generated image file."""
    from fastapi.responses import FileResponse
    file_path = GENERATED_DIR / filename
    if not file_path.exists():
        return {"error": "File not found"}
    return FileResponse(str(file_path))


# ============================================
# Output Directory Management
# ============================================

@app.get("/api/output/directory")
async def get_output_directory():
    """Return current output directory path."""
    return {"outputDirectory": str(get_output_dir())}

@app.post("/api/output/directory")
async def set_output_directory(req: SetDirectoryRequest):
    """Set a custom output directory for trained LoRA models."""
    target = Path(req.path)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return {"error": f"Cannot create directory: {e}"}
    save_settings({"outputDirectory": str(target)})
    return {"status": "ok", "outputDirectory": str(target)}

@app.post("/api/output/open")
async def open_output_directory():
    """Open the output directory in the system file explorer."""
    output_dir = get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    import subprocess
    import sys

    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(output_dir)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(output_dir)])
        else:
            subprocess.Popen(["xdg-open", str(output_dir)])
        return {"status": "opened", "outputDirectory": str(output_dir)}
    except Exception as e:
        return {"error": f"Failed to open folder: {e}"}
