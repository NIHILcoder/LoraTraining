import asyncio
import time
import uuid
import random
import os
import json
import shutil
import aiofiles
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticBase

# Windows stability fixes
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["ACCELERATE_USE_CPU_INIT"] = "0"

# CLIPTextModel shim only — do not disable torch.load pickle checks.
try:
    from transformers.models.clip.modeling_clip import CLIPTextModel
    if not hasattr(CLIPTextModel, "text_model"):
        CLIPTextModel.text_model = property(lambda self: self)
except Exception:
    pass

try:
    import transformers.modeling_utils
    transformers.modeling_utils._CONFIG_FOR_LOW_CPU_MEM_USAGE = False
except Exception:
    pass
# ----------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3005",
        "http://127.0.0.1:3005",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "app://.",
        "null",
    ],
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
        header_ok = bool(auth_header and auth_header.startswith("Bearer ") and auth_header[7:] == API_TOKEN)
        # <img>/<a> requests cannot set headers, so accept a matching ?token= query param too
        query_ok = request.query_params.get("token") == API_TOKEN
        if not header_ok and not query_ok:
            return JSONResponse(status_code=401, content={"error": "Unauthorized"})
            
    return await call_next(request)

active_training_sessions = {}
active_connections: list[WebSocket] = []
active_downloads: dict[str, asyncio.Task] = {}

def _log_dep_versions():
    """Log tested dependency versions at startup so mismatches are visible (P1-02)."""
    import importlib.metadata as _md
    names = ["torch", "transformers", "diffusers", "peft", "accelerate", "safetensors", "fastapi", "pydantic"]
    parts = []
    for n in names:
        try:
            parts.append(f"{n}={_md.version(n)}")
        except Exception:
            parts.append(f"{n}=?")
    print("[Startup] Dependency versions: " + ", ".join(parts))

_log_dep_versions()

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

ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}

def assert_under(root: Path, candidate: Path) -> Path:
    """Resolve candidate and require it to sit under root (blocks path traversal)."""
    root_r = root.resolve()
    path = candidate.resolve()
    if path != root_r and not path.is_relative_to(root_r):
        raise HTTPException(status_code=400, detail="Invalid path")
    return path

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


# Architectures this build can actually train / generate with.
TRAINING_ARCHITECTURES = {"sd15", "sdxl"}
INFERENCE_ARCHITECTURES = {"sd15", "sd21", "sdxl"}

def apply_arch_support(model: dict) -> dict:
    """Fill supportedTraining / supportedInference from architecture when missing."""
    arch = model.get("architecture", "sd15")
    model.setdefault("supportedTraining", arch in TRAINING_ARCHITECTURES)
    model.setdefault("supportedInference", arch in INFERENCE_ARCHITECTURES)
    return model


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
        "downloadUrl": "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
        "sha256": "6ce0161689b3853acaa03779ec93eafe75a02f4ced659bee03f50797806fa2fa",
        "supportedTraining": True,
        "supportedInference": True,
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
        "supportedTraining": False,
        "supportedInference": True,
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
        "sha256": "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b",
        "supportedTraining": True,
        "supportedInference": True,
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
        "supportedTraining": False,
        "supportedInference": False,
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
        "supportedTraining": False,
        "supportedInference": False,
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
        "supportedTraining": False,
        "supportedInference": False,
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
        "supportedTraining": False,
        "supportedInference": False,
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
    # Advanced knobs surfaced in the UI — must be declared or Pydantic silently drops them
    enableBucketing: Optional[bool] = True
    captionDropout: Optional[float] = 0.0
    noiseOffset: Optional[float] = 0.0
    datasetId: Optional[str] = None
    baseModel: str

class TrainingStartRequest(PydanticBase):
    config: TrainingConfig
    images: List[dict]  # [{filePath, captions: [...]}]

# Global trainer instance
from trainer import LoRATrainer, get_gpu_info, get_optimization_profile, prepare_dataset, estimate_training_time, ARCH_VRAM_MIN
from model_resolve import resolve_base_model
trainer_instance = LoRATrainer()

def load_custom_models() -> list:
    """Custom checkpoints registered in settings.json (local import / URL add)."""
    if SETTINGS_FILE.exists():
        try:
            settings = json.loads(SETTINGS_FILE.read_text())
            return list(settings.get("customModels") or [])
        except Exception:
            pass
    return []

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

from fastapi import APIRouter
from fastapi.responses import FileResponse

dataset_router = APIRouter(prefix="/api/datasets")

DATASETS_FILE = TRAINING_DATA_DIR / "datasets.json"

def load_datasets():
    if DATASETS_FILE.exists():
        try:
            return json.loads(DATASETS_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_datasets(data):
    DATASETS_FILE.write_text(json.dumps(data, indent=2))

@dataset_router.get("")
async def get_datasets():
    datasets = load_datasets()
    return list(datasets.values())

class CreateDatasetReq(PydanticBase):
    name: str

@dataset_router.post("")
async def create_dataset(req: CreateDatasetReq):
    ds_id = str(uuid.uuid4())[:8]
    ds = {
        "id": ds_id,
        "name": req.name,
        "images": [],
        "totalSize": 0,
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    datasets = load_datasets()
    datasets[ds_id] = ds
    save_datasets(datasets)
    return ds

@dataset_router.delete("/{ds_id}")
async def delete_dataset(ds_id: str):
    datasets = load_datasets()
    if ds_id in datasets:
        del datasets[ds_id]
        save_datasets(datasets)
        ds_dir = assert_under(TRAINING_DATA_DIR, TRAINING_DATA_DIR / ds_id)
        if ds_dir.exists():
            shutil.rmtree(ds_dir, ignore_errors=True)
    return {"status": "ok"}

class UpdateDatasetReq(PydanticBase):
    dataset: dict

@dataset_router.put("/{ds_id}")
async def update_dataset(ds_id: str, req: UpdateDatasetReq):
    datasets = load_datasets()
    if ds_id in datasets:
        datasets[ds_id] = req.dataset
        datasets[ds_id]["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save_datasets(datasets)
        return datasets[ds_id]
    raise HTTPException(status_code=404, detail="Dataset not found")

class UploadLocalImageReq(PydanticBase):
    filePath: str
    filename: str
    size: int

@dataset_router.post("/{ds_id}/images/local")
async def upload_local_image(ds_id: str, req: UploadLocalImageReq):
    datasets = load_datasets()
    if ds_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    ds_dir = assert_under(TRAINING_DATA_DIR, TRAINING_DATA_DIR / ds_id) / "images"
    ds_dir.mkdir(parents=True, exist_ok=True)

    src = Path(req.filePath)
    try:
        src = src.resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not src.is_file():
        raise HTTPException(status_code=400, detail="File not found")
    ext = src.suffix.lower()
    if ext not in ALLOWED_IMAGE_EXT:
        raise HTTPException(status_code=400, detail="Only PNG, JPEG, and WEBP images can be added.")

    img_id = str(uuid.uuid4())[:8]
    dest_path = ds_dir / f"{img_id}{ext}"
    
    try:
        shutil.copy2(str(src), dest_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to copy file: {e}")
        
    # Read real image dimensions (previously hardcoded to 1024x1024, corrupting bucket metadata)
    img_width, img_height = 0, 0
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(dest_path) as _im:
            img_width, img_height = _im.size
    except Exception:
        pass

    img_metadata = {
        "id": img_id,
        "filename": req.filename,
        "url": f"/api/datasets/{ds_id}/images/{img_id}/raw",
        "filePath": str(dest_path.resolve()),
        "size": req.size,
        "width": img_width,
        "height": img_height,
        "uploadedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "captions": []
    }
    
    datasets[ds_id]["images"].append(img_metadata)
    datasets[ds_id]["totalSize"] += req.size
    datasets[ds_id]["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_datasets(datasets)
    
    return img_metadata

@dataset_router.get("/{ds_id}/images/{img_id}/raw")
async def get_image_raw(ds_id: str, img_id: str):
    datasets = load_datasets()
    if ds_id in datasets:
        for img in datasets[ds_id]["images"]:
            if img["id"] == img_id:
                fp = Path(img["filePath"])
                safe = assert_under(TRAINING_DATA_DIR, fp)
                return FileResponse(str(safe))
    raise HTTPException(status_code=404, detail="Not found")

class PatchImageCaptionsReq(PydanticBase):
    captions: List[str]

@dataset_router.patch("/{ds_id}/images/{img_id}/captions")
async def patch_image_captions(ds_id: str, img_id: str, req: PatchImageCaptionsReq):
    """Update captions for a single image without rewriting the rest of the dataset."""
    datasets = load_datasets()
    if ds_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    target = next((i for i in datasets[ds_id].get("images", []) if i.get("id") == img_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail="Image not found")
    target["captions"] = req.captions
    datasets[ds_id]["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_datasets(datasets)
    return target

class CaptionUpdateItem(PydanticBase):
    imageId: str
    captions: List[str]

class PatchManyCaptionsReq(PydanticBase):
    updates: List[CaptionUpdateItem]

@dataset_router.patch("/{ds_id}/captions")
async def patch_many_captions(ds_id: str, req: PatchManyCaptionsReq):
    """Update captions for many images without replacing the rest of the dataset."""
    datasets = load_datasets()
    if ds_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    by_id = {i.get("id"): i for i in datasets[ds_id].get("images", [])}
    for u in req.updates:
        if u.imageId not in by_id:
            raise HTTPException(status_code=404, detail=f"Image not found: {u.imageId}")
    for u in req.updates:
        by_id[u.imageId]["captions"] = u.captions
    datasets[ds_id]["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_datasets(datasets)
    return {"status": "ok", "updated": len(req.updates)}

@dataset_router.delete("/{ds_id}/images/{img_id}")
async def delete_dataset_image(ds_id: str, img_id: str):
    """Delete a single image from a dataset (atomic on the server; no full-dataset rewrite race)."""
    datasets = load_datasets()
    if ds_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ds = datasets[ds_id]
    imgs = ds.get("images", [])
    target = next((i for i in imgs if i.get("id") == img_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        fp = target.get("filePath")
        if fp:
            safe = assert_under(TRAINING_DATA_DIR, Path(fp))
            if safe.is_file():
                safe.unlink()
    except HTTPException:
        pass
    except Exception:
        pass
    ds["images"] = [i for i in imgs if i.get("id") != img_id]
    ds["totalSize"] = max(0, ds.get("totalSize", 0) - int(target.get("size", 0) or 0))
    ds["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_datasets(datasets)
    return {"status": "ok"}

app.include_router(dataset_router)

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
        raise HTTPException(status_code=400, detail="Training is already in progress")
    
    config = req.config
    images = req.images
    session_id = str(uuid.uuid4())
    
    # Find the base model file
    models_dir = get_models_dir()
    model_path = None
    arch = config.baseModel
    
    if arch == "sd21":
        raise HTTPException(
            status_code=400,
            detail="SD 2.1 training is temporarily disabled: it needs the OpenCLIP ViT-H text encoder and v-prediction loss, which are not implemented yet. Use SD 1.5 or SDXL.",
        )
    if arch not in TRAINING_ARCHITECTURES:
        raise HTTPException(status_code=400, detail=f"Architecture '{arch}' is currently not supported for training.")
        
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
        raise HTTPException(status_code=400, detail=f"No downloaded base model found for architecture '{arch}'. Please download one from Models Hub.")
    
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
    # Launch from HTTP so training does not depend on a live WebSocket message.
    _ensure_training_task(session_id)
    
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
        raise HTTPException(status_code=404, detail="Output not found")
    safetensors = list(output_dir.glob("*.safetensors"))
    if safetensors:
        return {"path": str(safetensors[0]), "dir": str(output_dir)}
    raise HTTPException(status_code=404, detail="No .safetensors file found")


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

    # --- Resolve base model path (catalog + imported custom checkpoints) ---
    models_dir = get_models_dir()
    resolved_path, resolved_arch, resolved_name = resolve_base_model(
        req.baseModelId,
        models_dir,
        MODEL_CATALOG,
        load_custom_models(),
    )
    model_path = str(resolved_path) if resolved_path else None
    target_arch = resolved_arch or req.baseModelId or "sd15"
    requested_id = (req.baseModelId or "").strip()
    specific_request = bool(requested_id) and requested_id.lower() not in ("auto", "none")
    if specific_request and not model_path:
        raise HTTPException(
            status_code=400,
            detail=f"Base model '{requested_id}' is not available on disk. Re-import or download it.",
        )

    # --- Resolve LoRA path ---
    lora_path = None
    if req.loraModelId and req.loraModelId != "none":
        lora_dir = assert_under(get_output_dir(), get_output_dir() / req.loraModelId)
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
        # Serialize GPU work: the pipeline cache is shared and cleared on model switch,
        # so concurrent generations must not race on it.
        async with _inference_lock:
            pipe = await asyncio.to_thread(
                _load_pipeline, model_path, lora_path, req.loraWeight, target_arch
            )

            result_image = await asyncio.to_thread(
                _run_inference, pipe, req, actual_seed
            )

        img_id = str(uuid.uuid4())[:8]
        img_path = GENERATED_DIR / f"{img_id}.png"
        meta_path = GENERATED_DIR / f"{img_id}.json"

        # Resolve names for metadata
        lora_name = "None"
        if req.loraModelId and req.loraModelId != "none":
            lora_dir = assert_under(get_output_dir(), get_output_dir() / req.loraModelId)
            safetensors = list(lora_dir.glob("*.safetensors"))
            if safetensors:
                lora_name = safetensors[0].stem.replace("_", " ").title()

        base_name = resolved_name or "Auto"

        # Embed A1111/Civitai-compatible generation parameters into the PNG (PNG-info text chunk)
        params_text = (
            f"{req.prompt}\n"
            f"Negative prompt: {req.negativePrompt or ''}\n"
            f"Steps: {req.steps}, Sampler: {req.sampler}, CFG scale: {req.cfgScale}, "
            f"Seed: {actual_seed}, Size: {req.width}x{req.height}, Model: {base_name}"
        )
        if lora_name and lora_name != "None":
            params_text += f", LoRA: {lora_name}:{req.loraWeight}"

        from PIL.PngImagePlugin import PngInfo
        png_info = PngInfo()
        png_info.add_text("parameters", params_text)
        result_image.save(str(img_path), "PNG", pnginfo=png_info)

        # Save sidecar metadata for the Gallery
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

        # Return a served URL (not a multi-MB base64 payload); the frontend adds origin + token.
        return {"url": f"/api/generated/{img_id}.png", "seed": actual_seed, "mock": False}

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Inference] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

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
            repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

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
        # LoRA weight is already applied via pipe.set_adapters(...) in _load_pipeline;
        # passing a cross_attention scale here as well would double-scale it.
    )
    return result.images[0]


def _make_mock_svg(prompt: str, seed: int, sampler: str, lora_weight, reason: str) -> str:
    """Generate an informative SVG placeholder when inference can't run."""
    import urllib.parse
    from xml.sax.saxutils import escape
    prompt_short = (prompt[:55] + "…") if len(prompt) > 55 else prompt
    lora_line = f"LoRA Weight: {lora_weight} | {sampler}" if lora_weight is not None else f"No LoRA | {sampler}"
    # Escape any user prompt / exception text before it is interpolated into SVG/XML markup
    prompt_short = escape(prompt_short)
    lora_line = escape(lora_line)
    reason = escape(reason)

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
            raise HTTPException(status_code=503, detail=f"Captioning model failed to load: {e}")

    try:
        from PIL import Image
        import os
        
        # Load local image using the path passed from Electron
        if not req.imageUrl or not os.path.exists(req.imageUrl):
             raise HTTPException(status_code=400, detail="Image file not found locally")

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
        
    except HTTPException:
        raise
    except Exception as e:
        print("Captioning error:", e)
        raise HTTPException(status_code=500, detail=f"Captioning failed: {e}")


class BatchCaptionRequest(PydanticBase):
    images: List[dict]  # [{imageId, imageUrl}]

def _caption_image_file(image_path: str) -> list:
    """Caption a single local image with BLIP (assumes the model is already loaded)."""
    from PIL import Image
    if not image_path or not os.path.exists(image_path):
        raise HTTPException(status_code=400, detail="Image file not found locally")
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt")
    out = blip_model.generate(
        **inputs, max_new_tokens=50, num_return_sequences=3,
        do_sample=True, top_k=30, temperature=0.7,
    )
    captions = processor.batch_decode(out, skip_special_tokens=True)
    stop_words = {
        "a", "an", "the", "and", "but", "or", "for", "nor", "on", "at", "in", "with", "by", "from",
        "up", "about", "into", "over", "after", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "of", "to", "it", "that", "this", "there", "their",
        "they", "he", "she", "his", "her", "him", "its", "some", "many", "few", "all", "any", "no",
        "not", "very", "too", "so", "just", "can", "will", "would", "could", "should", "what", "which",
        "who", "when", "where", "why", "how", "background", "foreground", "picture", "image", "photo",
        "photography", "showing", "view", "close", "standing", "sitting", "looking",
    }
    keyword_set = set()
    for cap in captions:
        for w in (x.strip(".,!?\"'()[]") for x in cap.lower().split()):
            if len(w) > 2 and w not in stop_words and not w.isdigit():
                keyword_set.add(w)
    return [captions[0]] + sorted(keyword_set)

@app.post("/api/dataset/caption/batch")
async def auto_caption_batch(req: BatchCaptionRequest):
    """Caption many images in one request; per-image failures are reported, not fatal."""
    global processor, blip_model
    if processor is None or blip_model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        try:
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base", use_safetensors=True,
            )
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Captioning model failed to load: {e}")
    results = []
    for item in req.images:
        image_id = item.get("imageId")
        image_url = item.get("imageUrl")
        try:
            tags = await asyncio.to_thread(_caption_image_file, image_url)
            results.append({"imageId": image_id, "tags": tags})
        except HTTPException as he:
            results.append({"imageId": image_id, "error": he.detail})
        except Exception as e:
            results.append({"imageId": image_id, "error": str(e)})
    return {"results": results}



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
            # Fallback for older clients; HTTP /training/start now launches the loop.
            if data.get("type") == "start_training":
                session_id = (data.get("payload") or {}).get("sessionId")
                if session_id:
                    _ensure_training_task(session_id)
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)

def _ensure_training_task(session_id: str) -> None:
    """Start the trainer task once per session. Safe to call from HTTP and WebSocket."""
    session = active_training_sessions.get(session_id)
    if not session:
        return
    task = session.get("task")
    if task is not None and not task.done():
        return
    session["task"] = asyncio.create_task(run_real_training(session_id, session))

async def run_real_training(session_id: str, session: dict, websocket=None):
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
        result.append(apply_arch_support(m))
    
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
                result.append(apply_arch_support(cm))
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
    # Only .safetensors — pickle checkpoints (.ckpt/.bin/.pt) can execute code on load.
    if filename.endswith((".ckpt", ".bin", ".pt")):
        raise HTTPException(
            status_code=400,
            detail="Only .safetensors models can be added. .ckpt/.bin/.pt use Python pickle, which can run arbitrary code when loaded.",
        )
    if not filename.endswith(".safetensors"):
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

class ImportLocalModelReq(PydanticBase):
    filePath: str
    name: Optional[str] = None
    architecture: Optional[str] = "sd15"

@app.post("/api/models/base/import")
async def import_local_model(req: ImportLocalModelReq):
    """Register an existing local .safetensors file as a base model (copies it into the models dir)."""
    src = Path(req.filePath)
    if not src.exists() or not src.is_file():
        raise HTTPException(status_code=400, detail="File not found.")
    if src.suffix.lower() != ".safetensors":
        raise HTTPException(status_code=400, detail="Only .safetensors models can be imported.")

    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / src.name

    # Disk-space precheck (unless the file is already in place)
    try:
        size = src.stat().st_size
        if not dest.exists() and shutil.disk_usage(str(models_dir)).free < size * 1.05:
            raise HTTPException(status_code=507, detail="Not enough disk space to import this model.")
    except HTTPException:
        raise
    except Exception:
        pass

    if src.resolve() != dest.resolve():
        try:
            shutil.copy2(str(src), str(dest))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to copy file: {e}")

    model_id = f"custom-{uuid.uuid4().hex[:8]}"
    name = req.name or src.stem.replace(".", " ").replace("_", " ").replace("-", " ").title()
    custom_model = {
        "id": model_id,
        "name": name,
        "shortName": name[:12],
        "description": f"Imported from: {src.name}",
        "architecture": req.architecture or "sd15",
        "fileSize": dest.stat().st_size if dest.exists() else 0,
        "filename": dest.name,
        "downloadUrl": "",
        "isCustom": True,
        "status": "downloaded",
        "localPath": str(dest),
    }

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
        raise HTTPException(status_code=404, detail="Model not found")

    model_info = apply_arch_support(dict(model_info))
    if not model_info.get("supportedTraining") and not model_info.get("supportedInference"):
        raise HTTPException(
            status_code=400,
            detail="This architecture is not supported yet. Training and Playground currently work with SD 1.5 and SDXL.",
        )
    
    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    # Disk-space precheck — model files are 4–24 GB; fail early with a clear message
    expected = int(model_info.get("fileSize") or 0)
    if expected > 0:
        try:
            free = shutil.disk_usage(str(models_dir)).free
            if free < expected * 1.05:
                raise HTTPException(
                    status_code=507,
                    detail=f"Not enough disk space: need ~{expected / 1024**3:.1f} GB, only {free / 1024**3:.1f} GB free at {models_dir}.",
                )
        except HTTPException:
            raise
        except Exception:
            pass  # disk_usage failure shouldn't block the download

    task = asyncio.create_task(
        run_download(model_id, model_info["downloadUrl"], model_info["filename"], models_dir, model_info.get("sha256"))
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
    
    raise HTTPException(status_code=404, detail="Model not found")

class SetDirectoryRequest(PydanticBase):
    path: str

@app.post("/api/models/directory")
async def set_models_directory(req: SetDirectoryRequest):
    target = Path(req.path)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot create directory: {e}")
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

async def run_download(model_id: str, url: str, filename: str, models_dir: Path, expected_sha: Optional[str] = None):
    """Download a model file with resume support, optional SHA256 verification, and progress broadcasting."""
    import aiohttp
    import aiofiles
    import shutil
    import hashlib

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

        # Resume: continue from an existing .part file if one is present
        resume_from = part_path.stat().st_size if part_path.exists() else 0
        if resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"
            print(f"[Download] Found partial file ({resume_from} bytes) — attempting resume.")

        async with aiohttp.ClientSession(headers=headers) as session:
            print(f"[Download] Starting request to: {url}")
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=None), allow_redirects=True) as response:
                print(f"[Download] Response status: {response.status} for {url}")
                if response.status not in (200, 206):
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

                # Server ignored the Range request and is sending the whole file — restart cleanly
                if resume_from > 0 and response.status == 200:
                    print("[Download] Server does not support resume; restarting from scratch.")
                    resume_from = 0

                # Determine the full expected size
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    try:
                        total_size = int(content_range.rsplit("/", 1)[-1])
                    except (ValueError, IndexError):
                        total_size = resume_from + int(response.headers.get("Content-Length", 0))
                else:
                    total_size = int(response.headers.get("Content-Length", 0))

                # Integrity check only on a fresh full download (a resumed remainder can be small)
                if resume_from == 0 and total_size < 10 * 1024 * 1024:
                    error_msg = "Downloaded file is too small. Possibly an invalid URL or restricted access."
                    print(f"[Download] Error: {error_msg} (Size: {total_size} bytes)")
                    await broadcast_to_connections({
                        "type": "download_error",
                        "data": {"modelId": model_id, "message": error_msg}
                    })
                    return

                # Prepare hash only when we actually intend to verify
                sha = hashlib.sha256() if expected_sha else None
                if sha and resume_from > 0:
                    async with aiofiles.open(part_path, mode="rb") as pf:
                        while True:
                            block = await pf.read(1024 * 1024)
                            if not block:
                                break
                            sha.update(block)

                downloaded = resume_from
                last_broadcast = 0
                start_time = time.time()
                chunk_times = []

                print(f"[Download] Total size: {total_size / (1024**3):.2f} GB (resuming from {resume_from} bytes)")

                # Use aiofiles for non-blocking disk I/O; append when resuming
                async with aiofiles.open(part_path, mode=("ab" if resume_from > 0 else "wb")) as f:
                    async for chunk in response.content.iter_chunked(1024 * 1024):  # 1MB chunks
                        if model_id not in active_downloads:
                            print(f"[Download] Interrupted: {model_id}")
                            break
                        
                        await f.write(chunk)
                        if sha:
                            sha.update(chunk)
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

                    # Verify checksum when an expected digest is known
                    if sha and expected_sha:
                        digest = sha.hexdigest()
                        if digest.lower() != expected_sha.lower():
                            if part_path.exists():
                                part_path.unlink()
                            error_msg = f"Checksum mismatch: expected {expected_sha[:12]}…, got {digest[:12]}…"
                            print(f"[Download] {error_msg}")
                            await broadcast_to_connections({
                                "type": "download_error",
                                "data": {"modelId": model_id, "message": error_msg}
                            })
                            return
                        print(f"[Download] SHA256 verified: {digest[:12]}…")

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
        print(f"[Download] Cancelled: {filename} — keeping partial file for resume")
        # Do not delete .part; the next download attempt resumes from it.
    except Exception as e:
        error_msg = str(e)
        print(f"[Download] FAILED: {filename} - {error_msg}")
        await broadcast_to_connections({
            "type": "download_error",
            "data": {"modelId": model_id, "message": error_msg}
        })
        # Keep the .part file so the next download attempt can resume instead of restarting.
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
    target = assert_under(get_output_dir(), get_output_dir() / model_id)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Model not found")
        
    try:
        import shutil
        shutil.rmtree(target)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {e}")


@app.post("/api/gallery/models/{model_id}/open")
async def open_model_folder(model_id: str):
    """Open the model folder in the system file explorer."""
    target = assert_under(get_output_dir(), get_output_dir() / model_id)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Model folder not found")
    
    import subprocess
    import sys
    
    try:
        if sys.platform == "win32":
            os.startfile(target)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", target])
        else:
            subprocess.Popen(["xdg-open", target])
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open folder: {e}")


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
        metadata["url"] = f"/api/generated/{img_id}.png"
        metadata["path"] = str(img_file)
        
        images.append(metadata)
        
    return {"images": images}

@app.delete("/api/gallery/images/{image_id}")
async def delete_generated_image(image_id: str):
    """Delete a generated image and its metadata."""
    if "/" in image_id or "\\" in image_id or ".." in image_id:
        raise HTTPException(status_code=400, detail="Invalid image id")
    img_file = assert_under(GENERATED_DIR, GENERATED_DIR / f"{image_id}.png")
    meta_file = GENERATED_DIR / f"{image_id}.json"
    
    try:
        if img_file.exists():
            img_file.unlink()
        if meta_file.exists():
            meta_file.unlink()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/generated/{filename}")
async def get_generated_image(filename: str):
    """Serve a generated image file."""
    from fastapi.responses import FileResponse
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    file_path = assert_under(GENERATED_DIR, GENERATED_DIR / filename)
    if file_path.suffix.lower() not in ALLOWED_IMAGE_EXT or not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
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
        raise HTTPException(status_code=500, detail=f"Cannot create directory: {e}")
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
        raise HTTPException(status_code=500, detail=f"Failed to open folder: {e}")
