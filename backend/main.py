import asyncio
import time
import uuid
import random
import os
import json
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticBase

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_training_sessions = {}
active_connections: list[WebSocket] = []
active_downloads: dict[str, asyncio.Task] = {}

# --- Models directory ---
BACKEND_DIR = Path(__file__).parent.resolve()
DEFAULT_MODELS_DIR = BACKEND_DIR / "models"
SETTINGS_FILE = BACKEND_DIR / "settings.json"

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

# --- Base Model Catalog ---
MODEL_CATALOG = [
    {
        "id": "sd15",
        "name": "Stable Diffusion 1.5",
        "shortName": "SD 1.5",
        "description": "The classic. Lightweight, fast, with a massive community ecosystem of LoRAs. Best for anime and artistic styles.",
        "architecture": "sd15",
        "fileSize": 4265380864,
        "filename": "v1-5-pruned-emaonly.safetensors",
        "downloadUrl": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
    },
    {
        "id": "sdxl10",
        "name": "Stable Diffusion XL 1.0",
        "shortName": "SDXL",
        "description": "High-resolution powerhouse. 1024×1024 native. Superior detail and composition for photorealistic and artistic output.",
        "architecture": "sdxl",
        "fileSize": 6938078334,
        "filename": "sd_xl_base_1.0.safetensors",
        "downloadUrl": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
    },
    {
        "id": "flux-dev",
        "name": "Flux.1 Dev",
        "shortName": "Flux",
        "description": "Next-gen architecture by Black Forest Labs. State-of-the-art quality with superior prompt adherence and photorealism.",
        "architecture": "flux",
        "fileSize": 23802932552,
        "filename": "flux1-dev.safetensors",
        "downloadUrl": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors",
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
from trainer import LoRATrainer, get_gpu_info, get_optimization_profile, prepare_dataset
trainer_instance = LoRATrainer()

TRAINING_DATA_DIR = BACKEND_DIR / "training_data"
OUTPUT_DIR = BACKEND_DIR / "output"

@app.get("/api/gpu/info")
async def gpu_info():
    """Return GPU info and optimization profiles for all architectures."""
    info = get_gpu_info()
    profiles = {
        "sd15": get_optimization_profile(info["vram_gb"], "sd15"),
        "sdxl": get_optimization_profile(info["vram_gb"], "sdxl"),
        "flux": get_optimization_profile(info["vram_gb"], "flux"),
    }
    return {"gpu": info, "profiles": profiles}

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
    output_dir = OUTPUT_DIR / session_id
    
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
    return {"status": "stopped"}

@app.get("/api/training/output/{session_id}")
async def get_training_output(session_id: str):
    output_dir = OUTPUT_DIR / session_id
    if not output_dir.exists():
        return {"error": "Output not found"}
    safetensors = list(output_dir.glob("*.safetensors"))
    if safetensors:
        return {"path": str(safetensors[0]), "dir": str(output_dir)}
    return {"error": "No .safetensors file found"}

class CaptionRequest(PydanticBase):
    imageId: str
    imageUrl: Optional[str] = None # Will contain the local File path from Electron

# Global lazy-loaded models
processor = None
blip_model = None

@app.post("/api/dataset/caption")
async def auto_caption(req: CaptionRequest):
    global processor, blip_model
    
    if processor is None:
        print("Loading BLIP model on first request...")
        from transformers import BlipProcessor, BlipForConditionalGeneration
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        print("BLIP loaded successfully.")

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

class GenerateRequest(PydanticBase):
    prompt: str
    negativePrompt: str
    width: int
    height: int
    seed: int
    cfgScale: float
    steps: int
    loraWeight: Optional[float] = 1.0
    sampler: Optional[str] = "Euler a"

@app.post("/api/playground/generate")
async def playground_generate(req: GenerateRequest):
    await asyncio.sleep(2.0) # Simulate GPU Diffusers rendering latency
    
    import base64
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{req.width}" height="{req.height}">
      <rect width="{req.width}" height="{req.height}" fill="#0A0A0A"/>
      <path d="M0 0 L{req.width} {req.height} M{req.width} 0 L0 {req.height}" stroke="#333333" stroke-width="2" stroke-opacity="0.3"/>
      <rect width="{req.width}" height="{req.height}" fill="none" stroke="#333333" stroke-width="8"/>
      <text x="50%" y="42%" dominant-baseline="middle" text-anchor="middle" fill="#ffffff" font-family="sans-serif" font-size="24" font-weight="600">Simulated Inference</text>
      <text x="50%" y="49%" dominant-baseline="middle" text-anchor="middle" fill="#888888" font-family="sans-serif" font-size="14">Prompt: {req.prompt[:40] + ('...' if len(req.prompt) > 40 else '')}</text>
      <text x="50%" y="56%" dominant-baseline="middle" text-anchor="middle" fill="#00FF9D" font-family="monospace" font-size="14" font-weight="bold">LoRA Weight: {req.loraWeight} | {req.sampler}</text>
      <text x="50%" y="62%" dominant-baseline="middle" text-anchor="middle" fill="#666666" font-family="monospace" font-size="12">Seed: {req.seed} | CFG: {req.cfgScale} | Steps: {req.steps}</text>
    </svg>"""
    
    b64 = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
    data_url = f"data:image/svg+xml;base64,{b64}"
    
    return {"url": data_url, "seed": req.seed}

@app.websocket("/ws/training")
async def training_websocket(websocket: WebSocket):
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
    
    loop = asyncio.get_event_loop()
    
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
                        "phase": phase if phase == "training" else "preparing",
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
        
    except Exception as e:
        error_msg = str(e)
        print(f"[Training] Error: {error_msg}")
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

@app.get("/api/models/directory")
async def get_models_directory():
    return {"modelsDirectory": str(get_models_dir())}

async def run_download(model_id: str, url: str, filename: str, models_dir: Path):
    """Download a model file with real-time progress broadcasting."""
    import aiohttp
    
    part_path = models_dir / (filename + ".part")
    final_path = models_dir / filename
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    await broadcast_to_connections({
                        "type": "download_error",
                        "data": {"modelId": model_id, "message": f"HTTP {response.status}: {response.reason}"}
                    })
                    return
                
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                last_broadcast = 0
                start_time = time.time()
                chunk_times = []
                
                with open(part_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(1024 * 1024):  # 1MB chunks
                        if model_id not in active_downloads:
                            break
                        
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Track speed
                        now = time.time()
                        chunk_times.append((now, len(chunk)))
                        # Keep only last 10 seconds of data for speed calc
                        chunk_times = [(t, s) for t, s in chunk_times if now - t < 10]
                        
                        # Broadcast progress every 500ms
                        if now - last_broadcast >= 0.5:
                            last_broadcast = now
                            
                            # Calculate speed
                            if len(chunk_times) > 1:
                                duration = chunk_times[-1][0] - chunk_times[0][0]
                                total_bytes_in_window = sum(s for _, s in chunk_times)
                                speed = total_bytes_in_window / max(duration, 0.001)
                            else:
                                speed = 0
                            
                            # Format speed
                            if speed > 1024 * 1024:
                                speed_str = f"{speed / (1024 * 1024):.1f} MB/s"
                            elif speed > 1024:
                                speed_str = f"{speed / 1024:.1f} KB/s"
                            else:
                                speed_str = f"{speed:.0f} B/s"
                            
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
                
                # Rename .part to final filename
                if model_id in active_downloads:
                    if final_path.exists():
                        final_path.unlink()
                    part_path.rename(final_path)
                    
                    await broadcast_to_connections({
                        "type": "download_complete",
                        "data": {"modelId": model_id, "localPath": str(final_path)}
                    })
                    
    except asyncio.CancelledError:
        if part_path.exists():
            part_path.unlink()
    except Exception as e:
        await broadcast_to_connections({
            "type": "download_error",
            "data": {"modelId": model_id, "message": str(e)}
        })
        if part_path.exists():
            part_path.unlink()
    finally:
        active_downloads.pop(model_id, None)
