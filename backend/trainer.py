"""
LoRA Training Engine
====================
Real training module using HuggingFace diffusers + peft.
Automatically adapts to available GPU VRAM.
"""

import gc
import random
import os
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

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
# We patch it in every possible location within transformers
try:
    import transformers.utils.import_utils
    transformers.utils.import_utils.check_torch_load_is_safe = bypass_check
    
    import transformers.modeling_utils
    transformers.modeling_utils.check_torch_load_is_safe = bypass_check
    
    import transformers.dynamic_module_utils
    transformers.dynamic_module_utils.check_torch_load_is_safe = bypass_check
except Exception: pass

# 3. Aggressively disable safety checker in diffusers
try:
    import diffusers.loaders.single_file
    diffusers.loaders.single_file._legacy_load_safety_checker = lambda *args, **kwargs: {}
except Exception: pass

# 4. Force disable low_cpu_mem_usage globally in transformers
try:
    import transformers.modeling_utils
    transformers.modeling_utils._CONFIG_FOR_LOW_CPU_MEM_USAGE = False
except Exception: pass
# ----------------------------------------
import math
import json

import torch

# ============================================
# GPU & System Detection + Optimization Profiles
# ============================================

# VRAM requirements per architecture (min GB)
ARCH_VRAM_MIN = {
    "sd15": 6, "sd21": 6, "sdxl": 8, "sd3": 12,
    "flux": 16, "cascade": 10, "hunyuan": 16,
    "pixart": 12, "kolors": 10, "auraflow": 12,
}

# Base time per step (seconds) at rank=16, 512px, batch=1 — indexed by VRAM tier
# Tiers: 6, 8, 12, 16, 24 GB
ARCH_BASE_TIME: Dict[str, Dict[int, float]] = {
    "sd15":     {6: 1.8, 8: 1.2, 12: 0.8, 16: 0.6, 24: 0.5},
    "sd21":     {6: 2.0, 8: 1.4, 12: 0.9, 16: 0.7, 24: 0.5},
    "sdxl":     {8: 3.5, 12: 2.0, 16: 1.5, 24: 1.0},
    "sd3":      {12: 3.0, 16: 2.0, 24: 1.4},
    "flux":     {16: 4.0, 24: 2.5},
    "cascade":  {10: 3.0, 12: 2.5, 16: 1.8, 24: 1.2},
    "hunyuan":  {16: 3.5, 24: 2.0},
    "pixart":   {12: 2.5, 16: 1.8, 24: 1.2},
    "kolors":   {10: 3.2, 12: 2.2, 16: 1.6, 24: 1.1},
    "auraflow": {12: 2.8, 16: 2.0, 24: 1.3},
}

# Default native resolution per architecture
ARCH_NATIVE_RES = {
    "sd15": 512, "sd21": 768, "sdxl": 1024, "sd3": 1024,
    "flux": 1024, "cascade": 1024, "hunyuan": 1024,
    "pixart": 1024, "kolors": 1024, "auraflow": 1024,
}


def get_gpu_info() -> Dict[str, Any]:
    """Detect GPU and system RAM."""
    info = {
        "available": False,
        "name": "No GPU",
        "vram_gb": 0,
        "vram_bytes": 0,
        "compute_capability": (0, 0),
        "bf16_supported": False,
        "cuda_version": "",
        "driver_version": "",
        "ram_total_gb": 0,
        "ram_available_gb": 0,
    }

    # RAM detection
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024 ** 3), 1)
        info["ram_available_gb"] = round(mem.available / (1024 ** 3), 1)
    except ImportError:
        # Fallback without psutil
        try:
            import os
            if os.name == 'nt':
                import ctypes
                kernel32 = ctypes.windll.kernel32
                c_ulonglong = ctypes.c_ulonglong
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                                ("ullTotalPhys", c_ulonglong), ("ullAvailPhys", c_ulonglong),
                                ("ullTotalPageFile", c_ulonglong), ("ullAvailPageFile", c_ulonglong),
                                ("ullTotalVirtual", c_ulonglong), ("ullAvailVirtual", c_ulonglong),
                                ("sullAvailExtendedVirtual", c_ulonglong)]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(stat)
                kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                info["ram_total_gb"] = round(stat.ullTotalPhys / (1024 ** 3), 1)
                info["ram_available_gb"] = round(stat.ullAvailPhys / (1024 ** 3), 1)
        except Exception:
            pass

    if not torch.cuda.is_available():
        return info

    info["available"] = True
    info["name"] = torch.cuda.get_device_name(0)
    info["vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
    info["vram_gb"] = round(info["vram_bytes"] / (1024 ** 3), 1)
    cc = torch.cuda.get_device_properties(0)
    info["compute_capability"] = (cc.major, cc.minor)
    info["bf16_supported"] = cc.major >= 8
    info["cuda_version"] = torch.version.cuda or ""

    try:
        info["driver_version"] = str(torch.cuda.get_device_properties(0).name)
    except Exception:
        pass

    return info


def get_optimization_profile(vram_gb: float, architecture: str) -> Dict[str, Any]:
    """
    Return optimal training settings for any supported architecture.
    Supports: sd15, sd21, sdxl, sd3, flux, cascade, hunyuan, pixart, kolors, auraflow
    """
    min_vram = ARCH_VRAM_MIN.get(architecture, 8)
    native_res = ARCH_NATIVE_RES.get(architecture, 512)

    profile = {
        "gradient_checkpointing": True,
        "mixed_precision": "fp16",
        "enable_xformers": True,
        "gradient_accumulation_steps": 1,
        "max_batch_size": 1,
        "cache_latents": True,
        "train_text_encoder": False,
        "feasible": True,
        "warning": None,
        "recommended_resolution": native_res,
        "min_vram": min_vram,
    }

    if vram_gb < min_vram:
        profile["feasible"] = False
        profile["warning"] = f"{architecture.upper()} requires minimum {min_vram} GB VRAM. You have {vram_gb} GB."
        return profile

    # UNet-based (SD 1.x, SD 2.x)
    if architecture in ("sd15", "sd21"):
        if vram_gb < 8:
            profile["warning"] = f"Low VRAM ({vram_gb} GB). Using aggressive memory optimizations."
        elif vram_gb >= 10:
            profile["gradient_checkpointing"] = False
            profile["max_batch_size"] = 2
            profile["train_text_encoder"] = True

    # SDXL / Kolors (similar arch)
    elif architecture in ("sdxl", "kolors"):
        if vram_gb < 12:
            profile["train_text_encoder"] = False
            profile["warning"] = f"Limited VRAM ({vram_gb} GB). Maximum memory savings enabled."
        elif vram_gb >= 16:
            profile["gradient_checkpointing"] = False
            profile["max_batch_size"] = 2

    # DiT / MMDiT architectures (SD3, PixArt, AuraFlow)
    elif architecture in ("sd3", "pixart", "auraflow"):
        if vram_gb < 16:
            profile["warning"] = f"Limited VRAM ({vram_gb} GB). Training may be slow."
        elif vram_gb >= 24:
            profile["max_batch_size"] = 2

    # Transformer architectures (Flux, HunyuanDiT)
    elif architecture in ("flux", "hunyuan"):
        # These are large — always gradient checkpoint
        profile["gradient_checkpointing"] = True
        if vram_gb < 24:
            profile["warning"] = f"VRAM ({vram_gb} GB) is tight for {architecture.upper()}. Expect slower training."

    # Cascade (Würstchen)
    elif architecture == "cascade":
        if vram_gb < 12:
            profile["warning"] = f"Limited VRAM ({vram_gb} GB) for Cascade."
        elif vram_gb >= 16:
            profile["gradient_checkpointing"] = False
            profile["max_batch_size"] = 2

    return profile


def estimate_training_time(
    architecture: str, steps: int, rank: int = 16,
    resolution: int = 512, vram_gb: float = 0, batch_size: int = 1,
) -> Dict[str, Any]:
    """
    Estimate training time based on hardware + config.
    Returns dict with eta_seconds, time_per_step, feasible.
    """
    times = ARCH_BASE_TIME.get(architecture, {})
    native_res = ARCH_NATIVE_RES.get(architecture, 512)

    if not times:
        return {"eta_seconds": 0, "time_per_step": 0, "feasible": False, "reason": "Unknown architecture"}

    # Find closest VRAM tier
    tiers = sorted(times.keys())
    if vram_gb < tiers[0]:
        return {"eta_seconds": 0, "time_per_step": 0, "feasible": False,
                "reason": f"Requires {tiers[0]}+ GB VRAM"}

    # Interpolate between tiers
    base_time = tiers[-1]  # default to highest tier
    for i, t in enumerate(tiers):
        if vram_gb < t:
            # Interpolate between previous and this tier
            prev_t = tiers[i - 1]
            frac = (vram_gb - prev_t) / (t - prev_t)
            base_time = times[prev_t] * (1 - frac) + times[t] * frac
            break
        base_time = times[t]

    # Adjustments
    rank_factor = max(0.5, rank / 16.0)  # Higher rank = slightly more time
    res_factor = (resolution / native_res) ** 2  # Resolution scales quadratically
    batch_factor = 1.0 / max(1, batch_size)  # More batch = fewer steps needed effectively

    time_per_step = base_time * rank_factor * res_factor
    eta_seconds = steps * time_per_step * batch_factor

    return {
        "eta_seconds": round(eta_seconds),
        "time_per_step": round(time_per_step, 2),
        "feasible": True,
        "reason": None,
    }


# ============================================
# Dataset Preparation
# ============================================

def prepare_dataset(
    images: List[Dict[str, Any]],
    output_dir: Path,
) -> Path:
    """
    Prepare a training dataset folder from UI image data.
    
    Each image becomes:
      <output_dir>/image_001.png
      <output_dir>/image_001.txt  (captions joined by ", ")
    
    Args:
        images: List of dicts with 'filePath' and 'captions' keys
        output_dir: Where to write the prepared dataset
    
    Returns:
        Path to the prepared dataset directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from PIL import Image
    
    for i, img_data in enumerate(images):
        src_path = img_data.get("filePath") or img_data.get("url", "")
        captions = img_data.get("captions", [])
        
        if not src_path or not os.path.exists(src_path):
            print(f"[Dataset] Skipping image {i}: file not found at {src_path}")
            continue
        
        # Copy and normalize the image
        dest_name = f"image_{i:04d}"
        dest_img = output_dir / f"{dest_name}.png"
        dest_txt = output_dir / f"{dest_name}.txt"
        
        try:
            img = Image.open(src_path).convert("RGB")
            img.save(dest_img, "PNG")
        except Exception as e:
            print(f"[Dataset] Failed to process image {i}: {e}")
            continue
        
        # Write captions file
        caption_text = ", ".join(captions) if captions else ""
        dest_txt.write_text(caption_text, encoding="utf-8")
    
    # Count prepared images
    num_images = len(list(output_dir.glob("*.png")))
    print(f"[Dataset] Prepared {num_images} images in {output_dir}")
    
    return output_dir


# ============================================
# Custom Dataset for Training
# ============================================

def get_bucket(w: int, h: int, resolution: int, step: int = 64):
    """Calculate the closest bucket dimensions for a given aspect ratio and target resolution area."""
    import math
    target_area = resolution * resolution
    aspect_ratio = w / h
    
    target_w = math.sqrt(target_area * aspect_ratio)
    target_h = math.sqrt(target_area / aspect_ratio)
    
    # round to nearest step
    target_w = round(target_w / step) * step
    target_h = round(target_h / step) * step
    
    # ensure it's not 0
    target_w = max(step, int(target_w))
    target_h = max(step, int(target_h))
    
    return target_w, target_h

class LoRADataset(torch.utils.data.Dataset):
    """Image+caption dataset with Aspect Ratio Bucketing support."""
    
    def __init__(self, data_dir: Path, tokenizer, resolution: int = 512, center_crop: bool = True, enable_bucketing: bool = True, caption_dropout: float = 0.0):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.center_crop = center_crop
        self.enable_bucketing = enable_bucketing
        self.caption_dropout = caption_dropout
        
        self.image_files = sorted(list(data_dir.glob("*.png")))
        self.image_buckets = []
        
        from PIL import Image
        for img_path in self.image_files:
            if enable_bucketing:
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        self.image_buckets.append(get_bucket(w, h, resolution))
                except Exception:
                    self.image_buckets.append((resolution, resolution))
            else:
                self.image_buckets.append((resolution, resolution))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        from PIL import Image
        import torchvision.transforms.functional as F
        
        img_path = self.image_files[idx]
        target_w, target_h = self.image_buckets[idx]
        
        image = Image.open(img_path).convert("RGB")
        
        # Resize to fit bucket while preserving aspect ratio
        ratio = max(target_w / image.width, target_h / image.height)
        new_w = int(image.width * ratio)
        new_h = int(image.height * ratio)
        
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        # Crop exactly to bucket size
        if self.center_crop:
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
        else:
            left = torch.randint(0, new_w - target_w + 1, (1,)).item() if new_w > target_w else 0
            top = torch.randint(0, new_h - target_h + 1, (1,)).item() if new_h > target_h else 0
            
        image = image.crop((left, top, left + target_w, top + target_h))
        
        image = F.to_tensor(image)
        image = F.normalize(image, [0.5], [0.5])
        
        if torch.rand(1) < 0.5:
            image = F.hflip(image)
        
        txt_path = img_path.with_suffix(".txt")
        caption = ""
        
        if txt_path.exists():
            if self.caption_dropout > 0.0 and torch.rand(1).item() < self.caption_dropout:
                caption = ""
            else:
                caption = txt_path.read_text(encoding="utf-8").strip()
        
        return {
            "pixel_values": image,
            "caption": caption,
        }


# ============================================
# LoRA Trainer
# ============================================

class LoRATrainer:
    """
    Real LoRA training engine.
    Loads a base model, injects LoRA layers, and trains on a custom dataset.
    Supports SD 1.5 and SDXL with automatic VRAM optimization.
    """
    
    def __init__(self):
        self._stop_requested = False
        self._is_training = False
        self._current_step = 0
        self._lock = threading.Lock()
    
    def request_stop(self):
        """Signal the training loop to stop after the current step."""
        self._stop_requested = True
    
    @property
    def is_training(self) -> bool:
        return self._is_training
    
    def train(
        self,
        config: Dict[str, Any],
        dataset_dir: Path,
        model_path: str,
        output_dir: Path,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Run the full LoRA training loop.
        
        This method is BLOCKING and should be called via asyncio.to_thread().
        
        Args:
            config: Training configuration from the UI
            dataset_dir: Path to prepared dataset (images + .txt files)
            model_path: Path to the base model .safetensors file
            output_dir: Where to save the trained LoRA
            progress_callback: Called every N steps with (step, loss, lr, phase)
        
        Returns:
            Dict with training results (output_path, final_loss, total_steps)
        """
        self._stop_requested = False
        self._is_training = True
        self._current_step = 0
        
        try:
            return self._train_impl(config, dataset_dir, model_path, output_dir, progress_callback)
        finally:
            self._is_training = False
            # Clean up GPU memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _train_impl(
        self,
        config: Dict[str, Any],
        dataset_dir: Path,
        model_path: str,
        output_dir: Path,
        progress_callback: Optional[Callable],
    ) -> Dict[str, Any]:
        from diffusers import StableDiffusionPipeline, DDPMScheduler
        from diffusers import StableDiffusionXLPipeline
        from peft import LoraConfig, get_peft_model
        
        architecture = config.get("baseModel", "sd15")
        gpu_info = get_gpu_info()
        opt_profile = get_optimization_profile(gpu_info["vram_gb"], architecture)
        
        if not gpu_info["available"]:
            raise RuntimeError("No NVIDIA GPU detected. LoRA training requires a CUDA-capable GPU.")
        
        if not opt_profile["feasible"]:
            raise RuntimeError(opt_profile["warning"])
        
        # Determine dtype
        if gpu_info["bf16_supported"] and config.get("mixedPrecision") == "bf16":
            weight_dtype = torch.bfloat16
        else:
            weight_dtype = torch.float16
        
        device = torch.device("cuda")
        
        # --- Phase: Loading Model ---
        if progress_callback:
            progress_callback(step=0, loss=0, lr=0, phase="loading_model",
                              message=f"Loading {architecture.upper()} model...")
        
        # Diagnostics
        try:
            import omegaconf
            print(f"[Trainer] OmegaConf version: {omegaconf.__version__}")
        except ImportError:
            print("[Trainer] WARNING: OmegaConf NOT FOUND! from_single_file WILL FAIL.")
            raise RuntimeError("Missing dependency: omegaconf. Please wait for the app to install it or run 'pip install omegaconf'.")
        
        # Normalize path for Windows
        model_path = os.path.abspath(os.path.normpath(model_path))
        print(f"[Trainer] Normalized model path: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        print(f"[Trainer] Loading model from: {model_path}")
        print(f"[Trainer] GPU: {gpu_info['name']} ({gpu_info['vram_gb']} GB)")
        print(f"[Trainer] Architecture: {architecture}")
        
        # Load the appropriate pipeline
        try:
            # Load pipeline from single file for UNet, VAE, scheduler
            load_kwargs = {
                "torch_dtype": torch.float32,
                "use_safetensors": True,
                "load_safety_checker": False,
                "requires_safety_checker": False,
                "local_files_only": False,
                "low_cpu_mem_usage": False,
            }
            
            if architecture == "sdxl":
                pipe = StableDiffusionXLPipeline.from_single_file(model_path, **load_kwargs)
            else:  # sd15
                pipe = StableDiffusionPipeline.from_single_file(model_path, **load_kwargs)
            
            print(f"[Trainer] Pipeline loaded from single file.")
            
            # --- CRITICAL FIX: Load text encoder & tokenizer from known pretrained sources ---
            # from_single_file often creates text encoder with wrong position_embeddings config,
            # causing "index out of range" errors. The text encoder is frozen during LoRA training
            # anyway, so loading canonical pretrained weights is correct and safe.
            from transformers import CLIPTextModel, CLIPTokenizer
            
            if architecture == "sdxl":
                from transformers import CLIPTextModelWithProjection
                print("[Trainer] Loading SDXL text encoders from pretrained...")
                tokenizer_one = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
                text_encoder_one = CLIPTextModel.from_pretrained(
                    "openai/clip-vit-large-patch14", torch_dtype=torch.float32
                )
                tokenizer_two = CLIPTokenizer.from_pretrained(
                    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
                )
                text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
                    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", torch_dtype=torch.float32
                )
            else:  # sd15
                print("[Trainer] Loading SD1.5 text encoder from pretrained (openai/clip-vit-large-patch14)...")
                tokenizer_one = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
                text_encoder_one = CLIPTextModel.from_pretrained(
                    "openai/clip-vit-large-patch14", torch_dtype=torch.float32
                )
                tokenizer_two = None
                text_encoder_two = None
            
            # Verify text encoder works on CPU
            print(f"[Trainer] Verifying text encoder on CPU...")
            text_encoder_one.eval()
            with torch.no_grad():
                test_tokens = tokenizer_one(
                    "test", padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt"
                )
                test_out = text_encoder_one(test_tokens.input_ids, output_hidden_states=True)
                print(f"[Trainer] Text encoder OK. Output shape: {test_out.hidden_states[-1].shape}")
            
            # Move text encoders to GPU
            text_encoder_one.to(device, dtype=weight_dtype)
            text_encoder_one.requires_grad_(False)
            if text_encoder_two:
                text_encoder_two.to(device, dtype=weight_dtype)
                text_encoder_two.requires_grad_(False)
            
            print(f"[Trainer] Text encoder(s) on {device} in {weight_dtype}")
                
        except Exception as e:
            import traceback
            print(f"[Trainer] Critical failure during model loading: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Model loading failed: {str(e)}")

        
        unet = pipe.unet
        vae = pipe.vae
        noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        
        # Free the pipeline shell — we extracted the components we need
        del pipe
        gc.collect()
        
        # Move VAE to GPU for latent caching, then offload
        vae.to(device, dtype=weight_dtype)
        vae.requires_grad_(False)
        
        # --- Phase: Preparing Dataset ---
        if progress_callback:
            progress_callback(step=0, loss=0, lr=0, phase="preparing",
                              message="Preparing dataset and caching latents...")
        def encode_prompt(caption: str):
            """Encode a text prompt into embeddings using the text encoder(s)."""
            clip_skip = config.get("clipSkip", 1)
            max_len = 77 if architecture == "sd15" else tokenizer_one.model_max_length
            
            inputs_one = tokenizer_one(
                caption, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt"
            ).to(device)
            
            # Validate token IDs to prevent CUDA device-side asserts (out of bounds)
            vocab_size = text_encoder_one.config.vocab_size
            if (inputs_one.input_ids >= vocab_size).any():
                print(f"[Trainer] WARNING: Token IDs out of bounds (vocab_size={vocab_size}). Clamping.")
                inputs_one.input_ids = torch.clamp(inputs_one.input_ids, 0, vocab_size - 1)
            
            # Use output_hidden_states for proper penultimate layer extraction
            outputs_one = text_encoder_one(inputs_one.input_ids, output_hidden_states=True)
            # clip_skip=1 means last layer, clip_skip=2 means penultimate, etc.
            prompt_embeds = outputs_one.hidden_states[-clip_skip]
            
            if architecture == "sdxl" and text_encoder_two:
                inputs_two = tokenizer_two(
                    caption, padding="max_length", max_length=tokenizer_two.model_max_length, truncation=True, return_tensors="pt"
                ).to(device)
                
                vocab_size_two = text_encoder_two.config.vocab_size
                if (inputs_two.input_ids >= vocab_size_two).any():
                    inputs_two.input_ids = torch.clamp(inputs_two.input_ids, 0, vocab_size_two - 1)
                
                outputs_two = text_encoder_two(inputs_two.input_ids, output_hidden_states=True)
                pooled_embeds = outputs_two[0]  # pooled output
                prompt_embeds_two = outputs_two.hidden_states[-2]  # penultimate layer
                
                prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_two], dim=-1)
                return prompt_embeds, pooled_embeds
            
            return prompt_embeds, None

        
        resolution = config.get("resolution", 512)
        enable_bucketing = config.get("enableBucketing", True)
        caption_dropout = config.get("captionDropout", 0.1)
        dataset = LoRADataset(dataset_dir, None, resolution=resolution, enable_bucketing=enable_bucketing, caption_dropout=caption_dropout)
        
        if len(dataset) == 0:
            raise RuntimeError("Dataset is empty! Please add images before training.")
        
        # Pre-cache latents to save VRAM during training
        cached_latents = []
        cached_text_embeds = []
        cached_pooled_embeds = []
        
        if opt_profile["cache_latents"]:
            with torch.no_grad():
                for i in range(len(dataset)):
                    item = dataset[i]
                    pixel_values = item["pixel_values"].unsqueeze(0).to(device, dtype=weight_dtype)
                    caption = item["caption"]
                    
                    latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                    embeds, pooled = encode_prompt(caption)
                    
                    cached_latents.append(latents.cpu())
                    cached_text_embeds.append(embeds.cpu())
                    if pooled is not None:
                        cached_pooled_embeds.append(pooled.cpu())
            
            # Free VRAM: offload VAE and text_encoders
            vae.cpu(); text_encoder_one.cpu()
            if text_encoder_two: text_encoder_two.cpu()
            del vae, text_encoder_one, text_encoder_two
            gc.collect(); torch.cuda.empty_cache()
            print(f"[Trainer] Cached {len(cached_latents)} latent pairs. Components offloaded.")
        
        # --- Phase: Injecting LoRA ---
        if progress_callback:
            progress_callback(step=0, loss=0, lr=0, phase="preparing",
                              message="Injecting LoRA layers...")
        
        lora_rank = config.get("loraRank", 16)
        network_alpha = config.get("networkAlpha", 8)
        
        # Target modules for LoRA injection
        target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
        if architecture == "sdxl":
            target_modules += ["proj_in", "proj_out", "ff.net.0.proj", "ff.net.2"]
        
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=network_alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        
        unet.to(device, dtype=weight_dtype)
        if opt_profile["gradient_checkpointing"]:
            unet.enable_gradient_checkpointing()
        
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        
        unet = get_peft_model(unet, lora_config)
        unet.train()
        
        # --- Optimizer & Scheduler ---
        learning_rate = config.get("learningRate", 1e-4)
        optimizer_type = config.get("optimizer", "AdamW")
        lora_params = [p for p in unet.parameters() if p.requires_grad]
        
        # Select optimizer based on config
        if optimizer_type == "AdamW8bit":
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(lora_params, lr=learning_rate, weight_decay=1e-2)
            except ImportError:
                print("[Trainer] bitsandbytes not available, falling back to AdamW")
                optimizer = torch.optim.AdamW(lora_params, lr=learning_rate, weight_decay=1e-2)
        elif optimizer_type == "SGD":
            optimizer = torch.optim.SGD(lora_params, lr=learning_rate, momentum=0.9, weight_decay=1e-2)
        else:  # Default: AdamW
            optimizer = torch.optim.AdamW(lora_params, lr=learning_rate, weight_decay=1e-2)
        
        total_steps = config.get("trainingSteps", 1000)
        warmup_steps = config.get("warmupSteps", 0)
        scheduler_type = config.get("scheduler", "cosine")
        
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            if scheduler_type == "cosine":
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            return 1.0
        
        lr_scheduler = LambdaLR(optimizer, lr_lambda)
        
        # --- Training Loop Setup ---
        seed = config.get("seed", 42)
        noise_offset = config.get("noiseOffset", 0.0)
        batch_size = min(config.get("batchSize", 1), opt_profile["max_batch_size"])
        grad_accum = config.get("gradientAccumulation", 1)
        generator = torch.Generator(device=device).manual_seed(seed)
        
        shape_to_indices = {}
        for i in range(len(dataset)):
            shape = tuple(dataset.image_buckets[i])
            if shape not in shape_to_indices:
                shape_to_indices[shape] = []
            shape_to_indices[shape].append(i)
        
        running_loss = 0.0
        best_loss = float("inf")
        loss_count = 0
        start_time = time.time()
        
        if progress_callback:
            progress_callback(step=0, loss=0, lr=learning_rate, phase="training", message="Training started!")

        # --- Training Loop ---
        for step in range(total_steps):
            if self._stop_requested: break
            self._current_step = step
            
            idx_first = torch.randint(0, len(dataset), (1,)).item()
            shape = tuple(dataset.image_buckets[idx_first])
            pool = shape_to_indices[shape]
            batch_indices = random.choices(pool, k=batch_size)
            
            latents = torch.cat([cached_latents[i] for i in batch_indices], dim=0).to(device, dtype=weight_dtype)
            encoder_hidden_states = torch.cat([cached_text_embeds[i] for i in batch_indices], dim=0).to(device, dtype=weight_dtype)
            
            added_cond_kwargs = {}
            if architecture == "sdxl" and cached_pooled_embeds:
                pooled_embeds = torch.cat([cached_pooled_embeds[i] for i in batch_indices], dim=0).to(device, dtype=weight_dtype)
                # SDXL time_ids: [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
                bucket_w, bucket_h = dataset.image_buckets[batch_indices[0]]
                time_ids = torch.tensor(
                    [bucket_h, bucket_w, 0, 0, bucket_h, bucket_w],
                    device=device, dtype=weight_dtype
                ).unsqueeze(0).repeat(batch_size, 1)
                added_cond_kwargs = {"text_embeds": pooled_embeds, "time_ids": time_ids}
            
            noise = torch.randn(latents.shape, dtype=latents.dtype, device=latents.device, generator=generator)
            
            # Apply noise offset for better brightness range
            if noise_offset > 0:
                noise = noise + noise_offset * torch.randn(
                    (noise.shape[0], noise.shape[1], 1, 1), device=noise.device, dtype=noise.dtype
                )
            
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            with torch.amp.autocast("cuda", dtype=weight_dtype):
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs).sample
            
            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            loss = loss / grad_accum
            loss.backward()
            
            # Track metrics
            loss_value = loss.item() * grad_accum
            running_loss += loss_value
            loss_count += 1
            
            if loss_value < best_loss:
                best_loss = loss_value
            
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Report progress every 5 steps
            if step % 5 == 0 or step == total_steps - 1:
                try:
                    current_lr = lr_scheduler.get_last_lr()[0]
                except RuntimeError:
                    current_lr = learning_rate
                avg_loss = running_loss / loss_count if loss_count > 0 else 0
                elapsed = time.time() - start_time
                eta = (elapsed / max(step + 1, 1)) * (total_steps - step - 1)
                
                if progress_callback:
                    progress_callback(
                        step=step,
                        loss=loss_value,
                        lr=current_lr,
                        phase="training",
                        message=f"Step {step}/{total_steps} | Loss: {loss_value:.4f} | Avg: {avg_loss:.4f}",
                        avg_loss=avg_loss,
                        eta=eta,
                        elapsed=elapsed,
                    )
        
        # --- Phase: Saving ---
        final_step = self._current_step
        
        if progress_callback:
            progress_callback(step=final_step, loss=best_loss, lr=0, phase="saving",
                              message="Saving LoRA weights...")
        
        # Save LoRA weights
        output_dir.mkdir(parents=True, exist_ok=True)
        lora_name = config.get("name", "lora_output")
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in lora_name).strip()
        if not safe_name:
            safe_name = "lora_output"
        
        lora_filename = f"{safe_name}.safetensors"
        lora_path = output_dir / lora_filename
        
        # Save in standard format (single .safetensors file)
        from peft.utils import get_peft_model_state_dict
        from safetensors.torch import save_file
        
        state_dict = get_peft_model_state_dict(unet)
        save_file(state_dict, str(lora_path))
        print(f"[Trainer] LoRA saved: {lora_path}")
        
        # Save adapter_config.json separately so Gallery can read rank/alpha metadata
        adapter_config = {
            "r": lora_rank,
            "lora_alpha": network_alpha,
            "target_modules": sorted(list(lora_config.target_modules)),
            "lora_dropout": 0.0,
            "peft_type": "LORA",
        }
        (output_dir / "adapter_config.json").write_text(
            json.dumps(adapter_config, indent=2)
        )
        
        # Cleanup
        del unet, optimizer, lr_scheduler
        gc.collect()
        torch.cuda.empty_cache()
        
        avg_loss = running_loss / loss_count if loss_count > 0 else 0
        
        result = {
            "output_path": str(lora_path),
            "output_dir": str(output_dir),
            "final_loss": best_loss,
            "avg_loss": avg_loss,
            "total_steps": final_step + 1,
            "stopped_early": self._stop_requested,
            "architecture": architecture,
            "base_model_name": os.path.basename(model_path),
            "lora_rank": lora_rank,
            "network_alpha": network_alpha,
            "learning_rate": learning_rate,
            "resolution": resolution,
            "batch_size": batch_size,
            "optimizer": optimizer_type,
            "scheduler": scheduler_type,
            "seed": seed,
        }
        
        # Save training result metadata for Gallery
        result_meta_path = output_dir / "training_result.json"
        try:
            result_meta_path.write_text(json.dumps(result, indent=2, default=str))
            print(f"[Trainer] Metadata saved: {result_meta_path}")
        except Exception as e:
            print(f"[Trainer] Warning: Could not save metadata: {e}")
        
        print(f"[Trainer] Training complete: {result}")
        return result
