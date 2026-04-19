"""
LoRA Training Engine
====================
Real training module using HuggingFace diffusers + peft.
Automatically adapts to available GPU VRAM.
"""

import os
import gc
import time
import uuid
import shutil
import threading
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List

import torch

# ============================================
# GPU Detection & Optimization Profiles
# ============================================

def get_gpu_info() -> Dict[str, Any]:
    """Detect GPU and return optimization profile."""
    info = {
        "available": False,
        "name": "No GPU",
        "vram_gb": 0,
        "vram_bytes": 0,
        "compute_capability": (0, 0),
        "bf16_supported": False,
    }
    
    if not torch.cuda.is_available():
        return info
    
    info["available"] = True
    info["name"] = torch.cuda.get_device_name(0)
    info["vram_bytes"] = torch.cuda.get_device_properties(0).total_mem
    info["vram_gb"] = round(info["vram_bytes"] / (1024 ** 3), 1)
    info["compute_capability"] = torch.cuda.get_device_properties(0).major, torch.cuda.get_device_properties(0).minor
    # bf16 requires compute capability >= 8.0 (Ampere+)
    info["bf16_supported"] = info["compute_capability"][0] >= 8
    
    return info


def get_optimization_profile(vram_gb: float, architecture: str) -> Dict[str, Any]:
    """
    Return optimal training settings based on available VRAM and target architecture.
    
    Profiles:
    - SD 1.5: Works on 6GB+ VRAM
    - SDXL:   Works on 8GB+ VRAM (with optimizations), comfortable at 12GB+
    - Flux:   Requires 16GB+ VRAM minimum
    """
    profile = {
        "gradient_checkpointing": True,
        "mixed_precision": "fp16",
        "enable_xformers": True,
        "gradient_accumulation_steps": 1,
        "max_batch_size": 1,
        "cache_latents": True,  # Pre-encode images to latents to save VRAM during training
        "train_text_encoder": False,
        "feasible": True,
        "warning": None,
    }
    
    if architecture == "sd15":
        if vram_gb < 4:
            profile["feasible"] = False
            profile["warning"] = f"SD 1.5 requires minimum 6 GB VRAM. You have {vram_gb} GB."
        elif vram_gb < 6:
            profile["gradient_checkpointing"] = True
            profile["max_batch_size"] = 1
            profile["train_text_encoder"] = False
            profile["warning"] = f"Low VRAM ({vram_gb} GB). Training with aggressive memory optimizations."
        elif vram_gb < 10:
            profile["gradient_checkpointing"] = True
            profile["max_batch_size"] = 1
        else:
            profile["gradient_checkpointing"] = False
            profile["max_batch_size"] = 2
            profile["train_text_encoder"] = True
            
    elif architecture == "sdxl":
        if vram_gb < 8:
            profile["feasible"] = False
            profile["warning"] = f"SDXL requires minimum 8 GB VRAM. You have {vram_gb} GB."
        elif vram_gb < 12:
            profile["gradient_checkpointing"] = True
            profile["max_batch_size"] = 1
            profile["train_text_encoder"] = False
            profile["warning"] = f"Limited VRAM ({vram_gb} GB) for SDXL. Using maximum memory savings."
        elif vram_gb < 16:
            profile["gradient_checkpointing"] = True
            profile["max_batch_size"] = 1
        else:
            profile["gradient_checkpointing"] = False
            profile["max_batch_size"] = 2
            
    elif architecture == "flux":
        if vram_gb < 16:
            profile["feasible"] = False
            profile["warning"] = f"Flux requires minimum 16 GB VRAM. You have {vram_gb} GB."
        elif vram_gb < 24:
            profile["gradient_checkpointing"] = True
            profile["max_batch_size"] = 1
        else:
            profile["gradient_checkpointing"] = True
            profile["max_batch_size"] = 1
    
    return profile


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

class LoRADataset(torch.utils.data.Dataset):
    """Simple image+caption dataset for LoRA training."""
    
    def __init__(self, data_dir: Path, tokenizer, resolution: int = 512, center_crop: bool = True):
        from torchvision import transforms
        
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.resolution = resolution
        
        # Find all image/caption pairs
        self.image_files = sorted(list(data_dir.glob("*.png")))
        
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        img_path = self.image_files[idx]
        txt_path = img_path.with_suffix(".txt")
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        # Load caption
        caption = ""
        if txt_path.exists():
            caption = txt_path.read_text(encoding="utf-8").strip()
        
        # Tokenize caption
        tokens = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "pixel_values": image,
            "input_ids": tokens.input_ids.squeeze(0),
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
        
        print(f"[Trainer] Loading model from: {model_path}")
        print(f"[Trainer] GPU: {gpu_info['name']} ({gpu_info['vram_gb']} GB)")
        print(f"[Trainer] Architecture: {architecture}")
        print(f"[Trainer] Optimization: gc={opt_profile['gradient_checkpointing']}, "
              f"precision={weight_dtype}, cache_latents={opt_profile['cache_latents']}")
        
        # Load the appropriate pipeline
        if architecture == "sdxl":
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=weight_dtype,
                use_safetensors=True,
            )
        else:  # sd15
            pipe = StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=weight_dtype,
                use_safetensors=True,
            )
        
        unet = pipe.unet
        text_encoder = pipe.text_encoder
        vae = pipe.vae
        tokenizer = pipe.tokenizer
        noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        
        # Move VAE and text_encoder to GPU for latent caching, then offload
        vae.to(device, dtype=weight_dtype)
        text_encoder.to(device, dtype=weight_dtype)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        
        # --- Phase: Preparing Dataset ---
        if progress_callback:
            progress_callback(step=0, loss=0, lr=0, phase="preparing",
                              message="Preparing dataset and caching latents...")
        
        resolution = config.get("resolution", 512)
        dataset = LoRADataset(dataset_dir, tokenizer, resolution=resolution)
        
        if len(dataset) == 0:
            raise RuntimeError("Dataset is empty! Please add images before training.")
        
        print(f"[Trainer] Dataset: {len(dataset)} images at {resolution}x{resolution}")
        
        # Pre-cache latents to save VRAM during training
        cached_latents = []
        cached_text_embeds = []
        
        if opt_profile["cache_latents"]:
            dataloader_cache = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
            
            with torch.no_grad():
                for batch in dataloader_cache:
                    pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
                    input_ids = batch["input_ids"].to(device)
                    
                    latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                    text_embeds = text_encoder(input_ids)[0]
                    
                    cached_latents.append(latents.cpu())
                    cached_text_embeds.append(text_embeds.cpu())
            
            # Free VRAM: offload VAE and text_encoder
            vae.cpu()
            text_encoder.cpu()
            del vae, text_encoder
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[Trainer] Cached {len(cached_latents)} latent pairs. VAE/TextEncoder offloaded.")
        
        # --- Phase: Injecting LoRA ---
        if progress_callback:
            progress_callback(step=0, loss=0, lr=0, phase="preparing",
                              message="Injecting LoRA layers...")
        
        lora_rank = config.get("loraRank", 16)
        network_alpha = config.get("networkAlpha", 8)
        
        # Target modules for LoRA injection
        if architecture == "sdxl":
            target_modules = [
                "to_q", "to_k", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "ff.net.0.proj", "ff.net.2",
            ]
        else:
            target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
        
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=network_alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        
        unet.to(device, dtype=weight_dtype)
        
        if opt_profile["gradient_checkpointing"]:
            unet.enable_gradient_checkpointing()
        
        # Try to enable xformers for memory efficiency
        try:
            unet.enable_xformers_memory_efficient_attention()
            print("[Trainer] xformers memory efficient attention enabled.")
        except Exception:
            print("[Trainer] xformers not available, using default attention.")
        
        # Inject LoRA
        unet = get_peft_model(unet, lora_config)
        unet.train()
        
        trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in unet.parameters())
        print(f"[Trainer] LoRA injected: {trainable_params:,} trainable params / {total_params:,} total "
              f"({100 * trainable_params / total_params:.2f}%)")
        
        # --- Optimizer ---
        learning_rate = config.get("learningRate", 1e-4)
        optimizer_type = config.get("optimizer", "AdamW")
        
        # Collect LoRA parameters
        lora_params = [p for p in unet.parameters() if p.requires_grad]
        
        if optimizer_type == "Prodigy":
            try:
                from prodigyopt import Prodigy
                optimizer = Prodigy(lora_params, lr=1.0, weight_decay=0.01)
            except ImportError:
                print("[Trainer] Prodigy not installed, falling back to AdamW")
                optimizer = torch.optim.AdamW(lora_params, lr=learning_rate, weight_decay=1e-2)
        elif optimizer_type == "DAdaptAdam":
            try:
                from dadaptation import DAdaptAdam
                optimizer = DAdaptAdam(lora_params, lr=1.0, weight_decay=1e-2)
            except ImportError:
                print("[Trainer] DAdaptAdam not installed, falling back to AdamW")
                optimizer = torch.optim.AdamW(lora_params, lr=learning_rate, weight_decay=1e-2)
        else:
            optimizer = torch.optim.AdamW(lora_params, lr=learning_rate, weight_decay=1e-2)
        
        # --- Learning Rate Scheduler ---
        total_steps = config.get("trainingSteps", 1000)
        warmup_steps = config.get("warmupSteps", 0)
        scheduler_type = config.get("scheduler", "cosine")
        
        from torch.optim.lr_scheduler import LambdaLR
        import math
        
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            if scheduler_type == "cosine":
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            elif scheduler_type == "linear":
                return max(0.0, 1.0 - progress)
            elif scheduler_type == "cosine_with_restarts":
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((progress * 3) % 1.0))))
            else:  # constant
                return 1.0
        
        lr_scheduler = LambdaLR(optimizer, lr_lambda)
        
        # --- Training Loop ---
        if progress_callback:
            progress_callback(step=0, loss=0, lr=learning_rate, phase="training",
                              message="Training started!")
        
        seed = config.get("seed", 42)
        batch_size = min(config.get("batchSize", 1), opt_profile["max_batch_size"])
        grad_accum = config.get("gradientAccumulation", 1)
        generator = torch.Generator(device=device).manual_seed(seed)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        running_loss = 0.0
        best_loss = float("inf")
        loss_count = 0
        start_time = time.time()
        
        print(f"[Trainer] Starting training: {total_steps} steps, batch_size={batch_size}, "
              f"lr={learning_rate}, rank={lora_rank}, alpha={network_alpha}")
        
        for step in range(total_steps):
            if self._stop_requested:
                print(f"[Trainer] Stop requested at step {step}")
                break
            
            self._current_step = step
            
            # Get a random training sample
            if opt_profile["cache_latents"]:
                idx = torch.randint(0, len(cached_latents), (1,)).item()
                latents = cached_latents[idx].to(device, dtype=weight_dtype)
                encoder_hidden_states = cached_text_embeds[idx].to(device, dtype=weight_dtype)
            else:
                idx = torch.randint(0, len(dataset), (1,)).item()
                batch = dataset[idx]
                pixel_values = batch["pixel_values"].unsqueeze(0).to(device, dtype=weight_dtype)
                input_ids = batch["input_ids"].unsqueeze(0).to(device)
                
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                    encoder_hidden_states = text_encoder(input_ids)[0]
            
            # Sample noise
            noise = torch.randn_like(latents, generator=generator)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            
            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Predict the noise
            with torch.cuda.amp.autocast(dtype=weight_dtype):
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            # Calculate loss
            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            loss = loss / grad_accum
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Track metrics
            loss_value = loss.item() * grad_accum
            running_loss += loss_value
            loss_count += 1
            
            if loss_value < best_loss:
                best_loss = loss_value
            
            # Report progress every 5 steps
            if step % 5 == 0 or step == total_steps - 1:
                current_lr = lr_scheduler.get_last_lr()[0] * learning_rate if optimizer_type not in ["Prodigy", "DAdaptAdam"] else lr_scheduler.get_last_lr()[0]
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
        lora_name = config.get("name", "lora_output")
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in lora_name).strip()
        if not safe_name:
            safe_name = "lora_output"
        
        lora_filename = f"{safe_name}.safetensors"
        lora_path = output_dir / lora_filename
        
        # Save using peft
        unet.save_pretrained(output_dir)
        
        # Also try to save in the standard kohya format for compatibility
        try:
            from peft.utils import get_peft_model_state_dict
            state_dict = get_peft_model_state_dict(unet)
            from safetensors.torch import save_file
            save_file(state_dict, str(lora_path))
            print(f"[Trainer] LoRA saved: {lora_path}")
        except Exception as e:
            print(f"[Trainer] Warning: Could not save in standard format: {e}")
            # The peft save_pretrained should still work
            lora_path = output_dir / "adapter_model.safetensors"
        
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
        }
        
        print(f"[Trainer] Training complete: {result}")
        return result
