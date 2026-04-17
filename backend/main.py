import asyncio
import time
import uuid
import random
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

from typing import Optional

class TrainingConfig(BaseModel):
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

@app.post("/api/training/start")
async def start_training(config: TrainingConfig):
    session_id = str(uuid.uuid4())
    active_training_sessions[session_id] = {
        "status": "preparing",
        "config": config.model_dump(),
        "task": None
    }
    return {"sessionId": session_id}

@app.post("/api/training/stop/{session_id}")
async def stop_training(session_id: str):
    if session_id in active_training_sessions:
        task = active_training_sessions[session_id].get("task")
        if task:
            task.cancel()
        del active_training_sessions[session_id]
    return {"status": "stopped"}

class CaptionRequest(BaseModel):
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
        
        # Generate caption
        inputs = processor(raw_image, return_tensors="pt")
        out = blip_model.generate(**inputs, max_new_tokens=40)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        # Convert sentence into tags for the UI (simple splitting logic)
        tags = [caption] # Keep full sentence as a tag too
        words = [w.replace(',','').strip() for w in caption.split(" ")]
        clean_words = [w for w in words if len(w) > 2 and w not in ["the", "and", "with", "a", "an", "of", "in", "on", "at"]]
        
        # Combine unique items
        return {"tags": list(set(tags + clean_words))}
        
    except Exception as e:
        print("Captioning error:", e)
        return {"tags": ["error"]}

class GenerateRequest(BaseModel):
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
                    config = TrainingConfig(**active_training_sessions[session_id]["config"])
                    task = asyncio.create_task(run_training_simulation(session_id, config, websocket))
                    active_training_sessions[session_id]["task"] = task
    except WebSocketDisconnect:
        active_connections.remove(websocket)

async def run_training_simulation(session_id: str, config: TrainingConfig, websocket: WebSocket):
    total_steps = config.trainingSteps
    
    await websocket.send_json({
        "type": "training_update",
        "data": {
            "phase": "training",
            "totalSteps": total_steps,
            "currentStep": 0,
            "currentLoss": 2.5,
            "eta": total_steps * 0.5 
        }
    })
    
    current_loss = 2.5
    for step in range(0, total_steps, 10):
        await asyncio.sleep(0.5) 
        if session_id not in active_training_sessions:
            break
            
        current_loss = max(0.02, current_loss * 0.98 + (random.random() * 0.04 - 0.02))
        
        # Emit step metrics
        await websocket.send_json({
            "type": "training_step",
            "data": {
                "step": step,
                "loss": current_loss,
                "learningRate": config.learningRate * (1 - step/total_steps),
                "timestamp": int(time.time() * 1000)
            }
        })
        
        # Emit status update
        await websocket.send_json({
            "type": "training_update",
            "data": {
                "phase": "training",
                "currentStep": step,
                "currentLoss": current_loss,
                "eta": (total_steps - step) * 0.5
            }
        })
        
        # Emit log every 50 steps
        if step % 50 == 0:
            await websocket.send_json({
                "type": "training_log",
                "data": {
                    "id": str(uuid.uuid4()),
                    "timestamp": int(time.time() * 1000),
                    "level": "info",
                    "message": f"[Epoch {step//100}/{total_steps//100}] Step {step}/{total_steps} | Loss: {current_loss:.4f} | LR: {config.learningRate:.2e}"
                }
            })

    if session_id in active_training_sessions:
        await websocket.send_json({
            "type": "training_update",
            "data": {"phase": "completed"}
        })
        del active_training_sessions[session_id]
