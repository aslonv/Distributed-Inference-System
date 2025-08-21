"""
Individual worker node that loads and runs AI models
"""

import asyncio
import argparse
import random
import time
import logging
import base64
import io
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Configuration ====================

class WorkerConfig:
    def __init__(self, port: int, model_name: str):
        self.port = port
        self.model_name = model_name
        self.worker_id = f"worker_{port}"
        self.coordinator_url = "http://localhost:8000"
        
        self.chaos_enabled = False
        self.failure_rate = 0.1  # 10% failure rate when chaos enabled
        self.min_delay = 0  # ms
        self.max_delay = 500  # ms
        
        self.max_concurrent = 10
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== Data Models ====================

class InferenceRequest(BaseModel):
    data: str  # Base64 encoded image
    request_id: Optional[str] = None

class InferenceResponse(BaseModel):
    predictions: list
    model: str
    worker_id: str
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    load: float
    requests_processed: int
    uptime_seconds: float

# ==================== Model Management ====================

class ModelManager:
    """Manages AI model loading and inference"""
    
    # ImageNet class labels (top 5 for demo)
    IMAGENET_CLASSES = [
        "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian cat",
        "mountain lion", "lynx", "leopard", "snow leopard", "jaguar",
        "lion", "tiger", "cheetah", "brown bear", "American black bear",
        "golden retriever", "Labrador retriever", "cocker spaniel", "collie", "Border collie"
    ]
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.transform = None
        self.load_model()
        
    def load_model(self):
        """Load the specified model"""
        logger.info(f"Loading model: {self.model_name}")
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        if self.model_name == "mobilenet":
            self.model = models.mobilenet_v2(pretrained=True)
        elif self.model_name == "resnet18":
            self.model = models.resnet18(pretrained=True)
        elif self.model_name == "efficientnet":
            self.model = models.efficientnet_b0(pretrained=True)
        elif self.model_name == "squeezenet":
            self.model = models.squeezenet1_0(pretrained=True)
        else:
            logger.warning(f"Unknown model {self.model_name}, using MobileNet")
            self.model = models.mobilenet_v2(pretrained=True)
            self.model_name = "mobilenet"
      
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model {self.model_name} loaded on {self.device}")
    
    def preprocess_image(self, base64_data: str) -> torch.Tensor:
        """Decode and preprocess base64 image"""
        try:
            image_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
   
            tensor = self.transform(image)
 
            return tensor.unsqueeze(0).to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return torch.randn(1, 3, 224, 224).to(self.device)
    
    def inference(self, input_tensor: torch.Tensor) -> list:
        """Run inference on the input"""
        with torch.no_grad():
            outputs = self.model(input_tensor)

            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top5_prob, top5_idx = torch.topk(probabilities, 5)

            predictions = []
            for i in range(5):
                idx = top5_idx[i].item()
                class_name = self.IMAGENET_CLASSES[idx % len(self.IMAGENET_CLASSES)]
                predictions.append({
                    "class": class_name,
                    "confidence": float(top5_prob[i].item()),
                    "index": idx
                })
            
            return predictions

# ==================== Chaos Engineering ====================

class ChaosMonkey:
    """Simulates failures and delays for resilience testing"""
    
    def __init__(self, config: WorkerConfig):
        self.config = config
        
    async def maybe_fail(self):
        """Randomly fail if chaos is enabled"""
        if self.config.chaos_enabled and random.random() < self.config.failure_rate:
            raise Exception("Chaos monkey induced failure!")
    
    async def maybe_delay(self):
        """Add random delay if chaos is enabled"""
        if self.config.chaos_enabled:
            delay = random.randint(self.config.min_delay, self.config.max_delay) / 1000
            await asyncio.sleep(delay)
    
    def maybe_corrupt_result(self, predictions: list) -> list:
        """Randomly corrupt results if chaos is enabled"""
        if self.config.chaos_enabled and random.random() < self.config.failure_rate / 2:
            for pred in predictions:
                pred["confidence"] = random.random()
        return predictions

# ==================== Worker Service ====================

class WorkerService:
    """Main worker service implementation"""
    
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.model_manager = ModelManager(config.model_name, config.device)
        self.chaos_monkey = ChaosMonkey(config)

        self.start_time = time.time()
        self.requests_processed = 0
        self.current_load = 0
        self.semaphore = asyncio.Semaphore(config.max_concurrent)
        
    async def process_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Process a single inference request"""
        async with self.semaphore:
            start_time = time.time()
            self.current_load += 1
            
            try:
                await self.chaos_monkey.maybe_delay()
                await self.chaos_monkey.maybe_fail()

                input_tensor = self.model_manager.preprocess_image(request.data)

                predictions = self.model_manager.inference(input_tensor)

                predictions = self.chaos_monkey.maybe_corrupt_result(predictions)

                self.requests_processed += 1
                processing_time = (time.time() - start_time) * 1000
                
                return InferenceResponse(
                    predictions=predictions,
                    model=self.config.model_name,
                    worker_id=self.config.worker_id,
                    processing_time_ms=processing_time
                )
                
            finally:
                self.current_load -= 1
    
    def get_health(self) -> HealthResponse:
        """Get current health status"""
        uptime = time.time() - self.start_time
        load = self.current_load / self.config.max_concurrent
        
        return HealthResponse(
            status="healthy",
            model=self.config.model_name,
            device=self.config.device,
            load=load,
            requests_processed=self.requests_processed,
            uptime_seconds=uptime
        )
    
    async def register_with_coordinator(self):
        """Register this worker with the coordinator"""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"{self.config.coordinator_url}/workers/register",
                    params={
                        "worker_id": self.config.worker_id,
                        "url": f"http://localhost:{self.config.port}",
                        "model": self.config.model_name
                    }
                )
                logger.info(f"Registered with coordinator as {self.config.worker_id}")
        except Exception as e:
            logger.warning(f"Failed to register with coordinator: {e}")

# ==================== FastAPI Application ====================

def create_app(config: WorkerConfig) -> FastAPI:
    """Create FastAPI application"""
    
    worker_service = WorkerService(config)
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        await worker_service.register_with_coordinator()
        logger.info(f"Worker {config.worker_id} started on port {config.port}")
        yield
        # Shutdown
        logger.info(f"Worker {config.worker_id} shutting down")
    
    app = FastAPI(
        title=f"Worker Service - {config.model_name}",
        version="1.0.0",
        lifespan=lifespan
    )
    
    @app.post("/inference", response_model=InferenceResponse)
    async def inference(request: InferenceRequest):
        """Process inference request"""
        try:
            return await worker_service.process_inference(request)
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint"""
        return worker_service.get_health()
    
    @app.post("/chaos/enable")
    async def enable_chaos():
        """Enable chaos engineering mode"""
        config.chaos_enabled = True
        return {"status": "chaos enabled"}
    
    @app.post("/chaos/disable")
    async def disable_chaos():
        """Disable chaos engineering mode"""
        config.chaos_enabled = False
        return {"status": "chaos disabled"}
    
    @app.post("/simulate/crash")
    async def simulate_crash():
        """Simulate a worker crash"""
        logger.error("Simulating worker crash!")
        raise HTTPException(status_code=500, detail="Simulated crash")
    
    @app.post("/simulate/timeout")
    async def simulate_timeout():
        """Simulate a timeout"""
        await asyncio.sleep(60)
        return {"status": "timeout simulated"}
    
    return app

# ==================== Main Entry Point ====================

def main():
    """Main entry point for worker service"""
    parser = argparse.ArgumentParser(description="Worker Service")
    parser.add_argument("--port", type=int, default=8001, help="Port to run on")
    parser.add_argument("--model", type=str, default="mobilenet", 
                       choices=["mobilenet", "resnet18", "efficientnet", "squeezenet"],
                       help="Model to load")
    parser.add_argument("--chaos", action="store_true", help="Enable chaos mode")
    
    args = parser.parse_args()

    config = WorkerConfig(args.port, args.model)
    if args.chaos:
        config.chaos_enabled = True
        logger.warning("Chaos engineering mode enabled!")

    app = create_app(config)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.port,
        log_level="info"
    )

if __name__ == "__main__":
    main()