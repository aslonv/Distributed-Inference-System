"""
Main orchestrator managing worker nodes and inference requests
"""

import asyncio
import uuid
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from collections import deque
import random
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import aiohttp
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Configuration ====================

class Config:
    WORKER_TIMEOUT = 30  # seconds
    HEALTH_CHECK_INTERVAL = 5  
    MAX_RETRIES = 3
    RETRY_DELAY = 1  
    
    BATCH_SIZE = 10
    QUEUE_SIZE = 1000
    BATCH_TIMEOUT = 2  # seconds
    
    LOAD_BALANCE_STRATEGY = "least_loaded"  # "round_robin", "random", "least_loaded"
    
    METRICS_WINDOW = 60  # seconds
    LOG_LEVEL = "INFO"

# ==================== Data Models ====================

class Priority(str, Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

class InferenceRequest(BaseModel):
    data: str  # Base64 encoded image or text
    model_type: str = "any"  # "mobilenet", "resnet18", "efficientnet", "any"
    priority: Priority = Priority.NORMAL
    request_id: Optional[str] = None

class BatchInferenceRequest(BaseModel):
    requests: List[InferenceRequest]

class InferenceResponse(BaseModel):
    request_id: str
    status: str
    model: Optional[str] = None
    worker_id: Optional[str] = None
    predictions: Optional[List[Dict]] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    retries: int = 0

class WorkerInfo(BaseModel):
    id: str
    url: str
    model: str
    status: str = "unknown"
    last_heartbeat: float = 0
    load: float = 0
    processed: int = 0
    errors: int = 0
    avg_latency: float = 0

# ==================== Worker Management ====================

class WorkerManager:
    def __init__(self):
        self.workers: Dict[str, WorkerInfo] = {}
        self.round_robin_index = 0
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=Config.WORKER_TIMEOUT)
        )
        
    async def cleanup(self):
        """Cleanup HTTP session"""
        if self.session:
            await self.session.close()
    
    def register_worker(self, worker_id: str, url: str, model: str):
        """Register a new worker"""
        self.workers[worker_id] = WorkerInfo(
            id=worker_id,
            url=url,
            model=model,
            status="initializing",
            last_heartbeat=time.time()
        )
        logger.info(f"Registered worker {worker_id} with model {model}")
        
    async def health_check(self, worker_id: str) -> bool:
        """Check health of a specific worker"""
        if worker_id not in self.workers:
            return False
            
        worker = self.workers[worker_id]
        try:
            async with self.session.get(f"{worker.url}/health", timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    worker.status = "healthy"
                    worker.last_heartbeat = time.time()
                    worker.load = data.get("load", 0)
                    return True
        except Exception as e:
            logger.warning(f"Health check failed for {worker_id}: {e}")
            worker.status = "unhealthy"
            
        return False
    
    async def monitor_workers(self):
        """Continuously monitor worker health"""
        while True:
            for worker_id in list(self.workers.keys()):
                await self.health_check(worker_id)
            await asyncio.sleep(Config.HEALTH_CHECK_INTERVAL)
    
    def get_available_workers(self, model_type: str = "any") -> List[WorkerInfo]:
        """Get list of available workers for a model type"""
        available = []
        for worker in self.workers.values():
            if worker.status == "healthy":
                if model_type == "any" or worker.model == model_type:
                    available.append(worker)
        return available
    
    def select_worker(self, model_type: str = "any") -> Optional[WorkerInfo]:
        """Select a worker based on load balancing strategy"""
        available = self.get_available_workers(model_type)
        if not available:
            return None
            
        strategy = Config.LOAD_BALANCE_STRATEGY
        
        if strategy == "round_robin":
            worker = available[self.round_robin_index % len(available)]
            self.round_robin_index += 1
            return worker
            
        elif strategy == "random":
            return random.choice(available)
            
        elif strategy == "least_loaded":
            return min(available, key=lambda w: w.load)
            
        return available[0]

# ==================== Request Queue ====================

class RequestQueue:
    def __init__(self, max_size: int = 1000):
        self.high_priority = deque()
        self.normal_priority = deque()
        self.low_priority = deque()
        self.max_size = max_size
        self.processing = {}
        
    def add(self, request: InferenceRequest) -> bool:
        """Add request to appropriate priority queue"""
        if self.size() >= self.max_size:
            return False
            
        if not request.request_id:
            request.request_id = f"req_{uuid.uuid4().hex[:8]}"
            
        if request.priority == Priority.HIGH:
            self.high_priority.append(request)
        elif request.priority == Priority.LOW:
            self.low_priority.append(request)
        else:
            self.normal_priority.append(request)
            
        return True
    
    def get_batch(self, batch_size: int) -> List[InferenceRequest]:
        """Get a batch of requests prioritizing high priority"""
        batch = []
        
        # High priority
        while self.high_priority and len(batch) < batch_size:
            batch.append(self.high_priority.popleft())
            
        # Normal priority
        while self.normal_priority and len(batch) < batch_size:
            batch.append(self.normal_priority.popleft())
            
        # Low priority
        while self.low_priority and len(batch) < batch_size:
            batch.append(self.low_priority.popleft())
            
        return batch
    
    def size(self) -> int:
        """Get total queue size"""
        return len(self.high_priority) + len(self.normal_priority) + len(self.low_priority)

# ==================== Metrics Collection ====================

class MetricsCollector:
    def __init__(self):
        self.requests_total = 0
        self.requests_success = 0
        self.requests_failed = 0
        self.latencies = deque(maxlen=1000)
        self.errors = deque(maxlen=100)
        self.start_time = time.time()
        
    def record_request(self, success: bool, latency: float, error: str = None):
        """Record request metrics"""
        self.requests_total += 1
        if success:
            self.requests_success += 1
            self.latencies.append(latency)
        else:
            self.requests_failed += 1
            if error:
                self.errors.append({
                    "timestamp": time.time(),
                    "error": error
                })
                
    def get_stats(self) -> Dict:
        """Get current statistics"""
        uptime = time.time() - self.start_time
        latencies_list = list(self.latencies)
        
        stats = {
            "uptime_seconds": uptime,
            "requests_total": self.requests_total,
            "requests_success": self.requests_success,
            "requests_failed": self.requests_failed,
            "success_rate": self.requests_success / max(1, self.requests_total),
            "queue_size": request_queue.size(),
            "workers_active": len([w for w in worker_manager.workers.values() if w.status == "healthy"])
        }
        
        if latencies_list:
            latencies_sorted = sorted(latencies_list)
            stats.update({
                "latency_p50": latencies_sorted[len(latencies_sorted)//2],
                "latency_p99": latencies_sorted[int(len(latencies_sorted)*0.99)],
                "latency_avg": sum(latencies_list) / len(latencies_list)
            })
            
        return stats

# ==================== Inference Engine ====================

class InferenceEngine:
    def __init__(self, worker_manager: WorkerManager, metrics: MetricsCollector):
        self.worker_manager = worker_manager
        self.metrics = metrics
        
    async def process_request(self, request: InferenceRequest) -> InferenceResponse:
        """Process a single inference request with retries"""
        start_time = time.time()
        retries = 0
        last_error = None
        
        while retries <= Config.MAX_RETRIES:
            worker = self.worker_manager.select_worker(request.model_type)
            
            if not worker:
                last_error = "No available workers"
                await asyncio.sleep(Config.RETRY_DELAY * (2 ** retries))
                retries += 1
                continue
                
            try:
                async with self.worker_manager.session.post(
                    f"{worker.url}/inference",
                    json={"data": request.data, "request_id": request.request_id},
                    timeout=Config.WORKER_TIMEOUT
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        latency = (time.time() - start_time) * 1000
                        
                        worker.processed += 1
                        worker.avg_latency = (worker.avg_latency * 0.9 + latency * 0.1)
                        
                        self.metrics.record_request(True, latency)
                        
                        return InferenceResponse(
                            request_id=request.request_id,
                            status="success",
                            model=worker.model,
                            worker_id=worker.id,
                            predictions=result.get("predictions"),
                            latency_ms=latency,
                            retries=retries
                        )
                    else:
                        last_error = f"Worker returned {resp.status}"
                        
            except asyncio.TimeoutError:
                last_error = "Request timeout"
                worker.errors += 1
                worker.status = "unhealthy"
            except Exception as e:
                last_error = str(e)
                worker.errors += 1

            await asyncio.sleep(Config.RETRY_DELAY * (2 ** retries))
            retries += 1

        latency = (time.time() - start_time) * 1000
        self.metrics.record_request(False, latency, last_error)
        
        return InferenceResponse(
            request_id=request.request_id,
            status="failed",
            error=last_error,
            latency_ms=latency,
            retries=retries
        )
    
    async def process_batch(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Process multiple requests concurrently"""
        tasks = [self.process_request(req) for req in requests]
        return await asyncio.gather(*tasks)

# ==================== Global Instances ====================

worker_manager = WorkerManager()
request_queue = RequestQueue(Config.QUEUE_SIZE)
metrics_collector = MetricsCollector()
inference_engine = InferenceEngine(worker_manager, metrics_collector)

# ==================== FastAPI Application ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    await worker_manager.initialize()
    
    # Register default workers (they will auto-register when they start)
    worker_manager.register_worker("worker_8001", "http://localhost:8001", "mobilenet")
    worker_manager.register_worker("worker_8002", "http://localhost:8002", "resnet18")
    worker_manager.register_worker("worker_8003", "http://localhost:8003", "efficientnet")
    
    # Start background tasks
    asyncio.create_task(worker_manager.monitor_workers())
    asyncio.create_task(batch_processor())
    
    logger.info("Coordinator service started")
    
    yield
    
    # Shutdown
    await worker_manager.cleanup()
    logger.info("Coordinator service stopped")

app = FastAPI(
    title="Distributed Inference Coordinator",
    version="1.0.0",
    lifespan=lifespan
)

# ==================== Background Tasks ====================

async def batch_processor():
    """Process queued requests in batches"""
    while True:
        if request_queue.size() > 0:
            batch = request_queue.get_batch(Config.BATCH_SIZE)
            if batch:
                asyncio.create_task(inference_engine.process_batch(batch))
        await asyncio.sleep(Config.BATCH_TIMEOUT)

# ==================== API Endpoints ====================

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """Process a single inference request"""
    if not request.request_id:
        request.request_id = f"req_{uuid.uuid4().hex[:8]}"
        
    # Add to queue for batch processing or process immediately
    if request_queue.size() < Config.BATCH_SIZE:
        # Process immediately if queue is small
        return await inference_engine.process_request(request)
    else:
        # Add to queue
        if not request_queue.add(request):
            raise HTTPException(status_code=503, detail="Queue is full")
        
        # Wait for processing (with timeout)
        timeout = Config.WORKER_TIMEOUT * (Config.MAX_RETRIES + 1)
        start = time.time()
        
        while time.time() - start < timeout:
            if request.request_id in request_queue.processing:
                return request_queue.processing.pop(request.request_id)
            await asyncio.sleep(0.1)
            
        raise HTTPException(status_code=504, detail="Request timeout")

@app.post("/batch_inference")
async def batch_inference(batch: BatchInferenceRequest):
    """Process multiple inference requests"""
    # Add request IDs
    for req in batch.requests:
        if not req.request_id:
            req.request_id = f"req_{uuid.uuid4().hex[:8]}"
    
    # Process batch
    results = await inference_engine.process_batch(batch.requests)
    
    return {"results": results}

@app.get("/health")
async def health():
    """Health check endpoint"""
    stats = metrics_collector.get_stats()
    return {
        "status": "healthy" if stats["workers_active"] > 0 else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "stats": stats
    }

@app.get("/workers/status")
async def workers_status():
    """Get status of all workers"""
    return {
        "workers": [worker.dict() for worker in worker_manager.workers.values()],
        "total_active": len([w for w in worker_manager.workers.values() if w.status == "healthy"]),
        "total_requests": metrics_collector.requests_total
    }

@app.post("/workers/register")
async def register_worker(worker_id: str, url: str, model: str):
    """Register a new worker"""
    worker_manager.register_worker(worker_id, url, model)
    return {"status": "registered", "worker_id": worker_id}

@app.get("/metrics")
async def metrics():
    """Get detailed metrics"""
    return metrics_collector.get_stats()

@app.get("/queue/status")
async def queue_status():
    """Get queue status"""
    return {
        "size": request_queue.size(),
        "high_priority": len(request_queue.high_priority),
        "normal_priority": len(request_queue.normal_priority),
        "low_priority": len(request_queue.low_priority),
        "max_size": request_queue.max_size
    }

# ==================== Main Entry Point ====================

if __name__ == "__main__":
    uvicorn.run(
        "coordinator.app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )