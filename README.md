# Distributed AI Inference System

A scalable, fault-tolerant microservice-based AI inference system that provides distributed processing capabilities with multiple worker nodes, intelligent load balancing, comprehensive monitoring, and robust fault tolerance mechanisms.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [License](#license)

## Overview

The Distributed AI Inference System is a production-ready microservice architecture designed to handle AI model inference at scale. The system demonstrates enterprise-grade patterns including:

- **Microservice Architecture**: Decoupled services with clear responsibilities
- **Load Balancing**: Intelligent request distribution across worker nodes
- **Fault Tolerance**: Graceful handling of worker failures and network issues
- **Real-time Monitoring**: Comprehensive metrics and dashboard
- **Horizontal Scaling**: Easy addition of worker nodes
- **Priority Queuing**: Request prioritization for critical workloads
- **Chaos Engineering**: Built-in failure simulation for resilience testing

## Architecture

The system consists of four main components:

### 1. Coordinator Service (`coordinator/`)
- Central orchestrator managing the entire system
- Routes inference requests to available workers
- Implements load balancing strategies (round-robin, random, least-loaded)
- Handles priority queuing (high, normal, low)
- Provides health monitoring and metrics collection
- RESTful API with OpenAPI/Swagger documentation

### 2. Worker Services (`workers/`)
- Individual inference nodes running AI models
- Supports multiple model types: MobileNet, ResNet18, EfficientNet, SqueezeNet
- Auto-registration with coordinator
- Health status reporting and load metrics
- Chaos engineering capabilities for testing failure scenarios

### 3. Monitoring Dashboard (`monitoring/`)
- Real-time Streamlit-based web dashboard
- System health and performance metrics
- Request latency and success rate visualization
- Worker status and load distribution
- Interactive testing capabilities

### 4. Testing Suite (`tests/`)
- Comprehensive load testing framework
- Resilience and fault tolerance testing
- Performance benchmarking tools
- Automated test scenarios

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Apps   │    │  Load Balancer  │    │   Monitoring    │
│                 │    │                 │    │   Dashboard     │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Coordinator Service                          │
│                      (Port 8000)                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐  │
│  │Load Balancer│ │Queue Manager│ │Health Check │ │Metrics  │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────┘  │
└─────────────┬───────────────────────┬─────────────────┬───────┘
              │                       │                 │
              ▼                       ▼                 ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Worker Node   │    │   Worker Node   │    │   Worker Node   │
│   (Port 8001)   │    │   (Port 8002)   │    │   (Port 8003)   │
│                 │    │                 │    │                 │
│   MobileNet     │    │   ResNet18      │    │  EfficientNet   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Features

### Core Capabilities
- **Multi-Model Support**: MobileNet, ResNet18, EfficientNet, SqueezeNet
- **Intelligent Load Balancing**: Round-robin, random, and least-loaded strategies
- **Priority Queue System**: High, normal, and low priority request handling
- **Batch Processing**: Efficient batch inference processing
- **Health Monitoring**: Continuous health checks and status reporting
- **Fault Recovery**: Automatic worker failure detection and recovery
- **Metrics Collection**: Comprehensive performance and operational metrics

### Advanced Features
- **Chaos Engineering**: Built-in failure simulation for resilience testing
- **Request Retry Logic**: Configurable retry mechanisms with exponential backoff
- **Graceful Degradation**: System continues operating during partial failures
- **Hot Reloading**: Dynamic worker registration and deregistration
- **API Documentation**: Interactive OpenAPI/Swagger documentation
- **Real-time Dashboard**: Live system monitoring and testing interface

### Development Features
- **VS Code Integration**: Pre-configured debug settings and compound launches
- **Automated Testing**: Load testing, resilience testing, and performance benchmarks
- **Docker Support**: Complete containerization with docker-compose
- **Make Commands**: Simplified system management and automation
- **Logging**: Structured logging with configurable levels

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 8GB+ RAM recommended (for running multiple models)
- **Storage**: 2GB+ free disk space
- **OS**: Linux, macOS, or Windows with WSL2

### Development Environment (Recommended)
- **VS Code**: For integrated debugging and development
- **Docker**: For containerized deployment (optional)
- **Git**: For version control

### Python Dependencies
All dependencies are managed via `requirements.txt`:
- FastAPI & Uvicorn (Web framework and ASGI server)
- PyTorch & TorchVision (AI/ML framework)
- Streamlit & Plotly (Monitoring dashboard)
- aiohttp & httpx (Async HTTP clients)
- pytest (Testing framework)

## Installation

### Method 1: Automated Setup (Recommended)

1. **Clone the repository**:
```bash
git clone https://github.com/aslonv/distributed-inference-system.git
cd distributed-inference-system
```

2. **Run the setup script**:
```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
- Check Python installation
- Create virtual environment (optional)
- Install all dependencies
- Verify system compatibility
- Create startup scripts

### Method 2: Manual Setup

1. **Clone and navigate**:
```bash
git clone https://github.com/aslonv/distributed-inference-system.git
cd distributed-inference-system
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Method 3: Using Make Commands

```bash
# Complete system setup
make setup

# Or install dependencies only
make install
```

## Quick Start

### Option 1: Using Make Commands (Recommended)

```bash
# Start all services
make start

# Check system health
make health

# Open monitoring dashboard
make monitor

# Run a quick demo
make quick-demo
```

### Option 2: Using Scripts

```bash
# Start all services
./start_all.sh

# Or run services individually in separate terminals
python coordinator/app.py
python workers/worker.py --port 8001 --model mobilenet
python workers/worker.py --port 8002 --model resnet18
python workers/worker.py --port 8003 --model efficientnet
streamlit run monitoring/dashboard.py
```

### Option 3: VS Code Integration

1. Open the project in VS Code
2. Install Python extension if not already installed
3. Press `F5` and select "Full System" configuration
4. All services will start automatically in the integrated terminal

### Verify Installation

After starting the services, verify everything is working:

1. **API Documentation**: http://localhost:8000/docs
2. **Coordinator Health**: http://localhost:8000/health
3. **Worker Status**: http://localhost:8000/workers/status
4. **Monitoring Dashboard**: http://localhost:8501

## Usage

### Basic Inference Request

Send a POST request to the coordinator:

```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "data": "base64_encoded_image_data",
    "model_type": "mobilenet",
    "priority": "normal"
  }'
```

### Python Client Example

```python
import requests
import base64

# Prepare image data
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Send inference request
response = requests.post(
    "http://localhost:8000/inference",
    json={
        "data": image_data,
        "model_type": "any",  # or specific: mobilenet, resnet18, efficientnet
        "priority": "high"    # high, normal, or low
    }
)

result = response.json()
print(f"Predictions: {result['predictions']}")
print(f"Processed by: {result['worker_id']}")
```

### Batch Processing

```bash
curl -X POST http://localhost:8000/batch_inference \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"data": "image1_base64", "model_type": "mobilenet"},
      {"data": "image2_base64", "model_type": "resnet18"},
      {"data": "image3_base64", "model_type": "any"}
    ]
  }'
```

### Interactive Demo

Run the comprehensive demo to see all system capabilities:

```bash
# Interactive demo with all features
make demo

# Or run directly
python demo.py
```

The demo includes:
1. System health verification
2. Basic inference testing
3. Load balancing demonstration
4. Fault tolerance testing
5. Performance benchmarking

## API Reference

### Coordinator API Endpoints

#### POST /inference
Submit a single inference request.

**Request Body:**
```json
{
  "data": "base64_encoded_image",
  "model_type": "any|mobilenet|resnet18|efficientnet",
  "priority": "high|normal|low",
  "request_id": "optional_custom_id"
}
```

**Response:**
```json
{
  "request_id": "req_12345678",
  "status": "success|failed",
  "predictions": [
    {"class": "class_name", "confidence": 0.95}
  ],
  "worker_id": "worker_8001",
  "processing_time": 0.123,
  "timestamp": "2024-01-01T10:00:00Z"
}
```

#### POST /batch_inference
Submit multiple inference requests.

**Request Body:**
```json
{
  "requests": [
    {"data": "base64_image1", "model_type": "mobilenet"},
    {"data": "base64_image2", "model_type": "resnet18"}
  ]
}
```

#### GET /health
System health status.

**Response:**
```json
{
  "status": "healthy|degraded|unhealthy",
  "timestamp": "2024-01-01T10:00:00Z",
  "stats": {
    "workers_active": 3,
    "requests_total": 1234,
    "requests_success": 1200,
    "requests_failed": 34,
    "avg_latency": 0.15
  }
}
```

#### GET /workers/status
Worker node information.

**Response:**
```json
{
  "workers": [
    {
      "id": "worker_8001",
      "url": "http://localhost:8001",
      "model": "mobilenet",
      "status": "healthy|unhealthy",
      "load": 0.65,
      "last_heartbeat": "2024-01-01T10:00:00Z"
    }
  ],
  "total_active": 3,
  "total_requests": 1234
}
```

#### GET /metrics
System performance metrics.

**Response:**
```json
{
  "requests_total": 1234,
  "requests_success": 1200,
  "requests_failed": 34,
  "workers_active": 3,
  "avg_latency": 0.15,
  "queue_size": 0,
  "uptime": 3600
}
```

#### GET /queue/status
Request queue information.

**Response:**
```json
{
  "size": 5,
  "high_priority": 2,
  "normal_priority": 2,
  "low_priority": 1,
  "max_size": 1000
}
```

### Worker API Endpoints

#### POST /inference
Process inference request (internal use).

#### GET /health
Worker health status.

#### GET /metrics
Worker performance metrics.

#### POST /simulate/crash
Simulate worker failure (chaos engineering).

## Configuration

### Coordinator Configuration

Edit `coordinator/app.py` to modify the `Config` class:

```python
class Config:
    WORKER_TIMEOUT = 30           # Worker request timeout (seconds)
    HEALTH_CHECK_INTERVAL = 5     # Health check frequency (seconds)
    MAX_RETRIES = 3               # Maximum retry attempts
    RETRY_DELAY = 1               # Delay between retries (seconds)
    
    BATCH_SIZE = 10               # Maximum batch size
    QUEUE_SIZE = 1000             # Maximum queue size
    BATCH_TIMEOUT = 2             # Batch processing timeout
    
    # Load balancing strategy: round_robin, random, least_loaded
    LOAD_BALANCE_STRATEGY = "least_loaded"
    
    METRICS_WINDOW = 60           # Metrics collection window (seconds)
    LOG_LEVEL = "INFO"            # Logging level
```

### Worker Configuration

Configure workers via command line arguments:

```bash
python workers/worker.py \
  --port 8001 \
  --model mobilenet \
  --chaos            # Enable chaos engineering mode
```

### Environment Variables

You can override configuration using environment variables:

```bash
export LOG_LEVEL=DEBUG
export WORKER_TIMEOUT=60
export MAX_RETRIES=5
```

### Docker Configuration

Modify `docker-compose.yml` for containerized deployment settings:

```yaml
environment:
  - LOG_LEVEL=INFO
  - WORKER_TIMEOUT=30
  - MAX_RETRIES=3
```

## Monitoring

### Real-time Dashboard

Access the monitoring dashboard at http://localhost:8501

**Features:**
- System health overview
- Worker status and load distribution
- Request latency charts
- Success/failure rate pie charts
- Real-time metrics updates
- Interactive load testing tools

### Metrics Collection

The system collects comprehensive metrics:

**System Metrics:**
- Total requests processed
- Success/failure rates
- Average response latency
- Worker availability
- Queue size and depth

**Worker Metrics:**
- Individual worker load
- Processing times
- Error rates
- Model-specific performance

### Health Checks

**Coordinator Health Check:**
```bash
curl http://localhost:8000/health
```

**Worker Health Check:**
```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
```

**System Health via Make:**
```bash
make health
```

### Log Monitoring

**View all service logs:**
```bash
make logs
```

**Individual service logs:**
```bash
# Coordinator logs
tail -f coordinator.log

# Worker logs  
tail -f worker_8001.log
tail -f worker_8002.log
tail -f worker_8003.log
```

## Testing

### Load Testing

Run comprehensive load tests to evaluate system performance:

```bash
# Standard load test
make load-test

# Custom load test
python tests/load_test.py --requests 1000 --concurrent 20 --verbose

# Stress test
make stress-test

# Performance benchmark
make benchmark
```

**Load Test Options:**
```bash
python tests/load_test.py \
  --requests 500 \          # Number of requests
  --concurrent 10 \         # Concurrent connections
  --burst-mode \           # Enable burst testing
  --burst-size 50 \        # Requests per burst
  --burst-interval 5.0 \   # Seconds between bursts
  --timeout 30 \           # Request timeout
  --verbose                # Detailed output
```

### Resilience Testing

Test system fault tolerance:

```bash
# Resilience test suite
make resilience

# Run resilience tests directly
python tests/test_resilience.py
```

**Resilience Test Scenarios:**
- Worker failure simulation
- Network partition testing
- High load stress testing
- Recovery time measurement
- Graceful degradation verification

### Unit Testing

```bash
# Run all tests
make test

# Run specific test files
pytest tests/test_resilience.py -v
pytest tests/load_test.py -v
```

### Interactive Demo

```bash
# Full interactive demonstration
make demo

# Quick demo (automated)
make quick-demo

# Run demo directly
python demo.py
```

The interactive demo showcases:
1. System startup and health verification
2. Basic inference functionality
3. Load balancing across workers
4. Fault tolerance during worker failures
5. Performance under various load conditions

## Docker Deployment

### Quick Docker Start

```bash
# Start all services with Docker
make docker-up

# Stop Docker services
make docker-down
```

### Manual Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Scale workers
docker-compose up -d --scale worker1=2 --scale worker2=2

# Stop services
docker-compose down
```

### Docker Services

The docker-compose configuration includes:

- **coordinator**: Main coordinator service
- **worker1**: MobileNet worker (port 8001)
- **worker2**: ResNet18 worker (port 8002)  
- **worker3**: EfficientNet worker (port 8003)
- **monitoring**: Streamlit dashboard

### Production Deployment

For production deployment:

1. **Update resource limits** in docker-compose.yml:
```yaml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2.0'
    reservations:
      memory: 2G
      cpus: '1.0'
```

2. **Configure environment variables**:
```yaml
environment:
  - LOG_LEVEL=INFO
  - WORKER_TIMEOUT=30
  - MAX_RETRIES=3
```

3. **Set up reverse proxy** (nginx/traefik) for load balancing
4. **Configure monitoring** and log aggregation
5. **Set up backup** and recovery procedures

## Performance

### Benchmarks

On a typical development machine (8GB RAM, 4 cores):

**Throughput:**
- Single worker: ~50 requests/second
- Three workers: ~150 requests/second  
- Peak throughput: ~200 requests/second

**Latency:**
- Average response time: 150ms
- 95th percentile: 300ms
- 99th percentile: 500ms

**Resource Usage:**
- Coordinator: ~100MB RAM
- Each worker: ~800MB RAM (including model)
- Total system: ~2.5GB RAM

### Optimization Tips

**System-Level:**
- Increase worker instances for higher throughput
- Use GPU workers for faster model inference
- Implement request batching for efficiency
- Add Redis for persistent queuing

**Configuration:**
- Tune `BATCH_SIZE` for your workload
- Adjust `WORKER_TIMEOUT` based on model complexity
- Configure load balancing strategy based on usage patterns

**Infrastructure:**
- Deploy workers on separate machines
- Use load balancers for external traffic
- Implement connection pooling
- Add caching for frequent requests

### Monitoring Performance

Use the built-in monitoring tools:

```bash
# Performance benchmark
make benchmark

# System health check
make health

# Real-time dashboard
make monitor
```

## Troubleshooting

### Common Issues

#### Services Won't Start

**Problem**: Port already in use errors
```bash
# Check what's using the ports
lsof -i :8000
lsof -i :8001

# Kill processes if needed
sudo kill -9 <PID>

# Or use different ports
python workers/worker.py --port 8011 --model mobilenet
```

**Problem**: Import errors or missing dependencies
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python version
python --version  # Should be 3.8+
```

#### Workers Not Registering

**Problem**: Workers showing as unhealthy
```bash
# Check coordinator logs
make logs | grep coordinator

# Verify worker health manually
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health

# Check network connectivity
ping localhost
```

#### High Latency or Timeouts

**Problem**: Slow response times
```bash
# Check system resources
htop
df -h

# Increase timeout settings
export WORKER_TIMEOUT=60

# Check for memory issues
free -h
```

#### Dashboard Not Loading

**Problem**: Monitoring dashboard shows errors
```bash
# Check Streamlit service
streamlit run monitoring/dashboard.py --server.port=8501

# Verify coordinator API
curl http://localhost:8000/health

# Clear Streamlit cache
rm -rf ~/.streamlit
```

### Debug Mode

Enable debug logging for detailed information:

```bash
export LOG_LEVEL=DEBUG
python coordinator/app.py
```

Or modify the Config class:
```python
class Config:
    LOG_LEVEL = "DEBUG"
```

### System Reset

If you encounter persistent issues:

```bash
# Stop all services
make stop

# Clean temporary files
make clean

# Reset system state
make reset

# Restart everything
make start
```

### Health Checks

Verify each component:

```bash
# System health
make health

# Individual service health
curl http://localhost:8000/health  # Coordinator
curl http://localhost:8001/health  # Worker 1
curl http://localhost:8002/health  # Worker 2
curl http://localhost:8003/health  # Worker 3
```

### Log Analysis

Check service logs for errors:

```bash
# All logs
make logs

# Specific service logs
tail -f coordinator.log
tail -f worker_8001.log

# Error patterns
grep -i error *.log
grep -i exception *.log
```

## Development

### Setting Up Development Environment

1. **Clone and setup**:
```bash
git clone https://github.com/aslonv/distributed-inference-system.git
cd distributed-inference-system
make setup
```

2. **VS Code configuration**:
- Install Python extension
- Open project folder
- Use provided debug configurations
- Launch with F5 (Full System configuration)

### Code Structure

```
distributed-inference-system/
├── coordinator/           # Central coordinator service
│   └── app.py            # Main coordinator application
├── workers/              # Worker node implementations  
│   └── worker.py         # Worker service with model loading
├── monitoring/           # Real-time monitoring dashboard
│   └── dashboard.py      # Streamlit-based dashboard
├── tests/                # Testing suite
│   ├── load_test.py      # Load testing framework
│   └── test_resilience.py # Resilience testing
├── .vscode/              # VS Code debug configurations
│   └── launch.json       # Debug launch settings
├── requirements.txt      # Python dependencies
├── Makefile             # Automation commands
├── docker-compose.yml   # Container orchestration
└── setup.sh            # Automated setup script
```

### Contributing Guidelines

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Write tests** for new functionality
4. **Follow code style** (PEP 8 for Python)
5. **Update documentation** as needed
6. **Submit pull request** with clear description

### Code Quality

**Linting and formatting**:
```bash
# Run linter
make lint

# Type checking
make type-check

# Format code
black coordinator/ workers/ monitoring/ tests/
```

**Testing**:
```bash
# Run all tests
make test

# Run specific tests
pytest tests/load_test.py -v
pytest tests/test_resilience.py -v
```

### Adding New Features

#### Adding a New Model

1. **Update worker.py** to support the new model:
```python
# In ModelManager class
SUPPORTED_MODELS = {
    "mobilenet": models.mobilenet_v2,
    "resnet18": models.resnet18,
    "efficientnet": models.efficientnet_b0,
    "your_model": your_model_loader  # Add here
}
```

2. **Update argument parser**:
```python
parser.add_argument("--model", choices=["mobilenet", "resnet18", "efficientnet", "your_model"])
```

3. **Test the new model**:
```bash
python workers/worker.py --port 8004 --model your_model
```

#### Adding New Load Balancing Strategy

1. **Update coordinator Config**:
```python
LOAD_BALANCE_STRATEGY = "your_strategy"  # Add to options
```

2. **Implement in WorkerManager.select_worker()**:
```python
elif strategy == "your_strategy":
    # Your load balancing logic here
    return selected_worker
```

#### Adding New Monitoring Metrics

1. **Update MetricsCollector** in coordinator/app.py
2. **Add visualization** in monitoring/dashboard.py
3. **Test with** `make monitor`

### Debugging

**VS Code Debugging:**
1. Set breakpoints in your code
2. Press F5 to start debugging
3. Select "Full System" or individual services
4. Use debug console for inspection

**Manual Debugging:**
```bash
# Start services with debug output
export LOG_LEVEL=DEBUG
python coordinator/app.py

# Use Python debugger
python -m pdb coordinator/app.py
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/aslonv/distributed-inference-system) or open an issue.
