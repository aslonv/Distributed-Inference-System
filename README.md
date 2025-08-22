# Distributed AI Inference System 

A scalable, fault-tolerant microservice-based AI inference system that simulates distributed processing with multiple worker nodes, intelligent load balancing, and comprehensive monitoring.

## Features

- Microservice Architecture: Coordinator service managing multiple worker nodes
- Multi-Model Support: Workers run different models (MobileNet, ResNet18, EfficientNet)
- Fault Tolerance: Automatic retry mechanism with exponential backoff
- Health Monitoring: Heartbeat-based health checks for all workers
- Load Balancing: Intelligent routing based on worker load and availability
- Request Batching: Efficient batch processing with configurable batch sizes
- Async I/O: Built with FastAPI and asyncio for high performance
- Live Monitoring: Real-time Streamlit dashboard showing system metrics
- Comprehensive Logging: Detailed logging with request tracking
- Chaos Engineering: Built-in failure simulation for testing resilience

## Project Structure
distributed-inference-system/
├── coordinator/
│   ├── app.py              # Main coordinator service
│   ├── models.py           # Data models
│   └── config.py           # Configuration
├── workers/
│   ├── worker.py           # Worker service implementation
│   └── model_loader.py     # Model loading utilities
├── monitoring/
│   ├── dashboard.py        # Streamlit monitoring dashboard
│   └── metrics.py          # Metrics collection
├── tests/
│   ├── load_test.py        # Load testing script
│   └── test_resilience.py  # Resilience testing
├── docker-compose.yml      # Docker orchestration
├── requirements.txt        # Python dependencies
├── setup.sh               # Setup script
└── README.md              # This file


## Quick Start
### Prerequisites
- Python 3.8+
- VS Code
- 8GB+ RAM (for running multiple models)
- Git

### Installation
1. Clone the repository:
```bash
git clone https://github.com/aslonv/distributed-inference-system.git
cd distributed-inference-system
```