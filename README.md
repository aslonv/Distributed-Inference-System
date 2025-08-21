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