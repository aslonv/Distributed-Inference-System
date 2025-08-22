.PHONY: help setup install start stop clean test demo monitor health docker-up docker-down

help:
	@echo "================================================"
	@echo "Distributed Inference System - Make Commands"
	@echo "================================================"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup       - Complete system setup"
	@echo "  make install     - Install Python dependencies"
	@echo ""
	@echo "System Control:"
	@echo "  make start       - Start all services"
	@echo "  make stop        - Stop all services"
	@echo "  make restart     - Restart all services"
	@echo ""
	@echo "Testing:"
	@echo "  make test        - Run all tests"
	@echo "  make load-test   - Run load test"
	@echo "  make resilience  - Run resilience test"
	@echo "  make demo        - Run interactive demo"
	@echo "  make quick-demo  - Run quick demo"
	@echo ""
	@echo "Monitoring:"
	@echo "  make monitor     - Open monitoring dashboard"
	@echo "  make health      - Check system health"
	@echo "  make logs        - Show service logs"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up   - Start with Docker"
	@echo "  make docker-down - Stop Docker services"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean       - Clean temporary files"
	@echo "  make reset       - Reset system state"
	@echo ""

setup:
	@echo "Setting up Distributed Inference System..."
	@chmod +x setup.sh
	@./setup.sh

install:
	@echo "Installing Python dependencies..."
	@pip install -r requirements.txt
	@echo "Dependencies installed successfully!"

venv:
	@echo "Creating virtual environment..."
	@python -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

start:
	@echo "Starting all services..."
	@chmod +x start_all.sh
	@./start_all.sh

stop:
	@echo "Stopping all services..."
	@chmod +x stop_all.sh
	@./stop_all.sh

restart: stop start
	@echo "Services restarted"

coordinator:
	@echo "Starting Coordinator..."
	@python coordinator/app.py

worker1:
	@echo "Starting Worker 1 (MobileNet)..."
	@python workers/worker.py --port 8001 --model mobilenet

worker2:
	@echo "Starting Worker 2 (ResNet18)..."
	@python workers/worker.py --port 8002 --model resnet18

worker3:
	@echo "Starting Worker 3 (EfficientNet)..."
	@python workers/worker.py --port 8003 --model efficientnet

dashboard:
	@echo "Starting Monitoring Dashboard..."
	@streamlit run monitoring/dashboard.py

test: load-test resilience
	@echo "All tests completed"

load-test:
	@echo "Running load test..."
	@python tests/load_test.py --requests 100 --concurrent 10

load-test-heavy:
	@echo "Running heavy load test..."
	@python tests/load_test.py --requests 1000 --concurrent 50

load-test-burst:
	@echo "Running burst load test..."
	@python tests/load_test.py --requests 500 --burst --burst-size 100 --burst-interval 2

resilience:
	@echo "Running resilience tests..."
	@python tests/test_resilience.py

demo:
	@echo "Starting interactive demo..."
	@python demo.py

quick-demo:
	@echo "Running quick demo..."
	@python demo.py --quick

monitor:
	@echo "Opening monitoring dashboard..."
	@open http://localhost:8501 2>/dev/null || xdg-open http://localhost:8501 2>/dev/null || echo "Open http://localhost:8501 in your browser"

health:
	@echo "Checking system health..."
	@curl -s http://localhost:8000/health | python -m json.tool

workers-status:
	@echo "Checking workers status..."
	@curl -s http://localhost:8000/workers/status | python -m json.tool

metrics:
	@echo "Fetching system metrics..."
	@curl -s http://localhost:8000/metrics | python -m json.tool

queue-status:
	@echo "Checking queue status..."
	@curl -s http://localhost:8000/queue/status | python -m json.tool

logs:
	@echo "Showing recent logs..."
	@tail -n 50 logs/*.log 2>/dev/null || echo "No log files found"


docker-build:
	@echo "Building Docker images..."
	@docker-compose build

docker-up:
	@echo "Starting services with Docker..."
	@docker-compose up -d
	@echo "Services started. Dashboard at http://localhost:8501"

docker-down:
	@echo "Stopping Docker services..."
	@docker-compose down

docker-logs:
	@echo "Showing Docker logs..."
	@docker-compose logs -f

docker-ps:
	@echo "Docker container status..."
	@docker-compose ps


chaos-enable:
	@echo "Enabling chaos mode on all workers..."
	@curl -X POST http://localhost:8001/chaos/enable
	@curl -X POST http://localhost:8002/chaos/enable
	@curl -X POST http://localhost:8003/chaos/enable
	@echo "Chaos mode enabled!"

chaos-disable:
	@echo "Disabling chaos mode on all workers..."
	@curl -X POST http://localhost:8001/chaos/disable
	@curl -X POST http://localhost:8002/chaos/disable
	@curl -X POST http://localhost:8003/chaos/disable
	@echo "Chaos mode disabled!"

simulate-crash:
	@echo "Simulating worker crash..."
	@curl -X POST http://localhost:8001/simulate/crash


clean:
	@echo "Cleaning temporary files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@rm -rf .pytest_cache 2>/dev/null || true
	@rm -rf .coverage 2>/dev/null || true
	@rm -rf htmlcov 2>/dev/null || true
	@rm -rf dist 2>/dev/null || true
	@rm -rf build 2>/dev/null || true
	@rm -rf *.egg-info 2>/dev/null || true
	@echo "Cleanup completed!"

reset: stop clean
	@echo "Resetting system state..."
	@rm -f .pids 2>/dev/null || true
	@rm -f logs/*.log 2>/dev/null || true
	@rm -f resilience_test_results.json 2>/dev/null || true
	@echo "System reset completed!"


format:
	@echo "Formatting Python code..."
	@black coordinator/ workers/ monitoring/ tests/ --line-length 100

lint:
	@echo "Linting Python code..."
	@pylint coordinator/ workers/ monitoring/ tests/ || true

type-check:
	@echo "Type checking..."
	@mypy coordinator/ workers/ monitoring/ tests/ --ignore-missing-imports


benchmark:
	@echo "Running performance benchmark..."
	@python tests/load_test.py --requests 1000 --concurrent 20 --verbose

stress-test:
	@echo "Running stress test..."
	@python tests/load_test.py --requests 5000 --concurrent 100


api-docs:
	@echo "Opening API documentation..."
	@open http://localhost:8000/docs 2>/dev/null || xdg-open http://localhost:8000/docs 2>/dev/null || echo "Open http://localhost:8000/docs in your browser"

test-inference:
	@echo "Testing single inference..."
	@curl -X POST http://localhost:8000/inference \
		-H "Content-Type: application/json" \
		-d '{"data": "test_image_base64", "model_type": "mobilenet", "priority": "normal"}' | python -m json.tool


verify:
	@echo "Verifying installation..."
	@python -c "import torch; print('PyTorch:', torch.__version__)"
	@python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
	@python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
	@python -c "import torchvision; print('Torchvision:', torchvision.__version__)"
	@echo "All packages verified!"


git-init:
	@echo "Initializing git repository..."
	@git init
	@echo "venv/" >> .gitignore
	@echo "__pycache__/" >> .gitignore
	@echo "*.pyc" >> .gitignore
	@echo ".DS_Store" >> .gitignore
	@echo "logs/" >> .gitignore
	@echo ".pids" >> .gitignore
	@git add .
	@git commit -m "Initial commit: Distributed Inference System"
	@echo "Git repository initialized!"


s: start
st: stop
r: restart
m: monitor
h: health
t: test
d: demo
c: clean