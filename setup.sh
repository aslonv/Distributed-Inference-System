#!/bin/bash

set -e  

echo "================================================"
echo "🚀 Distributed Inference System Setup"
echo "================================================"
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' 

print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d " " -f 2 | cut -d "." -f 1,2)
    print_status "Python 3 found (version $PYTHON_VERSION)"
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version | cut -d " " -f 2 | cut -d "." -f 1,2)
    if [[ $PYTHON_VERSION == 3* ]]; then
        print_status "Python 3 found (version $PYTHON_VERSION)"
        PYTHON_CMD="python"
    else
        print_error "Python 3 is required but not found"
        exit 1
    fi
else
    print_error "Python is not installed"
    exit 1
fi

echo ""
echo "Creating project structure..."

mkdir -p coordinator
mkdir -p workers
mkdir -p monitoring
mkdir -p tests
mkdir -p logs
mkdir -p .vscode

print_status "Project directories created"

touch coordinator/__init__.py
touch workers/__init__.py
touch monitoring/__init__.py
touch tests/__init__.py

print_status "Python package files created"

echo ""
echo "Setting up virtual environment..."

if [ -d "venv" ]; then
    print_warning "Virtual environment already exists"
else
    $PYTHON_CMD -m venv venv
    print_status "Virtual environment created"
fi

echo ""
echo "Activating virtual environment..."

if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Unix-like (Linux, macOS)
    source venv/bin/activate
fi

print_status "Virtual environment activated"

echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet
print_status "pip upgraded"

echo ""
echo "Installing dependencies..."
echo "This may take a few minutes..."

pip install -r requirements.txt --quiet

print_status "Dependencies installed"

echo ""
echo "Pre-downloading AI models..."
echo "This may take several minutes on first run..."

$PYTHON_CMD -c "
import torch
import torchvision.models as models
print('Downloading MobileNet...')
models.mobilenet_v2(pretrained=True)
print('Downloading ResNet18...')
models.resnet18(pretrained=True)
print('Downloading EfficientNet...')
models.efficientnet_b0(pretrained=True)
print('Models cached successfully!')
" 2>/dev/null

print_status "Models downloaded and cached"

echo ""
echo "Creating VS Code configuration..."

cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Coordinator",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "coordinator.app:app",
                "--reload",
                "--port", "8000",
                "--log-level", "info"
            ],
            "console": "integratedTerminal",
            "justMyCode": false
        },
        {
            "name": "Worker 1 (MobileNet)",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/workers/worker.py",
            "args": ["--port", "8001", "--model", "mobilenet"],
            "console": "integratedTerminal"
        },
        {
            "name": "Worker 2 (ResNet18)",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/workers/worker.py",
            "args": ["--port", "8002", "--model", "resnet18"],
            "console": "integratedTerminal"
        },
        {
            "name": "Worker 3 (EfficientNet)",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/workers/worker.py",
            "args": ["--port", "8003", "--model", "efficientnet"],
            "console": "integratedTerminal"
        },
        {
            "name": "Monitoring Dashboard",
            "type": "python",
            "request": "launch",
            "module": "streamlit",
            "args": ["run", "monitoring/dashboard.py"],
            "console": "integratedTerminal"
        },
        {
            "name": "Load Test",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/tests/load_test.py",
            "args": ["--requests", "100", "--concurrent", "10"],
            "console": "integratedTerminal"
        }
    ],
    "compounds": [
        {
            "name": "Full System",
            "configurations": [
                "Coordinator",
                "Worker 1 (MobileNet)",
                "Worker 2 (ResNet18)",
                "Worker 3 (EfficientNet)",
                "Monitoring Dashboard"
            ],
            "stopAll": true
        }
    ]
}
EOF

print_status "VS Code configuration created"

echo ""
echo "Creating sample test data..."

$PYTHON_CMD -c "
from PIL import Image
import numpy as np
import base64
import io

# Create a sample image
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
img.save('sample_image.jpg')
print('Sample image created: sample_image.jpg')

# Create base64 encoded version
buffer = io.BytesIO()
img.save(buffer, format='JPEG')
img_str = base64.b64encode(buffer.getvalue()).decode()
with open('sample_image_base64.txt', 'w') as f:
    f.write(img_str)
print('Base64 encoded image saved: sample_image_base64.txt')
"

print_status "Sample test data created"

echo ""
echo "Creating convenience scripts..."

cat > start_all.sh << 'EOF'
#!/bin/bash
echo "Starting Distributed Inference System..."
echo ""

# Start coordinator
echo "Starting Coordinator..."
python coordinator/app.py &
COORD_PID=$!
sleep 2

# Start workers
echo "Starting Worker 1 (MobileNet)..."
python workers/worker.py --port 8001 --model mobilenet &
WORKER1_PID=$!
sleep 1

echo "Starting Worker 2 (ResNet18)..."
python workers/worker.py --port 8002 --model resnet18 &
WORKER2_PID=$!
sleep 1

echo "Starting Worker 3 (EfficientNet)..."
python workers/worker.py --port 8003 --model efficientnet &
WORKER3_PID=$!
sleep 1

# Start monitoring
echo "Starting Monitoring Dashboard..."
streamlit run monitoring/dashboard.py &
MONITOR_PID=$!

echo ""
echo "================================================"
echo "System started successfully!"
echo "================================================"
echo ""
echo "Services:"
echo "  • Coordinator API: http://localhost:8000"
echo "  • Worker 1: http://localhost:8001"
echo "  • Worker 2: http://localhost:8002"
echo "  • Worker 3: http://localhost:8003"
echo "  • Monitoring Dashboard: http://localhost:8501"
echo ""
echo "Process IDs:"
echo "  • Coordinator: $COORD_PID"
echo "  • Worker 1: $WORKER1_PID"
echo "  • Worker 2: $WORKER2_PID"
echo "  • Worker 3: $WORKER3_PID"
echo "  • Monitor: $MONITOR_PID"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Save PIDs to file for stop script
echo "$COORD_PID" > .pids
echo "$WORKER1_PID" >> .pids
echo "$WORKER2_PID" >> .pids
echo "$WORKER3_PID" >> .pids
echo "$MONITOR_PID" >> .pids

# Wait for interrupt
trap "echo 'Stopping all services...'; kill $COORD_PID $WORKER1_PID $WORKER2_PID $WORKER3_PID $MONITOR_PID; rm .pids; exit" INT
wait
EOF

chmod +x start_all.sh

cat > stop_all.sh << 'EOF'
#!/bin/bash
echo "Stopping all services..."

if [ -f .pids ]; then
    while read pid; do
        kill $pid 2>/dev/null
    done < .pids
    rm .pids
    echo "All services stopped"
else
    echo "No running services found"
    # Try to kill by port
    lsof -ti:8000,8001,8002,8003,8501 | xargs kill 2>/dev/null
fi
EOF

chmod +x stop_all.sh

print_status "Convenience scripts created"

cat > run_test.sh << 'EOF'
#!/bin/bash
echo "Running system test..."
python tests/load_test.py --requests 50 --concurrent 5
EOF

chmod +x run_test.sh

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Quick Start Guide:"
echo ""
echo "1. Start the entire system:"
echo "   ${GREEN}./start_all.sh${NC}"
echo ""
echo "2. Or start services individually:"
echo "   Terminal 1: ${GREEN}python coordinator/app.py${NC}"
echo "   Terminal 2: ${GREEN}python workers/worker.py --port 8001 --model mobilenet${NC}"
echo "   Terminal 3: ${GREEN}python workers/worker.py --port 8002 --model resnet18${NC}"
echo "   Terminal 4: ${GREEN}python workers/worker.py --port 8003 --model efficientnet${NC}"
echo "   Terminal 5: ${GREEN}streamlit run monitoring/dashboard.py${NC}"
echo ""
echo "3. Run load test:"
echo "   ${GREEN}./run_test.sh${NC}"
echo ""
echo "4. Access services:"
echo "   • API Documentation: ${YELLOW}http://localhost:8000/docs${NC}"
echo "   • Monitoring Dashboard: ${YELLOW}http://localhost:8501${NC}"
echo ""
echo "5. Stop all services:"
echo "   ${GREEN}./stop_all.sh${NC}"
echo ""
echo "VS Code:"
echo "   Open the project in VS Code and use the debug configurations"
echo "   or press F5 to start the 'Full System' compound configuration"
echo ""
echo "================================================"