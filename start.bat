@echo off
start cmd /k "venv\Scripts\activate && python coordinator/app.py"
timeout 2
start cmd /k "venv\Scripts\activate && python workers/worker.py --port 8001 --model mobilenet"
start cmd /k "venv\Scripts\activate && python workers/worker.py --port 8002 --model resnet18"
start cmd /k "venv\Scripts\activate && python workers/worker.py --port 8003 --model efficientnet"
start cmd /k "venv\Scripts\activate && streamlit run monitoring/dashboard.py"
echo All services started! Check the opened windows.