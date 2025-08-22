import requests
import base64
import time
import json
import asyncio
import aiohttp
from typing import Dict, List
from datetime import datetime
import sys
import os

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_step(step: int, text: str):
    """Print formatted step"""
    print(f"{Colors.CYAN}Step {step}:{Colors.ENDC} {text}")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

class SystemDemo:
    def __init__(self):
        self.coordinator_url = "http://localhost:8000"
        self.worker_urls = [
            "http://localhost:8001",
            "http://localhost:8002",
            "http://localhost:8003"
        ]
        self.dashboard_url = "http://localhost:8501"
        
    def check_service(self, url: str, name: str) -> bool:
        """Check if a service is running"""
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                print_success(f"{name} is running at {url}")
                return True
        except:
            pass
        print_error(f"{name} is not accessible at {url}")
        return False
    
    def check_system(self) -> bool:
        """Check if all services are running"""
        print_header("SYSTEM STATUS CHECK")
        
        services_ok = True

        if not self.check_service(self.coordinator_url, "Coordinator"):
            services_ok = False

        for i, url in enumerate(self.worker_urls, 1):
            if not self.check_service(url, f"Worker {i}"):
                services_ok = False

        try:
            response = requests.get(self.dashboard_url, timeout=2)
            if response.status_code == 200:
                print_success(f"Dashboard is running at {self.dashboard_url}")
        except:
            print_warning(f"Dashboard is not running (optional)")
        
        return services_ok
    
    def generate_test_image(self) -> str:
        """Generate a base64 encoded test image"""
        from PIL import Image
        import numpy as np
        import io

        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
 
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
    
    def demo_single_inference(self):
        """Demonstrate single inference request"""
        print_header("DEMO 1: SINGLE INFERENCE REQUEST")
        
        print_step(1, "Generating test image...")
        image_data = self.generate_test_image()
        print_success(f"Generated test image (224x224 RGB)")
        
        print_step(2, "Sending inference request...")
        
        request_data = {
            "data": image_data,
            "model_type": "mobilenet",
            "priority": "normal"
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.coordinator_url}/inference",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                latency = (time.time() - start_time) * 1000
                
                print_success(f"Request completed in {latency:.1f}ms")
                print(f"\n{Colors.BOLD}Results:{Colors.ENDC}")
                print(f"  Request ID: {result.get('request_id')}")
                print(f"  Model: {result.get('model')}")
                print(f"  Worker: {result.get('worker_id')}")
                print(f"  Status: {result.get('status')}")
                print(f"  Retries: {result.get('retries')}")
                
                if result.get('predictions'):
                    print(f"\n{Colors.BOLD}Top Predictions:{Colors.ENDC}")
                    for i, pred in enumerate(result['predictions'][:3], 1):
                        print(f"  {i}. {pred['class']}: {pred['confidence']*100:.1f}%")
            else:
                print_error(f"Request failed with status {response.status_code}")
                
        except Exception as e:
            print_error(f"Request failed: {e}")
    
    def demo_batch_inference(self):
        """Demonstrate batch inference"""
        print_header("DEMO 2: BATCH INFERENCE")
        
        print_step(1, "Preparing batch of 5 requests...")
        
        batch_requests = []
        for i in range(5):
            batch_requests.append({
                "data": self.generate_test_image(),
                "model_type": ["mobilenet", "resnet18", "efficientnet", "any"][i % 4],
                "priority": ["high", "normal", "low"][i % 3]
            })
        
        print_success("Batch prepared with different models and priorities")
        
        print_step(2, "Sending batch request...")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.coordinator_url}/batch_inference",
                json={"requests": batch_requests},
                timeout=30
            )
            
            if response.status_code == 200:
                results = response.json()
                latency = (time.time() - start_time) * 1000
                
                print_success(f"Batch completed in {latency:.1f}ms")
                
                print(f"\n{Colors.BOLD}Batch Results:{Colors.ENDC}")
                for i, result in enumerate(results['results'], 1):
                    status = "✅" if result['status'] == 'success' else "❌"
                    print(f"  Request {i}: {status} Model: {result.get('model', 'N/A')}, "
                          f"Latency: {result.get('latency_ms', 0):.1f}ms")
                          
        except Exception as e:
            print_error(f"Batch request failed: {e}")
    
    def demo_load_balancing(self):
        """Demonstrate load balancing across workers"""
        print_header("DEMO 3: LOAD BALANCING")
        
        print_step(1, "Sending 10 requests to observe load balancing...")
        
        worker_counts = {}
        
        for i in range(10):
            try:
                response = requests.post(
                    f"{self.coordinator_url}/inference",
                    json={
                        "data": self.generate_test_image(),
                        "model_type": "any",
                        "priority": "normal"
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    worker = result.get('worker_id', 'unknown')
                    worker_counts[worker] = worker_counts.get(worker, 0) + 1
                    print(f"  Request {i+1}: Processed by {worker}")
                    
            except Exception as e:
                print_error(f"Request {i+1} failed: {e}")
        
        print(f"\n{Colors.BOLD}Load Distribution:{Colors.ENDC}")
        for worker, count in worker_counts.items():
            print(f"  {worker}: {count} requests ({count/10*100:.0f}%)")
    
    def demo_fault_tolerance(self):
        """Demonstrate fault tolerance"""
        print_header("DEMO 4: FAULT TOLERANCE")
        
        print_warning("This demo will simulate a worker failure")
        input("Press Enter to continue...")
        
        print_step(1, "Checking initial worker status...")
        
        try:
            response = requests.get(f"{self.coordinator_url}/workers/status")
            if response.status_code == 200:
                workers = response.json()
                print(f"Active workers: {workers['total_active']}")
        except:
            pass
        
        print_step(2, "Simulating worker crash on port 8001...")
        
        try:
            requests.post("http://localhost:8001/simulate/crash", timeout=1)
        except:
            pass  # Expected to fail
        
        print_warning("Worker 8001 crashed")
        time.sleep(2)
        
        print_step(3, "Sending requests to test fault tolerance...")
        
        success_count = 0
        for i in range(5):
            try:
                response = requests.post(
                    f"{self.coordinator_url}/inference",
                    json={
                        "data": self.generate_test_image(),
                        "model_type": "any",
                        "priority": "high"
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result['status'] == 'success':
                        success_count += 1
                        print(f"  Request {i+1}: ✅ Success (handled by {result.get('worker_id')})")
                    else:
                        print(f"  Request {i+1}: ❌ Failed")
                        
            except Exception as e:
                print(f"  Request {i+1}: ❌ Error: {e}")
        
        print(f"\n{Colors.BOLD}Fault Tolerance Result:{Colors.ENDC}")
        print(f"  Success rate during failure: {success_count}/5 ({success_count/5*100:.0f}%)")
        
        if success_count >= 4:
            print_success("Excellent fault tolerance!")
        elif success_count >= 3:
            print_warning("Good fault tolerance, some issues")
        else:
            print_error("Poor fault tolerance")
    
    def demo_priority_queue(self):
        """Demonstrate priority queue functionality"""
        print_header("DEMO 5: PRIORITY QUEUE")
        
        print_step(1, "Sending requests with different priorities...")
        
        requests_sent = []
        priorities = ["low", "normal", "high", "high", "low", "normal", "high"]
        
        print("Sending in order: " + ", ".join(priorities))
        
        for i, priority in enumerate(priorities):
            try:
                response = requests.post(
                    f"{self.coordinator_url}/inference",
                    json={
                        "data": self.generate_test_image(),
                        "model_type": "any",
                        "priority": priority,
                        "request_id": f"demo_priority_{i}_{priority}"
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    requests_sent.append({
                        "id": result['request_id'],
                        "priority": priority,
                        "order": i
                    })
                    
            except Exception as e:
                print_error(f"Failed to send request: {e}")
        
        print_success(f"Sent {len(requests_sent)} requests with mixed priorities")
        
        print_step(2, "Checking queue status...")
        
        try:
            response = requests.get(f"{self.coordinator_url}/queue/status")
            if response.status_code == 200:
                queue = response.json()
                print(f"\n{Colors.BOLD}Queue Status:{Colors.ENDC}")
                print(f"  High Priority: {queue['high_priority']}")
                print(f"  Normal Priority: {queue['normal_priority']}")
                print(f"  Low Priority: {queue['low_priority']}")
                print(f"  Total: {queue['size']}")
        except:
            pass
    
    def demo_metrics_monitoring(self):
        """Demonstrate metrics and monitoring"""
        print_header("DEMO 6: METRICS & MONITORING")
        
        print_step(1, "Fetching current system metrics...")
        
        try:
            response = requests.get(f"{self.coordinator_url}/metrics")
            if response.status_code == 200:
                metrics = response.json()
                
                print(f"\n{Colors.BOLD}System Metrics:{Colors.ENDC}")
                print(f"  Uptime: {metrics['uptime_seconds']:.1f} seconds")
                print(f"  Total Requests: {metrics['requests_total']}")
                print(f"  Success Rate: {metrics['success_rate']*100:.1f}%")
                print(f"  Active Workers: {metrics['workers_active']}")
                
                if 'latency_avg' in metrics:
                    print(f"\n{Colors.BOLD}Latency Statistics:{Colors.ENDC}")
                    print(f"  Average: {metrics['latency_avg']:.1f}ms")
                    print(f"  P50: {metrics['latency_p50']:.1f}ms")
                    print(f"  P99: {metrics['latency_p99']:.1f}ms")
                    
        except Exception as e:
            print_error(f"Failed to fetch metrics: {e}")
        
        print_info(f"For live monitoring, open: {self.dashboard_url}")
    
    def run_interactive_demo(self):
        """Run interactive demonstration"""
        print_header("DISTRIBUTED INFERENCE SYSTEM - INTERACTIVE DEMO")
        
        if not self.check_system():
            print_error("\nSome services are not running!")
            print_info("Please start the system using: ./start_all.sh")
            return
        
        demos = [
            ("Single Inference", self.demo_single_inference),
            ("Batch Inference", self.demo_batch_inference),
            ("Load Balancing", self.demo_load_balancing),
            ("Fault Tolerance", self.demo_fault_tolerance),
            ("Priority Queue", self.demo_priority_queue),
            ("Metrics & Monitoring", self.demo_metrics_monitoring)
        ]
        
        while True:
            print(f"\n{Colors.BOLD}Available Demos:{Colors.ENDC}")
            for i, (name, _) in enumerate(demos, 1):
                print(f"  {i}. {name}")
            print(f"  0. Exit")
            
            try:
                choice = input(f"\n{Colors.CYAN}Select demo (0-{len(demos)}): {Colors.ENDC}")
                choice = int(choice)
                
                if choice == 0:
                    print_info("Exiting demo. Thank you!")
                    break
                elif 1 <= choice <= len(demos):
                    demos[choice-1][1]()
                    input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")
                else:
                    print_error("Invalid choice")
                    
            except KeyboardInterrupt:
                print_info("\n\nDemo interrupted. Goodbye!")
                break
            except Exception as e:
                print_error(f"Error: {e}")

def main():
    """Main entry point"""
    demo = SystemDemo()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            print_header("QUICK DEMO MODE")
            
            if demo.check_system():
                print_info("Running quick demo sequence...")
                demo.demo_single_inference()
                time.sleep(1)
                demo.demo_batch_inference()
                time.sleep(1)
                demo.demo_metrics_monitoring()
                print_success("\nQuick demo completed!")
                print_info(f"Visit {demo.dashboard_url} for live monitoring")
            else:
                print_error("System not running. Start with: ./start_all.sh")
        else:
            print("Usage: python demo.py [--quick]")
    else:
        demo.run_interactive_demo()

if __name__ == "__main__":
    main()