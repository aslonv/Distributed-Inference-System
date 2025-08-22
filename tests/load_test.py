import asyncio
import aiohttp
import argparse
import time
import random
import base64
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import statistics

@dataclass
class LoadTestConfig:
    coordinator_url: str = "http://localhost:8000"
    num_requests: int = 100
    concurrent_requests: int = 10
    burst_mode: bool = False
    burst_size: int = 50
    burst_interval: float = 5.0
    model_types: List[str] = field(default_factory=lambda: ["any", "mobilenet", "resnet18", "efficientnet"])
    priorities: List[str] = field(default_factory=lambda: ["high", "normal", "low"])
    timeout: float = 30.0
    verbose: bool = False

@dataclass
class TestResults:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    latencies: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    start_time: float = 0
    end_time: float = 0
    
    def calculate_stats(self) -> Dict:
        duration = self.end_time - self.start_time
        
        stats = {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "timeout_requests": self.timeout_requests,
            "success_rate": self.successful_requests / max(1, self.total_requests) * 100,
            "duration_seconds": duration,
            "requests_per_second": self.total_requests / max(1, duration)
        }
        
        if self.latencies:
            sorted_latencies = sorted(self.latencies)
            stats.update({
                "latency_min": min(self.latencies),
                "latency_max": max(self.latencies),
                "latency_avg": statistics.mean(self.latencies),
                "latency_median": statistics.median(self.latencies),
                "latency_p95": sorted_latencies[int(len(sorted_latencies) * 0.95)],
                "latency_p99": sorted_latencies[int(len(sorted_latencies) * 0.99)]
            })
        
        return stats

class LoadGenerator:
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results = TestResults()
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def generate_request(self) -> Dict:
        """Generate random test data for inference request"""
        dummy_data = f"test_image_{random.randint(1000, 9999)}"
        encoded_data = base64.b64encode(dummy_data.encode()).decode()
        
        return {
            "data": encoded_data,
            "model_type": random.choice(self.config.model_types),
            "priority": random.choice(self.config.priorities)
        }
    
    async def send_request(self, request_num: int) -> Dict:
        request_data = self.generate_request()
        start_time = time.time()
        
        try:
            async with self.session.post(
                f"{self.config.coordinator_url}/inference",
                json=request_data
            ) as response:
                latency = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "latency": latency,
                        "response": result,
                        "request_num": request_num
                    }
                else:
                    return {
                        "success": False,
                        "latency": latency,
                        "error": f"HTTP {response.status}",
                        "request_num": request_num
                    }
                    
        except asyncio.TimeoutError:
            return {
                "success": False,
                "timeout": True,
                "error": "Request timeout",
                "request_num": request_num
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "request_num": request_num
            }
    
    async def worker(self, queue: asyncio.Queue):
        while True:
            request_num = await queue.get()
            if request_num is None:
                break
                
            result = await self.send_request(request_num)
            
            self.results.total_requests += 1
            
            if result.get("success"):
                self.results.successful_requests += 1
                self.results.latencies.append(result["latency"])
                
                if self.config.verbose:
                    print(f"[OK] Request {request_num}: {result['latency']:.1f}ms")
            else:
                if result.get("timeout"):
                    self.results.timeout_requests += 1
                else:
                    self.results.failed_requests += 1
                    
                self.results.errors.append(result.get("error", "Unknown error"))
                
                if self.config.verbose:
                    print(f"[FAIL] Request {request_num}: {result.get('error')}")
            
            queue.task_done()
    
    async def run_steady_load(self):
        print(f"\nStarting steady load test:")
        print(f"   Requests: {self.config.num_requests}")
        print(f"   Concurrent: {self.config.concurrent_requests}")
        print(f"   Target: {self.config.coordinator_url}\n")
        
        queue = asyncio.Queue()
        workers = []
        
        for _ in range(self.config.concurrent_requests):
            worker_task = asyncio.create_task(self.worker(queue))
            workers.append(worker_task)
        
        self.results.start_time = time.time()
        
        for i in range(self.config.num_requests):
            await queue.put(i)
            
            if not self.config.verbose and i % 10 == 0:
                progress = (i + 1) / self.config.num_requests * 100
                print(f"Progress: {progress:.1f}% ({i+1}/{self.config.num_requests})", end="\r")
        
        await queue.join()
        
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)
        
        self.results.end_time = time.time()
        
        if not self.config.verbose:
            print()
    
    async def run_burst_load(self):
        print(f"\nStarting burst load test:")
        print(f"   Total Requests: {self.config.num_requests}")
        print(f"   Burst Size: {self.config.burst_size}")
        print(f"   Burst Interval: {self.config.burst_interval}s")
        print(f"   Target: {self.config.coordinator_url}\n")
        
        self.results.start_time = time.time()
        
        num_bursts = self.config.num_requests // self.config.burst_size
        remainder = self.config.num_requests % self.config.burst_size
        
        request_counter = 0
        
        for burst_num in range(num_bursts):
            print(f"Burst {burst_num + 1}/{num_bursts}: Sending {self.config.burst_size} requests...")
            
            tasks = []
            for _ in range(self.config.burst_size):
                task = self.send_request(request_counter)
                tasks.append(task)
                request_counter += 1
            
            results = await asyncio.gather(*tasks)
            
            for result in results:
                self.results.total_requests += 1
                if result.get("success"):
                    self.results.successful_requests += 1
                    self.results.latencies.append(result["latency"])
                else:
                    if result.get("timeout"):
                        self.results.timeout_requests += 1
                    else:
                        self.results.failed_requests += 1
                    self.results.errors.append(result.get("error", "Unknown"))
            
            if burst_num < num_bursts - 1:
                print(f"   Waiting {self.config.burst_interval}s before next burst...")
                await asyncio.sleep(self.config.burst_interval)
        
        # Handle remaining requests
        if remainder > 0:
            print(f"Final burst: Sending {remainder} requests...")
            tasks = []
            for _ in range(remainder):
                task = self.send_request(request_counter)
                tasks.append(task)
                request_counter += 1
            
            results = await asyncio.gather(*tasks)
            
            for result in results:
                self.results.total_requests += 1
                if result.get("success"):
                    self.results.successful_requests += 1
                    self.results.latencies.append(result["latency"])
                else:
                    if result.get("timeout"):
                        self.results.timeout_requests += 1
                    else:
                        self.results.failed_requests += 1
                    self.results.errors.append(result.get("error", "Unknown"))
        
        self.results.end_time = time.time()
    
    async def run(self):
        if self.config.burst_mode:
            await self.run_burst_load()
        else:
            await self.run_steady_load()
        
        return self.results

def print_results(results: TestResults):
    stats = results.calculate_stats()
    
    print("\n" + "="*60)
    print("LOAD TEST RESULTS")
    print("="*60)
    
    print(f"\nSummary:")
    print(f"   Total Requests: {stats['total_requests']}")
    print(f"   Successful: {stats['successful_requests']} ({stats['success_rate']:.1f}%)")
    print(f"   Failed: {stats['failed_requests']}")
    print(f"   Timeouts: {stats['timeout_requests']}")
    print(f"   Duration: {stats['duration_seconds']:.2f}s")
    print(f"   Throughput: {stats['requests_per_second']:.1f} req/s")
    
    if results.latencies:
        print(f"\nLatency Statistics (ms):")
        print(f"   Min: {stats['latency_min']:.1f}")
        print(f"   Max: {stats['latency_max']:.1f}")
        print(f"   Average: {stats['latency_avg']:.1f}")
        print(f"   Median: {stats['latency_median']:.1f}")
        print(f"   P95: {stats['latency_p95']:.1f}")
        print(f"   P99: {stats['latency_p99']:.1f}")
    
    if results.errors:
        print(f"\nError Summary:")
        error_counts = {}
        for error in results.errors[:10]:
            error_counts[error] = error_counts.get(error, 0) + 1
        
        for error, count in error_counts.items():
            print(f"   {error}: {count} occurrences")
    
    print("\n" + "="*60)
    
    # Performance assessment
    if stats['success_rate'] >= 99:
        print("Excellent performance")
    elif stats['success_rate'] >= 95:
        print("Good performance")
    elif stats['success_rate'] >= 90:
        print("Acceptable performance, but could be improved")
    else:
        print("Poor performance - investigation needed")
    
    print("="*60 + "\n")

async def main():
    parser = argparse.ArgumentParser(description="Load test the distributed inference system")
    parser.add_argument("--requests", "-r", type=int, default=100,
                       help="Total number of requests to send")
    parser.add_argument("--concurrent", "-c", type=int, default=10,
                       help="Number of concurrent requests")
    parser.add_argument("--burst", action="store_true",
                       help="Enable burst mode")
    parser.add_argument("--burst-size", type=int, default=50,
                       help="Number of requests per burst")
    parser.add_argument("--burst-interval", type=float, default=5.0,
                       help="Seconds between bursts")
    parser.add_argument("--url", type=str, default="http://localhost:8000",
                       help="Coordinator URL")
    parser.add_argument("--timeout", type=float, default=30.0,
                       help="Request timeout in seconds")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    config = LoadTestConfig(
        coordinator_url=args.url,
        num_requests=args.requests,
        concurrent_requests=args.concurrent,
        burst_mode=args.burst,
        burst_size=args.burst_size,
        burst_interval=args.burst_interval,
        timeout=args.timeout,
        verbose=args.verbose
    )
    
    try:
        async with LoadGenerator(config) as generator:
            results = await generator.run()
            print_results(results)
    except KeyboardInterrupt:
        print("\n\nLoad test interrupted by user")
    except Exception as e:
        print(f"\nLoad test failed: {e}")

if __name__ == "__main__":
    import sys
    import os
    
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    asyncio.run(main())