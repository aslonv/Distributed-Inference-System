import asyncio
import aiohttp
import time
import random
import json
import base64
from typing import List, Dict, Any
from datetime import datetime
from enum import Enum

class FailureScenario(Enum):
    WORKER_CRASH = "worker_crash"
    WORKER_TIMEOUT = "worker_timeout"
    NETWORK_PARTITION = "network_partition"
    HIGH_LATENCY = "high_latency"
    CHAOS_MODE = "chaos_mode"
    CASCADING_FAILURE = "cascading_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"

class ResilienceTest:
    def __init__(self, coordinator_url: str = "http://localhost:8000"):
        self.coordinator_url = coordinator_url
        self.worker_ports = [8001, 8002, 8003]
        self.test_results = {}
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_system_status(self) -> Dict:
        try:
            async with self.session.get(f"{self.coordinator_url}/health") as resp:
                if resp.status == 200:
                    return await resp.json()
        except:
            pass
        return {"status": "unknown", "stats": {}}
    
    async def get_worker_status(self) -> List[Dict]:
        try:
            async with self.session.get(f"{self.coordinator_url}/workers/status") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("workers", [])
        except:
            pass
        return []
    
    async def send_test_request(self) -> Dict:
        dummy_data = base64.b64encode(b"test_image").decode()
        
        try:
            start = time.time()
            async with self.session.post(
                f"{self.coordinator_url}/inference",
                json={
                    "data": dummy_data,
                    "model_type": "any",
                    "priority": "normal"
                },
                timeout=30
            ) as resp:
                latency = (time.time() - start) * 1000
                
                if resp.status == 200:
                    result = await resp.json()
                    return {
                        "success": True,
                        "latency": latency,
                        "retries": result.get("retries", 0)
                    }
                else:
                    return {"success": False, "error": f"HTTP {resp.status}"}
        except asyncio.TimeoutError:
            return {"success": False, "error": "timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def kill_worker(self, port: int):
        print(f"    Killing worker on port {port}")
        try:
            async with self.session.post(f"http://localhost:{port}/simulate/crash"):
                pass
        except:
            pass
    
    async def enable_chaos(self, port: int):
        print(f"   Enabling chaos on worker {port}")
        try:
            async with self.session.post(f"http://localhost:{port}/chaos/enable"):
                pass
        except:
            pass
    
    async def disable_chaos(self, port: int):
        try:
            async with self.session.post(f"http://localhost:{port}/chaos/disable"):
                pass
        except:
            pass
    
    async def test_worker_crash(self) -> Dict:
        """Test system behavior when a worker crashes"""
        print("\nTesting Worker Crash Scenario...")
        
        results = {
            "scenario": "Worker Crash",
            "start_time": datetime.now().isoformat(),
            "steps": []
        }
        
        # Step 1: Verify system is healthy
        print("   Step 1: Checking initial system health...")
        initial_status = await self.get_system_status()
        initial_workers = len(await self.get_worker_status())
        results["steps"].append({
            "step": "Initial health check",
            "workers_active": initial_status.get("stats", {}).get("workers_active", 0),
            "status": initial_status.get("status")
        })
        
        # Step 2: Send baseline requests
        print("   Step 2: Sending baseline requests...")
        baseline_results = []
        for _ in range(5):
            result = await self.send_test_request()
            baseline_results.append(result)
        
        baseline_success = sum(1 for r in baseline_results if r["success"])
        results["steps"].append({
            "step": "Baseline requests",
            "success_rate": baseline_success / 5,
            "avg_latency": sum(r.get("latency", 0) for r in baseline_results) / 5
        })
        
        # Step 3: Kill a worker
        print("   Step 3: Simulating worker crash...")
        await self.kill_worker(self.worker_ports[0])
        await asyncio.sleep(2)
        
        # Step 4: Test system during failure
        print("   Step 4: Testing system during failure...")
        failure_results = []
        for _ in range(10):
            result = await self.send_test_request()
            failure_results.append(result)
        
        failure_success = sum(1 for r in failure_results if r["success"])
        avg_retries = sum(r.get("retries", 0) for r in failure_results) / 10
        
        results["steps"].append({
            "step": "During failure",
            "success_rate": failure_success / 10,
            "avg_retries": avg_retries,
            "workers_active": (await self.get_system_status()).get("stats", {}).get("workers_active", 0)
        })
        
        # Step 5: Verify recovery
        print("   Step 5: Checking system recovery...")
        await asyncio.sleep(5)
        
        recovery_status = await self.get_system_status()
        results["steps"].append({
            "step": "Recovery check",
            "status": recovery_status.get("status"),
            "workers_active": recovery_status.get("stats", {}).get("workers_active", 0)
        })
        
        results["passed"] = failure_success >= 8  # 80% success during failure
        results["end_time"] = datetime.now().isoformat()
        
        return results
    
    async def test_chaos_mode(self) -> Dict:
        """Test system behavior with chaos engineering enabled"""
        print("\nTesting Chaos Mode Scenario...")
        
        results = {
            "scenario": "Chaos Mode",
            "start_time": datetime.now().isoformat(),
            "steps": []
        }
        
        print("   Enabling chaos on all workers...")
        for port in self.worker_ports:
            await self.enable_chaos(port)
        
        results["steps"].append({
            "step": "Chaos enabled",
            "workers_affected": len(self.worker_ports)
        })
        
        print("   Sending requests during chaos...")
        chaos_results = []
        for i in range(20):
            result = await self.send_test_request()
            chaos_results.append(result)
            if i % 5 == 0:
                print(f"      Progress: {i+1}/20 requests")
        
        success_count = sum(1 for r in chaos_results if r["success"])
        avg_retries = sum(r.get("retries", 0) for r in chaos_results) / 20
        
        results["steps"].append({
            "step": "During chaos",
            "total_requests": 20,
            "successful": success_count,
            "success_rate": success_count / 20,
            "avg_retries": avg_retries
        })
        
        print("   Disabling chaos...")
        for port in self.worker_ports:
            await self.disable_chaos(port)
        
        print("   Testing recovery after chaos...")
        recovery_results = []
        for _ in range(10):
            result = await self.send_test_request()
            recovery_results.append(result)
        
        recovery_success = sum(1 for r in recovery_results if r["success"])
        
        results["steps"].append({
            "step": "After chaos",
            "success_rate": recovery_success / 10
        })
        
        results["passed"] = success_count >= 15  # 75% success during chaos
        results["end_time"] = datetime.now().isoformat()
        
        return results
    
    async def test_cascading_failure(self) -> Dict:
        """Test system behavior during cascading failures"""
        print("\nTesting Cascading Failure Scenario...")
        
        results = {
            "scenario": "Cascading Failure",
            "start_time": datetime.now().isoformat(),
            "steps": []
        }
        
        # Kill workers sequentially
        for i, port in enumerate(self.worker_ports[:2]):  # Kill 2 out of 3
            print(f"   Stage {i+1}: Killing worker on port {port}")
            await self.kill_worker(port)
            
            stage_results = []
            for _ in range(5):
                result = await self.send_test_request()
                stage_results.append(result)
            
            success_rate = sum(1 for r in stage_results if r["success"]) / 5
            
            results["steps"].append({
                "step": f"After killing worker {i+1}",
                "workers_remaining": len(self.worker_ports) - (i + 1),
                "success_rate": success_rate
            })
            
            await asyncio.sleep(2)
        
        print("   Testing with minimal workers...")
        minimal_results = []
        for _ in range(10):
            result = await self.send_test_request()
            minimal_results.append(result)
        
        minimal_success = sum(1 for r in minimal_results if r["success"])
        
        results["steps"].append({
            "step": "Minimal workers",
            "success_rate": minimal_success / 10
        })
        
        results["passed"] = minimal_success >= 7  # 70% success with 1 worker
        results["end_time"] = datetime.now().isoformat()
        
        return results
    
    async def test_load_during_failure(self) -> Dict:
        """Test system under load while experiencing failures"""
        print("\nTesting Load During Failure Scenario...")
        
        results = {
            "scenario": "Load During Failure",
            "start_time": datetime.now().isoformat(),
            "steps": []
        }
        
        print("   Establishing baseline...")
        baseline = []
        for _ in range(10):
            result = await self.send_test_request()
            baseline.append(result)
        
        baseline_success = sum(1 for r in baseline if r["success"]) / 10
        results["steps"].append({
            "step": "Baseline",
            "success_rate": baseline_success
        })
        
        print("   Enabling chaos and sending burst...")
        await self.enable_chaos(self.worker_ports[0])
        
        tasks = []
        for i in range(50):
            task = self.send_test_request()
            tasks.append(task)
        
        burst_results = await asyncio.gather(*tasks)
        burst_success = sum(1 for r in burst_results if r["success"])
        
        results["steps"].append({
            "step": "Burst during chaos",
            "total_requests": 50,
            "successful": burst_success,
            "success_rate": burst_success / 50
        })
        
        await self.disable_chaos(self.worker_ports[0])
        
        results["passed"] = burst_success >= 35  # 70% success during burst with chaos
        results["end_time"] = datetime.now().isoformat()
        
        return results
    
    async def run_all_tests(self) -> Dict:
        """Run all resilience tests"""
        print("\n" + "="*60)
        print("DISTRIBUTED INFERENCE SYSTEM - RESILIENCE TEST SUITE")
        print("="*60)
        
        all_results = {
            "test_suite": "Resilience Tests",
            "start_time": datetime.now().isoformat(),
            "tests": []
        }
        
        test_scenarios = [
            ("Worker Crash", self.test_worker_crash),
            ("Chaos Mode", self.test_chaos_mode),
            ("Cascading Failure", self.test_cascading_failure),
            ("Load During Failure", self.test_load_during_failure)
        ]
        
        for name, test_func in test_scenarios:
            try:
                print(f"\n{'='*40}")
                result = await test_func()
                all_results["tests"].append(result)
                
                if result["passed"]:
                    print(f"{name}: PASSED")
                else:
                    print(f"{name}: FAILED")
                    
            except Exception as e:
                print(f"{name}: ERROR - {e}")
                all_results["tests"].append({
                    "scenario": name,
                    "passed": False,
                    "error": str(e)
                })
            
            await asyncio.sleep(3)
        
        all_results["end_time"] = datetime.now().isoformat()
        
        total_tests = len(all_results["tests"])
        passed_tests = sum(1 for t in all_results["tests"] if t.get("passed", False))
        
        all_results["summary"] = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0
        }
        
        return all_results

def print_test_summary(results: Dict):
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    summary = results.get("summary", {})
    
    print(f"\nTotal Tests: {summary.get('total_tests', 0)}")
    print(f"Passed: {summary.get('passed', 0)}")
    print(f"Failed: {summary.get('failed', 0)}")
    print(f"Pass Rate: {summary.get('pass_rate', 0)*100:.1f}%")
    
    print("\nDetailed Results:")
    for test in results.get("tests", []):
        status = "PASS" if test.get("passed") else "FAIL"
        print(f"   {test.get('scenario')}: {status}")
        
        if not test.get("passed") and "error" in test:
            print(f"      Error: {test['error']}")
    
    print("\n" + "="*60)
    pass_rate = summary.get("pass_rate", 0)
    
    if pass_rate == 1.0:
        print("All resilience tests passed!")
    elif pass_rate >= 0.75:
        print("Most resilience tests passed.")
        print("The system shows good fault tolerance with minor issues.")
    elif pass_rate >= 0.5:
        print("ACCEPTABLE. Some resilience issues detected.")
    else:
        print("POOR. Significant resilience issues detected.")
    
    print("="*60)

async def main():
    print("\nStarting Resilience Test Suite...")
    print("Make sure the distributed inference system is running")
    print("Waiting 3 seconds for system to be ready...")
    await asyncio.sleep(3)
    
    try:
        async with ResilienceTest() as tester:
            status = await tester.get_system_status()
            if status.get("status") == "unknown":
                print("\nError: Cannot connect to coordinator!")
                print("Please make sure the system is running:")
                print("  ./start_all.sh")
                return
            
            results = await tester.run_all_tests()
            print_test_summary(results)
            
            with open("resilience_test_results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"\nDetailed results saved to: resilience_test_results.json")
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest suite failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())