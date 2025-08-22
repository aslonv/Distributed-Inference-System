#!/usr/bin/env python
"""
Simple test to populate dashboard with data
"""

import requests
import base64
import time
import json

def main():
    print("Sending test requests to populate dashboard...")
    
    # Base URL
    url = "http://localhost:8000/inference"
    
    # Create dummy image data
    dummy_image = base64.b64encode(b"test_image_data").decode()
    
    # Send multiple requests with different priorities
    for i in range(30):
        request_data = {
            "data": dummy_image,
            "model_type": ["mobilenet", "resnet18", "efficientnet", "any"][i % 4],
            "priority": ["high", "normal", "low"][i % 3]
        }
        
        try:
            response = requests.post(url, json=request_data, timeout=5)
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Request {i+1}: Success - {result['model']} ({result['latency_ms']:.1f}ms)")
            else:
                print(f"✗ Request {i+1}: Failed - Status {response.status_code}")
        except Exception as e:
            print(f"✗ Request {i+1}: Error - {e}")
        
        # Small delay between requests
        time.sleep(0.2)
    
    print("\n✅ Done! Check the dashboard at http://localhost:8501")
    print("The graphs should now show:")
    print("  - Latency distribution")
    print("  - Request success/failure pie chart")
    print("  - Worker performance charts")

if __name__ == "__main__":
    main()