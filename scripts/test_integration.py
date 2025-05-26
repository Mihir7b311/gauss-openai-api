"""
Integration tests for the Gauss OpenAI API
"""

import requests
import json
import time


def test_health_endpoints():
    """Test health check endpoints"""
    print("=== Testing Health Endpoints ===")
    
    # Test basic health
    response = requests.get("http://localhost:8000/health")
    print(f"Health check: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test detailed health
    response = requests.get("http://localhost:8000/health/detailed")
    print(f"Detailed health: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test root endpoint
    response = requests.get("http://localhost:8000/")
    print(f"Root endpoint: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_models_api():
    """Test models API endpoint"""
    print("\n=== Testing Models API ===")
    
    response = requests.get("http://localhost:8000/v1/models")
    print(f"Models API: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Models found: {len(data['data'])}")
        for model in data['data']:
            print(f"  - {model['id']}")
    else:
        print(f"Error: {response.text}")


def test_chat_api():
    """Test chat completions API"""
    print("\n=== Testing Chat API ===")
    
    payload = {
        "model": "gauss",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "temperature": 0.7,
        "max_tokens": 50
    }
    
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Chat API: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Response ID: {data['id']}")
        print(f"Content: {data['choices'][0]['message']['content']}")
        print(f"Usage: {data['usage']}")
    else:
        print(f"Error: {response.text}")


def test_streaming_api():
    """Test streaming chat API"""
    print("\n=== Testing Streaming API ===")
    
    payload = {
        "model": "gauss",
        "messages": [
            {"role": "user", "content": "Count from 1 to 5"}
        ],
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        stream=True
    )
    
    print(f"Streaming API: {response.status_code}")
    
    if response.status_code == 200:
        print("Stream chunks:")
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                print(f"  {decoded_line}")
                if decoded_line.strip() == "data: [DONE]":
                    break
    else:
        print(f"Error: {response.text}")


if __name__ == "__main__":
    print("Running Gauss OpenAI API Integration Tests...")
    
    # Wait a moment for server to be ready
    time.sleep(1)
    
    try:
        test_health_endpoints()
        test_models_api()
        test_chat_api()
        test_streaming_api()
        print("\n=== Integration Tests Completed ===")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API server. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"Error during testing: {e}")