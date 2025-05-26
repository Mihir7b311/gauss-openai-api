"""
Test OpenAI client usage with the Gauss API
"""

import asyncio
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Configure OpenAI client to use local server
client = OpenAI(
    api_key="sk-dummy-key",  # Not used but required by OpenAI client
    base_url="http://localhost:8000/v1"
)


def test_models():
    """Test the models endpoint"""
    print("=== Testing Models Endpoint ===")
    try:
        models = client.models.list()
        print(f"Available models: {len(models.data)}")
        for model in models.data:
            print(f"  - {model.id} (owned by: {model.owned_by})")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_chat_completion():
    """Test non-streaming chat completion"""
    print("\n=== Testing Chat Completion ===")
    try:
        response = client.chat.completions.create(
            model="gauss",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! How are you today?"}
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        print(f"Response ID: {response.id}")
        print(f"Model: {response.model}")
        print(f"Content: {response.choices[0].message.content}")
        print(f"Usage: {response.usage}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_streaming_chat():
    """Test streaming chat completion"""
    print("\n=== Testing Streaming Chat ===")
    try:
        stream = client.chat.completions.create(
            model="gauss",
            messages=[
                {"role": "user", "content": "Write a short poem about artificial intelligence"}
            ],
            stream=True,
            temperature=0.8,
            max_tokens=200
        )
        
        print("Streaming response:")
        content = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
                print(chunk.choices[0].delta.content, end="", flush=True)
        
        print(f"\n\nTotal content length: {len(content)} characters")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_error_handling():
    """Test error handling"""
    print("\n=== Testing Error Handling ===")
    try:
        # Test with invalid parameters
        response = client.chat.completions.create(
            model="invalid-model",
            messages=[
                {"role": "user", "content": "Test"}
            ],
            temperature=5.0  # Invalid temperature
        )
        print("Error: Should have failed with invalid parameters")
        return False
    except Exception as e:
        print(f"Expected error caught: {type(e).__name__}")
        return True


if __name__ == "__main__":
    print("Testing Gauss OpenAI Compatible API...")
    
    tests = [
        ("Models", test_models),
        ("Chat Completion", test_chat_completion),
        ("Streaming Chat", test_streaming_chat),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        result = test_func()
        results.append((test_name, result))
    
    print(f"\n{'='*50}")
    print("TEST RESULTS:")
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")