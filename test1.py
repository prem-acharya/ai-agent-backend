import json
import requests
from dotenv import load_dotenv
import os
import asyncio
import httpx
from typing import AsyncGenerator, Dict

load_dotenv()

async def test_streaming_chat(query: str) -> AsyncGenerator[str, None]:
    """Test the streaming chat endpoint with async support"""
    url = "http://localhost:8000/api/v1/chat"
    
    data = {
        "content": query
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                url,
                json=data,
                headers=headers,
                timeout=30.0
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_text():
                    yield chunk
    except httpx.HTTPError as e:
        yield f"HTTP Error: {str(e)}"
    except Exception as e:
        yield f"Error: {str(e)}"

async def run_chat_tests():
    """Run a series of chat tests"""
    test_queries = [
        "What is artificial intelligence?",
        "Write a short poem about coding",
        "Explain quantum computing in simple terms"
    ]
    
    for query in test_queries:
        print(f"\n\nTesting query: {query}")
        print("Response:")
        async for chunk in test_streaming_chat(query):
            print(chunk, end='', flush=True)
        print("\n---")

def test_sync_chat(query: str):
    """Test the chat endpoint with synchronous requests"""
    url = "http://localhost:8000/api/v1/chat"
    
    data = {
        "content": query
    }

    headers = {"Content-Type": "application/json"}

    try:
        with requests.post(url, json=data, headers=headers, stream=True) as r:
            r.raise_for_status()
            print(f"\nTesting query: {query}")
            print("Response:")
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    print(chunk.decode('utf-8'), end='')
            print("\n---\n")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Test both sync and async methods
    print("Running synchronous tests...")
    test_sync_chat("Tell me a joke about programming")
    
    print("\nRunning asynchronous tests...")
    asyncio.run(run_chat_tests())
