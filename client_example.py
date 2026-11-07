"""
Hermes-3 8B Decoupled Gateway Client Example
Demonstrates how to interact with the gateway using both standard HTTP requests
and real-time streaming Server-Sent Events (SSE).

Reads the target URL directly from your .env file:
- GATEWAY_URL or HF_SPACE_URL
"""

import os
import sys
import re
import json
import httpx
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

def resolve_gateway_url() -> str:
    """
    Resolves the base URL from .env.
    If the URL is a Hugging Face Space repository URL (e.g. https://huggingface.co/spaces/User/Repo),
    it converts it to the direct API domain: https://user-repo.hf.space
    """
    raw_url = os.getenv("GATEWAY_URL") or os.getenv("HF_SPACE_URL") or os.getenv("HF_SPACE_REPO") or "http://127.0.0.1:7860"
    raw_url = raw_url.strip().rstrip("/")
    
    # Check if user provided standard Hugging Face space web URL:
    # https://huggingface.co/spaces/{owner}/{space_name}
    hf_match = re.search(r"huggingface\.co/spaces/([^/]+)/([^/]+)", raw_url, re.IGNORECASE)
    if hf_match:
        owner = hf_match.group(1).lower().replace("_", "-")
        space = hf_match.group(2).lower().replace("_", "-")
        direct_url = f"https://{owner}-{space}.hf.space"
        return direct_url
    
    return raw_url

GATEWAY_URL = resolve_gateway_url()
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HF_ACCESS_TOKEN")

def get_client_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    # If space is private, Bearer token is needed
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN.strip()}"
    return headers

def print_separator(title: str):
    print(f"\n{'=' * 20} {title} {'=' * 20}")

def test_health():
    print_separator("1. Testing /health")
    headers = get_client_headers()
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(f"{GATEWAY_URL}/health", headers=headers)
        print(f"Status: {resp.status_code}")
        try:
            print(json.dumps(resp.json(), indent=2))
        except Exception:
            print(resp.text[:300])

def test_models():
    print_separator("2. Testing /v1/models")
    headers = get_client_headers()
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(f"{GATEWAY_URL}/v1/models", headers=headers)
        print(f"Status: {resp.status_code}")
        try:
            print(json.dumps(resp.json(), indent=2))
        except Exception:
            print(resp.text[:300])

def test_warmup():
    print_separator("3. Testing /v1/warmup")
    headers = get_client_headers()
    with httpx.Client(timeout=35.0) as client:
        resp = client.get(f"{GATEWAY_URL}/v1/warmup", headers=headers)
        print(f"Status: {resp.status_code}")
        try:
            print(json.dumps(resp.json(), indent=2))
        except Exception:
            print(resp.text[:300])

def test_chat_completion(prompt: str = "Explain the advantages of event-driven architecture in 2 short bullet points."):
    print_separator("4. Testing /v1/chat/completions (Non-Streaming)")
    headers = get_client_headers()
    payload = {
        "model": "NousResearch/Hermes-3-Llama-3.1-8B",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": False
    }
    
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(f"{GATEWAY_URL}/v1/chat/completions", headers=headers, json=payload)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            print("\n--- Assistant Response ---")
            print(content)
            print("--------------------------")
            print(f"Usage: {data.get('usage')}")
            print(f"Metadata: {data.get('_gateway_metadata')}")
        else:
            print(f"Response ({resp.status_code}): {resp.text}")

def test_streaming_chat(prompt: str = "Write a Python generator function that yields fibonacci numbers."):
    print_separator("5. Testing /v1/chat/completions (SSE Streaming)")
    headers = get_client_headers()
    payload = {
        "model": "NousResearch/Hermes-3-Llama-3.1-8B",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": True
    }
    
    print(f"Connecting to stream on {GATEWAY_URL}...")
    with httpx.Client(timeout=120.0) as client:
        with client.stream("POST", f"{GATEWAY_URL}/v1/chat/completions", headers=headers, json=payload) as response:
            print(f"Stream Status: {response.status_code}")
            if response.status_code != 200:
                print(f"Error: {response.read().decode('utf-8')}")
                return
            
            print("\n--- Streaming Tokens ---")
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                raw_data = line[len("data:"):].strip()
                if raw_data == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw_data)
                    token = chunk["choices"][0].get("delta", {}).get("content", "")
                    print(token, end="", flush=True)
                except Exception:
                    continue
            print("\n------------------------\n[Stream Finished]")

if __name__ == "__main__":
    print(f"Resolved Gateway Base URL: {GATEWAY_URL}")
    print(f"Configured from .env (HF_SPACE_URL / GATEWAY_URL)")
    
    try:
        test_health()
        test_models()
        test_warmup()
        test_chat_completion()
        test_streaming_chat()
    except httpx.ConnectError:
        print(f"\n[!] Could not connect to {GATEWAY_URL}.")
        print("    If testing locally, start the gateway server using: python app.py")
        print("    If testing remote HF Space, ensure the space has finished building and is Running.")
