---
title: Hermes 3 8B Decoupled Gateway
emoji: ⚡
colorFrom: gray
colorTo: yellow
sdk: gradio
python_version: '3.11'
app_file: app.py
pinned: false
---

# ⚡ Hermes-3 8B Decoupled Brain & Gateway

A lightweight, production-grade **OpenAI-compatible LLM Gateway and Orchestrator** for **NousResearch/Hermes-3-Llama-3.1-8B**.

Deployed directly on Hugging Face's **Free Tier (2 vCPU - 16 GB RAM)** by offloading heavy matrix multiplication to Hugging Face's **Serverless Inference API (Dedicated GPU Cluster)**.

---

## 🏛️ Architecture Overview

Running an 8B model locally on a CPU space generates tokens at an impractical **0.5 – 1 token/sec** and consumes 14GB+ RAM. 

This **Decoupled Brain & Gateway** pattern splits the responsibilities:
1. **Lightweight FastAPI Orchestrator (CPU Space):** Consumes `< 150 MB RAM`, formats ChatML prompts, handles cold-start retries, enforces in-flight throttling, and streams OpenAI chunks.
2. **Hugging Face Serverless GPU Cluster (Remote Cloud):** Executes Multi-Head Attention on enterprise NVIDIA GPUs with high throughput (**30–60+ tokens/sec**).

```
[Telegram Bot / VS Code / Client]
               │
               ▼
┌────────────────────────────────────────────────────────┐
│   HF Space Orchestrator (FastAPI on 2 vCPU)            │
│                                                        │
│  1. Parse & validate incoming OpenAI completion payload │
│  2. Injects Hardened Lead Architect persona if needed  │
│  3. Construct ChatML prompt: <|im_start|>...<|im_end|> │
│  4. In-flight request throttling (asyncio.Semaphore)   │
└──────────────────────────┬─────────────────────────────┘
                           │ (HTTPS REST API / Bearer Token)
                           ▼
┌────────────────────────────────────────────────────────┐
│   Hugging Face Serverless GPU Cluster                  │
│   Model: NousResearch/Hermes-3-Llama-3.1-8B            │
│   • Enterprise GPU Execution (A10G / A100)             │
│   • Blazing Token Generation (30-60+ tokens/sec)       │
└──────────────────────────┬─────────────────────────────┘
                           │ (Streamed Chunks / JSON Response)
                           ▼
┌────────────────────────────────────────────────────────┐
│   HF Space Orchestrator                                │
│                                                        │
│  5. 503 Auto-Recovery / Cold-Start Exponential Backoff │
│  6. Special token preserver (keeps <tool_call>)        │
│  7. Returns OpenAI JSON or SSE Streaming response      │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
[Telegram Bot / Client Delivery]
```

---

## 🚀 Key Features

* **OpenAI-Compatible API:** Drop-in replacement for OpenAI endpoints (`/v1/chat/completions`, `/v1/models`).
* **Cold-Start Auto-Recovery Engine:** Automatically catches HF `503 Service Unavailable` / `Model is currently loading` and applies exponential backoff instead of failing.
* **ChatML & Special Token Preserver:** Fully maintains ChatML delimiters (`<|im_start|>`, `<|im_end|>`) while cleanly preserving agentic tool tags (`<tool_call>`, `</tool_call>`).
* **Dynamic System Persona:** Injects a default **Lead Co-Founder & Senior Technical Architect** persona with anti-laziness and production engineering rules when no system prompt is provided.
* **Token-Level SSE Streaming:** Supports `stream: true` for real-time typing indicators and low-latency delivery.
* **In-Flight Request Throttling:** Uses `asyncio.Semaphore` to protect Free Tier limits and prevent `429 Too Many Requests`.
* **Zero Hardware Crashes:** Minimal CPU footprint, boots in seconds, zero out-of-memory errors.

---

## 📡 API Endpoints Reference

| Route | Method | Description | Primary Use Case |
|---|---|---|---|
| `/health` | `GET` | Gateway uptime & status check | Keep-alive pings (UptimeRobot, cron) |
| `/v1/models` | `GET` | List available models | OpenAI SDK / IDE plugin handshake |
| `/v1/chat/completions` | `POST` | Core LLM reasoning & generation | Telegram bot, IDE plugins, REST clients |
| `/v1/warmup` | `GET` / `POST` | Proactive background model warmup | Wake up cold GPU cluster |

---

## 💻 Integration Examples

### 1. Python (`openai` SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<YOUR_HF_SPACE_HOST>/v1",
    api_key="not-needed"  # Handled server-side by Space secrets
)

response = client.chat.completions.create(
    model="NousResearch/Hermes-3-Llama-3.1-8B",
    messages=[
        {"role": "user", "content": "Explain microservices decomposition in 3 concise bullet points."}
    ],
    temperature=0.7,
    stream=False
)

print(response.choices[0].message.content)
```

### 2. Token-Level Streaming (SSE)

```python
stream = client.chat.completions.create(
    model="NousResearch/Hermes-3-Llama-3.1-8B",
    messages=[{"role": "user", "content": "Write a FastAPI rate-limiter middleware."}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 3. Telegram Bot (`httpx` / `python-telegram-bot`)

```python
import httpx

async def get_hermes_reply(user_prompt: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://<YOUR_HF_SPACE_HOST>/v1/chat/completions",
            json={
                "model": "Hermes-3-8B",
                "messages": [{"role": "user", "content": user_prompt}],
                "temperature": 0.7,
                "max_tokens": 1024
            },
            timeout=90.0
        )
        data = response.json()
        return data["choices"][0]["message"]["content"]
```

### 4. cURL

```bash
curl -X POST "https://<YOUR_HF_SPACE_HOST>/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "NousResearch/Hermes-3-Llama-3.1-8B",
    "messages": [
      {"role": "user", "content": "Hello Hermes, write a quick health check function in Python."}
    ],
    "temperature": 0.7
  }'
```

---

## ⚙️ Environment Variables

Set these in your Hugging Face Space **Settings -> Variables and secrets**:

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` (or `HF_ACCESS_TOKEN`) | **Yes** | None | Hugging Face User Access Token (read permissions) |
| `MODEL_ID` | No | `NousResearch/Hermes-3-Llama-3.1-8B` | Target Hugging Face model repository |
| `MAX_CONCURRENCY` | No | `1` | Maximum parallel in-flight generations |
| `MAX_RETRIES` | No | `4` | Max retry attempts for 503 cold starts |
| `BASE_RETRY_DELAY` | No | `5.0` | Initial exponential backoff delay in seconds |
