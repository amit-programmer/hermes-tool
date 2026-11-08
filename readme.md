---
title: Hermes 3 8B Local API Server
emoji: 🚀
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
python_version: '3.11'
app_file: app.py
pinned: false
---

# 🚀 Hermes-3 8B Standalone In-Memory Inference Server

This repository runs **NousResearch/Hermes-3-Llama-3.1-8B** directly from **local model weights in memory**, giving you:
* **Zero API Limits:** No rate limits, no quotas, no external router errors.
* **100% Offline / Self-Contained:** Server starts up, downloads model weights, and executes inference directly using PyTorch and Hugging Face Transformers.
* **OpenAI-Compatible API:** Full compatibility with `/v1/chat/completions` (JSON and real-time SSE streaming) and `/v1/models`.
* **ZeroGPU Ready:** Includes native `@spaces.GPU` integration to leverage Hugging Face's **Free Nvidia A100 ZeroGPU** infrastructure.

---

## ⚡ Deployment on Hugging Face Spaces (Free Nvidia A100 ZeroGPU)

1. Push this repository to your Hugging Face Space: `https://huggingface.co/spaces/Amitansuranjan/testing-tool`
2. In your Space **Settings**:
   * Set **Space Hardware** to: **ZeroGPU (Nvidia A100 - Free)**
   * In **Variables and secrets**, add: `HF_TOKEN` = `<your_hf_token>`
3. Once the Space builds, it will mount the model with 4-bit quantization (~5.5GB VRAM) on the A100 GPU and generate tokens at **30–50+ tokens/second**!

---

## 💻 Running Locally

### Requirements
* **RAM:** For Hermes 8B, 16GB+ RAM (CPU) or 6GB+ VRAM (NVIDIA GPU).  
  *(Tip: If testing on an 8GB RAM PC, you can set `MODEL_ID=meta-llama/Llama-3.2-1B-Instruct` or `Qwen/Qwen2.5-1.5B-Instruct` in `.env` so your local machine runs smoothly without out-of-memory errors!)*

### Start Server
```powershell
python app.py
```

### Endpoints
* `POST /v1/chat/completions` (Supports standard JSON & SSE Streaming)
* `GET /health` (Reports memory and GPU status)
* `GET /v1/models`
* `GET /docs` (FastAPI Swagger UI)
