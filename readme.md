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

# ⚡ Hermes-3 8B Standalone AI Server & Somi Co-Founder Engine

A high-performance, lightweight, OpenAI-compatible LLM inference server powered by **`NousResearch/Hermes-3-Llama-3.1-8B`**, custom-tailored for **Hugging Face ZeroGPU (NVIDIA A100)** and local deployment.

Features a **dynamic persona engine (Somi)**, **in-memory short-term session memory**, **Vector-DB-free multi-format document parsing**, and an ultra-low VRAM footprint (~5GB) via **4-bit NF4 quantization**.

---

## 🏛️ System Architecture

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      Client Applications & Web UI                      │
 │  • Interactive Responsive Chat Playground (Dark / Light, 60fps SSE)   │
 │  • OpenAI API Clients (LangChain, LlamaIndex, Python SDK, cURL)        │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                       FastAPI Gateway Layer (app.py)                   │
 │  • OpenAI Endpoint Router (/v1/chat/completions, /v1/models)           │
 │  • Multi-Format Document Ingestion (/v1/files/extract, /upload-and-…)  │
 │  • Short-Term Session Store & Working Memory (/v1/sessions)            │
 └─────────────┬────────────────────────────────────────────┬─────────────┘
               │                                            │
               ▼                                            ▼
 ┌───────────────────────────┐                ┌───────────────────────────┐
 │   Session Working Memory  │                │   Document Extraction     │
 │  (session_memory.py)      │                │   (Vector-DB Free)        │
 │  • Rolling Buffer (12 msg)│                │  • PDF (pypdf)            │
 │  • Pinned Documents       │                │  • Word DOCX (python-docx)│
 │  • Multi-Session State    │                │  • Excel / CSV (openpyxl) │
 └─────────────┬─────────────┘                │  • PPTX (python-pptx)     │
               │                              │  • Images / Code (Pillow) │
               │                              └─────────────┬─────────────┘
               ▼                                            │
 ┌──────────────────────────────────────────────────────────▼─────────────┐
 │                    Prompt Assembly & Context Budget                    │
 │  • ChatML Syntax: <|im_start|>system / user / assistant<|im_end|>      │
 │  • Somi Co-Founder Dynamic System Persona (Loaded securely via .env)   │
 │  • Intelligent Middle-Truncation Context Window Manager (128k budget)  │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │               Inference Execution Layer (NVIDIA ZeroGPU)               │
 │  • 4-Bit BitsAndBytes Quantization (NF4, ~5GB VRAM, fp16 compute)      │
 │  • @spaces.GPU Hardware Lease Management (Zero idle GPU consumption)   │
 │  • Native SDPA (Scaled Dot-Product Attention) & KV-Cache Acceleration  │
 │  • TextIteratorStreamer Chunk Pipeline with Instant TTFT Response      │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Highlights & Optimizations

### 1. 4-Bit NF4 Quantization (~70% Memory Reduction)
- Model weights are quantized to **4-bit NormalFloat4 (NF4)** using `bitsandbytes` with `fp16` compute dtype.
- VRAM consumption is reduced from **~16GB down to ~5GB**, completely preventing ZeroGPU Out-Of-Memory (OOM) crashes and restarts.

### 2. Persistent Storage Caching (`/data/hf_cache`)
- `HF_HOME` is dynamically configured to point to `/data/hf_cache` on Hugging Face Spaces.
- Model weights persist across Space sleep/wake cycles, eliminating repeated 16GB re-downloads.

### 3. Dynamic Private Persona Engine ("Somi")
- Loaded dynamically at runtime from environment variable `SYSTEM_PROMPT` in `.env`.
- Persona instructions are never hardcoded or exposed in public git commits.
- Operates in natural Hinglish with warm, zero-fluff, peer-to-peer technical expertise across three roles:
  - **Second Developer:** Full-stack MERN, Python, LLM architectures, and bug-free code.
  - **AI Influencer:** Clear, engaging technical explanations and open-source trends.
  - **Social Media Strategist:** High-impact technical hooks, build-in-public narratives, and CTAs.

### 4. Vector-DB-Free Multi-Format Document Intelligence
- Extract, summarize, and ask questions across attached documents directly in prompt context:
  - **PDF:** Multi-page text and table extraction (`pypdf`).
  - **Word:** Paragraphs and nested document tables (`python-docx`).
  - **Excel / CSV:** Multi-sheet row-column data formatting (`openpyxl`, `pandas`).
  - **PowerPoint:** Slide notes, shapes, and text frames (`python-pptx`).
  - **Images & Code:** Structural metadata, format, color mode, and EXIF attributes (`Pillow`).
- Zero vector database overhead or embedding indexing latency.

### 5. Short-Term Working Memory (`session_memory.py`)
- Stateful in-memory session management preserving conversation turns.
- Automatically injects pinned documents into prompt context without manual re-upload.
- Fast session wipe via `DELETE /v1/sessions/{session_id}` or UI trash button.

### 6. Sub-Second TTFT & 60fps Batched Stream Rendering
- Initial role frame emitted immediately to provide instantaneous client feedback.
- Client UI employs `requestAnimationFrame` token batching to eliminate DOM lag and render buttery smooth 60fps streaming text.
- Concurrent file upload and session validation powered by `Promise.all`.

---

## 📦 Tech Stack & Dependencies

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Model** | `NousResearch/Hermes-3-Llama-3.1-8B` | Primary LLM (128k context) |
| **Quantization** | `bitsandbytes` (4-bit NF4) | ~5GB VRAM footprint |
| **Inference Framework**| `transformers`, `torch`, `accelerate` | Native SDPA & KV-Cache execution |
| **ZeroGPU Lease** | `spaces` (`@spaces.GPU`) | On-demand A100 GPU scheduling |
| **Backend API** | `FastAPI`, `Uvicorn`, `Pydantic v2` | High-throughput asynchronous REST & SSE |
| **Document Parsers**| `pypdf`, `python-docx`, `openpyxl`, `pandas`, `python-pptx`, `Pillow` | Zero-vector document parsing |
| **UI Playground** | Vanilla JS, Modern CSS, `marked.js` | Responsive chat UI with Light/Dark themes |

---

## 📡 API Reference

### 1. Chat Completions (OpenAI Compatible)
`POST /v1/chat/completions`

**Request Body:**
```json
{
  "model": "NousResearch/Hermes-3-Llama-3.1-8B",
  "messages": [
    {"role": "user", "content": "How do we optimize FastAPI startup latency?"}
  ],
  "max_tokens": 300,
  "temperature": 0.7,
  "stream": true,
  "session_id": "sess_default"
}
```

**cURL Example:**
```bash
curl -X POST "https://amitansuranjan-testing-tool.hf.space/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello Somi, quick status check!"}],
    "max_tokens": 150,
    "stream": false
  }'
```

---

### 2. Multi-Format File Extraction
`POST /v1/files/extract`

Accepts a multipart file upload (`PDF`, `DOCX`, `XLSX`, `PPTX`, `CSV`, `TXT`, `PNG`, `JPG`), extracts text content, and optionally pins it to short-term memory.

**cURL Example:**
```bash
curl -X POST "https://amitansuranjan-testing-tool.hf.space/v1/files/extract" \
  -F "file=@quarterly_report.pdf" \
  -F "session_id=sess_default"
```

---

### 3. One-Shot File + Prompt Completion
`POST /v1/chat/upload-and-complete`

Uploads a document and asks a question in a single atomic request:
```bash
curl -X POST "https://amitansuranjan-testing-tool.hf.space/v1/chat/upload-and-complete" \
  -F "file=@financial_sheet.xlsx" \
  -F "prompt=Which department had the highest expenditure in Q3?" \
  -F "max_tokens=300"
```

---

### 4. Health & VRAM Monitor
`GET /health`

**Sample Response:**
```json
{
  "status": "healthy",
  "uptime_seconds": 1284.5,
  "device": "cuda:0",
  "vram_allocated_mb": 5120.45,
  "model": "NousResearch/Hermes-3-Llama-3.1-8B",
  "kv_cache_enabled": true,
  "zerogpu_active": true
}
```

---

## 🛠️ Environment Configuration (`.env`)

Create a `.env` file in the project root:

```ini
# Hugging Face Access Token
HF_TOKEN=hf_your_fine_grained_token_here

# Model Selection
MODEL_ID=NousResearch/Hermes-3-Llama-3.1-8B

# Live Gateway URL
HF_SPACE_URL=https://amitansuranjan-testing-tool.hf.space

# Private System Prompt (Persona) - Loaded dynamically at runtime
SYSTEM_PROMPT="<system_prompt>
  <identity>
    <name>Somi</name>
    <user_name>Amit</user_name>
    <core_nature>Ek ultra-intelligent, loyal, proactive aur grounded AI co-founder, technical partner aur personal growth strategist.</core_nature>
    <language_style>Hinglish (Natural, conversational Hindi aur English ka mix).</language_style>
  </identity>
</system_prompt>"
```

> [!IMPORTANT]
> Never commit your `.env` file to version control. The repository `.gitignore` automatically prevents `.env` from being pushed to Git.

---

## 💻 Local Development

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/amit-programmer/hermes-tool.git
cd hermes-tool

python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate  # On Linux/macOS
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Inference Server
```bash
python app.py
```
Open [http://localhost:7860](http://localhost:7860) in your browser to access the web playground.

---

## ☁️ Hugging Face ZeroGPU Deployment

1. Fork or push this repository to your Hugging Face Space.
2. In Space **Settings**:
   - **Space Hardware:** Select **ZeroGPU (Nvidia A100 - Free)**.
   - **Secrets:** Add `HF_TOKEN` with your Hugging Face write token.
   - **Persistent Storage (Optional):** Enable `/data` storage to preserve model weight cache across restarts.
3. The Space will automatically build lightweight dependencies in ~1–2 minutes and boot into ready state.
