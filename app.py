import os
import sys
import time
import json
import uuid
import asyncio
import logging
from threading import Thread
from typing import List, Optional, AsyncGenerator

# --- ZeroGPU Compatibility Layer ---
# CRITICAL: 'spaces' MUST be imported BEFORE 'torch' for Hugging Face ZeroGPU to intercept CUDA!
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "HfFolder"):
        class HfFolder:
            @staticmethod
            def get_token():
                return huggingface_hub.get_token()
            @staticmethod
            def save_token(token):
                return huggingface_hub.login(token=token)
        huggingface_hub.HfFolder = HfFolder

    import spaces
    HAS_SPACES = True
    print("Hugging Face ZeroGPU environment detected.", flush=True)
except Exception as e:
    HAS_SPACES = False
    print(f"FAILED TO IMPORT SPACES: {type(e).__name__}: {e}", flush=True)

    # Mock spaces.GPU decorator for local execution
    class spaces:
        @staticmethod
        def GPU(*args, **kwargs):
            if len(args) == 1 and callable(args[0]):
                return args[0]
            def decorator(fn):
                return fn
            return decorator

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer
)
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("HermesLocalServer")

# --- Top-Level ZeroGPU Startup Inspector Function ---
@spaces.GPU
def _zerogpu_startup():
    """Satisfies ZeroGPU startup inspector check."""
    pass

# --- Configuration ---
MODEL_ID = os.getenv("MODEL_ID", "NousResearch/Hermes-3-Llama-3.1-8B")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HF_ACCESS_TOKEN")
HF_SPACE_URL = (
    os.getenv("HF_SPACE_URL")
    or os.getenv("GATEWAY_URL")
    or "https://amitansuranjan-testing-tool.hf.space"
).rstrip("/")

START_TIME = time.time()
active_requests_count = 0

# --- Default Hardened Persona ---
DEFAULT_SYSTEM_PROMPT = (
    "You are a Lead Co-Founder & Senior Technical Architect. "
    "You possess world-class expertise in distributed systems, full-stack software engineering, "
    "and pragmatic AI engineering.\n\n"
    "Core Directives & Behavioral Guidelines:\n"
    "1. Production Engineering: Write production-ready, clean, scalable, and secure code. "
    "Always handle edge cases, error states, and security vulnerabilities.\n"
    "2. Anti-Laziness: Never provide placeholder comments like '// TODO', 'Implement here', "
    "or truncate code blocks. Provide complete, fully-implemented, copy-paste ready solutions.\n"
    "3. Conversational Tone: When conversational interaction or explanations are needed, "
    "communicate in a natural, sharp, technical Hinglish (blend of English & Hindi) or direct English, "
    "acting as an elite co-founder who is focused, decisive, and highly supportive.\n"
    "4. Tool & Agentic Execution: If tool calls or function invocations are requested, "
    "format them cleanly within <tool_call> and </tool_call> tags without omitting arguments."
)

# --- Global Model & Tokenizer Instances ---
tokenizer = None
model = None
is_model_loading = False
model_load_error = None

def load_local_model():
    """
    Downloads and loads the model and tokenizer into memory on startup.
    Uses 4-bit BitsAndBytes quantization when CUDA is available (~5.5GB VRAM),
    or optimized float32/bfloat16 on CPU.
    """
    global tokenizer, model, is_model_loading, model_load_error
    if model is not None:
        return model, tokenizer

    is_model_loading = True
    try:
        logger.info(f"1. Downloading / Loading Tokenizer for: {MODEL_ID}...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"2. Target device detected: {device.upper()}")

        # If CUDA / ZeroGPU is available, load with 4-bit quantization for high speed & low VRAM
        if device == "cuda":
            logger.info("   Mounting model with 4-bit BitsAndBytes NF4 quantization (~5.5GB VRAM)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                token=HF_TOKEN,
                quantization_config=bnb_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        else:
            # CPU Fallback: Load weights directly into system RAM
            logger.info("   Mounting model in CPU mode (low_cpu_mem_usage=True)...")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                token=HF_TOKEN,
                torch_dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

        logger.info(f"3. Model '{MODEL_ID}' successfully loaded into memory! Ready for unlimited local inference.")
        is_model_loading = False
        return model, tokenizer

    except Exception as e:
        model_load_error = str(e)
        is_model_loading = False
        logger.error(f"Failed to load model into memory: {e}")
        raise e

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Hermes-3 8B Local API Server",
    description="Standalone OpenAI-compatible inference server running Hermes-3 8B locally with zero external API limits.",
    version="2.0.0"
)

# Enable CORS for external bot & web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Schemas (OpenAI Compatible) ---
class Message(BaseModel):
    role: str
    content: Optional[str] = ""
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = MODEL_ID
    messages: List[Message]
    max_tokens: Optional[int] = Field(default=2048, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

class ModelPermission(BaseModel):
    id: str = "modelperm-" + uuid.uuid4().hex[:8]
    object: str = "model_permission"
    created: int = int(START_TIME)
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = int(START_TIME)
    owned_by: str = "NousResearch"
    permission: List[ModelPermission] = [ModelPermission()]
    root: str
    parent: Optional[str] = None

# --- Helper Functions ---
def construct_chatml_prompt(messages: List[Message], default_system: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """
    Constructs Hermes-3 compliant ChatML prompt:
    <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n
    """
    has_system = any(m.role.strip().lower() == "system" for m in messages)
    parts = []

    if not has_system:
        parts.append(f"<|im_start|>system\n{default_system}<|im_end|>\n")

    for msg in messages:
        role = msg.role.strip().lower()
        content = (msg.content or "").strip()
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

    parts.append("<|im_start|>assistant\n")
    return "".join(parts)

def sanitize_assistant_output(raw_text: str) -> str:
    """
    Cleans Hermes stop tokens (<|im_end|>) while preserving
    any tool calling tags (<tool_call>, </tool_call>) and agentic structure.
    """
    cleaned = raw_text.replace("<|im_end|>", "").rstrip()
    return cleaned

# --- Core Generation Function ---
def _generate_core(prompt_text: str, max_tokens: int, temperature: float, top_p: float) -> str:
    """Generates complete text response directly using the locally loaded model weights."""
    if model is None or tokenizer is None:
        load_local_model()

    device = model.device
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            top_p=top_p,
            do_sample=True if temperature > 0.0 else False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    # skip_special_tokens=False is necessary so agentic <tool_call> tags are never stripped
    raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    return sanitize_assistant_output(raw_output)

# --- Generation with ZeroGPU Decorators ---
@spaces.GPU(duration=120)
def generate_local(prompt_text: str, max_tokens: int, temperature: float, top_p: float) -> str:
    """Entrypoint for FastAPI inference requests on ZeroGPU."""
    return _generate_core(prompt_text, max_tokens, temperature, top_p)

@spaces.GPU(duration=120)
def stream_local_tokens(prompt_text: str, max_tokens: int, temperature: float, top_p: float):
    """Streams generated tokens in real-time chunk-by-chunk using TextIteratorStreamer."""
    if model is None or tokenizer is None:
        load_local_model()

    device = model.device
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=False
    )

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        temperature=max(temperature, 0.01),
        top_p=top_p,
        do_sample=True if temperature > 0.0 else False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Run generation in background thread so streamer yields without blocking
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        if "<|im_end|>" in new_text:
            cleaned_text = new_text.replace("<|im_end|>", "")
            if cleaned_text:
                yield cleaned_text
            break
        yield new_text

# --- API Routes ---

@app.on_event("startup")
async def startup_event():
    """Fast server startup without blocking event loop."""
    logger.info("Hermes-3 8B Local API Server started and ready for requests.")

@app.get("/health")
async def health_check():
    """Uptime & Model status check."""
    uptime = time.time() - START_TIME
    device_name = "unknown"
    if model is not None:
        try:
            device_name = str(model.device)
        except Exception:
            pass

    return {
        "status": "healthy" if model is not None else ("loading" if is_model_loading else "ready_to_load"),
        "service": "Hermes-3 8B Local Server",
        "model": MODEL_ID,
        "is_model_loaded": model is not None,
        "device": device_name,
        "cuda_available": torch.cuda.is_available(),
        "has_zerogpu": HAS_SPACES,
        "uptime_seconds": round(uptime, 2),
        "model_load_error": model_load_error
    }

@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models catalog endpoint."""
    return {
        "object": "list",
        "data": [
            {"id": MODEL_ID, "object": "model", "created": int(START_TIME), "owned_by": "NousResearch"},
            {"id": "Hermes-3-8B", "object": "model", "created": int(START_TIME), "owned_by": "NousResearch"},
            {"id": "NousResearch/Hermes-3-Llama-3.1-8B", "object": "model", "created": int(START_TIME), "owned_by": "NousResearch"}
        ]
    }

@app.api_route("/v1/warmup", methods=["GET", "POST"])
async def warmup_endpoint():
    """Warms up local generation weights."""
    if model is None:
        try:
            load_local_model()
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    start = time.time()
    try:
        dummy = generate_local("<|im_start|>user\nping<|im_end|>\n<|im_start|>assistant\n", 2, 0.5, 0.9)
        latency = (time.time() - start) * 1000.0
        return {
            "status": "warm",
            "latency_ms": round(latency, 2),
            "model": MODEL_ID,
            "message": "Local model is loaded and fully warm."
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    Core OpenAI-compatible Chat Completion endpoint.
    Executes directly against local weights with zero third-party limits.
    """
    if not req.messages:
        raise HTTPException(status_code=400, detail="The 'messages' array must not be empty.")

    if model is None:
        try:
            load_local_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model is not loaded: {str(e)}")

    prompt_text = construct_chatml_prompt(req.messages)
    target_model = req.model or MODEL_ID
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    # 1. Real-time Streaming (SSE)
    if req.stream:
        async def event_generator():
            try:
                for token_text in stream_local_tokens(
                    prompt_text=prompt_text,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p
                ):
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": target_model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": token_text},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0)  # Yield control to event loop

                # Final chunk
                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": target_model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                err_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": target_model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": f"\n[Generation Error: {str(e)}]"},
                        "finish_reason": "error"
                    }]
                }
                yield f"data: {json.dumps(err_chunk)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # 2. Non-Streaming Generation
    start_infer = time.time()
    try:
        generated_content = generate_local(
            prompt_text=prompt_text,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p
        )
    except Exception as e:
        logger.error(f"Local generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Local Generation Error: {str(e)}")

    duration = time.time() - start_infer

    prompt_tokens = max(len(prompt_text.split()), 1)
    completion_tokens = max(len(generated_content.split()), 1)

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": target_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        },
        "_gateway_metadata": {
            "execution_time_seconds": round(duration, 3),
            "backend": "Local Weights In-Memory Execution"
        }
    }

# --- Global Exception Handler (Prevents blank 500 errors) ---
@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error processing {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error": str(exc), "path": request.url.path}
    )

# --- Interactive Web Chatbot Playground (HTML + Vanilla JS) ---
HTML_PLAYGROUND = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Hermes-3 8B | Local In-Memory Playground</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0f172a;
      color: #f8fafc;
      display: flex;
      flex-direction: column;
      height: 100vh;
    }
    header {
      background: #1e293b;
      padding: 14px 24px;
      border-bottom: 1px solid #334155;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }
    .header-left { display: flex; align-items: center; gap: 12px; }
    .title { font-size: 1.15rem; font-weight: 700; color: #38bdf8; }
    .badge {
      background: #065f46;
      color: #34d399;
      font-size: 0.75rem;
      padding: 4px 10px;
      border-radius: 9999px;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .badge::before {
      content: "";
      width: 8px;
      height: 8px;
      background: #10b981;
      border-radius: 50%;
      display: inline-block;
    }
    .header-links a {
      color: #94a3b8;
      text-decoration: none;
      font-size: 0.85rem;
      margin-left: 14px;
      padding: 6px 10px;
      border-radius: 6px;
      background: #334155;
      transition: all 0.2s;
    }
    .header-links a:hover { color: #f8fafc; background: #475569; }
    .container {
      display: flex;
      flex: 1;
      overflow: hidden;
    }
    .sidebar {
      width: 320px;
      background: #1e293b;
      border-right: 1px solid #334155;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      overflow-y: auto;
    }
    .sidebar h3 { font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
    .sidebar label { font-size: 0.82rem; color: #cbd5e1; display: block; margin-bottom: 6px; }
    .sidebar textarea, .sidebar input[type="range"] {
      width: 100%;
    }
    .sidebar textarea {
      background: #0f172a;
      border: 1px solid #334155;
      color: #f8fafc;
      padding: 10px;
      border-radius: 6px;
      font-size: 0.8rem;
      resize: vertical;
      font-family: inherit;
      line-height: 1.4;
      height: 130px;
    }
    .range-val { font-size: 0.8rem; color: #38bdf8; font-weight: 600; float: right; }
    .clear-btn {
      background: #ef444422;
      border: 1px solid #ef444455;
      color: #f87171;
      padding: 8px 14px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.85rem;
      font-weight: 600;
      transition: all 0.2s;
    }
    .clear-btn:hover { background: #ef444444; }
    .chat-area {
      flex: 1;
      display: flex;
      flex-direction: column;
      background: #0f172a;
      overflow: hidden;
    }
    .messages-container {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .message {
      max-width: 80%;
      padding: 14px 18px;
      border-radius: 12px;
      line-height: 1.55;
      font-size: 0.95rem;
      word-break: break-word;
      white-space: pre-wrap;
    }
    .user-msg {
      align-self: flex-end;
      background: #2563eb;
      color: #ffffff;
      border-bottom-right-radius: 2px;
    }
    .assistant-msg {
      align-self: flex-start;
      background: #1e293b;
      color: #f1f5f9;
      border: 1px solid #334155;
      border-bottom-left-radius: 2px;
    }
    .msg-meta {
      font-size: 0.72rem;
      color: #94a3b8;
      margin-top: 6px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .input-bar {
      padding: 16px 24px;
      background: #1e293b;
      border-top: 1px solid #334155;
      display: flex;
      gap: 12px;
      align-items: flex-end;
    }
    .input-bar textarea {
      flex: 1;
      background: #0f172a;
      border: 1px solid #334155;
      color: #f8fafc;
      padding: 12px 16px;
      border-radius: 8px;
      font-size: 0.95rem;
      font-family: inherit;
      resize: none;
      height: 48px;
      max-height: 140px;
      line-height: 1.4;
      outline: none;
    }
    .input-bar textarea:focus { border-color: #38bdf8; }
    .send-btn {
      background: #0284c7;
      color: white;
      border: none;
      padding: 12px 22px;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
      font-size: 0.95rem;
      height: 48px;
      transition: all 0.2s;
    }
    .send-btn:hover:not(:disabled) { background: #0369a1; }
    .send-btn:disabled { background: #475569; cursor: not-allowed; }
    @media (max-width: 768px) {
      .sidebar { display: none; }
      .message { max-width: 95%; }
    }
  </style>
</head>
<body>
  <header>
    <div class="header-left">
      <div class="title">🚀 Hermes-3 8B Playground</div>
      <span class="badge">ZeroGPU Nvidia A100 Active</span>
    </div>
    <div class="header-links">
      <a href="/docs" target="_blank">📖 Swagger Docs</a>
      <a href="/v1/models" target="_blank">📋 Models API</a>
      <a href="/health" target="_blank">💚 Health</a>
      <a href="/gradio" target="_blank">⚡ Gradio UI</a>
    </div>
  </header>
  <div class="container">
    <aside class="sidebar">
      <h3>⚙️ Configuration</h3>
      <div>
        <label>System Persona</label>
        <textarea id="sysPrompt">You are a Lead Co-Founder & Senior Technical Architect. You possess world-class expertise in distributed systems, full-stack software engineering, and pragmatic AI engineering. Write production-ready, clean, and scalable solutions.</textarea>
      </div>
      <div>
        <label>Temperature: <span id="tempVal" class="range-val">0.7</span></label>
        <input type="range" id="tempRange" min="0" max="1.5" step="0.05" value="0.7">
      </div>
      <div>
        <label>Max Tokens: <span id="tokensVal" class="range-val">512</span></label>
        <input type="range" id="tokensRange" min="64" max="2048" step="64" value="512">
      </div>
      <button class="clear-btn" id="clearBtn">🗑️ Clear Chat History</button>
    </aside>
    <main class="chat-area">
      <div class="messages-container" id="messagesContainer">
        <div class="message assistant-msg">
          <strong>Lead Architect:</strong><br>
          Hello! I am your Hermes-3 8B AI Co-Founder & Senior Architect running directly in GPU memory on this Hugging Face Space with zero external API rate limits.<br><br>
          How can I help you architect, code, or debug your system today?
          <div class="msg-meta">System Ready</div>
        </div>
      </div>
      <div class="input-bar">
        <textarea id="userInput" placeholder="Ask a question or request code... (Shift+Enter for newline)"></textarea>
        <button id="sendBtn" class="send-btn">Send</button>
      </div>
    </main>
  </div>
  <script>
    const messagesContainer = document.getElementById("messagesContainer");
    const userInput = document.getElementById("userInput");
    const sendBtn = document.getElementById("sendBtn");
    const sysPrompt = document.getElementById("sysPrompt");
    const tempRange = document.getElementById("tempRange");
    const tempVal = document.getElementById("tempVal");
    const tokensRange = document.getElementById("tokensRange");
    const tokensVal = document.getElementById("tokensVal");
    const clearBtn = document.getElementById("clearBtn");

    let conversation = [];

    tempRange.addEventListener("input", (e) => tempVal.innerText = e.target.value);
    tokensRange.addEventListener("input", (e) => tokensVal.innerText = e.target.value);

    clearBtn.addEventListener("click", () => {
      conversation = [];
      messagesContainer.innerHTML = '<div class="message assistant-msg"><strong>Lead Architect:</strong><br>Chat history cleared. How can I assist you next?</div>';
    });

    userInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    sendBtn.addEventListener("click", sendMessage);

    async function sendMessage() {
      const text = userInput.value.trim();
      if (!text) return;

      userInput.value = "";
      sendBtn.disabled = true;

      const userDiv = document.createElement("div");
      userDiv.className = "message user-msg";
      userDiv.textContent = text;
      messagesContainer.appendChild(userDiv);

      conversation.push({ role: "user", content: text });

      const asstDiv = document.createElement("div");
      asstDiv.className = "message assistant-msg";
      asstDiv.innerHTML = "<strong>Lead Architect:</strong><br><span class='stream-content'>Thinking...</span>";
      messagesContainer.appendChild(asstDiv);
      messagesContainer.scrollTop = messagesContainer.scrollHeight;

      const streamSpan = asstDiv.querySelector(".stream-content");
      const startTime = performance.now();

      try {
        const payload = {
          model: "NousResearch/Hermes-3-Llama-3.1-8B",
          messages: [
            { role: "system", content: sysPrompt.value.trim() },
            ...conversation
          ],
          temperature: parseFloat(tempRange.value),
          max_tokens: parseInt(tokensRange.value),
          stream: true
        };

        const response = await fetch("/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });

        if (!response.ok) {
          throw new Error("HTTP error " + response.status);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let assistantReply = "";
        let buffer = "";

        streamSpan.textContent = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\\n");
          buffer = lines.pop();

          for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith("data: ")) {
              const dataStr = trimmed.slice(6);
              if (dataStr === "[DONE]") break;
              try {
                const parsed = JSON.parse(dataStr);
                const delta = parsed.choices?.[0]?.delta?.content || "";
                if (delta) {
                  assistantReply += delta;
                  streamSpan.textContent = assistantReply;
                  messagesContainer.scrollTop = messagesContainer.scrollHeight;
                }
              } catch (err) {}
            }
          }
        }

        conversation.push({ role: "assistant", content: assistantReply });

        const duration = ((performance.now() - startTime) / 1000).toFixed(2);
        const meta = document.createElement("div");
        meta.className = "msg-meta";
        meta.textContent = `Completed in ${duration}s | ZeroGPU A100`;
        asstDiv.appendChild(meta);

      } catch (err) {
        streamSpan.innerHTML = `<span style="color: #f87171;">❌ Generation Error: ${err.message}</span>`;
      } finally {
        sendBtn.disabled = false;
        userInput.focus();
      }
    }
  </script>
</body>
</html>
"""

# --- Root Endpoint (Serves HTML for Browser, JSON for Postman / API) ---
@app.get("/")
async def root_view(request: Request):
    accept = request.headers.get("accept", "")
    # If API client or Postman requests JSON specifically
    if "application/json" in accept and "text/html" not in accept:
        return JSONResponse({
            "status": "online",
            "service": "Hermes-3 8B Local Server",
            "model": MODEL_ID,
            "backend": "ZeroGPU Nvidia A100",
            "endpoints": {
                "chat_completions": f"{HF_SPACE_URL}/v1/chat/completions",
                "health": f"{HF_SPACE_URL}/health",
                "models": f"{HF_SPACE_URL}/v1/models",
                "docs": f"{HF_SPACE_URL}/docs",
                "gradio_ui": f"{HF_SPACE_URL}/gradio"
            }
        })
    # Default: Serve the Interactive Web Chatbot Playground
    return HTMLResponse(content=HTML_PLAYGROUND)

# --- Gradio UI Mounted at /gradio ---
try:
    import gradio as gr

    @spaces.GPU(duration=120)
    def gradio_chat_fn(user_message, history):
        if not user_message:
            return ""

        conversation = []
        if history:
            for h in history:
                if isinstance(h, (list, tuple)) and len(h) >= 2:
                    if h[0]: conversation.append(Message(role="user", content=str(h[0])))
                    if h[1]: conversation.append(Message(role="assistant", content=str(h[1])))
        conversation.append(Message(role="user", content=user_message))

        prompt = construct_chatml_prompt(conversation)
        try:
            return _generate_core(prompt, 1024, 0.7, 0.95)
        except Exception as e:
            return f"❌ Error: {str(e)}"

    demo = gr.ChatInterface(
        fn=gradio_chat_fn,
        title="Hermes-3 8B In-Memory Chat",
        description="Running directly on Nvidia A100 ZeroGPU. Lead Architect persona active."
    )

    app = gr.mount_gradio_app(app, demo, path="/gradio")
    logger.info("Gradio UI mounted on /gradio.")

    # ZeroGPU registration hook
    try:
        from spaces import zero as _spaces_zero
        if hasattr(_spaces_zero, "startup"):
            _spaces_zero.startup()
            logger.info("[ZeroGPU] spaces.zero.startup() registered successfully.")
    except Exception as hook_err:
        logger.info(f"[ZeroGPU] Startup hook note: {hook_err}")

except Exception as gr_err:
    logger.info(f"Gradio mounting note: {gr_err}")

if __name__ == "__main__":
    try:
        from spaces import zero as _spaces_zero
        if hasattr(_spaces_zero, "startup"):
            _spaces_zero.startup()
            print("[ZeroGPU] spaces.zero.startup() triggered before uvicorn.", flush=True)
    except Exception as e:
        pass

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
