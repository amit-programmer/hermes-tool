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
    print(f"ZeroGPU not detected or failed to import: {type(e).__name__}: {e}", flush=True)

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

def get_eos_token_ids(tok) -> List[int]:
    """Retrieves all stop token IDs for Hermes-3 / Llama 3.1."""
    stop_ids = set()
    if tok.eos_token_id is not None:
        if isinstance(tok.eos_token_id, list):
            stop_ids.update(tok.eos_token_id)
        else:
            stop_ids.add(tok.eos_token_id)
    # Crucial Hermes-3 & Llama 3.1 stop tokens
    for token_str in ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>"]:
        try:
            tid = tok.convert_tokens_to_ids(token_str)
            if tid is not None and isinstance(tid, int) and tid > 0:
                stop_ids.add(tid)
        except Exception:
            pass
    return list(stop_ids)

def load_local_model():
    """
    Downloads and loads the model and tokenizer into memory.
    Uses native bfloat16 on CUDA / ZeroGPU for maximum Tensor Core throughput.
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
        logger.info(f"2. Target device detected: {device.upper()} (ZeroGPU: {HAS_SPACES})")

        dtype = torch.bfloat16 if (torch.cuda.is_available() or HAS_SPACES or torch.cuda.is_bf16_supported()) else torch.float32
        logger.info(f"   Mounting model with native dtype: {dtype}...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        logger.info(f"3. Model '{MODEL_ID}' successfully loaded into memory! Ready for ultra-low latency inference.")
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
    version="2.1.0"
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
    cleaned = (
        raw_text.replace("<|im_end|>", "")
        .replace("<|eot_id|>", "")
        .replace("<|end_of_text|>", "")
        .rstrip()
    )
    return cleaned

# --- Core Generation Function ---
def _generate_core(prompt_text: str, max_tokens: int, temperature: float, top_p: float) -> str:
    """Generates complete text response directly using the locally loaded model weights."""
    if model is None or tokenizer is None:
        load_local_model()

    device = model.device
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    stop_ids = get_eos_token_ids(tokenizer)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            top_p=top_p,
            do_sample=True if temperature > 0.0 else False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_ids
        )

    generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
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
    stop_ids = get_eos_token_ids(tokenizer)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,
        timeout=60.0
    )

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        temperature=max(temperature, 0.01),
        top_p=top_p,
        do_sample=True if temperature > 0.0 else False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=stop_ids
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    stop_tokens = ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>"]
    for new_text in streamer:
        has_stop = False
        for st in stop_tokens:
            if st in new_text:
                cleaned = new_text.replace(st, "")
                if cleaned:
                    yield cleaned
                has_stop = True
                break
        if has_stop:
            break
        yield new_text

# --- API Routes ---

@app.on_event("startup")
async def startup_event():
    """Fast server startup with background model preloader and JIT warmup."""
    logger.info("Hermes-3 8B Local API Server initialized.")
    def _preload_and_warm():
        try:
            logger.info("Background preloading model into memory...")
            load_local_model()
            logger.info("Background model loading complete! Pre-warming JIT kernels...")
            try:
                generate_local("<|im_start|>user\nping<|im_end|>\n<|im_start|>assistant\n", 2, 0.5, 0.9)
                logger.info("JIT kernels pre-warmed successfully!")
            except Exception as w_err:
                logger.warning(f"Startup warmup note: {w_err}")
        except Exception as e:
            logger.warning(f"Background preload note: {e}")

    Thread(target=_preload_and_warm, daemon=True).start()

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

# --- Clean Full-Width Interactive Web Chatbot Playground (HTML5 + CSS3 + Vanilla JS) ---
HTML_PLAYGROUND = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Hermes-3 8B | Enterprise AI Architect</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <!-- Markdown & Syntax Highlighting -->
  <script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <style>
    :root {
      --bg-dark: #090d16;
      --bg-card: #111827;
      --bg-surface: #1e293b;
      --border-color: #27354a;
      --border-hover: #3b82f6;
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --accent: #8b5cf6;
      --accent-glow: rgba(139, 92, 246, 0.25);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --success: #10b981;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg-dark);
      color: var(--text-main);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }

    /* Top Navigation Header */
    .top-header {
      background: #0d1322;
      border-bottom: 1px solid var(--border-color);
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      z-index: 20;
    }
    .brand-group {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-avatar {
      width: 36px;
      height: 36px;
      border-radius: 9px;
      background: linear-gradient(135deg, #3b82f6, #8b5cf6);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.2rem;
      box-shadow: 0 0 14px var(--accent-glow);
    }
    .brand-details {
      display: flex;
      flex-direction: column;
    }
    .brand-title {
      font-size: 1.05rem;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .brand-badge {
      font-size: 0.65rem;
      color: #38bdf8;
      background: rgba(56, 189, 248, 0.12);
      padding: 2px 7px;
      border-radius: 4px;
      font-weight: 600;
      letter-spacing: 0.04em;
    }
    .status-pill {
      font-size: 0.72rem;
      color: #34d399;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      margin-top: 2px;
    }
    .status-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 8px #10b981;
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0% { opacity: 0.4; }
      50% { opacity: 1; }
      100% { opacity: 0.4; }
    }

    /* Header Nav Actions */
    .header-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .nav-btn {
      background: transparent;
      border: 1px solid var(--border-color);
      color: var(--text-muted);
      padding: 7px 12px;
      border-radius: 7px;
      font-size: 0.8rem;
      font-weight: 500;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      text-decoration: none;
      transition: all 0.2s;
    }
    .nav-btn:hover {
      background: var(--bg-surface);
      color: #fff;
      border-color: #3b82f6;
    }
    .nav-btn.primary {
      background: linear-gradient(135deg, #2563eb, #4f46e5);
      border: 1px solid rgba(255, 255, 255, 0.15);
      color: #ffffff;
      font-weight: 600;
      box-shadow: 0 4px 10px rgba(37, 99, 235, 0.25);
    }
    .nav-btn.primary:hover {
      background: linear-gradient(135deg, #1d4ed8, #4338ca);
      transform: translateY(-1px);
    }
    .nav-btn.danger:hover {
      background: rgba(239, 68, 68, 0.1);
      border-color: rgba(239, 68, 68, 0.4);
      color: #f87171;
    }

    /* Main Chat Layout */
    .chat-layout {
      flex: 1;
      display: flex;
      flex-direction: column;
      height: calc(100vh - 65px);
      background: var(--bg-dark);
      position: relative;
    }

    /* Message Stream */
    .chat-scroll {
      flex: 1;
      overflow-y: auto;
      padding: 24px 16px;
      scroll-behavior: smooth;
    }
    .chat-container {
      max-width: 880px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 22px;
    }

    /* Welcome Hero */
    .hero-container {
      text-align: center;
      padding: 40px 20px 20px;
      animation: fadeIn 0.4s ease-out;
    }
    .hero-orb {
      width: 64px;
      height: 64px;
      margin: 0 auto 18px;
      border-radius: 20px;
      background: linear-gradient(135deg, #3b82f6, #8b5cf6);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 2rem;
      box-shadow: 0 0 30px rgba(99, 102, 241, 0.4);
    }
    .hero-title {
      font-size: 1.6rem;
      font-weight: 700;
      color: #fff;
      margin-bottom: 8px;
      letter-spacing: -0.02em;
    }
    .hero-subtitle {
      font-size: 0.92rem;
      color: var(--text-muted);
      max-width: 520px;
      margin: 0 auto 30px;
      line-height: 1.5;
    }
    .prompt-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
      text-align: left;
    }
    .prompt-card {
      background: #111827;
      border: 1px solid var(--border-color);
      padding: 14px 16px;
      border-radius: 10px;
      cursor: pointer;
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .prompt-card:hover {
      border-color: var(--border-hover);
      transform: translateY(-2px);
      box-shadow: 0 4px 14px rgba(59, 130, 246, 0.15);
    }
    .prompt-card-icon {
      font-size: 1.2rem;
      margin-bottom: 6px;
    }
    .prompt-card-title {
      font-size: 0.88rem;
      font-weight: 600;
      color: #f1f5f9;
      margin-bottom: 4px;
    }
    .prompt-card-desc {
      font-size: 0.78rem;
      color: var(--text-dim);
      line-height: 1.35;
    }

    /* Message Bubbles */
    .message-row {
      display: flex;
      gap: 14px;
      animation: messageSlide 0.25s ease-out;
    }
    @keyframes messageSlide {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .message-row.user {
      justify-content: flex-end;
    }
    .msg-avatar {
      width: 34px;
      height: 34px;
      border-radius: 8px;
      flex-shrink: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1rem;
    }
    .msg-avatar.assistant {
      background: linear-gradient(135deg, #3b82f6, #8b5cf6);
      box-shadow: 0 0 10px var(--accent-glow);
    }
    .msg-avatar.user {
      background: #334155;
    }
    .msg-body {
      max-width: 84%;
      display: flex;
      flex-direction: column;
    }
    .message-row.user .msg-body {
      align-items: flex-end;
    }
    .msg-header {
      font-size: 0.76rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 6px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .architect-tag {
      background: rgba(59, 130, 246, 0.15);
      color: #60a5fa;
      font-size: 0.68rem;
      padding: 1px 6px;
      border-radius: 4px;
    }
    .msg-bubble {
      padding: 14px 18px;
      border-radius: 12px;
      font-size: 0.94rem;
      line-height: 1.6;
      word-break: break-word;
    }
    .message-row.user .msg-bubble {
      background: #2563eb;
      color: #ffffff;
      border-bottom-right-radius: 2px;
      white-space: pre-wrap;
    }
    .message-row.assistant .msg-bubble {
      background: #111827;
      border: 1px solid var(--border-color);
      color: #e2e8f0;
      border-bottom-left-radius: 2px;
    }

    /* Markdown Inside Assistant Bubble */
    .msg-bubble p { margin-bottom: 12px; }
    .msg-bubble p:last-child { margin-bottom: 0; }
    .msg-bubble h1, .msg-bubble h2, .msg-bubble h3 {
      color: #fff;
      margin: 16px 0 8px;
      font-weight: 600;
    }
    .msg-bubble h1 { font-size: 1.25rem; }
    .msg-bubble h2 { font-size: 1.1rem; }
    .msg-bubble h3 { font-size: 1.0rem; }
    .msg-bubble ul, .msg-bubble ol {
      margin-left: 20px;
      margin-bottom: 12px;
    }
    .msg-bubble li { margin-bottom: 4px; }
    .msg-bubble blockquote {
      border-left: 3px solid var(--primary);
      padding-left: 12px;
      color: var(--text-muted);
      margin: 12px 0;
      font-style: italic;
    }
    .msg-bubble code {
      background: rgba(255, 255, 255, 0.08);
      font-family: 'JetBrains Mono', monospace;
      padding: 2px 5px;
      border-radius: 4px;
      font-size: 0.86rem;
      color: #f43f5e;
    }

    /* Code Blocks */
    .code-block-wrapper {
      position: relative;
      margin: 14px 0;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid #334155;
      background: #0f172a;
    }
    .code-block-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #1e293b;
      padding: 6px 14px;
      font-size: 0.75rem;
      color: #94a3b8;
      font-family: 'JetBrains Mono', monospace;
      border-bottom: 1px solid #334155;
    }
    .copy-btn {
      background: transparent;
      border: none;
      color: #cbd5e1;
      font-size: 0.72rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 4px;
      transition: color 0.2s;
    }
    .copy-btn:hover { color: #38bdf8; }
    .code-block-wrapper pre {
      margin: 0;
      padding: 14px;
      overflow-x: auto;
    }
    .code-block-wrapper pre code {
      background: transparent !important;
      padding: 0 !important;
      color: inherit;
      font-size: 0.85rem;
      font-family: 'JetBrains Mono', monospace;
      line-height: 1.5;
    }

    /* Telemetry Footer */
    .msg-telemetry {
      font-size: 0.7rem;
      color: var(--text-dim);
      margin-top: 6px;
      display: flex;
      align-items: center;
      gap: 8px;
      font-family: 'JetBrains Mono', monospace;
    }

    /* Typing Dots */
    .thinking-box {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--text-muted);
      font-size: 0.85rem;
      padding: 6px 0;
    }
    .typing-dots span {
      display: inline-block;
      width: 6px;
      height: 6px;
      background: #60a5fa;
      border-radius: 50%;
      margin-right: 3px;
      animation: bounce 1.2s infinite ease-in-out;
    }
    .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce {
      0%, 80%, 100% { transform: scale(0); opacity: 0.4; }
      40% { transform: scale(1); opacity: 1; }
    }

    /* Live Cursor */
    .streaming-cursor {
      display: inline-block;
      width: 7px;
      height: 14px;
      background: #38bdf8;
      margin-left: 2px;
      animation: blink 0.9s infinite;
      vertical-align: middle;
    }
    @keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0; } }

    /* Floating Input Dock */
    .input-dock {
      padding: 14px 20px 20px;
      background: linear-gradient(180deg, transparent 0%, var(--bg-dark) 40%);
    }
    .input-wrapper {
      max-width: 880px;
      margin: 0 auto;
      background: #111827;
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 10px 14px;
      display: flex;
      align-items: flex-end;
      gap: 10px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
      transition: border-color 0.2s;
    }
    .input-wrapper:focus-within {
      border-color: var(--border-hover);
      box-shadow: 0 8px 24px rgba(59, 130, 246, 0.15);
    }
    .chat-textarea {
      flex: 1;
      background: transparent;
      border: none;
      color: #f8fafc;
      font-size: 0.95rem;
      font-family: inherit;
      resize: none;
      height: 38px;
      max-height: 160px;
      line-height: 1.45;
      outline: none;
      padding: 6px 4px;
    }
    .chat-textarea::placeholder { color: var(--text-dim); }
    .send-action-btn {
      width: 38px;
      height: 38px;
      border-radius: 8px;
      background: var(--primary);
      color: white;
      border: none;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.1rem;
      transition: all 0.2s;
      flex-shrink: 0;
    }
    .send-action-btn:hover:not(:disabled) {
      background: var(--primary-hover);
      transform: scale(1.03);
    }
    .send-action-btn:disabled {
      background: #1e293b;
      color: #475569;
      cursor: not-allowed;
    }
    .send-action-btn.stop {
      background: #ef4444;
    }
    .send-action-btn.stop:hover {
      background: #dc2626;
    }
    .input-hint {
      text-align: center;
      font-size: 0.72rem;
      color: var(--text-dim);
      margin-top: 8px;
    }
  </style>
</head>
<body>
  <!-- Top Navigation Header -->
  <header class="top-header">
    <div class="brand-group">
      <div class="brand-avatar">⚡</div>
      <div class="brand-details">
        <div class="brand-title">
          <span>Hermes-3 8B</span>
          <span class="brand-badge">NVIDIA A100 ZEROGPU</span>
        </div>
        <div class="status-pill">
          <div class="status-dot"></div>
          <span>Connected to Local In-Memory Weights</span>
        </div>
      </div>
    </div>
    <div class="header-actions">
      <button class="nav-btn primary" id="newChatBtn">
        <span>+</span> New Chat
      </button>
      <a href="/docs" target="_blank" class="nav-btn">
        <span>📖</span> Docs
      </a>
      <a href="/v1/models" target="_blank" class="nav-btn">
        <span>📋</span> Models
      </a>
      <a href="/health" target="_blank" class="nav-btn">
        <span>💚</span> Health
      </a>
      <a href="/gradio" target="_blank" class="nav-btn">
        <span>⚡</span> Gradio
      </a>
      <button class="nav-btn danger" id="clearChatBtn" title="Clear Chat">
        <span>🗑️</span> Clear
      </button>
    </div>
  </header>

  <!-- Main Chat Layout -->
  <main class="chat-layout">
    <!-- Scrollable Messages Container -->
    <div class="chat-scroll" id="chatScroll">
      <div class="chat-container" id="chatContainer">
        <!-- Hero Welcome -->
        <div class="hero-container" id="heroWelcome">
          <div class="hero-orb">⚡</div>
          <h1 class="hero-title">How can I help you build today?</h1>
          <p class="hero-subtitle">
            Hermes-3 8B running locally on Nvidia A100 ZeroGPU with native bfloat16 acceleration.
            Zero external rate limits, instant streaming, and production-grade engineering.
          </p>

          <div class="prompt-grid">
            <div class="prompt-card" onclick="useStarterPrompt('Design a distributed real-time notification engine with Redis and WebSockets.')">
              <div class="prompt-card-icon">🏗️</div>
              <div class="prompt-card-title">Distributed Systems</div>
              <div class="prompt-card-desc">Design a distributed notification engine with Redis pub/sub.</div>
            </div>
            <div class="prompt-card" onclick="useStarterPrompt('Write an async FastAPI microservice with Pydantic v2 validation and JWT authentication.')">
              <div class="prompt-card-icon">⚡</div>
              <div class="prompt-card-title">FastAPI Backend</div>
              <div class="prompt-card-desc">Write an async FastAPI service with Pydantic v2 schemas.</div>
            </div>
            <div class="prompt-card" onclick="useStarterPrompt('Demonstrate the Hermes-3 XML tool calling schema for web browsing and SQL queries.')">
              <div class="prompt-card-icon">🧠</div>
              <div class="prompt-card-title">Agentic Tool Calling</div>
              <div class="prompt-card-desc">Demonstrate tool calling schemas formatted in XML tags.</div>
            </div>
            <div class="prompt-card" onclick="useStarterPrompt('Review this Python async code for concurrency race conditions, memory leaks, and performance bottlenecks.')">
              <div class="prompt-card-icon">🛡️</div>
              <div class="prompt-card-title">Code Audit & Security</div>
              <div class="prompt-card-desc">Audit async Python code for race conditions and memory leaks.</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Input Dock -->
    <div class="input-dock">
      <div class="input-wrapper">
        <textarea id="chatInput" class="chat-textarea" placeholder="Message Hermes-3 8B... (Shift + Enter for newline)" rows="1"></textarea>
        <button id="sendBtn" class="send-action-btn" title="Send Message">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
        </button>
      </div>
      <div class="input-hint">
        Hermes-3 8B running in-memory on Hugging Face ZeroGPU • Press Enter to send, Shift + Enter for newline
      </div>
    </div>
  </main>

  <script>
    // State
    let conversation = [];
    let isGenerating = false;
    let abortController = null;

    // Elements
    const chatContainer = document.getElementById("chatContainer");
    const chatScroll = document.getElementById("chatScroll");
    const chatInput = document.getElementById("chatInput");
    const sendBtn = document.getElementById("sendBtn");
    const heroWelcome = document.getElementById("heroWelcome");
    const newChatBtn = document.getElementById("newChatBtn");
    const clearChatBtn = document.getElementById("clearChatBtn");

    // Configure Marked
    marked.setOptions({
      highlight: function(code, lang) {
        if (lang && hljs.getLanguage(lang)) {
          return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
      },
      breaks: true
    });

    // Custom Renderer for Copy Code buttons
    const renderer = new marked.Renderer();
    renderer.code = function(code, language) {
      const validLang = language || 'plaintext';
      const highlighted = (language && hljs.getLanguage(language))
        ? hljs.highlight(code, { language }).value
        : hljs.highlightAuto(code).value;
      const id = 'code_' + Math.random().toString(36).substr(2, 9);
      return `
        <div class="code-block-wrapper">
          <div class="code-block-header">
            <span>${validLang.toUpperCase()}</span>
            <button class="copy-btn" onclick="copyCode('${id}')">📋 Copy</button>
          </div>
          <pre><code id="${id}" class="hljs ${validLang}">${highlighted}</code></pre>
        </div>
      `;
    };
    marked.use({ renderer });

    window.copyCode = function(id) {
      const el = document.getElementById(id);
      if (el) {
        navigator.clipboard.writeText(el.innerText).then(() => {
          const btn = el.closest('.code-block-wrapper').querySelector('.copy-btn');
          const original = btn.innerHTML;
          btn.innerHTML = '✅ Copied!';
          setTimeout(() => { btn.innerHTML = original; }, 2000);
        });
      }
    };

    // Textarea Auto-Resize
    chatInput.addEventListener("input", function() {
      this.style.height = 'auto';
      this.style.height = Math.min(this.scrollHeight, 160) + 'px';
    });

    // Keydown Handling
    chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (!isGenerating) {
          sendMessage();
        }
      }
    });

    sendBtn.addEventListener("click", () => {
      if (isGenerating) {
        if (abortController) abortController.abort();
      } else {
        sendMessage();
      }
    });

    if (newChatBtn) newChatBtn.addEventListener("click", resetChat);
    if (clearChatBtn) clearChatBtn.addEventListener("click", resetChat);

    function resetChat() {
      if (isGenerating && abortController) abortController.abort();
      conversation = [];
      chatContainer.innerHTML = "";
      chatContainer.appendChild(heroWelcome);
      heroWelcome.style.display = "block";
      chatInput.value = "";
      chatInput.style.height = "38px";
      chatInput.focus();
    }

    window.useStarterPrompt = function(promptText) {
      chatInput.value = promptText;
      sendMessage();
    };

    async function sendMessage() {
      const userText = chatInput.value.trim();
      if (!userText || isGenerating) return;

      if (heroWelcome) heroWelcome.style.display = "none";

      // Append User Message
      appendMessage("user", userText);
      conversation.push({ role: "user", content: userText });

      chatInput.value = "";
      chatInput.style.height = "38px";

      // Prepare Assistant Placeholder
      isGenerating = true;
      setButtonState(true);
      abortController = new AbortController();

      const startTime = performance.now();
      let firstTokenTime = null;
      let tokenCount = 0;

      const assistantMsgElements = createAssistantMessageHolder();
      const contentEl = assistantMsgElements.content;
      const telemetryEl = assistantMsgElements.telemetry;

      let fullResponse = "";

      try {
        const payload = {
          model: "NousResearch/Hermes-3-Llama-3.1-8B",
          messages: [
            { role: "system", content: "You are a Lead Co-Founder & Senior Technical Architect. You possess world-class expertise in distributed systems, full-stack software engineering, and pragmatic AI engineering. Write production-ready, clean, scalable, and secure code." },
            ...conversation
          ],
          temperature: 0.7,
          max_tokens: 1024,
          stream: true
        };

        const response = await fetch("/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: abortController.signal
        });

        if (!response.ok) {
          throw new Error(`Server returned HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        contentEl.innerHTML = ""; // Clear "thinking..." dots

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          if (!firstTokenTime) {
            firstTokenTime = performance.now();
          }

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
                  fullResponse += delta;
                  tokenCount++;
                  renderStreamedContent(contentEl, fullResponse, true);
                  chatScroll.scrollTop = chatScroll.scrollHeight;
                }
              } catch (err) {}
            }
          }
        }

        // Final clean render without cursor
        renderStreamedContent(contentEl, fullResponse, false);
        conversation.push({ role: "assistant", content: fullResponse });

        const totalDuration = ((performance.now() - startTime) / 1000).toFixed(2);
        const tokPerSec = tokenCount > 0 ? (tokenCount / Math.max(totalDuration, 0.1)).toFixed(1) : 0;
        telemetryEl.textContent = `⚡ ${totalDuration}s • ${tokPerSec} tok/s • ZeroGPU A100`;

      } catch (err) {
        if (err.name === 'AbortError') {
          renderStreamedContent(contentEl, fullResponse, false);
          telemetryEl.textContent = `⚠️ Generation stopped by user.`;
        } else {
          contentEl.innerHTML = `<span style="color: #ef4444;">❌ Error: ${err.message}</span>`;
        }
      } finally {
        isGenerating = false;
        setButtonState(false);
        abortController = null;
        chatInput.focus();
      }
    }

    function renderStreamedContent(el, text, isStreaming) {
      let rendered = marked.parse(text);
      if (isStreaming) {
        rendered += `<span class="streaming-cursor"></span>`;
      }
      el.innerHTML = rendered;
    }

    function appendMessage(role, text) {
      const row = document.createElement("div");
      row.className = `message-row ${role}`;

      const avatar = document.createElement("div");
      avatar.className = `msg-avatar ${role}`;
      avatar.textContent = role === "user" ? "👤" : "⚡";

      const body = document.createElement("div");
      body.className = "msg-body";

      const bubble = document.createElement("div");
      bubble.className = "msg-bubble";
      bubble.textContent = text;

      body.appendChild(bubble);
      row.appendChild(body);
      row.appendChild(avatar);

      chatContainer.appendChild(row);
      chatScroll.scrollTop = chatScroll.scrollHeight;
    }

    function createAssistantMessageHolder() {
      const row = document.createElement("div");
      row.className = "message-row assistant";

      const avatar = document.createElement("div");
      avatar.className = "msg-avatar assistant";
      avatar.textContent = "⚡";

      const body = document.createElement("div");
      body.className = "msg-body";

      const header = document.createElement("div");
      header.className = "msg-header";
      header.innerHTML = `<span>Lead Architect</span><span class="architect-tag">ZeroGPU</span>`;

      const bubble = document.createElement("div");
      bubble.className = "msg-bubble";

      const content = document.createElement("div");
      content.innerHTML = `
        <div class="thinking-box">
          <div class="typing-dots">
            <span></span><span></span><span></span>
          </div>
          <span>Lead Architect is thinking...</span>
        </div>
      `;

      const telemetry = document.createElement("div");
      telemetry.className = "msg-telemetry";

      bubble.appendChild(content);
      body.appendChild(header);
      body.appendChild(bubble);
      body.appendChild(telemetry);

      row.appendChild(avatar);
      row.appendChild(body);

      chatContainer.appendChild(row);
      chatScroll.scrollTop = chatScroll.scrollHeight;

      return { row, content, telemetry };
    }

    function setButtonState(generating) {
      if (generating) {
        sendBtn.classList.add("stop");
        sendBtn.title = "Stop Generation";
        sendBtn.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <rect x="4" y="4" width="16" height="16" rx="2"></rect>
          </svg>
        `;
      } else {
        sendBtn.classList.remove("stop");
        sendBtn.title = "Send Message";
        sendBtn.innerHTML = `
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
        `;
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
                "warmup": f"{HF_SPACE_URL}/v1/warmup",
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
