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
    import spaces
    HAS_SPACES = True
    print("Hugging Face ZeroGPU environment detected.")
except ImportError:
    HAS_SPACES = False
    print("Running in standard environment (local / CPU).")

    # Mock spaces.GPU decorator for local execution
    class spaces:
        @staticmethod
        def GPU(duration=120):
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
from fastapi.responses import StreamingResponse, JSONResponse
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
    """Trigger model loading on server startup."""
    logger.info("Initializing server and preparing local model weights...")
    try:
        # Load in background or immediately
        load_local_model()
    except Exception as e:
        logger.error(f"Startup model load notice: {e}")

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
        "status": "healthy" if model is not None else ("loading" if is_model_loading else "error"),
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
    models = [
        ModelCard(id=MODEL_ID),
        ModelCard(id="Hermes-3-8B"),
        ModelCard(id="NousResearch/Hermes-3-Llama-3.1-8B")
    ]
    return {
        "object": "list",
        "data": [m.model_dump() for m in models]
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

# --- Gradio UI Dashboard & Playground ---
try:
    import gradio as gr

    @spaces.GPU(duration=120)
    def gradio_chat_fn(user_message, history):
        if not user_message:
            return ""

        conversation = []
        if history:
            for h in history:
                if isinstance(h, dict):
                    conversation.append(Message(role=h.get("role", "user"), content=h.get("content", "")))
                elif isinstance(h, (list, tuple)) and len(h) >= 2:
                    if h[0]:
                        conversation.append(Message(role="user", content=str(h[0])))
                    if h[1]:
                        conversation.append(Message(role="assistant", content=str(h[1])))
        conversation.append(Message(role="user", content=user_message))

        prompt = construct_chatml_prompt(conversation)
        try:
            return _generate_core(prompt, 1024, 0.7, 0.95)
        except Exception as e:
            return f"❌ Error: {str(e)}"

    with gr.Blocks(title="Hermes-3 8B Local Server", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"""
            # 🚀 Hermes-3 8B Local In-Memory Server
            **Model:** `{MODEL_ID}` | **Execution:** Direct Local Weights (Zero API Limits)
            """
        )

        with gr.Tab("💬 Interactive Playground"):
            gr.ChatInterface(
                fn=gradio_chat_fn,
                title="Hermes-3 8B Chat",
                description="Running directly on local weights. Lead Architect persona active."
            )

        with gr.Tab("📖 API Integration"):
            gr.Markdown(
                f"""
                ### OpenAI-Compatible API Endpoints
                - `POST {HF_SPACE_URL}/v1/chat/completions` (JSON & SSE Streaming)
                - `GET {HF_SPACE_URL}/health`
                - `GET {HF_SPACE_URL}/v1/models`
                """
            )

    app = gr.mount_gradio_app(app, demo, path="/")
    logger.info("Gradio UI mounted on root path.")

    # Explicitly trigger ZeroGPU registration hook so Uvicorn ASGI registers GPU functions
    try:
        from spaces import zero as _spaces_zero
        if hasattr(_spaces_zero, "startup"):
            _spaces_zero.startup()
            logger.info("[ZeroGPU] spaces.zero.startup() registered successfully.")
    except Exception as hook_err:
        logger.info(f"[ZeroGPU] Startup hook note: {hook_err}")

except ImportError:
    @app.get("/")
    def root_endpoint():
        return {
            "status": "online",
            "service": "Hermes-3 8B Local Server",
            "endpoints": {
                "health": "/health",
                "models": "/v1/models",
                "chat_completions": "/v1/chat/completions",
                "docs": "/docs"
            }
        }

if __name__ == "__main__":
    try:
        from spaces import zero as _spaces_zero
        if hasattr(_spaces_zero, "startup"):
            _spaces_zero.startup()
            print("[ZeroGPU] spaces.zero.startup() triggered before uvicorn.", flush=True)
    except Exception as e:
        pass

    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
