import os
import sys

# HF_HOME Cache directory setup (preserves downloaded model in /data across ZeroGPU restarts)
if sys.platform != "win32":
    os.environ["HF_HOME"] = "/data/hf_cache"
    try:
        os.makedirs("/data/hf_cache", exist_ok=True)
    except Exception:
        pass
else:
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import time
import json
import uuid
import asyncio
import logging
import re
from threading import Thread, Lock
from typing import List, Optional, AsyncGenerator, Union

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
    TextIteratorStreamer,
    BitsAndBytesConfig
)
import io
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Safe Document & Media Extraction Parsers (No Vector DB Required)
try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None

try:
    import openpyxl
    import pandas as pd
except ImportError:
    openpyxl = None
    pd = None

try:
    from pptx import Presentation
except ImportError:
    Presentation = None

try:
    from PIL import Image
except ImportError:
    Image = None


# Load environment variables from .env
load_dotenv()

# Import Short-Term Memory and Document Working Memory Store
from session_memory import SessionMemoryStore, session_memory_store, PinnedDocument, SessionState

# Security, Caching, and Big File Chunking Modules
from security_check import check_file, MAX_SIZE as MAX_FILE_SIZE
from cache_manager import get_from_cache, save_to_cache
from big_file_handler import process_big_file, extract_chunks

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

# --- Dynamic Persona Configuration (Private & Loaded Dynamically from .env) ---
def get_default_system_prompt() -> str:
    """Dynamically retrieves Somi persona from environment variable; stays private and never committed to code."""
    prompt = os.getenv("SYSTEM_PROMPT", "").strip()
    if prompt.startswith('"') and prompt.endswith('"') and len(prompt) >= 2:
        prompt = prompt[1:-1].strip()
    return prompt or (
        "You are Somi, an elite AI Co-Founder, Technical Partner, and Strategist to Amit. "
        "You deliver sharp, actionable, production-ready solutions in natural Hinglish/English with zero fluff."
    )

DEFAULT_SYSTEM_PROMPT = get_default_system_prompt()

# --- Global Model & Tokenizer Instances ---
tokenizer = None
model = None
is_model_loading = False
model_load_error = None
_model_load_lock = Lock()

# Hermes-3 Llama 3.1 8B context window (128k native)
MAX_CONTEXT_BUDGET = 131072

def get_optimal_torch_dtype() -> torch.dtype:
    """Selects bfloat16 for ZeroGPU / Ampere / Hopper / Ada, float16 for CUDA, float32 for CPU without top-level CUDA init."""
    if HAS_SPACES:
        return torch.bfloat16
    return torch.float16 if sys.platform != "win32" else torch.float32

def get_eos_token_ids(tok, custom_stops: Optional[List[str]] = None) -> List[int]:
    """Retrieves all stop token IDs for Hermes-3 / Llama 3.1 plus optional custom stop tokens."""
    stop_ids = set()
    if tok is None:
        return []

    if tok.eos_token_id is not None:
        if isinstance(tok.eos_token_id, (list, tuple, set)):
            stop_ids.update(tok.eos_token_id)
        else:
            stop_ids.add(tok.eos_token_id)

    # Core Hermes-3 and Llama 3.1 stop tokens
    stop_token_strings = ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|im_start|>"]
    if custom_stops:
        for cs in custom_stops:
            if cs and cs not in stop_token_strings:
                stop_token_strings.append(cs)

    for token_str in stop_token_strings:
        try:
            tid = tok.convert_tokens_to_ids(token_str)
            if tid is not None and isinstance(tid, int) and tid >= 0:
                if tok.unk_token_id is None or tid != tok.unk_token_id:
                    stop_ids.add(tid)
        except Exception:
            pass
        try:
            enc = tok.encode(token_str, add_special_tokens=False)
            if len(enc) == 1 and enc[0] >= 0:
                if tok.unk_token_id is None or enc[0] != tok.unk_token_id:
                    stop_ids.add(enc[0])
        except Exception:
            pass

    return list(stop_ids)

@spaces.GPU(duration=180)
def load_local_model():
    """
    Downloads and loads the model and tokenizer into memory with thread-safety inside @spaces.GPU.
    Uses 4-bit quantization (BitsAndBytes NF4, ~5GB VRAM) on CUDA/ZeroGPU for optimal efficiency.
    """
    global tokenizer, model, is_model_loading, model_load_error
    if model is not None and tokenizer is not None:
        return model, tokenizer

    with _model_load_lock:
        if model is not None and tokenizer is not None:
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

            dtype = get_optimal_torch_dtype()
            device_str = "cuda" if (HAS_SPACES or torch.cuda.is_available()) else "cpu"
            logger.info(f"2. Target device detected: {device_str.upper()} (ZeroGPU: {HAS_SPACES}, Dtype: {dtype})")

            # 4-bit Quantization Config (drastically drops memory from 16GB to ~5GB VRAM)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            load_kwargs = {
                "pretrained_model_name_or_path": MODEL_ID,
                "token": HF_TOKEN,
                "quantization_config": bnb_config,
                "device_map": "auto",
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            }

            try:
                logger.info("3. Loading 4-bit quantized model (BitsAndBytes NF4, ~5GB VRAM) with SDPA...")
                model = AutoModelForCausalLM.from_pretrained(
                    **load_kwargs,
                    attn_implementation="sdpa"
                )
            except Exception as quant_sdpa_err:
                try:
                    logger.info(f"   Note on 4-bit SDPA mount ({quant_sdpa_err}); attempting default attention...")
                    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
                except Exception as quant_err:
                    logger.warning(f"   4-bit load fallback ({quant_err}); loading standard precision...")
                    load_kwargs.pop("quantization_config", None)
                    load_kwargs["torch_dtype"] = dtype
                    load_kwargs["device_map"] = "auto" if torch.cuda.is_available() else None
                    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

            model.eval()
            # Ensure KV cache is enabled
            if hasattr(model, "generation_config") and model.generation_config is not None:
                model.generation_config.use_cache = True

            logger.info(f"4. Model '{MODEL_ID}' successfully loaded into memory! KV Cache enabled.")
            is_model_loading = False
            model_load_error = None
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
    version="2.4.0"
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
    max_tokens: Optional[int] = Field(default=300, ge=1, le=8192)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    session_id: Optional[str] = None



# --- Helper Functions ---
def construct_chatml_prompt(messages: List[Message], default_system: Optional[str] = None) -> str:
    """
    Constructs strict Hermes-3 compliant ChatML prompt:
    <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n
    Supports system, user, assistant, and tool roles with optional names.
    Dynamically falls back to get_default_system_prompt() to load Somi persona.
    """
    has_system = any(m.role.strip().lower() == "system" for m in messages if m.role)
    parts = []

    if not has_system:
        system_text = default_system or get_default_system_prompt()
        parts.append(f"<|im_start|>system\n{system_text.strip()}<|im_end|>\n")

    for msg in messages:
        role = (msg.role or "user").strip().lower()
        content = (msg.content or "").strip()
        name_attr = f" name={msg.name.strip()}" if msg.name and msg.name.strip() else ""
        parts.append(f"<|im_start|>{role}{name_attr}\n{content}<|im_end|>\n")

    # If messages do not conclude with an assistant prefill, prompt the assistant
    if not messages or (messages[-1].role or "").strip().lower() != "assistant":
        parts.append("<|im_start|>assistant\n")

    return "".join(parts)

def sanitize_assistant_output(raw_text: str, custom_stops: Optional[List[str]] = None) -> str:
    """
    Cleans Hermes stop tokens (<|im_end|>, <|eot_id|>, <|end_of_text|>) while preserving
    any tool calling tags (<tool_call>, </tool_call>) and agentic structure.
    Strictly truncates output at the earliest occurrence of any stop token.
    """
    stop_markers = ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|im_start|>"]
    if custom_stops:
        for cs in custom_stops:
            if cs and cs not in stop_markers:
                stop_markers.append(cs)

    earliest_pos = len(raw_text)
    for marker in stop_markers:
        pos = raw_text.find(marker)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos

    return raw_text[:earliest_pos].rstrip()

def enforce_context_budget(prompt_text: str, max_tokens: int, tok) -> str:
    """
    Ensures prompt_text fits within the 128k context budget (131,072 tokens).
    If prompt exceeds budget, gracefully truncates middle content while preserving
    system prompt directives, document metadata headers, and latest user query.
    """
    max_input_tokens = max(1024, MAX_CONTEXT_BUDGET - max_tokens - 256)

    # Fast estimation heuristic (~3.5 chars/token) before tokenizing
    if len(prompt_text) < max_input_tokens * 3:
        return prompt_text

    if tok is None:
        return prompt_text[:max_input_tokens * 4]

    token_ids = tok.encode(prompt_text, add_special_tokens=False)
    if len(token_ids) <= max_input_tokens:
        return prompt_text

    logger.warning(
        f"[ContextBudget] Input prompt ({len(token_ids)} tokens) exceeds max input budget "
        f"({max_input_tokens} tokens). Applying intelligent middle truncation..."
    )

    # Preserve beginning (system prompt, document head) and end (user query)
    head_tokens = int(max_input_tokens * 0.65)
    tail_tokens = max_input_tokens - head_tokens - 32

    head_ids = token_ids[:head_tokens]
    tail_ids = token_ids[-tail_tokens:] if tail_tokens > 0 else []

    head_text = tok.decode(head_ids, skip_special_tokens=False)
    tail_text = tok.decode(tail_ids, skip_special_tokens=False) if tail_ids else ""

    notice = "\n\n[... Notice: Content truncated to fit within 128k token context budget ...]\n\n"
    return head_text + notice + tail_text

# --- Core Generation Function ---
def _generate_core(
    prompt_text: str,
    max_tokens: int = 300,
    temperature: float = 0.7,
    top_p: float = 0.95,
    stop: Optional[List[str]] = None
) -> str:
    """Generates complete text response directly using locally loaded model weights with KV cache."""
    if model is None or tokenizer is None:
        load_local_model()

    device = model.device
    prompt_text = enforce_context_budget(prompt_text, max_tokens, tokenizer)
    max_input_len = max(1024, MAX_CONTEXT_BUDGET - max_tokens - 64)
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_input_len).to(device)
    stop_ids = get_eos_token_ids(tokenizer, custom_stops=stop)

    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": stop_ids,
    }

    if temperature is not None and temperature > 0.0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p) if top_p is not None else 0.95
    else:
        gen_kwargs["do_sample"] = False

    with torch.inference_mode():
        outputs = model.generate(**gen_kwargs)

    generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    return sanitize_assistant_output(raw_output, custom_stops=stop)

# --- Generation with ZeroGPU Decorators ---
@spaces.GPU(duration=180)
def generate_local(
    prompt_text: str,
    max_tokens: int = 300,
    temperature: float = 0.7,
    top_p: float = 0.95,
    stop: Optional[List[str]] = None
) -> str:
    """Entrypoint for FastAPI inference requests on ZeroGPU."""
    return _generate_core(prompt_text, max_tokens, temperature, top_p, stop=stop)

@spaces.GPU(duration=180)
def stream_local_tokens(
    prompt_text: str,
    max_tokens: int = 300,
    temperature: float = 0.7,
    top_p: float = 0.95,
    stop: Optional[List[str]] = None
):
    """Streams generated tokens in real-time chunk-by-chunk using TextIteratorStreamer."""
    if model is None or tokenizer is None:
        load_local_model()

    device = model.device
    prompt_text = enforce_context_budget(prompt_text, max_tokens, tokenizer)
    max_input_len = max(1024, MAX_CONTEXT_BUDGET - max_tokens - 64)
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_input_len).to(device)
    stop_ids = get_eos_token_ids(tokenizer, custom_stops=stop)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,
        timeout=60.0
    )

    gen_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_tokens,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": stop_ids,
    }

    if temperature is not None and temperature > 0.0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p) if top_p is not None else 0.95
    else:
        gen_kwargs["do_sample"] = False

    def _run_stream_worker():
        with torch.inference_mode():
            model.generate(**gen_kwargs)

    thread = Thread(target=_run_stream_worker, daemon=True)
    thread.start()

    stop_markers = ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|im_start|>"]
    if stop:
        for s in stop:
            if s and s not in stop_markers:
                stop_markers.append(s)

    for new_text in streamer:
        has_stop = False
        for marker in stop_markers:
            if marker in new_text:
                pre_stop = new_text.split(marker)[0]
                if pre_stop:
                    yield pre_stop
                has_stop = True
                break
        if has_stop:
            break
        yield new_text

# --- API Routes ---

@app.on_event("startup")
async def startup_event():
    """Fast server startup in Lazy-Loading mode (ZeroGPU compliant: model loads on first message inside @spaces.GPU)."""
    logger.info("Hermes-3 8B Local API Server initialized in Lazy-Loading mode (ZeroGPU compliant).")

@app.get("/health")
async def health_check():
    """Uptime, GPU, KV cache, and Model status check (ZeroGPU safe: does not trigger early CUDA initialization)."""
    uptime = time.time() - START_TIME
    device_name = str(model.device) if model is not None else "lazy_unloaded"
    vram_mb = 0.0
    if model is not None:
        try:
            vram_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        except Exception:
            pass

    return {
        "status": "healthy" if model is not None else "ready_to_load",
        "service": "Hermes-3 8B Local Server",
        "model": MODEL_ID,
        "is_model_loaded": model is not None,
        "device": device_name,
        "vram_allocated_mb": vram_mb,
        "context_window_tokens": MAX_CONTEXT_BUDGET,
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
    """Warms up local generation weights and pre-allocates KV cache."""
    if model is None:
        if is_model_loading:
            return JSONResponse(
                status_code=503,
                content={"status": "loading", "message": "Model is currently loading into memory. Please retry in a few moments."}
            )
        try:
            load_local_model()
        except Exception as e:
            return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

    start = time.time()
    try:
        dummy = generate_local("<|im_start|>user\nping<|im_end|>\n<|im_start|>assistant\n", 2, 0.5, 0.9)
        latency = (time.time() - start) * 1000.0
        return {
            "status": "warm",
            "latency_ms": round(latency, 2),
            "model": MODEL_ID,
            "device": str(model.device) if model is not None else "unknown",
            "message": "Local model is loaded, KV cache initialized, and kernels fully warm."
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    Core OpenAI-compatible Chat Completion endpoint.
    Executes directly against local weights with zero third-party limits.
    """
    if not req.messages:
        raise HTTPException(status_code=400, detail="The 'messages' array must not be empty.")

    if model is None:
        if is_model_loading:
            raise HTTPException(status_code=503, detail="Model is currently loading into memory. Please retry shortly.")
        try:
            load_local_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")

    # Parse and normalize custom stop tokens
    custom_stops = None
    if req.stop:
        if isinstance(req.stop, str):
            custom_stops = [req.stop]
        elif isinstance(req.stop, list):
            custom_stops = [s for s in req.stop if isinstance(s, str) and s]

    # Session Short-Term Working Memory handling
    session = None
    effective_messages = list(req.messages)

    if req.session_id:
        session = session_memory_store.get_or_create(req.session_id)

        # If client supplied only single user turn, restore previous turns from session rolling buffer
        if len(req.messages) == 1 and (req.messages[0].role or "").strip().lower() == "user" and len(session.messages) > 0:
            reconstructed = [
                Message(role=sm.role, content=sm.content, name=sm.name)
                for sm in session.messages
            ]
            if not (reconstructed and reconstructed[-1].content == req.messages[0].content):
                reconstructed.append(req.messages[0])
            effective_messages = reconstructed

        # Pinned Document Working Memory Injection
        pinned_context = session.get_pinned_context_block()
        if pinned_context:
            already_in_messages = any(
                pinned_context[:60] in (m.content or "") or
                "[WORKING MEMORY: PINNED DOCUMENTS" in (m.content or "") or
                "=== [DOCUMENT:" in (m.content or "")
                for m in effective_messages
            )
            if not already_in_messages:
                augmented = []
                system_found = False
                for m in effective_messages:
                    if (m.role or "").strip().lower() == "system" and not system_found:
                        augmented.append(
                            Message(
                                role="system",
                                content=f"{m.content.strip()}\n\n{pinned_context}",
                                name=m.name
                            )
                        )
                        system_found = True
                    else:
                        augmented.append(m)
                if not system_found:
                    augmented.insert(
                        0,
                        Message(role="system", content=f"{DEFAULT_SYSTEM_PROMPT.strip()}\n\n{pinned_context}")
                    )
                effective_messages = augmented

        # Record latest user turn into session rolling buffer
        for m in reversed(effective_messages):
            if (m.role or "").strip().lower() == "user" and m.content:
                session.add_message("user", m.content, name=m.name)
                break

    prompt_text = construct_chatml_prompt(effective_messages)
    target_model = req.model or MODEL_ID
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    # 1. Real-time Streaming (SSE)
    if req.stream:
        async def event_generator():
            try:
                # Initial role chunk conforming to OpenAI streaming standard
                initial_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": target_model,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(initial_chunk)}\n\n"

                accumulated_text = []
                for token_text in stream_local_tokens(
                    prompt_text=prompt_text,
                    max_tokens=req.max_tokens or 300,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    stop=custom_stops
                ):
                    accumulated_text.append(token_text)
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
                    await asyncio.sleep(0)

                # Persist completed assistant generation in session memory
                if session and accumulated_text:
                    session.add_message("assistant", "".join(accumulated_text))

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

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream; charset=utf-8"
            }
        )

    # 2. Non-Streaming Generation
    start_infer = time.time()
    try:
        generated_content = generate_local(
            prompt_text=prompt_text,
            max_tokens=req.max_tokens or 300,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=custom_stops
        )
        if session and generated_content:
            session.add_message("assistant", generated_content)
    except Exception as e:
        logger.error(f"Local generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Local Generation Error: {str(e)}")

    duration = time.time() - start_infer

    # Accurate token usage metrics calculated via tokenizer
    prompt_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False)) if tokenizer else max(len(prompt_text.split()), 1)
    completion_tokens = len(tokenizer.encode(generated_content, add_special_tokens=False)) if tokenizer else max(len(generated_content.split()), 1)

    res_payload = {
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
    if req.session_id:
        res_payload["session_id"] = req.session_id
    return res_payload

# --- File Extraction & Multipart Chat Endpoints (Zero Vector DB) ---

MAX_EXTRACT_CHARS = 120_000   # Global context budget cap for extracted text (~25k-30k tokens)
MAX_PDF_PAGES_SCAN = 100      # Max pages to scan/sample from a PDF
MAX_PDF_PAGE_CHARS = 3_500    # Max chars per individual PDF page
MAX_TABLE_ROWS_PREVIEW = 80   # Max rows previewed for tabular datasets (head + tail)
MAX_TABLE_COLS_PREVIEW = 25   # Max columns to render in markdown table
MAX_TEXT_LINES = 2_000        # Max lines for source code / plain text files
MAX_DOCX_PARAS = 1_500        # Max paragraphs in Word document
MAX_PPTX_SLIDES = 60          # Max slides in PowerPoint presentation
MAX_EXCEL_SHEETS = 10         # Max sheets to inspect in a workbook

# Safe PIL Decompression Bomb Guard
if Image is not None:
    try:
        Image.MAX_IMAGE_PIXELS = 60_000_000
    except Exception:
        pass

CODE_LANG_MAP = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".html": "html", ".htm": "html", ".css": "css", ".scss": "scss", ".sass": "sass", ".less": "less",
    ".json": "json", ".jsonl": "json", ".ndjson": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".xml": "xml", ".sql": "sql", ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".bat": "bat", ".cmd": "bat", ".ps1": "powershell", ".env": "ini", ".ini": "ini", ".conf": "ini",
    ".cfg": "ini", ".c": "c", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".h": "c", ".hpp": "cpp",
    ".cs": "csharp", ".java": "java", ".kt": "kotlin", ".kts": "kotlin", ".scala": "scala",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php", ".swift": "swift", ".m": "objectivec",
    ".mm": "objectivec", ".r": "r", ".dart": "dart", ".lua": "lua", ".pl": "perl",
    ".vue": "vue", ".svelte": "svelte", ".graphql": "graphql", ".gql": "graphql",
    ".proto": "protobuf", ".txt": "text", ".text": "text", ".md": "markdown",
    ".markdown": "markdown", ".rst": "rst", ".log": "text"
}

def decode_text_bytes(file_bytes: bytes) -> str:
    """Safely decodes raw bytes across standard character encodings with fallback."""
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            decoded = file_bytes.decode(enc)
            return decoded.replace("\x00", "")
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace").replace("\x00", "")

def truncate_text_content(text: str, filename: str, max_lines: int = MAX_TEXT_LINES, max_chars: int = MAX_EXTRACT_CHARS) -> tuple[str, bool]:
    """
    Truncates large text or code files smartly (first N lines + last M lines).
    Returns (processed_text, is_truncated).
    """
    lines = text.splitlines()
    total_lines = len(lines)
    total_chars = len(text)

    if total_lines <= max_lines and total_chars <= max_chars:
        return text, False

    head_lines = min(1500, max_lines - 300)
    tail_lines = min(300, total_lines - head_lines)
    
    first_part = "\n".join(lines[:head_lines])
    last_part = "\n".join(lines[-tail_lines:]) if tail_lines > 0 else ""
    omitted = total_lines - (head_lines + tail_lines)
    
    trunc_msg = f"\n\n// --- [Smart Truncation: Omitted {omitted:,} lines ({total_chars - len(first_part) - len(last_part):,} chars) to preserve context budget. Showing lines 1-{head_lines} and {total_lines - tail_lines + 1}-{total_lines}] ---\n\n"
    truncated_result = first_part + trunc_msg + last_part
    return truncated_result, True

def format_dataframe_to_markdown(df, max_rows: int = MAX_TABLE_ROWS_PREVIEW, max_cols: int = MAX_TABLE_COLS_PREVIEW) -> str:
    """
    Renders any pandas DataFrame into clean, resilient GitHub-Flavored Markdown.
    Zero external dependencies: does NOT require `tabulate`.
    Handles large tables with column truncation, head/tail row sampling, and pipe escaping.
    """
    if df is None or df.empty:
        return "*(Empty table)*"

    total_rows, total_cols = df.shape
    cols = [str(c) if c is not None else f"Col_{i}" for i, c in enumerate(df.columns)]

    # Determine column slicing
    is_cols_truncated = False
    if total_cols > max_cols:
        is_cols_truncated = True
        head_cols_count = max_cols - 5
        head_indices = list(range(head_cols_count))
        tail_indices = list(range(total_cols - 5, total_cols))
        selected_col_indices = head_indices + tail_indices
        display_col_names = [cols[i] for i in head_indices] + [f"... (+{total_cols - max_cols} cols)"] + [cols[i] for i in tail_indices]
    else:
        selected_col_indices = list(range(total_cols))
        display_col_names = cols

    # Determine row slicing
    is_rows_truncated = False
    if total_rows > max_rows:
        is_rows_truncated = True
        head_n = max(max_rows - 20, 10)
        tail_n = min(20, max_rows - head_n)
        head_indices = list(range(head_n))
        tail_indices = list(range(total_rows - tail_n, total_rows))
        row_indices = head_indices + [-1] + tail_indices  # -1 represents divider row
    else:
        row_indices = list(range(total_rows))

    def clean_cell(val) -> str:
        if val is None:
            return ""
        if pd is not None:
            try:
                if pd.isna(val):
                    return ""
            except Exception:
                pass
        s = str(val).replace("\r\n", " ").replace("\n", " ").replace("\r", " ").replace("|", "\\|")
        if len(s) > 120:
            s = s[:117] + "..."
        return s.strip()

    header_line = "| " + " | ".join(clean_cell(c) for c in display_col_names) + " |"
    separator_line = "| " + " | ".join(["---"] * len(display_col_names)) + " |"
    lines = [header_line, separator_line]

    for r_idx in row_indices:
        if r_idx == -1:
            lines.append("| " + " | ".join(["..."] * len(display_col_names)) + " |")
            continue
        
        row_series = df.iloc[r_idx]
        if is_cols_truncated:
            head_vals = [clean_cell(row_series.iloc[i]) for i in head_indices]
            tail_vals = [clean_cell(row_series.iloc[i]) for i in tail_indices]
            row_cells = head_vals + ["..."] + tail_vals
        else:
            row_cells = [clean_cell(row_series.iloc[i]) for i in selected_col_indices]

        lines.append("| " + " | ".join(row_cells) + " |")

    notice = []
    if is_rows_truncated:
        notice.append(f"Showing {max_rows} of {total_rows:,} total rows (Head + Tail preview)")
    if is_cols_truncated:
        notice.append(f"Showing {max_cols} of {total_cols:,} total columns")
    
    table_md = "\n".join(lines)
    if notice:
        table_md += f"\n\n*({'; '.join(notice)})*"

    return table_md

def summarize_dataframe_schema(df, sheet_name: str) -> str:
    """
    Produces high-density schema and statistical metadata for LLM tabular reasoning.
    Enables Hermes-3 to understand massive datasets (50,000+ rows) without needing
    every row in prompt context.
    """
    total_rows, total_cols = df.shape
    parts = [
        f"### Tabular Sheet: `{sheet_name}`",
        f"- **Total Rows:** {total_rows:,} | **Total Columns:** {total_cols:,}",
    ]

    col_details = []
    for i, c in enumerate(df.columns[:35]):
        col_series = df.iloc[:, i]
        dtype_str = str(col_series.dtype)
        null_count = int(col_series.isna().sum()) if pd is not None else 0
        null_info = f", {null_count} nulls" if null_count > 0 else ", 0 nulls"

        stat_info = ""
        try:
            if pd is not None and pd.api.types.is_numeric_dtype(col_series) and not col_series.dropna().empty:
                s_min = col_series.min()
                s_max = col_series.max()
                s_mean = col_series.mean()
                if isinstance(s_mean, (float, int)):
                    stat_info = f" [min: {s_min}, mean: {s_mean:.2f}, max: {s_max}]"
                else:
                    stat_info = f" [min: {s_min}, max: {s_max}]"
            elif pd is not None and (pd.api.types.is_object_dtype(col_series) or pd.api.types.is_string_dtype(col_series)):
                unique_cnt = col_series.nunique(dropna=True)
                if unique_cnt <= 5 and unique_cnt > 0:
                    sample_vals = [str(x) for x in col_series.dropna().unique()[:5]]
                    stat_info = f" [values: {', '.join(sample_vals)}]"
                else:
                    stat_info = f" [{unique_cnt:,} unique values]"
        except Exception:
            pass

        col_details.append(f"  - `{str(c)}` ({dtype_str}{null_info}){stat_info}")

    if total_cols > 35:
        col_details.append(f"  - *(+ {total_cols - 35} additional columns omitted from schema list)*")

    if col_details:
        parts.append("- **Column Schema & Statistics:**\n" + "\n".join(col_details))

    return "\n".join(parts)

def extract_text_code(filename: str, file_bytes: bytes, size_kb: float) -> dict:
    ext = os.path.splitext(filename)[1].lower()
    lang = CODE_LANG_MAP.get(ext, "text")
    raw_text = decode_text_bytes(file_bytes)
    
    total_lines = len(raw_text.splitlines())
    total_chars = len(raw_text)
    
    trunc_text, is_truncated = truncate_text_content(raw_text, filename)
    
    header = f"### File: `{filename}` ({lang.capitalize()})\n- **Lines:** {total_lines:,} | **Characters:** {total_chars:,} | **Size:** {size_kb:,} KB\n"
    if is_truncated:
        header += f"- **Notice:** *Smart Truncation active (large file condensed to fit LLM context).* \n"
    
    full_doc = f"{header}\n```{lang}\n{trunc_text}\n```"
    summary = f"{lang.capitalize()} file ({total_lines:,} lines, {total_chars:,} chars)"
    if is_truncated:
        summary += " [condensed]"

    return {
        "extracted_text": full_doc,
        "summary": summary,
        "metadata": {
            "language": lang,
            "lines": total_lines,
            "characters": total_chars,
            "is_truncated": is_truncated
        },
        "error": None
    }

def extract_pdf(filename: str, file_bytes: bytes, size_kb: float) -> dict:
    if pypdf is None:
        return {
            "extracted_text": f"[PDF file: {filename} uploaded, but 'pypdf' parser is pending install in environment]",
            "summary": "PDF document (pypdf unavailable)",
            "metadata": {"pages": 0, "status": "pypdf_missing"},
            "error": "pypdf library not found"
        }

    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        num_pages = len(reader.pages)
        if num_pages == 0:
            return {
                "extracted_text": f"### PDF Document: `{filename}`\n*(Empty PDF document with 0 pages)*",
                "summary": "Empty PDF document (0 pages)",
                "metadata": {"pages": 0, "status": "empty"},
                "error": None
            }

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                return {
                    "extracted_text": f"### PDF Document: `{filename}`\n*(This PDF is password-protected/encrypted and cannot be parsed without decryption credentials)*",
                    "summary": f"Encrypted PDF ({num_pages} pages)",
                    "metadata": {"pages": num_pages, "encrypted": True},
                    "error": "PDF is password protected"
                }

        pdf_meta = {}
        try:
            if reader.metadata:
                for k, v in reader.metadata.items():
                    clean_k = str(k).lstrip("/").strip()
                    if clean_k and v:
                        pdf_meta[clean_k] = str(v).strip()
        except Exception:
            pass

        selected_pages = []
        is_truncated = False
        if num_pages <= 50:
            selected_pages = list(range(num_pages))
        else:
            is_truncated = True
            head_pages = list(range(30))
            tail_pages = list(range(max(num_pages - 15, 30), num_pages))
            step = max((num_pages - 45) // 9, 1)
            mid_pages = [p for p in range(30, num_pages - 15, step)[:8] if p not in head_pages and p not in tail_pages]
            selected_pages = sorted(list(set(head_pages + mid_pages + tail_pages)))

        header_lines = [
            f"### PDF Document: `{filename}`",
            f"- **Total Pages:** {num_pages:,} | **File Size:** {size_kb:,} KB",
        ]
        if "Title" in pdf_meta:
            header_lines.append(f"- **Title:** {pdf_meta['Title']}")
        if "Author" in pdf_meta:
            header_lines.append(f"- **Author:** {pdf_meta['Author']}")
        if "Subject" in pdf_meta:
            header_lines.append(f"- **Subject:** {pdf_meta['Subject']}")

        if is_truncated:
            header_lines.append(f"- **Notice:** *Smart Truncation active. Showing {len(selected_pages)} key representative pages out of {num_pages:,} total pages (Pages 1-30, sampled middle sections, and concluding pages).*")

        pages_content = []
        total_extracted_chars = 0
        char_budget_exceeded = False

        for idx, p_num in enumerate(selected_pages):
            if total_extracted_chars >= MAX_EXTRACT_CHARS:
                char_budget_exceeded = True
                pages_content.append(f"\n*(Context character budget reached. Remaining {len(selected_pages) - idx} sampled pages omitted.)*")
                break

            try:
                page = reader.pages[p_num]
                p_text = (page.extract_text() or "").strip()
            except Exception as pe:
                p_text = f"*(Error extracting page {p_num + 1}: {str(pe)})*"

            if p_text:
                if len(p_text) > MAX_PDF_PAGE_CHARS:
                    p_text = p_text[:MAX_PDF_PAGE_CHARS] + "\n*(Page text truncated to preserve context budget)*"
                pages_content.append(f"#### [Page {p_num + 1} of {num_pages}]\n{p_text}")
                total_extracted_chars += len(p_text)

        is_scanned = (total_extracted_chars < 150 and num_pages > 1)
        if is_scanned:
            header_lines.append("- **Scan Advisory:** *⚠️ This document appears to be scanned or contains raster images with minimal selectable text.*")

        full_doc = "\n".join(header_lines) + "\n\n---\n\n" + "\n\n".join(pages_content)
        summary = f"PDF document with {num_pages} page(s)"
        if is_truncated or char_budget_exceeded:
            summary += f" ({len(selected_pages)} sampled pages extracted)"
        if is_scanned:
            summary += " [scanned/image-based]"

        return {
            "extracted_text": full_doc,
            "summary": summary,
            "metadata": {
                "total_pages": num_pages,
                "extracted_pages": len(selected_pages),
                "is_truncated": is_truncated or char_budget_exceeded,
                "is_scanned": is_scanned,
                **pdf_meta
            },
            "error": None
        }
    except Exception as e:
        return {
            "extracted_text": f"### PDF Document: `{filename}`\n*(Error reading PDF: {str(e)})*",
            "summary": f"PDF parse error: {str(e)}",
            "metadata": {"error": str(e)},
            "error": str(e)
        }

def extract_word(filename: str, file_bytes: bytes, size_kb: float) -> dict:
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == ".doc" or file_bytes.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        raw_text = decode_text_bytes(file_bytes)
        words = re.findall(r"[\x20-\x7E\t\n\r]{4,}", raw_text)
        cleaned = "\n".join(w.strip() for w in words if w.strip())
        if len(cleaned) > MAX_EXTRACT_CHARS:
            cleaned = cleaned[:MAX_EXTRACT_CHARS] + "\n\n*(Truncated legacy .doc content)*"
        
        md_text = (
            f"### Word Document (Legacy .doc): `{filename}`\n"
            f"- **File Size:** {size_kb:,} KB\n"
            f"> ℹ️ **Notice:** This file uses the legacy binary Word 97-2003 (.doc) format. "
            f"Extracted plain text stream is provided below. For full tables and styled headings, please convert to .docx.\n\n"
            f"```text\n{cleaned if cleaned else '*(No readable text stream detected)*'}\n```"
        )
        return {
            "extracted_text": md_text,
            "summary": f"Legacy Word .doc document ({size_kb} KB)",
            "metadata": {"format": "legacy_doc", "char_count": len(cleaned)},
            "error": None
        }

    if docx is None:
        return {
            "extracted_text": f"[Word document: {filename} uploaded, but 'python-docx' is not installed]",
            "summary": "Word document (python-docx unavailable)",
            "metadata": {"status": "docx_missing"},
            "error": "python-docx library not found"
        }

    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        doc_parts = []
        para_count = len(doc.paragraphs)
        table_count = len(doc.tables)
        
        doc_parts.append(f"### Word Document: `{filename}`")
        doc_parts.append(f"- **Paragraphs:** {para_count:,} | **Tables:** {table_count:,} | **Size:** {size_kb:,} KB\n---")

        para_rendered = 0
        is_truncated = False
        for p in doc.paragraphs:
            text = p.text.strip()
            if not text:
                continue
            
            style_name = (p.style.name if p.style else "").lower()
            if "title" in style_name:
                doc_parts.append(f"# {text}")
            elif "heading 1" in style_name:
                doc_parts.append(f"## {text}")
            elif "heading 2" in style_name:
                doc_parts.append(f"### {text}")
            elif "heading 3" in style_name:
                doc_parts.append(f"#### {text}")
            elif "heading" in style_name:
                doc_parts.append(f"##### {text}")
            elif "bullet" in style_name:
                doc_parts.append(f"- {text}")
            elif "number" in style_name:
                doc_parts.append(f"1. {text}")
            else:
                doc_parts.append(text)

            para_rendered += 1
            if para_rendered >= MAX_DOCX_PARAS:
                is_truncated = True
                doc_parts.append(f"\n*(Smart Truncation: Reached {MAX_DOCX_PARAS:,} paragraphs limit. Remaining paragraphs omitted to protect LLM context.)*")
                break

        for t_idx, table in enumerate(doc.tables):
            if len(doc_parts) > 2000 or sum(len(p) for p in doc_parts) >= MAX_EXTRACT_CHARS:
                doc_parts.append(f"\n*(Omitted remaining {table_count - t_idx} tables due to document size budget)*")
                break

            table_rows = []
            row_count = len(table.rows)
            sample_rows = table.rows if row_count <= 60 else (table.rows[:45] + table.rows[-10:])
            col_count = len(table.columns) if table.columns else 0

            for r_idx, row in enumerate(sample_rows):
                cells_text = []
                for cell in row.cells:
                    ctext = cell.text.strip().replace("\r\n", " ").replace("\n", " ").replace("|", "\\|")
                    if len(ctext) > 100:
                        ctext = ctext[:97] + "..."
                    cells_text.append(ctext)
                
                table_rows.append("| " + " | ".join(cells_text) + " |")

            if table_rows:
                if len(table_rows) > 1:
                    actual_cols = len(table.rows[0].cells)
                    table_rows.insert(1, "| " + " | ".join(["---"] * actual_cols) + " |")
                
                t_label = f"\n#### Table {t_idx + 1} ({row_count:,} rows × {col_count} columns)"
                if row_count > 60:
                    t_label += f" *(Preview showing first 45 and last 10 rows)*"
                doc_parts.append(t_label + "\n" + "\n".join(table_rows))

        full_doc = "\n\n".join(doc_parts)
        summary = f"Word document ({para_count} paragraphs, {table_count} tables)"
        if is_truncated:
            summary += " [condensed]"

        return {
            "extracted_text": full_doc,
            "summary": summary,
            "metadata": {
                "paragraphs": para_count,
                "tables": table_count,
                "is_truncated": is_truncated
            },
            "error": None
        }
    except Exception as e:
        return {
            "extracted_text": f"### Word Document: `{filename}`\n*(Error reading Word document: {str(e)})*",
            "summary": f"Word parse error: {str(e)}",
            "metadata": {"error": str(e)},
            "error": str(e)
        }

def extract_spreadsheet(filename: str, file_bytes: bytes, size_kb: float) -> dict:
    ext = os.path.splitext(filename)[1].lower()
    
    if pd is None:
        if ext in (".csv", ".tsv"):
            raw_text = decode_text_bytes(file_bytes)
            trunc_text, is_trunc = truncate_text_content(raw_text, filename)
            return {
                "extracted_text": f"### CSV/TSV Document: `{filename}`\n\n```csv\n{trunc_text}\n```",
                "summary": f"CSV/TSV table ({len(raw_text.splitlines()):,} lines)",
                "metadata": {"format": ext, "is_truncated": is_trunc},
                "error": None
            }
        return {
            "extracted_text": f"[Spreadsheet file: {filename} uploaded, but 'pandas/openpyxl' is not installed]",
            "summary": "Spreadsheet (parsers unavailable)",
            "metadata": {"status": "pandas_missing"},
            "error": "pandas library not found"
        }

    try:
        sheets = {}
        if ext == ".csv":
            df = None
            for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
                try:
                    df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc, on_bad_lines="skip")
                    break
                except Exception:
                    continue
            if df is None:
                df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python", on_bad_lines="skip", encoding_errors="replace")
            sheets = {"CSV_Data": df}
        elif ext == ".tsv":
            df = None
            for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
                try:
                    df = pd.read_csv(io.BytesIO(file_bytes), sep="\t", encoding=enc, on_bad_lines="skip")
                    break
                except Exception:
                    continue
            if df is None:
                df = pd.read_csv(io.BytesIO(file_bytes), sep="\t", on_bad_lines="skip", encoding_errors="replace")
            sheets = {"TSV_Data": df}
        else:
            try:
                sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
            except Exception:
                sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, engine="openpyxl")

        sheet_names = list(sheets.keys())
        total_sheets = len(sheet_names)
        
        doc_sections = [
            f"### Spreadsheet Workbook: `{filename}`",
            f"- **Total Sheets:** {total_sheets} ({', '.join(sheet_names[:10])}) | **File Size:** {size_kb:,} KB\n---"
        ]

        total_rows_all_sheets = sum(len(df) for df in sheets.values())
        is_massive = total_rows_all_sheets > 5000

        active_sheets = sheet_names[:MAX_EXCEL_SHEETS]
        for s_idx, s_name in enumerate(active_sheets):
            sheet_df = sheets[s_name]
            if sheet_df is None or sheet_df.empty:
                doc_sections.append(f"### Sheet: `{s_name}`\n*(Sheet contains no data)*")
                continue

            schema_summary = summarize_dataframe_schema(sheet_df, s_name)
            md_table = format_dataframe_to_markdown(
                sheet_df,
                max_rows=MAX_TABLE_ROWS_PREVIEW,
                max_cols=MAX_TABLE_COLS_PREVIEW
            )

            section = f"{schema_summary}\n\n#### Data Preview:\n{md_table}"
            doc_sections.append(section)

        if total_sheets > MAX_EXCEL_SHEETS:
            doc_sections.append(f"\n*(Notice: Showing first {MAX_EXCEL_SHEETS} of {total_sheets} total sheets. Remaining sheets omitted to respect context limits.)*")

        full_doc = "\n\n---\n\n".join(doc_sections)
        summary = f"Spreadsheet with {total_sheets} sheet(s), {total_rows_all_sheets:,} total rows"
        if is_massive:
            summary += " [smart truncated preview + schema]"

        return {
            "extracted_text": full_doc,
            "summary": summary,
            "metadata": {
                "sheets": total_sheets,
                "total_rows": total_rows_all_sheets,
                "sheet_names": sheet_names,
                "is_massive": is_massive
            },
            "error": None
        }
    except Exception as e:
        return {
            "extracted_text": f"### Spreadsheet: `{filename}`\n*(Error parsing spreadsheet: {str(e)})*",
            "summary": f"Spreadsheet parse error: {str(e)}",
            "metadata": {"error": str(e)},
            "error": str(e)
        }

def extract_powerpoint(filename: str, file_bytes: bytes, size_kb: float) -> dict:
    if Presentation is None:
        return {
            "extracted_text": f"[PowerPoint presentation: {filename} uploaded, but 'python-pptx' is not installed]",
            "summary": "PowerPoint (python-pptx unavailable)",
            "metadata": {"status": "pptx_missing"},
            "error": "python-pptx library not found"
        }

    try:
        prs = Presentation(io.BytesIO(file_bytes))
        num_slides = len(prs.slides)
        if num_slides == 0:
            return {
                "extracted_text": f"### Presentation: `{filename}`\n*(Presentation has 0 slides)*",
                "summary": "Empty PowerPoint presentation",
                "metadata": {"slides": 0},
                "error": None
            }

        is_truncated = False
        selected_slides = []
        if num_slides <= MAX_PPTX_SLIDES:
            selected_slides = list(range(num_slides))
        else:
            is_truncated = True
            head_slides = list(range(25))
            tail_slides = list(range(max(num_slides - 10, 25), num_slides))
            step = max((num_slides - 35) // 8, 1)
            mid_slides = [s for s in range(25, num_slides - 10, step)[:8] if s not in head_slides and s not in tail_slides]
            selected_slides = sorted(list(set(head_slides + mid_slides + tail_slides)))

        doc_parts = [
            f"### PowerPoint Presentation: `{filename}`",
            f"- **Total Slides:** {num_slides:,} | **File Size:** {size_kb:,} KB\n---"
        ]

        if is_truncated:
            doc_parts.append(f"> ℹ️ *Smart Truncation active: Showing {len(selected_slides)} key slides out of {num_slides:,} total slides.*")

        for s_idx in selected_slides:
            slide = prs.slides[s_idx]
            
            slide_title = ""
            try:
                if slide.shapes.title and slide.shapes.title.text.strip():
                    slide_title = f": {slide.shapes.title.text.strip()}"
            except Exception:
                pass

            slide_elements = [f"#### [Slide {s_idx + 1}{slide_title}]"]

            for shape in slide.shapes:
                if shape.has_text_frame:
                    if shape == getattr(slide.shapes, "title", None):
                        continue
                    shape_paras = [p.text.strip() for p in shape.text_frame.paragraphs if p.text.strip()]
                    if shape_paras:
                        slide_elements.append("\n".join(f"- {p}" for p in shape_paras))

            for shape in slide.shapes:
                if shape.has_table:
                    tbl = shape.table
                    tbl_rows = []
                    for row in tbl.rows:
                        row_cells = [cell.text.strip().replace("\r\n", " ").replace("\n", " ").replace("|", "\\|") for cell in row.cells]
                        tbl_rows.append("| " + " | ".join(row_cells) + " |")
                    if tbl_rows:
                        if len(tbl_rows) > 1:
                            col_count = len(tbl.columns)
                            tbl_rows.insert(1, "| " + " | ".join(["---"] * col_count) + " |")
                        slide_elements.append("\n" + "\n".join(tbl_rows))

            try:
                if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                    notes = slide.notes_slide.notes_text_frame.text.strip()
                    if notes:
                        slide_elements.append(f"\n> **Speaker Notes:** {notes}")
            except Exception:
                pass

            doc_parts.append("\n".join(slide_elements))

        full_doc = "\n\n---\n\n".join(doc_parts)
        summary = f"PowerPoint presentation with {num_slides} slide(s)"
        if is_truncated:
            summary += f" ({len(selected_slides)} sampled slides extracted)"

        return {
            "extracted_text": full_doc,
            "summary": summary,
            "metadata": {
                "slides": num_slides,
                "extracted_slides": len(selected_slides),
                "is_truncated": is_truncated
            },
            "error": None
        }
    except Exception as e:
        return {
            "extracted_text": f"### PowerPoint: `{filename}`\n*(Error reading PowerPoint: {str(e)})*",
            "summary": f"PowerPoint parse error: {str(e)}",
            "metadata": {"error": str(e)},
            "error": str(e)
        }

def extract_image(filename: str, file_bytes: bytes, size_kb: float) -> dict:
    if Image is None:
        return {
            "extracted_text": f"[Image file: {filename} ({size_kb:,} KB) uploaded, but 'Pillow' is not installed]",
            "summary": f"Image file ({size_kb} KB)",
            "metadata": {"status": "pillow_missing"},
            "error": "Pillow library not found"
        }

    try:
        img = Image.open(io.BytesIO(file_bytes))
        width, height = img.width, img.height
        img_format = img.format or "IMAGE"
        img_mode = img.mode

        meta_lines = [
            f"### Image Document: `{filename}`",
            f"- **Dimensions:** {width:,} × {height:,} px | **Aspect Ratio:** {round(width / max(height, 1), 2)}:1",
            f"- **Format:** {img_format} | **Color Mode:** {img_mode} | **File Size:** {size_kb:,} KB",
        ]

        exif_details = []
        try:
            exif = img.getexif()
            if exif:
                from PIL.ExifTags import TAGS
                for tag_id, value in exif.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    if tag_name in ("Make", "Model", "DateTime", "Software", "Orientation"):
                        exif_details.append(f"  - `{tag_name}`: {value}")
        except Exception:
            pass

        if exif_details:
            meta_lines.append("- **EXIF Metadata:**\n" + "\n".join(exif_details))

        summary = f"Image ({width}×{height}px, {img_format})"
        meta_lines.append(f"\n*(Image attachment: {width}×{height}px, format: {img_format}, mode: {img_mode}. Metadata parsed cleanly without external OCR engine.)*")

        full_doc = "\n".join(meta_lines)
        return {
            "extracted_text": full_doc,
            "summary": summary,
            "metadata": {
                "width": width,
                "height": height,
                "format": img_format,
                "color_mode": img_mode,
                "ocr_status": "disabled",
                "word_count": 0
            },
            "error": None
        }
    except Exception as e:
        return {
            "extracted_text": f"### Image: `{filename}`\n*(Error analyzing image: {str(e)})*",
            "summary": f"Image parse error: {str(e)}",
            "metadata": {"error": str(e)},
            "error": str(e)
        }

def extract_content_from_file(filename: str, file_bytes: bytes) -> dict:
    """
    Extracts structured Markdown, data tables, or OCR content from uploaded files without Vector DB.
    Directly extracts and formats PDF, Word, Excel, PPTX, CSV, TXT, Code, and Images.
    Implements smart truncation to handle massive 500+ page docs or 50,000+ row spreadsheets safely.
    """
    ext = os.path.splitext(filename)[1].lower()
    size_kb = round(len(file_bytes) / 1024, 2)
    result = {
        "filename": filename,
        "extension": ext,
        "size_kb": size_kb,
        "success": True,
        "extracted_text": "",
        "summary": "",
        "metadata": {},
        "error": None
    }

    try:
        # 1. Plain text & Code files
        if ext in CODE_LANG_MAP or filename.lower() in ("dockerfile", "makefile", "license", "readme", ".gitignore", ".env"):
            res = extract_text_code(filename, file_bytes, size_kb)
            result.update(res)
            return result

        # 2. PDF Documents (.pdf)
        if ext == ".pdf":
            res = extract_pdf(filename, file_bytes, size_kb)
            result.update(res)
            return result

        # 3. Word Documents (.docx, .doc)
        if ext in (".docx", ".doc"):
            res = extract_word(filename, file_bytes, size_kb)
            result.update(res)
            return result

        # 4. Spreadsheets (.xlsx, .xls, .csv, .tsv)
        if ext in (".xlsx", ".xls", ".csv", ".tsv"):
            res = extract_spreadsheet(filename, file_bytes, size_kb)
            result.update(res)
            return result

        # 5. PowerPoint Presentations (.pptx)
        if ext == ".pptx":
            res = extract_powerpoint(filename, file_bytes, size_kb)
            result.update(res)
            return result

        # 6. Images (.png, .jpg, .jpeg, .webp, .bmp, .gif, .tiff, .tif, .ico, .svg)
        if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif", ".ico", ".svg"):
            if ext == ".svg":
                res = extract_text_code(filename, file_bytes, size_kb)
                result.update(res)
                return result
            res = extract_image(filename, file_bytes, size_kb)
            result.update(res)
            return result

        # 7. Audio / Video files
        if ext in (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".mkv", ".webm", ".avi", ".mov"):
            media_type = "Audio" if ext in (".mp3", ".wav", ".m4a", ".ogg", ".flac") else "Video"
            desc = (
                f"### {media_type} File: `{filename}`\n"
                f"- **Format:** {ext.upper()[1:]} | **Size:** {size_kb:,} KB\n\n"
                f"Direct {media_type.lower()} file metadata parsed without vector database. "
                f"You may ask questions, transcription guidance, or processing instructions regarding this media."
            )
            result["extracted_text"] = desc
            result["summary"] = f"{media_type} file ({size_kb:,} KB)"
            result["metadata"] = {"media_type": media_type, "format": ext}
            return result

        # 8. Generic binary / unknown fallback
        raw_text = decode_text_bytes(file_bytes)
        if len(raw_text.strip()) > 30:
            trunc_text, is_trunc = truncate_text_content(raw_text, filename)
            result["extracted_text"] = f"### File: `{filename}` (Decoded Text)\n- **Size:** {size_kb:,} KB\n\n```text\n{trunc_text}\n```"
            result["summary"] = f"Decoded file content ({len(raw_text):,} chars)"
            result["metadata"] = {"is_truncated": is_trunc}
            return result

        result["extracted_text"] = f"### Attached Binary File: `{filename}`\n- **Size:** {size_kb:,} KB | **Type:** `{ext}`"
        result["summary"] = f"Attached file ({size_kb:,} KB)"
        result["metadata"] = {"binary": True}
        return result

    except Exception as e:
        logger.error(f"Error parsing file {filename}: {e}", exc_info=True)
        result["success"] = False
        result["error"] = str(e)
        result["extracted_text"] = f"### File Error: `{filename}`\n*(Error extracting file content: {str(e)})*"
        result["summary"] = f"File error: {str(e)}"
        return result

@app.post("/v1/files/extract")
async def extract_file_endpoint(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """
    Extracts text, data tables, or visual metadata from uploaded files without Vector DB.
    Guarded by security checks (MIME & size) and persistent disk caching.
    """
    temp_path = None
    try:
        content_bytes = await file.read()
        filename = file.filename or "unknown_file"

        # 1. Save to temporary file for security & chunk inspection
        import tempfile
        ext = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
            tf.write(content_bytes)
            temp_path = tf.name

        # 2. STEP 1: Security Check (Size, Magic MIME, Double Extension)
        is_safe, sec_msg = check_file(temp_path, filename)
        if not is_safe:
            logger.warning(f"Security check rejected file '{filename}': {sec_msg}")
            return JSONResponse(status_code=400, content={"success": False, "error": sec_msg, "extracted_text": ""})

        # 3. STEP 2: Cache Check
        cached_res = get_from_cache(filename, "extract")
        if cached_res and isinstance(cached_res, dict):
            if session_id and cached_res.get("success", True):
                sess = session_memory_store.get_or_create(session_id)
                pinned_doc = sess.pin_document(
                    filename=cached_res.get("filename", filename),
                    extracted_text=cached_res.get("extracted_text", ""),
                    summary=cached_res.get("summary", ""),
                    size_kb=cached_res.get("size_kb", 0.0)
                )
                cached_res["session_id"] = session_id
                cached_res["pinned_to_session"] = True
                cached_res["doc_id"] = pinned_doc.doc_id
            cached_res["from_cache"] = True
            return JSONResponse(status_code=200, content=cached_res)

        # 4. STEP 3: Multi-Format Extraction
        res = extract_content_from_file(filename, content_bytes)

        # Pin extracted document to session working memory if session_id provided
        if session_id and res.get("success", True):
            sess = session_memory_store.get_or_create(session_id)
            pinned_doc = sess.pin_document(
                filename=res.get("filename", filename),
                extracted_text=res.get("extracted_text", ""),
                summary=res.get("summary", ""),
                size_kb=res.get("size_kb", 0.0)
            )
            res["session_id"] = session_id
            res["pinned_to_session"] = True
            res["doc_id"] = pinned_doc.doc_id

        # 5. STEP 4: Cache the extraction
        if res.get("success", True):
            save_to_cache(filename, "extract", res)

        return JSONResponse(status_code=200, content=res)
    except Exception as e:
        logger.error(f"Error in /v1/files/extract: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e), "extracted_text": ""})
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.post("/v1/chat/upload-and-complete")
async def chat_upload_and_complete(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    max_tokens: Optional[int] = Form(300),
    temperature: Optional[float] = Form(0.7),
    session_id: Optional[str] = Form(None)
):
    """
    Convenience multipart endpoint to upload a file + question together in one request.
    Protected by security check, cache lookup, and big-file chunking.
    """
    if model is None:
        try:
            load_local_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model is not loaded: {str(e)}")

    content_bytes = await file.read()
    filename = file.filename or "file"
    user_prompt = (prompt or "Please analyze and summarize this attached file.").strip()

    # 1. Security Check
    import tempfile
    ext = os.path.splitext(filename)[1]
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
            tf.write(content_bytes)
            temp_path = tf.name

        is_safe, sec_msg = check_file(temp_path, filename)
        if not is_safe:
            raise HTTPException(status_code=400, detail=f"Security check failed: {sec_msg}")

        # 2. Cache Check (if exact question on this file was answered before)
        cached_answer = get_from_cache(filename, user_prompt)
        if cached_answer and isinstance(cached_answer, str):
            target_session_id = (session_id or "").strip() or f"sess_{uuid.uuid4().hex[:12]}"
            sess = session_memory_store.get_or_create(target_session_id)
            sess.add_message("user", user_prompt)
            sess.add_message("assistant", cached_answer)
            return {
                "status": "success",
                "session_id": target_session_id,
                "file": {"filename": filename, "size_kb": round(len(content_bytes) / 1024, 2)},
                "query": user_prompt,
                "completion": cached_answer,
                "from_cache": True
            }

        # 3. Extraction & Big File Handling
        extracted = extract_content_from_file(filename, content_bytes)
        file_context = extracted.get("extracted_text", "")

        # Session Short-Term Working Memory persistence
        target_session_id = (session_id or "").strip() or f"sess_{uuid.uuid4().hex[:12]}"
        sess = session_memory_store.get_or_create(target_session_id)
        if extracted.get("success", True):
            sess.pin_document(
                filename=extracted["filename"],
                extracted_text=file_context,
                summary=extracted.get("summary", ""),
                size_kb=extracted.get("size_kb", 0.0)
            )
        sess.add_message("user", user_prompt)

        # Check if file has many chunks (> 5 chunks)
        clean_ext = ext.lstrip(".").lower()
        chunks = extract_chunks(temp_path, clean_ext)
        if len(chunks) > 5:
            # Use big file handler to select only relevant chunks for crisp context
            def _local_caller(p):
                return generate_local(p, max_tokens=max_tokens or 300, temperature=temperature or 0.7)
            generated = process_big_file(temp_path, clean_ext, user_prompt, model_func=_local_caller)
        else:
            combined_user_content = (
                f"[ATTACHED FILE: {extracted['filename']} ({extracted['summary']})]\n"
                f"--- BEGIN FILE CONTENT ---\n{file_context}\n--- END FILE CONTENT ---\n\n"
                f"User Query:\n{user_prompt}"
            )
            messages = [
                Message(role="system", content=DEFAULT_SYSTEM_PROMPT),
                Message(role="user", content=combined_user_content)
            ]
            prompt_text = construct_chatml_prompt(messages)
            generated = generate_local(
                prompt_text=prompt_text,
                max_tokens=max_tokens or 300,
                temperature=temperature or 0.7,
                top_p=0.95
            )

        sess.add_message("assistant", generated)
        save_to_cache(filename, user_prompt, generated)

        return {
            "status": "success",
            "session_id": target_session_id,
            "file": {
                "filename": extracted["filename"],
                "size_kb": extracted["size_kb"],
                "summary": extracted["summary"]
            },
            "query": user_prompt,
            "completion": generated
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

# --- Short-Term Working Memory Management Endpoints ---

@app.get("/v1/sessions")
async def list_active_sessions():
    """Returns a list of all active sessions in the Short-Term Memory store."""
    sessions = session_memory_store.list_sessions()
    return {
        "status": "success",
        "active_session_count": len(sessions),
        "sessions": sessions
    }

@app.get("/v1/sessions/{session_id}/memory")
async def get_session_memory(session_id: str, include_doc_text: Optional[bool] = False):
    """
    Retrieves the short-term memory state for a session, including rolling buffer
    turns and pinned document working memory.
    """
    sess = session_memory_store.get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or expired.")
    return JSONResponse(status_code=200, content=sess.to_dict(include_doc_text=include_doc_text))

@app.delete("/v1/sessions/{session_id}/memory")
async def delete_session_memory(session_id: str):
    """
    Clears the short-term memory (rolling buffer and pinned documents) for a session.
    """
    deleted = session_memory_store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or already deleted.")
    return {
        "status": "deleted",
        "session_id": session_id,
        "message": f"Short-term working memory for session '{session_id}' successfully cleared."
    }

# --- Global Exception Handler (Prevents blank 500 errors) ---
@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error processing {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error": str(exc), "path": request.url.path}
    )

# --- Fully Responsive GenAI Chatbot with Dark & Light Theme Switcher ---
HTML_PLAYGROUND = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
  <title>Somi • AI Co-Founder | Hermes-3 8B Architecture</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <!-- Markdown & Syntax Highlighting -->
  <script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <style>
    /* Light Theme (Default Existing UI) */
    :root {
      --bg-page: #e8ecf4;
      --bg-app: #f1f4fa;
      --sidebar-bg: #ffffff;
      --card-bg: #ffffff;
      --card-border: rgba(226, 232, 240, 0.85);
      --card-shadow: rgba(9, 18, 42, 0.03);
      --card-shadow-hover: rgba(9, 18, 42, 0.08);
      --primary-navy: #09122a;
      --primary-btn: #fee75c;
      --primary-btn-hover: #fcd734;
      --border-soft: #e2e8f0;
      --text-heading: #1e293b;
      --text-sub: #64748b;
      --text-muted: #94a3b8;
      --user-bubble-bg: #09122a;
      --user-bubble-text: #ffffff;
      --asst-bubble-bg: #ffffff;
      --asst-bubble-text: #1e293b;
      --asst-bubble-border: #e2e8f0;
      --asst-code-inline: #f1f5f9;
      --pill-bg: #ffffff;
      --pill-border: #09122a;
      --pill-text: #1e293b;
      --pill-btn-bg: #09122a;
      --pill-btn-hover: #1e293b;
      --hero-title-color: #92a4c4;
      --history-active-bg: #e9eff8;
      --history-hover-bg: #f1f5f9;
      --history-icon-bg: #edf2f9;
      --theme-toggle-bg: #f1f5f9;
      --theme-toggle-border: #e2e8f0;
      --theme-toggle-text: #64748b;
      --window-border: rgba(255, 255, 255, 0.8);
      --window-shadow: rgba(9, 18, 42, 0.08);
    }

    /* Premium Dark Theme */
    body.dark-theme {
      --bg-page: #060911;
      --bg-app: #0c1220;
      --sidebar-bg: #101728;
      --card-bg: #152238;
      --card-border: #1e2e4a;
      --card-shadow: rgba(0, 0, 0, 0.25);
      --card-shadow-hover: rgba(0, 0, 0, 0.45);
      --primary-navy: #3b82f6;
      --primary-btn: #fee75c;
      --primary-btn-hover: #fcd734;
      --border-soft: #1c2b46;
      --text-heading: #f1f5f9;
      --text-sub: #94a3b8;
      --text-muted: #64748b;
      --user-bubble-bg: #2563eb;
      --user-bubble-text: #ffffff;
      --asst-bubble-bg: #152238;
      --asst-bubble-text: #f1f5f9;
      --asst-bubble-border: #1e2e4a;
      --asst-code-inline: #0d1527;
      --pill-bg: #152238;
      --pill-border: #3b82f6;
      --pill-text: #f8fafc;
      --pill-btn-bg: #3b82f6;
      --pill-btn-hover: #2563eb;
      --hero-title-color: #8da2c7;
      --history-active-bg: #192742;
      --history-hover-bg: #16223a;
      --history-icon-bg: #192742;
      --theme-toggle-bg: #16223a;
      --theme-toggle-border: #1c2b46;
      --theme-toggle-text: #94a3b8;
      --window-border: #1a2740;
      --window-shadow: rgba(0, 0, 0, 0.5);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body {
      height: 100%;
      width: 100%;
      overflow: hidden;
    }
    body {
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg-page);
      color: var(--text-heading);
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 16px;
      transition: background 0.25s ease, color 0.25s ease;
    }

    /* Outer Application Container Card (Desktop default) */
    .app-window {
      width: 100%;
      max-width: 1360px;
      height: 94vh;
      background: var(--bg-app);
      border-radius: 28px;
      box-shadow: 0 20px 50px var(--window-shadow);
      display: flex;
      position: relative;
      overflow: hidden;
      border: 1px solid var(--window-border);
      transition: background 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }

    /* Backdrop for Mobile Drawer */
    .sidebar-backdrop {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.55);
      backdrop-filter: blur(3px);
      z-index: 90;
      opacity: 0;
      transition: opacity 0.25s ease;
      pointer-events: none;
    }
    .sidebar-backdrop.active {
      display: block;
      opacity: 1;
      pointer-events: auto;
    }

    /* Left Sidebar */
    .sidebar {
      width: 280px;
      background: var(--sidebar-bg);
      border-right: 1px solid var(--border-soft);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      padding: 22px 18px 20px;
      flex-shrink: 0;
      z-index: 100;
      transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), background 0.25s ease, border-color 0.25s ease;
    }
    .sidebar-top {
      display: flex;
      flex-direction: column;
      gap: 12px;
      overflow: hidden;
    }
    .sidebar-drawer-header {
      display: none;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 6px;
      padding: 0 4px;
    }
    .sidebar-drawer-title {
      font-size: 1rem;
      font-weight: 700;
      color: var(--text-heading);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .drawer-close-btn {
      background: var(--theme-toggle-bg);
      border: 1px solid var(--border-soft);
      width: 32px;
      height: 32px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      font-size: 1.1rem;
      color: var(--text-sub);
    }

    /* Action Buttons Row in Sidebar */
    .btn-row {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .new-chat-btn {
      flex: 1;
      background: var(--primary-btn);
      color: #0f172a;
      border: none;
      padding: 12px 18px;
      border-radius: 9999px;
      font-size: 0.88rem;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      box-shadow: 0 4px 12px rgba(254, 231, 92, 0.4);
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .new-chat-btn:hover {
      background: var(--primary-btn-hover);
      transform: translateY(-1px);
    }
    .icon-action-btn {
      background: var(--theme-toggle-bg);
      color: var(--theme-toggle-text);
      border: 1px solid var(--border-soft);
      width: 42px;
      height: 42px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.05rem;
      cursor: pointer;
      transition: all 0.2s;
      flex-shrink: 0;
    }
    .icon-action-btn:hover {
      background: var(--history-hover-bg);
      color: var(--text-heading);
      transform: scale(1.05);
    }
    .icon-action-btn.danger:hover {
      background: #fee2e2;
      color: #ef4444;
      border-color: #fca5a5;
    }
    body.dark-theme .icon-action-btn.danger:hover {
      background: #7f1d1d;
      color: #fca5a5;
      border-color: #991b1b;
    }

    /* Inner Menu Theme Switcher (Segmented Pill) */
    .theme-switch-row {
      margin-top: 2px;
      display: flex;
      justify-content: center;
    }
    .theme-pill-selector {
      width: 100%;
      background: var(--bg-app);
      border: 1px solid var(--border-soft);
      border-radius: 9999px;
      padding: 3px;
      display: flex;
      gap: 2px;
    }
    .theme-pill-btn {
      flex: 1;
      border: none;
      background: transparent;
      padding: 7px 12px;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-sub);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .theme-pill-btn:hover:not(.active) {
      color: var(--text-heading);
    }
    .theme-pill-btn.active {
      background: var(--sidebar-bg);
      color: var(--text-heading);
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    body.dark-theme .theme-pill-btn.active {
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
    }

    /* Conversation History */
    .history-title {
      font-size: 0.92rem;
      font-weight: 700;
      color: var(--text-heading);
      margin-top: 14px;
      margin-bottom: 6px;
      padding-left: 6px;
    }
    .history-list {
      flex: 1;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-top: 4px;
      max-height: calc(94vh - 270px);
      padding-right: 4px;
    }
    .history-item {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      border-radius: 12px;
      color: var(--text-sub);
      font-size: 0.84rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s ease;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      position: relative;
    }
    .history-item:hover {
      background: var(--history-hover-bg);
      color: var(--text-heading);
    }
    .history-item.active {
      background: var(--history-active-bg);
      color: var(--primary-navy);
      font-weight: 600;
    }
    body.dark-theme .history-item.active {
      color: #60a5fa;
    }
    .chat-bubble-icon {
      width: 24px;
      height: 24px;
      border-radius: 6px;
      background: var(--history-icon-bg);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.8rem;
      flex-shrink: 0;
    }

    /* Bottom Profile / Model Card */
    .profile-card {
      background: var(--primary-navy);
      border-radius: 16px;
      padding: 12px 14px;
      display: flex;
      align-items: center;
      gap: 10px;
      color: #ffffff;
      box-shadow: 0 6px 18px rgba(9, 18, 42, 0.18);
      transition: background 0.25s ease;
    }
    body.dark-theme .profile-card {
      background: #090e1a;
      border: 1px solid #1a2740;
    }
    .profile-avatar {
      width: 36px;
      height: 36px;
      border-radius: 50%;
      background: linear-gradient(135deg, #3b82f6, #fee75c);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.1rem;
      flex-shrink: 0;
    }
    .profile-info {
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .profile-name {
      font-size: 0.85rem;
      font-weight: 700;
      color: #ffffff;
    }
    .profile-badge {
      font-size: 0.68rem;
      color: #38bdf8;
      font-weight: 500;
    }

    /* Main Chat Workspace */
    .main-chat {
      flex: 1;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      position: relative;
      background: var(--bg-app);
      overflow: hidden;
      height: 100%;
      transition: background 0.25s ease;
    }

    /* Mobile Top Header (Visible only on tablets & phones) */
    .mobile-header {
      display: none;
      align-items: center;
      justify-content: space-between;
      padding: 12px 18px;
      background: var(--sidebar-bg);
      border-bottom: 1px solid var(--border-soft);
      z-index: 40;
      transition: background 0.25s ease, border-color 0.25s ease;
    }
    .mobile-nav-toggle {
      background: var(--theme-toggle-bg);
      border: 1px solid var(--border-soft);
      width: 38px;
      height: 38px;
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.15rem;
      cursor: pointer;
      color: var(--text-heading);
    }
    .mobile-brand {
      display: flex;
      align-items: center;
      gap: 8px;
      font-weight: 700;
      font-size: 1rem;
      color: var(--text-heading);
    }
    .mobile-brand .badge {
      font-size: 0.65rem;
      background: #fee75c;
      color: #0f172a;
      padding: 2px 6px;
      border-radius: 4px;
      font-weight: 700;
    }
    .mobile-actions {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .mobile-action-btn {
      background: var(--theme-toggle-bg);
      border: 1px solid var(--border-soft);
      width: 36px;
      height: 36px;
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.95rem;
      cursor: pointer;
      color: var(--text-heading);
    }

    /* Scrollable Container */
    .chat-scroll {
      flex: 1;
      overflow-y: auto;
      padding: 30px 40px 10px;
      display: flex;
      flex-direction: column;
      scroll-behavior: smooth;
      -webkit-overflow-scrolling: touch;
    }

    /* Welcome Hero State */
    .hero-view {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      margin: auto 0;
      padding: 20px 0;
      animation: fadeIn 0.3s ease-out;
      width: 100%;
    }
    .hero-title {
      font-size: 2.1rem;
      font-weight: 700;
      color: var(--hero-title-color);
      margin-bottom: 34px;
      letter-spacing: -0.02em;
      text-align: center;
      transition: color 0.25s ease;
    }

    /* 6 Prompt Cards Grid */
    .prompt-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
      width: 100%;
      max-width: 920px;
    }
    .prompt-card {
      background: var(--card-bg);
      border-radius: 18px;
      padding: 28px 20px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      gap: 12px;
      box-shadow: 0 4px 16px var(--card-shadow);
      border: 1px solid var(--card-border);
      cursor: pointer;
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
      -webkit-tap-highlight-color: transparent;
    }
    .prompt-card:hover {
      transform: translateY(-4px);
      box-shadow: 0 10px 24px var(--card-shadow-hover);
      border-color: #3b82f6;
    }
    .prompt-card:active {
      transform: scale(0.98);
    }
    .prompt-icon {
      font-size: 1.8rem;
    }
    .prompt-text {
      font-size: 0.88rem;
      font-weight: 600;
      color: var(--text-heading);
    }

    /* Messages Stream View */
    .messages-stream {
      display: flex;
      flex-direction: column;
      gap: 18px;
      max-width: 900px;
      width: 100%;
      margin: 0 auto;
    }
    .msg-row {
      display: flex;
      gap: 12px;
      animation: slideIn 0.2s ease-out;
    }
    @keyframes slideIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .msg-row.user {
      justify-content: flex-end;
    }
    .bubble {
      padding: 14px 18px;
      border-radius: 18px;
      font-size: 0.94rem;
      line-height: 1.6;
      word-break: break-word;
      transition: background 0.2s ease, color 0.2s ease;
    }
    .msg-row.user .bubble {
      background: var(--user-bubble-bg);
      color: var(--user-bubble-text);
      border-bottom-right-radius: 4px;
      max-width: 80%;
      white-space: pre-wrap;
    }
    .msg-row.assistant .bubble {
      background: var(--asst-bubble-bg);
      color: var(--asst-bubble-text);
      border: 1px solid var(--asst-bubble-border);
      border-bottom-left-radius: 4px;
      box-shadow: 0 4px 16px var(--card-shadow);
      max-width: 86%;
    }

    /* Markdown inside Assistant Bubble */
    .bubble p { margin-bottom: 12px; }
    .bubble p:last-child { margin-bottom: 0; }
    .bubble h1, .bubble h2, .bubble h3 { color: var(--text-heading); margin: 14px 0 6px; font-weight: 700; }
    .bubble ul, .bubble ol { margin-left: 20px; margin-bottom: 12px; }
    .bubble li { margin-bottom: 4px; }
    .bubble code {
      background: var(--asst-code-inline);
      font-family: 'JetBrains Mono', monospace;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 0.86rem;
      color: #e11d48;
    }
    body.dark-theme .bubble code {
      color: #f43f5e;
    }

    /* Code Block inside Assistant Bubble */
    .code-wrapper {
      position: relative;
      margin: 12px 0;
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid #1e293b;
      background: #0f172a;
    }
    .code-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #1e293b;
      padding: 6px 14px;
      font-size: 0.74rem;
      color: #94a3b8;
      font-family: 'JetBrains Mono', monospace;
    }
    .copy-code-btn {
      background: transparent;
      border: none;
      color: #cbd5e1;
      font-size: 0.72rem;
      cursor: pointer;
    }
    .copy-code-btn:hover { color: #38bdf8; }
    .code-wrapper pre {
      margin: 0;
      padding: 14px;
      overflow-x: auto;
    }
    .code-wrapper pre code {
      background: transparent !important;
      padding: 0 !important;
      color: #e2e8f0;
      font-size: 0.85rem;
      font-family: 'JetBrains Mono', monospace;
      line-height: 1.5;
    }

    /* Telemetry Badge */
    .telemetry-meta {
      font-size: 0.72rem;
      color: var(--text-muted);
      margin-top: 6px;
      font-family: 'JetBrains Mono', monospace;
    }

    /* Thinking State */
    .thinking-dots {
      display: flex;
      align-items: center;
      gap: 6px;
      color: var(--text-sub);
      font-size: 0.86rem;
    }
    .dot-pulse {
      display: inline-block;
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #3b82f6;
      animation: pulseDot 1.2s infinite ease-in-out;
    }
    .dot-pulse:nth-child(2) { animation-delay: 0.2s; }
    .dot-pulse:nth-child(3) { animation-delay: 0.4s; }
    @keyframes pulseDot {
      0%, 80%, 100% { transform: scale(0); opacity: 0.3; }
      40% { transform: scale(1); opacity: 1; }
    }

    /* Streaming Cursor */
    .typing-cursor {
      display: inline-block;
      width: 7px;
      height: 14px;
      background: var(--text-heading);
      margin-left: 2px;
      animation: blink 0.9s infinite;
      vertical-align: middle;
    }
    @keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0; } }

    /* Bottom Navy Pill Input Bar */
    .input-dock {
      padding: 14px 40px 22px;
      display: flex;
      flex-direction: column;
      align-items: center;
      background: linear-gradient(180deg, transparent 0%, var(--bg-app) 45%);
    }
    .file-attachment-container {
      width: 100%;
      max-width: 920px;
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .file-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: var(--card-bg);
      border: 1.5px solid var(--card-border);
      border-radius: 9999px;
      padding: 5px 14px;
      font-size: 0.82rem;
      color: var(--text-heading);
      box-shadow: 0 4px 12px var(--card-shadow);
      transition: all 0.2s;
    }
    .file-chip-icon {
      font-size: 1.05rem;
    }
    .file-chip-name {
      font-weight: 600;
      max-width: 260px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .file-chip-size {
      color: var(--text-muted);
      font-size: 0.75rem;
      font-family: 'JetBrains Mono', monospace;
    }
    .file-chip-remove {
      background: transparent;
      border: none;
      color: var(--text-muted);
      cursor: pointer;
      font-size: 0.95rem;
      padding: 0 3px;
      line-height: 1;
      transition: color 0.15s;
    }
    .file-chip-remove:hover {
      color: #ef4444;
    }
    .attach-pill-btn {
      width: 36px;
      height: 36px;
      border-radius: 50%;
      background: transparent;
      color: var(--text-sub);
      border: none;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s;
      flex-shrink: 0;
    }
    .attach-pill-btn:hover {
      color: #3b82f6;
      background: var(--btn-hover-bg);
      transform: scale(1.05);
    }
    .user-attachment-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(255, 255, 255, 0.2);
      border: 1px solid rgba(255, 255, 255, 0.35);
      border-radius: 10px;
      padding: 3px 10px;
      font-size: 0.76rem;
      margin-bottom: 6px;
      font-weight: 600;
      width: fit-content;
    }
    body.dark-theme .user-attachment-badge {
      background: rgba(59, 130, 246, 0.2);
      border-color: rgba(59, 130, 246, 0.45);
    }
    .pill-input-box {
      width: 100%;
      max-width: 920px;
      background: var(--pill-bg);
      border: 2.5px solid var(--pill-border);
      border-radius: 9999px;
      padding: 6px 8px 6px 14px;
      display: flex;
      align-items: center;
      gap: 10px;
      box-shadow: 0 8px 24px var(--card-shadow);
      transition: all 0.2s;
    }
    .pill-input-box:focus-within {
      box-shadow: 0 10px 30px var(--card-shadow-hover);
      transform: translateY(-1px);
    }
    .chat-input {
      flex: 1;
      border: none;
      background: transparent;
      font-size: 0.98rem;
      font-family: inherit;
      color: var(--pill-text);
      outline: none;
      padding: 6px 0;
      min-width: 0;
    }
    .chat-input::placeholder {
      color: var(--text-muted);
    }
    .send-pill-btn {
      width: 44px;
      height: 44px;
      border-radius: 50%;
      background: var(--pill-btn-bg);
      color: #ffffff;
      border: none;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.15rem;
      transition: all 0.2s;
      flex-shrink: 0;
    }
    .send-pill-btn:hover:not(:disabled) {
      background: var(--pill-btn-hover);
      transform: scale(1.05);
    }
    .send-pill-btn:disabled {
      background: var(--text-muted);
      cursor: not-allowed;
    }
    .send-pill-btn.stop {
      background: #ef4444;
    }
    .send-pill-btn.stop:hover {
      background: #dc2626;
    }

    /* ========================================================= */
    /* RESPONSIVE BREAKPOINTS (Desktop, Tablet, Mobile)          */
    /* ========================================================= */

    /* 1. TABLET VIEW (768px to 1024px) */
    @media (max-width: 1024px) {
      body {
        padding: 0;
      }
      .app-window {
        width: 100vw;
        height: 100vh;
        border-radius: 0;
        border: none;
      }
      .sidebar {
        width: 250px;
        padding: 20px 14px 16px;
      }
      .chat-scroll {
        padding: 24px 28px 10px;
      }
      .prompt-grid {
        grid-template-columns: repeat(2, 1fr);
        gap: 14px;
      }
      .prompt-card {
        padding: 22px 16px;
      }
      .hero-title {
        font-size: 1.8rem;
        margin-bottom: 24px;
      }
      .input-dock {
        padding: 14px 28px 20px;
      }
    }

    /* 2. MOBILE VIEW (under 768px) */
    @media (max-width: 768px) {
      body {
        padding: 0;
      }
      .app-window {
        width: 100vw;
        height: 100vh;
        height: 100dvh;
        border-radius: 0;
        border: none;
      }

      /* Sliding Mobile Drawer */
      .sidebar {
        position: fixed;
        left: 0;
        top: 0;
        bottom: 0;
        width: 290px;
        max-width: 84vw;
        transform: translateX(-100%);
        box-shadow: none;
        z-index: 100;
      }
      .sidebar.open {
        transform: translateX(0);
        box-shadow: 12px 0 35px rgba(0, 0, 0, 0.4);
      }
      .sidebar-drawer-header {
        display: flex;
      }
      .history-list {
        max-height: calc(100vh - 280px);
      }

      /* Top Mobile App Header */
      .mobile-header {
        display: flex;
      }

      /* Main Area */
      .main-chat {
        height: 100%;
      }
      .chat-scroll {
        padding: 16px 14px 10px;
      }
      .hero-view {
        padding: 10px 0;
      }
      .hero-title {
        font-size: 1.45rem;
        margin-bottom: 18px;
      }

      /* Prompt Cards (Single Column) */
      .prompt-grid {
        grid-template-columns: 1fr;
        gap: 10px;
      }
      .prompt-card {
        padding: 16px 16px;
        flex-direction: row;
        justify-content: flex-start;
        text-align: left;
        gap: 14px;
        border-radius: 14px;
      }
      .prompt-icon {
        font-size: 1.4rem;
      }
      .prompt-text {
        font-size: 0.85rem;
      }

      /* Message Bubbles on Mobile */
      .msg-row.user .bubble {
        max-width: 90%;
        padding: 12px 14px;
        font-size: 0.9rem;
      }
      .msg-row.assistant .bubble {
        max-width: 95%;
        padding: 14px 14px;
        font-size: 0.9rem;
      }

      /* Bottom Input Dock on Mobile */
      .input-dock {
        padding: 10px 14px;
        padding-bottom: max(14px, env(safe-area-inset-bottom));
      }
      .pill-input-box {
        padding: 4px 6px 4px 16px;
        border-width: 2px;
      }
      .chat-input {
        font-size: 0.92rem;
        padding: 4px 0;
      }
      .send-pill-btn {
        width: 38px;
        height: 38px;
        font-size: 1rem;
      }
    }

    /* Small Mobile Phones (under 480px) */
    @media (max-width: 480px) {
      .hero-title {
        font-size: 1.35rem;
      }
      .sidebar {
        width: 270px;
      }
      .prompt-card {
        padding: 14px 14px;
      }
    }
  </style>
</head>
<body>

  <!-- Backdrop for Mobile Sidebar Drawer -->
  <div class="sidebar-backdrop" id="sidebarBackdrop"></div>

  <!-- Main Application Window -->
  <div class="app-window">

    <!-- Left Sidebar (Inner Menu with Drawer on Mobile / Fixed on Desktop) -->
    <aside class="sidebar" id="sidebar">
      <div class="sidebar-top">
        <!-- Drawer Header for Mobile -->
        <div class="sidebar-drawer-header">
          <div class="sidebar-drawer-title">
            <span>⚡ Somi • AI Co-Founder</span>
          </div>
          <button class="drawer-close-btn" id="closeDrawerBtn" title="Close Drawer">✕</button>
        </div>

        <!-- Top Action Buttons Row -->
        <div class="btn-row">
          <button class="new-chat-btn" id="newChatBtn">
            <span>+</span> New Conversation
          </button>
          <button class="icon-action-btn danger" id="clearChatBtn" title="Clear Chat History">
            🗑️
          </button>
        </div>

        <!-- Inner Menu Dark / Light Selector (Segmented Pill) -->
        <div class="theme-switch-row">
          <div class="theme-pill-selector">
            <button class="theme-pill-btn active" id="lightThemeOption">
              <span>☀️</span> Light
            </button>
            <button class="theme-pill-btn" id="darkThemeOption">
              <span>🌙</span> Dark
            </button>
          </div>
        </div>

        <div class="history-title">Conversation History</div>
        <div class="history-list" id="historyList">
          <!-- Dynamically populated conversations -->
        </div>
      </div>

      <!-- Bottom Profile / Model Card -->
      <div class="profile-card">
        <div class="profile-avatar">⚡</div>
        <div class="profile-info">
          <div class="profile-name">Somi • AI Co-Founder</div>
          <div class="profile-badge">Hermes-3 8B • A100</div>
        </div>
      </div>
    </aside>

    <!-- Main Chat Workspace -->
    <main class="main-chat">

      <!-- Top Header for Mobile & Tablet -->
      <header class="mobile-header">
        <button class="mobile-nav-toggle" id="openDrawerBtn" title="Open History">☰</button>
        <div class="mobile-brand">
          <span>Somi • AI Co-Founder</span>
          <span class="badge">A100</span>
        </div>
        <div class="mobile-actions">
          <button class="mobile-action-btn" id="mobileThemeToggleBtn" title="Toggle Theme">🌙</button>
          <button class="mobile-action-btn" id="mobileNewChatBtn" title="New Chat">➕</button>
        </div>
      </header>

      <!-- Scrollable Chat Stream -->
      <div class="chat-scroll" id="chatScroll">

        <!-- Welcome Hero View (Matching GenAI Screenshot) -->
        <div class="hero-view" id="heroView">
          <h1 class="hero-title">Start collaborating with Somi • AI Co-Founder</h1>

          <div class="prompt-grid">
            <div class="prompt-card" onclick="triggerPrompt('Write an engaging technical social media post about distributed systems.')">
              <div class="prompt-icon">📢</div>
              <div class="prompt-text">Write Social media post</div>
            </div>

            <div class="prompt-card" onclick="triggerPrompt('Create a 60-second video script explaining how Redis pub/sub works.')">
              <div class="prompt-icon">🎥</div>
              <div class="prompt-text">Create video script</div>
            </div>

            <div class="prompt-card" onclick="triggerPrompt('Write an official product press release for our new low-latency AI API.')">
              <div class="prompt-icon">📰</div>
              <div class="prompt-text">Write a press release</div>
            </div>

            <div class="prompt-card" onclick="triggerPrompt('Give me 5 actionable ergonomics and health tips for software engineers working long hours.')">
              <div class="prompt-icon">📋</div>
              <div class="prompt-text">Health Tips</div>
            </div>

            <div class="prompt-card" onclick="triggerPrompt('Design an optimal PostgreSQL database schema with indexing and relationships for an e-commerce platform.')">
              <div class="prompt-icon">🗄️</div>
              <div class="prompt-text">Design a database schema</div>
            </div>

            <div class="prompt-card" onclick="triggerPrompt('Write clean, responsive frontend code in React with Tailwind CSS for a dashboard card.')">
              <div class="prompt-icon">💻</div>
              <div class="prompt-text">Write frontend code</div>
            </div>
          </div>
        </div>

        <!-- Messages Stream (Shown when chatting) -->
        <div class="messages-stream" id="messagesStream" style="display: none;"></div>

      </div>

      <!-- Bottom Navy Pill Input Dock -->
      <div class="input-dock">
        <!-- File Attachment Preview Bar -->
        <div class="file-attachment-container" id="fileAttachmentContainer" style="display: none;">
          <div class="file-chip">
            <span class="file-chip-icon" id="fileChipIcon">📄</span>
            <span class="file-chip-name" id="fileChipName">document.pdf</span>
            <span class="file-chip-size" id="fileChipSize">(0 KB)</span>
            <button type="button" class="file-chip-remove" id="fileChipRemove" title="Remove attachment">✕</button>
          </div>
        </div>

        <div class="pill-input-box" id="pillInputBox">
          <!-- Hidden File Upload Input -->
          <input type="file" id="fileInput" style="display: none;" accept=".pdf,.docx,.doc,.xlsx,.xls,.csv,.tsv,.pptx,.txt,.md,.py,.js,.jsx,.ts,.tsx,.html,.css,.json,.xml,.sql,.yaml,.yml,.png,.jpg,.jpeg,.webp,.bmp,.mp3,.wav,.m4a">
          
          <!-- Attachment Paperclip Button -->
          <button type="button" id="attachBtn" class="attach-pill-btn" title="Attach Document, Sheet, Image or File">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path>
            </svg>
          </button>

          <input type="text" id="chatInput" class="chat-input" placeholder="Ask a question, request code, or attach a file..." autocomplete="off">
          <button id="sendBtn" class="send-pill-btn" title="Send Message">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          </button>
        </div>
      </div>

    </main>
  </div>

  <script>
    // State
    let conversation = [];
    let isGenerating = false;
    let abortController = null;
    let currentSessionId = null;
    let currentAttachedFile = null;

    // Elements
    const sidebar = document.getElementById("sidebar");
    const sidebarBackdrop = document.getElementById("sidebarBackdrop");
    const openDrawerBtn = document.getElementById("openDrawerBtn");
    const closeDrawerBtn = document.getElementById("closeDrawerBtn");
    const chatScroll = document.getElementById("chatScroll");
    const heroView = document.getElementById("heroView");
    const messagesStream = document.getElementById("messagesStream");
    const chatInput = document.getElementById("chatInput");
    const sendBtn = document.getElementById("sendBtn");
    const newChatBtn = document.getElementById("newChatBtn");
    const mobileNewChatBtn = document.getElementById("mobileNewChatBtn");
    const clearChatBtn = document.getElementById("clearChatBtn");
    const historyList = document.getElementById("historyList");

    // Attachment Elements
    const attachBtn = document.getElementById("attachBtn");
    const fileInput = document.getElementById("fileInput");
    const fileAttachmentContainer = document.getElementById("fileAttachmentContainer");
    const fileChipIcon = document.getElementById("fileChipIcon");
    const fileChipName = document.getElementById("fileChipName");
    const fileChipSize = document.getElementById("fileChipSize");
    const fileChipRemove = document.getElementById("fileChipRemove");
    const pillInputBox = document.getElementById("pillInputBox");

    // Theme Elements
    const mobileThemeToggleBtn = document.getElementById("mobileThemeToggleBtn");
    const lightThemeOption = document.getElementById("lightThemeOption");
    const darkThemeOption = document.getElementById("darkThemeOption");

    // --- Theme Switcher Logic ---
    function applyTheme(theme) {
      const isDark = theme === "dark";
      if (isDark) {
        document.body.classList.add("dark-theme");
        if (mobileThemeToggleBtn) {
          mobileThemeToggleBtn.innerHTML = "☀️";
          mobileThemeToggleBtn.title = "Switch to Light Mode";
        }
        if (darkThemeOption) darkThemeOption.classList.add("active");
        if (lightThemeOption) lightThemeOption.classList.remove("active");
      } else {
        document.body.classList.remove("dark-theme");
        if (mobileThemeToggleBtn) {
          mobileThemeToggleBtn.innerHTML = "🌙";
          mobileThemeToggleBtn.title = "Switch to Dark Mode";
        }
        if (lightThemeOption) lightThemeOption.classList.add("active");
        if (darkThemeOption) darkThemeOption.classList.remove("active");
      }
      localStorage.setItem("hermes_theme", theme);
    }

    function toggleTheme() {
      const current = localStorage.getItem("hermes_theme") || "light";
      applyTheme(current === "dark" ? "light" : "dark");
    }

    if (mobileThemeToggleBtn) mobileThemeToggleBtn.addEventListener("click", toggleTheme);
    if (lightThemeOption) lightThemeOption.addEventListener("click", () => applyTheme("light"));
    if (darkThemeOption) darkThemeOption.addEventListener("click", () => applyTheme("dark"));

    // Apply saved theme immediately
    const savedTheme = localStorage.getItem("hermes_theme") || "light";
    applyTheme(savedTheme);

    // Drawer Controls for Mobile/Tablet
    function openSidebar() {
      sidebar.classList.add("open");
      sidebarBackdrop.classList.add("active");
    }
    function closeSidebar() {
      sidebar.classList.remove("open");
      sidebarBackdrop.classList.remove("active");
    }
    if (openDrawerBtn) openDrawerBtn.addEventListener("click", openSidebar);
    if (closeDrawerBtn) closeDrawerBtn.addEventListener("click", closeSidebar);
    if (sidebarBackdrop) sidebarBackdrop.addEventListener("click", closeSidebar);

    // --- File Attachment & Drag-Drop Logic ---
    function getFileIcon(filename) {
      const ext = filename.split('.').pop().toLowerCase();
      if (['pdf'].includes(ext)) return '📄';
      if (['doc', 'docx'].includes(ext)) return '📝';
      if (['xls', 'xlsx', 'csv', 'tsv'].includes(ext)) return '📊';
      if (['ppt', 'pptx'].includes(ext)) return '📽️';
      if (['png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif', 'svg'].includes(ext)) return '🖼️';
      if (['mp3', 'wav', 'm4a', 'ogg', 'flac'].includes(ext)) return '🎵';
      if (['mp4', 'mkv', 'webm', 'mov'].includes(ext)) return '🎬';
      if (['js', 'ts', 'py', 'html', 'css', 'json', 'sql', 'cpp', 'java'].includes(ext)) return '💻';
      return '📎';
    }

    function formatFileSize(bytes) {
      if (!bytes) return '0 B';
      const k = 1024;
      const sizes = ['B', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    function handleFileSelection(file) {
      if (!file) return;
      const sizeStr = formatFileSize(file.size);
      const icon = getFileIcon(file.name);
      currentAttachedFile = {
        file: file,
        name: file.name,
        sizeFormatted: sizeStr,
        icon: icon
      };
      if (fileChipIcon) fileChipIcon.textContent = icon;
      if (fileChipName) fileChipName.textContent = file.name;
      if (fileChipSize) fileChipSize.textContent = `(${sizeStr})`;
      if (fileAttachmentContainer) fileAttachmentContainer.style.display = "flex";
      chatInput.focus();
    }

    function removeAttachedFile() {
      currentAttachedFile = null;
      if (fileInput) fileInput.value = "";
      if (fileAttachmentContainer) fileAttachmentContainer.style.display = "none";
    }

    if (attachBtn) {
      attachBtn.addEventListener("click", () => {
        if (fileInput) fileInput.click();
      });
    }

    if (fileInput) {
      fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
          handleFileSelection(e.target.files[0]);
        }
      });
    }

    if (fileChipRemove) {
      fileChipRemove.addEventListener("click", (e) => {
        e.stopPropagation();
        removeAttachedFile();
      });
    }

    // Drag and Drop support
    ['dragenter', 'dragover'].forEach(eventName => {
      window.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (pillInputBox) pillInputBox.style.borderColor = "#3b82f6";
      }, false);
    });
    ['dragleave', 'drop'].forEach(eventName => {
      window.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (pillInputBox) pillInputBox.style.borderColor = "";
      }, false);
    });
    window.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files.length > 0) {
        handleFileSelection(dt.files[0]);
      }
    });

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

    const renderer = new marked.Renderer();
    renderer.code = function(code, language) {
      const validLang = language || 'plaintext';
      const highlighted = (language && hljs.getLanguage(language))
        ? hljs.highlight(code, { language }).value
        : hljs.highlightAuto(code).value;
      const id = 'code_' + Math.random().toString(36).substr(2, 9);
      return `
        <div class="code-wrapper">
          <div class="code-header">
            <span>${validLang.toUpperCase()}</span>
            <button class="copy-code-btn" onclick="copyCodeSnippet('${id}')">📋 Copy</button>
          </div>
          <pre><code id="${id}" class="hljs ${validLang}">${highlighted}</code></pre>
        </div>
      `;
    };
    marked.use({ renderer });

    window.copyCodeSnippet = function(id) {
      const el = document.getElementById(id);
      if (el) {
        navigator.clipboard.writeText(el.innerText).then(() => {
          const btn = el.closest('.code-wrapper').querySelector('.copy-code-btn');
          const original = btn.innerText;
          btn.innerText = '✅ Copied!';
          setTimeout(() => { btn.innerText = original; }, 2000);
        });
      }
    };

    // --- Session History Storage ---
    function getStoredSessions() {
      try {
        return JSON.parse(localStorage.getItem("hermes_chat_sessions") || "[]");
      } catch (e) {
        return [];
      }
    }

    function saveSessions(sessions) {
      localStorage.setItem("hermes_chat_sessions", JSON.stringify(sessions));
    }

    function renderHistoryUI() {
      const sessions = getStoredSessions();
      historyList.innerHTML = "";

      if (sessions.length === 0) {
        historyList.innerHTML = `<div style="padding: 14px 12px; font-size: 0.82rem; color: var(--text-muted); text-align: center;">No saved conversations yet</div>`;
        return;
      }

      sessions.forEach(sess => {
        const item = document.createElement("div");
        item.className = "history-item" + (sess.id === currentSessionId ? " active" : "");
        item.innerHTML = `<span class="chat-bubble-icon">💬</span><span>${sess.title}</span>`;
        item.onclick = () => {
          closeSidebar();
          loadSession(sess.id);
        };
        historyList.appendChild(item);
      });
    }

    function startNewSession() {
      if (isGenerating && abortController) abortController.abort();
      closeSidebar();
      currentSessionId = "sess_" + Date.now();
      conversation = [];
      messagesStream.innerHTML = "";
      messagesStream.style.display = "none";
      heroView.style.display = "flex";
      chatInput.value = "";
      chatInput.focus();
      renderHistoryUI();
    }

    function clearAllHistory() {
      if (confirm("Are you sure you want to clear conversation history?")) {
        if (isGenerating && abortController) abortController.abort();
        closeSidebar();
        if (currentSessionId) {
          fetch(`/v1/sessions/${encodeURIComponent(currentSessionId)}/memory`, { method: "DELETE" }).catch(() => {});
        }
        localStorage.removeItem("hermes_chat_sessions");
        startNewSession();
      }
    }

    function loadSession(sessionId) {
      const sessions = getStoredSessions();
      const sess = sessions.find(s => s.id === sessionId);
      if (!sess) return;

      currentSessionId = sess.id;
      conversation = sess.messages || [];

      heroView.style.display = "none";
      messagesStream.style.display = "flex";
      messagesStream.innerHTML = "";

      conversation.forEach(msg => {
        appendRenderedBubble(msg.role, msg.content, msg.attachment);
      });

      renderHistoryUI();
      chatScroll.scrollTop = chatScroll.scrollHeight;
    }

    // --- Interaction ---
    newChatBtn.addEventListener("click", () => {
      removeAttachedFile();
      startNewSession();
    });
    if (mobileNewChatBtn) {
      mobileNewChatBtn.addEventListener("click", () => {
        removeAttachedFile();
        startNewSession();
      });
    }
    clearChatBtn.addEventListener("click", () => {
      removeAttachedFile();
      clearAllHistory();
    });

    chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (!isGenerating) sendUserMessage();
      }
    });

    sendBtn.addEventListener("click", () => {
      if (isGenerating) {
        if (abortController) abortController.abort();
      } else {
        sendUserMessage();
      }
    });

    window.triggerPrompt = function(promptText) {
      chatInput.value = promptText;
      sendUserMessage();
    };

    function escapeHtml(str) {
      return (str || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    }

    function appendRenderedBubble(role, content, attachment) {
      const row = document.createElement("div");
      row.className = `msg-row ${role}`;

      const bubble = document.createElement("div");
      bubble.className = "bubble";

      if (role === "user") {
        if (attachment) {
          const badge = document.createElement("div");
          badge.className = "user-attachment-badge";
          badge.innerHTML = `<span>${attachment.icon || "📎"}</span> <span>${escapeHtml(attachment.name)}</span> <span style="opacity: 0.8; font-size: 0.72rem;">${attachment.sizeFormatted ? `(${escapeHtml(attachment.sizeFormatted)})` : ""}</span>`;
          bubble.appendChild(badge);
        }
        if (content) {
          const txt = document.createElement("div");
          txt.className = "user-text-content";
          txt.textContent = content;
          bubble.appendChild(txt);
        }
      } else {
        bubble.innerHTML = marked.parse(content || "");
      }

      row.appendChild(bubble);
      messagesStream.appendChild(row);
    }

    async function sendUserMessage() {
      const text = chatInput.value.trim();
      if ((!text && !currentAttachedFile) || isGenerating) return;

      if (!currentSessionId) {
        currentSessionId = "sess_" + Date.now();
      }

      heroView.style.display = "none";
      messagesStream.style.display = "flex";

      const attachedSnapshot = currentAttachedFile;
      const userDisplayPrompt = text || (attachedSnapshot ? `Please analyze this file: ${attachedSnapshot.name}` : "");

      // Append user bubble to UI with attachment badge
      appendRenderedBubble("user", userDisplayPrompt, attachedSnapshot);
      conversation.push({
        role: "user",
        content: userDisplayPrompt,
        attachment: attachedSnapshot ? {
          name: attachedSnapshot.name,
          sizeFormatted: attachedSnapshot.sizeFormatted,
          icon: attachedSnapshot.icon
        } : null
      });

      // Clear input and attachment preview
      chatInput.value = "";
      removeAttachedFile();

      // Save/Update session title in storage
      let sessions = getStoredSessions();
      let sess = sessions.find(s => s.id === currentSessionId);
      if (!sess) {
        sess = { id: currentSessionId, title: (attachedSnapshot ? attachedSnapshot.name : userDisplayPrompt).slice(0, 26), messages: [] };
        sessions.unshift(sess);
      }
      sess.messages = conversation;
      saveSessions(sessions);
      renderHistoryUI();

      // Setup Assistant Holder
      isGenerating = true;
      setButtonState(true);
      abortController = new AbortController();

      const startTime = performance.now();
      let firstTokenTime = null;
      let tokenCount = 0;

      const assistantHolder = createAssistantHolder();
      const contentEl = assistantHolder.content;
      const metaEl = assistantHolder.meta;

      let fullResponse = "";

      try {
        let injectedUserContent = userDisplayPrompt;

        // If a file was attached, extract its content via backend first!
        if (attachedSnapshot && attachedSnapshot.file) {
          contentEl.innerHTML = `
            <div class="thinking-dots">
              <span class="dot-pulse"></span>
              <span class="dot-pulse"></span>
              <span class="dot-pulse"></span>
              <span>Extracting content from ${escapeHtml(attachedSnapshot.name)}...</span>
            </div>
          `;
          try {
            const formData = new FormData();
            formData.append("file", attachedSnapshot.file);
            formData.append("session_id", currentSessionId);
            const [extRes] = await Promise.all([
              fetch("/v1/files/extract", {
                method: "POST",
                body: formData,
                signal: abortController.signal
              }),
              Promise.resolve().then(() => saveSessions(sessions))
            ]);
            if (extRes.ok) {
              const fileData = await extRes.json();
              injectedUserContent = `[ATTACHED FILE: ${fileData.filename} (${fileData.summary})]\n--- BEGIN FILE CONTENT ---\n${fileData.extracted_text}\n--- END FILE CONTENT ---\n\nUser Question:\n${userDisplayPrompt}`;
            }
          } catch (extErr) {
            console.warn("File extraction fallback:", extErr);
          }

          contentEl.innerHTML = `
            <div class="thinking-dots">
              <span class="dot-pulse"></span>
              <span class="dot-pulse"></span>
              <span class="dot-pulse"></span>
              <span>Somi is analyzing document...</span>
            </div>
          `;
        }

        // Build messages payload for Hermes-3 (dynamically uses private Somi persona on backend)
        const apiMessages = conversation.map((msg, idx) => {
          if (idx === conversation.length - 1 && msg.role === "user") {
            return { role: "user", content: injectedUserContent };
          }
          return { role: msg.role, content: msg.content };
        });

        const payload = {
          model: "NousResearch/Hermes-3-Llama-3.1-8B",
          session_id: currentSessionId,
          messages: apiMessages,
          temperature: 0.7,
          max_tokens: 300,
          stream: true
        };

        const response = await fetch("/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: abortController.signal
        });

        if (!response.ok) {
          throw new Error("HTTP error " + response.status);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        contentEl.innerHTML = "";

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
                  scheduleBatchedStreamRender(contentEl, fullResponse);
                }
              } catch (err) {}
            }
          }
        }

        if (renderRafId) {
          cancelAnimationFrame(renderRafId);
          renderRafId = null;
        }
        renderStreamedContent(contentEl, fullResponse, false);
        chatScroll.scrollTop = chatScroll.scrollHeight;
        conversation.push({ role: "assistant", content: fullResponse });

        sess.messages = conversation;
        saveSessions(sessions);

        const totalDuration = ((performance.now() - startTime) / 1000).toFixed(2);
        const tokPerSec = tokenCount > 0 ? (tokenCount / Math.max(totalDuration, 0.1)).toFixed(1) : 0;
        metaEl.textContent = `⚡ ${totalDuration}s • ${tokPerSec} tok/s • Somi A100`;

      } catch (err) {
        if (err.name === 'AbortError') {
          if (renderRafId) {
            cancelAnimationFrame(renderRafId);
            renderRafId = null;
          }
          renderStreamedContent(contentEl, fullResponse, false);
          metaEl.textContent = "⚠️ Stopped by user.";
        } else {
          contentEl.innerHTML = `<span style="color: #ef4444;">❌ Error: ${err.message}</span>`;
        }
      } finally {
        if (renderRafId) {
          cancelAnimationFrame(renderRafId);
          renderRafId = null;
        }
        isGenerating = false;
        setButtonState(false);
        abortController = null;
        chatInput.focus();
      }
    }

    let renderRafId = null;
    let pendingStreamText = "";

    function scheduleBatchedStreamRender(el, text) {
      pendingStreamText = text;
      if (!renderRafId) {
        renderRafId = requestAnimationFrame(() => {
          renderRafId = null;
          renderStreamedContent(el, pendingStreamText, true);
          chatScroll.scrollTop = chatScroll.scrollHeight;
        });
      }
    }

    function renderStreamedContent(el, text, isStreaming) {
      let html = marked.parse(text || "");
      if (isStreaming) {
        html += `<span class="typing-cursor"></span>`;
      }
      el.innerHTML = html;
    }

    function createAssistantHolder() {
      const row = document.createElement("div");
      row.className = "msg-row assistant";

      const bubble = document.createElement("div");
      bubble.className = "bubble";

      const content = document.createElement("div");
      content.innerHTML = `
        <div class="thinking-dots">
          <span class="dot-pulse"></span>
          <span class="dot-pulse"></span>
          <span class="dot-pulse"></span>
          <span>Somi is thinking...</span>
        </div>
      `;

      const meta = document.createElement("div");
      meta.className = "telemetry-meta";

      bubble.appendChild(content);
      bubble.appendChild(meta);
      row.appendChild(bubble);
      messagesStream.appendChild(row);
      chatScroll.scrollTop = chatScroll.scrollHeight;

      return { row, content, meta };
    }

    function setButtonState(generating) {
      if (generating) {
        sendBtn.classList.add("stop");
        sendBtn.title = "Stop Generation";
        sendBtn.innerHTML = `
          <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
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

    // Init
    renderHistoryUI();
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
                "sessions": f"{HF_SPACE_URL}/v1/sessions",
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

    @spaces.GPU(duration=180)
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
            return _generate_core(prompt, 300, 0.7, 0.95)
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
