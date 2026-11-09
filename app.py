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

try:
    import pytesseract
except ImportError:
    pytesseract = None


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
                    await asyncio.sleep(0)

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

# --- File Extraction & Multipart Chat Endpoints (Zero Vector DB) ---
def extract_content_from_file(filename: str, file_bytes: bytes) -> dict:
    """
    Extracts text, data tables, or OCR content from uploaded files without Vector DB.
    Directly extracts and formats PDF, Word, Excel, PPTX, CSV, TXT, Code, and Images.
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
        "error": None
    }

    try:
        # 1. Plain text & Code files
        text_extensions = {
            ".txt", ".md", ".py", ".js", ".jsx", ".ts", ".tsx", ".html",
            ".css", ".json", ".yaml", ".yml", ".xml", ".sql", ".sh",
            ".bat", ".env", ".c", ".cpp", ".h", ".java", ".go", ".rs",
            ".php", ".rb", ".ini", ".conf", ".log"
        }
        if ext in text_extensions:
            try:
                text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = file_bytes.decode("latin-1", errors="replace")
            result["extracted_text"] = text
            result["summary"] = f"Text/code file ({len(text)} characters, {len(text.splitlines())} lines)"
            return result

        # 2. PDF Files (.pdf)
        if ext == ".pdf":
            if pypdf is not None:
                try:
                    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                    num_pages = len(reader.pages)
                    pages_content = []
                    for p_idx, page in enumerate(reader.pages):
                        p_text = page.extract_text() or ""
                        if p_text.strip():
                            pages_content.append(f"--- [Page {p_idx + 1}/{num_pages}] ---\n{p_text.strip()}")
                    combined_text = "\n\n".join(pages_content)
                    result["extracted_text"] = combined_text
                    result["summary"] = f"PDF document with {num_pages} page(s) ({len(combined_text.split())} words)"
                    return result
                except Exception as pe:
                    result["extracted_text"] = f"[Error reading PDF {filename}: {str(pe)}]"
                    result["error"] = str(pe)
                    return result
            else:
                result["extracted_text"] = f"[PDF file: {filename} uploaded, but 'pypdf' library is not available in environment]"
                result["summary"] = "PDF document (parser pending install)"
                result["error"] = "pypdf library not found"
                return result

        # 3. Word Documents (.docx, .doc)
        if ext in (".docx", ".doc"):
            if docx is not None:
                try:
                    doc = docx.Document(io.BytesIO(file_bytes))
                    doc_parts = []
                    for p in doc.paragraphs:
                        if p.text.strip():
                            doc_parts.append(p.text.strip())
                    for t_idx, table in enumerate(doc.tables):
                        table_rows = []
                        for row in table.rows:
                            row_cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                            table_rows.append("| " + " | ".join(row_cells) + " |")
                        if table_rows:
                            if len(table_rows) > 1:
                                col_count = len(table.rows[0].cells)
                                table_rows.insert(1, "| " + " | ".join(["---"] * col_count) + " |")
                            doc_parts.append(f"\n[Table {t_idx + 1}]\n" + "\n".join(table_rows))
                    
                    combined_text = "\n\n".join(doc_parts)
                    result["extracted_text"] = combined_text
                    result["summary"] = f"Word document ({len(doc.paragraphs)} paragraphs, {len(doc.tables)} tables)"
                    return result
                except Exception as de:
                    result["extracted_text"] = f"[Error reading Word document {filename}: {str(de)}]"
                    result["error"] = str(de)
                    return result
            else:
                result["extracted_text"] = f"[Word file: {filename} uploaded, but 'python-docx' is not available]"
                result["summary"] = "Word document (parser pending install)"
                result["error"] = "python-docx library not found"
                return result

        # 4. Excel & Spreadsheets (.xlsx, .xls, .csv, .tsv)
        if ext in (".xlsx", ".xls", ".csv", ".tsv"):
            try:
                if pd is not None:
                    if ext == ".csv":
                        df = pd.read_csv(io.BytesIO(file_bytes))
                        sheets = {"CSV_Data": df}
                    elif ext == ".tsv":
                        df = pd.read_csv(io.BytesIO(file_bytes), sep="\t")
                        sheets = {"TSV_Data": df}
                    else:
                        sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)

                    sheet_texts = []
                    for s_name, sheet_df in sheets.items():
                        total_rows, total_cols = sheet_df.shape
                        preview_df = sheet_df.head(100)
                        try:
                            md_table = preview_df.to_markdown(index=False)
                        except Exception:
                            md_table = preview_df.to_string(index=False)
                        desc = f"### Sheet: {s_name} (Total Rows: {total_rows}, Columns: {total_cols})\n"
                        desc += f"Columns: {', '.join(str(c) for c in sheet_df.columns)}\n\n"
                        desc += md_table
                        if total_rows > 100:
                            desc += f"\n\n*(Showing preview of first 100 rows out of {total_rows} total rows)*"
                        sheet_texts.append(desc)

                    combined_text = "\n\n---\n\n".join(sheet_texts)
                    result["extracted_text"] = combined_text
                    result["summary"] = f"Spreadsheet with {len(sheets)} sheet(s)"
                    return result
                elif ext in (".csv", ".tsv"):
                    text = file_bytes.decode("utf-8", errors="replace")
                    result["extracted_text"] = text
                    result["summary"] = f"CSV/TSV table ({len(text.splitlines())} lines)"
                    return result
                else:
                    result["extracted_text"] = f"[Excel file: {filename} uploaded, but 'pandas/openpyxl' is not available]"
                    result["summary"] = "Excel spreadsheet"
                    result["error"] = "pandas/openpyxl library not found"
                    return result
            except Exception as xe:
                result["extracted_text"] = f"[Error reading spreadsheet {filename}: {str(xe)}]"
                result["error"] = str(xe)
                return result

        # 5. PowerPoint Presentations (.pptx)
        if ext == ".pptx":
            if Presentation is not None:
                try:
                    prs = Presentation(io.BytesIO(file_bytes))
                    slide_texts = []
                    for s_idx, slide in enumerate(prs.slides):
                        texts = []
                        for shape in slide.shapes:
                            if shape.has_text_frame:
                                for p in shape.text_frame.paragraphs:
                                    if p.text.strip():
                                        texts.append(p.text.strip())
                        if texts:
                            slide_texts.append(f"--- [Slide {s_idx + 1}] ---\n" + "\n".join(texts))
                    combined_text = "\n\n".join(slide_texts)
                    result["extracted_text"] = combined_text
                    result["summary"] = f"PowerPoint presentation with {len(prs.slides)} slide(s)"
                    return result
                except Exception as ppe:
                    result["extracted_text"] = f"[Error reading PowerPoint {filename}: {str(ppe)}]"
                    result["error"] = str(ppe)
                    return result
            else:
                result["extracted_text"] = f"[PowerPoint file: {filename} uploaded, but 'python-pptx' is not available]"
                result["summary"] = "PowerPoint presentation"
                result["error"] = "python-pptx library not found"
                return result

        # 6. Images (.png, .jpg, .jpeg, .webp, .bmp, .gif)
        if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"):
            img_desc = f"[Attached Image File: {filename} (Size: {size_kb} KB)]"
            if Image is not None:
                try:
                    img = Image.open(io.BytesIO(file_bytes))
                    img_desc += f"\nImage Dimensions: {img.width}x{img.height}px | Format: {img.format} | Color Mode: {img.mode}"
                    
                    ocr_text = ""
                    if pytesseract is not None:
                        try:
                            ocr_text = pytesseract.image_to_string(img).strip()
                        except Exception:
                            ocr_text = ""
                    
                    if ocr_text:
                        img_desc += f"\n\n[Extracted OCR Text from Image]:\n{ocr_text}"
                        result["summary"] = f"Image ({img.width}x{img.height}px) with OCR text ({len(ocr_text.split())} words)"
                    else:
                        img_desc += "\n\n(No textual content detected via OCR; image metadata extracted.)"
                        result["summary"] = f"Image ({img.width}x{img.height}px)"
                    
                    result["extracted_text"] = img_desc
                    return result
                except Exception as ie:
                    result["extracted_text"] = f"[Image file: {filename} ({size_kb} KB)]"
                    result["summary"] = f"Image file ({size_kb} KB)"
                    return result
            else:
                result["extracted_text"] = f"[Image file: {filename} ({size_kb} KB)]"
                result["summary"] = f"Image file ({size_kb} KB)"
                return result

        # 7. Audio / Video files
        if ext in (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".mkv", ".webm", ".avi", ".mov"):
            media_type = "Audio" if ext in (".mp3", ".wav", ".m4a", ".ogg", ".flac") else "Video"
            desc = (
                f"[{media_type} File Attached: {filename} (Format: {ext.upper()[1:]}, Size: {size_kb} KB)]\n"
                f"Direct {media_type.lower()} file metadata parsed without vector database. "
                f"You may ask questions, transcription guidance, or processing instructions regarding this media."
            )
            result["extracted_text"] = desc
            result["summary"] = f"{media_type} file ({size_kb} KB)"
            return result

        # 8. Generic binary fallback
        try:
            raw_text = file_bytes.decode("utf-8", errors="ignore")
            if len(raw_text.strip()) > 20:
                result["extracted_text"] = raw_text
                result["summary"] = f"Decoded file content ({len(raw_text)} chars)"
                return result
        except Exception:
            pass

        result["extracted_text"] = f"[File Attached: {filename} (Size: {size_kb} KB, Type: {ext})]"
        result["summary"] = f"Attached file ({size_kb} KB)"
        return result

    except Exception as e:
        logger.error(f"Error parsing file {filename}: {e}")
        result["success"] = False
        result["error"] = str(e)
        result["extracted_text"] = f"[Error reading file {filename}: {str(e)}]"
        result["summary"] = f"File error: {str(e)}"
        return result

@app.post("/v1/files/extract")
async def extract_file_endpoint(file: UploadFile = File(...)):
    """
    Extracts text, data tables, or OCR content from uploaded files without Vector DB.
    Supports PDF, Word, Excel, PPTX, CSV, TXT, Code, Images, etc.
    """
    try:
        content_bytes = await file.read()
        res = extract_content_from_file(file.filename or "unknown_file", content_bytes)
        return JSONResponse(status_code=200, content=res)
    except Exception as e:
        logger.error(f"Error in /v1/files/extract: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e), "extracted_text": ""})

@app.post("/v1/chat/upload-and-complete")
async def chat_upload_and_complete(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    max_tokens: Optional[int] = Form(1024),
    temperature: Optional[float] = Form(0.7)
):
    """
    Convenience multipart endpoint to upload a file + question together in one request.
    Extracts file content, constructs the prompt, and returns the completion.
    """
    if model is None:
        try:
            load_local_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model is not loaded: {str(e)}")

    content_bytes = await file.read()
    extracted = extract_content_from_file(file.filename or "file", content_bytes)
    
    file_context = extracted.get("extracted_text", "")
    user_prompt = (prompt or "Please analyze and summarize this attached file.").strip()
    
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
        max_tokens=max_tokens or 1024,
        temperature=temperature or 0.7,
        top_p=0.95
    )
    
    return {
        "status": "success",
        "file": {
            "filename": extracted["filename"],
            "size_kb": extracted["size_kb"],
            "summary": extracted["summary"]
        },
        "query": user_prompt,
        "completion": generated
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
  <title>GenAI | Hermes-3 8B Architecture</title>
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
            <span>⚡ GenAI</span>
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
          <div class="profile-name">Hermes-3 8B</div>
          <div class="profile-badge">Nvidia A100 ZeroGPU</div>
        </div>
      </div>
    </aside>

    <!-- Main Chat Workspace -->
    <main class="main-chat">

      <!-- Top Header for Mobile & Tablet -->
      <header class="mobile-header">
        <button class="mobile-nav-toggle" id="openDrawerBtn" title="Open History">☰</button>
        <div class="mobile-brand">
          <span>GenAI Hermes-3</span>
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
          <h1 class="hero-title">Start writing with GenAI</h1>

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
    const themeToggleBtn = document.getElementById("themeToggleBtn");
    const mobileThemeToggleBtn = document.getElementById("mobileThemeToggleBtn");
    const lightThemeOption = document.getElementById("lightThemeOption");
    const darkThemeOption = document.getElementById("darkThemeOption");

    // --- Theme Switcher Logic ---
    function applyTheme(theme) {
      const isDark = theme === "dark";
      if (isDark) {
        document.body.classList.add("dark-theme");
        if (themeToggleBtn) {
          themeToggleBtn.innerHTML = "☀️";
          themeToggleBtn.title = "Switch to Light Mode";
        }
        if (mobileThemeToggleBtn) {
          mobileThemeToggleBtn.innerHTML = "☀️";
          mobileThemeToggleBtn.title = "Switch to Light Mode";
        }
        if (darkThemeOption) darkThemeOption.classList.add("active");
        if (lightThemeOption) lightThemeOption.classList.remove("active");
      } else {
        document.body.classList.remove("dark-theme");
        if (themeToggleBtn) {
          themeToggleBtn.innerHTML = "🌙";
          themeToggleBtn.title = "Switch to Dark Mode";
        }
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

    if (themeToggleBtn) themeToggleBtn.addEventListener("click", toggleTheme);
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
        const demoTitles = [
          "Stress & Anxiety Everytime",
          "Not Having Enough Money",
          "Mental Health",
          "Food Industry case study",
          "Team Management Plan",
          "Form validation code"
        ];
        demoTitles.forEach(title => {
          const item = document.createElement("div");
          item.className = "history-item";
          item.innerHTML = `<span class="chat-bubble-icon">💬</span><span>${title}</span>`;
          item.onclick = () => {
            closeSidebar();
            triggerPrompt("Tell me more about " + title);
          };
          historyList.appendChild(item);
        });
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
            const extRes = await fetch("/v1/files/extract", {
              method: "POST",
              body: formData,
              signal: abortController.signal
            });
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
              <span>Hermes-3 is analyzing document...</span>
            </div>
          `;
        }

        // Build messages payload for Hermes-3
        const apiMessages = conversation.map((msg, idx) => {
          if (idx === conversation.length - 1 && msg.role === "user") {
            return { role: "user", content: injectedUserContent };
          }
          return { role: msg.role, content: msg.content };
        });

        const payload = {
          model: "NousResearch/Hermes-3-Llama-3.1-8B",
          messages: [
            { role: "system", content: "You are a Lead Co-Founder & Senior Technical Architect. You possess world-class expertise in distributed systems, full-stack software engineering, pragmatic AI engineering, and comprehensive document analysis. When analyzing attached documents, spreadsheets, code, or images, provide precise, structured, and insightful answers." },
            ...apiMessages
          ],
          temperature: 0.7,
          max_tokens: 1536,
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
                  renderStreamedContent(contentEl, fullResponse, true);
                  chatScroll.scrollTop = chatScroll.scrollHeight;
                }
              } catch (err) {}
            }
          }
        }

        renderStreamedContent(contentEl, fullResponse, false);
        conversation.push({ role: "assistant", content: fullResponse });

        sess.messages = conversation;
        saveSessions(sessions);

        const totalDuration = ((performance.now() - startTime) / 1000).toFixed(2);
        const tokPerSec = tokenCount > 0 ? (tokenCount / Math.max(totalDuration, 0.1)).toFixed(1) : 0;
        metaEl.textContent = `⚡ ${totalDuration}s • ${tokPerSec} tok/s • ZeroGPU A100`;

      } catch (err) {
        if (err.name === 'AbortError') {
          renderStreamedContent(contentEl, fullResponse, false);
          metaEl.textContent = "⚠️ Stopped by user.";
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
      let html = marked.parse(text);
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
          <span>GenAI is writing...</span>
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
