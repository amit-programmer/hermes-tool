import os
import time
import json
import uuid
import asyncio
import logging
from typing import List, Optional, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
from dotenv import load_dotenv

# Load local environment variables if .env exists
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("HermesGateway")

# --- Configuration ---
MODEL_ID = os.getenv("MODEL_ID", "NousResearch/Hermes-3-Llama-3.1-8B")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HF_ACCESS_TOKEN")
HF_SPACE_URL = (
    os.getenv("HF_SPACE_URL")
    or os.getenv("GATEWAY_URL")
    or "https://amitansuranjan-testing-tool.hf.space"
).rstrip("/")
HF_API_URL = os.getenv(
    "HF_INFERENCE_ENDPOINT",
    f"https://api-inference.huggingface.co/models/{MODEL_ID}"
)

# Concurrency throttling to protect HF Serverless rate limits
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENCY", "1"))
_request_semaphore: Optional[asyncio.Semaphore] = None

def get_semaphore() -> asyncio.Semaphore:
    """Lazy initialization to ensure semaphore binds to the active asyncio event loop."""
    global _request_semaphore
    if _request_semaphore is None:
        _request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    return _request_semaphore

# Retry & Timeout parameters
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
BASE_RETRY_DELAY = float(os.getenv("BASE_RETRY_DELAY", "5.0"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120.0"))

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

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Hermes-3 8B Decoupled Gateway",
    description="Lightweight OpenAI-compatible proxy orchestrating HF Serverless GPU Inference for Hermes-3 8B.",
    version="1.0.0"
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
def get_auth_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN.strip()}"
    return headers

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

# --- Core Inference Engine with Cold-Start Auto-Recovery ---
async def execute_remote_inference(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float
) -> str:
    """
    Dispatches generation request to Hugging Face Serverless GPU cluster
    with Exponential Backoff Auto-Recovery for 503 Cold-Start delays.
    """
    global active_requests_count
    headers = get_auth_headers()
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": max(temperature, 0.01),
            "top_p": top_p,
            "return_full_text": False,
            "stop": ["<|im_end|>"]
        },
        "options": {
            "wait_for_model": True,
            "use_cache": False
        }
    }

    semaphore = get_semaphore()
    async with semaphore:
        active_requests_count += 1
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        logger.info(f"Dispatching inference to HF Serverless (Attempt {attempt}/{MAX_RETRIES})...")
                        response = await client.post(HF_API_URL, headers=headers, json=payload)
                        
                        # Case 1: Successful response
                        if response.status_code == 200:
                            data = response.json()
                            if isinstance(data, list) and len(data) > 0:
                                raw_text = data[0].get("generated_text", "")
                            elif isinstance(data, dict):
                                raw_text = data.get("generated_text", data.get("text", ""))
                            else:
                                raw_text = str(data)
                            return sanitize_assistant_output(raw_text)

                        # Case 2: 503 Model Loading / Cold Start
                        if response.status_code == 503:
                            err_info = {}
                            try:
                                err_info = response.json()
                            except Exception:
                                pass
                            
                            estimated_time = err_info.get("estimated_time", BASE_RETRY_DELAY * attempt)
                            wait_duration = min(max(float(estimated_time), 4.0), 30.0)
                            
                            logger.warning(
                                f"HF Model cold-start in progress (503). "
                                f"Waiting {wait_duration:.1f}s before retry ({attempt}/{MAX_RETRIES})..."
                            )
                            if attempt < MAX_RETRIES:
                                await asyncio.sleep(wait_duration)
                                continue
                            else:
                                raise HTTPException(
                                    status_code=503,
                                    detail=f"Model loading timed out after {MAX_RETRIES} attempts. Remote status: {response.text}"
                                )

                        # Case 3: 429 Concurrency Rate Limit
                        if response.status_code == 429:
                            backoff = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                            logger.warning(f"HF Rate Limited (429). Backing off for {backoff:.1f}s...")
                            if attempt < MAX_RETRIES:
                                await asyncio.sleep(backoff)
                                continue
                            else:
                                raise HTTPException(
                                    status_code=429,
                                    detail="HF Free Tier rate limit reached. Please throttle requests."
                                )

                        # Other status codes (e.g. 401 Unauthorized, 400 Bad Request)
                        error_detail = response.text
                        logger.error(f"HF Inference API Error HTTP {response.status_code}: {error_detail}")
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"HF Inference Error: {error_detail}"
                        )

                    except (httpx.TimeoutException, httpx.NetworkError) as net_err:
                        logger.warning(f"Network error on attempt {attempt}: {net_err}")
                        if attempt < MAX_RETRIES:
                            await asyncio.sleep(BASE_RETRY_DELAY * attempt)
                        else:
                            raise HTTPException(
                                status_code=504,
                                detail=f"Inference gateway network timeout: {str(net_err)}"
                            )
        finally:
            active_requests_count -= 1

# --- Token-Level Streaming Generator (SSE) ---
async def stream_remote_inference(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    model_name: str
) -> AsyncGenerator[str, None]:
    """
    Streams generated tokens chunk-by-chunk using Server-Sent Events (SSE)
    in standard OpenAI streaming chunk format.
    """
    global active_requests_count
    headers = get_auth_headers()
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": max(temperature, 0.01),
            "top_p": top_p,
            "return_full_text": False,
            "stop": ["<|im_end|>"]
        },
        "stream": True,
        "options": {
            "wait_for_model": True,
            "use_cache": False
        }
    }

    semaphore = get_semaphore()
    async with semaphore:
        active_requests_count += 1
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                async with client.stream("POST", HF_API_URL, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        # Fallback for error in stream initialization
                        err_text = await response.aread()
                        logger.error(f"Streaming error HTTP {response.status_code}: {err_text.decode('utf-8')}")
                        error_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": f"\n[Gateway Error HTTP {response.status_code}: {err_text.decode('utf-8')}]"},
                                "finish_reason": "error"
                            }]
                        }
                        yield f"data: {json.dumps(error_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        
                        if line.startswith("data:"):
                            raw_json = line[len("data:"):].strip()
                            if raw_json == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(raw_json)
                                token_text = ""
                                
                                # Handle HF Inference API format
                                if "token" in data and isinstance(data["token"], dict):
                                    token_text = data["token"].get("text", "")
                                elif "choices" in data:
                                    token_text = data["choices"][0].get("delta", {}).get("content", "")
                                
                                # Terminate on Hermes stop token
                                if "<|im_end|>" in token_text:
                                    token_text = token_text.replace("<|im_end|>", "")
                                    if token_text:
                                        chunk = {
                                            "id": completion_id,
                                            "object": "chat.completion.chunk",
                                            "created": created_ts,
                                            "model": model_name,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {"content": token_text},
                                                "finish_reason": None
                                            }]
                                        }
                                        yield f"data: {json.dumps(chunk)}\n\n"
                                    break

                                if token_text:
                                    chunk = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created_ts,
                                        "model": model_name,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"content": token_text},
                                            "finish_reason": None
                                        }]
                                    }
                                    yield f"data: {json.dumps(chunk)}\n\n"
                            except Exception as parse_err:
                                logger.debug(f"Non-fatal stream chunk parse warning: {parse_err}")
                                continue

            # Send finish chunk
            finish_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(finish_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        finally:
            active_requests_count -= 1

# --- API Routes ---

@app.get("/health")
async def health_check():
    """
    Health & Uptime endpoint.
    Used for monitoring and Keep-Alive pings (e.g. UptimeRobot)
    to prevent HF Space idle sleep.
    """
    uptime = time.time() - START_TIME
    return {
        "status": "healthy",
        "service": "Hermes-3 8B Decoupled Gateway",
        "model": MODEL_ID,
        "uptime_seconds": round(uptime, 2),
        "active_in_flight_requests": active_requests_count,
        "concurrency_limit": MAX_CONCURRENT_REQUESTS,
        "hf_token_configured": bool(HF_TOKEN),
        "gateway_architecture": "Decoupled Brain & Gateway (FastAPI Proxy -> HF Serverless GPU Cluster)"
    }

@app.get("/v1/models")
async def list_models():
    """
    OpenAI-compatible models catalog endpoint.
    Allows standard OpenAI SDKs and IDE plugins to discover the model.
    """
    models = [
        ModelCard(id=MODEL_ID, root=MODEL_ID),
        ModelCard(id="Hermes-3-8B", root=MODEL_ID),
        ModelCard(id="NousResearch/Hermes-3-Llama-3.1-8B", root=MODEL_ID)
    ]
    return {
        "object": "list",
        "data": [m.model_dump() for m in models]
    }

@app.api_route("/v1/warmup", methods=["GET", "POST"])
async def warmup_endpoint():
    """
    Proactively pings the remote Hugging Face GPU cluster to wake up
    cold instances in the background before actual user requests arrive.
    """
    start_warmup = time.time()
    headers = get_auth_headers()
    
    # Lightweight ping payload
    ping_payload = {
        "inputs": "<|im_start|>user\nping<|im_end|>\n<|im_start|>assistant\n",
        "parameters": {"max_new_tokens": 1, "return_full_text": False},
        "options": {"wait_for_model": True, "use_cache": False}
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(HF_API_URL, headers=headers, json=ping_payload)
            latency = (time.time() - start_warmup) * 1000.0
            
            if resp.status_code == 200:
                return {
                    "status": "warm",
                    "latency_ms": round(latency, 2),
                    "model": MODEL_ID,
                    "message": "Remote GPU cluster is warm and ready for instant generation."
                }
            elif resp.status_code == 503:
                return {
                    "status": "warming_up",
                    "latency_ms": round(latency, 2),
                    "model": MODEL_ID,
                    "remote_response": resp.json() if resp.headers.get("content-type") == "application/json" else resp.text,
                    "message": "Remote GPU is currently loading into memory. Auto-recovery will manage subsequent requests."
                }
            else:
                return {
                    "status": f"http_{resp.status_code}",
                    "latency_ms": round(latency, 2),
                    "model": MODEL_ID,
                    "detail": resp.text
                }
    except Exception as e:
        return {
            "status": "warmup_error",
            "model": MODEL_ID,
            "error": str(e)
        }

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    Core OpenAI-compatible Chat Completion endpoint.
    Handles message validation, ChatML formatting, special token preservation,
    cold-start auto-recovery, and optional token-level SSE streaming.
    """
    if not req.messages:
        raise HTTPException(status_code=400, detail="The 'messages' array must not be empty.")

    # 1. Construct ChatML formatted prompt with Dynamic System Prompt injection
    prompt_text = construct_chatml_prompt(req.messages)

    # 2. Check for Token-Level Streaming Request (SSE)
    if req.stream:
        return StreamingResponse(
            stream_remote_inference(
                prompt=prompt_text,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                model_name=req.model or MODEL_ID
            ),
            media_type="text/event-stream"
        )

    # 3. Non-streaming Execution
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())
    
    start_infer = time.time()
    generated_content = await execute_remote_inference(
        prompt=prompt_text,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p
    )
    duration = time.time() - start_infer

    # Approximate token calculation for standard OpenAI usage object
    prompt_tokens = max(len(prompt_text.split()), 1)
    completion_tokens = max(len(generated_content.split()), 1)

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": req.model or MODEL_ID,
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
            "backend": "Hugging Face Serverless GPU Cluster"
        }
    }

# --- Gradio UI Dashboard & Playground ---
# For Hugging Face Spaces (sdk: gradio), mount a live web UI on the root path
try:
    import gradio as gr

    async def gradio_chat_fn(user_message, history):
        if not user_message:
            return ""
        
        # Build message history supporting both dict and tuple formats
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
            result = await execute_remote_inference(
                prompt=prompt,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.95
            )
            return result
        except Exception as e:
            return f"❌ Error: {str(e)}"

    async def gradio_warmup_fn():
        res = await warmup_endpoint()
        return json.dumps(res, indent=2)

    with gr.Blocks(title="Hermes-3 8B Gateway", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"""
            # ⚡ Hermes-3 8B Decoupled Gateway & Orchestrator
            **Model:** `{MODEL_ID}` | **Architecture:** Decoupled Brain & Gateway (FastAPI Proxy ➔ Remote Serverless GPU)
            
            This gateway acts as an **OpenAI-compatible proxy** deployed on a Free 2 vCPU Hugging Face Space.
            All heavy matrix computation is offloaded to Hugging Face's dedicated Serverless GPU clusters.
            """
        )
        
        with gr.Tab("💬 Interactive Playground"):
            chatbot = gr.ChatInterface(
                fn=gradio_chat_fn,
                title="Hermes-3 8B Chat",
                description="Test the model live. Default Lead Architect & Co-Founder persona is active."
            )
        
        with gr.Tab("🚀 Remote GPU Warmup"):
            gr.Markdown("Click below to trigger a background ping to wake up the remote Hugging Face Serverless GPU cluster.")
            warmup_btn = gr.Button("🔥 Warmup GPU Now", variant="primary")
            warmup_output = gr.Code(label="Warmup Response", language="json")
            warmup_btn.click(fn=gradio_warmup_fn, outputs=warmup_output)
            
        with gr.Tab("📖 API Integration & Endpoints"):
            gr.Markdown(
                f"""
                ### Standard OpenAI-Compatible Endpoints
                - **Chat Completions:** `POST {HF_SPACE_URL}/v1/chat/completions` (Supports both JSON and SSE streaming)
                - **Models List:** `GET {HF_SPACE_URL}/v1/models`
                - **Health / Uptime:** `GET {HF_SPACE_URL}/health` (Ideal for UptimeRobot / cron keep-alive pings)
                - **Warmup:** `GET {HF_SPACE_URL}/v1/warmup`
                
                #### Python (OpenAI SDK) Example:
                ```python
                from openai import OpenAI

                client = OpenAI(
                    base_url="{HF_SPACE_URL}/v1",
                    api_key="none"  # Authentication is handled by the space's HF_TOKEN
                )

                response = client.chat.completions.create(
                    model="NousResearch/Hermes-3-Llama-3.1-8B",
                    messages=[
                        {{"role": "user", "content": "How do we scale a distributed message queue?"}}
                    ],
                    stream=False
                )
                print(response.choices[0].message.content)
                ```

                #### Telegram Bot (httpx / python) Example:
                ```python
                import httpx

                async def ask_hermes(prompt: str):
                    async with httpx.AsyncClient() as client:
                        resp = await client.post(
                            "{HF_SPACE_URL}/v1/chat/completions",
                            json={{
                                "model": "Hermes-3-8B",
                                "messages": [{{"role": "user", "content": prompt}}],
                                "temperature": 0.7
                            }},
                            timeout=60.0
                        )
                        return resp.json()["choices"][0]["message"]["content"]
                ```
                """
            )

    # Mount Gradio app onto FastAPI
    app = gr.mount_gradio_app(app, demo, path="/")
    logger.info("Gradio UI mounted successfully on root path.")

except ImportError:
    logger.warning("Gradio is not installed in the local environment; running in pure FastAPI mode.")

    @app.get("/")
    def root_endpoint():
        return {
            "status": "online",
            "service": "Hermes-3 8B Decoupled Gateway",
            "endpoints": {
                "health": "/health",
                "models": "/v1/models",
                "chat_completions": "/v1/chat/completions",
                "warmup": "/v1/warmup",
                "docs": "/docs"
            }
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
