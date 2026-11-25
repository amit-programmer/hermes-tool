# model_manager.py
"""
Single-Model Dynamic Memory Manager (CPU / GGUF).
Ensures only ONE LLM resides in RAM at any given time (RAM usage capped at ~1.5GB - 2GB).
Models are cached persistently in /data/models/ to survive restarts.

Models supported:
1. 'normal' -> Qwen2.5-1.5B-Instruct-GGUF (1.0 GB)
2. 'coding' -> Qwen2.5-Coder-3B-Instruct-GGUF (1.9 GB)
3. 'logical' -> DeepSeek-R1-Distill-Qwen-1.5B-GGUF (1.0 GB)
"""
import os
import sys
import gc
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("model_manager")

# Models Directory: /data/models for persistent Space storage, else local fallback
if sys.platform != "win32" and os.path.exists("/data"):
    BASE_MODEL_DIR = "/data/models"
else:
    BASE_MODEL_DIR = os.path.abspath("./models")

try:
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)
except Exception:
    pass

# Registry of GGUF models
MODEL_REGISTRY = {
    "normal": {
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "description": "General conversational queries & fast chat"
    },
    "coding": {
        "repo_id": "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
        "filename": "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        "description": "Code generation, debugging, refactoring"
    },
    "logical": {
        "repo_id": "bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
        "filename": "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        "description": "Logical reasoning, analysis, deep step-by-step thinking"
    },
    "vision": {
        "repo_id": "vikhyatk/moondream2",
        "filename": "moondream2-q4_k_m.gguf",
        "description": "Visual document and image analysis (Moondream2 GGUF)"
    },
    "audio": {
        "repo_id": "openai/whisper-tiny",
        "filename": "whisper-tiny.bin",
        "description": "Audio voice transcription and speech-to-text (Whisper-Tiny)"
    }
}

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except Exception:
    Llama = None
    HAS_LLAMA_CPP = False

try:
    from huggingface_hub import hf_hub_download
    HAS_HF_HUB = True
except Exception:
    hf_hub_download = None
    HAS_HF_HUB = False


class ModelManager:
    """Manages dynamic loading and unloading to guarantee single-model memory footprint."""
    
    def __init__(self, model_dir: str = BASE_MODEL_DIR):
        self.model_dir = model_dir
        self.current_model_type: Optional[str] = None
        self.current_model = None
        self._lock = None  # Lazily instantiated if asyncio is used

    def get_model_path(self, model_type: str) -> str:
        """Returns local path or downloads GGUF from Hugging Face Hub if missing."""
        info = MODEL_REGISTRY.get(model_type, MODEL_REGISTRY["normal"])
        filename = info["filename"]
        target_path = os.path.join(self.model_dir, filename)

        if os.path.exists(target_path):
            return target_path

        # Case-insensitive file check in model_dir
        for existing in os.listdir(self.model_dir):
            if existing.lower() == filename.lower():
                return os.path.join(self.model_dir, existing)

        # Download from HF Hub
        if HAS_HF_HUB:
            logger.info(f"Downloading {filename} from {info['repo_id']} into {self.model_dir}...")
            token = os.getenv("HF_TOKEN")
            token_arg = token.strip() if token and token.strip() else None
            downloaded = hf_hub_download(
                repo_id=info["repo_id"],
                filename=filename,
                local_dir=self.model_dir,
                token=token_arg
            )
            return downloaded
        else:
            raise RuntimeError(f"huggingface_hub is not available to download {filename}")

    def load_model(self, model_type: str):
        """
        Loads the requested model. If a different model is already loaded,
        the old model is evicted from RAM and garbage collected first.
        """
        model_type = model_type.lower()
        if model_type not in MODEL_REGISTRY:
            model_type = "normal"

        # If already loaded and active, return cached instance immediately
        if self.current_model_type == model_type and self.current_model is not None:
            return self.current_model

        if not HAS_LLAMA_CPP:
            raise RuntimeError("llama-cpp-python is not installed. Please check requirements.")

        # 1. Unload existing model from RAM
        if self.current_model is not None:
            logger.info(f"Unloading previous model '{self.current_model_type}' from RAM...")
            del self.current_model
            self.current_model = None
            self.current_model_type = None
            gc.collect()

        # 2. Locate model path
        model_path = self.get_model_path(model_type)
        logger.info(f"Loading GGUF model '{model_type}' from {model_path} into CPU RAM...")

        # 3. Instantiate Llama model
        # Optimal CPU parameters: 4 threads, 4096 context tokens
        self.current_model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=int(os.getenv("CPU_THREADS", "4")),
            verbose=False
        )
        self.current_model_type = model_type
        logger.info(f"Model '{model_type}' successfully loaded.")
        return self.current_model

    def generate(
        self,
        prompt: str,
        model_type: str = "normal",
        max_tokens: int = 300,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list] = None
    ) -> str:
        """
        Executes text generation on the single active model.
        Returns clean string output.
        """
        model = self.load_model(model_type)
        if stop is None:
            stop = ["<|im_end|>", "<|endoftext|>", "User:", "Amit:"]

        output = model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop
        )
        return output["choices"][0]["text"].strip()


# Global Singleton
manager = ModelManager()

def get_model(model_type: str = "normal"):
    """Entrypoint function to acquire current active model."""
    return manager.load_model(model_type)

def generate_text(prompt: str, model_type: str = "normal", max_tokens: int = 300) -> str:
    """Entrypoint function to generate text directly via model manager."""
    return manager.generate(prompt=prompt, model_type=model_type, max_tokens=max_tokens)
