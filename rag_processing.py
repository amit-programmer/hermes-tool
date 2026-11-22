# rag_processing.py
"""
Multimodal Knowledge Graph and RAG-Anything Processing Interface (Optional).
Connects multi-format documents to RAG-Anything knowledge graphs when needed,
while keeping Hermes-3 8B as the core inference model.
"""
import os
from typing import Optional, Callable

try:
    from raganything import RAGAnything, RAGAnythingConfig
    HAS_RAGANYTHING = True
except ImportError:
    RAGAnything = None
    RAGAnythingConfig = None
    HAS_RAGANYTHING = False

def create_rag(llm_func: Optional[Callable] = None, embedding_func: Optional[Callable] = None):
    """
    Initializes RAG-Anything instance with ZeroGPU-optimized parameters:
    - Working directory on persistent /data volume (or ./rag_storage)
    - Image processing disabled to preserve GPU budget
    - Table processing enabled for tabular accuracy
    """
    if not HAS_RAGANYTHING or RAGAnything is None or RAGAnythingConfig is None:
        return None

    working_dir = "/data/rag_storage" if os.path.exists("/data") else "./rag_storage"
    try:
        os.makedirs(working_dir, exist_ok=True)
    except Exception:
        pass

    config = RAGAnythingConfig(
        working_dir=working_dir,
        parser="mineru",
        parse_method="auto",
        enable_image_processing=False,  # Kept OFF to minimize ZeroGPU overhead
        enable_table_processing=True,
        enable_equation_processing=False,
    )
    return RAGAnything(config=config, llm_model_func=llm_func, embedding_func=embedding_func)
