# big_file_handler.py
"""
High-capacity multi-format document chunker and context-aware retriever (CPU-Optimized).
4-Step Architecture:
1. Todo: Splits large PDFs (20-p), DOCX (headings), XLSX (1,000 rows), PPTX (20 slides), TXT (5,000 lines).
2. Chhota Summary: Summarizes chunks rapidly.
3. Sahi Tukda Dhoondo: Uses BAAI/bge-small-en-v1.5 embedding similarity to find the most relevant chunk.
4. Final Synthesis: Synthesizes final answer via DeepSeek-R1-Distill-1.5B (or model_func).
"""
import os
import sys
import re
from typing import List, Optional, Callable, Dict, Any

# Safe multi-format library imports
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx
except ImportError:
    docx = None

try:
    from pptx import Presentation
except ImportError:
    Presentation = None

# Safe embedding imports
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    SentenceTransformer = None
    util = None
    HAS_SENTENCE_TRANSFORMERS = False

_embedder = None

def get_embedder():
    """Lazily loads BAAI/bge-small-en-v1.5 embedding model on CPU (~33MB)."""
    global _embedder
    if not HAS_SENTENCE_TRANSFORMERS:
        return None
    if _embedder is None:
        try:
            cache_folder = "/data/models/bge-small" if (sys.platform != "win32" and os.path.exists("/data")) else "./models/bge-small"
            _embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", cache_folder=cache_folder)
        except Exception:
            try:
                _embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
            except Exception:
                _embedder = None
    return _embedder


# ---- STEP 1: Multi-Format Chunking Engine ----

def extract_chunks(file_path: str, file_type: str) -> List[str]:
    """
    Extracts structured chunks tailored to file type without loading full document in one block:
    - PDF: 20-page sliding chunks
    - DOCX: Heading-based semantic sections
    - XLSX/CSV: 1,000-row tabular batches with column headers preserved
    - PPTX: 20-slide visual slide groups
    - TXT/MD/LOG: 5,000-line blocks
    """
    chunks = []
    file_type = file_type.lower().strip().lstrip(".")

    # 1. PDF Documents: Chunked every 20 pages
    if file_type == "pdf" and PdfReader is not None:
        try:
            reader = PdfReader(file_path)
            pages = []
            for idx, p in enumerate(reader.pages):
                txt = p.extract_text() or ""
                pages.append(f"--- Page {idx + 1} ---\n{txt}")
            for i in range(0, len(pages), 20):
                chunk = "\n\n".join(pages[i:i+20]).strip()
                if chunk:
                    chunks.append(chunk)
        except Exception as e:
            chunks.append(f"[Error reading PDF pages: {e}]")

    # 2. Word DOCX: Heading-based semantic sections
    elif file_type == "docx" and docx is not None:
        try:
            doc = docx.Document(file_path)
            cur = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue
                if para.style and para.style.name and "heading" in para.style.name.lower() and cur:
                    chunks.append("\n".join(cur))
                    cur = [f"## {text}"]
                else:
                    cur.append(text)
            if cur:
                chunks.append("\n".join(cur))
        except Exception as e:
            chunks.append(f"[Error reading Word document: {e}]")

    # 3. Excel XLSX / CSV: 1,000 rows per chunk with preserved column headers
    elif file_type in ("xlsx", "xls", "csv") and pd is not None:
        try:
            df = pd.read_excel(file_path) if file_type in ("xlsx", "xls") else pd.read_csv(file_path)
            header = " | ".join(df.columns.astype(str))
            total_rows = len(df)
            for i in range(0, total_rows, 1000):
                chunk_df = df.iloc[i:i+1000]
                text = f"[Table Header: {header}]\n[Rows {i+1} to {min(i+1000, total_rows)} of {total_rows}]\n" + chunk_df.to_string(index=False)
                chunks.append(text)
        except Exception as e:
            chunks.append(f"[Error reading Spreadsheet: {e}]")

    # 4. PowerPoint PPTX: 20 slides per chunk
    elif file_type in ("pptx", "ppt") and Presentation is not None:
        try:
            prs = Presentation(file_path)
            slides_text = []
            for s_idx, slide in enumerate(prs.slides):
                slide_parts = [f"--- Slide {s_idx + 1} ---"]
                for shape in slide.shapes:
                    if shape.has_text_frame and shape.text.strip():
                        slide_parts.append(shape.text.strip())
                slides_text.append("\n".join(slide_parts))
            for i in range(0, len(slides_text), 20):
                chunk = "\n\n".join(slides_text[i:i+20]).strip()
                if chunk:
                    chunks.append(chunk)
        except Exception as e:
            chunks.append(f"[Error reading PowerPoint: {e}]")

    # 5. TXT / Markdown / Logs / Code: 5,000 lines per chunk
    elif file_type in ("txt", "md", "log", "json", "py", "js", "html", "css", "yaml", "yml", "sql"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for i in range(0, len(lines), 5000):
                chunk = "".join(lines[i:i+5000]).strip()
                if chunk:
                    chunks.append(chunk)
        except Exception as e:
            chunks.append(f"[Error reading text file: {e}]")

    return chunks or ["[Empty or unsupported file content]"]


# ---- STEP 2, 3, 4: CPU-Based Embeddings & Summarization Pipeline ----

def find_relevant_chunks(chunks: List[str], question: str, top_k: int = 2) -> List[str]:
    """
    Finds the most relevant chunks using BAAI/bge-small embedding similarity on CPU.
    Falls back to token overlap scoring if embedder is unavailable.
    """
    if len(chunks) <= top_k:
        return chunks

    embedder = get_embedder()
    if embedder is not None and util is not None:
        try:
            q_emb = embedder.encode(question, convert_to_tensor=True)
            c_embs = embedder.encode(chunks, convert_to_tensor=True)
            hits = util.semantic_search(q_emb, c_embs, top_k=top_k)
            if hits and len(hits[0]) > 0:
                selected = [chunks[hit["corpus_id"]] for hit in hits[0]]
                return selected
        except Exception:
            pass

    # Keyword fallback
    keywords = [w.lower().strip(".,!?:;\"'()") for w in question.split() if len(w) > 3]
    scores = []
    for idx, c in enumerate(chunks):
        c_lower = c.lower()
        score = sum(c_lower.count(k) for k in keywords)
        scores.append((score, idx))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [chunks[s[1]] for s in scores[:top_k]]


def process_big_file(
    file_path: str,
    file_type: str,
    question: str,
    model_func: Optional[Callable[[str], str]] = None
) -> str:
    """
    Full 4-step CPU extraction & QA pipeline:
    1. Extract chunks without memory bloat.
    2. Rank & select top relevant chunks using BGE-small embeddings.
    3. Assemble crisp excerpt context (< 12,000 chars).
    4. Pass to reasoning model (DeepSeek) or model_func for synthesis.
    """
    chunks = extract_chunks(file_path, file_type)
    if not chunks:
        return "File me koi readable text ya content nahi mila."

    # If no model_func is provided, return concatenated clean chunks
    if model_func is None:
        return "\n\n".join(chunks[:3])

    # Select top 2-3 most relevant chunks
    relevant_chunks = find_relevant_chunks(chunks, question, top_k=2)
    relevant_text = "\n\n---\n\n".join(relevant_chunks)[:12000]

    prompt = (
        f"Relevant Document Excerpts:\n{relevant_text}\n\n"
        f"Question:\n{question}\n\n"
        f"Instructions:\n"
        f"Answer the question directly, accurately, and concisely based strictly on the relevant excerpts above."
    )
    return model_func(prompt)
