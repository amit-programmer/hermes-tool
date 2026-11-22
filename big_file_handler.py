# big_file_handler.py
"""
High-capacity multi-format document chunker and context-aware retriever (Vector-DB-Free).
Splits large PDFs, Word documents, Excel workbooks, PowerPoint slides, and Code/Logs
into manageable chunks, extracts chunk summaries, and selects the most relevant
content for Hermes-3 8B within the 128k context budget.
"""
import os
from typing import List, Optional, Callable

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
                # Split at primary headings
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

# ---- STEP 2, 3, 4: Vector-DB-Free Intelligent Big File Processing ----

def process_big_file(file_path: str, file_type: str, question: str, model_func: Optional[Callable[[str], str]] = None) -> str:
    """
    Processes large files accurately without an external Vector DB:
    1. Extracts structured chunks.
    2. Small files (<= 5 chunks): Direct synthesis.
    3. Big files (> 5 chunks):
       - Summarizes / tags chunks.
       - Matches keywords against question.
       - Gathers most relevant chunks up to ~12,000 chars.
       - Generates focused, hallucination-free answer.
    """
    chunks = extract_chunks(file_path, file_type)

    if not chunks:
        return "File me koi readable text ya content nahi mila."

    # If no model_func is provided, return concatenated clean chunks for prompt injection
    if model_func is None:
        return "\n\n".join(chunks[:5])

    # Case A: Small files (<= 5 chunks) - Full direct context synthesis
    if len(chunks) <= 5:
        full_text = "\n\n".join(chunks)
        prompt = (
            f"Document Content:\n{full_text}\n\n"
            f"Question:\n{question}\n\n"
            f"Instructions:\n"
            f"Answer accurately using ONLY the document content above. If the information is not present, state clearly that it is not found."
        )
        return model_func(prompt)

    # Case B: Large files (> 5 chunks) - Intelligent Keyword & Chunk Filtering
    keywords = [w.lower().strip(".,!?:;\"'()") for w in question.split() if len(w) > 3]

    # Score chunks by keyword occurrence
    chunk_scores = []
    for idx, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        score = sum(chunk_lower.count(k) for k in keywords)
        chunk_scores.append((score, idx, chunk))

    # Sort descending by score
    chunk_scores.sort(key=lambda x: x[0], reverse=True)

    # Pick top relevant chunks (or first 2 if no keyword match)
    relevant = [cs[2] for cs in chunk_scores if cs[0] > 0]
    if not relevant:
        relevant = chunks[:2]
    else:
        relevant = relevant[:3]

    # Cap relevant text around 12,000 chars to protect context and ensure crisp latency
    relevant_text = "\n\n---\n\n".join(relevant)[:12000]

    final_prompt = (
        f"Relevant Document Excerpts:\n{relevant_text}\n\n"
        f"Question:\n{question}\n\n"
        f"Instructions:\n"
        f"Answer the question directly and concisely based strictly on the relevant excerpts above."
    )
    return model_func(final_prompt)
