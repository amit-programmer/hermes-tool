# pinned_docs.py
"""
Dedicated working document storage for session state.
Persists heavy document content to disk (/data/pinned_docs/ or ./pinned_docs/)
instead of keeping hundreds of kilobytes of text in RAM.
"""
import os
import sys
import uuid
import time
import logging
import threading
from typing import Dict, List, Optional, Any

logger = logging.getLogger("PinnedDocs")

# Storage directory on persistent volume if available
if sys.platform != "win32" and os.path.exists("/data"):
    DOCS_DIR = "/data/pinned_docs"
else:
    DOCS_DIR = os.path.abspath("./pinned_docs")

try:
    os.makedirs(DOCS_DIR, exist_ok=True)
except Exception:
    pass


class PinnedDocRef:
    """Document reference holding metadata in RAM while text resides on disk."""
    def __init__(
        self,
        doc_id: str,
        filename: str,
        disk_path: str,
        summary: str = "",
        size_kb: float = 0.0,
        token_count: int = 0
    ):
        self.doc_id: str = doc_id
        self.filename: str = filename
        self.disk_path: str = disk_path
        self.summary: str = summary or f"Extracted file ({size_kb:.1f} KB)"
        self.size_kb: float = size_kb
        self.token_count: int = token_count
        self.pinned_at: float = time.time()

    @property
    def extracted_text(self) -> str:
        """Transparent backward compatibility for code accessing extracted_text."""
        return self.read_text()

    @property
    def extension(self) -> str:
        """Transparent backward compatibility for code accessing extension."""
        return self.filename.split(".")[-1].lower() if "." in self.filename else ""

    def read_text(self) -> str:
        """Reads document text from disk on demand."""
        if os.path.exists(self.disk_path):
            try:
                with open(self.disk_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading pinned doc from disk: {e}")
        return ""

    def to_dict(self, include_full_text: bool = False) -> Dict[str, Any]:
        data = {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "size_kb": self.size_kb,
            "summary": self.summary,
            "pinned_at": self.pinned_at,
            "token_count": self.token_count,
            "disk_path": self.disk_path
        }
        if include_full_text:
            data["extracted_text"] = self.read_text()
        return data


class PinnedDocs:
    """
    Decoupled document manager for a session.
    Keeps RAM footprint at 0KB by offloading file content to disk.
    """
    def __init__(self, session_id: str):
        self.session_id: str = session_id
        self.files: Dict[str, PinnedDocRef] = {}  # doc_id -> PinnedDocRef
        self._lock = threading.RLock()
        self.session_docs_dir = os.path.join(DOCS_DIR, session_id)
        try:
            os.makedirs(self.session_docs_dir, exist_ok=True)
        except Exception:
            pass

    def pin_file(
        self,
        filename: str,
        file_path: Optional[str] = None,
        extracted_text: Optional[str] = None,
        summary: str = "",
        size_kb: float = 0.0,
        doc_id: Optional[str] = None
    ) -> PinnedDocRef:
        """Pins a document by writing text to disk and keeping a reference."""
        with self._lock:
            if not doc_id:
                for existing_id, existing_ref in list(self.files.items()):
                    if existing_ref.filename == filename:
                        doc_id = existing_id
                        break
            if not doc_id:
                doc_id = f"doc_{uuid.uuid4().hex[:8]}"

            disk_file = os.path.join(self.session_docs_dir, f"{doc_id}.txt")
            text_to_save = extracted_text or ""
            if not text_to_save and file_path and os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as sf:
                        text_to_save = sf.read()
                except Exception:
                    pass

            try:
                with open(disk_file, "w", encoding="utf-8") as df:
                    df.write(text_to_save)
            except Exception as e:
                logger.error(f"Failed to persist doc to disk: {e}")

            token_count = max(1, int(len(text_to_save) / 3.8)) if text_to_save else 0
            ref = PinnedDocRef(
                doc_id=doc_id,
                filename=filename,
                disk_path=disk_file,
                summary=summary,
                size_kb=size_kb,
                token_count=token_count
            )
            self.files[doc_id] = ref
            logger.info(f"[PinnedDocs] Pinned '{filename}' ({token_count} est tokens) on disk for session '{self.session_id}'.")
            return ref

    def remove_document(self, doc_id_or_name: str) -> bool:
        """Unpins a document and cleans up disk file."""
        with self._lock:
            target_id = None
            if doc_id_or_name in self.files:
                target_id = doc_id_or_name
            else:
                for d_id, ref in self.files.items():
                    if ref.filename == doc_id_or_name:
                        target_id = d_id
                        break

            if target_id and target_id in self.files:
                ref = self.files.pop(target_id)
                if os.path.exists(ref.disk_path):
                    try:
                        os.remove(ref.disk_path)
                    except Exception:
                        pass
                return True
            return False

    def clear(self) -> int:
        """Removes all pinned documents for this session and frees disk space."""
        with self._lock:
            count = len(self.files)
            for ref in self.files.values():
                if os.path.exists(ref.disk_path):
                    try:
                        os.remove(ref.disk_path)
                    except Exception:
                        pass
            self.files.clear()
            return count

    def get_context_block(self, max_chars: int = 12000) -> str:
        """Assembles a compact context block of pinned documents for prompt injection."""
        with self._lock:
            if not self.files:
                return ""
            parts = ["[WORKING MEMORY: PINNED DOCUMENTS]"]
            for ref in self.files.values():
                text = ref.read_text()
                preview = text[:max_chars] if text else "[Empty Document]"
                parts.append(f"=== [DOCUMENT: {ref.filename} ({ref.summary})] ===\n{preview}")
            return "\n\n".join(parts)
