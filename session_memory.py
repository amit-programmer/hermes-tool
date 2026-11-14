"""
Enterprise-Grade In-Memory Short-Term Memory (STM) & Document Working Memory Store.
Zero-Vector-DB Architecture for Hermes-3 8B Local Server.

Features:
- Thread-safe rolling conversation buffers with configurable turn and token limits.
- Pinned document working memory per session (persists across turns in the session).
- Automatic TTL and idle session cleanup.
- Context injection for follow-up questions without re-parsing files.
"""

import time
import uuid
import logging
import threading
from typing import List, Dict, Optional, Any

logger = logging.getLogger("HermesMemoryStore")


def estimate_tokens(text: str) -> int:
    """Fast, safe token estimation (~3.8 characters per token for English & code)."""
    if not text:
        return 0
    return max(1, int(len(text) / 3.8))


class SessionMessage:
    """Represents a single message in a session's rolling buffer."""

    def __init__(self, role: str, content: str, name: Optional[str] = None):
        self.role: str = (role or "user").strip().lower()
        self.content: str = content or ""
        self.name: Optional[str] = name
        self.timestamp: float = time.time()
        self.token_count: int = estimate_tokens(self.content)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "token_count": self.token_count,
        }
        if self.name:
            data["name"] = self.name
        return data


class PinnedDocument:
    """Represents an extracted document pinned into a session's working memory."""

    def __init__(
        self,
        doc_id: str,
        filename: str,
        extracted_text: str,
        summary: str = "",
        size_kb: float = 0.0,
        extension: str = ""
    ):
        self.doc_id: str = doc_id
        self.filename: str = filename
        self.extracted_text: str = extracted_text or ""
        self.summary: str = summary or f"Extracted file ({size_kb:.1f} KB)"
        self.size_kb: float = size_kb
        self.extension: str = extension or (filename.split(".")[-1].lower() if "." in filename else "")
        self.pinned_at: float = time.time()
        self.token_count: int = estimate_tokens(self.extracted_text)

    def to_dict(self, include_full_text: bool = False) -> Dict[str, Any]:
        result = {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "extension": self.extension,
            "size_kb": self.size_kb,
            "summary": self.summary,
            "pinned_at": self.pinned_at,
            "token_count": self.token_count,
            "text_preview": (
                self.extracted_text[:300] + "..."
                if len(self.extracted_text) > 300
                else self.extracted_text
            )
        }
        if include_full_text:
            result["extracted_text"] = self.extracted_text
        return result


class SessionState:
    """
    Thread-safe state container for a single session.
    Maintains:
      - Rolling conversation buffer (pruned by turn count and token budget).
      - Pinned documents (working memory available to all prompts in this session).
      - Activity timestamps and TTL expiration tracking.
    """

    def __init__(
        self,
        session_id: str,
        max_turns: int = 30,
        max_token_budget: int = 32768,
        ttl_seconds: float = 3600.0
    ):
        self.session_id: str = session_id
        self.created_at: float = time.time()
        self.updated_at: float = self.created_at
        self.max_turns: int = max_turns  # Max total messages in buffer
        self.max_token_budget: int = max_token_budget
        self.ttl_seconds: float = ttl_seconds

        self.messages: List[SessionMessage] = []
        self.pinned_documents: Dict[str, PinnedDocument] = {}  # doc_id -> PinnedDocument
        self._lock: threading.RLock = threading.RLock()

    def touch(self) -> None:
        """Update last active timestamp."""
        self.updated_at = time.time()

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Check if this session has exceeded its idle TTL."""
        current = now or time.time()
        return (current - self.updated_at) > self.ttl_seconds

    def add_message(
        self,
        role: str,
        content: str,
        name: Optional[str] = None
    ) -> SessionMessage:
        """Add a message to the rolling buffer with automatic pruning."""
        with self._lock:
            self.touch()
            msg = SessionMessage(role=role, content=content, name=name)
            self.messages.append(msg)
            self._prune_rolling_buffer()
            return msg

    def pin_document(
        self,
        filename: str,
        extracted_text: str,
        summary: str = "",
        size_kb: float = 0.0,
        doc_id: Optional[str] = None
    ) -> PinnedDocument:
        """
        Pins or replaces a document in this session's working memory.
        If a document with the same filename already exists, updates it.
        """
        with self._lock:
            self.touch()
            # If doc_id not provided, check if filename already pinned
            if not doc_id:
                for existing_id, existing_doc in list(self.pinned_documents.items()):
                    if existing_doc.filename == filename:
                        doc_id = existing_id
                        break
            if not doc_id:
                doc_id = f"doc_{uuid.uuid4().hex[:8]}"

            doc = PinnedDocument(
                doc_id=doc_id,
                filename=filename,
                extracted_text=extracted_text,
                summary=summary,
                size_kb=size_kb
            )
            self.pinned_documents[doc_id] = doc
            logger.info(
                f"[SessionMemory] Pinned document '{filename}' ({doc.token_count} est tokens) "
                f"to session '{self.session_id}' (Total docs: {len(self.pinned_documents)})"
            )
            return doc

    def remove_document(self, doc_id_or_name: str) -> bool:
        """Unpins a document by doc_id or filename."""
        with self._lock:
            self.touch()
            if doc_id_or_name in self.pinned_documents:
                del self.pinned_documents[doc_id_or_name]
                return True
            for d_id, doc in list(self.pinned_documents.items()):
                if doc.filename == doc_id_or_name:
                    del self.pinned_documents[d_id]
                    return True
            return False

    def clear_documents(self) -> int:
        """Removes all pinned documents from working memory."""
        with self._lock:
            self.touch()
            count = len(self.pinned_documents)
            self.pinned_documents.clear()
            return count

    def clear_messages(self) -> int:
        """Clears conversation messages in this session."""
        with self._lock:
            self.touch()
            count = len(self.messages)
            self.messages.clear()
            return count

    def _prune_rolling_buffer(self) -> None:
        """
        Enforces turn count and token budget limits.
        Retains initial system message (if present) and drops oldest user/assistant pairs.
        """
        if not self.messages:
            return

        # 1. Turn count limit check
        while len(self.messages) > self.max_turns:
            # If the first message is system, keep it and remove the 2nd message
            if self.messages[0].role == "system" and len(self.messages) > 1:
                self.messages.pop(1)
            else:
                self.messages.pop(0)

        # 2. Token budget check
        current_tokens = sum(m.token_count for m in self.messages)
        while current_tokens > self.max_token_budget and len(self.messages) > 1:
            if self.messages[0].role == "system" and len(self.messages) > 1:
                removed = self.messages.pop(1)
            else:
                removed = self.messages.pop(0)
            current_tokens -= removed.token_count

    def get_pinned_context_block(self, max_doc_tokens: int = 65536) -> str:
        """
        Constructs the structured working memory context block for pinned documents.
        This is injected into Hermes-3 prompt for follow-up reasoning.
        """
        with self._lock:
            if not self.pinned_documents:
                return ""

            parts = [
                f"[WORKING MEMORY: PINNED DOCUMENTS IN SESSION '{self.session_id}']",
                "The following document context is pinned in short-term memory for this session.",
                "You can answer follow-up questions, analyze details, or reference sections directly:\n"
            ]

            total_tokens = 0
            for doc in self.pinned_documents.values():
                header = f"=== [DOCUMENT: {doc.filename} | Summary: {doc.summary}] ==="
                footer = f"=== [END DOCUMENT: {doc.filename}] ===\n"
                
                doc_text = doc.extracted_text
                # Budget check per document
                est_tok = doc.token_count
                if total_tokens + est_tok > max_doc_tokens:
                    allowed_chars = int((max_doc_tokens - total_tokens) * 3.8)
                    if allowed_chars > 200:
                        doc_text = doc_text[:allowed_chars] + "\n[... Document truncated to preserve context budget ...]"
                    else:
                        break

                parts.append(f"{header}\n{doc_text}\n{footer}")
                total_tokens += estimate_tokens(doc_text)

            return "\n".join(parts)

    def to_dict(self, include_doc_text: bool = False) -> Dict[str, Any]:
        """Serializes session state for the API endpoint."""
        with self._lock:
            now = time.time()
            ttl_remaining = max(0.0, self.ttl_seconds - (now - self.updated_at))
            msg_tokens = sum(m.token_count for m in self.messages)
            doc_tokens = sum(d.token_count for d in self.pinned_documents.values())

            return {
                "session_id": self.session_id,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "ttl_seconds_remaining": round(ttl_remaining, 1),
                "is_expired": self.is_expired(now),
                "message_count": len(self.messages),
                "pinned_document_count": len(self.pinned_documents),
                "total_estimated_tokens": msg_tokens + doc_tokens,
                "messages": [m.to_dict() for m in self.messages],
                "pinned_documents": [
                    d.to_dict(include_full_text=include_doc_text)
                    for d in self.pinned_documents.values()
                ]
            }


class SessionMemoryStore:
    """
    Enterprise-Grade In-Memory Session Store with thread safety,
    rolling conversation buffers, pinned document working memory, and auto TTL.
    """

    def __init__(
        self,
        default_max_turns: int = 30,
        default_max_token_budget: int = 32768,
        default_ttl_seconds: float = 3600.0,
        cleanup_interval_seconds: float = 300.0
    ):
        self._sessions: Dict[str, SessionState] = {}
        self._lock: threading.RLock = threading.RLock()
        self.default_max_turns: int = default_max_turns
        self.default_max_token_budget: int = default_max_token_budget
        self.default_ttl_seconds: float = default_ttl_seconds
        self.cleanup_interval_seconds: float = cleanup_interval_seconds
        self._last_cleanup: float = time.time()

    def get_or_create(
        self,
        session_id: Optional[str] = None,
        max_turns: Optional[int] = None,
        max_token_budget: Optional[int] = None,
        ttl_seconds: Optional[float] = None
    ) -> SessionState:
        """Retrieves an existing session or creates a new one thread-safely."""
        self._maybe_auto_cleanup()

        sid = (session_id or "").strip()
        if not sid:
            sid = f"sess_{uuid.uuid4().hex[:12]}"

        with self._lock:
            session = self._sessions.get(sid)
            if session is not None:
                if session.is_expired():
                    logger.info(f"[SessionMemory] Session '{sid}' expired. Reinitializing.")
                    del self._sessions[sid]
                    session = None
                else:
                    session.touch()
                    return session

            # Create new session
            new_session = SessionState(
                session_id=sid,
                max_turns=max_turns or self.default_max_turns,
                max_token_budget=max_token_budget or self.default_max_token_budget,
                ttl_seconds=ttl_seconds or self.default_ttl_seconds
            )
            self._sessions[sid] = new_session
            logger.info(f"[SessionMemory] Created new session '{sid}' (Total active: {len(self._sessions)})")
            return new_session

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Retrieves an existing session without auto-creating."""
        self._maybe_auto_cleanup()
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.is_expired():
                del self._sessions[session_id]
                return None
            session.touch()
            return session

    def delete_session(self, session_id: str) -> bool:
        """Deletes a session completely from memory."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"[SessionMemory] Deleted session '{session_id}'")
                return True
            return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Lists metadata summaries of all active sessions."""
        self._maybe_auto_cleanup()
        with self._lock:
            summaries = []
            now = time.time()
            for sid, sess in list(self._sessions.items()):
                if sess.is_expired(now):
                    continue
                summaries.append({
                    "session_id": sid,
                    "created_at": sess.created_at,
                    "updated_at": sess.updated_at,
                    "ttl_seconds_remaining": round(max(0.0, sess.ttl_seconds - (now - sess.updated_at)), 1),
                    "message_count": len(sess.messages),
                    "pinned_document_count": len(sess.pinned_documents)
                })
            return summaries

    def cleanup_expired(self) -> int:
        """Scans and evicts all expired sessions."""
        with self._lock:
            now = time.time()
            expired_keys = [sid for sid, sess in self._sessions.items() if sess.is_expired(now)]
            for sid in expired_keys:
                del self._sessions[sid]
            self._last_cleanup = now
            if expired_keys:
                logger.info(f"[SessionMemory] Cleaned up {len(expired_keys)} expired sessions. (Active: {len(self._sessions)})")
            return len(expired_keys)

    def _maybe_auto_cleanup(self) -> None:
        """Triggers passive cleanup if interval has elapsed."""
        if time.time() - self._last_cleanup > self.cleanup_interval_seconds:
            threading.Thread(target=self.cleanup_expired, daemon=True).start()


# Global Singleton Instance
session_memory_store = SessionMemoryStore()

