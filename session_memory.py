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

from chat_memory import ChatMemory, ChatMessage, estimate_tokens
from pinned_docs import PinnedDocs, PinnedDocRef

logger = logging.getLogger("HermesMemoryStore")

# Type aliases for 100% backward compatibility
SessionMessage = ChatMessage
PinnedDocument = PinnedDocRef


class SessionState:
    """
    Decoupled state container for a single session.
    Delegates conversation buffer to ChatMemory (RAM, <5KB, rolling 12/30 turns).
    Delegates heavy document working memory to PinnedDocs (disk-persisted, 0KB RAM).
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
        self.max_turns: int = max_turns
        self.max_token_budget: int = max_token_budget
        self.ttl_seconds: float = ttl_seconds

        # Decoupled components:
        self.chat: ChatMemory = ChatMemory(session_id=session_id, max_messages=max_turns)
        self.docs: PinnedDocs = PinnedDocs(session_id=session_id)
        self._lock: threading.RLock = threading.RLock()

    @property
    def messages(self) -> List[ChatMessage]:
        return self.chat.get_history()

    @property
    def pinned_documents(self) -> Dict[str, PinnedDocRef]:
        return self.docs.files

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
    ) -> ChatMessage:
        """Add a message to rolling buffer via ChatMemory."""
        with self._lock:
            self.touch()
            return self.chat.add_message(role=role, content=content, name=name)

    def pin_document(
        self,
        filename: str,
        extracted_text: str,
        summary: str = "",
        size_kb: float = 0.0,
        doc_id: Optional[str] = None
    ) -> PinnedDocRef:
        """Pins a document to disk working memory via PinnedDocs."""
        with self._lock:
            self.touch()
            return self.docs.pin_file(
                filename=filename,
                extracted_text=extracted_text,
                summary=summary,
                size_kb=size_kb,
                doc_id=doc_id
            )

    def remove_document(self, doc_id_or_name: str) -> bool:
        """Unpins a document and cleans up disk file."""
        with self._lock:
            self.touch()
            return self.docs.remove_document(doc_id_or_name)

    def clear_documents(self) -> int:
        """Removes all pinned documents from disk working memory."""
        with self._lock:
            self.touch()
            return self.docs.clear()

    def clear_messages(self) -> int:
        """Clears conversation messages."""
        with self._lock:
            self.touch()
            return self.chat.clear()

    def get_pinned_context_block(self, max_doc_tokens: int = 65536) -> str:
        """Returns structured prompt context block from PinnedDocs."""
        with self._lock:
            max_chars = int(max_doc_tokens * 3.8)
            return self.docs.get_context_block(max_chars=max_chars)

    def to_dict(self, include_doc_text: bool = False) -> Dict[str, Any]:
        """Serializes session state for the API endpoint."""
        with self._lock:
            now = time.time()
            ttl_remaining = max(0.0, self.ttl_seconds - (now - self.updated_at))
            history = self.chat.get_history()
            docs = list(self.docs.files.values())
            msg_tokens = self.chat.total_tokens
            doc_tokens = sum(d.token_count for d in docs)

            return {
                "session_id": self.session_id,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "ttl_seconds_remaining": round(ttl_remaining, 1),
                "is_expired": self.is_expired(now),
                "message_count": len(history),
                "pinned_document_count": len(docs),
                "total_estimated_tokens": msg_tokens + doc_tokens,
                "messages": [m.to_dict() for m in history],
                "pinned_documents": [
                    d.to_dict(include_full_text=include_doc_text)
                    for d in docs
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

