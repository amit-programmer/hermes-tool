# chat_memory.py
"""
Dedicated conversational short-term memory (RAM-based, <5KB, <0.001s).
Maintains a strict 12-message rolling buffer per session.
Independent from disk document storage to eliminate monolithic coupling.
"""
import time
import threading
from typing import List, Dict, Optional, Any

def estimate_tokens(text: str) -> int:
    """Fast estimation heuristic (~3.8 chars per token)."""
    if not text:
        return 0
    return max(1, int(len(text) / 3.8))

class ChatMessage:
    """Lightweight single message in conversation buffer."""
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

class ChatMemory:
    """
    Thread-safe 12-message rolling conversation buffer.
    Fast memory-only storage (0.001s latency).
    """
    def __init__(self, session_id: str, max_messages: int = 12):
        self.session_id: str = session_id
        self.max_messages: int = max_messages
        self.messages: List[ChatMessage] = []
        self._lock = threading.RLock()
        self.updated_at: float = time.time()

    def add_message(self, role: str, content: str, name: Optional[str] = None) -> ChatMessage:
        """Appends a new turn and automatically evicts oldest turns if > 12."""
        with self._lock:
            self.updated_at = time.time()
            msg = ChatMessage(role=role, content=content, name=name)
            self.messages.append(msg)
            while len(self.messages) > self.max_messages:
                self.messages.pop(0)
            return msg

    def get_history(self) -> List[ChatMessage]:
        """Returns snapshot of current rolling conversation buffer."""
        with self._lock:
            return list(self.messages)

    def clear(self) -> int:
        """Empties conversational history."""
        with self._lock:
            count = len(self.messages)
            self.messages.clear()
            self.updated_at = time.time()
            return count

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return sum(m.token_count for m in self.messages)
