import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SessionRecord:
    session_id: str
    updated_at: float
    history: List[dict] = field(default_factory=list)


class InMemoryChatSessionStore:
    """
    Lightweight in-memory chat session store.
    - TTL based eviction
    - bounded history length
    """

    def __init__(self, ttl_seconds: int = 1800, max_turns: int = 12):
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns
        self._lock = threading.Lock()
        self._sessions: Dict[str, SessionRecord] = {}

    def get_or_create_session_id(self, session_id: Optional[str] = None) -> str:
        with self._lock:
            self._cleanup_expired_locked()
            sid = (session_id or "").strip() or str(uuid.uuid4())
            if sid not in self._sessions:
                self._sessions[sid] = SessionRecord(
                    session_id=sid,
                    updated_at=time.time(),
                    history=[],
                )
            return sid

    def get_history(self, session_id: str) -> List[dict]:
        with self._lock:
            self._cleanup_expired_locked()
            rec = self._sessions.get(session_id)
            if not rec:
                return []
            rec.updated_at = time.time()
            return list(rec.history)

    def append_turn(self, session_id: str, role: str, content: str) -> None:
        if not content:
            return
        with self._lock:
            self._cleanup_expired_locked()
            rec = self._sessions.get(session_id)
            if not rec:
                rec = SessionRecord(session_id=session_id, updated_at=time.time(), history=[])
                self._sessions[session_id] = rec
            rec.history.append({"role": role, "content": content, "timestamp": time.time()})
            rec.history = rec.history[-self.max_turns :]
            rec.updated_at = time.time()

    def _cleanup_expired_locked(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, rec in self._sessions.items()
            if (now - rec.updated_at) > self.ttl_seconds
        ]
        for sid in expired:
            self._sessions.pop(sid, None)


chat_session_store = InMemoryChatSessionStore()
