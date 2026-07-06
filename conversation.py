from collections import defaultdict
import time
from typing import Dict, List

class ConversationMemory:
    def __init__(self, max_history: int = 10, ttl: int = 3600):
        self.sessions: Dict[str, dict] = defaultdict(lambda: {
            "history": [],       # [{"role": "user", "content": "..."}, ...]
            "last_active": time.time()
        })
        self.max_history = max_history
        self.ttl = ttl  # 会话过期时间（秒）

    def add_message(self, session_id: str, role: str, content: str):
        session = self.sessions[session_id]
        session["history"].append({"role": role, "content": content})
        session["last_active"] = time.time()
        # 保留最近 max_history*2 条（用户+助手各算一条）
        if len(session["history"]) > self.max_history * 2:
            session["history"] = session["history"][-self.max_history*2:]

    def get_history(self, session_id: str) -> List[dict]:
        self._clean_expired()
        return self.sessions.get(session_id, {}).get("history", [])

    def _clean_expired(self):
        now = time.time()
        expired = [sid for sid, s in self.sessions.items() if now - s["last_active"] > self.ttl]
        for sid in expired:
            del self.sessions[sid]