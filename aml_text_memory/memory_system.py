import sqlite3
import json
import hashlib
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

# 北京时间时区
BEIJING_TZ = timezone(timedelta(hours=8))

class MemorySystem:
    def __init__(self, db_path="memory.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                session_id TEXT,
                timestamp TEXT,
                content TEXT,
                keywords TEXT
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS profile (
                user_id TEXT PRIMARY KEY,
                profile_summary TEXT,
                updated_at TEXT
            )
        ''')
        self.conn.commit()
        self.corpus = []
        self.metadata = []
        self.bm25 = None
        self._rebuild_index()

    def _rebuild_index(self):
        self.cursor.execute("SELECT id, content, user_id, timestamp FROM facts")
        rows = self.cursor.fetchall()
        self.corpus = [row[1] for row in rows]
        self.metadata = [{"id": row[0], "user_id": row[2], "timestamp": row[3]} for row in rows]
        if self.corpus:
            tokenized_corpus = [doc.split() for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    def add_memory(self, user_id: str, session_id: str, content: str, timestamp: str = None):
        if timestamp is None:
            timestamp = datetime.now(BEIJING_TZ).isoformat()
        keywords = " ".join(content.split()[:20])
        unique_id = hashlib.md5(f"{user_id}{session_id}{content}{timestamp}".encode()).hexdigest()
        self.cursor.execute(
            "INSERT OR REPLACE INTO facts (id, user_id, session_id, timestamp, content, keywords) VALUES (?, ?, ?, ?, ?, ?)",
            (unique_id, user_id, session_id, timestamp, content, keywords)
        )
        self.conn.commit()
        self._update_profile(user_id, content)
        self._rebuild_index()
        return unique_id

    def _update_profile(self, user_id: str, content: str):
        profile_parts = []
        if "喜欢" in content or "偏好" in content:
            profile_parts.append("偏好明确")
        if "我是" in content or "职业" in content:
            profile_parts.append("身份明确")
        if len(content) > 50:
            profile_parts.append("长文本理解力强")
        if profile_parts:
            summary = "；".join(profile_parts)
            self.cursor.execute(
                "INSERT OR REPLACE INTO profile (user_id, profile_summary, updated_at) VALUES (?, ?, ?)",
                (user_id, summary, datetime.now(BEIJING_TZ).isoformat())
            )
            self.conn.commit()

    def search_memory(self, user_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results = []
        if self.bm25 and self.corpus:
            tokenized_query = query.split()
            scores = self.bm25.get_scores(tokenized_query)
            indexed_scores = list(enumerate(scores))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            for idx, score in indexed_scores[:10]:
                if score > 0:
                    meta = self.metadata[idx]
                    results.append({
                        "id": meta["id"],
                        "content": self.corpus[idx],
                        "timestamp": meta["timestamp"],
                        "score": float(score)
                    })
        if not results:
            self.cursor.execute(
                "SELECT id, content, timestamp FROM facts WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                (user_id, top_k)
            )
            fallbacks = self.cursor.fetchall()
            for row in fallbacks:
                results.append({
                    "id": row[0],
                    "content": row[1],
                    "timestamp": row[2],
                    "score": 0.0
                })
        self.cursor.execute("SELECT profile_summary FROM profile WHERE user_id = ?", (user_id,))
        profile_row = self.cursor.fetchone()
        if profile_row and profile_row[0]:
            profile_context = f"[用户画像]: {profile_row[0]}"
            results.insert(0, {
                "id": "profile_meta",
                "content": profile_context,
                "timestamp": datetime.now(BEIJING_TZ).isoformat(),
                "score": 999.0
            })
        seen_ids = set()
        final_results = []
        for r in results:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                final_results.append(r)
            if len(final_results) >= top_k:
                break
        return final_results
