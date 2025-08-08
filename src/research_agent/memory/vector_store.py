"""Local vector memory for research, analysis, and synthesis documents.

The original module targeted Pinecone. This store keeps the same document
shape (content + source, doc_type, task_id, tags, confidence) and ranks
neighbors with token overlap so the pipeline runs without an external index.
"""

from __future__ import annotations

import re
import uuid
from typing import Any


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]{3,}", text.lower())}


class VectorStore:
    """In-process vector store shared by the research pipeline."""

    def __init__(self) -> None:
        self._docs: list[dict[str, Any]] = []

    def add(
        self,
        content: str,
        *,
        source: str,
        doc_type: str = "research",
        task_id: str | None = None,
        tags: list[str] | None = None,
        confidence: float = 1.0,
        author: str | None = None,
    ) -> str:
        doc_id = str(uuid.uuid4())[:8]
        self._docs.append(
            {
                "id": doc_id,
                "content": content,
                "source": source,
                "doc_type": doc_type,
                "task_id": task_id,
                "author": author,
                "tags": list(tags or []),
                "confidence": float(confidence),
            }
        )
        return doc_id

    def add_evidence(self, evidence: list[dict[str, Any]], *, task_id: str | None) -> None:
        for item in evidence:
            self.add(
                f"{item.get('title', '')}. {item.get('content', '')}",
                source=str(item.get("source", "research")),
                doc_type="research",
                task_id=task_id,
                tags=list(item.get("tags") or []),
                confidence=float(item.get("score") or 0.5),
                author="research_agent",
            )

    def similarity_search(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        query_tokens = _tokens(query)
        if not query_tokens or not self._docs:
            return []
        scored: list[dict[str, Any]] = []
        for doc in self._docs:
            doc_tokens = _tokens(f"{doc['content']} {' '.join(doc['tags'])}")
            overlap = len(query_tokens & doc_tokens) / max(1, len(query_tokens))
            score = round(min(1.0, 0.75 * overlap + 0.25 * float(doc["confidence"])), 4)
            if score <= 0:
                continue
            hit = dict(doc)
            hit["score"] = score
            scored.append(hit)
        scored.sort(key=lambda item: float(item["score"]), reverse=True)
        return scored[:k]

    def to_dicts(self) -> list[dict[str, Any]]:
        return [dict(doc) for doc in self._docs]

    @classmethod
    def from_dicts(cls, rows: list[dict[str, Any]]) -> VectorStore:
        store = cls()
        for row in rows:
            store._docs.append(dict(row))
        return store

    def __len__(self) -> int:
        return len(self._docs)
