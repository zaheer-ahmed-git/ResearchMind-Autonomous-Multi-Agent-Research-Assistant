"""Adaptive ranking to surface trustworthy evidence and cut hallucinations."""

from __future__ import annotations

from typing import Any


def _token_set(text: str) -> set[str]:
    return {t for t in text.lower().replace("?", " ").split() if len(t) > 2}


def rank_evidence(
    evidence: list[dict[str, Any]],
    *,
    query: str,
    analysis_keywords: list[str] | None = None,
    recency_bias: float = 0.15,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score and re-order evidence using query overlap, source trust, and analysis hints."""
    if not evidence:
        return [], {"method": "adaptive", "count": 0}

    query_tokens = _token_set(query)
    hint_tokens = _token_set(" ".join(analysis_keywords or []))
    trusted = {"arxiv", "nature", "science", "nist", "who", "docs", "official", "wikipedia"}

    ranked: list[dict[str, Any]] = []
    for idx, row in enumerate(evidence):
        content = str(row.get("content", ""))
        title = str(row.get("title", ""))
        source = str(row.get("source", "")).lower()
        tokens = _token_set(f"{title} {content}")
        overlap = len(tokens & query_tokens) / max(1, len(query_tokens))
        hint_boost = 0.12 * (len(tokens & hint_tokens) / max(1, len(hint_tokens) or 1))
        trust = 0.2 if any(tag in source for tag in trusted) else 0.05
        base = float(row.get("score", 0.4))
        # Mild diversity penalty for near-duplicate titles
        dup_penalty = 0.0
        for prev in ranked:
            if str(prev.get("title", "")).lower() == title.lower():
                dup_penalty = 0.25
                break
        score = min(1.0, base * 0.45 + overlap * 0.35 + trust + hint_boost + recency_bias * (1.0 / (1 + idx)) - dup_penalty)
        updated = dict(row)
        updated["score"] = round(score, 4)
        ranked.append(updated)

    ranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank

    meta = {
        "method": "adaptive",
        "count": len(ranked),
        "top_score": ranked[0]["score"] if ranked else 0.0,
        "recency_bias": recency_bias,
    }
    return ranked, meta
