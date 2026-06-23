"""Retrieval-Augmented Generation over a product-copy knowledge base.

Embeds knowledge-base entries with a sentence-transformer and retrieves the
nearest entries to an image caption via cosine similarity. When embeddings are
unavailable it falls back to transparent keyword-overlap scoring, so retrieval
always returns useful category context.
"""
from __future__ import annotations

import json
import logging
import re

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-zA-Z]+")


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


class RagService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.entries: list[dict] = []
        self._embedder = None
        self._embeddings: np.ndarray | None = None
        self._load_attempted = False
        self.available = False
        self._load_knowledge_base()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _load_knowledge_base(self) -> None:
        try:
            with open(self.settings.knowledge_base_path, encoding="utf-8") as fh:
                self.entries = json.load(fh)
            logger.info("Loaded %d knowledge-base entries.", len(self.entries))
        except Exception as exc:
            logger.error("Failed to load knowledge base: %s", exc)
            self.entries = []

    def _try_load_embedder(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True

        if self.settings.mock_mode or not self.entries:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedder %s ...", self.settings.embed_model)
            self._embedder = SentenceTransformer(
                self.settings.embed_model, device=self.settings.resolve_device()
            )
            corpus = [self._entry_text(e) for e in self.entries]
            self._embeddings = self._embedder.encode(
                corpus, normalize_embeddings=True, convert_to_numpy=True
            )
            self.available = True
            logger.info("RAG embeddings ready.")
        except Exception as exc:  # pragma: no cover - depends on env
            logger.warning("Embedder unavailable, using keyword RAG: %s", exc)
            self.available = False

    @staticmethod
    def _entry_text(entry: dict) -> str:
        return f"{entry.get('category', '')}. {entry.get('snippet', '')}"

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return the top-k knowledge-base entries with a ``score`` field."""
        if not self.entries:
            return []
        top_k = top_k or self.settings.rag_top_k

        self._try_load_embedder()
        if self.available and self._embeddings is not None:
            base = self._semantic_scores(query)
        else:
            base = np.array(self._keyword_scores(query), dtype=float)

        # Blend in explicit keyword hits so a clearly-named object (e.g. "sari",
        # "headphones") anchors its category instead of letting the caption
        # embedding drift toward a loosely-related snippet. Each entry may set
        # its own boost: garment categories anchor harder than accessories,
        # because a fashion model wears jewelry incidentally while the garment
        # is the actual product being listed.
        kw_hits = np.array(self._keyword_hit_counts(query), dtype=float)
        boosts = np.array(
            [float(e.get("boost", 0.2)) for e in self.entries], dtype=float
        )
        scores = base + boosts * kw_hits

        ranked = sorted(
            zip(self.entries, scores), key=lambda p: p[1], reverse=True
        )
        results = []
        for entry, score in ranked[:top_k]:
            item = dict(entry)
            item["score"] = round(float(score), 4)
            results.append(item)
        return results

    def _semantic_scores(self, query: str) -> np.ndarray:
        q = self._embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )
        return (self._embeddings @ q[0])  # cosine sim (vectors are normalized)

    def _keyword_hit_counts(self, query: str) -> list[float]:
        """Number of an entry's category keywords present in the query."""
        q_tokens = _tokenize(query)
        return [
            float(len(q_tokens & {k.lower() for k in entry.get("keywords", [])}))
            for entry in self.entries
        ]

    def _keyword_scores(self, query: str) -> list[float]:
        q_tokens = _tokenize(query)
        scores: list[float] = []
        for entry in self.entries:
            kws = {k.lower() for k in entry.get("keywords", [])}
            entry_tokens = kws | _tokenize(entry.get("snippet", ""))
            if not entry_tokens:
                scores.append(0.05)  # general-merchandise catch-all baseline
                continue
            # Weight explicit keyword hits heavily; they map an object to a category.
            keyword_hits = len(q_tokens & kws)
            overlap = len(q_tokens & entry_tokens)
            scores.append(keyword_hits * 1.0 + overlap * 0.1)
        return scores


_service: RagService | None = None


def get_rag_service() -> RagService:
    global _service
    if _service is None:
        _service = RagService()
    return _service
