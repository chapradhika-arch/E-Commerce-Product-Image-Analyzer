"""Orchestrates caption -> RAG retrieval -> listing generation."""
from __future__ import annotations

import logging

from app.schemas import ProductAnalysis, RagSource
from app.services.captioning import get_captioning_service
from app.services.generator import get_generator_service
from app.services.rag import get_rag_service

logger = logging.getLogger(__name__)


def analyze_image(filename: str, image_bytes: bytes) -> ProductAnalysis:
    """Run the full analysis pipeline for one image, never raising."""
    try:
        caption, caption_is_mock = get_captioning_service().caption(image_bytes)
        rag_entries = get_rag_service().retrieve(caption)
        listing = get_generator_service().generate(caption, rag_entries)

        sources = [
            RagSource(
                category=e["category"],
                snippet=e.get("snippet", ""),
                score=e.get("score", 0.0),
            )
            for e in rag_entries
        ]

        return ProductAnalysis(
            filename=filename,
            caption=caption,
            title=listing["title"],
            description=listing["description"],
            tags=listing["tags"],
            category=listing["category"],
            rag_sources=sources,
            mock=caption_is_mock,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Analysis failed for %s", filename)
        return ProductAnalysis(
            filename=filename,
            caption="",
            title=filename,
            description="",
            tags=[],
            category="Unknown",
            mock=True,
            error=str(exc),
        )
