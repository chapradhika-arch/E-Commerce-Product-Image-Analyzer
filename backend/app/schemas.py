"""Pydantic request/response models for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class RagSource(BaseModel):
    """A knowledge-base entry retrieved by the RAG step."""

    category: str
    snippet: str
    score: float = Field(..., description="Cosine similarity to the image caption.")


class ProductAnalysis(BaseModel):
    """Generated listing for a single product image."""

    filename: str
    caption: str = Field(..., description="Raw image caption from the vision model.")
    title: str
    description: str
    tags: list[str]
    category: str
    rag_sources: list[RagSource] = []
    mock: bool = Field(
        False, description="True when produced by the deterministic fallback."
    )
    error: str | None = None


class BulkAnalysisResponse(BaseModel):
    count: int
    results: list[ProductAnalysis]


class HealthResponse(BaseModel):
    status: str
    mock_mode: bool
    device: str
    models: dict[str, str]
    knowledge_base_entries: int
