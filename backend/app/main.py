"""FastAPI application entrypoint."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import get_settings
from app.routers import analyze
from app.schemas import HealthResponse
from app.services.captioning import get_captioning_service
from app.services.generator import get_generator_service
from app.services.rag import get_rag_service

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up models once at startup.

    Loading eagerly here (in a clean, sequential import order) avoids the
    lazy-import race during the first request that intermittently triggered an
    `accelerate` circular-import. It also makes the first analyze fast. Each
    step is best-effort so a failure degrades to the mock fallback instead of
    crashing the server.
    """
    if not settings.mock_mode:
        logger.info("Warming up models (caption -> embedder -> generator)...")
        for name, warm in (
            ("caption", lambda: get_captioning_service()._try_load()),
            ("embedder", lambda: get_rag_service()._try_load_embedder()),
            ("generator", lambda: get_generator_service()._try_load()),
        ):
            try:
                warm()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Warmup for %s failed; using fallback.", name)
        logger.info("Model warmup complete.")
    yield


app = FastAPI(
    title="E-Commerce Product Image Analyzer",
    description=(
        "Auto-generate product titles, descriptions and tags from images using "
        "BLIP captioning, RAG retrieval and GPT-style text generation."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)


@app.get("/api/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    cap = get_captioning_service()
    gen = get_generator_service()
    rag = get_rag_service()
    # Touch lazy loaders so the report reflects what actually loaded.
    cap._try_load()
    gen._try_load()
    rag._try_load_embedder()

    real_models = cap.available or gen.available or rag.available
    return HealthResponse(
        status="ok",
        mock_mode=settings.mock_mode or not real_models,
        device=settings.resolve_device(),
        models={
            "caption": settings.caption_model if cap.available else "mock",
            "text": settings.text_model if gen.available else "mock",
            "embed": settings.embed_model if rag.available else "keyword-fallback",
        },
        knowledge_base_entries=len(rag.entries),
    )


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "name": "E-Commerce Product Image Analyzer API",
        "version": __version__,
        "docs": "/docs",
        "health": "/api/health",
    }
