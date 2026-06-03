"""GPT-style listing generation from a caption + retrieved RAG context.

Uses a FLAN-T5 instruction model when available; otherwise assembles a clean,
deterministic listing from the caption and knowledge-base snippets. Titles and
tags are always derived deterministically so output stays well-formed.
"""
from __future__ import annotations

import logging
import re

from app.config import get_settings

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "a", "an", "the", "of", "on", "in", "with", "and", "or", "for", "to", "at",
    "is", "are", "this", "that", "it", "its", "as", "by", "from", "photographed",
    "background", "product", "image", "photo", "picture", "front", "view", "some",
}
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]+")


def _keywords(text: str, limit: int = 6) -> list[str]:
    seen: list[str] = []
    for w in _WORD_RE.findall(text.lower()):
        if w in _STOPWORDS or len(w) < 3 or w in seen:
            continue
        seen.append(w)
        if len(seen) >= limit:
            break
    return seen


class GeneratorService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._tokenizer = None
        self._model = None
        self._device = self.settings.resolve_device()
        self._load_attempted = False
        self.available = False

    def _try_load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        if self.settings.mock_mode:
            return
        try:
            # Load the seq2seq model directly. The legacy "text2text-generation"
            # pipeline task was removed in transformers v5, so we drive the model
            # with tokenizer + .generate() which works across versions.
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            logger.info("Loading text model %s ...", self.settings.text_model)
            self._tokenizer = AutoTokenizer.from_pretrained(self.settings.text_model)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.settings.text_model
            ).to(self._device)
            self._model.eval()
            self.available = True
            logger.info("Text generation model ready on %s.", self._device)
        except Exception as exc:  # pragma: no cover - depends on env
            logger.warning("Text model unavailable, using template generator: %s", exc)
            self.available = False

    # ------------------------------------------------------------------ #
    def generate(self, caption: str, rag_entries: list[dict]) -> dict:
        category = rag_entries[0]["category"] if rag_entries else "General Merchandise"
        kws = _keywords(caption)

        title = self._make_title(caption, kws, category)
        tags = self._make_tags(kws, rag_entries)
        description = self._make_description(caption, category, rag_entries, kws)

        return {
            "title": title,
            "description": description,
            "tags": tags,
            "category": category,
        }

    # ------------------------------------------------------------------ #
    def _make_title(self, caption: str, kws: list[str], category: str) -> str:
        core = " ".join(kws[:4]) if kws else caption
        title = core.strip().title()
        if not title:
            title = category
        # Keep it punchy and listing-friendly.
        return title[:70]

    def _make_tags(self, kws: list[str], rag_entries: list[dict]) -> list[str]:
        tags: list[str] = []
        for kw in kws:
            tags.append(kw)
        for entry in rag_entries[:2]:
            for t in entry.get("tag_pool", []):
                if t not in tags:
                    tags.append(t)
        # De-dup while preserving order, cap the list.
        out: list[str] = []
        for t in tags:
            t = t.strip().lower()
            if t and t not in out:
                out.append(t)
        return out[:12]

    def _make_description(
        self, caption: str, category: str, rag_entries: list[dict], kws: list[str]
    ) -> str:
        context = " ".join(e.get("snippet", "") for e in rag_entries[:2]).strip()

        self._try_load()
        if self.available:
            prompt = (
                "Write a compelling, concise e-commerce product description "
                "(2-3 sentences) for an online store listing.\n"
                f"Product category: {category}\n"
                f"Image caption: {caption}\n"
                f"Reference style and selling points: {context}\n"
                "Description:"
            )
            try:
                import torch

                inputs = self._tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=512
                ).to(self._device)
                with torch.no_grad():
                    out_ids = self._model.generate(
                        **inputs, max_new_tokens=120, do_sample=False, num_beams=4
                    )
                text = self._tokenizer.decode(
                    out_ids[0], skip_special_tokens=True
                ).strip()
                if len(text) > 40:  # guard against degenerate short outputs
                    return text
            except Exception as exc:  # pragma: no cover - depends on env
                logger.warning("Text generation failed, using template: %s", exc)

        return self._template_description(caption, category, context, kws)

    @staticmethod
    def _template_description(
        caption: str, category: str, context: str, kws: list[str]
    ) -> str:
        subject = ", ".join(kws[:3]) if kws else "product"
        lead = (
            f"Discover this standout {category.lower()} — {caption.strip().rstrip('.')}"
            "."
        )
        highlight = (
            f" Featuring {subject}, it is designed to impress." if kws else ""
        )
        return f"{lead}{highlight} {context}".strip()


_service: GeneratorService | None = None


def get_generator_service() -> GeneratorService:
    global _service
    if _service is None:
        _service = GeneratorService()
    return _service
