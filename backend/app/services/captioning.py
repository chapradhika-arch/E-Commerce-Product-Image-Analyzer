"""Image captioning via Salesforce BLIP, with a deterministic mock fallback."""
from __future__ import annotations

import logging
from io import BytesIO

from PIL import Image

from app.config import get_settings

logger = logging.getLogger(__name__)


class CaptioningService:
    """Wraps the BLIP image-captioning model.

    Loading is lazy and best-effort: if torch/transformers are missing, the
    model fails to download, or MOCK_MODE is on, we fall back to a heuristic
    caption derived from the image so the rest of the pipeline keeps working.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._processor = None
        self._model = None
        self._device = self.settings.resolve_device()
        self._load_attempted = False
        self.available = False

    def _try_load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True

        if self.settings.mock_mode:
            logger.info("MOCK_MODE on — skipping BLIP model load.")
            return

        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor

            logger.info("Loading caption model %s ...", self.settings.caption_model)
            self._processor = BlipProcessor.from_pretrained(self.settings.caption_model)
            self._model = BlipForConditionalGeneration.from_pretrained(
                self.settings.caption_model
            ).to(self._device)
            self._model.eval()
            self.available = True
            logger.info("Caption model ready on %s.", self._device)
        except Exception as exc:  # pragma: no cover - depends on env
            logger.warning("Caption model unavailable, using mock captions: %s", exc)
            self.available = False

    def caption(self, image_bytes: bytes) -> tuple[str, bool]:
        """Return ``(caption, is_mock)`` for the given image bytes."""
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        self._try_load()
        if not self.available:
            return self._mock_caption(image), True

        try:
            import torch

            inputs = self._processor(image, return_tensors="pt").to(self._device)
            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=40)
            caption = self._processor.decode(out[0], skip_special_tokens=True).strip()
            return caption or self._mock_caption(image), False
        except Exception as exc:  # pragma: no cover - depends on env
            logger.warning("Caption inference failed, using mock: %s", exc)
            return self._mock_caption(image), True

    @staticmethod
    def _mock_caption(image: Image.Image) -> str:
        """Heuristic caption from basic image properties (no ML required)."""
        w, h = image.size
        orientation = (
            "portrait" if h > w * 1.15 else "landscape" if w > h * 1.15 else "square"
        )
        # Average colour -> rough dominant tone, just to make output feel grounded.
        small = image.resize((1, 1))
        r, g, b = small.getpixel((0, 0))[:3]
        if max(r, g, b) - min(r, g, b) < 25:
            tone = "neutral-toned" if (r + g + b) / 3 > 110 else "dark"
        else:
            tone = (
                "red" if r >= g and r >= b else "green" if g >= b else "blue"
            ) + "-toned"
        return f"a {tone} product photographed on a {orientation} background"


_service: CaptioningService | None = None


def get_captioning_service() -> CaptioningService:
    global _service
    if _service is None:
        _service = CaptioningService()
    return _service
