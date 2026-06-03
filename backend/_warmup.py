"""Pre-download & initialize all models so the first real request is fast.

Forces BLIP, the sentence-transformer embedder and FLAN-T5 to download and run
once. Safe to delete afterwards.
"""
import time
from io import BytesIO

from PIL import Image

from app.services.captioning import get_captioning_service
from app.services.generator import get_generator_service
from app.services.rag import get_rag_service


def banner(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


t0 = time.time()
img = Image.new("RGB", (64, 64), (120, 120, 120))
buf = BytesIO()
img.save(buf, format="PNG")
data = buf.getvalue()

banner("Loading BLIP caption model (largest download ~1.9 GB)")
cap = get_captioning_service()
caption, is_mock = cap.caption(data)
print(f"caption available={cap.available} mock={is_mock} -> {caption!r}", flush=True)

banner("Loading sentence-transformer embedder")
rag = get_rag_service()
sources = rag.retrieve(caption)
print(f"rag available={rag.available} top={sources[0]['category'] if sources else None}", flush=True)

banner("Loading FLAN-T5 text model")
gen = get_generator_service()
listing = gen.generate(caption, sources)
print(f"generator available={gen.available}", flush=True)
print(f"sample title -> {listing['title']!r}", flush=True)

banner("WARMUP COMPLETE")
print(f"caption_model_loaded={cap.available} "
      f"embedder_loaded={rag.available} "
      f"text_model_loaded={gen.available}", flush=True)
print(f"elapsed={time.time() - t0:.0f}s", flush=True)
