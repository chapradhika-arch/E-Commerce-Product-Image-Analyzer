"""Analysis endpoints: single image and bulk upload."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas import BulkAnalysisResponse, ProductAnalysis
from app.services.pipeline import analyze_image

router = APIRouter(prefix="/api", tags=["analyze"])

_ALLOWED = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}
_MAX_BYTES = 15 * 1024 * 1024  # 15 MB per file


async def _read_image(file: UploadFile) -> bytes:
    if file.content_type not in _ALLOWED:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}' for {file.filename}",
        )
    data = await file.read()
    if len(data) > _MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"{file.filename} exceeds 15 MB")
    if not data:
        raise HTTPException(status_code=400, detail=f"{file.filename} is empty")
    return data


@router.post("/analyze", response_model=ProductAnalysis)
async def analyze_single(file: UploadFile = File(...)) -> ProductAnalysis:
    data = await _read_image(file)
    return analyze_image(file.filename or "image", data)


@router.post("/analyze/bulk", response_model=BulkAnalysisResponse)
async def analyze_bulk(files: list[UploadFile] = File(...)) -> BulkAnalysisResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Max 20 images per request")

    results: list[ProductAnalysis] = []
    for file in files:
        try:
            data = await _read_image(file)
            results.append(analyze_image(file.filename or "image", data))
        except HTTPException as exc:
            results.append(
                ProductAnalysis(
                    filename=file.filename or "image",
                    caption="",
                    title=file.filename or "image",
                    description="",
                    tags=[],
                    category="Unknown",
                    mock=True,
                    error=str(exc.detail),
                )
            )
    return BulkAnalysisResponse(count=len(results), results=results)
