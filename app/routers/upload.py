from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from app.utils.ids import new_id
from app.utils.images import validate_mime, normalize_exif_orientation, process_image
from app.config import settings
from app.paths import UPLOADS_DIR, RESULTS_DIR
from pathlib import Path
from app.utils.limiter import limiter

router = APIRouter()

@router.post("")
@limiter.limit("20/minute")
async def upload_image(request: Request, style: str, file: UploadFile = File(...)):
    # 1) стиль валиден? (пока просто проверим тройку допустимых)
    if style not in {"comic", "oil", "retro"}:
        raise HTTPException(400, "Unsupported style")

    # 2) MIME и размер
    if not validate_mime(file.content_type or ""):
        raise HTTPException(415, "Unsupported media type")

    raw = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(413, "File too large")

    # 3) normalize EXIF
    try:
        fixed = normalize_exif_orientation(raw)
    except Exception:
        raise HTTPException(400, "Corrupted or unsupported image")

    # 4) сохраняем безопасно
    uid = new_id()
    src = (UPLOADS_DIR / f"{uid}.jpg").resolve()
    src.write_bytes(fixed)
    dst = (RESULTS_DIR / f"{uid}.jpg").resolve()
    # защитимся от path traversal (resolve + проверка родителей)
    if UPLOADS_DIR not in src.parents:
        raise HTTPException(400, "Invalid path")

    try:
        process_image(uid, style, src, dst)
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {e}")

    return {"id": uid, "result_url": f"/result/{uid}"}

from fastapi.responses import FileResponse
from app.paths import RESULTS_DIR

# @router.get("/{uid}")
# async def get_result(uid: str):
#     path = (RESULTS_DIR / f"{uid}.jpg").resolve()
#     if not path.exists():
#         raise HTTPException(404, "Result not found")
#     return FileResponse(path, media_type="image/jpeg", filename=f"{uid}.jpg")
