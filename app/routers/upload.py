from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from app.utils.ids import new_id
from app.utils.images import validate_mime, normalize_exif_orientation, process_image
from app.config import settings
from app.paths import UPLOADS_DIR, RESULTS_DIR
from pathlib import Path
from app.utils.limiter import limiter
from PIL import Image
from io import BytesIO

router = APIRouter()

@router.post("")
@limiter.limit("25/minute")
async def upload_image(request: Request, style: str, file: UploadFile = File(...)):
    if style not in {"comic", "oil", "retro", "anime"}:
        raise HTTPException(400, "Unsupported style")

    if not validate_mime(file.content_type or ""):
        raise HTTPException(415, "Unsupported media type")

    raw = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    max_side = 512  # Максимальный размер стороны в пикселях

    fixed = normalize_exif_orientation(raw)

    # Проверка размеров изображения
    try:
        img = Image.open(BytesIO(fixed)).convert("RGB")  # Конвертация в RGB для WebP
        width, height = img.size
        print(f"Original size: {width}x{height} pixels")  # Отладка

        # Если размер файла больше max_bytes или одна из сторон больше max_side
        if len(raw) > max_bytes or max(width, height) > max_side:
            print("Resizing image...")  # Отладка
            img.thumbnail((max_side, max_side))  # Уменьшаем до max_side
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=95)
            fixed = buf.getvalue()
            print(f"Resized size: {len(fixed)} bytes")  # Отладка
    except Exception as e:
        print(f"Resize error: {e}")  # Отладка
        raise HTTPException(400, "Corrupted or unsupported image")

    # Сохраняем
    uid = new_id()
    src = (UPLOADS_DIR / f"{uid}.jpg").resolve()
    src.write_bytes(fixed)
    dst = (RESULTS_DIR / f"{uid}.jpg").resolve()
    if UPLOADS_DIR not in src.parents:
        raise HTTPException(400, "Invalid path")

    try:
        process_image(uid, style, src, dst)
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {e}")

    return {"id": uid, "result_url": f"/result/{uid}"}
