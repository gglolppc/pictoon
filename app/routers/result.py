from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.paths import RESULTS_DIR

router = APIRouter()

@router.get("/{uid}")
async def get_result(uid: str):
    path = (RESULTS_DIR / f"{uid}.jpg").resolve()
    if not path.exists():
        raise HTTPException(404, "Result not found")
    return FileResponse(path, media_type="image/jpeg", filename=f"{uid}.jpg")
