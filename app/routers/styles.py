from fastapi import APIRouter
router = APIRouter()

STYLES = [
    {"key": "comic", "name": "Comic"},
    {"key": "oil", "name": "Oil Painting"},
    {"key": "retro", "name": "Retro Film"},
]

@router.get("")
def list_styles():
    return {"styles": STYLES}
