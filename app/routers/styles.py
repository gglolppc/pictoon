from fastapi import APIRouter
router = APIRouter()

STYLES = [
    {"key": "comic", "name": "Comic"},
    {"key": "oil", "name": "Oil Painting"},
    {"key": "retro", "name": "Retro Film"},
    {"key": "anime", "name": "anime"},
    {"key": "pencil", "name": "Pencil"},
    {"key": "vhs", "name": "Vhs / Glitch"},

]

@router.get("")
def list_styles():
    return {"styles": STYLES}
