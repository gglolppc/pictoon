from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = (BASE_DIR / "storage").resolve()
UPLOADS_DIR = STORAGE_DIR / "uploads"
RESULTS_DIR = STORAGE_DIR / "results"
TMP_DIR = STORAGE_DIR / "tmp"

for d in (STORAGE_DIR, UPLOADS_DIR, RESULTS_DIR, TMP_DIR):
    d.mkdir(parents=True, exist_ok=True)