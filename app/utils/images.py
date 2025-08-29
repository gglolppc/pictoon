from PIL import Image, ImageOps
from io import BytesIO
import cv2
import numpy as np
from pathlib import Path
from app.effects.oil_onnx import apply_oil_onnx
from app.effects.animegan2_onnx import apply_shinkai

def apply_comic(img: np.ndarray) -> np.ndarray:
    # 1. Уменьшаем шум
    blur = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    # 2. Получаем контуры
    gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 150)
    edges = cv2.cvtColor(255 - edges, cv2.COLOR_GRAY2RGB)
    # 3. Цветовая квантизация
    Z = blur.reshape((-1,3))
    Z = np.float32(Z)
    K = 8
    _, labels, centers = cv2.kmeans(Z, K, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape(blur.shape)
    # 4. Объединяем с контурами
    cartoon = cv2.bitwise_and(quantized, edges)
    return cartoon




def apply_retro(img: np.ndarray) -> np.ndarray:
    # 1. Sepia фильтр
    sepia = np.array([[0.393,0.769,0.189],
                      [0.349,0.686,0.168],
                      [0.272,0.534,0.131]])
    retro = cv2.transform(img, sepia)
    retro = np.clip(retro, 0, 255).astype(np.uint8)
    # 2. Лёгкий виньет
    rows, cols = retro.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/3)
    kernel_y = cv2.getGaussianKernel(rows, rows/3)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    vignette = np.copy(retro)
    for i in range(3):
        vignette[:,:,i] = vignette[:,:,i] * mask
    return vignette


ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}

def validate_mime(mime: str) -> bool:
    return mime in ALLOWED_MIME

def normalize_exif_orientation(data: bytes) -> bytes:
    img = Image.open(BytesIO(data))
    img = ImageOps.exif_transpose(img)
    out = BytesIO()

    fmt = "PNG" if img.mode in ("RGBA", "LA") else "JPEG"
    img.save(out, format=fmt, quality=95)
    return out.getvalue()


def process_image(uid: str, style: str, src_path: Path, dst_path: Path):
    img = cv2.imread(str(src_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if style == "comic":
        out = apply_comic(img)
    elif style == "retro":
        out = apply_retro(img)
    elif style == "oil":
        out = apply_oil_onnx(img)
    elif style == "anime":
        out = apply_shinkai(img)
    else:
        raise ValueError("Unsupported style")

    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(dst_path), out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])