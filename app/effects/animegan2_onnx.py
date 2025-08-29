from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort

MODEL_PATH = Path(__file__).resolve().parents[1] / "weights" / "Shinkai_53.onnx"
# CPU провайдер
_sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])

def _get_io():
    i = _sess.get_inputs()[0]
    o = _sess.get_outputs()[0]
    return i.name, o.name

_IN_NAME, _OUT_NAME = _get_io()

def apply_shinkai(img_rgb: np.ndarray) -> np.ndarray:
    """
    img_rgb: uint8 RGB [H,W,3]
    return:  uint8 RGB [H,W,3]
    """
    h, w = img_rgb.shape[:2]

    # White-Box обычно обучен на 256x256 (NHWC). Масштабируем.
    inp = cv2.resize(img_rgb, (700, 700), interpolation=cv2.INTER_AREA).astype(np.float32)

    # нормализация в [-1, 1] (типична для TF)
    inp = inp / 127.5 - 1.0
    # NHWC -> добавляем батч
    inp = np.expand_dims(inp, axis=0)  # [1,256,256,3]

    out = _sess.run([_OUT_NAME], {_IN_NAME: inp})[0]  # [1,256,256,3], float32 in [-1..1] или [0..1]
    out = out[0]

    # приводим к [0..255] uint8; если модель даёт [-1..1], конвертируем:
    if out.min() < 0:  # значит, в [-1..1]
        out = (out + 1.0) * 127.5
    else:              # иногда SavedModel уже в [0..1]
        out = out * 255.0

    out = np.clip(out, 0, 255).astype(np.uint8)
    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
    return out
