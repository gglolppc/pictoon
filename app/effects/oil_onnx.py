from pathlib import Path
import numpy as np
import onnxruntime as ort
import cv2
import logging

log = logging.getLogger("oil-onnx")

_session = None
_input_name = None
_output_name = None
_expected_hw = None  # (H, W) или None, если динамический вход

def _get_session() -> ort.InferenceSession:
    global _session, _input_name, _output_name, _expected_hw
    if _session is None:
        model_path = Path(__file__).resolve().parents[1] / "weights" / "candy_dynamic.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        _session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"]
        )
        _input_name  = _session.get_inputs()[0].name
        _output_name = _session.get_outputs()[0].name

        # Прочитаем форму входа: [N, 3, H, W]
        shape = _session.get_inputs()[0].shape
        # Если H/W — None или 'dynamic', значит можно любое
        def _num(x):
            if isinstance(x, (int, np.integer)): return int(x)
            try:
                return int(x)
            except Exception:
                return None

        n, c, h, w = (shape + [None]*4)[:4]
        h = _num(h); w = _num(w)
        _expected_hw = (h, w) if (h and w) else None

        log.warning(f"[oil] loaded: {model_path.name}, input={_input_name}, shape={shape}, expected_hw={_expected_hw}")

    return _session

def _preprocess_fit(img_rgb: np.ndarray, target_hw: tuple[int,int] | None):
    """
    Если target_hw задан (например, (224,224)), ресайзим к нему.
    Возвращаем: (tensor NCHW float32 0..255, (orig_h, orig_w))
    """
    h0, w0 = img_rgb.shape[:2]
    if target_hw:
        th, tw = target_hw
        img_in = cv2.resize(img_rgb, (tw, th), interpolation=cv2.INTER_AREA)
    else:
        img_in = img_rgb

    x = img_in.astype(np.float32).transpose(2, 0, 1)[None, ...]  # NCHW
    return x, (h0, w0)

def _postprocess(y: np.ndarray) -> np.ndarray:
    img = y[0].transpose(1, 2, 0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def apply_oil_onnx(img_rgb: np.ndarray) -> np.ndarray:
    sess = _get_session()
    x, orig_hw = _preprocess_fit(img_rgb, _expected_hw)
    y = sess.run([_output_name], {_input_name: x})[0]
    out = _postprocess(y)

    # если модель была 224×224 — вернём исходный размер
    oh, ow = orig_hw
    if out.shape[0] != oh or out.shape[1] != ow:
        out = cv2.resize(out, (ow, oh), interpolation=cv2.INTER_CUBIC)

    # чуть «звона»
    blur = cv2.GaussianBlur(out, (0, 0), 1.0)
    out = cv2.addWeighted(out, 1.20, blur, -0.20, 0)
    return out
