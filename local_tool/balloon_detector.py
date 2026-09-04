"""
Speech-balloon detector (YOLOv8, trained on Korean webtoons by xulihang's
balloon-dataset, shipped as ONNX). Fast (~0.1s/page on CPU) and precise,
so it serves as a hard constraint: a beat boundary must never pass
through a balloon.

Loaded once and cached; if the model or runtime is missing the detector
reports itself unavailable and the pipeline proceeds without it.
"""

import os

from detectors import Detection

_model = None
_loaded_path = None
_unavailable_reason = None


def load_model(model_path):
    global _model, _loaded_path, _unavailable_reason
    if _loaded_path == model_path:
        return _model
    _loaded_path = model_path
    _model = None
    if not os.path.isfile(model_path):
        _unavailable_reason = f"balloon model not found: {model_path}"
        return None
    try:
        from ultralytics import YOLO
        _model = YOLO(model_path)
        _unavailable_reason = None
    except Exception as exc:  # noqa: BLE001
        _unavailable_reason = f"balloon model failed to load: {exc}"
        _model = None
    return _model


def unavailable_reason():
    return _unavailable_reason


def detect_balloons_on_page(image, model_path, confidence=0.4, imgsz=640):
    """Balloon boxes on one page image, in page coordinates."""
    model = load_model(model_path)
    if model is None:
        return []
    results = model.predict(image, imgsz=imgsz, conf=confidence, verbose=False)
    if not results or results[0].boxes is None:
        return []
    boxes = results[0].boxes
    height, width = image.shape[:2]
    out = []
    for cls, conf, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
        if int(cls) != 0:  # class 0 = balloon; class 1 = 'other'
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 > x1 and y2 > y1:
            out.append(Detection(x1, y1, x2, y2, float(conf), "balloon", ["balloon"]))
    return out


def detect_balloons_on_strip(strip, model_path, confidence=0.4, imgsz=640):
    """Balloon boxes across the whole strip, in strip coordinates. Pages
    are run individually; a balloon straddling a page seam produces two
    partial boxes, which are merged when they touch at the seam."""
    boxes = []
    for i, page in enumerate(strip.pages):
        off = strip.offsets[i]
        for d in detect_balloons_on_page(page, model_path, confidence, imgsz):
            boxes.append(Detection(d.x1, d.y1 + off, d.x2, d.y2 + off, d.confidence, "balloon", ["balloon"]))

    boxes.sort(key=lambda d: d.y1)
    merged = []
    for d in boxes:
        if merged:
            m = merged[-1]
            seam_touch = abs(d.y1 - m.y2) <= 3 and d.y1 in strip.offsets
            x_overlap = min(d.x2, m.x2) - max(d.x1, m.x1)
            if seam_touch and x_overlap > 0.5 * min(d.w, m.w):
                m.x1, m.x2 = min(m.x1, d.x1), max(m.x2, d.x2)
                m.y2 = max(m.y2, d.y2)
                m.confidence = max(m.confidence, d.confidence)
                continue
        merged.append(d)
    return merged
