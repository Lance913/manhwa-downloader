"""
Detector C: a locally-run YOLO panel detector (Ultralytics format).

The model is loaded once and cached. If ultralytics/torch aren't installed
or the weights file is missing, the detector reports itself unavailable and
the pipeline simply runs without it - YOLO is an extra vote, never a
requirement.

Very tall webtoon strips are tiled into overlapping windows before
inference: letterboxing a 800x12000 strip into a 1024x1024 square would
shrink every panel to a few pixels tall. Each tile's boxes are shifted back
into page coordinates; overlapping duplicates from adjacent tiles are left
for the ensemble's IoU/containment merge to collapse.
"""

import os

from detectors import Detection

PANEL_CLASS_ID = 0
TEXT_CLASS_ID = 1

_model = None
_model_path_loaded = None
_unavailable_reason = None


def _resolve_device(requested):
    if requested and requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"


def load_model(model_path):
    global _model, _model_path_loaded, _unavailable_reason
    if _model is not None and _model_path_loaded == model_path:
        return _model
    if _unavailable_reason is not None and _model_path_loaded == model_path:
        return None

    _model_path_loaded = model_path
    if not os.path.isfile(model_path):
        _unavailable_reason = f"model file not found: {model_path}"
        return None
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        _unavailable_reason = f"ultralytics not installed ({exc})"
        return None
    try:
        _model = YOLO(model_path)
    except Exception as exc:  # noqa: BLE001 - any load failure just disables YOLO
        _unavailable_reason = f"failed to load model: {exc}"
        return None
    _unavailable_reason = None
    return _model


def unavailable_reason():
    return _unavailable_reason


def _tiles(height, width, tile_aspect, overlap):
    tile_height = max(int(width * tile_aspect), 1)
    if height <= tile_height:
        return [(0, height)]
    step = max(int(tile_height * (1 - overlap)), 1)
    tiles = []
    y = 0
    while True:
        y2 = min(y + tile_height, height)
        tiles.append((y, y2))
        if y2 >= height:
            break
        y += step
    return tiles


def yolo_detector(image, model_path, confidence_threshold=0.25, iou_threshold=0.45,
                  imgsz=1024, device="auto", weight=1.0, tile_aspect=1.5, tile_overlap=0.2):
    """Returns (panel_detections, text_detections). Text boxes are for
    debugging/visualization only and are never treated as panels."""
    model = load_model(model_path)
    if model is None:
        return [], []

    height, width = image.shape[:2]
    resolved_device = _resolve_device(device)

    panels, texts = [], []
    for y_start, y_end in _tiles(height, width, tile_aspect, tile_overlap):
        tile = image[y_start:y_end, :]
        results = model.predict(
            tile, imgsz=imgsz, conf=confidence_threshold, iou=iou_threshold,
            device=resolved_device, verbose=False,
        )
        if not results:
            continue
        boxes = results[0].boxes
        if boxes is None:
            continue
        for cls, conf, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
            x1, y1, x2, y2 = xyxy
            det = Detection(
                int(round(x1)), int(round(y1)) + y_start,
                int(round(x2)), int(round(y2)) + y_start,
                # Scale the model's own confidence by this detector's ensemble weight.
                float(conf) * weight,
                "yolo" if int(cls) == PANEL_CLASS_ID else "yolo-text",
            )
            det.x1 = max(0, min(det.x1, width))
            det.x2 = max(0, min(det.x2, width))
            det.y1 = max(0, min(det.y1, height))
            det.y2 = max(0, min(det.y2, height))
            if det.w <= 0 or det.h <= 0:
                continue
            (panels if int(cls) == PANEL_CLASS_ID else texts).append(det)

    return panels, texts
