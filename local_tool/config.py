"""Configurable thresholds, overridable via environment variables so a
future GUI/exe can expose them without code changes."""

import os
from dataclasses import dataclass


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    existing_weight: float = _env_float("EXISTING_DETECTOR_WEIGHT", 1.0)
    whitespace_weight: float = _env_float("WHITESPACE_DETECTOR_WEIGHT", 0.85)
    iou_threshold: float = _env_float("IOU_DUPLICATE_THRESHOLD", 0.5)
    # Safe to keep small: blank means zero ink, so the interior of a speech
    # bubble (whose outline crosses every row) can never look like a gutter.
    min_gutter_height: int = _env_int("MIN_GUTTER_HEIGHT", 12)
    min_segment_height: int = _env_int("MIN_SEGMENT_HEIGHT", 60)
    white_threshold: int = _env_int("GUTTER_WHITE_THRESHOLD", 235)
    black_threshold: int = _env_int("GUTTER_BLACK_THRESHOLD", 25)
    black_blank_max_std: float = _env_float("BLACK_BLANK_MAX_STD", 2.5)
    # Treat any flat single-color row as gutter (not just white/black).
    # Off by default: solid-color panel backgrounds would get sliced.
    gutter_uniform_rows: bool = os.environ.get("GUTTER_UNIFORM_ROWS", "0") == "1"
    gutter_max_row_std: float = _env_float("GUTTER_MAX_ROW_STD", 3.0)
    # A gutter segment is split into the sub-panels the other detectors
    # found only if those sub-panels cover at least this fraction of it.
    split_coverage: float = _env_float("SPLIT_COVERAGE", 0.6)

    # Row-profile segmentation (beats.py)
    # A blank row has essentially zero ink. Anything looser lets the empty
    # interior of a speech bubble (two outline pixels per row) count as
    # blank, and the bubble gets cut in half.
    gutter_uniform_fraction: float = _env_float("GUTTER_UNIFORM_FRACTION", 0.9995)
    # Relaxed: a long bold line of dialogue can cover half the row. The
    # chroma and midtone tests are what separate dialogue from art.
    text_white_fraction: float = _env_float("TEXT_WHITE_FRACTION", 0.45)
    text_max_midtone: float = _env_float("TEXT_MAX_MIDTONE", 0.25)
    text_max_chroma: float = _env_float("TEXT_MAX_CHROMA", 10.0)
    text_margin_white: float = _env_float("TEXT_MARGIN_WHITE", 0.90)
    margin_fraction: float = _env_float("MARGIN_FRACTION", 0.10)
    border_dark_value: int = _env_int("BORDER_DARK_VALUE", 60)
    border_dark_fraction: float = _env_float("BORDER_DARK_FRACTION", 0.85)
    # Taller than one line of lettering, so a text line misread as art can
    # never become a panel core of its own.
    min_art_rows: int = _env_int("MIN_ART_ROWS", 60)
    min_text_rows: int = _env_int("MIN_TEXT_ROWS", 40)
    # A flat-black run up to this tall is a frame line; taller is a gap.
    border_max_rows: int = _env_int("BORDER_MAX_ROWS", 8)
    # Minimum row-to-row pixel variation for an ART run to count as real
    # drawing rather than a smooth gradient/vignette. Measured gap: pure
    # gradients sit at ~0.3-0.8, real art (including dark scenes) at 30+.
    min_core_row_std: float = _env_float("MIN_CORE_ROW_STD", 5.0)
    rows_weight: float = _env_float("ROWS_DETECTOR_WEIGHT", 0.9)

    # Speech-balloon detector: a beat boundary is never allowed to pass
    # through a detected balloon.
    balloon_enabled: bool = os.environ.get("BALLOON_ENABLED", "1") == "1"
    balloon_model_path: str = os.environ.get(
        "BALLOON_MODEL_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "korean_webtoon", "model.onnx"),
    )
    # 0.4 missed a real bubble (unusual font/style) detected at 0.238 -
    # confirmed by direct inspection, not a guess. 0.2 matches the
    # model author's own documented usage example.
    balloon_confidence: float = _env_float("BALLOON_CONFIDENCE", 0.2)
    balloon_imgsz: int = _env_int("BALLOON_IMGSZ", 640)
    # Extra margin around a detected balloon (as a fraction of its own
    # height) before it's used to decide whether an ART run has real
    # content outside of it - covers spike/tail overshoot past the box.
    balloon_pad_fraction: float = _env_float("BALLOON_PAD_FRACTION", 0.4)
    missing_min_area_fraction: float = _env_float("MISSING_MIN_AREA_FRACTION", 0.02)
    padding_fraction: float = _env_float("PANEL_PADDING_FRACTION", 0.0)
    split_joint_panels: bool = os.environ.get("SPLIT_JOINT_PANELS", "0") == "1"
    reading_order_mode: str = os.environ.get("READING_ORDER_MODE", "vertical_webtoon")
    debug: bool = os.environ.get("DEBUG_VIZ", "1") == "1"

    # Off by default: measured on real chapters it never decided a boundary
    # correctly that the row profile hadn't already; it's an overlay/vote.
    yolo_enabled: bool = os.environ.get("YOLO_ENABLED", "0") == "1"
    yolo_model_path: str = os.environ.get(
        "YOLO_MODEL_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "manga_panel_detector_fp32.pt"),
    )
    yolo_weight: float = _env_float("YOLO_DETECTOR_WEIGHT", 1.0)
    yolo_confidence: float = _env_float("YOLO_CONFIDENCE", 0.25)
    yolo_iou: float = _env_float("YOLO_IOU", 0.45)
    yolo_imgsz: int = _env_int("YOLO_IMGSZ", 1024)
    yolo_device: str = os.environ.get("YOLO_DEVICE", "auto")
    # A YOLO box nothing else agrees with must be at least this confident
    # to become a panel on its own (the Manga109-trained model is noisy on
    # borderless webtoon layouts). Kept low: the real duplicate filter is
    # the content-based suppression below, which confidence can't replace
    # (the model is often *most* confident about its duplicates).
    yolo_solo_confidence: float = _env_float("YOLO_SOLO_CONFIDENCE", 0.3)
    # Drop a lone YOLO box if the only part of it not already inside another
    # panel is blank (edge density below this) - i.e. it recovers nothing.
    suppress_covered_solo: bool = os.environ.get("SUPPRESS_COVERED_SOLO", "1") == "1"
    new_content_edge_density: float = _env_float("NEW_CONTENT_EDGE_DENSITY", 0.01)
    yolo_tile_aspect: float = _env_float("YOLO_TILE_ASPECT", 2.0)
    yolo_tile_overlap: float = _env_float("YOLO_TILE_OVERLAP", 0.2)
