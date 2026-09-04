"""
Vercel serverless function: downloads a single manhwa page image and splits it
into its individual panels/images using classic OpenCV contour analysis.

The detection algorithm is ported from adenzu/Manga-Panel-Extractor
(src/image_processing/panel.py + image.py, MIT-style classical CV path).
The AI/torch model path from that repo is intentionally left out here -
it needs a multi-hundred-MB PyTorch + YOLO stack that doesn't fit a
serverless function well, so this uses the same background-mask /
contour-detection approach the repo's non-AI mode uses.
"""

from http.server import BaseHTTPRequestHandler
import base64
import io
import json

import cv2
import numpy as np
import requests
from PIL import Image

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

MODE_BOUNDING = "bounding"
MODE_MASKED = "masked"


# ---------------------------------------------------------------------------
# Ported panel-detection algorithm
# ---------------------------------------------------------------------------

def get_background_intensity_range(grayscale_image, min_range=1):
    edges = [
        grayscale_image[-1, :],
        grayscale_image[0, :],
        grayscale_image[:, 0],
        grayscale_image[:, -1],
    ]
    least_varied_edge = sorted(edges, key=lambda e: np.var(e))[0]
    max_intensity = max(least_varied_edge)
    min_intensity = max(min(min(least_varied_edge), max_intensity - min_range), 0)
    return min_intensity, max_intensity


def is_contour_rectangular(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
    return len(approx) == 4


def generate_background_mask(grayscale_image):
    WHITE = 255
    less_white, _ = get_background_intensity_range(grayscale_image, 25)
    less_white = max(less_white, 240)

    _, thresh = cv2.threshold(grayscale_image, less_white, WHITE, cv2.THRESH_BINARY)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)

    mask = np.zeros_like(thresh)
    PAGE_TO_SEGMENT_RATIO = 1024
    halting_area_size = mask.size // PAGE_TO_SEGMENT_RATIO

    mask_height, mask_width = mask.shape
    error = 0.05
    whole_min_width = mask_width * (1 - error)
    whole_min_height = mask_height * (1 - error)

    for i in np.argsort(stats[1:, 4])[::-1]:
        idx = i + 1
        x, y, w, h, area = stats[idx]
        if area < halting_area_size:
            break
        component = (labels == idx).astype(np.uint8)
        contour = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
        if w > whole_min_width or h > whole_min_height or is_contour_rectangular(contour):
            mask[labels == idx] = WHITE

    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    return mask


def preprocess_image_with_dilation(grayscale_image):
    processed = cv2.GaussianBlur(grayscale_image, (3, 3), 0)
    processed = cv2.Laplacian(processed, -1)
    processed = cv2.dilate(processed, np.ones((5, 5), np.uint8), iterations=1)
    return 255 - processed


def is_contour_sufficiently_big(contour, image_height, image_width):
    # Webtoon "pages" are often one continuous strip holding many panels
    # (tens of thousands of px tall), unlike the roughly page-shaped
    # images this ratio was designed for. Scaling the threshold off the
    # full page area would make it explode on tall strips and silently
    # drop legitimately-sized panels, so cap the height used for the
    # reference area at a small multiple of the page width instead.
    effective_height = min(image_height, image_width * 3)
    area_threshold = (image_width * effective_height) // 32
    return cv2.contourArea(contour) > area_threshold


def apply_adaptive_threshold(image):
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 0
    )


MIN_PANEL_HEIGHT = 350


def merge_fragmented_boxes(boxes, min_panel_height=MIN_PANEL_HEIGHT):
    """
    Reunites boxes that are really pieces of one panel rather than separate
    panels. Two failure modes from the contour detection get fixed here:

    - Overlapping/touching boxes: a patch of page background in the middle
      of a single panel (e.g. a plain-colored stretch) got misread as a
      gutter, splitting one contour into two.
    - A short box trailing close behind another one: usually a speech/
      thought bubble that visually bleeds past its panel's border into the
      gutter and gets picked up as its own tiny contour.

    Gap size alone does not reliably separate "two fragments of one panel"
    from "two genuinely separate small panels" in this kind of full-bleed
    webtoon art - real inter-panel gaps span the same range (tens to
    hundreds of px) as the erroneous-split gaps, with no clean cutoff. So
    this only merges when a box is itself short (a real panel is rarely
    under min_panel_height tall) and close to its neighbor, which is a much
    narrower - and safer - net than gap size on its own.

    Both are merged into the union of their bounding boxes so the final
    crop comes straight from the original image with no stitching seam.
    """
    TIGHT_GAP = 150  # a gap this small is almost never a real gutter, regardless of box size

    if not boxes:
        return boxes
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged = [list(boxes[0])]
    for x, y, w, h in boxes[1:]:
        px, py, pw, ph = merged[-1]
        gap = y - (py + ph)
        overlaps = gap < 0
        trailing_fragment = h < min_panel_height and gap < min_panel_height
        tight_gap = gap < TIGHT_GAP
        if overlaps or trailing_fragment or tight_gap:
            nx, ny = min(px, x), min(py, y)
            nx2, ny2 = max(px + pw, x + w), max(py + ph, y + h)
            merged[-1] = [nx, ny, nx2 - nx, ny2 - ny]
        else:
            merged.append([x, y, w, h])
    return [tuple(b) for b in merged]


def extract_panels(image, contours, accept_page_as_panel=True, mode=MODE_BOUNDING,
                    fill_in_color=(0, 0, 0)):
    height, width = image.shape[:2]
    found = []

    if mode == MODE_MASKED:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not accept_page_as_panel and (w >= width * 0.99 or h >= height * 0.99):
                continue
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
            masked_image = cv2.bitwise_and(image, mask)
            fitted = masked_image[y:y + h, x:x + w]
            inverse_mask = cv2.bitwise_not(mask[y:y + h, x:x + w])
            fill = np.full_like(fitted, fill_in_color)
            fitted = cv2.bitwise_or(cv2.bitwise_and(inverse_mask, fill), fitted)
            found.append((x, y, fitted))
    else:
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not accept_page_as_panel and (w >= width * 0.99 or h >= height * 0.99):
                continue
            boxes.append((x, y, w, h))
        for x, y, w, h in merge_fragmented_boxes(boxes):
            x2, y2 = min(x + w, width), min(y + h, height)
            found.append((x, y, image[y:y2, x:x2]))

    # Sort into reading order: top-to-bottom, then left-to-right within a row.
    ROW_BUCKET = 40
    found.sort(key=lambda p: (round(p[1] / ROW_BUCKET), p[0]))
    return [panel for _, _, panel in found]


def threshold_extraction(image, grayscale_image, mode=MODE_BOUNDING):
    processed = cv2.GaussianBlur(grayscale_image, (3, 3), 0)
    processed = cv2.Laplacian(processed, -1)
    _, thresh = cv2.threshold(processed, 8, 255, cv2.THRESH_BINARY)
    processed = apply_adaptive_threshold(processed)
    processed = cv2.subtract(processed, thresh)
    processed = cv2.dilate(processed, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if is_contour_sufficiently_big(c, image.shape[0], image.shape[1])]
    return extract_panels(image, contours, accept_page_as_panel=False, mode=mode)


def joint_panel_split_extraction(grayscale_image, background_mask):
    """Attempts to split panels that touch each other with no visible gutter."""
    pixels_before = np.count_nonzero(background_mask)
    background_mask = cv2.ximgproc.thinning(background_mask)

    up_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]], np.uint8)
    down_kernel = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], np.uint8)
    left_kernel = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]], np.uint8)
    right_kernel = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]], np.uint8)
    down_right_diag = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], np.uint8)
    down_left_diag = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0]], np.uint8)
    up_left_diag = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], np.uint8)
    up_right_diag = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]], np.uint8)

    PAGE_TO_JOINT_OBJECT_RATIO = 3
    image_height, image_width = grayscale_image.shape

    height_based_size = image_height // PAGE_TO_JOINT_OBJECT_RATIO
    width_based_size = (2 * image_width) // PAGE_TO_JOINT_OBJECT_RATIO
    height_based_size += height_based_size % 2 + 1
    width_based_size += width_based_size % 2 + 1

    up_dilation = np.zeros((height_based_size, height_based_size), np.uint8)
    up_dilation[height_based_size // 2:, height_based_size // 2] = 1

    down_dilation = np.zeros((height_based_size, height_based_size), np.uint8)
    down_dilation[:height_based_size // 2 + 1, height_based_size // 2] = 1

    left_dilation = np.zeros((width_based_size, width_based_size), np.uint8)
    left_dilation[width_based_size // 2, width_based_size // 2:] = 1

    right_dilation = np.zeros((width_based_size, width_based_size), np.uint8)
    right_dilation[width_based_size // 2, :width_based_size // 2 + 1] = 1

    min_based_size = min(width_based_size, height_based_size)

    down_right_dilation = np.identity(min_based_size // 2 + 1, dtype=np.uint8)
    down_right_dilation = np.pad(down_right_dilation, ((0, min_based_size // 2), (0, min_based_size // 2)))

    up_left_dilation = np.identity(min_based_size // 2 + 1, dtype=np.uint8)
    up_left_dilation = np.pad(up_left_dilation, ((min_based_size // 2, 0), (0, min_based_size // 2)))

    up_right_dilation = np.flip(np.identity(min_based_size // 2 + 1, dtype=np.uint8), axis=1)
    up_right_dilation = np.pad(up_right_dilation, ((min_based_size // 2, 0), (0, min_based_size // 2)))

    down_left_dilation = np.flip(np.identity(min_based_size // 2 + 1, dtype=np.uint8), axis=1)
    down_left_dilation = np.pad(down_left_dilation, ((0, min_based_size // 2), (min_based_size // 2, 0)))

    match_kernels = [up_kernel, down_kernel, left_kernel, right_kernel,
                     down_right_diag, down_left_diag, up_left_diag, up_right_diag]
    dilation_kernels = [up_dilation, down_dilation, left_dilation, right_dilation,
                        down_right_dilation, down_left_dilation, up_left_dilation, up_right_dilation]

    def get_dots(mask, kernel):
        temp = cv2.matchTemplate(mask, kernel, cv2.TM_CCOEFF_NORMED)
        _, temp = cv2.threshold(temp, 0.9, 1, cv2.THRESH_BINARY)
        temp = np.where(temp == 1, 255, 0).astype(np.uint8)
        pad_h = (kernel.shape[0] - 1) // 2
        pad_w = (kernel.shape[1] - 1) // 2
        return cv2.copyMakeBorder(
            temp, pad_h, kernel.shape[0] - pad_h - 1, pad_w, kernel.shape[1] - pad_w - 1,
            cv2.BORDER_CONSTANT, value=0,
        )

    for match_kernel, dilation_kernel in zip(match_kernels, dilation_kernels):
        dots = get_dots(background_mask, match_kernel)
        lines = cv2.dilate(dots, dilation_kernel, iterations=1)
        background_mask = cv2.bitwise_or(background_mask, lines)

    pixels_now = max(np.count_nonzero(background_mask), 1)
    dilation_size = pixels_before // (4 * pixels_now)
    dilation_size += dilation_size % 2 + 1
    background_mask = cv2.dilate(background_mask, np.ones((dilation_size, dilation_size), np.uint8), iterations=1)

    return 255 - background_mask


def get_page_without_background(grayscale_image, background_mask, split_joint_panels=False):
    STRIPE_FORMAT_MASK_AREA_RATIO = 0.3
    mask_area_ratio = np.count_nonzero(background_mask) / background_mask.size

    if split_joint_panels and mask_area_ratio < STRIPE_FORMAT_MASK_AREA_RATIO:
        try:
            return joint_panel_split_extraction(grayscale_image, background_mask)
        except Exception:
            pass  # cv2.ximgproc unavailable or extraction failed - fall back below

    return cv2.subtract(grayscale_image, background_mask)


def get_fallback_panels(image, grayscale_image, fallback, panels, mode=MODE_BOUNDING):
    if fallback and len(panels) < 2:
        alt = threshold_extraction(image, grayscale_image, mode=mode)
        if len(alt) > len(panels):
            return alt
    return panels


def generate_panel_blocks(image, split_joint_panels=False, fallback=True, mode=MODE_BOUNDING):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_image = preprocess_image_with_dilation(grayscale_image)
    background_mask = generate_background_mask(processed_image)
    page_without_background = get_page_without_background(grayscale_image, background_mask, split_joint_panels)

    contours, _ = cv2.findContours(page_without_background, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if is_contour_sufficiently_big(c, image.shape[0], image.shape[1])]

    panels = extract_panels(image, contours, mode=mode)
    panels = get_fallback_panels(image, grayscale_image, fallback, panels, mode=mode)

    if not panels:
        panels = [image]

    return panels


# ---------------------------------------------------------------------------
# Panel importance signals (no ML/API calls - cheap per-panel image stats
# the client can rank across a whole chapter to keep only the panels that
# most likely matter to the story).
# ---------------------------------------------------------------------------

def enclosed_white_ratio(panel):
    """Fraction of the panel that is white and NOT connected to the panel's
    outer edge - i.e. white area enclosed by ink, which speech/thought
    bubbles are, as opposed to plain white background or gutter bleed."""
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    height, width = white_mask.shape
    flood_fill_mask = np.zeros((height + 2, width + 2), np.uint8)
    enclosed = white_mask.copy()

    border_points = (
        [(x, 0) for x in range(width)] + [(x, height - 1) for x in range(width)] +
        [(0, y) for y in range(height)] + [(width - 1, y) for y in range(height)]
    )
    for x, y in border_points:
        if enclosed[y, x] == 255:
            cv2.floodFill(enclosed, flood_fill_mask, (x, y), 0)

    return float(np.count_nonzero(enclosed)) / white_mask.size


def edge_density(panel):
    """Fraction of pixels that are edges - a proxy for how much visual
    detail/action a panel holds, vs. a plain/empty establishing panel."""
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    return float(np.count_nonzero(edges)) / edges.size


def panel_importance_features(panel, page_area):
    height, width = panel.shape[:2]
    return {
        "areaRatio": (width * height) / page_area if page_area else 0,
        "edgeDensity": edge_density(panel),
        "bubbleDensity": enclosed_white_ratio(panel),
    }


# ---------------------------------------------------------------------------
# Serverless handler
# ---------------------------------------------------------------------------

def _download_image(url, referer=None):
    headers = {"User-Agent": USER_AGENT}
    if referer:
        headers["Referer"] = referer
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    pil_image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


class handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw or b"{}")

            url = (data.get("url") or "").strip()
            if not url:
                raise ValueError("Missing image url")

            mode = data.get("mode") or MODE_BOUNDING
            if mode not in (MODE_BOUNDING, MODE_MASKED):
                mode = MODE_BOUNDING
            split_joint_panels = bool(data.get("splitJointPanels", False))
            whole_page = bool(data.get("wholePage", False))
            referer = (data.get("referer") or "").strip() or None

            image = _download_image(url, referer=referer)
            page_height, page_width = image.shape[:2]

            if whole_page:
                panels = [image]
            else:
                panels = generate_panel_blocks(image, split_joint_panels=split_joint_panels, mode=mode)

            page_area = page_width * page_height
            encoded_panels = []
            panel_stats = []
            for panel in panels:
                ok, buf = cv2.imencode(".jpg", panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not ok:
                    continue
                encoded_panels.append(base64.b64encode(buf.tobytes()).decode("ascii"))
                panel_stats.append(panel_importance_features(panel, page_area))

            # Page dimensions let the client spot ad/credit-card pages that
            # scanlation groups splice into the chapter (they're a visibly
            # different shape from every real page) without needing OCR.
            # Panel stats let it rank panels chapter-wide by likely story
            # importance (size, detail, dialogue) without any AI/API calls.
            self._send(200, {
                "panels": encoded_panels,
                "panelStats": panel_stats,
                "pageWidth": page_width,
                "pageHeight": page_height,
            })
        except Exception as exc:  # noqa: BLE001 - surface any failure to the client
            self._send(400, {"error": str(exc)})
