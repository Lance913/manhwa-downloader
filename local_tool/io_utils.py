"""Loads chapter pages either from a URL (reusing the existing scraper) or
from a local folder of already-downloaded images."""

import os
import re
import sys
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
from scrape import _extract_ordered_images, REQUEST_HEADERS  # noqa: E402
from panels import _download_image  # noqa: E402

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _natural_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def sanitize_name(name):
    name = name or "chapter"
    name = re.sub(r'[\\/:*?"<>|]+', " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return (name[:80] or "chapter")


def load_from_url(url):
    """Returns (title, [BGR numpy image, ...])."""
    parsed = urlparse(url)
    resp = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
    resp.raise_for_status()

    images = _extract_ordered_images(resp.text, resp.url)
    if not images:
        raise ValueError("Couldn't find any chapter page images on that link")

    title_match = re.search(r"<title>(.*?)</title>", resp.text, re.I | re.S)
    title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else "chapter"
    title = title.split(" | ")[0].strip()
    title = re.sub(r"\s*[-|]\s*Read\s+Online\b.*$", "", title, flags=re.I).strip()
    title = sanitize_name(title)

    referer = f"{parsed.scheme}://{parsed.netloc}/"

    pages = []
    for img_url in images:
        pages.append(_download_with_retry(img_url, referer))

    return title, pages


def _download_with_retry(img_url, referer, attempts=5):
    """Image CDNs rate-limit bursts (HTTP 429) and hiccup (5xx); back off
    and retry instead of aborting the whole chapter."""
    import time
    delay = 2.0
    last_exc = None
    for attempt in range(attempts):
        try:
            return _download_image(img_url, referer=referer)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise
            last_exc = exc
        except requests.RequestException as exc:
            if attempt == attempts - 1:
                raise
            last_exc = exc
        time.sleep(delay)
        delay = min(delay * 2, 30)
    raise last_exc


def load_from_folder(folder):
    """Returns (title, [BGR numpy image, ...]) reading every image file in
    the folder, sorted in natural (page1, page2, ..., page10) order."""
    files = [f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS)]
    files.sort(key=_natural_key)
    if not files:
        raise ValueError(f"No image files found in {folder}")

    title = sanitize_name(os.path.basename(os.path.normpath(folder)))
    pages = []
    for f in files:
        path = os.path.join(folder, f)
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        pages.append(image)

    return title, pages
