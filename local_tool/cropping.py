"""Crops final panel boxes from the original full-resolution page image."""


def crop_panel(image, detection, padding_fraction=0.0):
    height, width = image.shape[:2]
    pad_x = int(detection.w * padding_fraction)
    pad_y = int(detection.h * padding_fraction)

    x1 = max(0, detection.x1 - pad_x)
    y1 = max(0, detection.y1 - pad_y)
    x2 = min(width, detection.x2 + pad_x)
    y2 = min(height, detection.y2 + pad_y)

    return image[y1:y2, x1:x2]
