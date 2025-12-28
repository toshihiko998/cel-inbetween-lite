from __future__ import annotations
import cv2
import numpy as np

def line_mask_from_rgba(rgba: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Create a 'line' mask (0..1), where 1 means strong dark line.
    Works best for typical cel line-art (dark contours).
    """
    rgb = rgba[:, :, :3]
    bgr_u8 = (np.clip(rgb[:, :, ::-1], 0, 1) * 255).astype(np.uint8)
    gray = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Emphasize dark strokes: invert
    inv = 1.0 - gray

    # Edge-ish enhancement (DoG-like)
    blur1 = cv2.GaussianBlur(inv, (0, 0), 0.8)
    blur2 = cv2.GaussianBlur(inv, (0, 0), 2.0)
    dog = np.clip(blur1 - blur2, 0.0, 1.0)

    # Threshold to isolate lines
    m = (dog > 0.08).astype(np.float32)

    k = max(1, int(kernel_size))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.dilate(m, kernel, iterations=1)

    # Softness
    m = cv2.GaussianBlur(m, (0, 0), 0.8)
    m = np.clip(m, 0.0, 1.0)
    return m

def reinject_lines(base_rgba: np.ndarray, line_mask01: np.ndarray, strength: float) -> np.ndarray:
    """
    Darken base image where line_mask is 1.
    Uses 'min' style darkening on RGB, preserves alpha.
    """
    s = float(np.clip(strength, 0.0, 1.0))
    out = base_rgba.copy()
    rgb = out[:, :, :3]
    a = out[:, :, 3:4]

    # Create a darkening factor: near lines -> multiply down
    # factor in [1-s .. 1]
    factor = 1.0 - s * line_mask01[:, :, None]
    rgb = np.clip(rgb * factor, 0.0, 1.0)

    out[:, :, :3] = rgb
    out[:, :, 3:4] = a
    return out

