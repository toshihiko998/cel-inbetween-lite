from __future__ import annotations
import cv2
import numpy as np

def edge_map_from_rgba(rgba: np.ndarray) -> np.ndarray:
    """
    Returns binary-ish edge map (0..1 float32).
    Uses RGB edges + alpha boundary.
    """
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]

    # RGB edges
    bgr_u8 = (np.clip(rgb[:, :, ::-1], 0, 1) * 255).astype(np.uint8)
    gray = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2GRAY)
    e1 = cv2.Canny(gray, 60, 150).astype(np.float32) / 255.0

    # Alpha edges (boundary)
    a_u8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    e2 = cv2.Canny(a_u8, 30, 90).astype(np.float32) / 255.0

    e = np.maximum(e1, e2)
    return e

def edge_distance_weight(edge01: np.ndarray, protect_radius_px: float) -> np.ndarray:
    """
    Builds a weight map in [0..1]:
    - near edges => weight ~0 (avoid blending)
    - far from edges => weight ~1 (allow blending)
    """
    # edge01: 0..1. Create binary edges
    edge_bin = (edge01 > 0.2).astype(np.uint8)
    inv = (1 - edge_bin)  # 1 where not edge
    dist = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3).astype(np.float32)

    # Normalize by protect radius
    r = max(1e-3, float(protect_radius_px))
    w = np.clip(dist / r, 0.0, 1.0)
    return w

