from __future__ import annotations
import cv2
import numpy as np

def warp_rgba(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Warp RGBA image using flow field.
    flow gives displacement from source to target in pixels (dx, dy).
    We produce target by sampling source at (x - dx, y - dy).
    """
    h, w = img.shape[:2]
    # grid of target coords
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x - flow[:, :, 0]).astype(np.float32)
    map_y = (grid_y - flow[:, :, 1]).astype(np.float32)

    # OpenCV remap expects channels last; works for RGBA too
    warped = cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return warped

