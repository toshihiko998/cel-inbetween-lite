from __future__ import annotations
import cv2
import numpy as np

def rgba_to_gray_u8(rgba: np.ndarray) -> np.ndarray:
    rgb = rgba[:, :, :3]
    # to BGR for OpenCV then gray
    bgr = rgb[:, :, ::-1]
    gray = cv2.cvtColor((bgr * 255.0).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return gray

def compute_flow_farneback(a_rgba: np.ndarray, b_rgba: np.ndarray) -> np.ndarray:
    """
    Compute dense optical flow from A to B.
    Returns flow (H,W,2) float32 in pixels: (dx, dy)
    """
    a = rgba_to_gray_u8(a_rgba)
    b = rgba_to_gray_u8(b_rgba)

    flow = cv2.calcOpticalFlowFarneback(
        a, b, None,
        pyr_scale=0.5,
        levels=5,
        winsize=25,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=0
    )
    return flow.astype(np.float32)

