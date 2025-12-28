from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

def imread_rgba(path: Path) -> np.ndarray:
    """Read image as RGBA float32 (0..1)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif img.shape[2] == 4:
        pass
    else:
        raise ValueError(f"Unsupported channels: {img.shape}")

    img = img.astype(np.float32) / 255.0
    # BGRA -> RGBA
    img = img[:, :, [2, 1, 0, 3]]
    return img

def imwrite_rgba(path: Path, rgba: np.ndarray) -> None:
    """Write RGBA float32 (0..1) to PNG."""
    x = np.clip(rgba, 0.0, 1.0)
    x = (x * 255.0 + 0.5).astype(np.uint8)
    # RGBA -> BGRA
    x = x[:, :, [2, 1, 0, 3]]
    cv2.imwrite(str(path), x)

def ensure_same_shape(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape[:2] != b.shape[:2]:
        raise ValueError(f"Image sizes differ: {a.shape[:2]} vs {b.shape[:2]}")
    if a.shape[2] != 4 or b.shape[2] != 4:
        raise ValueError("Expected RGBA images.")
    return a, b

