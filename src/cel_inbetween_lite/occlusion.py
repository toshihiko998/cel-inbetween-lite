from __future__ import annotations
import numpy as np

def occlusion_mask(flow_ab: np.ndarray, flow_ba: np.ndarray, th: float = 1.5) -> np.ndarray:
    """
    Simple forward-backward consistency check.
    If flow_ab + flow_ba is large => inconsistent => occluded/unreliable.
    Returns mask in [0..1] where 1 means "reliable", 0 means "unreliable".
    """
    # This is a simplified approximation (no proper warping of flow_ba to A space)
    fb = flow_ab + flow_ba
    mag = np.sqrt(fb[:, :, 0] ** 2 + fb[:, :, 1] ** 2)
    reliable = (mag < th).astype(np.float32)
    return reliable

