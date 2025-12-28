from __future__ import annotations
from pathlib import Path
import numpy as np

from .io import imread_rgba, imwrite_rgba, ensure_same_shape
from .flow import compute_flow_farneback
from .warp import warp_rgba
from .edge import edge_map_from_rgba, edge_distance_weight
from .occlusion import occlusion_mask
from .line_inject import line_mask_from_rgba, reinject_lines

def compose_edge_aware(
    wa: np.ndarray, wb: np.ndarray,
    blend_t: float,
    edge_w: np.ndarray,
    reliable_w: np.ndarray
) -> np.ndarray:
    """
    edge_w: 0..1, 0 near edges => do not blend; 1 far from edges => blend ok
    reliable_w: 0..1, 0 unreliable => avoid blending (choose one side)
    """
    t = float(np.clip(blend_t, 0.0, 1.0))

    # Base blend weight for B
    wB = t * edge_w * reliable_w
    # But if not blending allowed, pick closer keyframe (t<0.5 => A, else B)
    hard_pick_B = (t >= 0.5).astype(np.float32) if isinstance(t, np.ndarray) else None

    # For simplicity: make fallback pick map
    fallback_pick_B = np.full(edge_w.shape, 1.0 if t >= 0.5 else 0.0, dtype=np.float32)

    # When edge_w or reliable_w small, use fallback_pick_B instead of blending.
    blend_ok = (edge_w * reliable_w)  # 0..1
    wB = wB + (1.0 - blend_ok) * fallback_pick_B

    wB = np.clip(wB, 0.0, 1.0)
    wA = 1.0 - wB

    out = wa * wA[:, :, None] + wb * wB[:, :, None]
    return np.clip(out, 0.0, 1.0)

def inbetween_pair(
    a_path: Path,
    b_path: Path,
    n: int,
    out_dir: Path,
    prefix: str = "",
    start_index: int = 1,
    digits: int = 4,
    edge_protect: float = 6.0,
    occ_th: float = 1.5,
    line_strength: float = 0.85,
    line_kernel: int = 3,
    flow_scale: float = 1.0,
) -> None:
    if n <= 0:
        raise ValueError("--n must be >= 1")

    A = imread_rgba(a_path)
    B = imread_rgba(b_path)
    A, B = ensure_same_shape(A, B)

    # Compute flows
    flow_ab = compute_flow_farneback(A, B) * float(flow_scale)
    flow_ba = compute_flow_farneback(B, A) * float(flow_scale)

    # Edge maps / weights (use max edges from both to protect boundaries)
    eA = edge_map_from_rgba(A)
    eB = edge_map_from_rgba(B)
    e = np.maximum(eA, eB)
    edge_w = edge_distance_weight(e, protect_radius_px=edge_protect)  # 0..1

    # Occlusion reliability (simple)
    rel = occlusion_mask(flow_ab, flow_ba, th=occ_th)  # 0..1

    # Line masks (will be warped and blended)
    lineA = line_mask_from_rgba(A, kernel_size=line_kernel)
    lineB = line_mask_from_rgba(B, kernel_size=line_kernel)

    for i in range(1, n + 1):
        t = i / (n + 1)

        # Warp A towards B by t*flow_ab
        wa = warp_rgba(A, flow_ab * t)
        # Warp B towards A by (1-t)*flow_ba (so it aligns to intermediate)
        wb = warp_rgba(B, flow_ba * (1.0 - t))

        # Compose with edge-aware + occlusion-aware blending
        out = compose_edge_aware(wa, wb, t, edge_w=edge_w, reliable_w=rel)

        # Warp & blend line masks, then reinject lines
        wlineA = warp_rgba(np.dstack([lineA, lineA, lineA, lineA]), flow_ab * t)[:, :, 0]
        wlineB = warp_rgba(np.dstack([lineB, lineB, lineB, lineB]), flow_ba * (1.0 - t))[:, :, 0]
        line = np.clip((1.0 - t) * wlineA + t * wlineB, 0.0, 1.0)

        out = reinject_lines(out, line_mask01=line, strength=line_strength)

        idx = start_index + (i - 1)
        name = f"{prefix}{idx:0{digits}d}.png"
        imwrite_rgba(out_dir / name, out)

