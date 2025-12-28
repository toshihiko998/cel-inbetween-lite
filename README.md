# cel-inbetween-lite

A lightweight "cel animation" inbetweening tool using OpenCV optical flow (Farneback).
It focuses on:
- warping-based interpolation (not simple dissolve)
- edge-aware compositing to reduce color bleeding
- line reinjection to keep line-art crisp
- simple occlusion detection to avoid blending in unreliable regions

## Install
```bash
pip install -e .
