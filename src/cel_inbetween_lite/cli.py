
import argparse
from pathlib import Path
from .inbetween import inbetween_pair

def main():
    p = argparse.ArgumentParser(prog="cel-inbetween")
    sub = p.add_subparsers(dest="cmd", required=True)

    ib = sub.add_parser("inbetween", help="Generate inbetweens between two frames")
    ib.add_argument("--a", required=True, help="Path to keyframe A (png/jpg)")
    ib.add_argument("--b", required=True, help="Path to keyframe B (png/jpg)")
    ib.add_argument("--n", type=int, default=3, help="Number of inbetween frames")
    ib.add_argument("--out", required=True, help="Output directory")
    ib.add_argument("--prefix", default="", help="Optional filename prefix")
    ib.add_argument("--start-index", type=int, default=1, help="Start index for output numbering")
    ib.add_argument("--digits", type=int, default=4, help="Number of digits for numbering")

    # quality knobs (safe defaults)
    ib.add_argument("--edge-protect", type=float, default=6.0,
                    help="Edge protection radius (pixels). Higher reduces color bleeding but may create harder transitions.")
    ib.add_argument("--occ-th", type=float, default=1.5,
                    help="Occlusion threshold. Higher blends more; lower trusts less and avoids blending.")
    ib.add_argument("--line-strength", type=float, default=0.85,
                    help="Line reinjection strength (0..1). Higher keeps stronger lines.")
    ib.add_argument("--line-kernel", type=int, default=3,
                    help="Morph kernel size for line mask reinforcement (odd number).")
    ib.add_argument("--flow-scale", type=float, default=1.0,
                    help="Global scale applied to flow magnitude. >1 exaggerates motion, <1 dampens.")

    args = p.parse_args()

    if args.cmd == "inbetween":
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        inbetween_pair(
            a_path=Path(args.a),
            b_path=Path(args.b),
            n=args.n,
            out_dir=out_dir,
            prefix=args.prefix,
            start_index=args.start_index,
            digits=args.digits,
            edge_protect=args.edge_protect,
            occ_th=args.occ_th,
            line_strength=args.line_strength,
            line_kernel=args.line_kernel,
            flow_scale=args.flow_scale,
        )

if __name__ == "__main__":
    main()
