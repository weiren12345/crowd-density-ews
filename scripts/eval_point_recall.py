"""Measure where detection recall collapses, against point-annotated crowds.

This is the headline experiment the project is built around. Crowd-counting
datasets (ShanghaiTech and relatives) annotate every head as a point, in scenes
far denser than anything that can be filmed or staged safely. That makes them
the only practical way to answer the question the whole system depends on:

    at what crowding level does the detector stop finding people?

It matters because density is a COUNT. When recall falls, density reads low -
so a dangerous crowd looks safer than a moderate one, and the warning quietly
stops firing exactly when it should. The failure is in the worst direction, and
the curve this script produces is what bounds it.

    python scripts/eval_point_recall.py \
        --root .../ShanghaiTech/part_A --weights runs/detect/heads_n/weights/best.pt \
        --out runs/recall_shanghai

What it CANNOT tell you:

* The x-axis is heads per image, not people/m^2. These datasets ship no
  calibration, so there is no area to divide by. Treat the bins as crowding.
* The images are photographs from arbitrary angles, not elevated surveillance.
  This bounds dense-crowd recall; it does not simulate a bottleneck.
* Point annotations mark head centres, so this is a COUNT ratio. It does not
  check that a detection corresponds to the same head a human marked.

Read scripts/eval_recall.py for the hand-counted version on your own footage,
which is the one that produces people/m^2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CONF_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
DEFAULT_EDGES = [0, 50, 100, 200, 400, 700, 1200, 100_000]


def load_points(mat_path: Path, scale: float) -> np.ndarray:
    """Head centres from a ShanghaiTech-style .mat, in image pixels."""
    from scipy.io import loadmat

    inner = loadmat(str(mat_path))["image_info"][0, 0][0, 0]
    pts = np.asarray(inner["location"], dtype=np.float64).reshape(-1, 2)
    return pts * scale


def infer_scale(points: np.ndarray, width: int, height: int) -> float:
    """Work out the factor between annotation coordinates and this image.

    Re-uploads of these datasets are frequently resized without rescaling the
    labels. Assuming 1.0 silently destroys the result, so the factor is
    inferred per image and reported, rather than trusted.
    """
    if len(points) == 0:
        return 1.0
    span_x = points[:, 0].max() / width
    span_y = points[:, 1].max() / height
    span = max(span_x, span_y)
    if span <= 0:
        return 1.0
    # Annotations should very nearly fill the frame. If they occupy about half
    # of it, the image was upscaled 2x and the labels were not.
    for candidate in (1.0, 2.0, 4.0):
        if 0.6 <= span * candidate <= 1.0001:
            return candidate
    return 1.0


def nn_distance(points: np.ndarray, sample: int = 400) -> float:
    """Median nearest-neighbour distance in pixels.

    How close heads sit in the IMAGE, which is what determines whether they
    occlude one another and whether the detector can resolve them at all -
    a more direct predictor of failure than the raw count.
    """
    n = len(points)
    if n < 2:
        return float("nan")
    pts = points if n <= sample else points[np.random.default_rng(0).choice(n, sample, False)]
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="part_A or part_B directory")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train_data,test_data",
                    help="the detector trained on neither, so both are held out")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-det", type=int, default=4000)
    ap.add_argument("--edges", default=",".join(str(e) for e in DEFAULT_EDGES))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import cv2
    from cdw.detect import HeadDetector

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = [float(e) for e in args.edges.split(",")]

    pairs = []
    for split in args.splits.split(","):
        gt_dir, img_dir = root / split / "ground-truth", root / split / "images"
        if not gt_dir.is_dir():
            continue
        for gt in sorted(gt_dir.glob("*.mat")):
            stem = gt.stem.replace("GT_", "")
            for cand in (img_dir / f"processed_{stem}.jpg", img_dir / f"{stem}.jpg"):
                if cand.exists():
                    pairs.append((gt, cand))
                    break
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        print(f"no image/ground-truth pairs under {root}")
        return 1
    print(f"{len(pairs)} images from {args.splits}")

    det = HeadDetector(weights=args.weights, conf=min(CONF_GRID), imgsz=args.imgsz,
                       device=args.device, max_det=args.max_det)

    rows, scales = [], []
    for i, (gt, img_path) in enumerate(pairs):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        pts = load_points(gt, 1.0)
        scale = infer_scale(pts, w, h)
        pts = pts * scale
        scales.append(scale)

        confs = np.array([d.conf for d in det.detect(img)])
        rows.append({
            "image": img_path.name,
            "true": int(len(pts)),
            "nn_px": nn_distance(pts),
            "width": w,
            "height": h,
            "confs": confs,
        })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(pairs)}", end="\r")

    uniq, cnts = np.unique(scales, return_counts=True)
    print(f"\nannotation scale factors applied: "
          f"{dict(zip(uniq.tolist(), cnts.tolist()))}")
    if len(uniq) > 1:
        print("  (mixed factors - inspect before trusting these numbers)")

    truths = np.array([r["true"] for r in rows], float)
    nn = np.array([r["nn_px"] for r in rows], float)
    print(f"{len(rows)} images, {int(truths.sum())} annotated heads")
    print(f"count/image: min {truths.min():.0f}, median {np.median(truths):.0f}, "
          f"max {truths.max():.0f}")

    report = {
        "root": str(root), "weights": args.weights, "imgsz": args.imgsz,
        "max_det": args.max_det, "n_images": len(rows),
        "total_true_heads": int(truths.sum()),
        "scale_factors": {str(k): int(v) for k, v in zip(uniq.tolist(), cnts.tolist())},
        "curve_by_conf": {},
    }

    for conf in CONF_GRID:
        dets = np.array([float((r["confs"] >= conf).sum()) for r in rows])
        bins = []
        for lo, hi in zip(edges, edges[1:]):
            m = (truths >= lo) & (truths < hi)
            if not m.any():
                continue
            bins.append({
                "bin": [lo, hi],
                "n_images": int(m.sum()),
                "mean_true": float(truths[m].mean()),
                "mean_detected": float(dets[m].mean()),
                "pooled_ratio": float(dets[m].sum() / truths[m].sum()),
                "median_nn_px": float(np.nanmedian(nn[m])),
                "capped_images": int((dets[m] >= args.max_det).sum()),
            })
        report["curve_by_conf"][str(conf)] = bins

    headline = report["curve_by_conf"]["0.25"]
    print("\nat conf 0.25:")
    print(f"  {'heads/img':>14} {'imgs':>5} {'true':>7} {'detected':>9} "
          f"{'ratio':>6} {'nn_px':>6}")
    for b in headline:
        lo, hi = b["bin"]
        label = f"{lo:g}-{hi:g}" if hi < 10_000 else f"{lo:g}+"
        print(f"  {label:>14} {b['n_images']:5d} {b['mean_true']:7.0f} "
              f"{b['mean_detected']:9.0f} {b['pooled_ratio']:6.3f} "
              f"{b['median_nn_px']:6.1f}")

    capped = sum(b["capped_images"] for b in headline)
    if capped:
        print(f"\nWARNING: {capped} images hit the {args.max_det} detection cap; "
              "raise --max-det or the tail is an artefact")

    ok = [b for b in headline if b["pooled_ratio"] >= 0.8]
    ceiling = max((b["bin"][1] for b in ok), default=None)
    report["usable_ceiling_heads_per_image"] = ceiling
    print(f"\nusable ceiling (ratio >= 0.8): "
          f"{ceiling if ceiling else 'below the lowest bin'} heads/image")

    (out_dir / "point_recall.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for conf in CONF_GRID:
            b = report["curve_by_conf"][str(conf)]
            if b:
                ax1.plot([x["mean_true"] for x in b], [x["pooled_ratio"] for x in b],
                         marker="o", label=f"conf {conf:.2f}")
        ax1.axhline(1.0, color="grey", lw=0.8, ls="--")
        ax1.axhline(0.8, color="crimson", lw=0.8, ls=":")
        ax1.set_xscale("log")
        ax1.set_xlabel("true heads per image (log scale)")
        ax1.set_ylabel("detected / true")
        ax1.set_title("Recall against crowding")
        ax1.set_ylim(0, 1.2)
        ax1.legend(fontsize=7)
        ax1.grid(alpha=0.3)

        b = report["curve_by_conf"]["0.25"]
        ax2.plot([x["median_nn_px"] for x in b], [x["pooled_ratio"] for x in b],
                 marker="s", color="crimson")
        ax2.axhline(0.8, color="grey", lw=0.8, ls=":")
        ax2.set_xlabel("median nearest-neighbour distance (pixels)")
        ax2.set_ylabel("detected / true")
        ax2.set_title("Recall against head separation in the image\n(conf 0.25)")
        ax2.set_ylim(0, 1.2)
        ax2.grid(alpha=0.3)

        fig.suptitle(f"Head detector vs point-annotated crowds — {root.name}")
        fig.tight_layout()
        fig.savefig(out_dir / "point_recall.png", dpi=150)
        print(f"wrote {out_dir/'point_recall.png'}")
    except ImportError:
        pass

    print(f"wrote {out_dir/'point_recall.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
