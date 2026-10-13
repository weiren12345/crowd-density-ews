"""Count ratio against crowding, measured on a labelled YOLO split.

This is the cheap rehearsal for the real study in eval_recall.py. It asks the
same question - does the detector miss more people as the scene gets busier? -
but uses a dataset's own labels instead of hand counts, so it costs nothing and
can be run the moment training finishes.

What it CANNOT tell you:

* The x-axis is heads per image, not people/m^2. Without a homography there is
  no area, so these bins are crowding, not density.
* A classroom at 80 heads is not a bottleneck at 5 people/m^2. Students are
  seated, separated and mutually non-occluding. The occlusion that drives the
  real recall collapse is largely absent from this data, so the curve here is
  an optimistic bound - the honest reading is "it is at least this bad".
* Dataset labels are themselves imperfect, and unlike the hand-count study
  there is no second annotator to bound that.

Use it to check the analysis machinery and to get an early steer. Do not put
its numbers in the writeup as the recall result.

    python scripts/eval_dataset_recall.py \
        --weights runs/detect/heads_n/weights/best.pt \
        --data C:/path/to/PartA --split test --out runs/recall_dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CONF_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
DEFAULT_EDGES = [0, 5, 10, 20, 30, 40, 60, 10_000]


def true_count(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    return sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, help="dataset root containing the split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default=None)
    ap.add_argument("--edges", default=",".join(str(e) for e in DEFAULT_EDGES))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from cdw.detect import HeadDetector

    root = Path(args.data)
    images_dir = root / args.split / "images"
    labels_dir = root / args.split / "labels"
    if not images_dir.is_dir():
        images_dir, labels_dir = root / "images" / args.split, root / "labels" / args.split
    if not images_dir.is_dir():
        print(f"no {args.split} split under {root}")
        return 1

    files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} images in {images_dir}")

    edges = [float(e) for e in args.edges.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One low-confidence pass; the sweep is then done offline on stored scores.
    det = HeadDetector(weights=args.weights, conf=min(CONF_GRID), imgsz=args.imgsz,
                       device=args.device)

    truths, scores = [], []
    for i, f in enumerate(files):
        import cv2
        img = cv2.imread(str(f))
        if img is None:
            continue
        truths.append(true_count(labels_dir / (f.stem + ".txt")))
        scores.append(np.array([d.conf for d in det.detect(img)]))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)}", end="\r")

    truths = np.array(truths, float)
    print(f"\nprocessed {len(truths)} images, {int(truths.sum())} labelled heads")

    report = {"weights": args.weights, "split": args.split, "imgsz": args.imgsz,
              "n_images": len(truths), "curve_by_conf": {}}

    for conf in CONF_GRID:
        dets = np.array([float((s >= conf).sum()) for s in scores])
        rows = []
        for lo, hi in zip(edges, edges[1:]):
            m = (truths >= lo) & (truths < hi)
            if not m.any():
                continue
            rows.append({
                "bin": [lo, hi],
                "n_images": int(m.sum()),
                "mean_true": float(truths[m].mean()),
                "mean_detected": float(dets[m].mean()),
                "pooled_ratio": float(dets[m].sum() / truths[m].sum()) if truths[m].sum() else float("nan"),
            })
        report["curve_by_conf"][str(conf)] = rows

    headline = report["curve_by_conf"]["0.25"]
    print("\nat conf 0.25 — heads per image vs detected/true:")
    print(f"  {'bin':>12}  {'images':>6}  {'true':>7}  {'detected':>8}  {'ratio':>6}")
    for r in headline:
        lo, hi = r["bin"]
        label = f"{lo:g}-{hi:g}" if hi < 1000 else f"{lo:g}+"
        print(f"  {label:>12}  {r['n_images']:6d}  {r['mean_true']:7.1f}  "
              f"{r['mean_detected']:8.1f}  {r['pooled_ratio']:6.3f}")

    (out_dir / "dataset_recall.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for conf in CONF_GRID:
            rows = report["curve_by_conf"][str(conf)]
            if rows:
                ax.plot([r["mean_true"] for r in rows],
                        [r["pooled_ratio"] for r in rows],
                        marker="o", label=f"conf {conf:.2f}")
        ax.axhline(1.0, color="grey", lw=0.8, ls="--")
        ax.axhline(0.8, color="crimson", lw=0.8, ls=":")
        ax.set_xlabel("true heads per image (dataset labels)")
        ax.set_ylabel("detected / true")
        ax.set_title("Count ratio vs crowding — SCUT-HEAD test split\n"
                     "(seated, separated subjects: an optimistic bound)")
        ax.set_ylim(0, 1.3)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "dataset_recall.png", dpi=150)
        print(f"\nwrote {out_dir/'dataset_recall.png'}")
    except ImportError:
        pass

    print(f"wrote {out_dir/'dataset_recall.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
