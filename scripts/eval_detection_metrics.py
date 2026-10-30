"""Standard detection metrics for every trained model, on one held-out scene.

Count ratio answers "were the people found". It cannot answer "were the boxes
any good", and it cannot be computed against the box-annotated CroHD data at
all. This script fills that gap: precision, recall, F1 and mAP for each model,
scored identically.

Two operating points are reported. `max_f1` sweeps confidence from 0.001 and
takes the point maximising F1, which is what Ultralytics reports by default.
`conf_0.25` is the threshold the pipeline actually runs at, and is the honest
one to quote for deployed behaviour.

IMPORTANT, on reading the output: the held-out CroHD scene is the CroHD model's
*validation* set - the training framework selected its checkpoint on it - so
that model's column is an upper bound, not an independent result. For the
SCUT-HEAD models the same images are entirely unseen. The unbiased comparison
is scripts/eval_point_recall.py on ShanghaiTech, which no model trained on and
none was selected against.

`workers=0` is not a performance choice. On Windows the dataloader spawns
rather than forks, so each worker re-imports this module; without it (and the
__main__ guard below) the validation recursively spawns processes until it
dies.

Usage:
    python scripts/eval_detection_metrics.py --data configs/ht21.yaml \\
        --out results/detection
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODELS = {
    "SCUT-HEAD A": "runs/detect/heads_n/weights/best.pt",
    "SCUT-HEAD A+B": "runs/detect/heads_nAB/weights/best.pt",
    "CroHD": "runs/detect/heads_crohd/weights/best.pt",
}

OPERATING_POINTS = (("max_f1", 0.001), ("conf_0.25", 0.25))


def evaluate(weights: Path, data: Path, imgsz: int, max_det: int,
             iou: float, conf: float, run_dir: Path, name: str) -> dict:
    """One validation pass. Returns the summary metrics."""
    result = YOLO(str(weights)).val(
        data=str(data), imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
        workers=0, plots=False, verbose=False,
        project=str(run_dir), name=name, exist_ok=True,
    )
    box = result.box
    precision, recall = float(box.mp), float(box.mr)
    denom = precision + recall
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / denom, 4) if denom else 0.0,
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/ht21.yaml",
                    help="dataset yaml; its val split is what gets scored")
    ap.add_argument("--out", default="results/detection")
    ap.add_argument("--runs", default=None,
                    help="where to put Ultralytics run dirs (default: <out>/runs)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max-det", type=int, default=3000,
                    help="Ultralytics defaults to 300, which silently truncates "
                         "dense frames and looks exactly like poor recall")
    args = ap.parse_args()

    data = (ROOT / args.data).resolve()
    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.runs).resolve() if args.runs else out / "runs"

    missing = [n for n, p in DEFAULT_MODELS.items() if not (ROOT / p).exists()]
    if missing:
        print("missing weights for: " + ", ".join(missing))
        print("see docs/datasets.md - best.pt is on the GitHub release")
        return 1

    rows: dict[str, dict] = {}
    for name, rel in DEFAULT_MODELS.items():
        rows[name] = {}
        for label, conf in OPERATING_POINTS:
            rows[name][label] = evaluate(
                ROOT / rel, data, args.imgsz, args.max_det, args.iou, conf,
                run_dir, f"{name}_{label}".replace(" ", "_"),
            )
            m = rows[name][label]
            print(f"{name:16s} {label:10s} "
                  f"P={m['precision']:.3f} R={m['recall']:.3f} "
                  f"F1={m['f1']:.3f} mAP50={m['mAP50']:.3f}", flush=True)

    summary = {
        "val_split": str(data),
        "imgsz": args.imgsz,
        "iou": args.iou,
        "max_det": args.max_det,
        "caveat": ("the val split is the CroHD model's checkpoint-selection set, "
                   "so its figures are an upper bound; for the SCUT-HEAD models "
                   "these images are unseen"),
        "models": rows,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2),
                                      encoding="utf-8")
    print("\nwrote", out / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
