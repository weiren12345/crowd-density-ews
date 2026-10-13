"""Fine-tune YOLOv8 for single-class head detection on SCUT-HEAD.

Intended for Colab on a T4; a 6 GB laptop GPU will not fit imgsz=1280 at a
useful batch size, and CPU training is not viable.

    python scripts/train_head_detector.py --data /content/scut_head \
        --epochs 60 --batch 4 --imgsz 1280

--data should point at a directory laid out in Ultralytics' format:

    scut_head/
      images/train/*.jpg   labels/train/*.txt
      images/val/*.jpg     labels/val/*.txt

The script writes data.yaml itself so the class list cannot drift from what the
model was actually trained on.

Notes that matter for the writeup:

* SCUT-HEAD Part A is classroom surveillance. The elevated viewpoint
  transfers reasonably to our setting; the lighting, indoor scale and seated,
  regularly spaced subjects do not. Expect the domain gap to show up as a
  further recall drop on real pedestrian footage, on top of the density
  effect - and measure both separately (scripts/eval_recall.py) rather than
  reporting one number.
* imgsz=1280 is not a free choice. Heads in a crowded wide shot are a few
  dozen pixels; at 640 they fall below what the smallest anchor can resolve,
  and recall in dense frames collapses for reasons that have nothing to do
  with occlusion.
* Keep PartB in reserve as a held-out domain, not as extra training data.
  Training on both leaves nothing to test generalisation with.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


#: Directory layouts seen in the wild, as (train_images, val_images, test_images).
#: Roboflow exports use <split>/images; the Ultralytics docs use images/<split>.
#: Supporting both avoids duplicating hundreds of MB just to rename folders.
LAYOUTS = (
    ("train/images", "valid/images", "test/images"),
    ("train/images", "val/images", "test/images"),
    ("images/train", "images/val", "images/test"),
    ("images/train", "images/valid", "images/test"),
)


def detect_layout(root: Path):
    """Return (train, val, test) image dirs relative to root, or raise."""
    for train, val, test in LAYOUTS:
        if (root / train).is_dir() and (root / val).is_dir():
            return train, val, (test if (root / test).is_dir() else None)
    looked = "\n".join(f"  {t} + {v}" for t, v, _ in LAYOUTS)
    raise FileNotFoundError(
        f"no recognised YOLO layout under {root}. Looked for:\n{looked}"
    )


def write_data_yaml(root: Path, out: Path) -> Path:
    train, val, test = detect_layout(root)

    # A labels/ dir must sit beside each images/ dir, or Ultralytics will train
    # happily on zero annotations and report a meaningless mAP of 0.
    for rel in (train, val):
        labels = root / rel.replace("images", "labels")
        if not labels.is_dir():
            raise FileNotFoundError(f"missing labels directory {labels}")

    spec = {
        "path": str(root.resolve()),
        "train": train,
        "val": val,
        "names": {0: "head"},
    }
    if test:
        spec["test"] = test
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    print(f"layout: train={train}  val={val}  test={test or '(none)'}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="SCUT-HEAD root in YOLO layout")
    ap.add_argument("--weights", default="yolov8s.pt",
                    help="starting weights; yolov8n for speed, yolov8s for recall")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4,
                    help="at imgsz=1280 a T4 fits about 4; -1 lets Ultralytics autoscale")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None)
    ap.add_argument("--name", default="heads")
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument(
        "--workers", type=int, default=0,
        help="dataloader workers. On Windows each is a separate PROCESS that "
             "re-imports torch and reserves address space for the CUDA DLLs, "
             "so on a low-RAM machine even 2 can trigger "
             "'OSError [WinError 1455]: the paging file is too small'. 0 loads "
             "data in the main process: slower, but it is the reliable setting "
             "on 8 GB Windows. Raise it on a bigger machine or on Colab.",
    )
    ap.add_argument(
        "--mosaic", type=float, default=0.0,
        help="mosaic augmentation probability. Default OFF here, for two "
             "reasons: it composites 4 images into one canvas, roughly halving "
             "each object's scale - and SCUT-HEAD heads are already ~17 px "
             "median, so mosaic pushes them below what the finest detection "
             "stride can resolve. It also allocates a (2*imgsz)^2 buffer per "
             "sample, which is what exhausts RAM on an 8 GB machine.",
    )
    ap.add_argument("--data-yaml", default="configs/scut_head.yaml")
    args = ap.parse_args()

    from ultralytics import YOLO

    data_yaml = write_data_yaml(Path(args.data), Path(args.data_yaml))
    print(f"wrote {data_yaml}")

    # Ultralytics joins a RELATIVE project path onto its own configured
    # runs_dir, which silently nests results as runs/detect/runs/detect/<name>.
    # An absolute path keeps them where the caller asked for them.
    project = Path(args.project).resolve()

    model = YOLO(args.weights)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(project),
        name=args.name,
        patience=args.patience,
        # Heads are small and roughly upright in an elevated view, so the
        # defaults that help on COCO mostly hurt here.
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        mosaic=args.mosaic,
        scale=0.3,
        close_mosaic=10 if args.mosaic else 0,
        workers=args.workers,
    )

    metrics = model.val(data=str(data_yaml), imgsz=args.imgsz, device=args.device)
    print("\nvalidation:")
    try:
        print(f"  mAP50    {metrics.box.map50:.4f}")
        print(f"  mAP50-95 {metrics.box.map:.4f}")
        print(f"  recall   {metrics.box.mr:.4f}")
        print(f"  precision {metrics.box.mp:.4f}")
    except AttributeError:
        print(f"  {metrics}")

    # Ask the trainer where it actually wrote, rather than reconstructing the
    # path by hand and hoping the two agree.
    save_dir = Path(getattr(model.trainer, "save_dir", project / args.name))
    best = save_dir / "weights" / "best.pt"
    print(f"\nbest weights: {best}")
    print(f"exists: {best.exists()}")
    print()
    print("Validation mAP above is on the dataset you just trained on. If that")
    print("was SCUT-HEAD, it is evidence about classrooms, NOT about pedestrian")
    print("bottlenecks. Point configs/<site>.yaml at these weights, then run")
    print("scripts/eval_recall.py on your own footage to find out what it does")
    print("in the regime that matters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
