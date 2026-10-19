"""Convert MOT-format head annotations (CroHD / HT21) to YOLO format.

CroHD is the dataset this project should have started with: dense crowds, Full
HD, elevated cameras, head boxes with track ids. It is the domain SCUT-HEAD was
a poor proxy for.

Three choices made here, all of which affect the result and so are worth
stating in the writeup:

**Scene-level holdout.** Sequences are held out whole, never split by frame. At
25 fps consecutive frames are near-duplicates, so a random frame split would
put almost-identical images in train and test and report a score that means
nothing. Holding out an entire scene measures what we actually care about:
generalisation to a camera the model has never seen.

**Temporal subsampling.** Neighbouring frames carry almost no new information.
Taking every Nth frame cuts redundancy, training time and overfitting at once.

**Only images we keep are copied.** Copying all 5741 frames would duplicate
several GB for nothing.

    python scripts/convert_mot_to_yolo.py \\
        --root C:/path/to/HT21 --out C:/path/to/ht21_yolo \\
        --test-seq HT21-04 --stride 4

MOT ground-truth columns:
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility

``conf`` of 0 marks an entry to be ignored during evaluation, so those rows are
dropped rather than trained on.
"""

from __future__ import annotations

import argparse
import configparser
import shutil
from collections import defaultdict
from pathlib import Path


def read_seqinfo(seq_dir: Path) -> tuple[int, int]:
    cfg = configparser.ConfigParser()
    cfg.read(seq_dir / "seqinfo.ini")
    s = cfg["Sequence"]
    return int(s["imWidth"]), int(s["imHeight"])


def read_gt(gt_path: Path, min_visibility: float) -> dict[int, list]:
    """frame -> [(x, y, w, h), ...] in pixels."""
    out: dict[int, list] = defaultdict(list)
    dropped_conf = dropped_vis = 0
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        frame = int(float(parts[0]))
        x, y, w, h = (float(v) for v in parts[2:6])
        if len(parts) >= 7 and float(parts[6]) == 0:
            dropped_conf += 1
            continue
        if len(parts) >= 9 and float(parts[8]) < min_visibility:
            dropped_vis += 1
            continue
        if w <= 0 or h <= 0:
            continue
        out[frame].append((x, y, w, h))
    return out, dropped_conf, dropped_vis


def to_yolo(box, width: int, height: int):
    """(x, y, w, h) pixels -> normalised (cx, cy, w, h), clipped to frame."""
    x, y, w, h = box
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return (
        ((x1 + x2) / 2) / width,
        ((y1 + y2) / 2) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def convert_sequence(seq_dir: Path, split_dir: Path, stride: int,
                     min_visibility: float) -> tuple[int, int]:
    width, height = read_seqinfo(seq_dir)
    gt, dropped_conf, dropped_vis = read_gt(seq_dir / "gt" / "gt.txt", min_visibility)

    img_out = split_dir / "images"
    lab_out = split_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)

    n_images = n_boxes = 0
    for frame in sorted(gt):
        if (frame - 1) % stride:
            continue
        src = seq_dir / "img1" / f"{frame:06d}.jpg"
        if not src.exists():
            continue
        stem = f"{seq_dir.name}_{frame:06d}"
        rows = []
        for box in gt[frame]:
            yolo = to_yolo(box, width, height)
            if yolo:
                rows.append("0 " + " ".join(f"{v:.6f}" for v in yolo))
        if not rows:
            continue
        shutil.copy2(src, img_out / f"{stem}.jpg")
        (lab_out / f"{stem}.txt").write_text("\n".join(rows), encoding="utf-8")
        n_images += 1
        n_boxes += len(rows)

    print(f"  {seq_dir.name}: {n_images} images, {n_boxes} boxes "
          f"({n_boxes/max(1,n_images):.0f}/frame)"
          + (f", dropped {dropped_conf} conf=0" if dropped_conf else "")
          + (f", {dropped_vis} below visibility" if dropped_vis else ""))
    return n_images, n_boxes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="HT21 directory containing train/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-seq", default="HT21-04",
                    help="sequence held out whole, as an unseen scene")
    ap.add_argument("--stride", type=int, default=4,
                    help="keep every Nth frame; 25 fps is heavily redundant")
    ap.add_argument("--test-stride", type=int, default=10)
    ap.add_argument("--min-visibility", type=float, default=0.0,
                    help="drop boxes below this visibility; 0 keeps every "
                         "annotated head, which is what a counting system wants")
    args = ap.parse_args()

    train_root = Path(args.root) / "train"
    if not train_root.is_dir():
        print(f"no train/ directory under {args.root}")
        return 1
    seqs = sorted(d for d in train_root.iterdir() if (d / "gt" / "gt.txt").exists())
    if not seqs:
        print(f"no sequences with ground truth under {train_root}")
        return 1
    if args.test_seq not in {s.name for s in seqs}:
        print(f"--test-seq {args.test_seq} not among {[s.name for s in seqs]}")
        return 1

    out = Path(args.out)
    print(f"train sequences: {[s.name for s in seqs if s.name != args.test_seq]}")
    print(f"held-out scene:  {args.test_seq}\n")

    print("train:")
    tot_i = tot_b = 0
    for seq in seqs:
        if seq.name == args.test_seq:
            continue
        i, b = convert_sequence(seq, out / "train", args.stride, args.min_visibility)
        tot_i += i
        tot_b += b
    print(f"  total: {tot_i} images, {tot_b} boxes")

    print("val (held-out scene):")
    vi, vb = convert_sequence(train_root / args.test_seq, out / "val",
                              args.test_stride, args.min_visibility)

    print(f"\nwrote {out}")
    print("NOTE: val is an entirely unseen scene, so its metrics measure")
    print("cross-scene generalisation rather than fit. Do not tune on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
