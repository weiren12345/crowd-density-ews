# Datasets: what was used, where to get it, how to convert it

No dataset is redistributed in this repository. This file is the substitute:
enough provenance and tooling that the results can be reproduced from the
original sources.

Three reasons for that choice, any one of which would be sufficient:

1. **Licence.** All three datasets are third-party academic releases with their
   own terms. Permission to use them is not permission to republish them.
2. **Privacy.** CroHD and ShanghaiTech are real pedestrians who did not consent
   to appear here. `docs/ethics.md` argues that position for our own footage; it
   would be incoherent to ignore it for other people's.
3. **Size.** Tens of gigabytes. The repository is 11 MB.

What *is* committed: the conversion script, the exact split, the stride, and
every evaluation number the splits produced. Those are the parts that are ours.

---

## 1. The datasets

| | role | what it is |
|---|---|---|
| **CroHD (HT21)** | **training (final model)** and all real-footage evaluation | dense pedestrian crowds, Full HD video, elevated cameras, head boxes with track ids |
| **SCUT-HEAD** | training (runs 1 and 2 — the failed baseline) | classroom photographs, head boxes, camera facing the students |
| **ShanghaiTech Part A** | held-out cross-domain evaluation | still images of crowds, every head annotated as a point |

### Where to get them

| dataset | source | citation to use |
|---|---|---|
| CroHD / HT21 | MOTChallenge, "Head Tracking 21" — `motchallenge.net` | Sundararaman et al., *Tracking Pedestrian Heads in Dense Crowd*, CVPR 2021 |
| SCUT-HEAD | `github.com/HCIILAB/SCUT-HEAD-Dataset-Release` | the dataset release page's own requested citation |
| ShanghaiTech | the authors' release, or a mirror (ours came from a Kaggle copy) | Zhang et al., *Single-Image Crowd Counting via Multi-Column CNN*, CVPR 2016 |

> **VERIFY before submission.** Confirm each citation and each download URL
> against the primary source, and read each licence for the terms that apply to
> academic use. The URLs above are recorded as used, not verified as canonical.

### Why CroHD and not SCUT-HEAD

This is the project's central finding, not an implementation detail. SCUT-HEAD's
camera faces the students, so a model trained on it learns *faces from an
elevated-front angle*, not *heads from above*. Evaluated on real elevated
footage it misses the backs and tops of heads almost entirely — a count ratio of
0.269 where CroHD reaches 0.975. See `results/README.md` section 1 and the
diagnostic figures beside it.

Keep SCUT-HEAD in any reproduction: the comparison is the experiment.

---

## 2. Reproducing the training data

CroHD ships in MOT format (one `gt.txt` per sequence). Convert to YOLO:

```bash
python scripts/convert_mot_to_yolo.py \
    --root /path/to/HT21 \
    --out /path/to/ht21_yolo \
    --test-seq HT21-04 \
    --stride 4
```

Three things this does that matter for the numbers:

- **Scene-level holdout.** `--test-seq` holds out a whole sequence. At 25 fps a
  random frame split puts near-identical images on both sides of the split and
  reports a score that means nothing.
- **Temporal subsampling.** `--stride 4` keeps every fourth frame. Consecutive
  frames are near-duplicates; keeping all of them inflates the dataset without
  adding information.
- **Drops `conf=0` rows**, which in MOT format mark annotations to be ignored.

Result: 1,187 images carrying **253,376** head boxes.

Then train:

```bash
python scripts/train_head_detector.py \
    --data /path/to/ht21_yolo \
    --weights yolov8n.pt \
    --epochs 10 --imgsz 1280 --batch 4 \
    --name heads_crohd
```

`yolov8n.pt` downloads automatically from Ultralytics on first use. Expect
roughly 5m30s per epoch on an RTX 3050 6 GB. `--workers 0` is the default here
deliberately — Windows dataloader workers were the cause of several out-of-
memory failures — and it is also why the epoch time is what it is.

Mosaic augmentation is **off**. It composites four images into one canvas,
halving each object's scale; with heads already at ~17 px it pushes them below
what the stride-8 detection layer can resolve, and it allocates a `(2*imgsz)^2`
buffer per sample.

---

## 3. Reproducing the evaluation

Count ratio against ShanghaiTech point annotations:

```bash
python scripts/eval_point_recall.py \
    --root /path/to/ShanghaiTech/part_A \
    --weights runs/detect/heads_crohd/weights/best.pt \
    --out runs/recall_shanghai
```

> **A trap worth knowing about.** Some ShanghaiTech mirrors ship annotations
> scaled by 2x relative to the image dimensions. The symptom is quiet: every
> ground-truth point lands in the top-left quadrant, so recall looks
> catastrophic for reasons that have nothing to do with the model. The script
> calls `infer_scale()` per image and corrects it. We hit this, found it by
> checking `max_ann_x / width = 0.498`, and confirmed it visually —
> `results/recall/databug_shanghaitech_2x_scaling.jpg` is that check.

Measurement validation against CroHD ground-truth tracks, the real-footage
demo, the fundamental diagram and the flow/forecast study all take the same
`--root /path/to/HT21` argument. See each script's docstring, and
`results/README.md` for which command produced which number.

---

## 4. Trained weights

Not committed — `.pt` files are gitignored, because binaries in git history are
permanent and these are reproducible from section 2.

The trained CroHD model (`best.pt`, 6.3 MB) is attached to the repository's
GitHub release instead, so the results can be reproduced without re-training.
Drop it at `runs/detect/heads_crohd/weights/best.pt`, which is the path every
script's example command expects.
