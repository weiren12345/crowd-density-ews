# Machine learning in this system

A complete account of what is learned, what is classical, and what is
deterministic. The separation is deliberate and is a design argument in its own
right: the learned component reads pixels, and nothing downstream of it makes a
decision it cannot justify.

---

## 1. The learned component: YOLOv8 head detection

The only neural network in the pipeline.

### Architecture

| | |
|---|---|
| Model | YOLOv8n (nano) |
| Type | single-stage, anchor-free object detector |
| Backbone | CSPDarknet with C2f (cross-stage partial) blocks |
| Neck | PANet-style feature pyramid, top-down and bottom-up |
| Head | decoupled, predicting at strides 8, 16 and 32 |
| Parameters | **3,157,200** pretrained (80 COCO classes) -> **3,138,003** fine-tuned (1 class) |
| Inference | 30.8 ms/frame at imgsz 1280 on an RTX 3050 6GB |

yolov8s (11,166,560 parameters) was obtained as a larger alternative but the
nano model proved sufficient once the training data was right - which is itself
a finding: the bottleneck was data, not capacity.

### Learning setup

**Transfer learning.** Every run initialises from COCO-pretrained weights, so
the backbone begins with general visual features (edges, texture, shape) and
only has to specialise. Training a 3M-parameter detector from scratch on ~1200
images would not converge usefully.

**Task.** Single-class detection. The classification head is reduced from 80
classes to 1 ("head"), making this a detection-and-localisation problem rather
than a recognition one. The system never identifies anybody.

**Loss function**, three terms combined:
- *Distribution Focal Loss* + *CIoU* for box regression
- *Binary cross-entropy* for objectness/classification
- Label assignment by the **Task-Aligned Assigner**, which matches predictions
  to ground truth using a combined classification-and-localisation score rather
  than IoU alone

**Optimiser.** SGD with momentum, Ultralytics defaults, with warmup and a
cosine-style learning-rate schedule. Early stopping via `patience`.

**Augmentation**, chosen deliberately rather than left at defaults:
- horizontal flip (p = 0.5)
- random scaling (+/- 30%)
- rotation, shear and perspective **disabled** - heads are upright in an
  elevated view, so those transforms teach poses that never occur
- **mosaic disabled.** Mosaic composites four images into one canvas, roughly
  halving each object's scale. With heads already at ~17 px median in
  SCUT-HEAD, mosaic pushes them below what the stride-8 detection layer can
  resolve. It also allocates a (2 x imgsz)^2 buffer per sample, which exhausted
  an 8 GB machine.

### Three training runs, as a controlled experiment

The single variable changed between runs is the training data.

| Run | Data | Images | Boxes | Epochs | imgsz |
|---|---|---|---|---|---|
| 1 | SCUT-HEAD Part A | 1,100 | ~30k | 15 | 1024 |
| 2 | SCUT-HEAD A + B | 2,730 | ~45k | 10 | 1024 |
| 3 | **CroHD** | 1,187 | **253,376** | 10 | 1280 |

Result, measured identically on 482 held-out ShanghaiTech images:

| heads/image | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| 100-200 | 0.269 | 0.421 | **0.975** |
| 1200+ | 0.012 | 0.024 | **0.204** |

This is a **domain-adaptation study**: Run 1 failed because SCUT-HEAD's camera
faces the students, so the model learned faces from an elevated-front angle
rather than heads from any angle. Run 3 fixed it with dense, elevated footage
showing mostly backs and tops of heads.

### Evaluation methodology

- **Scene-level holdout.** CroHD sequences are held out whole. At 25 fps a
  random frame split would put near-identical images on both sides and report a
  meaningless score.
- **Cross-domain testing.** Trained on one dataset, evaluated on another
  (ShanghaiTech), which is a far harder and more honest test than in-domain
  validation. Run 1's in-domain mAP50 of 0.938 collapsed to a 0.27 count ratio
  the moment it met different data.
- **Metrics.** mAP50 and mAP50-95 for detection; count ratio against hand or
  point annotations for the counting task, because density is a count and what
  matters is whether a person was found, not whether the box hugs them.
- **Validation against ground truth.** Counts to within 14% (sd 3.4pp across
  four unrelated scenes); track-derived speeds to within 2-5%.

---

## 2. Classical, not learned: multi-object tracking

**ByteTrack** associates detections across frames to produce identities. It is
frequently described as part of the "AI" stack, and it is worth being precise:
ByteTrack contains **no learned parameters**. It is

- a **Kalman filter** predicting each track's next position (classical state
  estimation), and
- **Hungarian assignment** on IoU to match predictions to new detections,
  with a two-stage pass that gives low-confidence detections a second chance.

The learning sits entirely in the detections it consumes. Tracking-by-detection
is the honest description: a classical algorithm operating on neural-network
output.

---

## 3. Not machine learning at all

Listed explicitly, because the boundary is a design argument rather than an
omission. A warning system should be able to explain itself, and everything
below can be audited line by line.

| Component | What it actually is |
|---|---|
| Calibration | projective geometry - a homography fitted by least squares to 4+ measured correspondences |
| Head-height correction | similar triangles: a radial contraction about the camera nadir |
| Density | arithmetic: count divided by a measured area |
| Optical flow | **Farneback** - classical polynomial expansion, no learned parameters |
| Local velocity variance | descriptive statistics over k-nearest neighbours |
| Crowd pressure | the product of two measured quantities |
| Level-of-service bands | cited constants (Fruin), not fitted |
| Warning rule | a deterministic state machine with dwell time and hysteresis |
| Flow balance | conservation of people, plus a least-squares slope |
| Scene baseline | a percentile of observed occupancy |

Linear least-squares fitting appears in two places (the homography, and the
occupancy trend). That is statistical estimation rather than machine learning in
any meaningful sense, and is listed here rather than claimed above.

---

## 4. ML methodology demonstrated

Beyond the model itself, the project exercises:

- **transfer learning** from a large pretrained backbone
- **domain adaptation**, diagnosed from error analysis and fixed by changing the
  training distribution
- **controlled ablation** - three runs, one variable
- **train / validation / test discipline**, with scene-level rather than random
  splits
- **cross-domain generalisation testing**
- **error analysis driving design**, i.e. looking at *which* detections were
  missed, not just the aggregate score
- **baseline comparison** - the occupancy forecast was measured against a
  persistence baseline and **lost**, which is reported rather than hidden
- **negative results**, reported with the numbers

---

## 5. What is deliberately not learned, and why

The warning decision is not a model output. No network emits "danger: 0.87".

Density, pressure and the thresholds are deterministic code with cited or
derived constants, so any alarm can be traced to the exact measurement and rule
that produced it. For a safety system that is the correct trade: accountability
matters more than squeezing out accuracy, and an operator asked to act on an
alarm should be able to see why it fired.

Placing the learning where it genuinely helps - reading pixels - and keeping it
out of where it would obscure responsibility is a defensible architecture, and
worth stating as a design decision rather than leaving implicit.
