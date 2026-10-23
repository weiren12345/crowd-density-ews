# Crowd density early-warning system

Measuring indicators of dangerous crowd conditions at pedestrian bottlenecks,
from a fixed elevated camera.

> **This system detects dangerous conditions. It does not predict crushes.**
> It reports whether measured density and crowd pressure currently resemble
> values documented before past flow breakdowns. It has no model of what
> happens next. A warning is a prompt for a human to look at a camera, not a
> decision. See [docs/limitations.md](docs/limitations.md).

ENGG2112 group project. Built, measured, and reported with its negative results
intact.

## How it works

```
video ──▶ detection ──▶ tracking ──▶ calibration ──▶ density ──▶ turbulence ──▶ warning
          (YOLOv8)      (ByteTrack)   (homography)   (people/m²)  (pressure)    (two signals)
          └── the only learned component ──┘ └──── deterministic Python ────────────────────┘
```

Only detection is a model. Density, crowd pressure and the warning rule are
plain arithmetic over its output, driven by config files with citations. That
separation is deliberate: it keeps the warning logic auditable, and it lets the
whole non-ML stack be tested against exact ground truth. Full account in
[docs/ml_components.md](docs/ml_components.md).

The warning rule needs **two** signals together: density in a crowded
level-of-service band **and** crowd pressure rising, sustained for a dwell
time. Either alone is a poor indicator — a packed but calm platform is not
dangerous, and a pressure blip in a sparse crowd is noise.

## Headline results

Every number below has a producing script and a caveat recorded in
[results/README.md](results/README.md).

**Training data determines the usable range, by a factor of 3.6×.** Identical
evaluation of three models on 482 held-out ShanghaiTech images (241,275
annotated heads), count ratio against point annotations:

| heads/image | SCUT-HEAD A | A + B | **CroHD** |
|---|---|---|---|
| 100–200 | 0.269 | 0.421 | **0.975** |
| 400–700 | 0.100 | 0.160 | **0.490** |
| 1200+ | 0.012 | 0.024 | **0.204** |
| usable ceiling (ratio ≥ 0.8) | never | never | **200 heads/frame** |

The failure was **viewpoint**, diagnosed by looking at which heads were missed
rather than at the aggregate score: SCUT-HEAD's camera faces the students, so
the model learned faces from an elevated-front angle instead of heads from
above. See the diagnostic figures in `results/recall/`.

**The measurement stack is validated against ground truth.** Counts to within
14% (sd 3.4 pp across four unrelated CroHD scenes); track-derived speeds to
within 2–5%; a declared 30.0 m² region recovered exactly by the homography.

**It escalates on real footage.** On HT21-03, a street crossing where people
accumulate at a light, the system goes `NORMAL → WATCH → ELEVATED` and back as
inflow and outflow converge — real model, real detections, real crowd, no
calibration.

**And three things it cannot do**, reported rather than buried:

- The occupancy forecast was measured against a persistence baseline and
  **lost**. Negative skill. It is in the repository because the measurement is
  the result.
- **No critical density could be located**, because CroHD contains no flow
  breakdown. The fundamental diagram has no peak to find.
- **The warning layer cannot be validated.** No public dataset contains the
  condition it is designed to detect. The escalation demonstration at
  `results/demo/scenario_escalation.jpg` runs on a **fabricated** clip —
  watermarked on every frame — and is evidence about the logic, not about
  crowds.

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pytest -q
```

**152 tests**, none of which need a model, a dataset or any footage. The
model-dependent ones skip cleanly if weights are absent.

For GPU inference, install the CUDA build of torch (pick the index matching your
driver):

```bash
.venv/Scripts/python.exe -m pip install torch==2.11.0+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128
```

Measured on an RTX 3050 6 GB laptop GPU, yolov8n at `imgsz=1280`: **30.8 ms per
frame on GPU against 143.4 ms on CPU**.

### See it work without downloading anything

Both of these generate their own footage and replay detections from ground
truth, so they exercise calibration, density, crowd pressure and the warning
rule with no detector error and no dataset:

```bash
.venv/Scripts/python.exe scripts/make_scenario_clip.py
.venv/Scripts/python.exe scripts/run_pipeline.py --config configs/scenario.yaml --video data/raw/scenario_bottleneck.mp4 --out runs/scenario
```

Expect `NORMAL → WATCH → WARNING`, one sustained episode from 27.7 s to 53.1 s
peaking at 3.93 people/m², a stand-down when the crowd stays dense but stops
deteriorating, and an annotated `runs/scenario/overlay.mp4`.

The stand-down is the point of the two-signal rule: a density-only threshold
would still be firing at 53 s.

## Reproducing the measured results

The datasets are **not** in this repository — licence, privacy and size.
[docs/datasets.md](docs/datasets.md) has the sources, the citations, the exact
split and stride, and the conversion and training commands. The short version:

```bash
# 1. convert CroHD to YOLO format, holding out a whole sequence
python scripts/convert_mot_to_yolo.py --root /path/to/HT21 --out /path/to/ht21_yolo --test-seq HT21-04 --stride 4

# 2. train (or skip this and download best.pt from the release)
python scripts/train_head_detector.py --data /path/to/ht21_yolo --weights yolov8n.pt --epochs 10 --imgsz 1280 --batch 4 --name heads_crohd

# 3. validate the measurement stack against ground-truth tracks
python scripts/analyse_crohd.py --root /path/to/HT21 --weights runs/detect/heads_crohd/weights/best.pt --out runs/crohd

# 4. the real-footage escalation demo
python scripts/demo_uncalibrated.py --root /path/to/HT21 --sequence HT21-03 --weights runs/detect/heads_crohd/weights/best.pt --out runs/demo
```

The trained model (6.3 MB) is attached to the GitHub release rather than
committed, since it is reproducible from step 2 and binaries in git history are
permanent. Put it at `runs/detect/heads_crohd/weights/best.pt`.

## Working with your own footage

The calibrated path is retained and tested even though no available footage
could use it, because calibration is what makes thresholds **transferable**:
pixels are specific to one camera at one zoom, whereas people/m² means the same
thing everywhere.

1. **Calibrate.** Measure a rectangle on the ground, then click it. Full
   procedure in [docs/calibration.md](docs/calibration.md).

   ```bash
   python tools/pick_points.py --mode calibration --video data/raw/site1.mp4 --width 4.0 --height 3.0 --out configs/calibration_site1.json
   ```

2. **Pick the monitored region.**

   ```bash
   python tools/pick_points.py --mode region --video data/raw/site1.mp4 --calibration configs/calibration_site1.json
   ```

3. **Configure.** Copy `configs/camera_example.yaml`, paste in the calibration
   path and polygon.

4. **Run.** `scripts/run_pipeline.py --config configs/site1.yaml --video ...`

Report the reprojection error alongside every density figure. A calibration
fitted from no measured points reports `rms_m: null` by design — the code will
not claim a precision nobody measured.

## Layout

```
src/cdw/      the library: one module per pipeline stage
scripts/      pipeline runner, clip generators, training, evaluation studies
tools/        interactive: point picking, blind hand counting
tests/        152 tests, no model or footage required
configs/      camera configs, calibration, thresholds (with citations)
docs/         methods draft, limitations, calibration, ethics, datasets, ML account
results/      every measured finding, indexed — figures and JSON summaries only
data/         gitignored: footage and derived frames
runs/         gitignored: weights, per-frame CSVs, overlay video
```

Start with [results/README.md](results/README.md) for the findings and
[docs/methods_draft.md](docs/methods_draft.md) for how they were produced.

## Known limitations

Summarised in full in [docs/limitations.md](docs/limitations.md). The three that
most constrain what the system can claim:

1. **Detector recall degrades with crowding**, and because density is a count,
   under-detection makes a dangerous crowd look *safer*. The usable ceiling is
   measured, not assumed: 200 heads/frame.
2. **The thresholds are inherited, not derived.** The Fruin bands are cited
   constants; `pressure_slope_threshold` was set by a sweep on the fabricated
   scenario, because no real footage contains the condition to tune against.
3. **No forecast.** The system reports present conditions. The one forecast
   attempted does not beat persistence.

[docs/future_work.md](docs/future_work.md) sets out what data would change this
and how it would be used — including why training end-to-end on video is the
wrong answer even with the data.

## Licence and credits

Code, configuration and documentation: MIT, see [LICENSE](LICENSE).

The datasets are third-party and are not redistributed here; dataset frames
appearing inside the figures in `results/` are reproduced for academic
illustration and credited in [results/README.md](results/README.md). Sources,
citations and terms: [docs/datasets.md](docs/datasets.md).

## Before submission

- [ ] Verify the Fruin band edges against a primary source — see the VERIFY
      block in `configs/thresholds.yaml` and [docs/references.md](docs/references.md).
- [ ] Verify the crowd-pressure definition against the paper, and state any
      deviation.
- [ ] Verify each dataset citation, URL and licence ([docs/datasets.md](docs/datasets.md)).
- [ ] Replace the copyright holder in `LICENSE` with the group's names.
- [ ] Report calibration reprojection error alongside every density figure.
- [ ] State the usable ceiling from the recall study as a headline result.
- [ ] Pin `ultralytics` and `opencv-python` versions now that results are final.
