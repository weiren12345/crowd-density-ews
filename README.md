# Crowd density early-warning system

Forecasting indicators of dangerous crowd conditions at pedestrian
bottlenecks, from a fixed elevated camera.

> **This system detects dangerous conditions. It does not predict crushes.**
> It reports whether measured density and crowd pressure currently resemble
> values documented before past flow breakdowns. It has no model of what
> happens next. A warning is a prompt for a human to look at a camera, not a
> decision. See [docs/limitations.md](docs/limitations.md).

## How it works

```
video ──▶ detection ──▶ tracking ──▶ calibration ──▶ density ──▶ turbulence ──▶ warning
          (YOLOv8)      (ByteTrack)   (homography)   (people/m²)  (pressure)    (two signals)
          └── the only learned component ──┘ └──── deterministic Python ────────────────────┘
```

Only detection is a model. Density, crowd pressure and the warning rule are
plain arithmetic over its output, driven by config files with citations. That
separation is deliberate: it keeps the warning logic auditable, and it lets the
whole non-ML stack be tested against exact ground truth.

The warning rule needs **two** signals together: density in a crowded
level-of-service band **and** crowd pressure rising, sustained for a dwell
time. Either alone is a poor indicator — a packed but calm platform is not
dangerous, and a pressure blip in a sparse crowd is noise.

## Status

| Stage | State | Where |
|---|---|---|
| 1. Detection | **Done and tested** against pretrained COCO weights | `src/cdw/detect.py` |
| 2. Tracking | **Done and tested**, ids hold across a panning sequence | `src/cdw/detect.py` |
| 3. Calibration | **Done and tested**, incl. head-height correction | `src/cdw/calibrate.py` |
| 4. Density | **Done and tested** | `src/cdw/density.py` |
| 5. Turbulence | **Done and tested**, flow fallback included | `src/cdw/turbulence.py` |
| 6. Warning | **Done and tested** | `src/cdw/warning.py` |
| Recall study | Tooling written, not yet run | `scripts/`, `tools/` |
| Training | Script written, not yet run | `scripts/train_head_detector.py` |

Stages 3–6 are proven end to end on a synthetic clip with exact ground truth —
no model involved, so any wrong number there is our arithmetic. Stages 1–2 are
verified against pretrained COCO weights on a real image. 103 tests pass; the
model-dependent ones skip cleanly if `yolov8n.pt` is absent.

What is **not** yet tested: the head detector (not trained), and the whole
pipeline on real pedestrian footage. Those are the next two milestones.

## Quickstart

```bash
python -m venv --system-site-packages .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

For GPU inference, install the CUDA build of torch instead of the CPU one
(pick the index matching your driver):

```bash
.venv/Scripts/python.exe -m pip install torch==2.11.0+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128
```

Measured on an RTX 3050 6GB laptop GPU, yolov8n at `imgsz=1280`: **30.8 ms per
frame on GPU against 143.4 ms on CPU**, so roughly real-time locally. Training
still belongs on Colab - 6 GB will not fit `imgsz=1280` at a useful batch size.

Prove the pipeline with no model and no footage:

```bash
.venv/Scripts/python.exe scripts/make_synthetic_clip.py
.venv/Scripts/python.exe scripts/run_pipeline.py --config configs/synthetic.yaml --video data/raw/synthetic_bottleneck.mp4 --out runs/synthetic
```

The scripted scenario climbs from 1.3 to 4.3 people/m² in a 6 m² region while
flow breaks down. Expect `NORMAL → WATCH → WARNING`, one sustained episode, and
an annotated `runs/synthetic/overlay.mp4`.

Run the tests:

```bash
.venv/Scripts/python.exe -m pytest -q
```

## Working with real footage

1. **Calibrate.** Measure a rectangle on the ground, then click it.
   Full procedure, including what to record for the writeup, in
   [docs/calibration.md](docs/calibration.md).

   ```bash
   python tools/pick_points.py --mode calibration --video data/raw/site1.mp4 --width 4.0 --height 3.0 --out configs/calibration_site1.json
   ```

2. **Pick the region.**

   ```bash
   python tools/pick_points.py --mode region --video data/raw/site1.mp4 --calibration configs/calibration_site1.json
   ```

3. **Configure.** Copy `configs/camera_example.yaml` to `configs/site1.yaml`
   and paste in the calibration path and polygon.

4. **Run against pretrained weights first.** `yolov8n.pt` with
   `keep_classes: [0]` gives COCO person boxes; `box_bottom` puts the projected
   point at the feet, where the homography is valid. This proves the pipeline
   before any training.

5. **Then train.** `scripts/train_head_detector.py`, on Colab — see
   [docs/limitations.md](docs/limitations.md) section 7 on the domain gap.
   After switching to head weights, configure the height correction
   (`docs/calibration.md` section 5).

## The recall study

The headline limitation is that detector recall collapses above roughly
4–5 people/m² — exactly the danger regime — and because density is a count,
under-detection makes a dangerous crowd look *safer*. The project measures this
rather than hiding it:

```bash
python scripts/sample_frames.py --config configs/site1.yaml --video data/raw/site1.mp4 --out data/frames/site1 --per-bin 20
python tools/count_heads.py --manifest data/frames/site1/manifest.json --annotator alice
python tools/count_heads.py --manifest data/frames/site1/manifest.json --annotator bob --limit 30
python scripts/eval_recall.py --frames data/frames/site1 --out runs/recall_site1
```

Hand counting is blind: no model output is shown, frames are shuffled per
annotator, and time per frame is recorded. The second annotator on a subset
gives inter-annotator agreement, which bounds how much of the resulting curve
is real. The output defines a **usable ceiling** — the density above which the
system's own numbers cannot be trusted — and optionally fits a debiasing
correction.

## Layout

```
configs/      camera configs, calibration files, thresholds (with citations)
src/cdw/      the library: one module per pipeline stage
scripts/      pipeline runner, synthetic clip, training, recall analysis
tools/        interactive: point picking, blind hand counting
tests/        88 tests, no model or footage required
docs/         limitations, calibration, ethics, references
data/         gitignored: footage, sampled frames, annotations
runs/         gitignored: CSVs, overlays, reports
```

## Before submission

- [ ] Verify the Fruin band edges against a primary source — see the VERIFY
      block in `configs/thresholds.yaml` and [docs/references.md](docs/references.md).
- [ ] Verify the crowd-pressure definition against the paper, and state any
      deviation.
- [ ] Set `pressure_slope_threshold` from your own calm footage and record the
      procedure (`docs/calibration.md` section 6).
- [ ] Confirm the ethics position with the unit coordinator in writing
      ([docs/ethics.md](docs/ethics.md)).
- [ ] Report calibration reprojection error alongside every density figure.
- [ ] State the usable ceiling from the recall study as a headline result.
- [ ] Pin `ultralytics` and `opencv-python` versions once results are final.
