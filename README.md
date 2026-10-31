# Crowd density early-warning system

Measuring indicators of dangerous crowd conditions at pedestrian bottlenecks,
from a fixed elevated camera. ENGG2112 group project, built and reported with
its negative results intact.

> **This system detects dangerous conditions. It does not predict crushes.**
> It reports whether measured density and crowd pressure currently resemble
> values documented before past flow breakdowns. It has no model of what
> happens next. A warning is a prompt for a human to look at a camera, not a
> decision. See [docs/limitations.md](docs/limitations.md).

**Where to look:** [results/README.md](results/README.md) for every finding and
its caveat; [docs/methods_draft.md](docs/methods_draft.md) for how they were
produced.

## How it works

```
video ──▶ detection ──▶ tracking ──▶ calibration ──▶ density ──▶ turbulence ──▶ warning
          (YOLOv8)      (ByteTrack)   (homography)   (people/m²)  (pressure)    (two signals)
          └── the only trained component ──┘ └──── deterministic Python ───────────────────┘
```

Only detection is a model. Density, crowd pressure and the warning rule are
plain arithmetic over its output, driven by config files with citations. That
separation keeps the warning logic auditable and lets the whole non-ML stack be
tested against exact ground truth.

Escalation needs **two** signals together: density in a crowded Fruin band
**and** crowd pressure rising, sustained for a dwell time. Either alone is a
poor indicator — a packed but calm platform is not dangerous, and a pressure
blip in a sparse crowd is noise.

### Two paths, because no real footage is calibrated

Density in people/m² needs a surveyed homography, and no public dataset of dense
crowds supplies one. Hence a second path that needs no survey:

| | calibrated | scene-relative |
|---|---|---|
| Units | people/m² | people in the region |
| Escalates when | density in a crowded Fruin band **and** pressure rising | occupancy above this scene's own normal **and** the region filling |
| Levels | `NORMAL → WATCH → WARNING` | `LEARNING → WATCH → ELEVATED` |
| Thresholds from | published Fruin bands | the clip's own first 15 seconds |
| Claims | conditions resemble values documented before past flow breakdowns | conditions are unusual **for this camera** |
| Demonstrated on | the fabricated clip | real CroHD footage |

**One component is trained, and it is not the baseline.** The detector is a
YOLOv8n fine-tuned on 1,187 CroHD images, and it is the part that transfers to a
new camera. The scene baseline is not a model: it is the **85th percentile** of
occupancy over the clip's first 15 seconds, recomputed for every video and
meaningless on any other camera — which is why that path reports "unusual here"
and never "dangerous". A percentile rather than a mean, because occupancy
exceeds its own mean about half the time. And occupancy alone never escalates
without a net inflow above 0.3 people/s, so a packed but stable crowd stays
normal.

## Results

Every number has a producing script and a caveat recorded in
[results/README.md](results/README.md).

**Training data determines the usable range, by a factor of 3.6×.** Three
models, identical evaluation on 482 held-out ShanghaiTech images (241,275
annotated heads), count ratio against point annotations:

| heads/image | SCUT-HEAD A | A + B | **CroHD** |
|---|---|---|---|
| 100–200 | 0.269 | 0.421 | **0.975** |
| 400–700 | 0.100 | 0.160 | **0.490** |
| 1200+ | 0.012 | 0.024 | **0.204** |
| usable ceiling (ratio ≥ 0.8) | never | never | **200 heads/frame** |

![Count ratio against crowding for all three models, log-log](results/recall/three_way.png)

**The failure was viewpoint**, diagnosed by looking at which heads were missed
rather than at the aggregate score: SCUT-HEAD's camera faces the students, so
the model learned faces from an elevated-front angle instead of heads from
above. Same two frames, same evaluation, different training set:

| SCUT-HEAD model — 15 and 20 detections | CroHD model — 52 and 63 |
|---|---|
| ![SCUT-HEAD model detects only forward-facing people](results/recall/diagnosis_scuthead_misses_backs_of_heads.jpg) | ![CroHD model detects heads from all angles](results/recall/diagnosis_crohd_finds_them.jpg) |

The backs and tops of heads — what an elevated camera mostly sees — are missed
entirely. A training-distribution failure, not a capacity one: both models are
the same 3.1M-parameter YOLOv8n.

**The measurement stack is validated against ground truth.** Counts to within
14% (sd 3.4 pp across four CroHD scenes, three training and one held out — the
held-out scene scores 0.869, above the four-scene mean); track-derived speeds to
within 2–5%; a declared 30.0 m² region recovered exactly by the homography.

**It escalates on real footage.** On HT21-03, a street crossing where people
accumulate at a light, the system runs `NORMAL → WATCH → ELEVATED` and back.
Inflow is nearly double outflow at escalation and has converged by de-escalation
— the flow-balance diagnostic doing the work, with no calibration and no
borrowed thresholds.

![Real footage escalating through NORMAL, WATCH and ELEVATED](results/demo/real_crossing_escalation.jpg)

HT21-03 is a detector **training** scene, so this demonstrates the occupancy,
flow-balance and escalation logic on real crowd video — none of which was fitted
to it — not that the detector generalises. That claim rests on the ShanghaiTech
evaluation above, on images the model never saw.

### Three things it cannot do

- **The occupancy forecast lost to a persistence baseline.** Negative skill. It
  is in the repository because the measurement is the result.
- **No critical density could be located**, because CroHD contains no flow
  breakdown. The fundamental diagram has no peak to find.
- **The warning layer cannot be validated.** No public dataset contains the
  condition it is designed to detect, so the full `NORMAL → WATCH → WARNING`
  escalation can only be shown on a clip we generated ourselves:

![Fabricated scenario escalating through NORMAL, WATCH and WARNING](results/demo/scenario_escalation.jpg)

> **This clip is fabricated.** Geometry, people and movement were generated by
> `scripts/make_scenario_clip.py` and the frames are watermarked. Detections are
> replayed from ground truth, so it exercises calibration, density, crowd
> pressure and the warning rule with no detector error. It is evidence about the
> logic, **not** about crowds. The real-footage result above is the one that
> says anything about the world.

## Limitations

Full account in [docs/limitations.md](docs/limitations.md). The three that most
constrain what the system can claim:

1. **Recall degrades with crowding**, and because density is a count,
   under-detection makes a dangerous crowd look *safer*. The usable ceiling is
   measured, not assumed: 200 heads/frame.
2. **Thresholds are inherited, not derived.** The Fruin bands are cited
   constants; `pressure_slope_threshold` was set by a sweep on the fabricated
   scenario, because no real footage contains the condition to tune against.
3. **No forecast.** The system reports present conditions only.

[docs/future_work.md](docs/future_work.md) sets out what data would change this,
including why training end-to-end on video is the wrong answer even with it.

## Running it

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pytest -q
```

**152 tests**, none of which need a model, a dataset or any footage.

To see the whole pipeline with nothing downloaded — it generates its own clip:

```bash
.venv/Scripts/python.exe scripts/make_scenario_clip.py
.venv/Scripts/python.exe scripts/run_pipeline.py --config configs/scenario.yaml --video data/raw/scenario_bottleneck.mp4 --out runs/scenario
```

Expect `NORMAL → WATCH → WARNING`, one sustained episode from 27.7 s to 53.1 s
peaking at 3.93 people/m², a stand-down when the crowd stays dense but stops
deteriorating, and an annotated `runs/scenario/overlay.mp4`. The stand-down is
the point of the two-signal rule: a density-only threshold would still be firing
at 53 s.

GPU inference needs the CUDA build of torch (match the index to your driver).
Measured on an RTX 3050 6 GB, yolov8n at `imgsz=1280`: **30.8 ms per frame on
GPU against 143.4 ms on CPU**.

```bash
.venv/Scripts/python.exe -m pip install torch==2.11.0+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Reproducing the measured results

The datasets are **not** in this repository — licence, privacy and size. Sources,
citations, exact split and stride: [docs/datasets.md](docs/datasets.md).

```bash
# convert CroHD to YOLO format, holding out a whole sequence
python scripts/convert_mot_to_yolo.py --root /path/to/HT21 --out /path/to/ht21_yolo --test-seq HT21-04 --stride 4
# train, or skip this and download best.pt from the release below
python scripts/train_head_detector.py --data /path/to/ht21_yolo --weights yolov8n.pt --epochs 10 --imgsz 1280 --batch 4 --name heads_crohd
# validate the measurement stack against ground-truth tracks
python scripts/analyse_crohd.py --root /path/to/HT21 --weights runs/detect/heads_crohd/weights/best.pt --out runs/crohd
# the real-footage escalation demo
python scripts/demo_uncalibrated.py --root /path/to/HT21 --sequence HT21-03 --weights runs/detect/heads_crohd/weights/best.pt --out runs/demo
```

The trained model and the fabricated clip are attached to the
[v1.0 release](https://github.com/weiren12345/ENGG2112-TUT06-62/releases/tag/v1.0)
rather than committed, since both are reproducible and binaries in git history
are permanent:

```bash
mkdir -p runs/detect/heads_crohd/weights
curl -L -o runs/detect/heads_crohd/weights/best.pt https://github.com/weiren12345/ENGG2112-TUT06-62/releases/download/v1.0/best.pt
```

### Your own footage

The calibrated path is retained and tested even though no available footage
could exercise it, because calibration is what makes thresholds
**transferable**: pixels belong to one camera at one zoom, whereas people/m²
means the same thing everywhere. Measure a rectangle on the ground and click it,
pick the monitored region, copy `configs/camera_example.yaml`, then run
`scripts/run_pipeline.py`. Full procedure in
[docs/calibration.md](docs/calibration.md).

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
data/, runs/  gitignored: footage, weights, per-frame CSVs, overlay video
```

## Licence and credits

Code, configuration and documentation: MIT, see [LICENSE](LICENSE).

The datasets are third-party and are not redistributed here; dataset frames
appearing inside the figures in `results/` are reproduced for academic
illustration and credited in [results/README.md](results/README.md). Sources,
citations and terms: [docs/datasets.md](docs/datasets.md).
