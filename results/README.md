# Results index

Every measured finding, what it shows, the script that produced it, and the
caveat attached. Written to be the thing the report is drafted from, so the
numbers do not have to be reconstructed later.

Large artefacts (model weights, per-frame CSVs, overlay video) stay in `runs/`,
which is gitignored. Only summaries and figures live here.

---

## 1. Detector recall vs crowding

**Headline: the training set determines the usable range, by a factor of 3.6x.**

Identical evaluation for all three models: 482 ShanghaiTech Part A images,
241,275 annotated heads, confidence 0.25.

| heads/image | SCUT-HEAD A | A+B | CroHD |
|---|---|---|---|
| 100-200 | 0.269 | 0.421 | **0.975** |
| 200-400 | 0.180 | 0.288 | **0.740** |
| 400-700 | 0.100 | 0.160 | **0.490** |
| 1200+ | 0.012 | 0.024 | **0.204** |
| usable ceiling (ratio >= 0.8) | never | never | **200 heads/image** |

| file | what it is |
|---|---|
| `recall_scuthead_A.json` | Part A only, 15 epochs |
| `recall_scuthead_AB.json` | Part A + B, 10 epochs |
| `recall_crohd.json` | CroHD, 10 epochs |
| `three_way.png` | all three curves, log-log |
| `comparison_A_vs_AB.png` | the viewpoint-diversity experiment alone |
| `recall_scuthead_testsplit.json` | SCUT-HEAD's own test split: ratio ~0.95 at every density, which is why that dataset could not reveal the problem |

Produced by `scripts/eval_point_recall.py` and `scripts/eval_dataset_recall.py`.

**Caveats.** The 0-50 bin holds only 5 images; ignore it. Count ratio cannot
separate "found them all" from "missed fifty and invented fifty". No single
confidence threshold suits all densities - sparse scenes want 0.3, extreme
density wants 0.1 - and picking the best per bin would be selecting on the
evaluation data.

## 2. The diagnosis

| file | what it shows |
|---|---|
| `diagnosis_scuthead_misses_backs_of_heads.jpg` | SCUT-HEAD model: almost every detection is a forward-facing person. Trained on a classroom camera facing the students, it learned faces from an elevated-front angle, not heads from any angle |
| `diagnosis_crohd_finds_them.jpg` | same two images, CroHD model: 15 -> 52 and 20 -> 63 detections |
| `databug_shanghaitech_2x_scaling.jpg` | the Kaggle re-upload had been upscaled 2x without rescaling its point labels. Red is as-stored, green is x2. Used as-is, every result from that dataset would have been silently wrong |

## 3. Measurement validated against ground truth

**Headline: counting is biased low by a consistent 14%; speeds are within 2-5%.**

Four CroHD scenes, none of which the model trained on.

| sequence | count ratio | speed error | speed correlation |
|---|---|---|---|
| HT21-01 | 0.875 | 4.5% | 0.939 |
| HT21-02 | 0.886 | 2.8% | 0.996 |
| HT21-03 | 0.799 | 2.4% | 0.955 |
| HT21-04 | 0.869 | 4.6% | 0.722 |

Count ratio across scenes: **mean 0.857, sd 0.034**. A consistent bias is
correctable; an erratic one is not.

Counting without tracking does better still: **0.977 count ratio, 4% mean
absolute error** over 40 s of HT21-04 (`count_vs_truth_timeseries.png`). Use
`detect()` for counting and `track()` for velocity - ByteTrack discards
detections it cannot associate.

| file | what it is |
|---|---|
| `crohd_summary.json` | per-sequence counts, speeds, errors |
| `validation.png` | pipeline speed vs ground-truth speed, and the first fundamental-diagram look |
| `count_vs_truth_timeseries.png` | detected vs true count over 40 s |

Produced by `scripts/analyse_crohd.py`.

## 4. Fundamental diagram

**Headline: speed falls as density rises, in every scene with enough range -
but flow never turns over, so no critical density is locatable.**

From 1.18M person-observations of ground-truth tracks.

| sequence | slope (m/s per person/m2) | speed range |
|---|---|---|
| HT21-02 | -0.0198 | 0.41 -> 0.21 m/s |
| HT21-03 | -0.0136 | 0.71 -> 0.52 m/s |
| HT21-04 | -0.0063 | 0.35 -> 0.26 m/s |
| HT21-01 | +0.0100 | inconclusive: sparsest scene, too little density range |

Produced by `scripts/fundamental_diagram.py`.

**Caveats.** Density is a proxy resting on an assumed head width and a packing
constant: the shape and slope are sound, **the x-axis numbers are not - do not
quote them**. Scenes must be analysed separately; pooling them mixes "which
environment" with "how dense" and inverts the curve. Flow rises monotonically
because CroHD contains no breakdown, so the critical density is not in this
data to be found.

## 5. Flow balance

**Headline: the net occupancy rate is measurable; forecasting it is not
better than assuming nothing changes.**

Conservation check (occupancy change must equal entries minus exits):

| sequence | drift per 100 crossings |
|---|---|
| HT21-02 | **+0.5** - crossing counts sound |
| HT21-04 | **-12.5** - systematically biased outward |

The HT21-04 bias is structural: a person occluded inside the region is
forgotten by the tracker, returns as a first sighting (no entry counted), then
is counted leaving. Hence the net rate is taken from the occupancy trend, which
has no such failure mode.

Forecast skill against a persistence baseline, 2816 forecasts over 133 s:

| configuration | skill at 30 s |
|---|---|
| 30 s window, raw slope | -106% |
| 30 s window, sign-gated | -42% |
| 60 s window, sign-gated + 0.5 shrinkage | **-6%** |

**Extrapolation never wins.** Gating makes it nearly harmless, not useful.
Statistical significance was the wrong gate - a slope can be significant
without persisting. Report `net_rate` as a measurement; do not present
`time_to()` as a countdown.

Produced by `scripts/flow_forecast.py`.

## 6. End-to-end demonstration

Full stack on 133 s of HT21-02: detection, tracking, velocities, occupancy,
flow balance, and a scene-relative warning rule, uncalibrated.

| file | what it is |
|---|---|
| `summary.json` | settings and level counts |
| `overlay_still.jpg` | annotated frame with the live HUD |

The system reports **NORMAL throughout** (89% of frames, remainder spent
learning the baseline). That is the correct answer - HT21-02 is a steady-state
crowd where nothing unusual happens - and it is a true negative worth
reporting: a system that invented alarms on calm footage would be worse.

Escalation behaviour (NORMAL -> WATCH -> WARNING) is demonstrated separately on
the synthetic clip with exact ground truth, via `scripts/run_pipeline.py`.
No positive case exists on real footage because no public dataset contains one.

Produced by `scripts/demo_uncalibrated.py`.

---

## What none of this shows

- that the system detects or predicts a crush. No public dataset contains one,
  so the warning layer has never been tested against the event it is named for
- absolute density in people/m2 without a surveyed camera
- that any threshold is correct. The Fruin bands remain unverified against a
  primary source, and `pressure_slope_threshold` was never set
