# Methods — draft

Covers the ML workflow, the justification for each technical choice, and how the
original work plan was followed or adapted. Every number traces to
`results/README.md`.

---

## 1. Data preparation

Three datasets were used, each verified before use rather than trusted. They
served different roles, and the distinction organises what follows: **one
trained the delivered model, one was superseded and retained as a control, and
one was never trained on at all.** Two others, named in the original proposal,
were examined and set aside; they are recorded below because the reason they
failed is the same constraint that shaped the rest of the project.

### CroHD / HeadTracking21 — the training data for the delivered model

CroHD supplies dense crowds from elevated cameras as video with head boxes and
track identities: four training sequences, 5,741 frames, **1,188,496 annotated
heads** at 1920×1080 and 25 fps. Two properties distinguish it. The cameras are
mounted above pedestrian traffic, so the annotated heads are predominantly backs
and tops rather than faces — what a deployed camera sees. And because it is
video with persistent identities, the same data supports both detection training
and the velocity measurements the crowd-pressure signal depends on; no
still-image dataset can do the second.

Test-sequence ground truth is withheld by the benchmark and its evaluation
server is closed, so the four training sequences were re-split locally.
Conversion from MOT to YOLO format made three decisions worth recording:

- **Scene-level holdout.** Sequences are held out whole. At 25 fps consecutive
  frames are near-duplicates, so a random frame split would place almost
  identical images on both sides of the divide and report a score measuring
  memorisation rather than generalisation.
- **Temporal subsampling.** Every fourth frame of the training sequences was
  retained, and every tenth of the held-out scene, neighbouring frames carrying
  almost no additional information. This yielded 1,187 training images
  containing 253,376 boxes, and 100 images from the held-out scene containing
  17,665.
- **Ignore flags respected.** MOT rows with a confidence field of 0 mark entries
  excluded from evaluation and were dropped rather than trained on. Boxes were
  clipped to the frame boundary.

### The two supporting datasets

SCUT-HEAD (Parts A and B) was the initial choice and was later superseded. It
was obtained as a Roboflow export in YOLO format: Part A 1,100 train / 400
validation / 500 test; Part B 1,630 / 466 / 233. Part A is classroom
surveillance from an elevated camera across three lecture rooms; Part B is
internet-sourced photography. Labels were verified as normalised
`class cx cy w h`, single class, with zero malformed rows across a 400-image
sample. Median head width measured 17 px against a median image width of roughly
1,070 px, which set the training resolution for these runs. Why it was
superseded is §2's subject; the models trained on it are retained deliberately,
as the control that gives the final result its meaning.

ShanghaiTech Part A became the test set and was never trained on. It holds 482
images and 241,275 heads annotated as points, at 33–3,138 people per image.
Verification found a defect that would have invalidated every result derived
from it: the redistribution had **upscaled every image 2× without rescaling the
point labels**. The maximum annotation x-coordinate sat at 0.498 of image width
(median across all 1,198 images in both parts), and overlaying the raw points
showed them clustered in the top-left quadrant. The evaluation script now infers
the scale factor per image and reports it rather than assuming 1.0.

### Considered in the proposal, not used

Two further datasets were named in the original proposal and tested against the
same requirements before being set aside. Both were proposed for the warning
layer rather than for training, and both failed on the same axis.

UMN offers staged surveillance sequences of panic and rapid dispersal, three
scenes across eleven segments. It was attractive because it is one of the few
public datasets containing anything resembling a crowd emergency. Three things
ruled it out. The behaviour is simulated: participants are instructed to flee,
and each runs out of frame. The scenes are sparsely populated, so the density
regime is wrong. And it carries no head annotations, so it can train nothing and
cannot support even a count-ratio measurement. A fourth and more specific
objection — that the UMN anomaly produces high speed but *low* velocity
variance, which is precisely what the crowd-pressure signal is built to read as
benign — is set out in §5, and is the reason the planned evaluation against it
was abandoned rather than merely deprioritised.

PETS 2009 offers real-world crowd flows, bottleneck navigation and evacuation
from a multi-camera outdoor setup, with scenario subsets for density estimation,
tracking and flow analysis. On paper this was the closest match to the
deployment scenario of anything considered, and it is the **only candidate that
supplies camera calibration**, published as Tsai model parameters per view. It
was set aside on density and annotation type. The sequences hold tens of
participants where the system targets hundreds, and the annotations are person
positions and full-body boxes — several of the available sets produced by third
parties for later workshops rather than by the dataset authors — rather than
head boxes. Body boxes are the wrong target in dense crowds, which is the same
reasoning §2 applies to SCUT-HEAD.

PETS is worth naming explicitly rather than dropping, because it is the one
case where the calibration gap discussed in §2 and §5 was solvable and the
density gap was not. That is the trade the project could not escape: the
datasets with physical units do not have the crowds, and the datasets with the
crowds do not have physical units.

### What was not available

No dataset was found containing the target condition. Footage of crowd crushes
exists, but it is not publicly released, for substantive reasons: it shows
people dying, victims cannot consent, and the material is held under legal
restriction. Even with access the scale would be small, since major incidents
are rare. This absence is the single most important constraint on the project
and shapes every section that follows.

---

## 2. Data strategy

The strategy changed twice, each time in response to measurement rather than
preference, and converged on CroHD.

SCUT-HEAD Part A was the initial choice, selected for being head-annotated,
elevated, and already in YOLO format. A detector trained on it reached mAP50
0.938 on its own validation split and recovered 27% of people on ShanghaiTech.
Inspecting *which* detections were missed showed why: almost every success was a
forward-facing person, almost every back-of-head a miss. SCUT-HEAD's camera
faces the students from the front of a classroom, so the model had learned
**faces from an elevated-front angle, not heads from any angle**, where a camera
above a bottleneck sees mostly backs and tops.

Part B confirmed the diagnosis. Adding it — internet-sourced and
viewpoint-diverse — roughly doubled recall at every density, which identified
viewpoint as the operative variable; that it remained unusable showed classroom
photography could not supply enough of it. This reversed the plan that had
reserved Part B as a held-out domain, ShanghaiTech having meanwhile become the
better one.

CroHD was the dataset that worked, and the selection was made against the
failure diagnosis rather than by convenience, requirement by requirement:

| Requirement | Why it is necessary | SCUT-HEAD | CroHD |
|---|---|---|---|
| Heads, not bodies | bodies are occluded in dense crowds | yes | yes |
| Camera above the crowd | determines whether faces or backs of heads are seen | partly — elevated but front-facing | **yes** |
| Genuinely dense scenes | the regime of interest | no — tens per frame | **yes — hundreds** |
| Box annotations | point labels cannot train a detector | yes | **yes** |
| Video with identities | required for velocity and crowd pressure | no — stills | **yes** |

Only CroHD satisfies all five. The viewpoint row had been underweighted at the
start, treated as roughly satisfied by "elevated" when the operative question was
what the camera *faces*. The density row matters almost as much: 253,376 boxes
across 1,187 images means the model sees heavily occluded, overlapping heads as
the normal case rather than the exception.

ShanghaiTech could never be training data, because point annotations cannot
train a box detector — which is what made it available as the test set.
Dense-crowd data is abundant in point form and scarce in box form, which limits
what any detection-based approach can learn. Two further properties were wanted
and not obtained. No candidate combined **surveyed calibration**, which gives
physical units, with the density the system targets: PETS 2009 publishes camera
calibration but holds tens of people per scene, while CroHD has the crowds and
no calibration. And no candidate at any density contained **deterioration
episodes**, the condition to be detected. The two gaps are separable:
calibration is an engineering requirement any deployment would satisfy, while
incident data is the genuinely hard problem. Both reappear in §5.

---

## 3. Model development

The architecture is YOLOv8n, the smallest model in the YOLOv8 family: 3,157,200
parameters pretrained, 3,138,003 after reduction to a single class.

Transfer learning was used throughout. All runs initialise from COCO-pretrained
weights, so the network begins with general visual features and need only
specialise; a 3M-parameter detector trained from scratch on ~1,200 images would
not converge usefully.

The model uses a single-class formulation, the classification head having been
reduced from 80 classes to one (`head`), so that it answers only "is this a
head?". It detects and localises; it has no mechanism to identify anyone, which
is relevant to the ethics position.

A larger model was tried and proved unnecessary. YOLOv8s (11,166,560 parameters)
was obtained as an alternative, but the nano model reached a 0.975 count ratio
once the training data was appropriate. **The limiting factor was data, not
capacity** — a result rather than an assumption.

Augmentations were chosen from the geometry of the problem rather than left at
their defaults. Rotation, shear and perspective were disabled, since an elevated
camera sees heads upright and these transforms teach poses that never occur.
Mosaic, which tiles four images into one canvas, was disabled because it roughly
halves object scale — with heads already at a median of 17 pixels, that pushes
them below what the network can resolve — and because its memory cost was
prohibitive. Horizontal flip and ±30% scaling were kept, both producing
configurations that genuinely occur.

Ultralytics caps predictions at 300 per image by default. Evaluation images
contain up to 3,138 heads, so the default silently truncated predictions and
produced what looked like a recall collapse but was a library setting. The cap
was raised and the behaviour covered by a regression test.

The learned component is deliberately narrow: one network reads pixels, and
everything after it — ground-plane projection, density, crowd pressure, the
warning rule — is deterministic. An alarm prompts a human to act, and that human
needs to see which measurement and which rule produced it. The arrangement also
costs less data, since learning over a few physical quantities needs far less of
it than learning the same relationships from raw pixels; §5 returns to this.

### The delivered model

| | |
|---|---|
| Initialisation | COCO-pretrained YOLOv8n |
| Training data | CroHD, 1,187 images, 253,376 head boxes |
| Classes | 1 (`head`) |
| Resolution | 1280 |
| Epochs | 10, batch 2 |
| Augmentation | horizontal flip, ±30% scale; mosaic, rotation, shear and perspective disabled |
| Parameters | 3,138,003 |
| Inference | 30.8 ms/frame at `imgsz=1280` (RTX 3050 6 GB), against 143.4 ms on CPU |

Two figures summarise what it achieved, and they answer different questions. On
the **held-out CroHD sequence** it reaches precision 0.763 and recall 0.674 — a
validation score, since the checkpoint was selected there. On **ShanghaiTech**,
which played no part in training or checkpoint selection, it recovers 97.5% of
people at 100–200 heads per image, against 26.9% for the Part A model and 42.1%
for Part A+B. That is the independent test: a 3.6× improvement from changing the
training distribution alone.

The operating point is confidence 0.25 with the detection cap at 3,000, and
this is the configuration behind every downstream result in the project.

---

## 4. Training, validation and testing

`imgsz=1280` keeps CroHD's 26-34 px heads at 17-23 px after downscaling, above
the smallest size the network can resolve; batch size fell from 8 to 2 as
instance counts rose, limited by system RAM (7.6 GB) rather than the GPU.

The three roles below are easy to conflate, and the distinction governs what
each number can be claimed to show:

| Role | Data | Size | Purpose |
|---|---|---|---|
| Train | HT21-01, 02, 03 | 1,187 images / 253,376 boxes | fits the weights |
| Validation | HT21-04 (whole scene) | 100 images / 17,665 boxes | **selects the retained checkpoint** |
| **Test** | **ShanghaiTech Part A** | **482 images / 241,275 points** | **independent evaluation** |

The split is scene-level, for the reason given in §1. HT21-04 is held out from
training but not from model selection — the framework keeps whichever epoch
scores best on it — so any figure from it is an upper bound, not an independent
result. ShanghaiTech played no part in fitting or selection, and is the only
genuine test.

Testing was cross-domain rather than in-domain, because in-domain testing had
already been shown to be uninformative here: the SCUT-HEAD Part A control
scored well on SCUT-HEAD's *own test split* while being unusable on real
footage, all three of its splits sharing one camera and one viewpoint. All
three models were therefore scored identically on the 482 ShanghaiTech images
at confidence 0.25, `imgsz=1024`, detection cap 4,000, and additionally with
standard detection metrics on the held-out CroHD scene at the deployed
confidence of 0.25.

The two SCUT-HEAD models were retained as controls rather than discarded,
because the delivered model's score is uninterpretable without them: a recovery
figure means nothing until it is set against what the obvious first choice of
dataset produced on the same images. The three runs differ in resolution and
batch size as well as in training data, so this is a controlled comparison
rather than a clean ablation — though the confound runs against the winner,
which was scored below its training resolution.

Count ratio was used alongside mAP because density is a *count*: what matters is
whether a person was found, not whether the box hugs them to the pixel. Its
limitation is that a count cannot distinguish "found everyone" from "missed
fifty and invented fifty". Accuracy is not reported at all, because object
detection has no true-negative count and the quantity is undefined.

The warning layer could not be validated, because no data contains the condition
it is designed to recognise. Its thresholds are inherited or derived per scene:
the inherited ones are Fruin's **queuing** classes, chosen over his walkway
classes because the escalation case is an accumulating, near-stationary crowd,
and adopted from secondary sources.

---

## 5. Relationship to the original work plan

The original plan was simple: fine-tune a head detector on a crowd-crush
dataset; obtain a test set with measured ground points; compute density in
people/m²; and warn using Fruin's thresholds.

Three of those four were delivered: the detector, the density calculation and
the two-signal warning rule, and a homography that turns measured ground points
into metres. The first premise was not, and its failure propagated through
everything else.

**No crowd-crush dataset was available.** Such footage exists but is not
publicly released, for the legal and ethical reasons set out in §1, and nothing
within the project's control would have changed that. The detector was trained
on the nearest available proxy instead — first SCUT-HEAD, then CroHD once the
first was shown to be the wrong one (§2) — and the system was redefined to
detect conditions *associated with* flow breakdown rather than crushes
themselves. That redefinition is the largest single departure from the plan.

**No test set with measured ground points was obtained either.** Surveying a
site ourselves carried an ethics-approval lead time the schedule could not
absorb, and no public dataset combines surveyed calibration with the densities
the system targets (§2). Without a homography fitted to real measurements every
quantity stays in pixels. The real-footage path therefore reports occupancy and
dimensionless motion statistics against thresholds learned from each scene's own
baseline, while Fruin's bands — which need absolute density — can only be
exercised on a clip whose geometry is declared. Hence two decision rules where
the plan assumed one.

Two smaller departures: SCUT-HEAD Part B moved from held-out domain to training
data, viewpoint diversity being worth more there than as a test set (§2); and
the planned UMN anomaly-detection evaluation was dropped, the UMN anomaly being
coordinated escape, which produces high speed but **low local velocity
variance** — precisely what the crowd-pressure measure is built to score as
benign (§1).

### Two decision rules, and where their thresholds come from

The absence of calibration forced two decision rules rather than one. They make
different claims and use **different level vocabularies**:

| | calibrated rule | scene-relative rule |
|---|---|---|
| Implemented in | `src/cdw/warning.py` | `scripts/demo_uncalibrated.py` |
| Levels | NORMAL / WATCH / **WARNING** | LEARNING / WATCH / **ELEVATED** |
| Signals | density in a crowded Fruin band **and** crowd pressure rising | occupancy above this scene's own normal **and** net inflow positive |
| Units | people/m² — absolute | people — relative to this camera |
| Thresholds from | cited literature (Fruin) plus a swept parameter | the footage itself |
| Claims | conditions resemble values documented before past flow breakdowns | conditions are unusual **for this camera** |

The scene-relative threshold is computed rather than chosen: the system records
occupancy over an initial learning period and takes a high percentile of it.
What is chosen are the five parameters that produce it — the percentile, the
length of the learning window, the net-rate threshold, and the dwell and clear
times. The dwell and clear times are not arbitrary; without them the level flaps,
because occupancy sits on the threshold for extended periods.

ELEVATED is deliberately not called WARNING. It is an anomaly signal, needing no
survey and no literature, and correspondingly it cannot state that a crowd is
approaching danger in absolute terms — only that this scene is behaving
unusually. Its known weakness is the learning window, which assumes the opening
period is representative of the scene.

Because no real footage is both calibrated and deteriorating, the two rules were
demonstrated separately. The scene-relative rule was run on CroHD footage, using
the trained detector and real tracking with no calibration and no borrowed
thresholds; the calibrated rule on a watermarked synthetic clip whose geometry
is declared rather than surveyed, with detection replayed from ground truth
since the trained detector does not respond to synthetic imagery. One
qualification carries into the reading of the first: its sequence is a detector
**training** scene, so it establishes that the occupancy, flow-balance and
escalation logic behave correctly on real crowd video, not that the detector
generalises (§4). What each produced is reported in the findings section.

> Recommendations — what surveyed calibration and incident data would enable,
> and the three candidate routes for training on them — are in
> `docs/future_work.md`, which covers them far more fully than this section
> should.
