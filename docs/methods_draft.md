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
- **Temporal subsampling.** Every fourth frame was retained, neighbouring frames
  carrying almost no additional information. This yielded 1,187 training images
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

HT21-04 is held out from training but not from model selection - the framework
keeps whichever epoch scores best on it - so any figure from it is an upper
bound, not an independent result. ShanghaiTech played no part in fitting or
selection, and is the only genuine test. On validation the model reached mAP50
0.714, recall 0.674 and precision 0.763.

Testing was cross-domain by necessity. The Part A control scored about 0.95 on
SCUT-HEAD's *own test split* at every density: a correctly executed
train/validation/test discipline reported success on a model that was in fact
unusable, because all three splits share one camera. The failure stayed
invisible until the model met a different source. All three models were
therefore scored identically on 482 ShanghaiTech images at confidence 0.25,
`imgsz=1024`, detection cap 4,000.

| heads/image | SCUT-HEAD A | SCUT-HEAD A+B | **CroHD (delivered)** |
|---|---|---|---|
| 100-200 | 0.269 | 0.421 | **0.975** |
| 200-400 | 0.180 | 0.288 | **0.740** |
| 1200+ | 0.012 | 0.024 | **0.204** |
| usable ceiling (ratio >= 0.8) | never | never | **200 heads/image** |

The controls make the delivered figure interpretable: "97.5% recovered" means
nothing until set against what the obvious first choice of dataset produced on
the same images, and neither control is usable at any density. The runs differ
in resolution and batch as well as data, so this is a controlled comparison
rather than a clean ablation - though the confound runs against the winner,
scored below its training resolution.

Count ratio asks whether people were found, not whether boxes were placed well,
so the models were also scored with standard detection metrics on the held-out
CroHD scene:

| model | precision | recall | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| SCUT-HEAD A | 0.314 | 0.065 | 0.107 | 0.028 | 0.006 |
| SCUT-HEAD A+B | 0.535 | 0.328 | 0.406 | 0.213 | 0.041 |
| **CroHD (delivered)** | **0.747** | **0.696** | **0.720** | **0.631** | **0.202** |

The delivered model's precision and recall are balanced, where both controls are
lopsided toward precision - few heads found, moderate confidence in those few,
the wrong trade when the output is a count. HT21-04 is its validation scene, so
that column is an upper bound while for the controls these images are unseen.
Accuracy is not reported: object detection has no true-negative count.

The usable ceiling is the project's defining limitation, and it is measured
rather than assumed. Counting is reliable to roughly 200 heads per frame, above
which the ratio falls to 0.204. Because density is a count, missed detections
lower the reading rather than failing visibly, so the system reports a
dangerously packed crowd as safer than it is - worst in precisely the regime it
exists to monitor.

The measurement stack was validated across four CroHD scenes, three training and
one held out, so this measures agreement with ground truth rather than
generalisation. Counting is biased low by a consistent **14% (sd 3.4 percentage
points)**; a stable bias is correctable, an erratic one is not. The
held-out scene scores 0.869 against a four-scene mean of 0.857, the lowest
(0.799) being a training scene, so difficulty dominates memorisation.
Track-derived speeds match ground truth to within **2-5%**, and counting without
tracking does better still (0.977), so detection should be used for counting and
tracking only for velocity.

The warning layer could not be validated, because no data contains the condition
it is designed to recognise. Its thresholds are inherited or derived per scene:
the inherited ones are Fruin's **queuing** classes, chosen over his walkway
classes because the escalation case is an accumulating, near-stationary crowd,
and adopted from secondary sources.

---

## 5. Relationship to the original work plan

The original plan was: fine-tune a head detector on SCUT-HEAD; film our own
footage at a pedestrian bottleneck; calibrate with a homography from measured
ground points; compute density in people/m²; and warn using Fruin thresholds.

Much of that was followed. The staged architecture, the head-detection approach,
the homography-based calibration, the density and crowd-pressure formulation,
and the two-signal warning rule were all implemented as planned and are tested.
The separation of the learned component from the decision logic was maintained
throughout.

Four things changed. *Own footage was not obtained*: ethics approval for filming
identifiable people carries a lead time the schedule could not absorb — a risk
the plan had flagged from the outset — so public datasets were substituted.
*Absolute density was dropped for the real-footage path*, because people/m²
requires a surveyed camera; without one the system reports occupancy and
dimensionless motion statistics against thresholds learned from each scene's own
baseline. *Part B moved from held-out to training data*, as described in §2.
*The anomaly-detection evaluation on the UMN dataset was dropped*: the UMN
anomaly is coordinated escape, which produces high speed but **low local
velocity variance** — precisely the quantity the crowd-pressure measure is
designed to score as benign — so it would have measured the wrong thing.

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

The scene-relative threshold is computed, not chosen: the system records
occupancy over an initial learning period and takes a high percentile of it. On
HT21-03 that produced a threshold of **101 people**, from the 85th percentile of
the first 15 s; ELEVATED then requires occupancy above 101 **and** a net rate
above 0.3 people/s, sustained for 2 s. What is chosen rather than derived are
the five parameters producing it — the 85th percentile, the 15 s learning
window, the 0.3 people/s rate, and the 2 s dwell and 3 s clear times. The dwell
and clear times are not arbitrary: without them the level oscillated nineteen
times in eleven seconds, since occupancy sits on the threshold for extended
periods.

ELEVATED is deliberately not called WARNING. It is an anomaly signal, needing no
survey and no literature, and correspondingly it cannot state that a crowd is
approaching danger in absolute terms — only that this scene is behaving
unusually. Its weakness is the learning window: on HT21-02 the baseline landed
at 102 against a clip mean of 85, an unrepresentative quiet period setting the
bar too high, which is part of why that sequence never escalated.

Because no real footage is both calibrated and deteriorating, each rule was
demonstrated separately.

*Real footage.* On CroHD sequence HT21-03 — a street crossing where pedestrians
accumulate at a signal and discharge — the complete pipeline escalated from
NORMAL to WATCH at 15.1 s and to ELEVATED at 17.1 s, then stood down at 29.8 s.
The flow measurements show why: inflow of 16.2 people/s against outflow of 9.2
at escalation, converging to 14.6 against 12.4 at stand-down. This uses the
trained model, real detections and real tracking, with no calibration and no
borrowed thresholds. One qualification: HT21-03 is a detector **training**
scene, so what the demonstration establishes is that the occupancy, flow-balance
and escalation logic behave correctly on real crowd video — none of which was
fitted to this scene — not that the detector generalises, which is evidenced
separately and on different data in §4.

*Fabricated scenario.* A synthetic scenario with **declared** geometry (a
6.0 × 5.0 m region, recovered exactly by the homography) in which a bottleneck
fills faster than it empties. The clip is watermarked on every frame and the
geometry is asserted rather than surveyed. On it the calibrated pipeline reports
density in people/m², applies the Fruin bands, escalates to WARNING at 27.7 s,
and de-escalates at 53.1 s when the crowd — still at level-of-service E — stops
deteriorating. Detection is replayed from ground truth, since the trained
detector does not respond to synthetic imagery.

Together the pair supports the claim that **given surveyed footage of a
deteriorating crowd, the system would produce the calibrated output shown in the
scenario** — while making clear that this rests on demonstrated behaviour under
specified conditions, not on evidence about real disasters.

### What better data would enable

Surveyed calibration would make the learned component transferable, the less
obvious benefit and the more important one. Without it every quantity is in
pixels, so every threshold belongs to one camera at one zoom and a model trained
at one site is meaningless at another. With a homography, measurements become
physical — people/m², m/s — and mean the same thing everywhere, permitting
pooling across cameras, applying published thresholds at all, and carrying
validation between venues.

Incident data would let the thresholds be learned rather than inherited. The
realistic target is not crush footage but its neighbours: flow breakdown without
casualties (common at festivals, stadium egress and transit hubs), near misses
logged by venue operators, and instrumented experiments with consenting
volunteers at controlled densities.

Of three routes for using such data, *end-to-end video classification* is **not
recommended**: it requires hundreds of labelled events that do not exist and
produces an unauditable score, and for a system whose output prompts an
evacuation decision, "the network returned 0.87" is not a justification an
inquiry would accept. *A learned decision boundary over the existing physical
features* is the recommended route, and it is why the learned component was kept
small: retaining density, speed, local velocity variance, crowd pressure and the
inflow/outflow balance as inputs, a modest classifier — logistic regression or
gradient-boosted trees — would replace the hand-set thresholds, needing tens of
breakdown episodes rather than thousands because the features already encode the
physics, and remaining inspectable in a way a video network is not.
*Physics-informed training from simulation* is a third route already begun, the
scenario generator coupling speed to local density and giving individuals their
own walking paces; scaling it across varied geometry, inflow and exit capacity
would yield a labelled corpus at no ethical cost, with the caveat that a model
trained on simulation learns the simulator's assumptions.

What more data would not solve is threefold: label ambiguity (when a crush
"begins" is contested, and labels placed at the moment of injury teach a model to
fire too late); severe class imbalance, demanding precision-recall evaluation
rather than accuracy; and the choice of operating point, which is a value
judgement rather than a technical one. False negatives can kill, while false
alarms erode the operator trust the system depends on. Characterising that
trade-off is an engineering contribution; settling it belongs to venue operators
and regulators.
