# Methods — draft

Drafted against the rubric: the full ML workflow, justified technical choices,
and how the original work plan was followed or adapted. Rework the wording in
your own voice; every number traces to `results/README.md`.

A thread runs through all five sections: **what the available data allowed, what
it did not, and what different data would have enabled.** That is not an excuse
running through the methods — it is the reasoning behind most of the technical
choices, and it leads naturally to the recommendations.

---

## 1. Data preparation

Three datasets were used, each verified before use rather than trusted. They
served different roles, and the distinction organises everything that follows:
**one trained the delivered model, one was superseded and retained as a control,
and one was never trained on at all.**

### CroHD / HeadTracking21 — the dataset the delivered model was trained on

CroHD supplies dense crowds from elevated cameras as video with head boxes and
track identities: four training sequences, 5,741 frames, **1,188,496 annotated
heads** at 1920×1080 and 25 fps. It is the only accessible dataset matching
deployment conditions on every axis that matters — viewpoint, density,
annotation type and temporal continuity — and §2 sets out how the project
arrived at it.

Two properties distinguish it from the alternatives. The cameras are mounted
above pedestrian traffic, so the annotated heads are predominantly backs and
tops rather than faces, which is what a deployed camera sees. And because it is
video with persistent identities, the same data supports both detection training
and the velocity measurements the crowd-pressure signal depends on; no still-image
dataset can do the second.

Test-sequence ground truth is withheld by the benchmark and its evaluation server
is closed, so the four training sequences were re-split locally. Conversion from
MOT to YOLO format made three decisions worth recording:

- **Scene-level holdout.** Sequences are held out whole. At 25 fps consecutive
  frames are near-duplicates, so a random frame split would place almost
  identical images on both sides of the divide and report a score measuring
  memorisation rather than generalisation.
- **Temporal subsampling.** Every fourth frame was retained; neighbouring frames
  carry almost no additional information. This yielded 1,187 training images
  containing 253,376 boxes, and 100 validation images from the held-out scene
  containing 17,665.
- **Ignore flags respected.** MOT rows with a confidence field of 0 mark entries
  excluded from evaluation and were dropped rather than trained on. Boxes were
  clipped to the frame boundary.

Only retained frames were copied, since duplicating all 5,741 would have
consumed several gigabytes for no benefit.

### The two supporting datasets

**SCUT-HEAD (Parts A and B) — the initial choice, superseded.** Obtained as a
Roboflow export in YOLO format: Part A 1,100 train / 400 validation / 500 test;
Part B 1,630 / 466 / 233. Part A is classroom surveillance from an elevated
camera across three lecture rooms; Part B is internet-sourced photography.
Labels were verified as normalised `class cx cy w h`, single class, with zero
malformed rows across a 400-image sample. Median head width measured 17 px
against a median image width of roughly 1,070 px, which set the training
resolution for these runs. Why it was superseded is §2's subject; the models
trained on it are retained deliberately, as the control that gives the final
result its meaning.

**ShanghaiTech Part A — evaluation only, never trained on.** 482 images,
241,275 annotated heads, 33–3,138 people per image, annotated as points.
Verification found a defect that would have invalidated every result derived
from it: the redistribution had **upscaled every image 2× without rescaling the
point labels**. The maximum annotation x-coordinate sat at 0.498 of image width
(median across all 1,198 images in both parts), and overlaying the raw points
showed them clustered in the top-left quadrant. The evaluation script now infers
the scale factor per image and reports it rather than assuming 1.0.

### What was not available

No dataset was found containing the target condition. Footage of crowd crushes
exists — coroners, public inquiries and researchers examine it — but it is not
publicly released, and the reasons are substantive rather than bureaucratic: it
shows people dying, victims cannot consent, and the material is held under legal
restriction. Even with access the scale would be small, since major incidents
are rare; a research collection might hold tens of events, not thousands.

This absence is the single most important constraint on the project and shapes
the sections that follow.

---

## 2. Data strategy

The strategy changed twice, each time in response to measurement rather than
preference, and converged on CroHD.

**The initial choice and its failure.** SCUT-HEAD Part A was selected because it
is head-annotated (bodies are occluded in dense crowds, heads are not), elevated,
and already in YOLO format. A detector trained on it reached mAP50 0.938 on its
own validation split but recovered only 27% of people on ShanghaiTech.
Inspecting *which* detections were missed — rather than the aggregate score —
showed that almost every successful detection was a forward-facing person and
virtually every back-of-head was missed. SCUT-HEAD's camera sits at the front of
a classroom facing the students, so the model had learned **faces from an
elevated-front angle, not heads from any angle**, while a camera above a
pedestrian bottleneck sees mostly backs and tops of heads.

**An intermediate step that confirmed the diagnosis.** Part B — internet-sourced
and viewpoint-diverse — was added to training, reversing the original plan that
had reserved it as a held-out domain; ShanghaiTech had meanwhile become the
better held-out domain, being a genuinely different source never trained on.
Recall roughly doubled at every density. That it improved established viewpoint
diversity as the operative variable; that it remained unusable established that
classroom photography could not supply enough of it. The conclusion was that the
problem would not be solved by adding data of the same kind.

**The dataset that worked: CroHD.** Selection was made against the failure
diagnosis rather than by convenience, requirement by requirement:

| Requirement | Why it is necessary | SCUT-HEAD | CroHD |
|---|---|---|---|
| Heads, not bodies | bodies are occluded in dense crowds | yes | yes |
| Camera above the crowd | determines whether faces or backs of heads are seen | partly — elevated but front-facing | **yes** |
| Genuinely dense scenes | the regime of interest | no — tens per frame | **yes — hundreds** |
| Box annotations | point labels cannot train a detector | yes | **yes** |
| Video with identities | required for velocity and crowd pressure | no — stills | **yes** |

Only CroHD satisfies all five. The viewpoint row is the one that had been
underweighted at the start: it had been treated as roughly satisfied by "elevated",
when the operative question was what the camera *faces*. The density row matters
almost as much, since 253,376 boxes across 1,187 images means the model sees
heavily occluded, overlapping heads as the normal case rather than the exception.

**A constraint worth stating.** ShanghaiTech could be used for evaluation only,
because point annotations cannot train a box detector. Dense-crowd data is
abundant in point form and scarce in box form, which limits what any
detection-based approach can learn and is part of why CroHD is as close to
unique as it is.

### The dataset that was actually wanted

Stating the gap precisely clarifies both the choices above and the
recommendations below. An ideal dataset would have been:

| Property | Why | Available? |
|---|---|---|
| Dense crowds | the regime of interest | yes (ShanghaiTech, CroHD) |
| Elevated camera | matches deployment viewpoint | yes (CroHD) |
| Video with identities | needed for velocity | yes (CroHD) |
| **Surveyed calibration** | gives physical units | **no** |
| **Deterioration episodes** | the condition to detect | **no** |

The last two are what the project lacked, and they are separable. Calibration is
an engineering requirement that any deployment would satisfy. Incident data is
the genuinely hard problem.

---

## 3. Model development

**Architecture.** YOLOv8n — a single-stage, anchor-free detector with a
CSPDarknet backbone using C2f blocks, a PANet-style feature pyramid, and a
decoupled head operating at strides 8, 16 and 32. 3,157,200 parameters
pretrained; 3,138,003 after reduction to a single class.

**Transfer learning.** All runs initialise from COCO-pretrained weights, so the
backbone begins with general visual features and need only specialise. Training a
3M-parameter detector from scratch on ~1,200 images would not converge usefully.

**Single-class formulation.** The classification head was reduced from 80 classes
to one ("head"). The system performs detection and localisation only — it never
identifies anyone, which is relevant to the ethics position.

**Model size.** YOLOv8s (11,166,560 parameters) was obtained as a larger
alternative but proved unnecessary: the nano model reached a 0.975 count ratio
once the training data was appropriate. **The limiting factor was data, not
capacity** — a result rather than an assumption, and one that shapes the
recommendations.

**Augmentation, chosen rather than defaulted:**

- *Rotation, shear and perspective disabled.* Heads are upright in an elevated
  view; these transforms teach poses that never occur.
- *Mosaic disabled.* Mosaic composites four images into one canvas, roughly
  halving object scale. With heads at 17 px median this pushes them below what
  the stride-8 detection layer can resolve. It also allocates a (2·imgsz)² buffer
  per sample, which exhausted available memory.
- *Horizontal flip and ±30% scaling retained* — both produce configurations that
  genuinely occur.

**Detection cap.** Ultralytics limits predictions to 300 per image by default.
Evaluation images contain up to 3,138 annotated heads, so the default would have
truncated predictions and produced an apparent recall collapse that was in fact a
library setting. The cap was raised and the behaviour covered by a regression
test.

### The delivered model

The artefact the rest of the system is built on is a single YOLOv8n checkpoint
fine-tuned on CroHD. Its configuration:

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

Two figures summarise what it achieved. On **held-out CroHD footage** it reaches
precision 0.763 and recall 0.674 on a scene it never trained on. On
**cross-domain ShanghaiTech images** it recovers 97.5% of people at 100–200 heads
per image, against 26.9% for the Part A model and 42.1% for Part A+B — a 3.6×
improvement over the first attempt, from changing the training distribution
alone.

The operating point is confidence 0.25 with the detection cap raised to 3,000,
and this is the configuration used for every downstream result in the project:
the measurement validation, the fundamental diagram, the flow study and the
real-footage demonstration all run on this checkpoint.

### Why the learned component was kept deliberately small

The architecture places one neural network at the front — reading pixels — and
keeps everything after it deterministic: projection to the ground plane, density,
crowd pressure, and the warning rule. This was a design decision with two
justifications.

The first is accountability. An alarm prompts a human to act, and that human
needs to see which measurement and which rule produced it. A single risk score
emitted by a network cannot supply that.

The second is **data efficiency, and it is the same argument that governs the
recommendations**. Expressing the problem in a handful of physically meaningful
quantities means any future learning over them needs far less data than learning
the same relationships from raw pixels. The feature engineering has already done
the difficult part.

---

## 4. Training, validation and testing

**Hardware constraints shaped the configuration**, and the binding constraint was
not the expected one. The GPU (6 GB) peaked at 2.4 GB; **system RAM (7.6 GB) was
the limit.** Dataloader workers were reduced to zero, since on Windows each
worker is a separate process re-importing the CUDA runtime, and even two
exhausted the commit limit. Batch size fell from 8 to 4 and finally 2 as instance
counts rose, because label-assignment memory scales with objects per image rather
than batch size alone.

**Resolution was matched to the data.** SCUT-HEAD images are natively ~1,070 px
wide, so `imgsz=1024` was used — approximately native, where the originally
planned 1,280 would have upscaled without adding information. CroHD is 1920×1080
with heads at 26–34 px, so `imgsz=1280` kept heads at 17–23 px after downscaling,
above the stride-8 resolution limit.

**Three training runs: the delivered model, and two controls that establish what
it is being compared against.** They differ principally in training data.

| Run | Data | Images | Boxes | Epochs | imgsz | batch | Role |
|---|---|---|---|---|---|---|---|
| 1 | SCUT-HEAD A | 1,100 | ~30k | 15 | 1024 | 4 | control |
| 2 | SCUT-HEAD A+B | 2,730 | ~45k | 10 | 1024 | 4 | control |
| **3** | **CroHD** | 1,187 | **253,376** | 10 | 1280 | 2 | **delivered** |

The controls are retained rather than discarded because the delivered model's
score is uninterpretable without them. "97.5% of people recovered" means nothing
until it is set against what the obvious first choice of dataset produced on the
same images.

**The runs are not a clean ablation, and the claim is framed accordingly.**
Resolution and batch size also differ, for the reasons above: resolution was
matched to each dataset's native scale, and batch size was forced down by memory
as instance counts rose. What *is* controlled is the evaluation — all three
models were tested on the same 482 images at the same resolution and detection
cap. The confound also runs against the winner: Run 3 trained at 1280 but was
evaluated at 1024, below its training resolution, and still led by 3.6×.

**Validation** used held-out splits, scene-level for CroHD. The delivered model
achieved mAP50 0.714, recall 0.674 and precision 0.763 on a sequence it had never
seen. Those figures are *lower* than Run 1's in-domain 0.938, and the comparison
is instructive rather than unfavourable: each model was scored against its own
validation set, and CroHD's — dense crowds of small heads — is far harder than
SCUT-HEAD's classroom photographs. **A validation score is only meaningful
relative to the difficulty of the set that produced it**, which is precisely why
Run 1's 0.938 proved misleading, collapsing on contact with any other data.

**Testing** was cross-domain and identical for all three models: 482 ShanghaiTech
images at confidence 0.25, `imgsz=1024`, detection cap 4,000.

| heads/image | SCUT-HEAD A | SCUT-HEAD A+B | **CroHD (delivered)** |
|---|---|---|---|
| 100–200 | 0.269 | 0.421 | **0.975** |
| 200–400 | 0.180 | 0.288 | **0.740** |
| 1200+ | 0.012 | 0.024 | **0.204** |
| usable ceiling (ratio ≥ 0.8) | never | never | **200 heads/image** |

The delivered model is the only one of the three that is ever usable: the
controls never reach a count ratio of 0.8 at any density. The gap is widest where
it matters most, and it is attributable to the training distribution, since the
evaluation was held constant.

**Metric choice.** Count ratio was used alongside mAP because density is a
*count*: what matters is whether a person was found, not whether the box hugs
them to the pixel. mAP50-95 penalises imprecise corners heavily at 17 px object
sizes, which is not the error this system is sensitive to. The limitation is
stated: count ratio cannot distinguish "found everyone" from "missed fifty and
invented fifty".

**The usable ceiling is the project's defining limitation, and it is measured
rather than assumed.** Counting is reliable to roughly 200 heads per frame;
above that the ratio falls away, reaching 0.204 in the densest images. The
direction of the error is what makes this serious. Because density is a count,
missed detections do not produce a conspicuous failure — they produce a *lower*
density reading, so the system reports a dangerously packed crowd as safer than
it is. Degradation is therefore worst in precisely the regime the system exists
to monitor, and any deployment must treat a measurement above the ceiling as
unreliable rather than as a low number. The ceiling is now scale-limited rather
than viewpoint-limited, which is an improvement on where the project started but
not a solution. Evaluation breadth is a related constraint: the measurement
validation rests on four CroHD scenes and one held-out sequence, which is enough
to establish that the bias is consistent, not enough to establish that it is
consistent everywhere.

**Independent validation against ground truth.** Across four CroHD scenes,
counting is biased low by a consistent **14% (sd 3.4 percentage points)** — a
stable bias is correctable, an erratic one is not — and track-derived speeds
match ground-truth pedestrian speeds to within **2–5%** (correlation up to
0.996). Counting without tracking performs better still (0.977 ratio, 4% mean
absolute error), establishing that detection should be used for counting and
tracking only for velocity, since the tracker discards detections it cannot
associate.

### What could and could not be validated

The detection and measurement layers were validated against ground truth, as
above. **The warning layer could not be**, because no data contains the condition
it is designed to recognise. The thresholds are therefore inherited (Fruin level
of service, still to be verified against a primary source) or derived per scene,
rather than learned. Replacing them with learned values is the central
recommendation, and §5 sets out what that would require.

---

## 5. Relationship to the original work plan

The original plan was: fine-tune a head detector on SCUT-HEAD; film our own
footage at a pedestrian bottleneck; calibrate with a homography from measured
ground points; compute density in people/m²; and warn using Fruin thresholds.

**What was followed.** The staged architecture, the head-detection approach, the
homography-based calibration, the density and crowd-pressure formulation, and the
two-signal warning rule were all implemented as planned and are tested. The
deliberate separation of the learned component from the decision logic was
maintained throughout.

**What changed, and why.**

*Own footage was not obtained.* Ethics approval for filming identifiable people
carries a lead time the project schedule could not absorb — a risk the plan had
flagged from the outset. Public datasets were substituted.

*Absolute density was dropped for the real-footage path.* People/m² requires a
surveyed camera. Without one, the system reports occupancy and dimensionless
motion statistics, with thresholds learned from each scene's own baseline rather
than borrowed.

*Part B moved from held-out to training data*, as described in §2, because the
evidence indicated it was worth more as viewpoint diversity.

*The anomaly-detection evaluation on the UMN dataset was dropped.* The proposed
metric was inappropriate for this system: the UMN anomaly is coordinated escape,
which produces high speed but **low local velocity variance** — precisely the
quantity the crowd-pressure measure is designed to score as benign. Evaluating
against it would have measured the wrong thing.

### Two decision rules, and where their thresholds come from

The absence of calibration in the available footage forced two decision rules
rather than one, and they make different claims. The distinction matters for
reading the results, because the two use **different level vocabularies**.

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
the first 15 s. ELEVATED then requires occupancy above 101 **and** a net rate
above 0.3 people/s, sustained for 2 s.

What is chosen rather than derived are the five parameters that produce it — the
85th percentile, the 15 s learning window, the 0.3 people/s rate, and the 2 s
dwell and 3 s clear times. The dwell and clear times are not arbitrary: without
them the level oscillated nineteen times in eleven seconds, since occupancy sits
on the threshold for extended periods.

**ELEVATED is deliberately not called WARNING.** It is an anomaly signal. It
needs no survey and no literature, and correspondingly it cannot state that a
crowd is approaching danger in absolute terms — only that this scene is
behaving unusually.

Its weakness is the learning window. On HT21-02 the baseline landed at 102
against a clip mean of 85: an unrepresentative quiet period set the bar too
high, which is part of why that sequence never escalated. A longer or explicitly
quiet learning period would be more robust.

### Two complementary demonstrations

Because no real footage is both calibrated and deteriorating, two demonstrations
were produced rather than one.

*Real footage.* On CroHD sequence HT21-03 — a street crossing where pedestrians
accumulate at a signal and discharge — the complete pipeline escalated from
NORMAL to WATCH at 15.1 s and to ELEVATED at 17.1 s, then stood down at 29.8 s.
The flow measurements show why: inflow of 16.2 people/s against outflow of 9.2 at
escalation, converging to 14.6 against 12.4 at stand-down. This uses the trained
model, real detections and real tracking, with no calibration and no borrowed
thresholds — ELEVATED carrying the scene-relative meaning set out above, not the
absolute one.

*Fabricated scenario.* A synthetic scenario was constructed with **declared**
geometry (a 6.0 × 5.0 m region, recovered exactly by the homography) in which a
bottleneck fills faster than it empties. The clip is watermarked on every frame
and the geometry is asserted rather than surveyed. On it the calibrated pipeline
reports density in people/m², applies the Fruin bands, escalates to WARNING at
27.7 s, and de-escalates at 53.1 s when the crowd — still at level-of-service E —
stops deteriorating. Detection is replayed from ground truth, since the trained
detector does not respond to synthetic imagery.

**What the pairing supports.** The real demonstration establishes that the
measurement and decision stack functions on genuine crowds. The fabricated
scenario establishes what the same system reports when calibration is available
and conditions deteriorate. Together they support the claim that **given surveyed
footage of a deteriorating crowd, the system would produce the calibrated output
shown in the scenario** — while making clear this rests on a demonstration of
behaviour under specified conditions, not on evidence about real disasters.

### What better data would enable, and how it would be used

This follows directly from §2's gap analysis and §3's architectural choice.

**Surveyed calibration would make the learned component transferable.** This is
the less obvious benefit and the more important one. Without calibration every
quantity is in pixels, so every threshold belongs to one camera at one zoom and a
model trained at one site is meaningless at another. With a homography,
measurements become physical — people/m², m/s — and mean the same thing
everywhere. That permits pooling data across cameras, applying published
thresholds at all, and carrying validation between venues. **The calibration
requirement is the enabler, not merely a limitation.**

**Incident data would let the thresholds be learned rather than inherited.** The
realistic target is not crush footage but its neighbours: flow breakdown without
casualties (common at festivals, stadium egress and transit hubs), near misses
logged by venue operators, and instrumented crowd experiments with consenting
volunteers at controlled densities. A system trained on breakdown and checked for
whether it fires early in the few documented disasters is achievable; training
directly on crushes is not.

**How the model would then be trained.** Three routes were considered:

*End-to-end video classification* — feeding raw frames to a video network and
predicting whether an incident follows — is **not recommended**. It requires
hundreds of labelled events that do not exist, and it produces an unauditable
score. For a system whose output prompts an evacuation decision, "the network
returned 0.87" is not a justification an inquiry would accept.

*A learned decision boundary over the existing physical features* is the
recommended route, and it is why the architecture in §3 was kept small. Retaining
density, speed, local velocity variance, crowd pressure and the inflow/outflow
balance as inputs, a modest classifier — logistic regression or gradient-boosted
trees — would replace the hand-set thresholds. This would need perhaps tens of
breakdown episodes rather than thousands, because the features already encode the
physics. Such a model is also **inspectable**: the learned weight on each physical
quantity can be stated and checked against the literature, where a video network's
reasoning cannot.

*Physics-informed training from simulation* is a third route the project has
already begun. The scenario generator simulates a bottleneck filling faster than
it empties, with speed coupled to local density and individual walking paces
producing realistic velocity variance. Scaling it across varied geometry, inflow
and exit capacity would yield a labelled corpus of deterioration events at no
ethical cost. The caveat is real: a model trained on simulation learns the
simulator's assumptions, and demonstrating transfer is a research problem in its
own right.

**What more data would not solve**, and should be acknowledged: label ambiguity
(when a crush "begins" is contested, and labels placed at the moment of injury
teach a model to fire too late); severe class imbalance, demanding
precision-recall evaluation rather than accuracy; and the choice of operating
point, which is a value judgement rather than a technical one. False negatives
can kill, while false alarms erode the operator trust the system depends on.
Characterising that trade-off is an engineering contribution; settling it belongs
to venue operators and regulators.
