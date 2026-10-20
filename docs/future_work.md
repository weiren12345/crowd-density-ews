# What better data would enable — and how we would use it

Draft for the recommendations / end-to-end system section. The question this
answers: given incident data and surveyed cameras, how would the system learn
to recognise crush conditions, and what would it then look like in deployment?

---

## 1. What "crush data" would actually be

Worth stating precisely, because the honest answer shapes everything after it.

A usable dataset would be video of crowd incidents with frame-level labels —
normal, deteriorating, and the onset of failure. Such footage **does exist**:
CCTV from investigated disasters is examined by coroners, inquiries and
researchers. It is not publicly available, and the reasons are not
bureaucratic. It shows people dying. Victims cannot consent, families have
standing, and the material is usually held under legal restriction.

Even with access, the scale would be small. Major crowd disasters are rare;
a research collection might contain **tens of events, not thousands**. That
single fact rules out the approach most people reach for first.

**So the realistic target is not "crush footage" but its neighbours:**

- **Flow breakdown without casualties** — congestion events at festivals,
  stadium egress, transit hubs. Far more common, recordable, and the same
  physical phenomenon short of its extreme.
- **Near misses** recorded by venue operators who already run CCTV.
- **Instrumented crowd experiments** — controlled studies with volunteers at
  known densities, which exist in the pedestrian-dynamics literature.

A system trained on flow breakdown and evaluated for whether it fires early in
the few documented disasters is both achievable and defensible. Training
directly on crushes is neither.

---

## 2. What camera measurements would add

Calibration is usually framed as "getting the density number right". Its more
important consequence is **transferability**.

Without calibration, every quantity is in pixels, so every threshold is
specific to one camera at one zoom. A model trained at site A is meaningless at
site B.

With a surveyed homography, measurements become **physical**: people/m², m/s,
(people/m²)(m/s)². Those units mean the same thing everywhere. That unlocks:

- **Cross-site training.** Pool data from many cameras, because the features
  are comparable. A model learns about crowds rather than about one view.
- **Published thresholds become applicable.** Fruin's bands and the
  crowd-pressure literature are expressed in physical units and cannot be used
  at all without them.
- **Transferable validation.** A model validated at one venue carries evidence
  to another.

This is why the calibrated path was retained in the system even though the
available footage could not use it. **The requirement is not a limitation to
apologise for; it is the thing that would make the learned component
generalise.**

---

## 3. Three approaches, by data requirement

### A. End-to-end video → risk score
Feed raw frames to a video model (3D CNN or video transformer) and predict
whether an incident follows.

*Requires:* hundreds to thousands of labelled events.
*Verdict:* **not recommended**, and not only for data reasons. It produces an
unauditable score. For a system whose output prompts an evacuation decision,
"the network said 0.87" is not an acceptable justification, and an inquiry
would say so.

### B. Learned thresholds on physical features — **recommended**
Keep the existing measurement stack — density, speed, local velocity variance,
crowd pressure, inflow versus outflow — and replace the hand-set thresholds
with a classifier trained on those features.

*Requires:* perhaps tens of breakdown episodes, because the feature engineering
has already done the hard work. The model is choosing a decision boundary in a
handful of physically meaningful dimensions, not learning to see.

*Suitable models:* logistic regression or gradient-boosted trees. Both are
small, data-efficient, and **inspectable** — you can state the learned
coefficient on each physical quantity and sanity-check it against the
literature. A model that assigned negative weight to density would be visibly
wrong, which is not true of a video network.

*What it would fix:* `pressure_slope_threshold` is currently derived per scene
by a sweep, and the Fruin bands are inherited. A learned boundary over several
sites would replace both with something measured.

### C. Physics-informed training from simulation
Generate breakdown events with a crowd-dynamics model, train on simulation,
validate on real normal footage.

**This project already has the beginning of it.** `make_scenario_clip.py`
simulates a bottleneck that fills faster than it empties, with speed coupled to
local density and individual walking paces producing realistic velocity
variance. Scaling that to many scenarios — varying geometry, inflow, exit
capacity, crowd composition — would produce a labelled training set of
deterioration events at zero ethical cost.

*The catch, stated plainly:* a model trained on simulation learns the
simulator's assumptions. Validating that it transfers is a research problem in
its own right (the sim-to-real gap familiar from robotics). It is promising,
not free.

### D. Anomaly detection, as a complement
Train only on normal footage; flag departures. Needs no incident data at all,
and is partly implemented already in the scene-relative baseline. Best used
alongside B rather than instead of it: it catches the unusual, whereas B
recognises the specific condition of interest.

---

## 4. What the end-to-end deployed system would look like

**Commissioning (per camera, once)**
1. Survey the scene: measure ground features, fit the homography, record
   reprojection residuals.
2. Measure camera height and nadir, enabling the head-height correction.
3. Define the monitored region on the ground.
4. Record a baseline period of ordinary operation and fit the scene's own
   speed-density relationship.

**Runtime (per frame)**
Detection → tracking → projection to the ground plane → density, velocity
field, crowd pressure, inflow versus outflow → learned classifier → level.

**Outputs, and who uses them**
- **An operator alert**, not an automated action. No barrier opens, no
  announcement plays, no crowd is redirected without a human deciding.
- **The evidence behind it**: current density, which way it is trending, the
  inflow/outflow split, and which rule fired. An operator asked to act needs to
  see why.
- **A confidence state.** When measured density exceeds the level at which
  counting was shown to be reliable, the system reports
  `MEASUREMENT UNRELIABLE` rather than a falsely low number. This project
  measured that ceiling; a deployed system should respect it.
- **A logged record** for post-incident review, which is also how the training
  set grows.

**Maintenance**
Re-calibration after any camera disturbance; periodic re-validation of the
scene baseline; retraining as logged episodes accumulate.

---

## 5. The problems more data does not solve

Worth including, because they are the ones a careful reader will raise.

**Label ambiguity.** When does a crush "begin"? Investigators disagree, and the
label determines what the model learns. A system trained against labels placed
at the moment of injury learns to fire too late to be useful.

**Class imbalance.** Even a rich collection would contain perhaps one positive
frame in tens of thousands. This demands precision-recall evaluation rather
than accuracy, and cost-sensitive training.

**Choosing the operating point is a value judgement, not a technical one.**
False negatives can kill; false alarms erode the operator trust the system
depends on, and an alarm that is routinely ignored is worse than none. Where to
sit on that curve is a decision for venue operators and regulators, and the
engineering contribution is to *characterise* the trade-off, not to settle it.

**Validation remains partial.** A model that fires early on the few documented
disasters has evidence from a handful of events. That is better than nothing
and much less than proof.

---

## 6. What we would do first

With modest additional resources, in order:

1. **Instrumented crowd experiments** with consenting volunteers at controlled
   densities — ethically straightforward, calibration exact, and it produces
   the physical-unit data the thresholds need.
2. **Partner with a venue operator** who already runs calibrated CCTV, to log
   ordinary operation and the occasional congestion episode.
3. **Scale the simulator** to a labelled corpus of deterioration events, and
   measure how far a model trained on it transfers.
4. **Train approach B** on the combined set, and report the operating-point
   trade-off rather than a single accuracy figure.

None of these requires footage of a disaster. That is the point: the useful
version of this system is reachable without waiting for the thing it is meant
to prevent.
