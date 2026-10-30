# Limitations

Written to be quoted in the proposal. Each item says what the limitation is,
why it exists, and how we measure it rather than how we hope it away.

## 1. The system detects conditions. It does not predict crushes.

This is the single most important sentence in the writeup, and it should appear
in the abstract, the methods and the conclusion.

What the system measures is whether crowd density and crowd pressure currently
resemble values documented in the literature before past flow breakdowns. It
has no model of what happens next. It cannot distinguish a dense, rising-
pressure crowd that is about to disperse from one that is about to fail, and
nothing in its design gives it access to the information that would separate
those cases — intent, the state of exits outside the frame, what a marshal is
about to do.

Phrasings to use: "flags conditions associated with flow breakdown",
"early-warning indicator". Phrasings to avoid: "predicts crowd crushes",
"prevents disasters", "AI safety system".

A warning is a prompt for a human to look at a camera. It is not a decision.

## 2. Detector recall collapses in the density regime that matters

Above roughly 4–5 people/m², heads occlude one another and the detector misses
an increasing fraction of them. Density is a count divided by an area, so a
recall drop feeds straight through: **under-detection makes a dangerous crowd
look safer than a moderate one.** The failure is in the direction that matters,
which is the worst possible direction for a warning system.

This is a property of the measurement, not a bug to be fixed, and the project
measures it explicitly.

**How it was measured in the end.** `scripts/eval_point_recall.py` evaluates the
detector against 482 ShanghaiTech images carrying 241,275 **existing** point
annotations, binned by true heads per image. That gives a count ratio per
density band and a **usable ceiling**: 200 heads per frame, above which the
system's own numbers cannot be trusted. Report that ceiling as a headline
result. A system that knows where it stops working is a better engineering
outcome than one that claims it never does.

> **Note on an earlier plan.** This project originally intended to establish the
> ceiling by **blind hand counting** — `scripts/sample_frames.py` for
> density-stratified sampling, `tools/count_heads.py` for shuffled annotator
> sessions with a second annotator on a subset for inter-annotator agreement,
> and `scripts/eval_recall.py` to fit the curve and a debiasing correction. That
> tooling exists and is tested, but **the study was never run**: the switch to
> public point-annotated datasets supplied far more annotated heads than hand
> counting could, and on dense crowds that nobody on the team had to count by
> eye. Do not cite the hand-count protocol as something the project did. The
> `results/` directory contains no hand-count data, and that is why.

Caveats on the measurement as actually performed:

- It is a **count ratio, not detection-level recall**. It does not check that a
  detected head is the same head the annotation marked. For a density system
  that is the right quantity; for a detection paper it would not be.
- It cannot distinguish "found everyone" from "missed fifty and invented
  fifty". Two errors in opposite directions cancel in a count.
- The annotations are somebody else's and carry their own error. The
  ShanghaiTech redistribution we obtained had its labels scaled wrongly by 2×,
  which would have silently invalidated every number derived from it had it not
  been caught (`results/recall/databug_shanghaitech_2x_scaling.jpg`).
- No single confidence threshold suits all densities — sparse scenes favour
  0.3, extreme density 0.1 — and choosing the best per bin would be selecting
  on the evaluation data.

## 3. Head detections do not lie on the ground plane

The homography describes **one flat surface**. We fit it to the ground. A head
is about 1.7 m above that surface, so the ray from the camera through a head
meets the ground *beyond* the person. For a camera at height `H` whose nadir is
`N`, a head at true distance `R` from `N` projects to

```
R' = R · H / (H − h)
```

a radial dilation about the nadir. The error is not a rounding detail. At
`H = 6 m`, `h = 1.7 m`, someone 5 m from the nadir lands **2.0 m too far away** —
comparable to the width of the region being measured. Uncorrected, people near
the far edge of a region silently fall outside it, deflating density exactly
where the crowd is deepest.

`cdw.calibrate.HeightCorrection` inverts it, given two extra measurements: the
camera height, and where the nadir falls in world coordinates. Configure it
under `calibration.height_correction`. Residual error remains because real
stature varies by roughly ±0.1 m; at the distances above that is order 0.1 m,
which is worth stating and far better than ignoring the effect.

**This only applies once the head detector is in use.** During the pretrained
bootstrap the reference point is `box_bottom` — the feet, which *are* on the
ground plane — so no correction is needed or should be configured.

## 4. Tracking degrades where it is needed most

ByteTrack associates by overlap and motion continuity. In dense crowds, boxes
overlap heavily and motion is erratic, so identity switches rise. A switch
produces a spurious large displacement, which inflates velocity variance and
therefore crowd pressure — a false signal in the direction of a false alarm.

Mitigations in place:
- velocities above `max_speed_mps` (default 6 m/s) are discarded as switches,
  since no pedestrian crosses that;
- velocities are differenced over a lag (default 0.3 s) rather than a single
  frame, so box jitter does not dominate;
- the optical-flow path (`pressure.source: both`) provides an identity-free
  fallback when fewer than half the in-region detections carry usable
  velocities.

Flow has its own failure: it measures apparent motion of texture, not of
people, so it cannot distinguish one person moving fast from a group shuffling.
Running both and **reporting where they disagree** is more informative than
picking one; the disagreement is itself a density-dependent measurement.

## 5. The thresholds are inherited, and one of them is not

The level-of-service bands come from Fruin's queuing classes, converted to
people/m² in `configs/thresholds.yaml`. They are cited, not invented — but see
the VERIFY block at the top of that file: the values there were transcribed
from secondary sources and **must be checked against the primary source** before
submission. Fruin's queuing bands describe standing crowds; his walkway bands
are different and flow-based. A bottleneck is between the two cases. Say which
you chose and why.

`pressure_slope_threshold` is **not** from the literature. Crowd pressure is
scale-dependent — it depends on the region size, the frame rate, the velocity
lag and the detector — so no published value transfers. It has to be set from
the pressure series your own calibrated footage produces (procedure in
`calibration.md`). Record the value and the data it came from, and present it
as a tuned parameter, because that is what it is.

## 6. Single camera, single plane, fixed scene

- One homography per camera position. Move or bump the camera and every number
  is wrong, silently. Re-calibrate after any disturbance and note it.
- The ground must be flat within the region. Steps, ramps and kerbs break the
  single-plane assumption.
- People outside the frame do not exist to the system. A bottleneck is fed by
  approaches it cannot see.
- Extrapolating the homography beyond the quadrilateral it was fitted to is
  where area estimates go quietly wrong. Keep the region inside the calibrated
  points.

## 7. Training data domain gap

SCUT-HEAD Part A is classroom surveillance: indoor, static, seated, regularly
spaced subjects, consistent lighting. The elevated viewpoint transfers; almost
nothing else does. Expect a further recall drop on real pedestrian footage *on
top of* the density effect, and measure the two separately rather than
reporting a single number.

Validation mAP on SCUT-HEAD is evidence about SCUT-HEAD. It is not evidence
about pedestrian bottlenecks. This was demonstrated rather than assumed: mAP50
0.938 in-domain, 0.269 count ratio on real elevated crowds.

> **This section originally advised keeping Part B as a held-out domain, on the
> grounds that training on both leaves nothing to test with.** That advice was
> reversed by evidence. Part B is viewpoint-diverse, and viewpoint turned out to
> be the operative variable, so it was worth more as training data than as a
> test set — and ShanghaiTech had meanwhile become a stronger held-out domain,
> being a genuinely different source. The underlying principle still holds:
> never test on what you trained on. Only the choice of which dataset plays
> which role changed.

## 8. Evaluation is on a handful of clips

Whatever is concluded rests on a small number of scenes. Concretely: the
measurement validation covers **four CroHD sequences** (three of them training
scenes, one held out), the flow study two, and the end-to-end demonstrations
two — 133 s and 40 s respectively. The cross-domain detector evaluation is
broader, at 482 ShanghaiTech images, but those are stills from unknown cameras
and support no claim about tracking, velocity or flow.

That is enough to establish that the counting bias is consistent across the
scenes tested. It is not enough to establish that it is consistent everywhere,
and no result here should be generalised to a camera geometry or crowd type
that does not appear in CroHD.

## 9. No validated dangerous footage

The system is aimed at a regime we cannot ethically or practically record. The
synthetic clip (`scripts/make_synthetic_clip.py`) exercises the arithmetic at
high density but is not evidence about real crowds — it contains no occlusion,
no detector, and motion we scripted ourselves. It proves the pipeline is
self-consistent. It proves nothing about the world.
