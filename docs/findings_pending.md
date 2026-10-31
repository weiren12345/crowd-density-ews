# Pending: results content moved out of Methods §4

**Status: parked, not drafted.** This is the results material that was sitting
inside the Methods section, lifted out verbatim so that §4 could become a clean
account of *what was done* rather than *what was found*. It has not been
rewritten, reordered or turned into a section — that is the next job, and it
belongs in the findings, analysis and evidence section.

Nothing here is new. Every figure traces to `results/README.md`, which remains
the source of record and also holds the material this file does **not** cover:
the fundamental diagram, the flow study, the two end-to-end demonstrations, and
the per-sequence validation table.

Methods §4 now states the protocol behind each of these numbers — the split, the
evaluation settings, the role of the controls, and why count ratio is used
alongside mAP. Whoever writes the findings section should read §4 first, and
should not restate the protocol here.

---

## 1. Validation score on the held-out CroHD scene

The delivered model reached **mAP50 0.714, recall 0.674 and precision 0.763**
on HT21-04.

Carry the caveat from §4 whenever this is quoted: HT21-04 is the checkpoint
selection set, so this is an upper bound rather than an independent result.

## 2. Why in-domain testing was abandoned

Run 1 was evaluated on SCUT-HEAD's *own test split*, where it returned a count
ratio of approximately **0.95 at every density** — while recovering only 27% of
people on real elevated footage. A conventional train/validation/test
discipline, correctly executed, therefore reported success on a model that was
in fact unusable, because all three of its splits share one camera and one
viewpoint. The failure stayed invisible until the model met a different source.

This is the project's strongest argument for cross-domain evaluation, and it is
an argument about methodology rather than about this particular model. It may
be worth stating in both sections for that reason.

## 3. Cross-domain count ratio

482 ShanghaiTech images, all three models scored identically.

| heads/image | SCUT-HEAD A | SCUT-HEAD A+B | **CroHD (delivered)** |
|---|---|---|---|
| 100–200 | 0.269 | 0.421 | **0.975** |
| 200–400 | 0.180 | 0.288 | **0.740** |
| 1200+ | 0.012 | 0.024 | **0.204** |
| usable ceiling (ratio ≥ 0.8) | never | never | **200 heads/image** |

The delivered model is the only one of the three ever usable; neither control
reaches a count ratio of 0.8 at any density. The gap is attributable to the
training distribution, since the evaluation was held constant. Against the
first attempt this is a **3.6× improvement** from changing the training data
alone.

## 4. Standard detection metrics

Held-out CroHD scene, at the deployed confidence of 0.25.

| model | precision | recall | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| SCUT-HEAD A | 0.314 | 0.065 | 0.107 | 0.028 | 0.006 |
| SCUT-HEAD A+B | 0.535 | 0.328 | 0.406 | 0.213 | 0.041 |
| **CroHD (delivered)** | **0.747** | **0.696** | **0.720** | **0.631** | **0.202** |

Two things worth drawing out. The Part A model recovers **6.5% of heads**,
which is what mAP50 0.938 on classroom photographs amounts to on real elevated
footage. And the delivered model's precision and recall are **balanced** (0.747
against 0.696) where both controls are lopsided toward precision — they find
few heads and are moderately confident about those few, which for a system
whose output is a count is the wrong trade.

Caveat, as in §4: HT21-04 is the delivered model's validation scene, so its row
is an upper bound, while for the two SCUT-HEAD models these images are entirely
unseen. The unbiased three-way comparison is the count ratio in §3 above.

## 5. The usable ceiling

Counting is reliable to roughly **200 heads per frame**, above which the ratio
falls away to **0.204** in the densest images.

The direction of the error is what makes this serious. Because density is a
count, missed detections do not produce a conspicuous failure — they produce a
*lower* density reading, so the system reports a dangerously packed crowd as
safer than it is. Degradation is therefore worst in precisely the regime the
system exists to monitor, and any deployment must treat a measurement above the
ceiling as unreliable rather than as a low number.

This is the project's defining limitation and it is measured rather than
assumed. It needs to appear in the findings section *and* in limitations,
because a reader needs it to interpret everything downstream of the detector.

## 6. The measurement stack against ground truth

Four CroHD scenes — **three of them training scenes and one held out**, so this
measures agreement with ground truth rather than generalisation.

- Counting is biased low by a consistent **14% (sd 3.4 percentage points)**. A
  stable bias is correctable; an erratic one is not.
- The held-out scene scores **0.869**, above the four-scene mean of **0.857**,
  while the lowest ratio (**0.799**) belongs to a scene the model did train on —
  so scene difficulty dominates memorisation here, which is what makes the
  pooled figure quotable at all.
- Track-derived speeds match ground-truth pedestrian speeds to within **2–5%**,
  with correlation up to **0.996**.
- Counting *without* tracking performs better still (**0.977** ratio, **4%**
  mean absolute error), establishing that detection should be used for counting
  and tracking only for velocity, since the tracker discards detections it
  cannot associate.

Four scenes establish that the bias is consistent, not that it is consistent
everywhere.

---

## Not yet placed

Two figures were in Methods §4 before this move and are not reproduced above,
because they describe the training setup rather than a finding. They are in
`results/README.md` if wanted:

- the control training configurations (2,730 images, ~30k/~45k boxes, 15
  epochs, batch 4);
- GPU headroom during training (2.4 GB of 6 GB). The binding constraint it
  illustrated — system RAM at 7.6 GB — is kept in Methods §4.

---

# Moved out of Methods §5

The same treatment, applied to the work-plan section. Methods §5 now describes
what the two decision rules *are* and why there are two of them; what they
actually did on footage is below.

## 7. The scene-relative threshold, as measured

Methods §5 says the threshold is computed from a high percentile of an initial
learning period, and names the five parameters that produce it. The values:

- On **HT21-03** the learning period produced a threshold of **101 people**,
  from the **85th percentile of the first 15 s**. ELEVATED then required
  occupancy above 101 **and** a net rate above **0.3 people/s**, sustained for
  **2 s**, with a **3 s** clear time.
- The dwell and clear times are not arbitrary. Without them the level
  **oscillated nineteen times in eleven seconds**, because occupancy sits on
  the threshold for extended periods. This is the evidence for the hysteresis,
  and it is worth reporting as a result rather than asserting as a design note.
- The learning window's weakness is measured, not hypothesised. On **HT21-02**
  the baseline landed at **102** against a clip mean of **85** — an
  unrepresentative quiet period set the bar too high, which is part of why that
  sequence never escalated. A longer or explicitly quiet learning period would
  be more robust.

## 8. Demonstration on real footage (scene-relative rule)

CroHD sequence **HT21-03**, a street crossing where pedestrians accumulate at a
signal and discharge. The complete pipeline escalated:

| event | time |
|---|---|
| NORMAL → WATCH | **15.1 s** |
| WATCH → ELEVATED | **17.1 s** |
| stood down | **29.8 s** |

The flow measurements show why: inflow of **16.2 people/s** against outflow of
**9.2** at escalation, converging to **14.6** against **12.4** at stand-down.

This uses the trained model, real detections and real tracking, with no
calibration and no borrowed thresholds.

**Carry the qualification from Methods §5**: HT21-03 is a detector *training*
scene. The demonstration establishes that the occupancy, flow-balance and
escalation logic behave correctly on real crowd video — none of which was
fitted to this scene — not that the detector generalises.

## 9. Demonstration on the fabricated scenario (calibrated rule)

A synthetic clip with **declared** geometry: a **6.0 × 5.0 m** region, recovered
exactly by the homography, in which a bottleneck fills faster than it empties.
Watermarked on every frame; geometry asserted rather than surveyed; detection
replayed from ground truth.

On it the calibrated pipeline reports density in people/m², applies the Fruin
bands, escalates to **WARNING at 27.7 s**, and de-escalates at **53.1 s** when
the crowd — still at **level-of-service E** — stops deteriorating.

## 10. What the pair supports (analysis)

The real demonstration establishes that the measurement and decision stack
functions on genuine crowds. The fabricated scenario establishes what the same
system reports when calibration is available and conditions deteriorate.

Together they support the claim that **given surveyed footage of a deteriorating
crowd, the system would produce the calibrated output shown in the scenario** —
while making clear that this rests on demonstrated behaviour under specified
conditions, not on evidence about real disasters.

This is the strongest claim the project can make, and the sentence above is
deliberately constructed to not exceed it. Do not let it drift into "the system
detects crowd crushes".
