# References — TO BE VERIFIED

> **Status: none of the entries below have been checked against a primary
> source.** They were written from recollection to record what needs citing and
> where each claim is used in the code. Before anything here goes in the
> proposal, find the actual source, confirm the numbers and the wording, and
> replace the entry with a full citation in your unit's required style.
>
> Do not cite this file, and do not cite source comments or
> `configs/thresholds.yaml`. Cite the works themselves.

## Level-of-service bands

**Used in:** `configs/thresholds.yaml`, which drives every warning decision.

Fruin's pedestrian level-of-service classes. The values in the config are the
**queuing / waiting** classes, given in the original in square feet per person
and converted to people/m² (1 ft² = 0.092903 m²). The conversion arithmetic is
shown in the config header so it can be rechecked independently of the source
values.

To verify:
- the primary source and its edition (commonly attributed to J. J. Fruin,
  *Pedestrian Planning and Design*, 1971);
- that you are reading the **queuing** table, not the **walkway** table — they
  differ, and walkway LOS is expressed in flow terms;
- each band edge in the original units;
- whether a later standard (a transport authority handbook, or a national
  standard on pedestrian facilities) is a more appropriate citation for an
  engineering report than the 1971 original.

Also decide and justify which table applies. A bottleneck sits between a queue
and a walkway; the writeup should say which you chose and why.

## Crowd pressure / turbulence

**Used in:** `src/cdw/turbulence.py`, the second of the two warning signals.

The measure implemented is local density multiplied by the variance of local
velocity, attributed to work by Helbing, Johansson and Al-Abideen analysing
video of the 2006 Mina/Jamarat crowd disaster.

To verify:
- the exact definition of "crowd pressure" in the paper, including how the
  local density and local velocity variance are each estimated, and over what
  neighbourhood;
- whether the variance is of the velocity vector or of its magnitude;
- the units, and any smoothing applied before the product is formed;
- whether the paper reports a threshold value, and if so under what
  normalisation — our `pressure_slope_threshold` is a tuned parameter and must
  not be presented as theirs.

If the implementation here differs from the paper, say so explicitly and
describe the difference. An honest "we implement a variant of" is fine; an
unstated deviation is not.

## Detection and tracking

**Used in:** `src/cdw/detect.py`, `scripts/train_head_detector.py`.

- Ultralytics YOLOv8 — cite the software with its version, and the Ultralytics
  documentation or repository as appropriate.
- ByteTrack — cite the tracking paper, not just the Ultralytics config file.
- Record the exact `ultralytics` version you trained and evaluated with, and
  pin it in `requirements.txt` once results are final. Behaviour changes
  between minor releases.

## Dataset

**Used in:** training.

SCUT-HEAD, Part A (Part B held out as a separate domain).

To verify:
- the dataset paper and the correct citation;
- the licence, and whether it permits your use and any redistribution;
- the exact train/val split you used, recorded in the repository so results are
  reproducible.

## Optical flow

**Used in:** `src/cdw/turbulence.py`, the fallback motion signal.

Farnebäck's two-frame dense optical flow algorithm, via OpenCV's
`calcOpticalFlowFarneback`. Cite the original method and the OpenCV version.

## General background worth having

Claims likely to appear in the introduction that will each need a source:

- that crowd crushes arise from density and flow conditions rather than from
  panic — the "panic" framing is contested in the literature and should not be
  asserted casually;
- any figure quoted for injuries or deaths in a specific incident;
- any claim that early warning improves outcomes. This is the project's
  motivating premise and the easiest one to overstate. Find a source, or soften
  the claim.
