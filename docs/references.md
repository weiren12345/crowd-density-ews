# References — TO BE VERIFIED

> **Status: no entry below has been checked against a primary source.** They
> were written from recollection to record what needs citing and where each
> claim is used in the code. The level-of-service bands have since been
> cross-checked against two secondary sources (see below), which is weaker.
> Before anything here goes in the proposal, find the actual source, confirm
> the numbers and the wording, and replace the entry with a full citation in
> your unit's required style.
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

**Citation to use:** Fruin, J. J. (1971). *Pedestrian Planning and Design*.
Metropolitan Association of Urban Designers and Environmental Planners, New
York. Table number still to be recorded.

**Cross-check, 2026-10-30.** Two secondary sources were consulted. Neither is
citable in the report — both are commercial crowd-safety vendors — but the
check is worth recording:

- A vendor knowledge page (movetos.com) publishes the **queuing** table in
  metric. Its density column matches `configs/thresholds.yaml` to rounding,
  and its m² figures are the 2-decimal roundings of 13 / 10 / 7 / 3 / 2 ft²,
  which independently supports the imperial band edges the config converts
  from. It gives the full Fruin citation above. It also cites G. K. Still and
  the HSE *Event Safety Guide*; see the leads below.
- An earlier page (crowdmanagementindia.com) gave the **walkway** table
  instead (LOS F > 2.17 people/m²) and contradicted itself, quoting ~5.4
  people/m² for LOS F in a summary box while its own table said 2.17. It
  appears to have conflated the two tables. Not used.

**This is corroboration, not verification.** Both readings are secondary, and
the 1971 book has not been opened. Until it is, the writeup must either cite a
source someone has actually read or state that the bands are adopted from
secondary sources.

Still to do:
- consult the primary source and record the edition and table number, **or**
  state the adoption explicitly in the writeup;
- confirm each band edge in the original units (ft² per person);
- decide and justify which table applies. A bottleneck sits between a queue and
  a walkway. The project uses the **queuing** table; the stated reason should be
  that the escalation case is an accumulating, near-stationary crowd.

**Better leads for a citable secondary source**, both named by the movetos page
and both likely held by the university library:

- Still, G. K. (2014). *Introduction to Crowd Science*. CRC Press. Still is a
  recognised authority on crowd safety; a textbook a marker will accept.
- Still, G. K. (2000). *Crowd Dynamics*. PhD thesis, University of Warwick.
- Health and Safety Executive (1999). *The Event Safety Guide* ("Purple
  Guide"). HSE Books. Quoted as giving a maximum of 2 people/m² for open event
  areas — worth having, as a regulatory figure, but note that the Purple Guide
  is now published by the Events Industry Forum rather than HSE, so check which
  edition you are citing.

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

## Datasets

**Used in:** training and evaluation. Three datasets, in different roles — see
`docs/datasets.md` for provenance and the reproduction path.

- **CroHD / HeadTracking21** — the delivered model's training data. Cite the
  dataset paper (Sundararaman et al., *Tracking Pedestrian Heads in Dense
  Crowd*, CVPR 2021); confirm the author list and venue.
- **SCUT-HEAD, Parts A and B** — the two control models. Both parts are
  *training* data; Part B was moved out of the held-out role early on (§2 of
  the methods draft). Do not describe Part B as held out.
- **ShanghaiTech Part A** — evaluation only, never trained on. Cite Zhang et
  al., *Single-Image Crowd Counting via Multi-Column CNN*, CVPR 2016; confirm.

To verify for each:
- the dataset paper and the correct citation;
- the licence, and whether it permits your use and any redistribution;
- the exact splits used. These are recorded in `results/README.md` and §4 of
  the methods draft; keep the three accounts consistent.

### Considered and not used

Named in the original proposal, described in §1 of the methods draft, never
downloaded or run.

- **UMN** ("Unusual Crowd Activity"). Cite Mehran, Oyama and Shah, *Abnormal
  Crowd Behavior Detection Using Social Force Model*, CVPR 2009, pp. 935–942,
  doi:10.1109/CVPR.2009.5206641 — confirm.
- **PETS 2009.** Cite Ferryman and Shahrokni, *PETS2009: Dataset and
  Challenge*, IEEE Int. Workshop on Performance Evaluation of Tracking and
  Surveillance, 2009 — confirm author list and exact title.

> **VERIFY — weaker provenance than the entries above.** Both dataset homepages
> were dead when checked on 2026-10-31: the PETS host `cvg.reading.ac.uk` no
> longer resolves, and `mha.cs.umn.edu` served a rejected certificate. The
> descriptive details in §1 therefore come from **papers citing these datasets,
> not from the dataset authors**. Specifically unconfirmed: UMN's scene and
> segment counts (3 scenes / 11 segments / 7,740 frames) and its resolution
> (sources disagree between 320×240 and 640×480 — the draft avoids quoting
> either); and whether PETS's published annotation sets are official or
> third-party, which the draft hedges. PETS's Tsai calibration is corroborated
> by multiple independent papers, one of which also reports the calibration is
> imperfect in joint-consistency terms and that the providers acknowledged
> synchronisation problems — worth knowing if anyone proposes using it.
>
> If either dataset is going to be named in the submitted report, find a live
> authoritative page or the dataset paper itself and confirm the details, or
> cut the specifics and keep only the reasoning.

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
