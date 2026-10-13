# Calibration and threshold-setting

Everything downstream is a count divided by an area. If the area is wrong, every
density is wrong by the same factor and nothing in the pipeline will complain.
Calibration is the step most worth doing carefully and documenting fully.

## 1. On site, before filming

Measure and write down, in a logbook you can cite:

- **The reference shape.** Lay tape, or pick four features on the ground whose
  spacing you can measure — paving joints, line markings, bollard bases. Record
  the real distances. A rectangle 3–5 m on a side, covering a good fraction of
  the region, is ideal. Small shapes amplify clicking error across the frame.
- **The camera height** above the ground plane, if you intend to use the head
  detector (see §5).
- **The nadir**: where a plumb line from the camera meets the ground, expressed
  in the same world coordinates as the reference shape. Mark it, or measure its
  offset from a reference corner.
- **Anything that can move.** If the camera is on a tripod that gets bumped,
  the calibration is void from that moment on.

Take a still of the empty scene with the tape visible. You will calibrate from
it, and it is also the figure for the report.

## 2. Fit the homography

```bash
python tools/pick_points.py --mode calibration \
    --video data/raw/site1.mp4 --frame 0 \
    --width 4.0 --height 3.0 \
    --out configs/calibration_site1.json
```

Click the four corners **in order around the shape**, not diagonally. The tool
prints the reprojection error.

> With exactly four points the residual is zero by construction. That means the
> fit is exact, **not** that the calibration is accurate. Accuracy comes from
> your tape measure.

To get a residual that means something, measure more than four points and pass
them explicitly:

```bash
python tools/pick_points.py --mode calibration \
    --video data/raw/site1.mp4 --frame 0 \
    --world "0,0;4,0;4,3;0,3;2,0;2,3" \
    --out configs/calibration_site1.json
```

Now the printed RMS and max residuals are real. Report them. They bound the
precision of every density figure you quote — if the max residual is comparable
to a person's own footprint, calibration rather than detection is your limiting
factor, and you should say so.

## 3. Choose the region

```bash
python tools/pick_points.py --mode region \
    --video data/raw/site1.mp4 --frame 0 \
    --calibration configs/calibration_site1.json
```

Guidance:

- **Keep it inside the calibrated quadrilateral.** A homography extrapolated
  beyond its fitting points goes wrong quietly and fast.
- **Pick the actual constriction**, not a generous area around it. Density
  averaged over a large region hides the local peak that matters.
- **Avoid the far field.** Where the ground plane approaches the horizon, one
  pixel covers metres and both detection and projection fall apart.
- Sanity-check the printed area. The tool tells you how many people 4 people/m²
  implies. If that number looks absurd for the view, the calibration is wrong.

Paste the printed polygon and area into `configs/<site>.yaml`.

## 4. Verify visually

Run the pipeline with an overlay and look at it:

```bash
python scripts/run_pipeline.py --config configs/site1.yaml \
    --video data/raw/site1.mp4 --out runs/site1 --max-frames 300
```

Check in `runs/site1/overlay.mp4` that the region outline sits **on the ground**
where you measured it, and that the dot under each box lands at the person's
feet. This catches mistakes no unit test will.

## 5. Head-height correction (only with the head detector)

During the pretrained bootstrap the reference point is `box_bottom` — the feet,
which lie on the calibrated plane. No correction applies; do not configure one.

Once the head detector is in use, the reference point becomes the head, roughly
1.7 m off the plane, and the long-projection error described in
`limitations.md` §3 applies. Add to your calibration file:

```json
"height_correction": {
  "nadir": [2.1, -3.4],
  "camera_height_m": 6.0,
  "object_height_m": 1.7
}
```

`nadir` is in the same world coordinates as `world_points`. It will usually lie
*outside* the region, often behind it, so negative coordinates are normal and
expected.

To check it is right: pick a person standing on a known mark, and confirm the
corrected world position matches the mark. Without the correction it will be
metres long.

## 6. Setting `pressure_slope_threshold`

This is the one number with no citable source, because crowd pressure is scale
dependent — region size, frame rate, velocity lag and detector all change its
magnitude. Set it from your own data, and document the procedure:

1. Run the pipeline over your calmest footage with the threshold set absurdly
   high, so nothing ever triggers.
2. From `frames.csv`, take the distribution of `pressure_slope` over that calm
   footage. This is your noise floor.
3. Set the threshold well above it — a high percentile (95th–99th) of the calm
   distribution is a defensible starting point, because it bounds the
   false-positive rate on quiet data at 1–5% of frames *before* the dwell time
   is applied.
4. Re-run on your busiest footage and check the warnings land where a human
   watching the video would agree something changed.
5. Record the percentile, the calm clip it came from, and the resulting
   threshold. Present it as a tuned parameter, because it is one.

`min_dwell_s` and `clear_s` then control flapping. The dwell time is what makes
step 3 conservative: a 1–5% per-frame rate becomes far rarer once two seconds
of *continuous* exceedance is required.

## 7. Re-calibrate when

- the camera is moved, bumped, or re-mounted;
- zoom or focal length changes;
- the ground surface changes (resurfacing, temporary staging, barriers);
- you cannot account for a sudden step in measured density.

Keep every calibration file in git. They are experimental records, and a reader
should be able to tell which one produced which figure.
