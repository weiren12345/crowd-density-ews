"""Measure where detection recall collapses, and what that does to density.

Produces the central honest-limitation result of the project:

1. detected/true count ratio against TRUE density (hand counts, not estimates),
2. a confidence-threshold sweep showing the recall/precision-ish trade-off,
3. inter-annotator agreement on the doubly-annotated subset, which bounds how
   much of the curve is real and how much is annotation noise,
4. optionally, a fitted debiasing correction and what it does to the residual.

    python scripts/eval_recall.py --frames data/frames/site1 --out runs/recall_site1

Reads from --frames: manifest.json, detections.json, annotations_*.json.

Read the caveats in docs/limitations.md before quoting any number from here.
The ratio is a count ratio, not a detection-level recall: it does not know
whether a detected head is the *same* head a human marked. It is the right
quantity for a density system, and the wrong one for a detection paper.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

CONF_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]


def load_annotations(frames_dir: Path):
    """{annotator: {frame_index: count}} from every annotations_*.json present."""
    out = {}
    for path in sorted(frames_dir.glob("annotations_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        counts = {
            int(k): v["count"]
            for k, v in data["frames"].items()
            if not v.get("skipped")
        }
        if counts:
            out[data.get("annotator", path.stem)] = counts
    return out


def consensus(annotations):
    """Mean count per frame across annotators who saw it."""
    per_frame = {}
    for counts in annotations.values():
        for frame, c in counts.items():
            per_frame.setdefault(frame, []).append(c)
    return {f: float(np.mean(v)) for f, v in per_frame.items()}


def agreement(annotations):
    """Pairwise agreement on the frames two annotators both counted."""
    reports = []
    for a, b in combinations(sorted(annotations), 2):
        shared = sorted(set(annotations[a]) & set(annotations[b]))
        if len(shared) < 2:
            continue
        xa = np.array([annotations[a][f] for f in shared], float)
        xb = np.array([annotations[b][f] for f in shared], float)
        diff = xa - xb
        mean = (xa + xb) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(mean > 0, np.abs(diff) / mean, 0.0)
        reports.append({
            "pair": [a, b],
            "n_frames": len(shared),
            "mean_abs_diff": float(np.abs(diff).mean()),
            "mean_relative_diff": float(rel.mean()),
            "bias_a_minus_b": float(diff.mean()),
            "limits_of_agreement": [
                float(diff.mean() - 1.96 * diff.std(ddof=1)),
                float(diff.mean() + 1.96 * diff.std(ddof=1)),
            ],
            "pearson_r": float(np.corrcoef(xa, xb)[0, 1]) if len(shared) > 2 else None,
            "max_abs_diff": float(np.abs(diff).max()),
        })
    return reports


def counts_at_conf(detections, frame, conf):
    return sum(1 for d in detections.get(str(frame), []) if d["conf"] >= conf)


def curve(truth, detections, area_m2, conf, edges):
    """Detected/true ratio per TRUE-density bin at one confidence threshold."""
    rows = []
    for lo, hi in zip(edges, edges[1:]):
        frames = [f for f, t in truth.items() if lo <= t / area_m2 < hi]
        if not frames:
            continue
        true_counts = np.array([truth[f] for f in frames], float)
        det_counts = np.array([counts_at_conf(detections, f, conf) for f in frames], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(true_counts > 0, det_counts / true_counts, np.nan)
        rows.append({
            "bin": [lo, hi],
            "n_frames": len(frames),
            "mean_true_density": float(true_counts.mean() / area_m2),
            "mean_true_count": float(true_counts.mean()),
            "mean_detected_count": float(det_counts.mean()),
            # Pooled: total detected over total true. Less noisy than the mean
            # of per-frame ratios when some frames hold very few people.
            "pooled_ratio": float(det_counts.sum() / true_counts.sum())
            if true_counts.sum() > 0 else float("nan"),
            "mean_ratio": float(np.nanmean(ratios)) if len(ratios) else float("nan"),
            "std_ratio": float(np.nanstd(ratios)) if len(ratios) else float("nan"),
        })
    return rows


def usable_ceiling(rows, threshold=0.8):
    """Highest true density at which the pooled ratio still clears `threshold`."""
    ok = [r for r in rows if r["pooled_ratio"] >= threshold]
    return max((r["bin"][1] for r in ok), default=None)


def fit_correction(truth, detections, area_m2, conf):
    """Least-squares fit of true_count = a * detected + b * detected^2.

    Through the origin on purpose: zero detections must imply zero people, or
    the correction invents a crowd out of an empty frame. Quadratic because
    under-counting worsens with density. This is a calibration curve for one
    camera and one detector, not a general law - refit it per site, and never
    extrapolate past the densities you actually annotated.
    """
    frames = sorted(truth)
    if len(frames) < 5:
        return None
    x = np.array([counts_at_conf(detections, f, conf) for f in frames], float)
    y = np.array([truth[f] for f in frames], float)
    design = np.column_stack([x, x**2])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coeffs
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "a": float(coeffs[0]),
        "b": float(coeffs[1]),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rms_error_people": float(np.sqrt(((y - pred) ** 2).mean())),
        "rms_error_density": float(np.sqrt(((y - pred) ** 2).mean()) / area_m2),
        "raw_rms_error_people": float(np.sqrt(((y - x) ** 2).mean())),
        "max_detected_count_fitted": float(x.max()),
        "form": "true ~ a*detected + b*detected^2, through the origin",
    }


def plot(rows_by_conf, out_path, area_m2):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for conf, rows in sorted(rows_by_conf.items()):
        if not rows:
            continue
        xs = [r["mean_true_density"] for r in rows]
        ys = [r["pooled_ratio"] for r in rows]
        ax1.plot(xs, ys, marker="o", label=f"conf {conf:.2f}")

    ax1.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax1.axhline(0.8, color="crimson", lw=0.8, ls=":")
    ax1.set_xlabel("true density (people/m$^2$, hand counted)")
    ax1.set_ylabel("detected / true")
    ax1.set_title("Count ratio against true density")
    ax1.set_ylim(0, 1.25)
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    confs = sorted(rows_by_conf)
    for lo, hi in {tuple(r["bin"]) for rows in rows_by_conf.values() for r in rows}:
        ys, xs = [], []
        for c in confs:
            match = [r for r in rows_by_conf[c] if tuple(r["bin"]) == (lo, hi)]
            if match:
                xs.append(c)
                ys.append(match[0]["pooled_ratio"])
        if xs:
            ax2.plot(xs, ys, marker="s", label=f"{lo:g}-{hi:g} p/m$^2$")

    ax2.set_xlabel("confidence threshold")
    ax2.set_ylabel("detected / true")
    ax2.set_title("Threshold sweep per density band")
    ax2.set_ylim(0, 1.25)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Detector recall against hand counts (region {area_m2:.1f} m$^2$)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="directory from sample_frames.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--edges", default="0,1,2,3,4,5,99")
    ap.add_argument("--ratio-floor", type=float, default=0.8,
                    help="ratio defining the usable ceiling")
    args = ap.parse_args()

    frames_dir = Path(args.frames)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = [float(e) for e in args.edges.split(",")]

    det_blob = json.loads((frames_dir / "detections.json").read_text(encoding="utf-8"))
    detections = det_blob["frames"]
    area_m2 = float(det_blob["meta"]["region_area_m2"])

    annotations = load_annotations(frames_dir)
    if not annotations:
        print(f"no annotations_*.json in {frames_dir}; run tools/count_heads.py first")
        return 1
    truth = consensus(annotations)
    print(f"{len(annotations)} annotator(s), {len(truth)} frames with ground truth")

    agree = agreement(annotations)
    for rep in agree:
        print(f"  {rep['pair'][0]} vs {rep['pair'][1]}: n={rep['n_frames']}, "
              f"mean abs diff {rep['mean_abs_diff']:.2f} people "
              f"({rep['mean_relative_diff']*100:.1f}%), "
              f"bias {rep['bias_a_minus_b']:+.2f}")
    if len(annotations) > 1 and not agree:
        print("  (no doubly-annotated frames yet)")

    rows_by_conf = {c: curve(truth, detections, area_m2, c, edges) for c in CONF_GRID}

    default_conf = 0.25
    headline = rows_by_conf[default_conf]
    ceiling = usable_ceiling(headline, args.ratio_floor)
    correction = fit_correction(truth, detections, area_m2, default_conf)

    best_conf = max(
        CONF_GRID,
        key=lambda c: usable_ceiling(rows_by_conf[c], args.ratio_floor) or 0,
    )

    report = {
        "region_area_m2": area_m2,
        "n_frames": len(truth),
        "annotators": {k: len(v) for k, v in annotations.items()},
        "agreement": agree,
        "ratio_floor": args.ratio_floor,
        "headline_conf": default_conf,
        "usable_ceiling_density": ceiling,
        "best_conf_by_ceiling": best_conf,
        "best_conf_ceiling": usable_ceiling(rows_by_conf[best_conf], args.ratio_floor),
        "curve_by_conf": rows_by_conf,
        "debias_correction": correction,
        "caveats": [
            "Count ratio, not detection-level recall: identity of detections is not matched to annotations.",
            "Strata were defined by detector output, so bin membership is itself biased.",
            "Ground truth is human and imperfect; see agreement for its own spread.",
            "The correction is fitted per camera and detector; do not extrapolate beyond the annotated range.",
        ],
    }
    (out_dir / "recall_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nat conf {default_conf}:")
    for r in headline:
        print(f"  {r['bin'][0]:4.1f}-{r['bin'][1]:4.1f} p/m2  n={r['n_frames']:3d}  "
              f"true {r['mean_true_count']:5.1f}  detected {r['mean_detected_count']:5.1f}  "
              f"ratio {r['pooled_ratio']:.3f}")
    print(f"\nusable ceiling (ratio >= {args.ratio_floor}): "
          f"{ceiling if ceiling else 'below the lowest bin'} people/m2")
    print(f"best confidence by that criterion: {best_conf}")
    if correction:
        print(f"debias fit: true ~ {correction['a']:.3f}*d + {correction['b']:.5f}*d^2, "
              f"R2={correction['r2']:.3f}, rms {correction['rms_error_people']:.2f} people "
              f"(raw {correction['raw_rms_error_people']:.2f})")

    plot(rows_by_conf, out_dir / "recall_curve.png", area_m2)
    print(f"wrote {out_dir/'recall_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
