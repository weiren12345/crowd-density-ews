"""Run the full pipeline over a video and write a CSV, a summary and an overlay.

Usage:
    python scripts/run_pipeline.py --config configs/synthetic.yaml \
        --video data/raw/synthetic_bottleneck.mp4 --out runs/synthetic

Outputs in --out:
    frames.csv           one row per processed frame
    frames.summary.json  calibration error, region area, settings
    overlay.mp4          annotated video (skip with --no-overlay)
    episodes.json        contiguous WARNING episodes with start/end times
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdw.pipeline import load_config, run_video  # noqa: E402


def episodes(records, level="WARNING"):
    """Contiguous runs at a given level, as start/end/peak summaries."""
    out, current = [], None
    for r in records:
        if r.level == level:
            if current is None:
                current = {
                    "start_t": r.t, "end_t": r.t,
                    "peak_density": r.density, "peak_pressure": r.pressure,
                    "frames": 1,
                }
            else:
                current["end_t"] = r.t
                current["frames"] += 1
                current["peak_density"] = max(current["peak_density"], r.density)
                current["peak_pressure"] = max(current["peak_pressure"], r.pressure)
        elif current is not None:
            current["duration_s"] = round(current["end_t"] - current["start_t"], 2)
            out.append(current)
            current = None
    if current is not None:
        current["duration_s"] = round(current["end_t"] - current["start_t"], 2)
        out.append(current)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)

    last = {"level": None}

    def progress(rec):
        if args.quiet:
            return
        if rec.level != last["level"]:
            print(f"  t={rec.t:6.2f}s  {last['level'] or 'start'} -> {rec.level}"
                  f"   density={rec.density:.2f}  slope={rec.pressure_slope:+.4f}"
                  f"   ({rec.reason})")
            last["level"] = rec.level

    records, summary = run_video(
        cfg,
        args.video,
        out_csv=out_dir / "frames.csv",
        out_video=None if args.no_overlay else out_dir / "overlay.mp4",
        stride=args.stride,
        max_frames=args.max_frames,
        progress=progress,
    )

    eps = episodes(records)
    (out_dir / "episodes.json").write_text(json.dumps(eps, indent=2), encoding="utf-8")

    if not records:
        print("no frames processed")
        return 1

    peak = max(records, key=lambda r: r.density)
    cal = summary["calibration_error"]
    print()
    print(f"region          {summary['region']}  {summary['region_area_m2']} m2")
    if cal.get("fitted", True) and cal.get("rms_m") is not None:
        print(f"calibration     rms {cal['rms_m']*100:.1f} cm, "
              f"max {cal['max_m']*100:.1f} cm")
    else:
        print("calibration     supplied matrix, no residuals available")
    print(f"frames          {summary['frames_processed']} at {summary['fps']:g} fps"
          f" (stride {summary['stride']})")
    print(f"peak density    {peak.density:.2f} people/m2 (LOS {peak.los}) at t={peak.t:.2f}s")
    print(f"levels          " + ", ".join(
        f"{lvl}={sum(1 for r in records if r.level == lvl)}"
        for lvl in ("NORMAL", "WATCH", "WARNING")
    ))
    print(f"warning episodes {len(eps)}")
    for e in eps:
        print(f"  {e['start_t']:6.2f}s - {e['end_t']:6.2f}s  ({e['duration_s']}s)"
              f"  peak density {e['peak_density']:.2f}, peak pressure {e['peak_pressure']:.4f}")
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
