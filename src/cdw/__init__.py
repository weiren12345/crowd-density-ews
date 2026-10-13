"""Crowd density early-warning system.

Stage order matches the pipeline:

    detect -> (track) -> calibrate -> density -> turbulence -> warning

Only ``detect`` contains a learned model. Everything after it is deterministic
Python, which is what makes the warning logic auditable.
"""

__version__ = "0.1.0"

__all__ = [
    "calibrate",
    "density",
    "detect",
    "pipeline",
    "turbulence",
    "viz",
    "warning",
]
