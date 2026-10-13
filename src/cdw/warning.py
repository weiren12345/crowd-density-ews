"""Warning logic: a two-signal rule over density and crowd pressure.

Deliberately simple and fully inspectable. Every number it uses comes from a
config file with a citation field, so the proposal can defend each threshold by
pointing at a source rather than at a tuned constant.

The rule is: raise a warning when density sits in a crowded level-of-service
band AND crowd pressure is rising. Either signal alone is a poor predictor - a
packed but calmly queueing platform is not dangerous, and a brief pressure
spike in a sparse crowd is noise. Requiring both, sustained for a dwell time,
is what keeps the false-positive rate tolerable.

What this does NOT do: predict a crush. It reports that measured conditions
resemble those documented before past flow breakdowns. See docs/limitations.md.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import yaml

__all__ = ["LOSBand", "WarningLevel", "WarningRule", "WarningSample", "load_thresholds"]


class WarningLevel(str, Enum):
    NORMAL = "NORMAL"
    WATCH = "WATCH"      # density is in a crowded band, pressure is not rising
    WARNING = "WARNING"  # both signals present and sustained


@dataclass(frozen=True, slots=True)
class LOSBand:
    """One Fruin level-of-service band, in people per square metre."""

    name: str
    min_density: float
    max_density: float = float("inf")
    description: str = ""

    def contains(self, density: float) -> bool:
        return self.min_density <= density < self.max_density


@dataclass(slots=True)
class WarningSample:
    t: float
    level: WarningLevel
    los: str
    density: float
    pressure: float
    pressure_slope: float      # (people/m^2)(m/s)^2 per second
    density_signal: bool
    pressure_signal: bool
    held_for_s: float
    reason: str


def load_thresholds(path) -> dict:
    """Read configs/thresholds.yaml."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


class WarningRule:
    """Stateful two-signal rule with a trend window and hysteresis.

    Parameters
    ----------
    bands:
        Level-of-service bands, ascending by density.
    escalate_from:
        Name of the lowest band that counts as "crowded". Density at or above
        this band satisfies the density signal.
    pressure_slope_threshold:
        Pressure must be rising faster than this for the pressure signal to
        hold. Units: (people/m^2)(m/s)^2 per second.
    trend_window_s:
        Window over which the slope is fitted by least squares.
    min_dwell_s:
        Both signals must hold continuously for this long before WARNING is
        raised. Suppresses single-frame coincidences.
    clear_s:
        Once raised, conditions must be absent this long before standing down.
        Prevents the warning flickering on and off at the boundary.
    """

    def __init__(
        self,
        bands: list[LOSBand],
        escalate_from: str = "D",
        pressure_slope_threshold: float = 0.02,
        trend_window_s: float = 5.0,
        min_dwell_s: float = 2.0,
        clear_s: float = 5.0,
    ):
        if not bands:
            raise ValueError("at least one level-of-service band is required")
        self.bands = sorted(bands, key=lambda b: b.min_density)
        names = [b.name for b in self.bands]
        if escalate_from not in names:
            raise ValueError(f"escalate_from={escalate_from!r} is not one of {names}")
        self._escalate_index = names.index(escalate_from)
        self.pressure_slope_threshold = pressure_slope_threshold
        self.trend_window_s = trend_window_s
        self.min_dwell_s = min_dwell_s
        self.clear_s = clear_s

        self._pressure_hist: deque[tuple[float, float]] = deque()
        self._both_since: float | None = None
        self._clear_since: float | None = None
        self._level = WarningLevel.NORMAL

    # -- classification -----------------------------------------------------

    def band_for(self, density: float) -> LOSBand:
        for band in reversed(self.bands):
            if band.contains(density):
                return band
        return self.bands[0]

    def _density_signal(self, density: float) -> bool:
        band = self.band_for(density)
        return [b.name for b in self.bands].index(band.name) >= self._escalate_index

    def pressure_slope(self, t: float, pressure: float) -> float:
        """Least-squares slope of pressure over the trend window."""
        self._pressure_hist.append((t, pressure))
        cutoff = t - self.trend_window_s
        while self._pressure_hist and self._pressure_hist[0][0] < cutoff:
            self._pressure_hist.popleft()
        if len(self._pressure_hist) < 3:
            return 0.0
        ts = np.array([a for a, _ in self._pressure_hist])
        ps = np.array([b for _, b in self._pressure_hist])
        span = ts[-1] - ts[0]
        if span < 1e-6:
            return 0.0
        return float(np.polyfit(ts, ps, 1)[0])

    # -- main update --------------------------------------------------------

    def update(self, t: float, density: float, pressure: float) -> WarningSample:
        slope = self.pressure_slope(t, pressure)
        band = self.band_for(density)
        d_sig = self._density_signal(density)
        p_sig = slope > self.pressure_slope_threshold
        both = d_sig and p_sig

        if both:
            self._clear_since = None
            if self._both_since is None:
                self._both_since = t
        else:
            self._both_since = None
            if self._clear_since is None:
                self._clear_since = t

        held = (t - self._both_since) if self._both_since is not None else 0.0

        if self._level is WarningLevel.WARNING:
            # Stand down only after conditions have been absent for clear_s.
            if self._clear_since is not None and (t - self._clear_since) >= self.clear_s:
                self._level = WarningLevel.WATCH if d_sig else WarningLevel.NORMAL
        elif both and held >= self.min_dwell_s:
            self._level = WarningLevel.WARNING
        elif d_sig:
            self._level = WarningLevel.WATCH
        else:
            self._level = WarningLevel.NORMAL

        return WarningSample(
            t=t,
            level=self._level,
            los=band.name,
            density=density,
            pressure=pressure,
            pressure_slope=slope,
            density_signal=d_sig,
            pressure_signal=p_sig,
            held_for_s=held,
            reason=self._reason(self._level, band, d_sig, p_sig, held),
        )

    def _reason(self, level, band, d_sig, p_sig, held) -> str:
        if level is WarningLevel.WARNING:
            if d_sig and p_sig:
                return (
                    f"density in LOS {band.name} and pressure rising, "
                    f"sustained {held:.1f}s"
                )
            # Still raised, but only because of hysteresis. Say so rather than
            # reporting signals that are no longer present - the CSV is
            # evidence, and it should not claim a rise that has stopped.
            missing = "pressure no longer rising" if d_sig else f"density fell to LOS {band.name}"
            waited = (self._clear_since is not None) and (held == 0.0)
            return f"warning held during stand-down ({missing})" if waited else missing
        if level is WarningLevel.WATCH:
            return f"density in LOS {band.name}, pressure not rising"
        if p_sig and not d_sig:
            return f"pressure rising but density only LOS {band.name}"
        return f"LOS {band.name}"

    def reset(self) -> None:
        self._pressure_hist.clear()
        self._both_since = None
        self._clear_since = None
        self._level = WarningLevel.NORMAL

    # -- construction from config ------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "WarningRule":
        bands = [
            LOSBand(
                name=b["name"],
                min_density=float(b["min_density"]),
                max_density=float(b.get("max_density", float("inf"))),
                description=b.get("description", ""),
            )
            for b in cfg["los_bands"]
        ]
        rule = cfg.get("rule", {})
        return cls(
            bands=bands,
            escalate_from=rule.get("escalate_from", "D"),
            pressure_slope_threshold=float(rule.get("pressure_slope_threshold", 0.02)),
            trend_window_s=float(rule.get("trend_window_s", 5.0)),
            min_dwell_s=float(rule.get("min_dwell_s", 2.0)),
            clear_s=float(rule.get("clear_s", 5.0)),
        )

    @classmethod
    def from_file(cls, path) -> "WarningRule":
        return cls.from_config(load_thresholds(path))
