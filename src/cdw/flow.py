"""Flow balance: count people crossing a boundary, and project occupancy.

People are conserved. If more enter a region than leave it, occupancy rises at
a rate you can measure and extrapolate. That makes this the only part of the
system that genuinely looks forward - everything else reports the present.

What it forecasts is **occupancy**, not an incident:

    "at the current net inflow, this region reaches 500 people in ~27 s"

Every term in that sentence is measured. It makes no claim about what happens
at 500, which is the point: the projection is defensible precisely because it
stops at the quantity we can actually observe.

The assumption it rests on, stated plainly: **current rates persist.** That is
exactly what stops being true near breakdown, when outflow collapses. So the
forecast degrades where it would be most valuable. A rising net inflow is still
actionable; the arrival time attached to it is not a promise.

**Measured skill, and it is not good.** Evaluated on 2816 forecasts over 133 s
of CroHD footage against a persistence baseline (assume occupancy does not
change), extrapolation never won:

    30 s window, raw slope          -106% skill at a 30 s horizon
    30 s window, sign-gated          -42%
    60 s window, sign-gated + 0.5     -6%

In a steady-state crowd occupancy wanders rather than trends, so projecting a
short-term slope forward mostly amplifies noise. The defaults below are the
configuration that makes projection very nearly harmless; none makes it
helpful. Projection would only earn its place on footage containing sustained
accumulation, which no public dataset we could find provides - the same gap
that prevents locating a critical density.

Treat ``net_rate`` as a measurement, which the conservation checks support, and
``time_to`` as an indication rather than a countdown.

Counting crossings from track ids needs one piece of care. In a dense crowd
tracks are lost and reacquired constantly, and a track that simply *appears*
inside the region is not someone who walked in - it is usually the same person
with a new id. Only a track seen on one side and then the other is counted, so
appearances and disappearances contribute nothing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

__all__ = ["FlowBalance", "FlowSample"]


@dataclass(slots=True)
class FlowSample:
    t: float
    occupancy: int            # people currently inside the region
    entries: int              # crossings inward, within the window
    exits: int                # crossings outward, within the window
    inflow_rate: float        # people/s, from crossings
    outflow_rate: float       # people/s, from crossings
    crossing_net_rate: float  # inflow - outflow
    net_rate: float           # people/s from the occupancy trend: the one to use
    window_s: float
    tracked_fraction: float   # share of in-region people carrying a usable id

    def projected_occupancy(self, horizon_s: float) -> float:
        """Occupancy in ``horizon_s`` seconds if the current net rate holds."""
        return max(0.0, self.occupancy + self.net_rate * horizon_s)

    @property
    def crossing_discrepancy(self) -> float:
        """How far the crossing count disagrees with the occupancy trend.

        Should be near zero - people are conserved. A large value means track
        churn is corrupting the crossing counts, so the inflow/outflow split
        should be read with suspicion even though the net rate is still sound.
        """
        return self.crossing_net_rate - self.net_rate

    def time_to(self, target: int) -> float | None:
        """Seconds until occupancy reaches ``target``, or None.

        None when the region is already there, or is not filling fast enough to
        get there - a projection from a flat or falling rate is meaningless and
        should not be dressed up as a countdown.
        """
        if self.occupancy >= target:
            return 0.0
        if self.net_rate <= 1e-6:
            return None
        return (target - self.occupancy) / self.net_rate


class FlowBalance:
    """Directional crossing counts over a trailing window.

    Parameters
    ----------
    window_s:
        How far back crossings are counted. Short windows are responsive but
        noisy; long ones lag a genuine surge.
    forget_s:
        How long an unseen track is remembered. A track lost briefly and
        reacquired keeps its side, so a momentary detection failure does not
        register as a crossing.
    """

    def __init__(self, window_s: float = 60.0, forget_s: float = 1.5,
                 require_consistent_sign: bool = True, shrink: float = 0.5):
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self.window_s = window_s
        self.forget_s = forget_s
        # A slope can be statistically significant and still not persist, so
        # significance is the wrong gate - it made no measurable difference.
        # Requiring the same sign in both halves of the window tests
        # persistence directly, and shrinkage damps what survives, because
        # even a real trend over-predicts.
        self.require_consistent_sign = require_consistent_sign
        self.shrink = shrink
        self._side: dict[int, bool] = {}        # track id -> was inside
        self._last_seen: dict[int, float] = {}
        self._events: deque[tuple[float, int]] = deque()   # (t, +1 in / -1 out)
        self._occupancy: deque[tuple[float, float]] = deque()

    def update(self, t: float, inside_ids, visible_ids, occupancy: int | None = None
               ) -> FlowSample:
        """Feed one frame.

        ``inside_ids``  track ids currently inside the region.
        ``visible_ids`` every track id detected this frame, inside or out.
            Needed to tell "left the region" from "lost by the tracker": a
            track that is simply absent tells us nothing and is not counted.
        ``occupancy``   the count to report, if it is measured more reliably
            than tracking manages - detection without tracking usually is.
            Defaults to the number of tracked ids inside.
        """
        inside = set(inside_ids)
        visible = set(visible_ids) | inside

        for tid in visible:
            was = self._side.get(tid)
            now = tid in inside
            if was is None:
                # First sighting. Record the side, count nothing: an id that
                # appears mid-region is a reacquisition far more often than it
                # is an arrival.
                self._side[tid] = now
            elif was != now:
                self._events.append((t, 1 if now else -1))
                self._side[tid] = now
            self._last_seen[tid] = t

        for tid in [k for k, seen in self._last_seen.items() if t - seen > self.forget_s]:
            del self._last_seen[tid]
            self._side.pop(tid, None)

        cutoff = t - self.window_s
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

        entries = sum(1 for _, d in self._events if d > 0)
        exits = sum(1 for _, d in self._events if d < 0)
        span = min(self.window_s, max(t - self._events[0][0], 1e-6)) if self._events \
            else self.window_s

        measured = len(inside) if occupancy is None else occupancy

        # The net rate comes from the occupancy TREND, not from crossings.
        # Validation on ground-truth tracks showed crossing counts drift about
        # -12 per 100 even with perfect identities: a person occluded inside
        # the region is forgotten, returns as a first sighting (no entry), and
        # is later counted leaving. Every occlusion biases the count outward.
        # Occupancy itself has no such failure mode, so the projection is built
        # on its slope and the crossings are kept as a diagnostic.
        self._occupancy.append((t, float(measured)))
        while self._occupancy and self._occupancy[0][0] < cutoff:
            self._occupancy.popleft()
        occ_rate = _slope(self._occupancy)
        if self.require_consistent_sign and len(self._occupancy) >= 8:
            mid = self._occupancy[0][0] + (t - self._occupancy[0][0]) / 2
            first = [q for q in self._occupancy if q[0] < mid]
            second = [q for q in self._occupancy if q[0] >= mid]
            if first and second and _slope(first) * _slope(second) <= 0:
                occ_rate = 0.0
        occ_rate *= self.shrink

        return FlowSample(
            t=t,
            occupancy=measured,
            entries=entries,
            exits=exits,
            inflow_rate=entries / span,
            outflow_rate=exits / span,
            crossing_net_rate=(entries - exits) / span,
            net_rate=occ_rate,
            window_s=self.window_s,
            tracked_fraction=(len(inside) / measured) if measured else 0.0,
        )

    def reset(self) -> None:
        self._side.clear()
        self._last_seen.clear()
        self._events.clear()
        self._occupancy.clear()


def _slope(series) -> float:
    """Least-squares slope of (t, value) pairs, in value per second."""
    if len(series) < 3:
        return 0.0
    ts = [a for a, _ in series]
    vs = [b for _, b in series]
    span = ts[-1] - ts[0]
    if span < 1e-6:
        return 0.0
    n = len(ts)
    mt = sum(ts) / n
    mv = sum(vs) / n
    denom = sum((a - mt) ** 2 for a in ts)
    if denom < 1e-12:
        return 0.0
    return sum((a - mt) * (b - mv) for a, b in zip(ts, vs)) / denom
