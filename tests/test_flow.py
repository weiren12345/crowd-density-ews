"""Flow-balance tests.

The failure mode that matters is miscounting crossings in a dense crowd, where
tracks churn constantly. A reacquired id must not read as an arrival, or the
net rate inflates and the projection becomes fiction.
"""

import pytest

from cdw.flow import FlowBalance, FlowSample


def feed(fb, steps, fps=10.0):
    """steps: list of (inside_ids, visible_ids). Returns the final sample."""
    sample = None
    for i, (inside, visible) in enumerate(steps):
        sample = fb.update(i / fps, inside, visible)
    return sample


# -- crossing counting ------------------------------------------------------


def test_first_sighting_is_not_a_crossing():
    """An id appearing inside the region is a reacquisition, not an arrival."""
    fb = FlowBalance(window_s=10.0)
    s = feed(fb, [({1, 2, 3}, {1, 2, 3})] * 5)
    assert s.entries == 0
    assert s.exits == 0
    assert s.net_rate == 0.0


def test_a_genuine_entry_is_counted():
    fb = FlowBalance(window_s=10.0)
    s = feed(fb, [(set(), {1})] * 3 + [({1}, {1})] * 3)
    assert s.entries == 1
    assert s.exits == 0


def test_a_genuine_exit_is_counted():
    fb = FlowBalance(window_s=10.0)
    s = feed(fb, [({1}, {1})] * 3 + [(set(), {1})] * 3)
    assert s.exits == 1
    assert s.entries == 0


def test_a_lost_track_is_not_an_exit():
    """Disappearing tells us nothing about which side the person is on."""
    fb = FlowBalance(window_s=10.0)
    s = feed(fb, [({1}, {1})] * 3 + [(set(), set())] * 3)
    assert s.exits == 0


def test_a_brief_dropout_does_not_fabricate_crossings():
    """Lost for one frame, back inside the next: nobody moved."""
    fb = FlowBalance(window_s=10.0, forget_s=1.5)
    steps = [({1}, {1})] * 3 + [(set(), set())] + [({1}, {1})] * 3
    s = feed(fb, steps)
    assert s.entries == 0 and s.exits == 0


def test_a_long_dropout_forgets_the_track():
    """Past forget_s the id is released, so its return is a first sighting."""
    fb = FlowBalance(window_s=100.0, forget_s=0.5)
    fb.update(0.0, {1}, {1})
    fb.update(5.0, set(), set())          # long gone
    s = fb.update(5.1, {1}, {1})
    assert s.entries == 0                 # treated as new, not as an arrival


def test_repeated_crossings_all_count():
    fb = FlowBalance(window_s=100.0)
    steps = []
    for _ in range(3):
        steps += [(set(), {1})] * 2 + [({1}, {1})] * 2
    s = feed(fb, steps)
    assert s.entries == 3
    assert s.exits == 2


# -- rates ------------------------------------------------------------------


def test_net_rate_is_positive_when_filling():
    fb = FlowBalance(window_s=10.0)
    steps = [(set(), set(range(10)))]
    for i in range(10):                    # one person enters per frame
        steps.append((set(range(i + 1)), set(range(10))))
    s = feed(fb, steps, fps=10.0)
    assert s.entries == 10
    assert s.net_rate > 0


def test_net_rate_is_negative_when_emptying():
    fb = FlowBalance(window_s=10.0)
    steps = [(set(range(10)), set(range(10)))]
    for i in range(10):
        steps.append((set(range(10 - i - 1)), set(range(10))))
    s = feed(fb, steps, fps=10.0)
    assert s.exits == 10
    assert s.net_rate < 0


def test_balanced_flow_gives_zero_net_rate():
    """Equal in and out: busy, but not filling."""
    fb = FlowBalance(window_s=100.0)
    steps = []
    for i in range(6):
        a, b = 100 + i, 200 + i
        steps += [({a}, {a, b}), ({a, b}, {a, b}), ({b}, {a, b})]
    s = feed(fb, steps)
    assert s.entries == s.exits
    assert s.net_rate == pytest.approx(0.0)


def test_window_drops_old_crossings():
    fb = FlowBalance(window_s=1.0)
    fb.update(0.0, set(), {1})
    fb.update(0.1, {1}, {1})               # an entry at t=0.1
    late = fb.update(9.0, {1}, {1})        # far outside the window
    assert late.entries == 0


# -- projection -------------------------------------------------------------


def test_projection_extrapolates_the_net_rate():
    s = FlowSample(t=0, occupancy=100, entries=60, exits=0, inflow_rate=6.0,
                   outflow_rate=0.0, crossing_net_rate=6.0, net_rate=6.0, window_s=10.0,
                   tracked_fraction=1.0)
    assert s.projected_occupancy(10.0) == pytest.approx(160.0)
    assert s.time_to(400) == pytest.approx(50.0)


def test_projection_refuses_to_extrapolate_a_flat_rate():
    """A countdown from a rate of zero is fiction; say nothing instead."""
    s = FlowSample(t=0, occupancy=100, entries=5, exits=5, inflow_rate=0.5,
                   outflow_rate=0.5, crossing_net_rate=0.0, net_rate=0.0, window_s=10.0,
                   tracked_fraction=1.0)
    assert s.time_to(400) is None


def test_projection_refuses_when_emptying():
    s = FlowSample(t=0, occupancy=100, entries=0, exits=30, inflow_rate=0.0,
                   outflow_rate=3.0, crossing_net_rate=-3.0, net_rate=-3.0, window_s=10.0,
                   tracked_fraction=1.0)
    assert s.time_to(400) is None


def test_already_at_target_is_zero_not_none():
    s = FlowSample(t=0, occupancy=500, entries=0, exits=0, inflow_rate=0.0,
                   outflow_rate=0.0, crossing_net_rate=0.0, net_rate=0.0, window_s=10.0,
                   tracked_fraction=1.0)
    assert s.time_to(400) == 0.0


def test_projected_occupancy_cannot_go_negative():
    s = FlowSample(t=0, occupancy=10, entries=0, exits=100, inflow_rate=0.0,
                   outflow_rate=10.0, crossing_net_rate=-10.0, net_rate=-10.0, window_s=10.0,
                   tracked_fraction=1.0)
    assert s.projected_occupancy(60.0) == 0.0


# -- reporting --------------------------------------------------------------


def test_occupancy_can_come_from_a_better_counter():
    """Detection counts more reliably than tracking, so allow it to be passed."""
    fb = FlowBalance(window_s=10.0)
    s = fb.update(0.0, {1, 2}, {1, 2}, occupancy=180)
    assert s.occupancy == 180
    assert s.tracked_fraction == pytest.approx(2 / 180)


def test_tracked_fraction_is_safe_when_empty():
    fb = FlowBalance(window_s=10.0)
    assert fb.update(0.0, set(), set()).tracked_fraction == 0.0


def test_reset_clears_state():
    fb = FlowBalance(window_s=100.0)
    feed(fb, [(set(), {1})] * 2 + [({1}, {1})] * 2)
    fb.reset()
    s = fb.update(99.0, {1}, {1})
    assert s.entries == 0 and s.exits == 0


def test_non_positive_window_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        FlowBalance(window_s=0.0)


# -- occupancy trend vs crossings -------------------------------------------


def test_net_rate_comes_from_occupancy_not_crossings():
    """The projection must survive crossings being miscounted.

    Validation on ground-truth tracks showed crossing counts drift about -12
    per 100 even with perfect identities, because a person occluded inside the
    region is forgotten and later counted leaving. Occupancy has no such
    failure mode, so the net rate is taken from its trend.
    """
    fb = FlowBalance(window_s=10.0)
    # Occupancy climbs steadily while NO crossing is ever observable: every id
    # is new each frame, exactly what heavy track churn looks like.
    sample = None
    for i in range(20):
        ids = set(range(1000 + i * 50, 1000 + i * 50 + 10 + i))
        sample = fb.update(i / 10.0, ids, ids, occupancy=10 + i)
    assert sample.entries == 0            # churn hid every crossing
    assert sample.crossing_net_rate == 0.0
    # Trend is unmistakable. Halved by the default shrinkage, which exists
    # because even a real trend over-predicts (see flow.py on measured skill).
    assert sample.net_rate > 3.0
    assert sample.time_to(100) is not None


def test_crossing_discrepancy_flags_churn():
    fb = FlowBalance(window_s=10.0)
    s = None
    for i in range(20):
        ids = set(range(1000 + i * 50, 1000 + i * 50 + 10 + i))
        s = fb.update(i / 10.0, ids, ids, occupancy=10 + i)
    # crossings say nothing is happening, occupancy says it is filling
    assert abs(s.crossing_discrepancy) > 1.0


def test_discrepancy_is_small_when_tracking_is_clean():
    fb = FlowBalance(window_s=10.0)
    s = None
    for i in range(20):
        ids = set(range(1, 11))           # stable identities, steady occupancy
        s = fb.update(i / 10.0, ids, ids)
    assert abs(s.crossing_discrepancy) < 0.5


# -- projection safeguards --------------------------------------------------


def test_shrinkage_damps_the_projected_rate():
    """Measured on CroHD: even a real trend over-predicts, so damp it."""
    def rate_with(shrink):
        fb = FlowBalance(window_s=10.0, shrink=shrink, require_consistent_sign=False)
        s = None
        for i in range(20):
            s = fb.update(i / 10.0, set(), set(), occupancy=10 + i)
        return s.net_rate

    assert rate_with(0.5) == pytest.approx(rate_with(1.0) * 0.5)


def test_an_inconsistent_trend_is_suppressed():
    """Occupancy that rises then falls is not a trend worth extrapolating."""
    fb = FlowBalance(window_s=10.0, shrink=1.0, require_consistent_sign=True)
    s = None
    occ = list(range(10, 30)) + list(range(30, 10, -1))
    for i, o in enumerate(occ):
        s = fb.update(i / 10.0, set(), set(), occupancy=o)
    assert s.net_rate == 0.0


def test_a_consistent_trend_survives_the_gate():
    fb = FlowBalance(window_s=10.0, shrink=1.0, require_consistent_sign=True)
    s = None
    for i in range(30):
        s = fb.update(i / 10.0, set(), set(), occupancy=10 + i)
    assert s.net_rate > 5.0


def test_gate_can_be_disabled():
    fb = FlowBalance(window_s=10.0, shrink=1.0, require_consistent_sign=False)
    s = None
    occ = list(range(10, 30)) + list(range(30, 10, -1))
    for i, o in enumerate(occ):
        s = fb.update(i / 10.0, set(), set(), occupancy=o)
    assert s.net_rate != 0.0
