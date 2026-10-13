"""Warning-rule tests.

Two failure modes matter and pull against each other: a rule that flickers is
ignored by operators, and a rule that latches forever is ignored too. The dwell
and clear times are what balance them, so both are pinned here.
"""

import pytest

from cdw.warning import LOSBand, WarningLevel, WarningRule, load_thresholds

BANDS = [
    LOSBand("A", 0.0, 0.83),
    LOSBand("B", 0.83, 1.08),
    LOSBand("C", 1.08, 1.54),
    LOSBand("D", 1.54, 3.59),
    LOSBand("E", 3.59, 5.38),
    LOSBand("F", 5.38),
]


def make_rule(**kw):
    defaults = dict(
        bands=BANDS,
        escalate_from="D",
        pressure_slope_threshold=0.02,
        trend_window_s=5.0,
        min_dwell_s=2.0,
        clear_s=5.0,
    )
    defaults.update(kw)
    return WarningRule(**defaults)


def feed(rule, series, fps=15.0):
    """Feed (density, pressure) pairs at a fixed rate; return all samples."""
    return [rule.update(i / fps, d, p) for i, (d, p) in enumerate(series)]


# -- band classification ----------------------------------------------------


@pytest.mark.parametrize(
    "density,expected",
    [(0.0, "A"), (0.5, "A"), (0.9, "B"), (1.2, "C"), (2.0, "D"), (4.0, "E"), (9.0, "F")],
)
def test_band_classification(density, expected):
    assert make_rule().band_for(density).name == expected


def test_band_edges_are_lower_inclusive():
    rule = make_rule()
    assert rule.band_for(1.54).name == "D"
    assert rule.band_for(1.539).name == "C"


def test_unknown_escalate_band_is_rejected():
    with pytest.raises(ValueError, match="escalate_from"):
        make_rule(escalate_from="Z")


def test_empty_bands_are_rejected():
    with pytest.raises(ValueError, match="at least one"):
        WarningRule(bands=[])


# -- the two-signal requirement ---------------------------------------------


def test_high_density_alone_never_warns():
    """A packed but calm crowd is WATCH, not WARNING."""
    rule = make_rule()
    samples = feed(rule, [(4.5, 0.5)] * 150)   # 10 s, dense, pressure flat
    assert {s.level for s in samples} <= {WarningLevel.WATCH}
    assert samples[-1].los == "E"


def test_rising_pressure_alone_never_warns():
    """A pressure spike in a sparse crowd is noise."""
    rule = make_rule()
    samples = feed(rule, [(0.4, i * 0.05) for i in range(150)])
    assert all(s.level is WarningLevel.NORMAL for s in samples)
    assert samples[-1].pressure_signal is True
    assert samples[-1].density_signal is False


def test_both_signals_sustained_raise_a_warning():
    rule = make_rule()
    samples = feed(rule, [(2.5, i * 0.02) for i in range(150)])
    assert samples[-1].level is WarningLevel.WARNING


def test_warning_waits_for_the_dwell_time():
    """Nothing should fire inside min_dwell_s of the conditions appearing."""
    rule = make_rule(min_dwell_s=2.0)
    samples = feed(rule, [(2.5, i * 0.02) for i in range(150)], fps=15.0)
    first = next(i for i, s in enumerate(samples) if s.level is WarningLevel.WARNING)
    # Slope needs 3 samples before it is defined, so the earliest possible
    # warning is at (2 s dwell + a couple of frames).
    assert first / 15.0 >= 2.0


def test_a_momentary_coincidence_does_not_warn():
    rule = make_rule(min_dwell_s=2.0)
    series = [(2.5, 0.0)] * 30 + [(2.5, 1.0)] * 3 + [(2.5, 1.0)] * 30
    samples = feed(rule, series)
    # A single jump gives one steep slope reading, not 2 s of rising pressure.
    assert all(s.level is not WarningLevel.WARNING for s in samples[:33])


def test_falling_pressure_does_not_warn():
    rule = make_rule()
    samples = feed(rule, [(4.0, 5.0 - i * 0.02) for i in range(150)])
    assert all(s.level is not WarningLevel.WARNING for s in samples)


# -- hysteresis -------------------------------------------------------------


def test_warning_holds_through_a_brief_dip():
    """The rule must not flicker when pressure momentarily plateaus."""
    rule = make_rule(clear_s=5.0)
    rising = [(2.5, i * 0.02) for i in range(90)]        # 6 s, warns
    plateau = [(2.5, 90 * 0.02)] * 15                    # 1 s flat
    samples = feed(rule, rising + plateau)
    assert samples[len(rising) - 1].level is WarningLevel.WARNING
    assert samples[-1].level is WarningLevel.WARNING


def test_warning_stands_down_after_clear_time():
    rule = make_rule(clear_s=2.0)
    rising = [(2.5, i * 0.02) for i in range(90)]
    calm = [(0.3, 1.8)] * 90                             # 6 s sparse and flat
    samples = feed(rule, rising + calm)
    assert samples[len(rising) - 1].level is WarningLevel.WARNING
    assert samples[-1].level is WarningLevel.NORMAL


def test_stand_down_goes_to_watch_while_still_dense():
    rule = make_rule(clear_s=2.0)
    rising = [(4.0, i * 0.02) for i in range(90)]
    # The 5 s trend window still holds the rise, so the slope needs that long
    # to fall below threshold before the 2 s clear timer can even start.
    dense_calm = [(4.0, 1.8)] * 180                      # 12 s still packed, now steady
    samples = feed(rule, rising + dense_calm)
    assert samples[-1].level is WarningLevel.WATCH


# -- reporting --------------------------------------------------------------


def test_sample_reports_both_signals_and_a_reason():
    rule = make_rule()
    samples = feed(rule, [(2.5, i * 0.02) for i in range(150)])
    final = samples[-1]
    assert final.density_signal and final.pressure_signal
    assert "pressure rising" in final.reason
    assert final.held_for_s > 2.0


def test_slope_is_zero_before_enough_history():
    rule = make_rule()
    assert rule.update(0.0, 2.0, 1.0).pressure_slope == 0.0


def test_slope_recovers_a_known_gradient():
    rule = make_rule(trend_window_s=100.0)
    for i in range(50):
        s = rule.update(i * 0.1, 2.0, i * 0.1 * 0.37)    # 0.37 per second
    assert s.pressure_slope == pytest.approx(0.37, rel=1e-6)


def test_reset_returns_to_normal():
    rule = make_rule()
    feed(rule, [(2.5, i * 0.02) for i in range(150)])
    rule.reset()
    assert rule.update(0.0, 0.2, 0.0).level is WarningLevel.NORMAL


# -- config -----------------------------------------------------------------


def test_shipped_thresholds_load_and_are_ordered():
    cfg = load_thresholds("configs/thresholds.yaml")
    rule = WarningRule.from_config(cfg)
    mins = [b.min_density for b in rule.bands]
    assert mins == sorted(mins)
    assert [b.name for b in rule.bands] == ["A", "B", "C", "D", "E", "F"]


def test_shipped_bands_are_contiguous():
    """No density should fall into a gap between bands."""
    rule = WarningRule.from_config(load_thresholds("configs/thresholds.yaml"))
    for lower, upper in zip(rule.bands, rule.bands[1:]):
        assert lower.max_density == pytest.approx(upper.min_density)
