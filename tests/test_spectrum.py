"""Tests for feather.util.spectrum — power spectrum utilities."""

import numpy as np
import pytest

from feather.util.spectrum import power_spectrum


class TestPowerSpectrum:
    """Test Welch power spectral density estimation."""

    def test_sine_wave_peak(self):
        """A pure sine wave should peak at its known period."""
        # 10 years of monthly data with 3-year period signal
        n = 120  # 10 years
        dt = 1 / 12  # monthly
        t = np.arange(n) * dt
        period_true = 3.0  # years
        signal = np.sin(2 * np.pi * t / period_true)

        periods, power = power_spectrum(signal, dt=dt)
        assert len(periods) > 0
        assert len(power) == len(periods)

        # Peak should be near the true period (within resolution)
        peak_idx = np.argmax(power)
        peak_period = periods[peak_idx]
        assert abs(peak_period - period_true) < 1.5, (
            f"Peak at {peak_period:.2f} yr, expected ~{period_true} yr"
        )

    def test_returns_positive_power(self):
        rng = np.random.default_rng(42)
        ts = rng.standard_normal(120)
        periods, power = power_spectrum(ts)
        assert np.all(power >= 0)

    def test_periods_are_positive(self):
        rng = np.random.default_rng(42)
        ts = rng.standard_normal(120)
        periods, power = power_spectrum(ts)
        assert np.all(periods > 0)

    def test_sorted_longest_first(self):
        rng = np.random.default_rng(42)
        ts = rng.standard_normal(120)
        periods, _ = power_spectrum(ts)
        assert np.all(np.diff(periods) <= 0), "Periods should be sorted longest first"

    def test_short_series(self):
        """Very short series should return empty or valid arrays."""
        ts = np.array([1.0, 2.0, 3.0])
        periods, power = power_spectrum(ts)
        # Should return empty for < 8 points
        assert len(periods) == 0
        assert len(power) == 0

    def test_handles_nan(self):
        """NaN values should be replaced with zero."""
        rng = np.random.default_rng(42)
        ts = rng.standard_normal(120)
        ts[10] = np.nan
        ts[50] = np.nan
        periods, power = power_spectrum(ts)
        assert len(periods) > 0
        assert np.all(np.isfinite(power))

    def test_custom_dt(self):
        """Test with different sampling interval."""
        rng = np.random.default_rng(42)
        ts = rng.standard_normal(60)
        # Daily data (dt = 1/365.25)
        periods, power = power_spectrum(ts, dt=1 / 365.25)
        assert len(periods) > 0

    def test_white_noise_flat_spectrum(self):
        """White noise should have a roughly flat spectrum."""
        rng = np.random.default_rng(42)
        ts = rng.standard_normal(1200)  # 100 years monthly
        periods, power = power_spectrum(ts)
        # Ratio of max to min should be moderate (not dominated by one peak)
        if len(power) > 2:
            ratio = power.max() / power[power > 0].min()
            assert ratio < 100, f"White noise spectrum too peaked: ratio={ratio}"

    def test_empty_input(self):
        periods, power = power_spectrum(np.array([]))
        assert len(periods) == 0
        assert len(power) == 0
