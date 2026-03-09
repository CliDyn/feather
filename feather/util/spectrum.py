"""Power spectrum utilities for climate variability analysis.

Provides Welch spectral density estimation for teleconnection indices.
"""

import numpy as np
from scipy.signal import welch


def power_spectrum(ts, dt=1 / 12):
    """Welch power spectral density of a 1D time series.

    Parameters
    ----------
    ts : array-like
        1D time series (e.g., monthly teleconnection index).
        NaN values are replaced with zero for spectral estimation.
    dt : float
        Sampling interval in years (default ``1/12`` for monthly data).

    Returns
    -------
    periods : np.ndarray
        Periods in years (``1 / frequency``), sorted from longest to
        shortest. The zero-frequency (infinite period) bin is excluded.
    power : np.ndarray
        Power spectral density at each period.
    """
    ts = np.asarray(ts, dtype=np.float64)

    # Replace NaN with zero (safe for spectral estimation)
    ts = np.where(np.isfinite(ts), ts, 0.0)

    n = len(ts)
    if n < 8:
        return np.array([]), np.array([])

    # Welch with reasonable defaults for climate indices
    # nperseg ~ 1/4 of total length gives good frequency resolution
    nperseg = min(n, max(8, n // 4))

    freqs, psd = welch(ts, fs=1.0 / dt, nperseg=nperseg,
                       detrend="linear", scaling="density")

    # Exclude zero frequency
    nonzero = freqs > 0
    freqs = freqs[nonzero]
    psd = psd[nonzero]

    # Convert frequency (cycles/year) to period (years)
    periods = 1.0 / freqs

    # Sort by period (longest first)
    sort_idx = np.argsort(periods)[::-1]
    return periods[sort_idx], psd[sort_idx]
