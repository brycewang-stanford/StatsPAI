"""Gaussian kernel density on a regular grid by linear binning.

Shared by :func:`statspai.regression.interflex.interflex` (density weights
for the kernel estimator) and the RIF quantile influence function in
:mod:`statspai.decomposition._common` (density at the quantile).

Clean-room implementation, written from documented definitions only (the
specification is ``docs/dev/cleanroom/binned_kde.md``):

* the bandwidth is Silverman's rule of thumb (Silverman 1986, eq. 3.31,
  ``silverman1986density``), ``0.9 * min(sd, IQR / 1.34) * n^(-1/5)``, with
  the fall-backs for a zero spread that R's ``?bw.nrd0`` help page promises
  ("a positive result will be guaranteed"), pinned by black-box runs;
* the estimate is the binned kernel estimator (linear binning onto an
  equispaced grid, then a discrete convolution with the kernel evaluated
  at the grid offsets; Silverman 1982, ``silverman1982algorithm``; Wand
  1994, ``wand1994fast``), read off a grid that
  extends the data range, then linearly interpolated onto the output points.

The grid geometry (``cut`` and ``ext`` bandwidths beyond the data, at least
512 bins, rounded up to a power of two) follows the ``?density`` help page.
R's ``stats::density`` was used only as a black box: the fixture
``tests/reference_parity/_fixtures/binned_kde_blackbox_R.json`` (generator
next to it) agrees with this module to about 1e-14 relative.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["silverman_bandwidth", "binned_gaussian_kde", "binned_density_at"]

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))


def silverman_bandwidth(x: np.ndarray) -> float:
    """Silverman's rule-of-thumb bandwidth for a Gaussian kernel.

    ``0.9 * s * n^(-1/5)`` with ``s = min(sd, IQR / 1.34)``; ``sd`` uses the
    ``n - 1`` divisor and the IQR the linearly interpolated (type 7)
    quartiles. When ``s`` is zero (coinciding quartiles) the standard
    deviation is used; when that is zero too (constant data) the absolute
    value of the first observation; and when that is zero, 1. A single
    observation has no spread and goes straight to the last two steps.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n == 0:
        raise ValueError("silverman_bandwidth needs at least one observation")
    if n > 1:
        sd = float(np.std(x, ddof=1))
        q25, q75 = np.percentile(x, [25.0, 75.0])
        spread = min(sd, float(q75 - q25) / 1.34)
    else:
        sd = 0.0
        spread = 0.0
    if not spread > 0.0:
        spread = sd
    if not spread > 0.0:
        spread = abs(float(x[0]))
    if not spread > 0.0:
        spread = 1.0
    return 0.9 * spread * n ** (-0.2)


def _bin_count(n_points: int) -> int:
    """Number of bins: at least 512, a power of two above 512."""
    m = max(int(n_points), 512)
    if m > 512:
        m = 1 << int(np.ceil(np.log2(m)))
    return m


def _binned_density_on_grid(
    x: np.ndarray, mass: np.ndarray, bw: float, lo: float, hi: float, m: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Density at ``m`` equispaced points spanning ``[lo, hi]``.

    Each observation's mass is split between the two grid points around it
    in proportion to proximity (linear binning); observations outside
    ``[lo, hi]`` contribute nothing. The binned masses are convolved with the
    Gaussian kernel evaluated at every grid offset.
    """
    step = (hi - lo) / (m - 1)
    grid = lo + step * np.arange(m)
    pos = (x - lo) / step
    left = np.floor(pos).astype(np.int64)
    frac = pos - left
    counts = np.zeros(m + 2)
    # Shift by one so that index -1 (just below lo) lands in a dropped slot.
    for idx, share in ((left, 1.0 - frac), (left + 1, frac)):
        inside = (idx >= 0) & (idx < m)
        np.add.at(counts, idx[inside] + 1, (mass * share)[inside])
    binned = counts[1 : m + 1]
    offsets = step * np.arange(-(m - 1), m)
    kernel = np.exp(-0.5 * (offsets / bw) ** 2) / (bw * _SQRT_2PI)
    nfft = 1 << int(np.ceil(np.log2(3 * m - 2)))
    conv = np.fft.irfft(np.fft.rfft(binned, nfft) * np.fft.rfft(kernel, nfft), nfft)
    dens = conv[m - 1 : 2 * m - 1]
    return grid, dens


def binned_gaussian_kde(
    x: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    bw: Optional[float] = None,
    n: int = 512,
    cut: float = 3.0,
    ext: float = 4.0,
    from_: Optional[float] = None,
    to: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Gaussian kernel density estimate on ``n`` equispaced points.

    Parameters
    ----------
    x : array-like
        Observations.
    weights : array-like, optional
        Observation masses. ``None`` gives each observation ``1 / len(x)``.
        They are used as given (not rescaled) and do not enter the default
        bandwidth.
    bw : float, optional
        Kernel standard deviation; default :func:`silverman_bandwidth`.
    n : int
        Number of output points.
    cut : float
        Default output range is ``[min(x) - cut*bw, max(x) + cut*bw]``.
    ext : float
        The binning grid extends ``ext * bw`` beyond the output range.
    from_, to : float, optional
        Override the ends of the output range.

    Returns
    -------
    (points, density, bw)
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("binned_gaussian_kde needs at least one observation")
    mass = (
        np.full(x.size, 1.0 / x.size)
        if weights is None
        else np.asarray(weights, dtype=float).ravel()
    )
    if mass.shape != x.shape:
        raise ValueError("weights must have the same length as x")
    h = silverman_bandwidth(x) if bw is None else float(bw)
    if not h > 0.0:
        raise ValueError("bandwidth must be positive")
    start = float(x.min()) - cut * h if from_ is None else float(from_)
    stop = float(x.max()) + cut * h if to is None else float(to)
    lo, hi = start - ext * h, stop + ext * h
    grid, dens = _binned_density_on_grid(x, mass, h, lo, hi, _bin_count(n))
    points = np.linspace(start, stop, int(n))
    values = np.maximum(np.interp(points, grid, dens), 0.0)
    return points, values, h


def binned_density_at(
    x: np.ndarray,
    point: float,
    *,
    weights: Optional[np.ndarray] = None,
    bw: Optional[float] = None,
    n: int = 512,
    ext: float = 4.0,
) -> float:
    """Binned Gaussian density at one point.

    The output range collapses to ``[point, point]``, so the binning grid
    spans ``point +/- ext*bw``; observations farther away than that carry
    no mass into the estimate.
    """
    _, values, _ = binned_gaussian_kde(
        x, weights=weights, bw=bw, n=n, ext=ext, from_=point, to=point
    )
    return float(values[0])
