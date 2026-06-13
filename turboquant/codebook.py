"""Scalar Lloyd-Max codebook generation for TurboQuant."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln


@dataclass(frozen=True)
class ScalarCodebook:
    dimension: int
    bits: int
    centroids: np.ndarray
    boundaries: np.ndarray
    cost: float


def beta_coordinate_pdf(xs: np.ndarray, dimension: int) -> np.ndarray:
    """Density of one coordinate of a random point on S^{d-1}."""
    if dimension < 2:
        raise ValueError("dimension must be >= 2")
    xs = np.asarray(xs, dtype=np.float64)
    log_coeff = gammaln(dimension / 2.0) - 0.5 * np.log(np.pi) - gammaln((dimension - 1.0) / 2.0)
    one_minus = np.clip(1.0 - xs * xs, 0.0, None)
    exponent = (dimension - 3.0) / 2.0
    pdf = np.exp(log_coeff) * np.power(one_minus, exponent)
    pdf[(xs < -1.0) | (xs > 1.0)] = 0.0
    return pdf


def _normal_quantile_init(levels: int, dimension: int) -> np.ndarray:
    # Deterministic high-dimensional initialization. This keeps Lloyd iterations
    # stable for the narrow coordinate distribution at d=1536/3072.
    from scipy.stats import norm

    ps = (np.arange(levels, dtype=np.float64) + 0.5) / levels
    centroids = norm.ppf(ps) / np.sqrt(dimension)
    return np.clip(centroids, -1.0 + 1e-12, 1.0 - 1e-12)


def lloyd_max_codebook(
    dimension: int,
    bits: int,
    *,
    grid_size: int = 200_001,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> ScalarCodebook:
    """Compute a symmetric scalar Lloyd-Max codebook on [-1, 1].

    The paper solves a continuous 1D k-means problem for the beta coordinate
    density. For reproducibility and robustness, this implementation performs
    deterministic numerical quadrature on a dense grid.
    """
    if bits < 0:
        raise ValueError("bits must be non-negative")
    if bits == 0:
        centroids = np.array([0.0], dtype=np.float64)
        boundaries = np.array([-1.0, 1.0], dtype=np.float64)
        xs = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
        weights = beta_coordinate_pdf(xs, dimension)
        weights /= np.trapz(weights, xs)
        cost = float(np.trapz((xs**2) * weights, xs))
        return ScalarCodebook(dimension, bits, centroids, boundaries, cost)

    levels = 1 << bits
    xs = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    weights = beta_coordinate_pdf(xs, dimension)
    weights_sum = np.trapz(weights, xs)
    if not np.isfinite(weights_sum) or weights_sum <= 0:
        raise RuntimeError("invalid beta-coordinate density normalization")
    weights /= weights_sum

    centroids = _normal_quantile_init(levels, dimension)
    for _ in range(max_iter):
        boundaries = np.empty(levels + 1, dtype=np.float64)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])

        new_centroids = centroids.copy()
        for idx in range(levels):
            left = boundaries[idx]
            right = boundaries[idx + 1]
            if idx == levels - 1:
                mask = (xs >= left) & (xs <= right)
            else:
                mask = (xs >= left) & (xs < right)
            if not np.any(mask):
                continue
            mass = np.trapz(weights[mask], xs[mask])
            if mass > 0:
                new_centroids[idx] = np.trapz(xs[mask] * weights[mask], xs[mask]) / mass

        # Enforce symmetry; the target density is exactly symmetric.
        new_centroids = 0.5 * (new_centroids - new_centroids[::-1])[::-1] * -1.0
        new_centroids = np.sort(new_centroids)

        delta = float(np.max(np.abs(new_centroids - centroids)))
        centroids = new_centroids
        if delta < tol:
            break

    boundaries = np.empty(levels + 1, dtype=np.float64)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])

    nearest = np.searchsorted(boundaries[1:-1], xs, side="right")
    sqerr = (xs - centroids[nearest]) ** 2
    cost = float(np.trapz(sqerr * weights, xs))
    return ScalarCodebook(dimension, bits, centroids.astype(np.float64), boundaries, cost)


def gaussian_approx_codebook(bits: int, dimension: int) -> np.ndarray:
    """Fast deterministic fallback centroids for N(0, 1/d)."""
    return _normal_quantile_init(1 << bits, dimension).astype(np.float64)
