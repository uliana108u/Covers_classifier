"""Classical (non-learned) CQT descriptors used by the fast CPU baseline.

Each track is a small ``(84, 50)`` CQT spectrogram. Because covers of the same
recording share harmonic content but can differ in tempo/timing, we collapse the
time axis into summary statistics, yielding a tempo-robust low-dimensional feature
vector compared with cosine similarity.
"""
from __future__ import annotations

import numpy as np


def _log_magnitude(cqt: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.log10(np.abs(cqt) + eps)


def describe(
    cqt: np.ndarray,
    n_segments: int = 5,
    stats: tuple = ("mean", "std"),
    log: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """Return a fixed-length descriptor vector for one CQT spectrogram.

    The time axis (50 frames) is split into ``n_segments`` contiguous blocks; for each
    block we compute ``stats`` over the remaining time dimension per frequency bin,
    concatenated into a single flat vector.

    Example (default): 84 bins x 5 segments x 2 stats = 840-d vector.
    """
    x = _log_magnitude(cqt, eps) if log else np.abs(cqt)
    n_frames = x.shape[1]
    seg_bounds = np.linspace(0, n_frames, n_segments + 1, dtype=int)
    chunks = []
    for i in range(n_segments):
        seg = x[:, seg_bounds[i] : seg_bounds[i + 1]]
        for stat in stats:
            if stat == "mean":
                chunks.append(seg.mean(axis=1))
            elif stat == "std":
                chunks.append(seg.std(axis=1))
            elif stat == "max":
                chunks.append(seg.max(axis=1))
            else:
                raise ValueError(f"unknown stat: {stat}")
    return np.concatenate(chunks).astype(np.float32)


def describe_batch(
    cqts: np.ndarray,
    n_segments: int = 5,
    stats: tuple = ("mean", "std"),
    log: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """Vectorized descriptor for a batch of CQT spectrograms (B, 84, 50) -> (B, D)."""
    if log:
        x = np.log10(np.abs(cqts) + eps)
    else:
        x = np.abs(cqts)
    n_frames = x.shape[2]
    seg_bounds = np.linspace(0, n_frames, n_segments + 1, dtype=int)
    parts = []
    for i in range(n_segments):
        seg = x[:, :, seg_bounds[i] : seg_bounds[i + 1]]
        if "mean" in stats:
            parts.append(seg.mean(axis=2))
        if "std" in stats:
            parts.append(seg.std(axis=2))
        if "max" in stats:
            parts.append(seg.max(axis=2))
    return np.concatenate(parts, axis=1).astype(np.float32)
