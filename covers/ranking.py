"""Chunked top-K cosine ranking.

For N test tracks we must rank every track against the other N-1. A full N x N
similarity matrix is too large to hold in memory (e.g. 55170 x 55170), so we:
  1. L2-normalize the embedding matrix.
  2. Process queries in batches, computing ``Q_batch @ E.T`` and taking the top-K
     candidates (excluding self) per query.

This mirrors the baseline but uses a compact row-wise accumulation that keeps peak
memory to ``batch_size * N`` floats.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def rank_top_k(
    embeddings: np.ndarray,
    k: int = 100,
    batch_size: Optional[int] = None,
    query_ids: Optional[Sequence[int]] = None,
    exclude_self: bool = True,
) -> Dict[int, List[int]]:
    """Return {query_id: [candidate_id, ...]} ranking by descending cosine similarity.

    ``embeddings`` is (N, D) where rows align with ``query_ids`` (defaults to 0..N-1).
    Only the *similarity* between queries and the candidate set is considered; the
    candidate set is the same as the query set.
    """
    n = embeddings.shape[0]
    emb_norm = normalize(embeddings)
    if query_ids is None:
        query_ids = list(range(n))
    if batch_size is None:
        batch_size = max(1, int(4_000_000_000 / max(1, n * 4)))  # ~4GB peak cap
    batch_size = max(1, min(batch_size, n))

    rankings: Dict[int, List[int]] = {}
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = emb_norm[start:end] @ emb_norm.T  # (B, N)
        # Exclude self from candidates.
        if exclude_self:
            row_idx = np.arange(start, end)
            sims[row_idx - start, row_idx] = -np.inf
        top = np.argpartition(-sims, kth=k, axis=1)[:, :k]
        # Sort each row's top-k by descending similarity.
        batch_sims = np.take_along_axis(sims, top, axis=1)
        sorted_idx = np.argsort(-batch_sims, axis=1)
        for local, q in enumerate(range(start, end)):
            cands = top[local, sorted_idx[local]].tolist()
            rankings[query_ids[q]] = cands
    return rankings
