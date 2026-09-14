"""Evaluation: nDCG@100 as defined by the task.

The submission is a ranking of candidate track ids per query track. A retrieved
candidate is *relevant* if it belongs to the same cover clique as the query
(knowledge of cliques comes from ``cliques2versions.tsv`` for val; the test ground
truth is hidden and provided separately for offline scoring).

Metric: ``nDCG@K`` with gain=1 for relevant hits and discount ``1/sqrt(position)``
(so position 1 -> 1.0, position 2 -> 0.707, ...). DCG is normalized by the ideal
DCG computed from the same relevance labels.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np


def dcg(relevance: Sequence[int], k: int, discount_base: float = 2.0) -> float:
    """Discounted cumulative gain at K with discount 1/sqrt(position)."""
    k = min(k, len(relevance))
    return sum(
        rel / math.sqrt(pos + 1.0) for pos, rel in enumerate(relevance[:k])
    )


def ndcg_at_k(ranked_ids: Sequence[int], relevant: Set[int], k: int = 100) -> float:
    """nDCG@k for one query given its ranked candidate ids and the relevant set.

    Only candidates that appear in the ranked list contribute to the ideal DCG
    denominator (this is the standard formulation when labels beyond K are unknown).
    """
    k = min(k, len(ranked_ids))
    relevance = [1 if rid in relevant else 0 for rid in ranked_ids[:k]]
    num_relevant = relevance.count(1)
    if num_relevant == 0:
        return 0.0
    ideal = dcg([1] * num_relevant, k)
    return dcg(relevance, k) / ideal


def mean_ndcg_at_100(
    rankings: Dict[int, Sequence[int]],
    track_to_clique: Dict[int, int],
) -> float:
    """Mean nDCG@100 across queries.

    ``rankings`` maps query track id -> ranked candidate ids (in relevance order).
    ``track_to_clique`` maps every track id (query + candidates) -> clique id.
    """
    total = 0.0
    count = 0
    for query_id, ranked in rankings.items():
        q_clique = track_to_clique.get(query_id)
        if q_clique is None:
            continue
        relevant = {
            cand
            for cand in ranked
            if track_to_clique.get(cand) == q_clique
        }
        total += ndcg_at_100(list(ranked), relevant)
        count += 1
    return total / count if count else 0.0


def ndcg_at_100(ranked_ids: Sequence[int], relevant: Set[int]) -> float:
    return ndcg_at_k(ranked_ids, relevant, k=100)


def parse_submission(path: str) -> Dict[int, List[int]]:
    """Parse a submission file.

    Format: one ``query_trackid [space] trackid1 [space] trackid2 ...`` line per query.
    """
    rankings: Dict[int, List[int]] = {}
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            query = int(parts[0])
            # Keep the ordering as written; the file may already be sorted or not.
            rankings[query] = [int(p) for p in parts[1:]]
    return rankings
