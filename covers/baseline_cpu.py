"""Fast CPU baseline for cover detection.

Runs without a GPU: extracts compact descriptors from the CQT spectrograms and
ranks test tracks by cosine similarity, writing a ``submission.txt`` file in the
contest format.

Usage:
    python -m covers.baseline_cpu --dataset_path <path> --out submission.txt
    python -m covers.baseline_cpu --mode val   # score nDCG@100 on validation
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from covers import data as data_mod
from covers.eval import mean_ndcg_at_100
from covers.features import describe_batch
from covers.ranking import rank_top_k


def build_embeddings(
    track_ids: Sequence[int],
    dataset_path: str,
    n_segments: int,
    stats: tuple,
    batch_io: int = 1024,
) -> np.ndarray:
    """Compute descriptors for many tracks, batching file I/O."""
    from tqdm import tqdm

    out = []
    for start in tqdm(range(0, len(track_ids), batch_io), desc="embedding", ncols=80):
        end = min(start + batch_io, len(track_ids))
        cqts = np.stack([data_mod.load_cqt(int(tid), dataset_path) for tid in track_ids[start:end]])
        out.append(describe_batch(cqts, n_segments=n_segments, stats=stats))
    return np.vstack(out)


def write_submission(rankings: Dict[int, List[int]], out_path: str, k: int = 100) -> None:
    with open(out_path, "w") as fh:
        for query_id in sorted(rankings):
            cands = rankings[query_id][:k]
            fh.write(f"{query_id} " + " ".join(map(str, cands)) + "\n")


def run_test(dataset_path: str, out_path: str, n_segments: int, stats: tuple,
             k: int, rank_batch: Optional[int]) -> None:
    cqt_path = os.path.join(dataset_path, "test")
    test_ids = [int(t) for t in data_mod.load_split_ids("test", dataset_path)]
    emb = build_embeddings(test_ids, cqt_path, n_segments, stats)
    ranking = rank_top_k(emb, k=k, batch_size=rank_batch, query_ids=list(test_ids))
    ranking = {q: [test_ids[c] for c in cands] for q, cands in ranking.items()}
    write_submission(ranking, out_path, k)
    print(f"wrote {len(ranking)} queries to {out_path}")


def build_val_track_to_clique(dataset_path: str):
    """Return (val_track_ids, track_id -> dense clique id) for the val subset."""
    _, val_map, _, val_track_ids, _ = data_mod.train_val_clique_maps(dataset_path)
    track_to_clique: Dict[int, int] = {}
    for clique, row in val_map.iterrows():
        for tid in row["versions"]:
            if int(tid) in set(val_track_ids):
                track_to_clique[int(tid)] = int(clique)
    return val_track_ids, track_to_clique


def run_val(dataset_path: str, n_segments: int, stats: tuple,
            k: int, rank_batch: Optional[int]) -> float:
    cqt_path = os.path.join(dataset_path, "train")
    val_track_ids, track_to_clique = build_val_track_to_clique(dataset_path)
    emb = build_embeddings(val_track_ids, cqt_path, n_segments, stats)
    ranking = rank_top_k(emb, k=k, batch_size=rank_batch, query_ids=list(val_track_ids))
    ranking = {int(q): [int(val_track_ids[c]) for c in cands] for q, cands in ranking.items()}
    score = mean_ndcg_at_100(ranking, track_to_clique)
    print(f"val nDCG@{k}: {score:.4f}")
    return score


def _parse_stats(s: str) -> tuple:
    return tuple(st for st in s.split(",") if st)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast CPU cover-detection baseline")
    parser.add_argument("--dataset_path", default=None,
                        help="dir containing splits/, cliques2versions.tsv and train/ test/")
    parser.add_argument("--out", default="submission.txt")
    parser.add_argument("--mode", choices=["test", "val"], default="test")
    parser.add_argument("--n_segments", type=int, default=5)
    parser.add_argument("--stats", default="mean,std")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--rank_batch", type=int, default=None)
    args = parser.parse_args()

    dataset_path = data_mod.resolve_data_path(args.dataset_path)
    stats = _parse_stats(args.stats)
    if args.mode == "test":
        run_test(dataset_path, args.out, args.n_segments, stats, args.k, args.rank_batch)
    else:
        run_val(dataset_path, args.n_segments, stats, args.k, args.rank_batch)


if __name__ == "__main__":
    main()
