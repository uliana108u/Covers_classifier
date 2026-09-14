"""Evaluate a submission file with nDCG@100.

The official test ground truth is hidden; for validation you can score a submission
built from the val subset using ``cliques2versions.tsv``. This script can also score
an arbitrary ranking file against a provided ground-truth clique map.

Usage:
    python -m covers.evaluate --submission submission.txt [--mode val] [--dataset_path ...]
"""
from __future__ import annotations

import argparse

from covers import data as data_mod
from covers.baseline_cpu import build_val_track_to_clique
from covers.eval import mean_ndcg_at_100, parse_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a submission with nDCG@100")
    parser.add_argument("--submission", required=True, help="path to submission file")
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--ground_truth", default=None,
                        help="optional col1=track_id col2=clique_id TSV override")
    args = parser.parse_args()

    dataset_path = data_mod.resolve_data_path(args.dataset_path)
    rankings = parse_submission(args.submission)

    if args.ground_truth:
        track_to_clique = {}
        with open(args.ground_truth) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2:
                    track_to_clique[int(parts[0])] = int(parts[1])
    else:
        _, track_to_clique = build_val_track_to_clique(dataset_path)

    score = mean_ndcg_at_100(rankings, track_to_clique)
    print(f"nDCG@100: {score:.4f}")


if __name__ == "__main__":
    main()
