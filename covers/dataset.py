"""Torch datasets/dataloaders for cover-detection training.

Ported from ``baseline/baseline/models/data_loader.py``. Train mode samples triplets
(anchor, positive-in-same-clique, negative-from-other-clique); val/test use the track
id + CQT only.
"""
from __future__ import annotations

import os
from typing import Callable, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from covers import data as data_mod


class CoverDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        dataset_path: str,
        data_split: Literal["train", "val", "test"],
        file_ext: str = "npy",
        clique_map: pd.DataFrame | None = None,
        track_ids: list[int] | None = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.dataset_path = dataset_path
        self.file_ext = file_ext
        self.data_split = data_split
        self.clique_map = clique_map  # maps dense clique id -> row with 'versions'
        self.track_ids = track_ids
        if track_ids is None:
            self.track_ids = self._load_track_ids()

        if data_split == "train":
            self._build_triplet_state()

    # ----- setup ---------------------------------------------------------
    def _load_track_ids(self) -> list[int]:
        if self.data_split == "test":
            return [int(t) for t in data_mod.load_split_ids("test", self.data_path)]
        # train/val: derive from clique versions so we only touch existing files
        ids: list[int] = []
        for _, row in self.clique_map.iterrows():
            for tid in row["versions"]:
                ids.append(int(tid))
        return ids

    def _build_triplet_state(self) -> None:
        """Index tracks by clique for fast positive/negative sampling."""
        self.clique_of_track: dict[int, int] = {}
        self.tracks_in_clique: dict[int, list[int]] = {}
        for clique, row in self.clique_map.iterrows():
            members = [int(t) for t in row["versions"]]
            self.tracks_in_clique[int(clique)] = members
            for tid in members:
                self.clique_of_track[tid] = int(clique)
        self.track_index = {tid: i for i, tid in enumerate(self.track_ids)}
        self._rnd = np.random.default_rng(0)
        self._perm = self._rnd.permutation(len(self.track_ids))
        self._perm_pos = 0

    # ----- data access ---------------------------------------------------
    def __len__(self) -> int:
        return len(self.track_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        track_id = self.track_ids[index]
        anchor = self._load_cqt(track_id)
        if self.data_split != "train":
            return {
                "track_id": torch.tensor(track_id, dtype=torch.int64),
                "cqt": anchor,
                "clique": torch.tensor(-1),
            }
        clique = self.clique_of_track[track_id]
        pos_id, neg_id = self._sample_triplet(track_id, clique)
        return {
            "track_id": torch.tensor(track_id, dtype=torch.int64),
            "cqt": anchor,
            "clique": torch.tensor(clique, dtype=torch.int64),
            "positive_id": torch.tensor(pos_id, dtype=torch.int64),
            "positive_cqt": self._load_cqt(pos_id),
            "negative_id": torch.tensor(neg_id, dtype=torch.int64),
            "negative_cqt": self._load_cqt(neg_id),
        }

    def _sample_triplet(self, track_id: int, clique: int) -> tuple[int, int]:
        members = self.tracks_in_clique[clique]
        pos_list = [m for m in members if m != track_id]
        pos_id = int(self._rnd.choice(pos_list))
        return pos_id, self._sample_negative(clique)

    def _sample_negative(self, clique: int) -> int:
        while True:
            if self._perm_pos >= len(self._perm):
                self._perm = self._rnd.permutation(len(self.track_ids))
                self._perm_pos = 0
            neg_track = self.track_ids[self._perm[self._perm_pos]]
            self._perm_pos += 1
            if self.clique_of_track[neg_track] != clique:
                return int(neg_track)

    def _load_cqt(self, track_id: int) -> torch.Tensor:
        filename = os.path.join(self.dataset_path, data_mod.make_file_path(int(track_id), self.file_ext))
        return torch.from_numpy(np.load(filename))


def _collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Stack per-sample tensors; handle empty tensors for non-train splits."""
    out: dict[str, torch.Tensor] = {}
    for key in batch[0].keys():
        out[key] = torch.stack([b[key] for b in batch])
    return out


def make_dataloaders(
    data_path: str,
    dataset_path: str,
    clique_map: pd.DataFrame | None,
    train_ids: list[int] | None,
    val_ids: list[int] | None,
    test_ids: list[int] | None,
    batch_size: int = 64,
    num_workers: int = 0,
    train_shuffle: bool = True,
    drop_last: bool = True,
) -> dict[str, DataLoader]:
    """Build train/val/test dataloaders. Pass ``None`` ids to skip a split."""
    loaders: dict[str, DataLoader] = {}
    common = dict(num_workers=num_workers, collate_fn=_collate)

    if train_ids is not None:
        ds = CoverDataset(
            data_path=data_path,
            dataset_path=dataset_path,
            data_split="train",
            clique_map=clique_map,
            track_ids=train_ids,
        )
        loaders["train"] = DataLoader(ds, batch_size=batch_size, shuffle=train_shuffle, drop_last=drop_last, **common)

    for split, ids in (("val", val_ids), ("test", test_ids)):
        if ids is None:
            continue
        ds = CoverDataset(
            data_path=data_path,
            dataset_path=dataset_path,
            data_split=split,
            track_ids=list(ids),
        )
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, **common)
    return loaders