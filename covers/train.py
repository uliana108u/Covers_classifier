"""GPU training + test inference for the neural cover-detection model.

Reusable core behind ``train_gpu.ipynb``. Handles:
- train/val loop with triplet (cosine) + soft-label CE losses and mixed precision
- validation: embed val tracks -> pairwise cosine ranking -> nDCG@100 / MRR
- early stopping on nDCG@100 with best-model checkpointing + resume
- test inference: embed test tracks -> top-K ranking -> submission.txt

(Original baseline early-stopped on mAP; we report nDCG@100 since it is the contest
metric and behaves the same way for early stopping.)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from covers import data as data_mod
from covers.dataset import CoverDataset, _collate, make_dataloaders
from covers.eval import mean_ndcg_at_100
from covers.model import Resnet50, soft_label_ce_loss, triplet_loss_fn
from covers.ranking import rank_top_k


@dataclass
class TrainCfg:
    data_path: str
    dataset_path: str
    num_classes: int
    test_dataset_path: Optional[str] = None  # dir with test CQTs (defaults to <data_path>/test)
    device: str = "cuda"
    batch_size: int = 128
    num_workers: int = 2
    epochs: int = 25
    lr: float = 1e-4
    patience: int = 4
    triplet_margin: float = 0.3
    dropout: float = 0.1
    mixed_precision: bool = True
    val_k: int = 100
    resume: Optional[str] = None  # path to checkpoint (.pt) to resume from
    checkpoint_dir: str = "checkpoints"
    max_epochs_override: Optional[int] = None
    limit_train: Optional[int] = None  # cap train tracks (smoke tests / sanity runs)
    limit_val: Optional[int] = None  # cap val tracks
    limit_test: Optional[int] = None  # cap test tracks (inference smoke tests)


def get_device(name: str) -> torch.device:
    if name.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        print("WARNING: CUDA requested but not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(name)


class EarlyStopper:
    def __init__(self, patience: int, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_map = -np.inf

    def __call__(self, score: float) -> bool:
        if score > self.best_map + self.delta:
            self.best_map = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def embed_tracks(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "embed",
) -> tuple[List[int], np.ndarray]:
    """Forward all CQTs through the model, returning (track_ids, f_c matrix (N, 2048))."""
    model.eval()
    track_ids: List[int] = []
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, ncols=80, leave=False):
            feats = model(batch["cqt"].to(device))
            emb = feats["f_c"].cpu().numpy()
            if emb.ndim == 1:
                emb = emb[None]
            track_ids.extend(batch["track_id"].tolist())
            chunks.append(emb)
    return track_ids, np.vstack(chunks)


def build_track_to_clique(val_map: pd.DataFrame, val_track_ids: List[int]) -> Dict[int, int]:
    valid = set(val_track_ids)
    mapping: Dict[int, int] = {}
    for clique, row in val_map.iterrows():
        for tid in row["versions"]:
            tid = int(tid)
            if tid in valid:
                mapping[tid] = int(clique)
    return mapping


def compute_val_score(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    track_to_clique: Dict[int, int],
    k: int = 100,
) -> Dict[str, float]:
    """Embed val tracks and return {'ndcg@100': ..., 'mrr': ...}."""
    track_ids, emb = embed_tracks(model, val_loader, device, desc="val")
    ranking = rank_top_k(emb, k=k)
    ranked_by_id = {track_ids[q]: [track_ids[c] for c in cands[:k]] for q, cands in ranking.items()}
    ndcg = mean_ndcg_at_100(ranked_by_id, track_to_clique)

    # MRR: reciprocal rank of the first relevant candidate
    import math
    rrs = []
    for qid, cands in ranked_by_id.items():
        qc = track_to_clique.get(qid)
        if qc is None:
            continue
        for pos, cid in enumerate(cands, start=1):
            if track_to_clique.get(cid) == qc:
                rrs.append(1.0 / pos)
                break
    mrr = float(np.mean(rrs)) if rrs else 0.0
    return {"ndcg@100": float(ndcg), "mrr": mrr, "num_queries": len(ranked_by_id)}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainCfg,
    epoch: int,
    score: float,
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "score": score,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_classes": cfg.num_classes,
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: torch.device,
) -> tuple[int, float]:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    return int(state["epoch"]) + 1, float(state["score"])


def run(cfg: TrainCfg, progress: Optional[callable] = None) -> Resnet50:
    """Train/validate, early-stop on val nDCG@100, save checkpoints. Returns model."""
    device = get_device(cfg.device)
    cfg.device = device
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    train_map, val_map, train_ids, val_ids, num_classes = data_mod.train_val_clique_maps(cfg.data_path)
    if num_classes != cfg.num_classes:
        raise ValueError(f"num_classes {num_classes} != cfg.num_classes {cfg.num_classes}")
    if cfg.limit_train:
        train_ids = train_ids[: cfg.limit_train]
    if cfg.limit_val:
        val_ids = val_ids[: cfg.limit_val]
    track_to_clique = build_track_to_clique(val_map, val_ids)

    loaders = make_dataloaders(
        data_path=cfg.data_path,
        dataset_path=cfg.dataset_path,
        clique_map=train_map,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=None,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    train_loader, val_loader = loaders["train"], loaders["val"]

    model = Resnet50(num_channels=1, num_classes=cfg.num_classes, dropout=cfg.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    triplet_loss = triplet_loss_fn(cfg.triplet_margin).to(device)
    use_amp = cfg.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    stopper = EarlyStopper(cfg.patience)

    start_epoch, best_score = 0, -np.inf
    if cfg.resume and os.path.exists(cfg.resume):
        start_epoch, best_score = load_checkpoint(model, optimizer, cfg.resume, device)
        print(f"resumed from {cfg.resume} (next epoch {start_epoch}, best {best_score:.4f})")

    history: List[dict] = []
    epochs = cfg.max_epochs_override or cfg.epochs
    for epoch in range(start_epoch, epochs):
        model.train()
        ep_loss = ep_tri = ep_ce = 0.0
        steps = 0
        pbar = tqdm(train_loader, desc=f"train {epoch}", ncols=100)
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                a = model(batch["cqt"].to(device))
                p = model(batch["positive_cqt"].to(device))
                n = model(batch["negative_cqt"].to(device))
                l_tri = triplet_loss(a["f_t"], p["f_t"], n["f_t"])
                l_ce = soft_label_ce_loss(a["cls"], batch["clique"], cfg.num_classes, device)
                loss = l_tri + l_ce
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ep_loss += loss.item()
            ep_tri += l_tri.item()
            ep_ce += l_ce.item()
            steps += 1
            pbar.set_postfix(tri=f"{ep_tri/steps:.3f}", ce=f"{ep_ce/steps:.3f}")

        scores = compute_val_score(model, val_loader, device, track_to_clique, cfg.val_k)
        val_score = scores["ndcg@100"]
        row = {
            "epoch": epoch,
            "train_loss": ep_loss / max(steps, 1),
            "train_triplet": ep_tri / max(steps, 1),
            "train_ce": ep_ce / max(steps, 1),
            **scores,
        }
        history.append(row)
        with open(os.path.join(cfg.checkpoint_dir, "history.jsonl"), "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row, indent=2))
        if progress:
            progress(row)

        if val_score > stopper.best_map:
            stopper.best_map = val_score
            stopper.counter = 0
            best_path = os.path.join(cfg.checkpoint_dir, "best-model.pt")
            save_checkpoint(model, optimizer, cfg, epoch, val_score, best_path)
            print(f"epoch {epoch}: saved best nDCG@100 {val_score:.4f} -> {best_path}")
        else:
            stopper.counter += 1
            print(f"epoch {epoch}: val nDCG@100 {val_score:.4f} did not improve")
            if stopper.counter >= cfg.patience:
                print("early stopping")
                break

    # load best weights back into the model for inference
    best_path = os.path.join(cfg.checkpoint_dir, "best-model.pt")
    if os.path.exists(best_path):
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        print(f"loaded best weights ({state['epoch']}) for inference")
    return model


def infer_test(
    model: nn.Module,
    cfg: TrainCfg,
    submission_path: str = "submission.txt",
    k: int = 100,
    batch_size: int = 256,
) -> Dict[int, List[int]]:
    """Embed all test tracks and write top-K ``submission.txt`` (contest format)."""
    device = get_device(cfg.device)
    model.to(device)
    test_ids = [int(t) for t in data_mod.load_split_ids("test", cfg.data_path)]
    if cfg.limit_test:
        test_ids = test_ids[: cfg.limit_test]
    test_dataset_path = cfg.test_dataset_path or os.path.join(cfg.data_path, "test")
    loader = DataLoader(
        CoverDataset(
            data_path=cfg.data_path,
            dataset_path=test_dataset_path,
            data_split="test",
            track_ids=test_ids,
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
    )
    track_ids, emb = embed_tracks(model, loader, device, desc="infer-test")
    ranking = rank_top_k(emb, k=k)
    ranked_by_id = {track_ids[q]: [track_ids[c] for c in cands[:k]] for q, cands in ranking.items()}
    with open(submission_path, "w") as fh:
        for qid in sorted(ranked_by_id):
            fh.write(f"{qid} " + " ".join(map(str, ranked_by_id[qid])))
            fh.write("\n")
    print(f"submission written to {submission_path} ({len(ranked_by_id)} queries)")
    return ranked_by_id


__all__ = ["TrainCfg", "run", "infer_test", "compute_val_score", "embed_tracks", "EarlyStopper", "get_device"]