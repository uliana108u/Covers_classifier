"""Neural model: ResNet50 with IBN-Net + GeM pooling + dual classifier head.

Ported from ``baseline/baseline/models/modules.py`` (architecture unchanged) so the
training notebook can import it from this package. The model maps a single CQT
spectrogram (84 freq bins x 50 time frames, 1 channel) to:

- ``f_t`` — 2048-d embedding used for the triplet (cosine) loss
- ``f_c`` — 2048-d embedding (BatchNorm-ed ``f_t``) used for retrieval
- ``cls`` — logits over ~39.5k cliques for the classification auxiliary loss
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """Generalized Mean pooling (trainable p)."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(
            1.0 / self.p
        )


class IBN(nn.Module):
    """Instance-Batch Normalization: first half of channels InstanceNorm, rest BatchNorm."""

    def __init__(self, planes: int, ratio: float = 0.5):
        super().__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        return torch.cat((out1, out2), 1)


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        last: bool = False,
        downsample: nn.Module | None = None,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.ibn = IBN(out_channels, ratio=0.5) if not last else nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=bias
        )
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.clone()
        x = self.conv1(x)
        x = self.ibn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out = residual + x
        return self.relu(out)


class Resnet50(nn.Module):
    def __init__(
        self,
        emb_dim: int = 2048,
        num_channels: int = 1,
        num_classes: int = 8858,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(blocks=3, planes=64, stride=1)
        self.layer2 = self._make_layer(blocks=4, planes=128, stride=2)
        self.layer3 = self._make_layer(blocks=6, planes=256, stride=2)
        self.layer4 = self._make_layer(blocks=3, planes=512, stride=1, last=True)

        self.gem_pool = GeM()
        self.dropout = nn.Dropout(p=dropout)

        self.bn_fc = nn.BatchNorm1d(emb_dim)
        self.fc = nn.Linear(emb_dim, num_classes, bias=False)
        nn.init.kaiming_normal_(self.fc.weight)

    def _make_layer(self, blocks: int, planes: int, stride: int = 1, last: bool = False) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [
            Bottleneck(
                in_channels=self.in_channels,
                out_channels=planes,
                stride=stride,
                downsample=downsample,
                last=last,
            )
        ]
        self.in_channels = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(in_channels=self.in_channels, out_channels=planes, last=last))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Expect (B, 84, 50) CQT; internally unsqueezed to (B, 1, 84, 50)."""
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.max_pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        f_t = self.gem_pool(x)
        f_t = self.dropout(torch.flatten(f_t, start_dim=1))

        f_c = self.bn_fc(f_t)
        cls = self.fc(f_c)
        return {"f_t": f_t, "f_c": f_c, "cls": cls}


def build_embeddings_for_batch(batch: dict, model: nn.Module, device: torch.device) -> dict[str, torch.Tensor]:
    """Run forward passes for a triplet batch (anchor/positive/negative)."""
    anchor = model(batch["anchor"].to(device))
    positive = model(batch["positive"].to(device))
    negative = model(batch["negative"].to(device))
    return {"anchor": anchor, "positive": positive, "negative": negative}


def triplet_loss_fn(margin: float = 0.3) -> nn.Module:
    """Cosine-distance triplet margin loss (as used by the original baseline)."""
    return nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=margin
    )


def soft_label_ce_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    """Cross-entropy with label smoothing via one-hot soft labels."""
    one_hot = F.one_hot(labels.long(), num_classes=num_classes).float().to(device)
    return F.cross_entropy(logits, one_hot)