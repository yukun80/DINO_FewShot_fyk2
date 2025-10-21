import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Optional


class DPTDecoder(nn.Module):
    """
    DPT-style multi-layer decoder aligned with modules/module_segdino.

    - Takes 4 intermediate feature maps from the ViT backbone (same spatial size):
      feats: Sequence[Tensor] of length 4, each [B, C, H, W]
    - For each layer: 1x1 projection to configured out_channels[i], then 3x3 conv to `features`.
    - Upsamples to the first layer's spatial size (for robustness), concatenates, and outputs logits via 1x1 conv.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        features: int = 128,
        out_channels: Optional[Sequence[int]] = None,
        use_bn: bool = False,
    ) -> None:
        super().__init__()

        if out_channels is None:
            # Default aligned with module_segdino for DINOv{2,3} small/base
            out_channels = [96, 192, 384, 768]

        assert len(out_channels) == 4, "out_channels must contain 4 entries"

        # 1x1 per-layer projection from backbone embed_dim -> configured out_channels[i]
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(in_channels, oc, kernel_size=1, stride=1, padding=0, bias=True)
                for oc in out_channels
            ]
        )

        # 3x3 per-layer conv to unify channels to `features`
        self.refine = nn.ModuleList(
            [
                nn.Conv2d(oc, features, kernel_size=3, stride=1, padding=1, bias=False)
                for oc in out_channels
            ]
        )

        self.use_bn = use_bn
        if use_bn:
            self.refine_bn = nn.ModuleList([nn.BatchNorm2d(features) for _ in range(4)])

        self.act = nn.GELU()

        # Final fusion and classification
        self.output_conv = nn.Conv2d(features * 4, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        assert isinstance(feats, (list, tuple)) and len(feats) == 4, "Expect 4 feature maps"

        # Per-layer projection + refinement
        proc = []
        for i, x in enumerate(feats):
            x = self.projects[i](x)
            x = self.refine[i](x)
            if self.use_bn:
                x = self.refine_bn[i](x)
            x = self.act(x)
            proc.append(x)

        # Align to the first layer's spatial size for robustness
        target_hw = proc[0].shape[-2:]
        proc = [
            p if p.shape[-2:] == target_hw else F.interpolate(p, size=target_hw, mode="bilinear", align_corners=True)
            for p in proc
        ]

        fused = torch.cat(proc, dim=1)
        out = self.output_conv(fused)
        return out
