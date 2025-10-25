import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple

from models.backbones.dino import DINO_linear
from modules.module_IFA.ifa_head import IFAHead


def extract_encoder_features(
    model: DINO_linear,
    image: torch.Tensor,
    version: int,
    input_size: int,
) -> List[torch.Tensor]:
    """Extract intermediate encoder features as list of [1, C, H, W]."""
    if version == 3:
        input_dim = int(input_size / 16) * 16
    elif version == 2:
        input_dim = int(input_size / 14) * 14
    else:
        raise ValueError(f"Unsupported DINO version: {version}")

    x = F.interpolate(image, size=[input_dim, input_dim], mode="bilinear", align_corners=False)
    feats = model.encoder(x)
    feats_list = list(feats) if isinstance(feats, (list, tuple)) else [feats]
    feats_list = [f.contiguous() for f in feats_list]
    return feats_list


def _ensure_apm_shape_for_feat(apm, feat: torch.Tensor, apm_mode: str) -> None:
    """Initialize APM parameters to be batch-agnostic for a given feature tensor."""
    if apm is None:
        return
    _, c, h, w = feat.shape
    mode = (apm_mode or "S").upper()
    if mode not in ("S", "M"):
        mode = "S"
    target = (1, 1, h, w) if mode == "S" else (1, c, h, w)
    if getattr(apm, "target_shape", None) != target:
        apm.target_shape = target
    apm._ensure_initialized(feat)


def apply_fdm_to_features(
    model: DINO_linear,
    feats: List[torch.Tensor],
    method: str,
) -> List[torch.Tensor]:
    """
    Optionally apply FDM (APM → ACPA) to encoder features, following the same policy
    as training: deeper-two only for multilayer, last for linear.
    """
    # If neither module is enabled, fast-return
    has_apm = getattr(model, "apm", None) is not None and bool(getattr(model, "fdm_enable_apm", False))
    has_acpa = getattr(model, "acpa", None) is not None and bool(getattr(model, "fdm_enable_acpa", False))
    if not (has_apm or has_acpa):
        return feats

    out = list(feats)
    if method == "multilayer":
        n = len(out)
        apply_ids = list(range(max(0, n - 2), n))  # deeper-two
    else:
        apply_ids = [len(out) - 1]  # last only

    for i in apply_ids:
        f = out[i]
        if has_apm:
            _ensure_apm_shape_for_feat(model.apm, f, getattr(model, "fdm_apm_mode", "S"))
            f = model.apm(f)
        if has_acpa:
            f = model.acpa(f)
        out[i] = f

    return out


@torch.no_grad()
def run_ifa_inference(
    model: DINO_linear,
    image: torch.Tensor,
    method: str,
    version: int,
    input_size: int,
    ifa_cfg: Dict[str, Any],
    support_pack: Dict[str, Any],
    out_size: Tuple[int, int],
    use_fdm_on_feats: bool = False,
) -> torch.Tensor:
    """
    Run IFA on top of encoder features with optional FDM pre-processing of features.
    Returns logits tensor [1, 2, out_h, out_w].
    """
    # Build IFA head
    ifa_head = IFAHead(
        temperature=float(ifa_cfg.get("ifa_temp", 10.0)),
        fg_thresh=float(ifa_cfg.get("ifa_fg_thresh", 0.7)),
        bg_thresh=float(ifa_cfg.get("ifa_bg_thresh", 0.6)),
        iters=int(ifa_cfg.get("ifa_iters", 3)),
        use_refine=bool(ifa_cfg.get("ifa_refine", True)),
    )

    # Query features
    feats_q_ms = extract_encoder_features(model, image, version, input_size)
    # Support features (cached per-scale per-K)
    feats_s_ms: List[List[torch.Tensor]] = support_pack["feats_s_ms"]
    masks_s: List[torch.Tensor] = support_pack["masks_s"]

    if use_fdm_on_feats:
        feats_q_ms = apply_fdm_to_features(model, feats_q_ms, method)
        # Transform support features per scale respecting scale indices (not K)
        has_apm = getattr(model, "apm", None) is not None and bool(getattr(model, "fdm_enable_apm", False))
        has_acpa = getattr(model, "acpa", None) is not None and bool(getattr(model, "fdm_enable_acpa", False))
        if has_apm or has_acpa:
            n_scales = len(feats_s_ms)
            if method == "multilayer":
                apply_ids = list(range(max(0, n_scales - 2), n_scales))
            else:
                apply_ids = [n_scales - 1]
            new_feats_s_ms: List[List[torch.Tensor]] = []
            for si in range(n_scales):
                scale_feats = []
                for f in feats_s_ms[si]:
                    ff = f
                    if si in apply_ids:
                        if has_apm:
                            _ensure_apm_shape_for_feat(model.apm, ff, getattr(model, "fdm_apm_mode", "S"))
                            ff = model.apm(ff)
                        if has_acpa:
                            ff = model.acpa(ff)
                    scale_feats.append(ff)
                new_feats_s_ms.append(scale_feats)
            feats_s_ms = new_feats_s_ms

    if method == "linear":
        last_q = feats_q_ms[-1]
        last_s = [feats_s_ms[-1][k] for k in range(len(masks_s))]
        logits = ifa_head.run_single_scale(last_s, masks_s, last_q)
        logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        return logits
    elif method == "multilayer":
        ms_weights = [float(w) for w in ifa_cfg.get("ifa_ms_weights", [0.1, 0.2, 0.3, 0.4])]
        # Skip zero-weight scales to save memory/compute
        idxs = [i for i, w in enumerate(ms_weights) if w > 1e-8]
        if len(idxs) == 0:
            # Fallback: keep last scale
            idxs = [len(ms_weights) - 1]
            ms_weights = [1.0]
        else:
            ms_weights = [ms_weights[i] for i in idxs]
        feats_q_ms_sel = [feats_q_ms[i] for i in idxs]
        feats_s_ms_sel = [feats_s_ms[i] for i in idxs]
        logits = ifa_head.run_multi_scale(
            feats_s_ms=feats_s_ms_sel,
            masks_s=masks_s,
            feats_q_ms=feats_q_ms_sel,
            out_size=out_size,
            weights=ms_weights,
        )
        return logits
    else:
        raise NotImplementedError(f"IFA inference not supported for method '{method}'.")


def run_ifa_training_logits(
    model: DINO_linear,
    image: torch.Tensor,
    method: str,
    version: int,
    input_size: int,
    ifa_cfg: Dict[str, Any],
    support_pack: Dict[str, Any],
    out_size: Tuple[int, int],
    use_fdm_on_feats: bool = False,
) -> torch.Tensor:
    """
    Differentiable IFA forward for training. Mirrors run_ifa_inference but without no_grad.
    Returns logits tensor [1, 2, out_h, out_w].
    """
    ifa_head = IFAHead(
        temperature=float(ifa_cfg.get("ifa_temp", 10.0)),
        fg_thresh=float(ifa_cfg.get("ifa_fg_thresh", 0.7)),
        bg_thresh=float(ifa_cfg.get("ifa_bg_thresh", 0.6)),
        iters=int(ifa_cfg.get("ifa_iters", 3)),
        use_refine=bool(ifa_cfg.get("ifa_refine", True)),
    )

    feats_q_ms = extract_encoder_features(model, image, version, input_size)
    feats_s_ms: List[List[torch.Tensor]] = support_pack["feats_s_ms"]
    masks_s: List[torch.Tensor] = support_pack["masks_s"]

    if use_fdm_on_feats:
        feats_q_ms = apply_fdm_to_features(model, feats_q_ms, method)
        has_apm = getattr(model, "apm", None) is not None and bool(getattr(model, "fdm_enable_apm", False))
        has_acpa = getattr(model, "acpa", None) is not None and bool(getattr(model, "fdm_enable_acpa", False))
        if has_apm or has_acpa:
            n_scales = len(feats_s_ms)
            if method == "multilayer":
                apply_ids = list(range(max(0, n_scales - 2), n_scales))
            else:
                apply_ids = [n_scales - 1]
            new_feats_s_ms: List[List[torch.Tensor]] = []
            for si in range(n_scales):
                scale_feats = []
                for f in feats_s_ms[si]:
                    ff = f
                    if si in apply_ids:
                        if has_apm:
                            _ensure_apm_shape_for_feat(model.apm, ff, getattr(model, "fdm_apm_mode", "S"))
                            ff = model.apm(ff)
                        if has_acpa:
                            ff = model.acpa(ff)
                    scale_feats.append(ff)
                new_feats_s_ms.append(scale_feats)
            feats_s_ms = new_feats_s_ms

    if method == "linear":
        last_q = feats_q_ms[-1]
        last_s = [feats_s_ms[-1][k] for k in range(len(masks_s))]
        logits = ifa_head.run_single_scale(last_s, masks_s, last_q)
        logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        return logits
    elif method == "multilayer":
        ms_weights = [float(w) for w in ifa_cfg.get("ifa_ms_weights", [0.1, 0.2, 0.3, 0.4])]
        logits = ifa_head.run_multi_scale(
            feats_s_ms=feats_s_ms,
            masks_s=masks_s,
            feats_q_ms=feats_q_ms,
            out_size=out_size,
            weights=ms_weights,
        )
        return logits
    else:
        raise NotImplementedError(f"IFA training not supported for method '{method}'.")
