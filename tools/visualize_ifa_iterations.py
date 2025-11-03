"""
IFA iteration visualization for Few-Shot Segmentation.

This script replays inference on selected query samples, captures the logits
produced by the decoder and IFA head at every iteration, and renders aligned
panels using either segmentation masks or pre-activation probability heatmaps:
    - RGB input
    - Optional ground-truth reference
    - Decoder result (mask or heatmap)
    - Each IFA iteration (mask or heatmap)
    - Final decoder+IFA fusion (mask or heatmap)

Outputs are written under `experiments/FSS_IFAIterViz/<run_id>/ifa_iterations`.

Example:
python -m tools.visualize_ifa_iterations with \
    checkpoint_path='experiments/FSS_Training/LoRA_20shot/best_model.pth' \
    num_samples=-1 use_ifa=True ifa_iters=3 viz_mode=heatmap
"""

import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from sacred import Experiment  # noqa: E402
from sacred.observers import FileStorageObserver  # noqa: E402

from datasets.disaster import DisasterDataset  # noqa: E402
from models.backbones.dino import DINOMultilayer  # noqa: E402
from utils.feature_viz import (  # noqa: E402
    denormalize_to_numpy,
    logits_to_probability_map,
    probability_to_heatmap_overlay,
    summarize_probability_map,
)
from utils.ifa import build_support_pack, run_ifa_inference  # noqa: E402

ex = Experiment("FSS_IFAIterViz")
ex.observers.append(FileStorageObserver("experiments/FSS_IFAIterViz"))

COLOR_PALETTE = np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8)


@ex.config
def cfg():
    with open("configs/disaster.yaml") as file:
        config = yaml.full_load(file)

    checkpoint_path = None
    sample_indices: Sequence[int] = []
    num_samples = -1  # <=0 processes the entire query split
    random_seed = 0
    max_iters_to_plot = -1  # <=0 means plot all
    save_mask_arrays = False
    save_prob_arrays = False
    viz_mode = "heatmap"  # {"heatmap", "mask"}
    target_class = 1
    show_mask_reference = True
    heatmap_cmap = "inferno"
    heatmap_alpha = 0.65
    heatmap_value_range = None  # e.g., [0.0, 1.0]
    # IFA knobs exposed for CLI overrides
    use_ifa = True
    ifa_iters = config.get("ifa_iters", 3)
    ifa_refine = config.get("ifa_refine", True)
    ifa_alpha = config.get("ifa_alpha", 0.3)
    ifa_ms_weights = list(config.get("ifa_ms_weights", [0.1, 0.2, 0.3, 0.4]))
    ifa_temp = config.get("ifa_temp", 10.0)
    ifa_fg_thresh = config.get("ifa_fg_thresh", 0.7)
    ifa_bg_thresh = config.get("ifa_bg_thresh", 0.6)
    ifa_use_fdm = config.get("ifa_use_fdm", True)

    config.update(
        {
            "checkpoint_path": checkpoint_path,
            "sample_indices": list(sample_indices),
            "num_samples": num_samples,
            "random_seed": random_seed,
            "max_iters_to_plot": max_iters_to_plot,
            "save_mask_arrays": save_mask_arrays,
            "save_prob_arrays": save_prob_arrays,
            "viz_mode": viz_mode,
            "target_class": target_class,
            "show_mask_reference": show_mask_reference,
            "heatmap_cmap": heatmap_cmap,
            "heatmap_alpha": heatmap_alpha,
            "heatmap_value_range": heatmap_value_range,
            "use_ifa": use_ifa,
            "ifa_iters": ifa_iters,
            "ifa_refine": ifa_refine,
            "ifa_alpha": ifa_alpha,
            "ifa_ms_weights": ifa_ms_weights,
            "ifa_temp": ifa_temp,
            "ifa_fg_thresh": ifa_fg_thresh,
            "ifa_bg_thresh": ifa_bg_thresh,
            "ifa_use_fdm": ifa_use_fdm,
        }
    )


def _sync_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    effective = dict(config)
    run_dir = os.path.dirname(effective["checkpoint_path"]) if effective.get("checkpoint_path") else None
    cfg_path = os.path.join(run_dir, "config.json") if run_dir else None
    if cfg_path and os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                train_meta = json.load(f)
            train_cfg = train_meta.get("config", train_meta)
            keys = [
                "model_name",
                "dino_version",
                "dinov2_size",
                "dinov3_size",
                "dinov3_weights_path",
                "dinov3_rope_dtype",
                "input_size",
                "num_classes",
                "dataset",
                "dataset_name",
                "number_of_shots",
                "dataset_dir",
                "model_path",
                "model_repo_path",
                "split_file",
                "train_split",
                "val_split",
                "fdm",
                "fdm_enable_apm",
                "fdm_apm_mode",
                "fdm_enable_acpa",
                "encoder_adapters",
                "ifa_iters",
                "ifa_refine",
                "ifa_alpha",
                "ifa_ms_weights",
                "ifa_temp",
                "ifa_fg_thresh",
                "ifa_bg_thresh",
                "ifa_use_fdm",
            ]
            for k in keys:
                if k in train_cfg:
                    effective[k] = train_cfg[k]
            print(f"[IFAIterViz] Synced config from training run at {cfg_path}")
        except Exception as exc:
            print(f"[IFAIterViz] Warning: failed to read training config at {cfg_path}: {exc}")
    effective["method"] = "multilayer"
    return effective


def _select_indices(total: int, requested: Sequence[int], num_samples: int, seed: int) -> List[int]:
    if requested:
        valid = sorted({idx for idx in requested if 0 <= idx < total})
        if not valid:
            raise ValueError("Provided sample_indices are outside dataset range.")
        return valid
    if num_samples <= 0 or num_samples >= total:
        return list(range(total))
    rng = random.Random(seed)
    pool = list(range(total))
    rng.shuffle(pool)
    return sorted(pool[:num_samples])


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = COLOR_PALETTE[np.clip(mask.astype(np.int64), 0, COLOR_PALETTE.shape[0] - 1)]
    return rgb.astype(np.float32) / 255.0


def _compute_iou(pred: torch.Tensor, target: torch.Tensor, class_id: int = 1) -> float:
    pred_class = (pred == class_id).cpu()
    target_class = (target == class_id).cpu()
    inter = torch.logical_and(pred_class, target_class).sum().item()
    union = torch.logical_or(pred_class, target_class).sum().item()
    if union == 0:
        return 0.0
    return inter / union


def _parse_value_range(value: Any) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"auto", "", "none"}:
            return None
        parts = [p for p in value.replace(",", " ").split() if p]
        if len(parts) == 2:
            vmin, vmax = float(parts[0]), float(parts[1])
            if vmax <= vmin:
                raise ValueError(f"Invalid heatmap_value_range ({vmin}, {vmax}); expected vmax > vmin")
            return vmin, vmax
        raise ValueError(
            f"Could not parse heatmap_value_range '{value}'. Expected two numbers or 'auto'."
        )
    if isinstance(value, (list, tuple)) and len(value) == 2:
        vmin, vmax = float(value[0]), float(value[1])
        if vmax <= vmin:
            raise ValueError(f"Invalid heatmap_value_range ({vmin}, {vmax}); expected vmax > vmin")
        return vmin, vmax
    raise ValueError("`heatmap_value_range` must be null or a sequence with two numeric values.")


def _normalize_viz_mode(mode: Any) -> str:
    if not isinstance(mode, str):
        return "heatmap"
    lowered = mode.strip().lower()
    if lowered not in {"heatmap", "mask"}:
        raise ValueError(f"Unsupported viz_mode '{mode}'. Use 'heatmap' or 'mask'.")
    return lowered


def _format_prob_stats(stats: Tuple[float, float, float]) -> str:
    return f"min {stats[0]:.2f} · mean {stats[1]:.2f} · max {stats[2]:.2f}"


def _make_heatmap_overlay(
    rgb_image: np.ndarray,
    logits: torch.Tensor,
    class_id: int,
    cmap_name: str,
    alpha: float,
    value_range: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, torch.Tensor, Tuple[float, float, float]]:
    prob_map = logits_to_probability_map(logits, class_index=class_id, reduce_batch=True)
    stats = summarize_probability_map(prob_map)
    overlay = probability_to_heatmap_overlay(
        image_rgb=rgb_image,
        prob_map=prob_map,
        cmap_name=cmap_name,
        alpha=alpha,
        value_range=value_range,
    ).astype(np.float32) / 255.0
    return overlay, prob_map, stats


def _prepare_model(config: Dict[str, Any], device: torch.device) -> DINOMultilayer:
    if config.get("model_name", "DINO") != "DINO":
        raise NotImplementedError("IFA iteration visualization currently supports only the DINO backbone.")

    legacy_method = config.get("method", "multilayer")
    if legacy_method not in (None, "multilayer"):
        raise ValueError(f"Only 'multilayer' method is supported, but the config requested '{legacy_method}'.")
    config["method"] = "multilayer"

    model = DINOMultilayer(
        version=config.get("dino_version", 2),
        num_classes=config["num_classes"],
        input_size=config["input_size"],
        model_repo_path=config["model_repo_path"],
        model_path=config["model_path"],
        dinov2_size=config.get("dinov2_size", "base"),
        dinov3_size=config.get("dinov3_size", "base"),
        dinov3_weights_path=config.get("dinov3_weights_path", None),
        dinov3_rope_dtype=config.get("dinov3_rope_dtype", "bf16"),
        encoder_adapters=config.get("encoder_adapters", "none"),
        fdm_enable_apm=(
            config.get("fdm", {}).get("enable_apm", False)
            if isinstance(config.get("fdm", {}), dict)
            else config.get("fdm_enable_apm", False)
        ),
        fdm_apm_mode=(
            config.get("fdm", {}).get("apm_mode", config.get("fdm_apm_mode", "S"))
            if isinstance(config.get("fdm", {}), dict)
            else config.get("fdm_apm_mode", "S")
        ),
        fdm_enable_acpa=(
            config.get("fdm", {}).get("enable_acpa", False)
            if isinstance(config.get("fdm", {}), dict)
            else config.get("fdm_enable_acpa", False)
        ),
    )
    state = torch.load(config["checkpoint_path"], map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


@ex.automain
def main(_run, config: Dict[str, Any]):
    if config["checkpoint_path"] is None:
        raise ValueError("A `checkpoint_path` must be provided for visualization.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join(_run.observers[0].dir, "ifa_iterations")
    os.makedirs(output_dir, exist_ok=True)

    effective_config = _sync_training_config(config)
    if not effective_config.get("use_ifa", False):
        raise ValueError("Set `use_ifa=True` (either in the checkpoint config or CLI) for IFA visualization.")

    print(f"[IFAIterViz] Loading model from {effective_config['checkpoint_path']} on device {device}")
    model = _prepare_model(effective_config, device)

    dataset_name = effective_config.get("dataset_name", "disaster")
    if dataset_name != "disaster":
        raise NotImplementedError("IFA visualization currently supports the 'disaster' dataset only.")

    split_file = effective_config["split_file"]
    train_split = str(effective_config.get("train_split", "support")).strip("\"'")
    val_split = str(effective_config.get("val_split", "query")).strip("\"'")
    support_set = DisasterDataset(root=".", split_file=split_file, mode=train_split)
    query_set = DisasterDataset(root=".", split_file=split_file, mode=val_split)

    support_pack = build_support_pack(
        model=model,
        support_dataset=support_set,
        config=effective_config,
        device=device,
        max_support=effective_config.get("number_of_shots", 1),
    )

    viz_mode = _normalize_viz_mode(effective_config.get("viz_mode", "heatmap"))
    target_class = int(effective_config.get("target_class", 1))
    show_mask_ref = bool(effective_config.get("show_mask_reference", False))
    heatmap_cmap = effective_config.get("heatmap_cmap", "inferno")
    heatmap_alpha = float(effective_config.get("heatmap_alpha", 0.65))
    heatmap_range = None
    if viz_mode == "heatmap":
        try:
            heatmap_range = _parse_value_range(effective_config.get("heatmap_value_range", None))
        except ValueError as exc:
            raise ValueError(f"Failed to parse heatmap_value_range: {exc}") from exc

    indices = _select_indices(
        total=len(query_set),
        requested=effective_config.get("sample_indices", []),
        num_samples=effective_config.get("num_samples", -1),
        seed=effective_config.get("random_seed", 0),
    )
    max_iters = int(effective_config.get("max_iters_to_plot", -1))
    save_masks = bool(effective_config.get("save_mask_arrays", False))
    save_probs = bool(effective_config.get("save_prob_arrays", False))
    arrays_dir = None
    if save_masks or save_probs:
        arrays_dir = os.path.join(output_dir, "npz")
        os.makedirs(arrays_dir, exist_ok=True)

    alpha = float(effective_config.get("ifa_alpha", 0.3))
    use_fdm = bool(effective_config.get("ifa_use_fdm", True))

    summaries: List[Dict[str, Any]] = []
    print(f"[IFAIterViz] Visualizing {len(indices)} samples...")

    for idx in indices:
        image_t, mask_t, image_path = query_set[idx]
        if image_t.numel() == 0:
            print(f"[IFAIterViz] Skipping sample {idx} due to loading failure.")
            continue

        image_batch = image_t.unsqueeze(0).to(device)
        mask = mask_t.to(device)

        with torch.no_grad():
            decoder_logits = model(image_batch)
            decoder_logits = F.interpolate(
                decoder_logits,
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            logits_ifa_with_hist = run_ifa_inference(
                model=model,
                image=image_batch,
                version=effective_config.get("dino_version", 2),
                input_size=effective_config.get("input_size", 512),
                ifa_cfg=effective_config,
                support_pack=support_pack,
                out_size=mask.shape[-2:],
                use_fdm_on_feats=use_fdm,
                capture_history=True,
            )

        if not isinstance(logits_ifa_with_hist, tuple):
            raise RuntimeError("Expected run_ifa_inference to return history when capture_history=True.")
        logits_ifa, history = logits_ifa_with_hist
        fused_logits = (1.0 - alpha) * decoder_logits + alpha * logits_ifa

        iter_logits = history.get("fused_iter_logits", []) or []
        if max_iters > 0:
            iter_logits = iter_logits[:max_iters]

        decoder_pred = decoder_logits.argmax(1)
        ifa_pred = logits_ifa.argmax(1)
        fused_pred = fused_logits.argmax(1)
        iter_preds = [logit.argmax(1) for logit in iter_logits]

        decoder_iou = _compute_iou(decoder_pred, mask, target_class)
        ifa_iou = _compute_iou(ifa_pred, mask, target_class)
        fused_iou = _compute_iou(fused_pred, mask, target_class)
        iter_ious = [_compute_iou(pred, mask, target_class) for pred in iter_preds]

        rgb = denormalize_to_numpy(image_t)

        decoder_prob = None
        ifa_prob = None
        fused_prob = None
        iter_probs: List[torch.Tensor] = []
        decoder_stats: Optional[Tuple[float, float, float]] = None
        ifa_stats: Optional[Tuple[float, float, float]] = None
        fused_stats: Optional[Tuple[float, float, float]] = None
        iter_stats: List[Tuple[float, float, float]] = []
        iter_heatmaps: List[np.ndarray] = []

        if viz_mode == "heatmap":
            decoder_overlay, decoder_prob, decoder_stats = _make_heatmap_overlay(
                rgb_image=rgb,
                logits=decoder_logits,
                class_id=target_class,
                cmap_name=heatmap_cmap,
                alpha=heatmap_alpha,
                value_range=heatmap_range,
            )
            ifa_overlay, ifa_prob, ifa_stats = _make_heatmap_overlay(
                rgb_image=rgb,
                logits=logits_ifa,
                class_id=target_class,
                cmap_name=heatmap_cmap,
                alpha=heatmap_alpha,
                value_range=heatmap_range,
            )
            fused_overlay, fused_prob, fused_stats = _make_heatmap_overlay(
                rgb_image=rgb,
                logits=fused_logits,
                class_id=target_class,
                cmap_name=heatmap_cmap,
                alpha=heatmap_alpha,
                value_range=heatmap_range,
            )
            for logit in iter_logits:
                overlay, prob_map, stats = _make_heatmap_overlay(
                    rgb_image=rgb,
                    logits=logit,
                    class_id=target_class,
                    cmap_name=heatmap_cmap,
                    alpha=heatmap_alpha,
                    value_range=heatmap_range,
                )
                iter_heatmaps.append(overlay)
                iter_probs.append(prob_map)
                iter_stats.append(stats)
        else:
            if save_probs:
                decoder_prob = logits_to_probability_map(decoder_logits, class_index=target_class, reduce_batch=True)
                ifa_prob = logits_to_probability_map(logits_ifa, class_index=target_class, reduce_batch=True)
                fused_prob = logits_to_probability_map(fused_logits, class_index=target_class, reduce_batch=True)
                iter_probs = [
                    logits_to_probability_map(logit, class_index=target_class, reduce_batch=True) for logit in iter_logits
                ]
                decoder_stats = summarize_probability_map(decoder_prob)
                ifa_stats = summarize_probability_map(ifa_prob)
                fused_stats = summarize_probability_map(fused_prob)
                iter_stats = [summarize_probability_map(prob) for prob in iter_probs]

        panels: List[Tuple[str, np.ndarray]] = [("RGB Input", rgb)]
        if show_mask_ref:
            panels.append(("Ground Truth", _mask_to_rgb(mask.cpu().numpy())))

        if viz_mode == "heatmap":
            panels.append((f"Decoder probs ({_format_prob_stats(decoder_stats)})", decoder_overlay))
        else:
            panels.append(
                (
                    f"Decoder mask (IoU {decoder_iou*100:.1f})",
                    _mask_to_rgb(decoder_pred.squeeze(0).cpu().numpy()),
                )
            )

        for i, pred in enumerate(iter_preds, start=1):
            if viz_mode == "heatmap":
                stats = iter_stats[i - 1] if i - 1 < len(iter_stats) else (0.0, 0.0, 0.0)
                panels.append((f"IFA iter {i} ({_format_prob_stats(stats)})", iter_heatmaps[i - 1]))
            else:
                panels.append(
                    (
                        f"IFA iter {i} (IoU {iter_ious[i-1]*100:.1f})",
                        _mask_to_rgb(pred.squeeze(0).cpu().numpy()),
                    )
                )

        if viz_mode == "heatmap":
            panels.append((f"IFA logits ({_format_prob_stats(ifa_stats)})", ifa_overlay))
            panels.append((f"Fused probs ({_format_prob_stats(fused_stats)})", fused_overlay))
        else:
            panels.append(
                (
                    f"IFA mask (IoU {ifa_iou*100:.1f})",
                    _mask_to_rgb(ifa_pred.squeeze(0).cpu().numpy()),
                )
            )
            panels.append(
                (
                    f"Fused mask (IoU {fused_iou*100:.1f})",
                    _mask_to_rgb(fused_pred.squeeze(0).cpu().numpy()),
                )
            )

        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(4 * len(panels), 4),
            gridspec_kw={"wspace": 0.02},
        )
        if len(panels) == 1:
            axes = [axes]
        for ax, (title, data) in zip(axes, panels):
            ax.imshow(data)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle(os.path.basename(image_path), fontsize=12)
        fig.tight_layout(pad=0.5, w_pad=0.2, h_pad=0.4)
        fig.subplots_adjust(wspace=0.02)

        fig_path = os.path.join(output_dir, f"sample_{idx:04d}.png")
        fig.savefig(fig_path, dpi=160, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        _run.add_artifact(fig_path, name=f"ifa_iterations/sample_{idx:04d}.png")

        if (save_masks or save_probs) and arrays_dir is not None:
            npz_path = os.path.join(arrays_dir, f"sample_{idx:04d}.npz")
            payload: Dict[str, np.ndarray] = {}
            if save_masks:
                iter_arrays = [pred.squeeze(0).cpu().numpy().astype(np.uint8) for pred in iter_preds]
                payload.update(
                    {
                        "decoder_mask": decoder_pred.squeeze(0).cpu().numpy().astype(np.uint8),
                        "ifa_mask": ifa_pred.squeeze(0).cpu().numpy().astype(np.uint8),
                        "fused_mask": fused_pred.squeeze(0).cpu().numpy().astype(np.uint8),
                        "iter_masks": np.stack(iter_arrays)
                        if iter_arrays
                        else np.zeros((0, *mask.shape[-2:]), dtype=np.uint8),
                        "gt_mask": mask.cpu().numpy().astype(np.uint8),
                    }
                )
            if save_probs:
                if decoder_prob is None:
                    decoder_prob = logits_to_probability_map(decoder_logits, class_index=target_class, reduce_batch=True)
                    ifa_prob = logits_to_probability_map(logits_ifa, class_index=target_class, reduce_batch=True)
                    fused_prob = logits_to_probability_map(fused_logits, class_index=target_class, reduce_batch=True)
                    iter_probs = [
                        logits_to_probability_map(logit, class_index=target_class, reduce_batch=True) for logit in iter_logits
                    ]
                    decoder_stats = summarize_probability_map(decoder_prob)
                    ifa_stats = summarize_probability_map(ifa_prob)
                    fused_stats = summarize_probability_map(fused_prob)
                    iter_stats = [summarize_probability_map(prob) for prob in iter_probs]
                iter_prob_arrays = [prob.cpu().numpy().astype(np.float32) for prob in iter_probs]
                payload.update(
                    {
                        "decoder_prob": decoder_prob.cpu().numpy().astype(np.float32),
                        "ifa_prob": ifa_prob.cpu().numpy().astype(np.float32),
                        "fused_prob": fused_prob.cpu().numpy().astype(np.float32),
                        "iter_probs": np.stack(iter_prob_arrays)
                        if iter_prob_arrays
                        else np.zeros((0, *mask.shape[-2:]), dtype=np.float32),
                    }
                )
            if payload:
                np.savez_compressed(npz_path, **payload)
                _run.add_artifact(npz_path, name=f"ifa_iterations_npz/sample_{idx:04d}.npz")

        summaries.append(
            {
                "index": idx,
                "image_path": image_path,
                "decoder_iou": decoder_iou,
                "ifa_iou": ifa_iou,
                "fused_iou": fused_iou,
                "iter_ious": iter_ious,
                "decoder_prob_stats": decoder_stats if decoder_stats is not None else (0.0, 0.0, 0.0),
                "ifa_prob_stats": ifa_stats if ifa_stats is not None else (0.0, 0.0, 0.0),
                "fused_prob_stats": fused_stats if fused_stats is not None else (0.0, 0.0, 0.0),
                "iter_prob_stats": iter_stats,
            }
        )

    summary_path = os.path.join(output_dir, "ifa_iteration_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    _run.add_artifact(summary_path, name="ifa_iterations/metrics.json")

    print(f"[IFAIterViz] Saved {len(summaries)} visualizations to {output_dir}")
    return f"Completed IFA iteration visualization for {len(summaries)} samples."
