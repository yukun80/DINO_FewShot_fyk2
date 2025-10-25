"""
IFA iteration visualization for Few-Shot Segmentation.

This script replays inference on selected query samples, captures the logits
produced by the IFA head at every iteration, and renders side-by-side masks for:
    - RGB input
    - Ground-truth mask
    - Decoder-only prediction
    - Each IFA iteration (fused across scales)
    - Final decoder+IFA fusion

Outputs are written under `experiments/FSS_IFAIterViz/<run_id>/ifa_iterations`.

Example:
python -m tools.visualize_ifa_iterations with \
    checkpoint_path='experiments/FSS_Training/dinov2_multilayer+svf+IFA+FDM_5shot_mIoU-78/best_model.pth' \
    num_samples=-1 use_ifa=True ifa_iters=3
"""

import json
import os
import random
from typing import Any, Dict, List, Sequence

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
from models.backbones.dino import DINO_linear  # noqa: E402
from utils.feature_viz import denormalize_to_numpy  # noqa: E402
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
                "method",
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


def _prepare_model(config: Dict[str, Any], device: torch.device) -> DINO_linear:
    if config.get("model_name", "DINO") != "DINO":
        raise NotImplementedError("IFA iteration visualization currently supports only the DINO backbone.")

    model = DINO_linear(
        version=config.get("dino_version", 2),
        method=config["method"],
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
    if effective_config.get("method") not in ("linear", "multilayer"):
        raise ValueError("IFA visualization is supported for `linear` and `multilayer` methods only.")
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

    indices = _select_indices(
        total=len(query_set),
        requested=effective_config.get("sample_indices", []),
        num_samples=effective_config.get("num_samples", -1),
        seed=effective_config.get("random_seed", 0),
    )
    max_iters = int(effective_config.get("max_iters_to_plot", -1))
    save_npz = bool(effective_config.get("save_mask_arrays", False))
    arrays_dir = None
    if save_npz:
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
                method=effective_config["method"],
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

        decoder_pred = decoder_logits.argmax(1)
        ifa_pred = logits_ifa.argmax(1)
        fused_pred = fused_logits.argmax(1)

        iter_logits = history.get("fused_iter_logits", []) or []
        if max_iters > 0:
            iter_logits = iter_logits[:max_iters]
        iter_preds = [logit.argmax(1) for logit in iter_logits]

        decoder_iou = _compute_iou(decoder_pred, mask)
        ifa_iou = _compute_iou(ifa_pred, mask)
        fused_iou = _compute_iou(fused_pred, mask)
        iter_ious = [_compute_iou(pred, mask) for pred in iter_preds]

        rgb = denormalize_to_numpy(image_t)
        panels = [
            ("RGB Input", rgb, None),
            ("Ground Truth", _mask_to_rgb(mask.cpu().numpy()), None),
            (f"Decoder (IoU {decoder_iou*100:.1f})", _mask_to_rgb(decoder_pred.squeeze(0).cpu().numpy()), None),
        ]

        for i, (pred, iou) in enumerate(zip(iter_preds, iter_ious), start=1):
            panels.append(
                (
                    f"IFA iter {i} (IoU {iou*100:.1f})",
                    _mask_to_rgb(pred.squeeze(0).cpu().numpy()),
                    None,
                )
            )

        panels.append(
            (
                f"Final Fusion (IoU {fused_iou*100:.1f})",
                _mask_to_rgb(fused_pred.squeeze(0).cpu().numpy()),
                None,
            )
        )

        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
        if len(panels) == 1:
            axes = [axes]
        for ax, (title, data, _) in zip(axes, panels):
            ax.imshow(data)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle(os.path.basename(image_path), fontsize=12)
        fig.tight_layout()

        fig_path = os.path.join(output_dir, f"sample_{idx:04d}.png")
        fig.savefig(fig_path, dpi=160, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        _run.add_artifact(fig_path, name=f"ifa_iterations/sample_{idx:04d}.png")

        if save_npz:
            npz_path = os.path.join(arrays_dir, f"sample_{idx:04d}.npz")
            iter_arrays = [pred.squeeze(0).cpu().numpy().astype(np.uint8) for pred in iter_preds]
            np.savez_compressed(
                npz_path,
                decoder=decoder_pred.squeeze(0).cpu().numpy().astype(np.uint8),
                fused=fused_pred.squeeze(0).cpu().numpy().astype(np.uint8),
                ifa=ifa_pred.squeeze(0).cpu().numpy().astype(np.uint8),
                iterations=np.stack(iter_arrays) if iter_arrays else np.zeros((0, *mask.shape[-2:]), dtype=np.uint8),
                mask=mask.cpu().numpy().astype(np.uint8),
            )
            _run.add_artifact(npz_path, name=f"ifa_iterations_npz/sample_{idx:04d}.npz")

        summaries.append(
            {
                "index": idx,
                "image_path": image_path,
                "decoder_iou": decoder_iou,
                "ifa_iou": ifa_iou,
                "fused_iou": fused_iou,
                "iter_ious": iter_ious,
            }
        )

    summary_path = os.path.join(output_dir, "ifa_iteration_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    _run.add_artifact(summary_path, name="ifa_iterations/metrics.json")

    print(f"[IFAIterViz] Saved {len(summaries)} visualizations to {output_dir}")
    return f"Completed IFA iteration visualization for {len(summaries)} samples."
