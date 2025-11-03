"""
Feature activation visualization for Few-Shot Segmentation models.

This script runs a trained checkpoint on selected query samples and produces
Grad-CAM overlays for three stages:
    1) Raw backbone activations (before APM/ACPA).
    2) Post-FDM activations (after APM+ACPA).
    3) Decoder fused tensor prior to the classification head.

Results are saved under `experiments/FSS_FeatureViz/<run_id>/feature_viz`.

python -m tools.visualize_features with checkpoint_path='experiments/FSS_Training/1/best_model.pth'
"""

import json
import os
import random
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch
import torch.nn.functional as F
import yaml
from sacred import Experiment
from sacred.observers import FileStorageObserver

from datasets.disaster import DisasterDataset
from models.backbones.dino import DINOMultilayer
from utils.feature_viz import compute_grad_cam, denormalize_to_numpy, overlay_heatmap

ex = Experiment("FSS_FeatureViz")
ex.observers.append(FileStorageObserver("experiments/FSS_FeatureViz"))


@ex.config
def cfg():
    with open("configs/disaster.yaml") as file:
        config = yaml.full_load(file)

    checkpoint_path = None
    num_samples = -1
    sample_indices: Sequence[int] = []
    target_class = 1
    random_seed = 0
    cam_blend_alpha = 0.7
    cam_cmap = "inferno"
    save_heatmap_npy = False

    config.update(
        {
            "checkpoint_path": checkpoint_path,
            "num_samples": num_samples,
            "sample_indices": list(sample_indices),
            "target_class": target_class,
            "random_seed": random_seed,
            "cam_blend_alpha": cam_blend_alpha,
            "cam_cmap": cam_cmap,
            "save_heatmap_npy": save_heatmap_npy,
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
                "number_of_shots",
                "dataset_dir",
                "model_path",
                "model_repo_path",
                "split_file",
                "val_split",
                "fdm",
                "encoder_adapters",
            ]
            for k in keys:
                if k in train_cfg:
                    effective[k] = train_cfg[k]
            print(f"[FeatureViz] Synced config from training run at {cfg_path}")
        except Exception as exc:
            print(f"[FeatureViz] Warning: failed to read training config at {cfg_path}: {exc}")
    effective["method"] = "multilayer"
    return effective


def _select_indices(total: int, requested: Sequence[int], num_samples: int, seed: int) -> List[int]:
    if requested:
        valid = sorted({idx for idx in requested if 0 <= idx < total})
        if not valid:
            raise ValueError("Provided sample_indices are outside dataset range.")
        return valid
    if num_samples < 0 or num_samples >= total:
        return list(range(total))
    count = min(max(1, num_samples), total)
    rng = random.Random(seed)
    all_idx = list(range(total))
    rng.shuffle(all_idx)
    return sorted(all_idx[:count])


def _compute_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, class_id: int) -> float:
    pred_class = pred_mask == class_id
    gt_class = gt_mask == class_id
    inter = torch.logical_and(pred_class, gt_class).sum().item()
    union = torch.logical_or(pred_class, gt_class).sum().item()
    if union == 0:
        return 0.0
    return inter / union


def _extract_model_state(state: Any) -> Dict[str, torch.Tensor]:
    if isinstance(state, dict):
        for key in ("model_state", "model", "state_dict"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    if isinstance(state, dict):
        return state
    raise ValueError("Checkpoint format not recognized; expected dict with model weights.")


def _infer_adapter_mode(state_dict: Dict[str, torch.Tensor]) -> str:
    keys = state_dict.keys()
    if any("lora_" in k for k in keys):
        return "lora"
    if any("vector_S" in k for k in keys):
        return "svf"
    return "none"


def _prepare_model(config: Dict[str, Any], device: torch.device) -> DINOMultilayer:
    if config.get("model_name", "DINO") != "DINO":
        raise NotImplementedError("Feature visualization currently supports only the DINO backbone.")

    raw_ckpt = torch.load(config["checkpoint_path"], map_location="cpu")
    state_dict = _extract_model_state(raw_ckpt)
    inferred_adapter = _infer_adapter_mode(state_dict)
    cfg_adapter = (config.get("encoder_adapters", "none") or "none").lower()
    if cfg_adapter not in ("none", "lora", "svf"):
        cfg_adapter = "none"
    if cfg_adapter != inferred_adapter:
        print(
            f"[FeatureViz] Adapter mismatch: config='{cfg_adapter}', checkpoint='{inferred_adapter}'. "
            f"Using '{inferred_adapter}' to match checkpoint structure."
        )
        cfg_adapter = inferred_adapter

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
        encoder_adapters=cfg_adapter,
        fdm_enable_apm=(
            config.get("fdm", {}).get("enable_apm", False)
            if isinstance(config.get("fdm", {}), dict)
            else config.get("fdm_enable_apm", False)
        ),
        fdm_apm_mode=(
            config.get("fdm", {}).get("apm_mode", "S")
            if isinstance(config.get("fdm", {}), dict)
            else config.get("fdm_apm_mode", "S")
        ),
        fdm_enable_acpa=(
            config.get("fdm", {}).get("enable_acpa", False)
            if isinstance(config.get("fdm", {}), dict)
            else config.get("fdm_enable_acpa", False)
        ),
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _render_panel(
    rgb_image: torch.Tensor,
    cams: List[Any],
    config: Dict[str, Any],
    titles: Sequence[str],
    save_path: str,
) -> None:
    rgb_np = denormalize_to_numpy(rgb_image.cpu())
    overlays = [
        overlay_heatmap(rgb_np, cam, cmap_name=config["cam_cmap"], alpha=config["cam_blend_alpha"]) for cam in cams
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow((rgb_np * 255).astype("uint8"))
    axes[0].set_title("Query Image")
    axes[0].axis("off")
    for ax, heatmap_img, title in zip(axes[1:], overlays, titles):
        ax.imshow(heatmap_img)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


@ex.automain
def main(_run, config: Dict[str, Any]):
    if not config.get("checkpoint_path"):
        raise ValueError(
            "`checkpoint_path` is required. Example: with checkpoint_path='experiments/.../best_model.pth'"
        )

    torch.set_grad_enabled(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FeatureViz] Using device: {device}")

    effective_cfg = _sync_training_config(config)

    model = _prepare_model(effective_cfg, device)
    dataset = DisasterDataset(
        root=".",
        split_file=effective_cfg["split_file"],
        mode=effective_cfg.get("val_split", "query"),
    )

    indices = _select_indices(
        total=len(dataset),
        requested=effective_cfg.get("sample_indices", []),
        num_samples=int(effective_cfg.get("num_samples", 1)),
        seed=int(effective_cfg.get("random_seed", 0)),
    )
    print(f"[FeatureViz] Visualizing samples: {indices}")

    output_root = os.path.join(_run.observers[0].dir, "feature_viz")
    os.makedirs(output_root, exist_ok=True)

    target_class = int(effective_cfg.get("target_class", 1))
    class_names = {1: "Landslide"}
    titles = [
        f"Stage1 CAM (Backbone)",
        f"Stage2 CAM (APM+ACPA)",
        f"Stage3 CAM (Decoder Input)",
    ]

    for idx in indices:
        image_tensor, mask_tensor, image_path = dataset[idx]
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        img = image_tensor.unsqueeze(0).to(device)
        mask = mask_tensor.to(device)

        logits, stage1, stage2, stage3 = model.forward_with_feature_maps(img, keep_encoder_grad=True)
        for tensor in (stage1, stage2, stage3):
            tensor.retain_grad()

        logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        score = logits[:, target_class, :, :].mean()
        model.zero_grad(set_to_none=True)
        score.backward()

        cams = [
            compute_grad_cam(stage1, stage1.grad, target_size=mask.shape[-2:]),
            compute_grad_cam(stage2, stage2.grad, target_size=mask.shape[-2:]),
            compute_grad_cam(stage3, stage3.grad, target_size=mask.shape[-2:]),
        ]

        pred_mask = logits.argmax(dim=1).squeeze(0)
        miou = _compute_iou(pred_mask, mask, target_class)
        _run.log_scalar("sample.iou_target", miou)

        panel_path = os.path.join(output_root, f"{idx:04d}_{image_name}_cams.png")
        _render_panel(
            rgb_image=image_tensor,
            cams=cams,
            config=effective_cfg,
            titles=[
                f"{titles[0]}",
                f"{titles[1]}",
                f"{titles[2]}",
            ],
            save_path=panel_path,
        )
        _run.add_artifact(panel_path, name=f"feature_viz/{os.path.basename(panel_path)}")

        if effective_cfg.get("save_heatmap_npy", False):
            npy_path = os.path.join(output_root, f"{idx:04d}_{image_name}_cams.npy")
            torch.save(
                {
                    "cams": cams,
                    "iou": miou,
                    "image_path": image_path,
                    "target_class": target_class,
                    "class_name": class_names.get(target_class, f"class_{target_class}"),
                },
                npy_path,
            )
            _run.add_artifact(npy_path, name=f"feature_viz/{os.path.basename(npy_path)}")

        print(f"[FeatureViz] Saved CAMs for sample {idx} at {panel_path}")
        model.zero_grad(set_to_none=True)

    print("[FeatureViz] Completed visualization run.")
