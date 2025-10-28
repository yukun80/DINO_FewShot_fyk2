"""
t-SNE feature visualization over the query set.

This utility replays inference on the query split, collects embeddings from a
selected feature stage (e.g., backbone, decoder input, decoder/IFA logits),
and projects them to 2-D with t-SNE for qualitative analysis. Results are saved
under `experiments/FSS_TSNEViz/<run_id>/tsne_plots`.

Example:
python -m tools.visualize_tsne with \
    checkpoint_path='experiments/FSS_Training/dinov2_multilayer+svf+IFA+FDM_5shot_mIoU-78/best_model.pth' \
    feature_stage='decoder_input' sample_mode='pixel' pixels_per_image=512
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from sacred import Experiment  # noqa: E402
from sacred.observers import FileStorageObserver  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from datasets.disaster import DisasterDataset  # noqa: E402
from models.backbones.dino import DINO_linear  # noqa: E402
from utils.ifa import build_support_pack, run_ifa_inference  # noqa: E402

ex = Experiment("FSS_TSNEViz")
ex.observers.append(FileStorageObserver("experiments/FSS_TSNEViz"))

VALID_SAMPLE_MODES = {"image", "pixel"}
VALID_FEATURE_STAGES = {
    "backbone_raw",
    "backbone_fdm",
    "decoder_input",
    "decoder_logits",
    "ifa_logits",
    "decoder_fused",
}


@ex.config
def cfg():
    with open("configs/disaster.yaml") as file:
        config = yaml.full_load(file)

    checkpoint_path = None
    sample_indices: Sequence[int] = []
    num_samples = -1
    sample_mode = "image"  # {"image", "pixel"}
    pixels_per_image = 2048
    per_class_limit = -1  # limit samples per class when sample_mode="pixel"
    max_points = 5000  # safety cap on total collected embeddings (keeps t-SNE feasible)
    feature_stage = "decoder_input"
    normalize_features = True
    use_pca = True
    pca_dim = 50
    tsne_perplexity = 30.0
    tsne_learning_rate = 200.0
    tsne_metric = "euclidean"
    tsne_n_iter = 1500
    tsne_verbose = False
    random_seed = 0
    save_embeddings_npy = False
    figure_dpi = 300

    config.update(
        {
            "checkpoint_path": checkpoint_path,
            "sample_indices": list(sample_indices),
            "num_samples": num_samples,
            "sample_mode": sample_mode,
            "pixels_per_image": pixels_per_image,
            "per_class_limit": per_class_limit,
            "max_points": max_points,
            "feature_stage": feature_stage,
            "normalize_features": normalize_features,
            "use_pca": use_pca,
            "pca_dim": pca_dim,
            "tsne_perplexity": tsne_perplexity,
            "tsne_learning_rate": tsne_learning_rate,
            "tsne_metric": tsne_metric,
            "tsne_n_iter": tsne_n_iter,
            "tsne_verbose": tsne_verbose,
            "random_seed": random_seed,
            "save_embeddings_npy": save_embeddings_npy,
            "figure_dpi": figure_dpi,
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
                "val_split",
                "train_split",
                "fdm",
                "fdm_enable_apm",
                "fdm_apm_mode",
                "fdm_enable_acpa",
                "encoder_adapters",
                "use_ifa",
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
            print(f"[TSNEViz] Synced config from training run at {cfg_path}")
        except Exception as exc:
            print(f"[TSNEViz] Warning: failed to read training config at {cfg_path}: {exc}")
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


def _cap_indices_by_points(
    indices: List[int],
    config: Dict[str, Any],
    seed: int,
) -> Tuple[List[int], int]:
    mode = config.get("sample_mode", "image")
    max_points = max(1, int(config.get("max_points", 50000)))
    if mode == "image":
        total = len(indices)
        if total <= max_points:
            return indices, total
        rng = random.Random(seed)
        pool = list(indices)
        rng.shuffle(pool)
        trimmed = sorted(pool[:max_points])
        print(
            f"[TSNEViz] Capping image-level samples from {total} to {len(trimmed)} to respect max_points={max_points}."
        )
        return trimmed, len(trimmed)

    pixels_per_image = max(1, int(config.get("pixels_per_image", 2048)))
    per_class_limit = int(config.get("per_class_limit", -1))
    if per_class_limit > 0:
        per_img_budget = min(pixels_per_image, 2 * per_class_limit)
    else:
        per_img_budget = pixels_per_image
    total_points = per_img_budget * len(indices)
    if total_points <= max_points:
        return indices, total_points

    rng = random.Random(seed)
    pool = list(indices)
    rng.shuffle(pool)
    max_images = max(1, max_points // per_img_budget)
    trimmed = sorted(pool[:max_images])
    print(
        "[TSNEViz] Capping pixel samples: "
        f"{len(indices)} images -> {len(trimmed)} images to respect max_points={max_points}."
    )
    if len(trimmed) < len(indices):
        capped_pixels = max(1, max_points // max(1, len(trimmed)))
        if capped_pixels < pixels_per_image:
            print(
                f"[TSNEViz] Adjusting pixels_per_image from {pixels_per_image} to {capped_pixels} to stay within budget."
            )
            config["pixels_per_image"] = capped_pixels
            per_img_budget = capped_pixels
    effective_pixels = min(max_points, per_img_budget * len(trimmed))
    return trimmed, effective_pixels


def _prepare_model(config: Dict[str, Any], device: torch.device) -> DINO_linear:
    if config.get("model_name", "DINO") != "DINO":
        raise NotImplementedError("t-SNE visualization currently supports only the DINO backbone.")

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
    checkpoint = config["checkpoint_path"]
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint}'.")
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


@dataclass
class SampleMetadata:
    image_path: str
    dataset_index: int
    fg_ratio: float
    decoder_iou: float
    sample_mode: str
    point_label: int
    extra: Dict[str, Any]


def _compute_iou(pred_mask: torch.Tensor, target: torch.Tensor, class_id: int = 1) -> float:
    pred_class = (pred_mask == class_id)
    target_class = (target == class_id)
    inter = torch.logical_and(pred_class, target_class).sum().item()
    union = torch.logical_or(pred_class, target_class).sum().item()
    return 0.0 if union == 0 else inter / union


def _maybe_resize(feat: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    if feat.shape[-2:] == size_hw:
        return feat
    return F.interpolate(feat, size=size_hw, mode="bilinear", align_corners=False)


def _resolve_stage_tensor(
    stage_name: str,
    stages: Dict[str, torch.Tensor],
    history: List[torch.Tensor] | None,
) -> torch.Tensor:
    if stage_name in stages:
        tensor = stages[stage_name]
        if tensor is None:
            raise ValueError(
                f"Feature stage '{stage_name}' is unavailable for this run. "
                "Ensure the stage is compatible with the selected options (e.g., use_ifa=True for IFA logits)."
            )
        return tensor
    if stage_name.startswith("ifa_iter_"):
        if history is None or len(history) == 0:
            raise ValueError("Requested IFA iteration stage but no history is available.")
        try:
            iter_id = int(stage_name.split("_")[-1])
        except ValueError as exc:
            raise ValueError(f"Invalid IFA iteration stage '{stage_name}'.") from exc
        if iter_id <= 0 or iter_id > len(history):
            raise ValueError(
                f"Requested IFA iteration {iter_id} but history has {len(history)} iterations."
            )
        return history[iter_id - 1]
    raise ValueError(f"Unsupported feature_stage '{stage_name}'. Available: {sorted(stages.keys())}")


def _gather_embeddings(
    feature_map: torch.Tensor,
    mask: torch.Tensor,
    image_path: str,
    dataset_index: int,
    iou: float,
    mode: str,
    rng: random.Random,
    pixels_per_image: int,
    per_class_limit: int,
) -> Tuple[List[np.ndarray], List[int], List[SampleMetadata]]:
    feature_map = feature_map.squeeze(0).cpu()
    mask = mask.cpu()
    c, h, w = feature_map.shape
    embeddings: List[np.ndarray] = []
    labels: List[int] = []
    metadata: List[SampleMetadata] = []

    fg_ratio = float((mask == 1).float().mean().item())

    if mode == "image":
        vec = feature_map.view(c, -1).mean(dim=1).numpy()
        dominant = 1 if fg_ratio >= 0.5 else 0
        metadata.append(
            SampleMetadata(
                image_path=image_path,
                dataset_index=dataset_index,
                fg_ratio=fg_ratio,
                decoder_iou=iou,
                sample_mode="image",
                point_label=dominant,
                extra={"h": h, "w": w},
            )
        )
        embeddings.append(vec)
        labels.append(dominant)
        return embeddings, labels, metadata

    # Pixel sampling mode
    flat_feat = feature_map.permute(1, 2, 0).reshape(-1, c)
    flat_mask = mask.reshape(-1)
    total_pixels = flat_feat.shape[0]
    budget = min(pixels_per_image, total_pixels)
    per_class_limit = per_class_limit if per_class_limit > 0 else budget
    order = list(range(total_pixels))
    rng.shuffle(order)
    taken = 0
    class_counts = {0: 0, 1: 0}
    for idx in order:
        cls = int(flat_mask[idx].item())
        if cls not in class_counts:
            class_counts[cls] = 0
        if class_counts[cls] >= per_class_limit:
            continue
        vec = flat_feat[idx].numpy()
        embeddings.append(vec)
        labels.append(cls)
        metadata.append(
            SampleMetadata(
                image_path=image_path,
                dataset_index=dataset_index,
                fg_ratio=fg_ratio,
                decoder_iou=iou,
                sample_mode="pixel",
                point_label=cls,
                extra={"pixel_index": int(idx)},
            )
        )
        class_counts[cls] += 1
        taken += 1
        if taken >= budget:
            break
    return embeddings, labels, metadata


def _fit_tsne(
    embeddings: List[np.ndarray],
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    if len(embeddings) < 5:
        raise RuntimeError("Need at least 5 embeddings for a meaningful t-SNE projection.")
    matrix = np.stack(embeddings, axis=0)
    if config.get("normalize_features", True):
        matrix = StandardScaler().fit_transform(matrix)
    if config.get("use_pca", True):
        pca_dim = int(config.get("pca_dim", 50))
        if matrix.shape[1] > pca_dim:
            pca_dim = min(pca_dim, matrix.shape[1] - 1)
            matrix = PCA(n_components=pca_dim, random_state=config.get("random_seed", 0)).fit_transform(matrix)
    perplexity = float(config.get("tsne_perplexity", 30.0))
    perplexity = min(perplexity, max(5.0, len(matrix) - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=float(config.get("tsne_learning_rate", 200.0)),
        metric=config.get("tsne_metric", "euclidean"),
        max_iter=int(config.get("tsne_n_iter", 1500)),
        init="pca",
        random_state=config.get("random_seed", 0),
        verbose=1 if config.get("tsne_verbose", False) else 0,
    )
    coords = tsne.fit_transform(matrix)
    return matrix, coords


def _render_plot(
    coords: np.ndarray,
    labels: List[int],
    config: Dict[str, Any],
    output_path: str,
):
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab10", len(unique_labels))
    colors = [cmap(unique_labels.index(lbl)) for lbl in labels]
    plt.figure(figsize=(6, 6), dpi=config.get("figure_dpi", 300))
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, s=12, alpha=0.7, edgecolors="none")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=f"class {lbl}", markerfacecolor=cmap(i), markersize=6)
        for i, lbl in enumerate(unique_labels)
    ]
    plt.legend(handles=handles, loc="best")
    plt.title(
        f"t-SNE ({config['feature_stage']}) - mode={config['sample_mode']} - samples={len(labels)}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


@ex.automain
def main(_run, config: Dict[str, Any]):
    if config["checkpoint_path"] is None:
        raise ValueError("A `checkpoint_path` must be provided.")
    config = _sync_training_config(config)
    feature_stage = config["feature_stage"]
    if feature_stage not in VALID_FEATURE_STAGES and not feature_stage.startswith("ifa_iter_"):
        raise ValueError(
            f"Invalid feature_stage '{feature_stage}'. "
            f"Supported: {sorted(VALID_FEATURE_STAGES.union({'ifa_iter_<k>'}))}"
        )
    sample_mode = config.get("sample_mode", "image")
    if sample_mode not in VALID_SAMPLE_MODES:
        raise ValueError(f"sample_mode must be one of {VALID_SAMPLE_MODES}, got '{sample_mode}'.")

    requires_ifa = feature_stage in {"ifa_logits", "decoder_fused"} or feature_stage.startswith("ifa_iter_")
    supports_ifa = config["method"] in ("linear", "multilayer")
    if requires_ifa and not supports_ifa:
        raise ValueError(f"Feature stage '{feature_stage}' requires IFA, but method='{config['method']}' is unsupported.")

    seed = int(config.get("random_seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join(_run.observers[0].dir, "tsne_plots")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[TSNEViz] Saving outputs to: {output_dir}")

    model = _prepare_model(config, device)

    dataset = DisasterDataset(
        root=".",
        split_file=config["split_file"],
        mode=config.get("val_split", "query"),
    )
    indices = _select_indices(len(dataset), config.get("sample_indices", []), config.get("num_samples", -1), seed)
    if not indices:
        raise RuntimeError("No samples selected for visualization.")
    indices, est_points = _cap_indices_by_points(indices, config, seed)
    print(
        f"[TSNEViz] Using {len(indices)} query samples (~{est_points} embeddings) after safety capping."
    )

    support_pack: Dict[str, Any] = {}
    capture_history = feature_stage.startswith("ifa_iter_")
    need_ifa = supports_ifa and (bool(config.get("use_ifa", False)) or requires_ifa)
    if need_ifa:
        train_split = config.get("train_split", "support")
        support_dataset = DisasterDataset(
            root=".",
            split_file=config["split_file"],
            mode=train_split,
        )
        support_pack = build_support_pack(
            model=model,
            support_dataset=support_dataset,
            config=config,
            device=device,
            max_support=config.get("number_of_shots", 1),
        )

    rng = random.Random(seed)
    embeddings: List[np.ndarray] = []
    labels: List[int] = []
    metadata: List[SampleMetadata] = []

    for ds_idx in indices:
        image, mask, image_path = dataset[ds_idx]
        if image.numel() == 0:
            continue
        image = image.unsqueeze(0).to(device)
        mask = mask.to(torch.long)
        with torch.no_grad():
            logits, stage1, stage2, stage3 = model.forward_with_feature_maps(image)
        stage_tensors = {
            "backbone_raw": stage1,
            "backbone_fdm": stage2,
            "decoder_input": stage3,
        }
        mask_hw = mask.shape[-2:]
        logits = F.interpolate(logits, size=mask_hw, mode="bilinear", align_corners=False)
        stage_tensors["decoder_logits"] = logits
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu()
        iou = _compute_iou(pred, mask, class_id=1)

        if need_ifa:
            ifa_result = run_ifa_inference(
                model=model,
                image=image,
                method=config["method"],
                version=config.get("dino_version", 2),
                input_size=config.get("input_size", 512),
                ifa_cfg=config,
                support_pack=support_pack,
                out_size=mask_hw,
                use_fdm_on_feats=bool(config.get("ifa_use_fdm", True)),
                capture_history=capture_history,
            )
            if capture_history:
                ifa_logits, history_payload = ifa_result  # type: ignore[misc]
                fused_hist = history_payload.get("fused_iter_logits", [])
            else:
                ifa_logits = ifa_result  # type: ignore[assignment]
                fused_hist = []
            stage_tensors["ifa_logits"] = ifa_logits
            alpha = float(config.get("ifa_alpha", 0.3))
            stage_tensors["decoder_fused"] = (1.0 - alpha) * logits + alpha * ifa_logits
            history_tensors = [_maybe_resize(h, mask_hw).cpu() for h in fused_hist]
        else:
            stage_tensors["ifa_logits"] = None
            stage_tensors["decoder_fused"] = logits
            history_tensors = []

        stage_tensor = _resolve_stage_tensor(feature_stage, stage_tensors, history_tensors)
        stage_tensor = _maybe_resize(stage_tensor, mask_hw).cpu()
        emb, lbls, metas = _gather_embeddings(
            stage_tensor,
            mask,
            image_path,
            ds_idx,
            iou,
            sample_mode,
            rng,
            int(config.get("pixels_per_image", 2048)),
            int(config.get("per_class_limit", -1)),
        )
        embeddings.extend(emb)
        labels.extend(lbls)
        metadata.extend(metas)

    max_points = max(1, int(config.get("max_points", 5000)))
    if len(embeddings) > max_points:
        print(
            f"[TSNEViz] Downsampling collected embeddings from {len(embeddings)} to {max_points} before t-SNE."
        )
        idxs = rng.sample(range(len(embeddings)), max_points)
        embeddings = [embeddings[i] for i in idxs]
        labels = [labels[i] for i in idxs]
        metadata = [metadata[i] for i in idxs]

    feature_matrix, coords = _fit_tsne(embeddings, config)
    plot_path = os.path.join(output_dir, f"tsne_{feature_stage}_{sample_mode}.png")
    _render_plot(coords, labels, config, plot_path)
    _run.add_artifact(plot_path, name=os.path.relpath(plot_path, output_dir))

    meta_path = os.path.join(output_dir, f"tsne_{feature_stage}_{sample_mode}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump([meta.__dict__ for meta in metadata], f, indent=2)
    _run.add_artifact(meta_path, name=os.path.relpath(meta_path, output_dir))

    if config.get("save_embeddings_npy", False):
        npz_path = os.path.join(output_dir, f"tsne_{feature_stage}_{sample_mode}_embeddings.npz")
        np.savez(
            npz_path,
            embeddings=feature_matrix,
            coords=coords,
            labels=np.array(labels),
        )
        _run.add_artifact(npz_path, name=os.path.relpath(npz_path, output_dir))

    print(f"[TSNEViz] Saved plot to {plot_path}")
    return f"Completed t-SNE visualization over {len(embeddings)} embeddings."
