"""
This script is for performing inference and generating visual predictions.
It has been refactored to use Sacred for experiment tracking.

--- Sacred Integration Details ---

Purpose:
    - To run inference with a trained model and save the visual results in a
      reproducible and organized manner.
    - Each prediction run is saved to `experiments/FSS_Prediction`.
    - The generated segmentation masks are saved as Sacred artifacts for easy access.

--- Example Usage ---

`python3 predict.py with checkpoint_path='path/to/your/model.pth' nb_shots=10`

# IFA
python3 predict.py with checkpoint_path='experiments/FSS_Training/dinov2_multilayer+fdm_5shot/best_model.pth' nb_shots=45 use_ifa=True ifa_iters=3 ifa_refine=True

python3 predict.py with checkpoint_path='experiments/FSS_Training/dinov3_large_1026/best_model.pth'

- The `model_path` is required.
- The output directory is managed automatically by Sacred.
- Other parameters should match the model's training configuration.
"""

# --- Example Command ---
# python3 predict.py with checkpoint_path='experiments/FSS_Training/1/best_model.pth' nb_shots=10
# -----------------------

import os
import json
import yaml
import torch
import warnings
import numpy as np
from typing import Dict, Any
from PIL import Image
from sacred import Experiment
from sacred.observers import FileStorageObserver

# --- Project-specific Imports ---
from utils.train_utils import get_dataset_loaders
from models.backbones.dino import DINOMultilayer
from utils.ifa import build_support_pack, run_ifa_inference
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# --- Sacred Experiment Setup ---
ex = Experiment("FSS_Prediction")
ex.observers.append(FileStorageObserver("experiments/FSS_Prediction"))


@ex.config
def cfg():
    """
    Defines the default configuration for the prediction experiment.
    """
    # Load base configuration from the YAML file
    with open("configs/disaster.yaml") as file:
        config = yaml.full_load(file)

    # --- Command-line accessible parameters ---
    checkpoint_path = None  # REQUIRED: Path to the trained model .pth file
    model_name = "DINO"
    legacy_method = config.get("method", "multilayer")
    if legacy_method not in (None, "multilayer"):
        raise ValueError(f"Only 'multilayer' method is supported, but the config requested '{legacy_method}'.")
    method = "multilayer"
    dataset = "disaster"
    nb_shots = 10
    input_size = 512
    # Backbone/version options (exposed for CLI override)
    dino_version = config.get("dino_version", 2)
    dinov2_size = config.get("dinov2_size", "base")
    dinov3_size = config.get("dinov3_size", "base")
    dinov3_weights_path = config.get("dinov3_weights_path", None)
    dinov3_rope_dtype = config.get("dinov3_rope_dtype", "bf16")

    # IFA options (explicit keys so Sacred recognizes CLI overrides)
    use_ifa = False
    ifa_iters = 3
    ifa_refine = True
    ifa_alpha = 0.3
    ifa_ms_weights = [0.1, 0.2, 0.3, 0.4]
    ifa_temp = 10.0
    ifa_fg_thresh = 0.7
    ifa_bg_thresh = 0.6
    ifa_use_fdm = True  # Apply FDM to features before IFA (parity with training)

    # Merge CLI-accessible parameters into the main config dictionary
    config.update(
        {
            "checkpoint_path": checkpoint_path,
            "model_name": model_name,
            "method": method,
            "dataset": dataset,
            "number_of_shots": nb_shots,
            "input_size": input_size,
            "dino_version": dino_version,
            "dinov2_size": dinov2_size,
            "dinov3_size": dinov3_size,
            "dinov3_weights_path": dinov3_weights_path,
            "dinov3_rope_dtype": dinov3_rope_dtype,
            # IFA prediction options (inference-only)
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


# --- Color Palette Definition ---
# Class 0 (Background): Black (0, 0, 0)
# Class 1 (Landslide): Red (255, 0, 0)
COLOR_PALETTE = np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8)


@torch.no_grad()
def predict_and_visualize(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str,
    _run: Experiment.run,
    base_config: Dict[str, Any],
    support_pack: Dict[str, Any],
) -> None:
    """
    The main prediction and visualization function.

    Args:
        model: The model to use for inference.
        data_loader: The DataLoader for the validation set.
        device: The device to run inference on.
        output_dir: The directory where predicted images will be saved.
        _run: The Sacred run object for adding artifacts.
    """
    model.eval()
    # Store visual PNGs in a dedicated subfolder to separate from Sacred JSONs
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "predictions")
    os.makedirs(images_dir, exist_ok=True)
    print(f"Starting prediction... Visualizations will be saved to '{images_dir}'")

    for i, (image, target, image_path) in enumerate(data_loader):
        if not isinstance(image_path, str):
            image_path = image_path[0]
        if not image_path:
            print(f"Skipping sample at index {i} due to a loading error.")
            continue

        image = image.to(device)
        output = model(image)
        output = torch.nn.functional.interpolate(output, size=target.shape[-2:], mode="bilinear", align_corners=False)

        # Optional IFA enhancement (inference-time)
        if base_config.get("use_ifa", False):
            logits_ifa = run_ifa_inference(
                model=model,
                image=image,
                version=base_config.get("dino_version", 2),
                input_size=base_config.get("input_size", 512),
                ifa_cfg=base_config,
                support_pack=support_pack,
                out_size=target.shape[-2:],
                use_fdm_on_feats=bool(base_config.get("ifa_use_fdm", True)),
            )
            alpha = float(base_config.get("ifa_alpha", 0.3))
            output = (1.0 - alpha) * output + alpha * logits_ifa
        prediction = torch.argmax(output, 1).squeeze(0).cpu().numpy().astype(np.uint8)

        # --- Visualization ---
        pred_image = Image.fromarray(prediction, mode="P")
        pred_image.putpalette(COLOR_PALETTE.flatten())
        pred_image = pred_image.convert("RGB")

        # --- Save the Result and Add as Artifact ---
        base_name = os.path.basename(image_path)
        file_name, _ = os.path.splitext(base_name)
        output_path = os.path.join(images_dir, f"{file_name}_pred.png")
        pred_image.save(output_path)

        # Add the generated image as a named artifact under the same subfolder
        _run.add_artifact(output_path, name=f"predictions/{file_name}_pred.png")

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(data_loader)} images...")

    print("\nPrediction and visualization complete.")


@ex.automain
def main(_run, config: Dict[str, Any]):
    """
    Main entry point for prediction, managed by Sacred.
    """
    if config["checkpoint_path"] is None:
        raise ValueError("A `checkpoint_path` must be provided. Ex: `with checkpoint_path='path/to/model.pth'`")

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = _run.observers[0].dir  # Use the Sacred run directory for output
    print(f"Using device: {device}")
    print(f"Predicting with model from: {config['checkpoint_path']}")

    # Sync runtime config with training run's config.json (if available)
    effective_config: Dict[str, Any] = dict(config)
    run_dir = os.path.dirname(effective_config["checkpoint_path"]) if effective_config["checkpoint_path"] else None
    train_cfg_path = os.path.join(run_dir, "config.json") if run_dir else None
    if train_cfg_path and os.path.exists(train_cfg_path):
        try:
            with open(train_cfg_path, "r") as f:
                train_meta = json.load(f)
            train_cfg = train_meta.get("config", train_meta)
            keys_to_copy = [
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
                "number_of_shots",
                "dataset_dir",
                "model_path",
                "model_repo_path",
                "split_file",
                "val_input_size",
                "val_label_size",
                "max_eval",
                "fdm",
                "fdm_enable_apm",
                "fdm_apm_mode",
                "fdm_enable_acpa",
                "encoder_adapters",
            ]
            for k in keys_to_copy:
                if k in train_cfg:
                    effective_config[k] = train_cfg[k]
            print(f"Loaded training config from: {train_cfg_path}")
            print(
                f"Using version={effective_config.get('dino_version', 2)}, decoder='multilayer', "
                f"dinov2_size='{effective_config.get('dinov2_size', 'base')}', dinov3_size='{effective_config.get('dinov3_size', 'base')}', "
                f"input_size={effective_config['input_size']}"
            )
            effective_config["method"] = "multilayer"
        except Exception as e:
            print(f"Warning: failed to read training config at {train_cfg_path}: {e}")

    effective_config["method"] = "multilayer"

    # --- Model Initialization and Loading ---
    print("Initializing model...")
    if effective_config["model_name"] == "DINO":
        model = DINOMultilayer(
            version=effective_config.get("dino_version", 2),
            num_classes=effective_config["num_classes"],
            input_size=effective_config["input_size"],
            model_repo_path=effective_config["model_repo_path"],
            model_path=effective_config["model_path"],
            dinov2_size=effective_config.get("dinov2_size", "base"),
            dinov3_size=effective_config.get("dinov3_size", "base"),
            dinov3_weights_path=effective_config.get("dinov3_weights_path", None),
            dinov3_rope_dtype=effective_config.get("dinov3_rope_dtype", "bf16"),
            encoder_adapters=effective_config.get("encoder_adapters", "none"),
            # FDM flags
            fdm_enable_apm=(
                effective_config.get("fdm", {}).get("enable_apm", False)
                if isinstance(effective_config.get("fdm", {}), dict)
                else effective_config.get("fdm_enable_apm", False)
            ),
            fdm_apm_mode=(
                effective_config.get("fdm", {}).get("apm_mode", effective_config.get("fdm_apm_mode", "S"))
                if isinstance(effective_config.get("fdm", {}), dict)
                else effective_config.get("fdm_apm_mode", "S")
            ),
            fdm_enable_acpa=(
                effective_config.get("fdm", {}).get("enable_acpa", False)
                if isinstance(effective_config.get("fdm", {}), dict)
                else effective_config.get("fdm_enable_acpa", False)
            ),
        )
    else:
        raise NotImplementedError(f"Model '{config['model_name']}' is not supported.")

    if not os.path.exists(effective_config["checkpoint_path"]):
        raise FileNotFoundError(f"Model checkpoint not found at: {effective_config['checkpoint_path']}")

    model.load_state_dict(torch.load(effective_config["checkpoint_path"], map_location=device))
    model.to(device)

    # --- Data Loading ---
    print("Loading datasets (support/query)...")
    train_loader, val_loader, train_set = get_dataset_loaders(effective_config)

    # Prepare support features for IFA if enabled
    support_pack: Dict[str, Any] = {}
    if effective_config.get("use_ifa", False):
        support_pack = build_support_pack(
            model=model,
            support_dataset=train_set,
            config=effective_config,
            device=device,
            max_support=effective_config.get("number_of_shots", 1),
        )

    # --- Run Prediction and Visualization ---
    predict_and_visualize(model, val_loader, device, output_dir, _run, effective_config, support_pack)

    print(f"Prediction complete. Visualizations saved to: {output_dir}")
    return f"Completed prediction run {_run._id}."
