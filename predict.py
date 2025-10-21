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

- The `model_path` is required.
- The output directory is managed automatically by Sacred.
- Other parameters should match the model's training configuration.
"""

# --- Example Command ---
# python3 predict.py with checkpoint_path='experiments/FSS_Training/1/best_model.pth' nb_shots=10
# -----------------------

import os
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
from models.backbones.dino import DINO_linear

warnings.filterwarnings("ignore")

# --- Sacred Experiment Setup ---
ex = Experiment("FSS_Prediction")
ex.observers.append(FileStorageObserver('experiments/FSS_Prediction'))


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
    method = "linear"
    dataset = "disaster"
    nb_shots = 10
    input_size = 512
    # Backbone/version options (exposed for CLI override)
    dino_version = config.get("dino_version", 2)
    dinov2_size = config.get("dinov2_size", "base")
    dinov3_size = config.get("dinov3_size", "base")
    dinov3_weights_path = config.get("dinov3_weights_path", None)
    dinov3_rope_dtype = config.get("dinov3_rope_dtype", "bf16")

    # Merge CLI-accessible parameters into the main config dictionary
    config.update({
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
    })


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
    _run: Experiment.run
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
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting prediction... Visualizations will be saved to '{output_dir}'")

    for i, (image, target, image_path) in enumerate(data_loader):
        if not isinstance(image_path, str):
            image_path = image_path[0]
        if not image_path:
            print(f"Skipping sample at index {i} due to a loading error.")
            continue

        image = image.to(device)
        output = model(image)
        output = torch.nn.functional.interpolate(output, size=target.shape[-2:], mode='bilinear', align_corners=False)
        prediction = torch.argmax(output, 1).squeeze(0).cpu().numpy().astype(np.uint8)

        # --- Visualization ---
        pred_image = Image.fromarray(prediction, mode='P')
        pred_image.putpalette(COLOR_PALETTE.flatten())
        pred_image = pred_image.convert('RGB')

        # --- Save the Result and Add as Artifact ---
        base_name = os.path.basename(image_path)
        file_name, _ = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{file_name}_pred.png")
        pred_image.save(output_path)

        # Add the generated image as a named artifact to the Sacred run
        _run.add_artifact(output_path, name=f"prediction_{file_name}.png")

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = _run.observers[0].dir  # Use the Sacred run directory for output
    print(f"Using device: {device}")
    print(f"Predicting with model from: {config['checkpoint_path']}")

    # --- Model Initialization and Loading ---
    print("Initializing model...")
    if config["model_name"] == "DINO":
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
        )
    else:
        raise NotImplementedError(f"Model '{config['model_name']}' is not supported.")

    if not os.path.exists(config["checkpoint_path"]):
        raise FileNotFoundError(f"Model checkpoint not found at: {config['checkpoint_path']}")

    model.load_state_dict(torch.load(config["checkpoint_path"], map_location=device))
    model.to(device)

    # --- Data Loading ---
    print("Loading validation (query) dataset...")
    _, val_loader, _ = get_dataset_loaders(config)

    # --- Run Prediction and Visualization ---
    predict_and_visualize(model, val_loader, device, output_dir, _run)

    print(f"Prediction complete. Visualizations saved to: {output_dir}")
    return f"Completed prediction run {_run._id}."
