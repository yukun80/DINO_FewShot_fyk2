"""
This script is the main entry point for training a Few-Shot Semantic Segmentation model.
It has been refactored to use Sacred for robust experiment tracking and management.

--- Example Usage ---

2.  **Run Training with Sacred:**
    The syntax is `python3 train.py with <key>=<value>`.

    - **Linear Probing (10-shot, LR=0.01, Run 1):**
      python3 train.py with method=linear nb_shots=10 dino_version=2 dinov2_size=base run_id=1
      python3 train.py with method=linear nb_shots=10 dino_version=3 dinov3_size=base run_id=1

    - **multilayer**
      python3 train.py with method=multilayer nb_shots=9 dino_version=2 dinov2_size=base run_id=1
      python3 train.py with method=multilayer nb_shots=45 dino_version=2 dinov2_size=base run_id=1
      
      python3 train.py with method=multilayer nb_shots=20 dino_version=3 dinov3_size=base run_id=3

    - **SVF (10-shot, LR=0.0001, Run 1, requires a pre-trained linear decoder):**
      python3 train.py with method=svf nb_shots=10 lr=0.0001 run_id=1

"""

# --- Example Command ---
# python3 train.py with method=linear nb_shots=10 lr=0.01 run_id=1
# -----------------------

import datetime
import time
import torch
import yaml
import os
import random
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver

# --- Project-specific Imports ---
from utils.train_utils import get_lr_function, get_loss_fun, get_optimizer, get_dataset_loaders
from utils.precise_bn import compute_precise_bn_stats
from models.backbones.dino import DINO_linear
import warnings

warnings.filterwarnings("ignore")

# --- Sacred Experiment Setup ---
ex = Experiment("FSS_Training")
# All experiment artifacts will be stored in 'experiments/FSS_Training/{run_id}'
ex.observers.append(FileStorageObserver("experiments/FSS_Training"))


@ex.config
def cfg():
    """
    Defines the default configuration for the experiment using Sacred.
    These values can be easily overridden from the command line.
    """
    # Load base configuration from the YAML file
    with open("configs/disaster.yaml") as file:
        config = yaml.full_load(file)

    # --- Command-line accessible parameters ---
    model_name = "DINO"
    method = "linear"
    dataset = "disaster"
    nb_shots = 10
    lr = 0.01
    input_size = 512
    run_id = 1  # Used to distinguish between multiple runs of the same configuration
    # Backbone/version options (exposed for CLI override)
    dino_version = config.get("dino_version", 2)
    dinov2_size = config.get("dinov2_size", "base")
    dinov3_size = config.get("dinov3_size", "base")
    dinov3_weights_path = config.get("dinov3_weights_path", None)

    # Merge CLI-accessible parameters into the main config dictionary
    config.update(
        {
            "model_name": model_name,
            "method": method,
            "dataset": dataset,
            "number_of_shots": nb_shots,
            "lr": lr,
            "input_size": input_size,
            "run": run_id,
            "RNG_seed": run_id - 1,  # Seed depends on the run_id for reproducibility
        }
    )
    # Also surface backbone selection into the unified config dict
    config.update(
        {
            "dino_version": dino_version,
            "dinov2_size": dinov2_size,
            "dinov3_size": dinov3_size,
            "dinov3_weights_path": dinov3_weights_path,
        }
    )


def print_trainable_parameters(model):
    """Prints the number of trainable parameters in a model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


class ConfusionMatrix:
    """
    A robust confusion matrix for evaluating semantic segmentation.
    """

    def __init__(self, num_classes, exclude_classes):
        self.num_classes = num_classes
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        self.exclude_classes = exclude_classes

    def update(self, pred, target):
        pred = pred.cpu()
        target = target.cpu()
        n = self.num_classes
        k = (target >= 0) & (target < n)
        inds = n * target + pred
        inds = inds[k]
        self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global.item() * 100, (acc * 100).tolist(), (iu * 100).tolist()

    def __str__(self):
        acc_global, _, iu = self.compute()
        mIOU = sum(iu) / len(iu)
        reduced_iu = [iu[i] for i in range(self.num_classes) if i not in self.exclude_classes]
        mIOU_reduced = sum(reduced_iu) / len(reduced_iu)
        return f"mIoU: {mIOU:.2f} | mIoU (reduced): {mIOU_reduced:.2f} | " f"Global Accuracy: {acc_global:.2f}"


def evaluate(model, data_loader, device, confmat, max_eval):
    """Evaluates the model on the validation set."""
    model.eval()
    with torch.no_grad():
        for i, (image, target, _) in enumerate(data_loader):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = torch.nn.functional.interpolate(
                output, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
            confmat.update(output.argmax(1).flatten(), target.flatten())
            if i + 1 == max_eval:
                break
    return confmat


def train_one_epoch(
    model, loss_fun, optimizer, loader, lr_scheduler, _run, epoch, clip_grad_norm=None
):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    for t, (image, target, _) in enumerate(loader):
        image, target = image.to("cuda"), target.to("cuda")
        output = model(image)
        output = torch.nn.functional.interpolate(
            output, size=target.shape[-2:], mode="bilinear", align_corners=False
        )
        loss = loss_fun(output, target.long())

        optimizer.zero_grad()
        loss.backward()
        # Optional gradient clipping for stability
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        _run.log_scalar("metrics.batch_loss", loss.item(), step=epoch * len(loader) + t)

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch} | Average Loss: {avg_loss:.4f}")
    return avg_loss


def get_epochs_to_eval(config):
    """Determines which epochs should trigger an evaluation."""
    epochs = config["epochs"]
    eval_every = config["eval_every_k_epochs"]
    eval_epochs = {i * eval_every - 1 for i in range(1, epochs // eval_every + 1)}
    eval_epochs.add(0)
    eval_epochs.add(epochs - 1)
    if "eval_last_k_epochs" in config:
        eval_epochs.update(range(max(0, epochs - config["eval_last_k_epochs"]), epochs))
    return sorted(list(eval_epochs))


def setup_env(config):
    """Sets up the environment for reproducibility."""
    torch.backends.cudnn.benchmark = True
    seed = config.get("RNG_seed", 0)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Environment set up with seed: {seed}")


@ex.automain
def main(_run, config):
    """
    The main entry point for a training run, managed by Sacred.
    """
    setup_env(config)

    # --- Configuration & Setup ---
    save_dir = _run.observers[0].dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Starting Run {_run._id}: {config['method']} on {config['dataset']} ({config['number_of_shots']}-shot)")
    print(f"Artifacts will be saved to: {save_dir}")

    # --- Model Initialization ---
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
            # FDM flags
            fdm_enable_apm=config.get("fdm", {}).get("enable_apm", False) if isinstance(config.get("fdm", {}), dict) else config.get("fdm_enable_apm", False),
            fdm_apm_mode=config.get("fdm", {}).get("apm_mode", "S") if isinstance(config.get("fdm", {}), dict) else config.get("fdm_apm_mode", "S"),
            fdm_enable_acpa=config.get("fdm", {}).get("enable_acpa", False) if isinstance(config.get("fdm", {}), dict) else config.get("fdm_enable_acpa", False),
        )
    else:
        raise NotImplementedError(f"Model '{config['model_name']}' is not supported.")

    print_trainable_parameters(model)

    # --- Load Pre-trained Decoder if Applicable ---
    if config["method"] in ["svf", "lora"]:
        linear_weights_path = (
            config.get("linear_weights_path")
            or f"./exp/models_disaster/{config['model_name']}_linear_{config['dataset']}_{config['number_of_shots']}shot_run{config['run']}_best.pth"
        )

        print(f"Attempting to load pre-trained decoder from: {linear_weights_path}")
        if not os.path.exists(linear_weights_path):
            raise FileNotFoundError(
                f"Pre-trained decoder not found at {linear_weights_path}. Please run the 'linear' method first."
            )

        state_dict = torch.load(linear_weights_path, map_location="cpu")
        model.decoder.load_state_dict(
            {k.replace("decoder.", ""): v for k, v in state_dict.items() if k.startswith("decoder.")}
        )
        model.bn.load_state_dict({k.replace("bn.", ""): v for k, v in state_dict.items() if k.startswith("bn.")})
        print("Successfully loaded pre-trained decoder and batch norm layers.")

    model.to(device)

    # --- Data, Optimizer, and Scheduler ---
    train_loader, val_loader, train_set = get_dataset_loaders(config)
    optimizer = get_optimizer(model, config)
    loss_fun = get_loss_fun(config)
    total_iterations = len(train_loader) * config["epochs"]
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=total_iterations, power=config["poly_power"]
    )

    # --- Training & Evaluation Loop ---
    start_time = time.time()
    best_mIU = 0
    eval_on_epochs = get_epochs_to_eval(config)

    for epoch in range(config["epochs"]):
        if hasattr(train_set, "build_epoch"):
            train_set.build_epoch()

        avg_loss = train_one_epoch(
            model,
            loss_fun,
            optimizer,
            train_loader,
            lr_scheduler,
            _run,
            epoch,
            clip_grad_norm=config.get("clip_grad_norm", None),
        )
        _run.log_scalar("metrics.avg_epoch_loss", avg_loss, step=epoch)

        if epoch in eval_on_epochs:
            if config["bn_precise_stats"]:
                print("Calculating precise BN stats...")
                compute_precise_bn_stats(model, train_loader, config["bn_precise_num_samples"])

            confmat = ConfusionMatrix(config["num_classes"], config["exclude_classes"])
            evaluate(model, val_loader, device, confmat, config["max_eval"])

            print(f"--- Evaluation at Epoch {epoch} ---")
            print(confmat)

            acc_global, _, iu = confmat.compute()
            mIOU = sum(iu) / len(iu)

            _run.log_scalar("eval.mIoU", mIOU, step=epoch)
            _run.log_scalar("eval.global_accuracy", acc_global, step=epoch)

            if mIOU > best_mIU:
                best_mIU = mIOU
                print(f"New best mIoU: {best_mIU:.2f}. Saving model...")
                save_path = os.path.join(save_dir, "best_model.pth")
                torch.save(model.state_dict(), save_path)
                _run.log_scalar("eval.best_mIoU", best_mIU, step=epoch)

    # --- Finalization ---
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training finished in {total_time_str}.")
    print(f"Final Best mIoU: {best_mIU:.2f}")

    # Add the final best model as a named artifact for easy access
    final_model_path = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(final_model_path):
        _run.add_artifact(final_model_path, name="best_model.pth")

    return best_mIU
