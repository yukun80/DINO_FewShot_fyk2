"""
This script is the main entry point for training a Few-Shot Semantic Segmentation model.
It has been refactored to use Sacred for robust experiment tracking and management.

--- Example Usage ---

2.  **Run Training with Sacred:**
    The syntax is `python3 train.py with <key>=<value>`.

    - **Multilayer decoder (default)**
      python3 train.py with dataset=disaster nb_shots=20 dino_version=3 dinov3_size=base run_id=1
      python3 train.py with dataset=disaster nb_shots=10 dino_version=2 dinov2_size=base run_id=1
      
      
    - **IFA**
    python3 train.py with run_id=1

"""

# --- Example Command ---
# python3 train.py with dataset=disaster nb_shots=20 run_id=1
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
from models.backbones.dino import DINOMultilayer
from utils.ifa import extract_encoder_features, run_ifa_training_logits
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

    # --- Command-line accessible parameters (now default to YAML values when present) ---
    model_name = config.get("model_name", "DINO")
    legacy_method = config.get("method", "multilayer")
    if legacy_method not in (None, "multilayer"):
        raise ValueError(f"Only 'multilayer' method is supported, but the config requested '{legacy_method}'.")
    method = "multilayer"
    dataset = config.get("dataset", "disaster")
    nb_shots = config.get("number_of_shots", 10)
    lr = config.get("lr", 0.01)
    input_size = config.get("input_size", 512)
    run_id = config.get("run", 1)  # Used to distinguish between multiple runs of the same configuration
    # Backbone/version options (exposed for CLI override)
    dino_version = config.get("dino_version", 2)
    dinov2_size = config.get("dinov2_size", "base")
    dinov3_size = config.get("dinov3_size", "base")
    dinov3_weights_path = config.get("dinov3_weights_path", None)

    # IFA training options (parallel enhancement branch)
    use_ifa_train = config.get("use_ifa_train", False)
    ifa_iters = config.get("ifa_iters", 3)
    ifa_refine = config.get("ifa_refine", True)
    ifa_alpha = config.get("ifa_alpha", 0.3)
    ifa_ms_weights = config.get("ifa_ms_weights", [0.1, 0.2, 0.3, 0.4])
    ifa_temp = config.get("ifa_temp", 10.0)
    ifa_fg_thresh = config.get("ifa_fg_thresh", 0.7)
    ifa_bg_thresh = config.get("ifa_bg_thresh", 0.6)
    ifa_use_fdm = config.get("ifa_use_fdm", True)
    ifa_loss_w_main = config.get("ifa_loss_w_main", 0.3)
    ifa_loss_w_aux = config.get("ifa_loss_w_aux", 0.0)
    # Memory-control for training
    ifa_train_support_k = config.get("ifa_train_support_k", None)  # Limit supports per-iter (<= number_of_shots)
    ifa_detach_support = config.get("ifa_detach_support", True)   # Extract support feats under no_grad

    # Encoder adapters decoupled from decoder type
    encoder_adapters = config.get("encoder_adapters", "none")  # {none|lora|svf}

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
            # IFA training options
            "use_ifa_train": use_ifa_train,
            "ifa_iters": ifa_iters,
            "ifa_refine": ifa_refine,
            "ifa_alpha": ifa_alpha,
            "ifa_ms_weights": ifa_ms_weights,
            "ifa_temp": ifa_temp,
            "ifa_fg_thresh": ifa_fg_thresh,
            "ifa_bg_thresh": ifa_bg_thresh,
            "ifa_use_fdm": ifa_use_fdm,
            "ifa_loss_w_main": ifa_loss_w_main,
            "ifa_loss_w_aux": ifa_loss_w_aux,
            "ifa_train_support_k": ifa_train_support_k,
            "ifa_detach_support": ifa_detach_support,
            # encoder adapters
            "encoder_adapters": encoder_adapters,
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


def save_periodic_checkpoint(model, optimizer, lr_scheduler, best_mIU, epoch_idx, save_dir, run):
    """
    Serializes a training snapshot so runs can be resumed or inspected mid-flight.
    """
    checkpoint = {
        "epoch": epoch_idx + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "lr_scheduler_state": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "best_mIoU": best_mIU,
        "timestamp": time.time(),
    }
    filename = f"checkpoint_epoch_{epoch_idx + 1:04d}.pth"
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    run.add_artifact(checkpoint_path, name=filename)
    print(f"[Checkpoint] Saved periodic checkpoint to {checkpoint_path}")


def train_one_epoch(
    model, loss_fun, optimizer, loader, lr_scheduler, _run, epoch, config, train_set, clip_grad_norm=None
):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    for t, (image, target, _) in enumerate(loader):
        image, target = image.to("cuda"), target.to("cuda")
        output = model(image)
        output = torch.nn.functional.interpolate(output, size=target.shape[-2:], mode="bilinear", align_corners=False)
        loss = loss_fun(output, target.long())

        # Optional IFA training branch (parallel to decoder CE)
        if config.get("use_ifa_train", False):
            k = int(config.get("number_of_shots", 1))
            k_eff = int(config.get("ifa_train_support_k", k) or k)
            k_eff = max(1, min(k, k_eff))
            support_imgs = []
            support_msks = []
            # Simple strategy: fixed first-K supports; can be randomized later
            for idx in range(min(k_eff, len(train_set))):
                img_s, msk_s, _ = train_set[idx]
                support_imgs.append(img_s.unsqueeze(0).to(image.device))
                support_msks.append(msk_s.to(image.device))

            # Per-scale features for supports: [S][K]
            feats_per_support = []
            detach_support_feats = bool(config.get("ifa_detach_support", True))
            for img in support_imgs:
                feats = extract_encoder_features(
                    model,
                    img,
                    version=config.get("dino_version", 2),
                    input_size=config.get("input_size", 512),
                    keep_encoder_grad=not detach_support_feats,
                )
                if detach_support_feats:
                    feats = [f.detach() for f in feats]
                feats_per_support.append(feats)
            num_scales = len(feats_per_support[0]) if len(feats_per_support) > 0 else 0
            if num_scales > 0:
                feats_s_ms = []
                for s in range(num_scales):
                    feats_s_ms.append([feats_per_support[k_i][s] for k_i in range(len(feats_per_support))])

                support_pack = {"feats_s_ms": feats_s_ms, "masks_s": support_msks}
                logits_ifa = run_ifa_training_logits(
                    model=model,
                    image=image,
                    version=config.get("dino_version", 2),
                    input_size=config.get("input_size", 512),
                    ifa_cfg=config,
                    support_pack=support_pack,
                    out_size=target.shape[-2:],
                    use_fdm_on_feats=bool(config.get("ifa_use_fdm", True)),
                )

                alpha = float(config.get("ifa_alpha", 0.3))
                logits_fused = (1.0 - alpha) * output + alpha * logits_ifa
                w_main = float(config.get("ifa_loss_w_main", 0.3))
                w_aux = float(config.get("ifa_loss_w_aux", 0.0))
                loss = loss + w_main * loss_fun(logits_fused, target.long())
                if w_aux > 0.0:
                    loss = loss + w_aux * loss_fun(logits_ifa, target.long())

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

    # Mark nested/optional keys as 'used' for Sacred usage tracking to avoid ConfigAddedError
    _ = (
        config.get("use_ifa_train", False),
        config.get("ifa_iters", 3),
        config.get("ifa_refine", True),
        config.get("ifa_alpha", 0.3),
        config.get("ifa_ms_weights", [0.1, 0.2, 0.3, 0.4]),
        config.get("ifa_temp", 10.0),
        config.get("ifa_fg_thresh", 0.7),
        config.get("ifa_bg_thresh", 0.6),
        config.get("ifa_use_fdm", True),
        config.get("ifa_loss_w_main", 0.3),
        config.get("ifa_loss_w_aux", 0.0),
        config.get("encoder_adapters", "none"),
    )
    # Also touch FDM nested dict via indexing so Sacred tracks nested keys
    try:
        _fdm = config["fdm"]
        _ = (_fdm["enable_apm"], _fdm["apm_mode"], _fdm["enable_acpa"])  # noqa: F841
    except Exception:
        pass

    # --- Configuration & Setup ---
    save_dir = _run.observers[0].dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Starting Run {_run._id}: {config['method']} on {config['dataset']} ({config['number_of_shots']}-shot)")
    print(f"Artifacts will be saved to: {save_dir}")

    # --- Model Initialization ---
    if config["model_name"] == "DINO":
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
            # Encoder adapters decoupled from decoder type
            encoder_adapters=config.get("encoder_adapters", "none"),
            # FDM flags
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
    else:
        raise NotImplementedError(f"Model '{config['model_name']}' is not supported.")

    print_trainable_parameters(model)

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
    checkpoint_every = config.get("checkpoint_every_epochs", None)
    if checkpoint_every is not None:
        try:
            checkpoint_every = int(checkpoint_every)
        except (TypeError, ValueError):
            checkpoint_every = None
    if checkpoint_every is not None and checkpoint_every <= 0:
        checkpoint_every = None

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
            config,
            train_set,
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

        if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
            save_periodic_checkpoint(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                best_mIU=best_mIU,
                epoch_idx=epoch,
                save_dir=save_dir,
                run=_run,
            )

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
