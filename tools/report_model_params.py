"""
Utility script to report total and learnable parameter counts for the current
few-shot segmentation framework. Designed for readability and portability.

Example usage:
    # Default config (DINOv3 large, pretrained weights under ./pretrain)
    python3 -m tools.report_model_params --override model_repo_path=. model_path=pretrain

    # Small configuration that matches the checkpoint stored at
    # experiments/FSS_Training/dinov3_small/best_model.pth
python3 -m tools.report_model_params \
    --override model_repo_path=. \
                model_path=pretrain \
                dino_version=3 \
                dinov3_size=small \
                dinov3_weights_path=pretrain/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
                
python3 -m tools.report_model_params \
    --override model_repo_path=. \
                model_path=pretrain \
                dino_version=3 \
                dinov3_size=base \
                dinov3_weights_path=pretrain/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth

Training the matching model:
python3 train.py with run_id=1 dino_version=3 dinov3_size=small model_repo_path=. model_path=pretrain
    
The trained weights are saved by Sacred under experiments/FSS_Training/<run_id>,
e.g. experiments/FSS_Training/dinov3_small/best_model.pth for the small setup.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.backbones.dino import DINOMultilayer

"""
python3 -m tools.report_model_params --override model_repo_path=. model_path=../pretrain dinov3_weights_path=../pretrain/dinov3_vits16_pretrain_lvd1689m-08c60483.pth dinov3_size=small
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report total parameters (Params) and learnable parameters (L-Params) for the model."
    )
    parser.add_argument(
        "--config",
        default="configs/disaster.yaml",
        help="Path to the base YAML configuration file.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=None,
        metavar="KEY=VALUE",
        help="Override configuration keys, e.g. encoder_adapters=lora fdm.enable_acpa=false",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device to place the model on before reporting. Falls back to CPU if CUDA is unavailable.",
    )
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="Skip dummy forward pass. Parameters from lazily initialised modules may be missing.",
    )
    parser.add_argument(
        "--dump-json",
        default=None,
        help="Optional path to save the parameter statistics as JSON.",
    )
    return parser.parse_args()


def safe_yaml_load(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    updated = deepcopy(config)
    if not overrides:
        return updated
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' is invalid; expected KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Override '{item}' has an empty key.")
        parsed_value = safe_yaml_load(value.strip())
        target = updated
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = parsed_value
    return updated


def load_config(path: str, overrides: Iterable[str] | None) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        base_cfg = yaml.full_load(handle) or {}
    cfg = apply_overrides(base_cfg, overrides)
    return cfg


def build_model(cfg: Dict[str, Any]) -> DINOMultilayer:
    legacy_method = cfg.get("method", "multilayer")
    if legacy_method not in (None, "multilayer"):
        raise ValueError(f"Only 'multilayer' method is supported, but the config requested '{legacy_method}'.")
    model_kwargs = dict(
        version=cfg.get("dino_version", 2),
        num_classes=cfg.get("num_classes", 2),
        input_size=cfg.get("input_size", 512),
        model_repo_path=cfg.get("model_repo_path"),
        model_path=cfg.get("model_path"),
        dinov2_size=cfg.get("dinov2_size", "base"),
        dinov3_size=cfg.get("dinov3_size", "base"),
        dinov3_weights_path=cfg.get("dinov3_weights_path"),
        dinov3_rope_dtype=cfg.get("dinov3_rope_dtype", "bf16"),
        encoder_adapters=cfg.get("encoder_adapters", "none"),
        fdm_enable_apm=cfg.get("fdm", {}).get("enable_apm", cfg.get("fdm_enable_apm", False)),
        fdm_apm_mode=cfg.get("fdm", {}).get("apm_mode", cfg.get("fdm_apm_mode", "S")),
        fdm_enable_acpa=cfg.get("fdm", {}).get("enable_acpa", cfg.get("fdm_enable_acpa", False)),
    )
    missing = [k for k in ("model_repo_path", "model_path") if not model_kwargs.get(k)]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Configuration must provide {joined} for model initialisation.")
    model = DINOMultilayer(**model_kwargs)
    return model


def run_dummy_forward(model: DINOMultilayer, device: torch.device, cfg: Dict[str, Any]) -> None:
    input_size = cfg.get("input_size", 512)
    if isinstance(input_size, (list, tuple)):
        h, w = input_size
    else:
        h = w = int(input_size)
    dummy = torch.zeros((1, 3, h, w), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        _ = model(dummy)


def _group_key(param_name: str) -> str:
    if param_name.startswith("encoder."):
        return "encoder"
    if param_name.startswith("decoder."):
        return "decoder"
    if param_name.startswith("apm."):
        return "apm"
    if param_name.startswith("acpa."):
        return "acpa"
    if param_name.startswith("bn.") or ".bn." in param_name:
        return "batch_norm"
    return "others"


def collect_param_stats(model: torch.nn.Module) -> Tuple[Dict[str, Any], Dict[str, Dict[str, int]]]:
    totals = {"total_params": 0, "learnable_params": 0}
    groups: Dict[str, Dict[str, int]] = {}
    for name, param in model.named_parameters():
        numel = param.numel()
        learnable = numel if param.requires_grad else 0
        totals["total_params"] += numel
        totals["learnable_params"] += learnable
        group = _group_key(name)
        if group not in groups:
            groups[group] = {"total": 0, "learnable": 0}
        groups[group]["total"] += numel
        groups[group]["learnable"] += learnable
    return totals, groups


def to_millions(value: int) -> float:
    return round(value / 1e6, 6)


def format_report(
    cfg_path: str, overrides: Iterable[str], totals: Dict[str, Any], groups: Dict[str, Dict[str, int]]
) -> str:
    lines = []
    lines.append("===== Model Parameter Report =====")
    overrides_str = " ".join(overrides) if overrides else "(none)"
    lines.append(f"Config: {cfg_path}")
    lines.append(f"Overrides: {overrides_str}")
    total_m = to_millions(totals["total_params"])
    learnable_m = to_millions(totals["learnable_params"])
    ratio = (totals["learnable_params"] / totals["total_params"] * 100) if totals["total_params"] else 0.0
    lines.append("")
    lines.append(f"Params (M): {total_m}")
    lines.append(f"L-Params (M): {learnable_m}")
    lines.append(f"Trainable Ratio: {ratio:.4f}%")
    lines.append("")
    lines.append("Breakdown (M):")
    for key in sorted(groups.keys()):
        total = to_millions(groups[key]["total"])
        learnable = to_millions(groups[key]["learnable"])
        lines.append(f"- {key:12s} total {total:10.6f} | learnable {learnable:10.6f}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.override)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("CUDA requested but not available; using CPU instead.")

    model = build_model(cfg)
    model.to(device)

    if not args.skip_forward:
        run_dummy_forward(model, device, cfg)

    totals, groups = collect_param_stats(model)
    report = format_report(args.config, args.override or [], totals, groups)
    print(report)

    if args.dump_json:
        payload = {
            "config": args.config,
            "overrides": args.override or [],
            "totals": totals,
            "groups": groups,
        }
        with open(args.dump_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nSaved JSON report to {args.dump_json}")


if __name__ == "__main__":
    main()
