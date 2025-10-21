import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from functools import partial
from typing import List, Sequence, Union, Optional
import glob
from peft import LoraConfig, get_peft_model
from models.svf import *
from models.decoders.dpt import DPTDecoder
from modules.module_FDM.freq_masker import MaskModule
from modules.module_FDM.phase_attn import PhaseAttention

def create_backbone_dinov2(method, model_repo_path, model_path, dinov2_size="base") : 
    sys.path.insert(0,os.path.join(model_repo_path, "dinov2"))
    from dinov2.models.vision_transformer import vit_base, vit_large, vit_small
    
    if dinov2_size == 'base':
        dino_backbone = vit_base(patch_size = 14, img_size = 524, init_values=1.0, block_chunks=0, num_register_tokens=0)
        dino_backbone.load_state_dict(torch.load(os.path.join(model_path,"dinov2_vitb14_pretrain.pth")))
    elif dinov2_size == 'small':
        dino_backbone = vit_small(patch_size = 14, img_size = 524, init_values=1.0, block_chunks=0, num_register_tokens=0)
        dino_backbone.load_state_dict(torch.load(os.path.join(model_path,"dinov2_vits14_pretrain.pth")))
    elif dinov2_size == 'large':
        dino_backbone = vit_large(patch_size = 14, img_size = 524, init_values=1.0, block_chunks=0, num_register_tokens=0)
        dino_backbone.load_state_dict(torch.load(os.path.join(model_path,"dinov2_vitl14_pretrain.pth")))
    else:
        raise ValueError(f"Unsupported DINOv2 size: '{dinov2_size}'. Please choose 'small', 'base' or 'large'.")

    # Determine depth and select layers
    depth = getattr(dino_backbone, 'n_blocks', None) or len(getattr(dino_backbone, 'blocks'))
    if method == "multilayer":
        n = _select_spread_layers(depth, 4)
    else:
        n = _select_last_k_layers(depth, 1)
    dino_backbone.forward = partial(
            dino_backbone.get_intermediate_layers,
            n=n,
            reshape=True,
        )
    return dino_backbone

def _select_last_k_layers(total_depth: int, k: int) -> List[int]:
    if k <= 0:
        return []
    start = max(0, total_depth - k)
    return list(range(start, total_depth))

def _select_spread_layers(total_depth: int, k: int = 4) -> List[int]:
    """Select k layers approximately evenly spaced across depth.
    For total_depth=12 and k=4, returns [2,5,8,11] aligning with module_segdino.
    """
    if k <= 0 or total_depth <= 0:
        return []
    idxs: List[int] = []
    for i in range(k):
        # round to nearest int index in [0, total_depth-1]
        idx = max(0, min(total_depth - 1, round((i + 1) * total_depth / k) - 1))
        idxs.append(idx)
    # ensure strictly non-decreasing and unique by adjusting backward if necessary
    for i in range(1, len(idxs)):
        if idxs[i] <= idxs[i - 1]:
            idxs[i] = min(total_depth - 1, idxs[i - 1] + 1)
    if len(idxs) > k:
        idxs = idxs[-k:]
    return idxs

def _safe_import_dinov3(model_repo_path: str):
    """
    Try to import dinov3 from model_repo_path/dinov3 first; if that fails, fall back to local package.
    """
    repo_candidate = os.path.join(model_repo_path, "dinov3")
    if os.path.isdir(repo_candidate):
        if repo_candidate not in sys.path:
            sys.path.insert(0, repo_candidate)
    try:
        from dinov3.models.vision_transformer import vit_base as v3_vit_base  # type: ignore
        from dinov3.models.vision_transformer import vit_large as v3_vit_large  # type: ignore
        from dinov3.models.vision_transformer import vit_small as v3_vit_small  # type: ignore
        return v3_vit_small, v3_vit_base, v3_vit_large
    except Exception:
        # Fallback: try importing the top-level dinov3 package present in this repo
        if os.path.isdir("dinov3") and (os.getcwd() not in sys.path):
            sys.path.insert(0, os.getcwd())
        from dinov3.models.vision_transformer import vit_base as v3_vit_base  # type: ignore
        from dinov3.models.vision_transformer import vit_large as v3_vit_large  # type: ignore
        from dinov3.models.vision_transformer import vit_small as v3_vit_small  # type: ignore
        return v3_vit_small, v3_vit_base, v3_vit_large

def _find_weight(model_path: str, patterns: Sequence[str]) -> Optional[str]:
    for pat in patterns:
        full_pat = os.path.join(model_path, pat)
        matches = sorted(glob.glob(full_pat))
        if matches:
            return matches[0]
    return None

def _normalize_state_dict_keys(state: dict) -> dict:
    # Handle common wrappers
    for k in ("state_dict", "model", "teacher", "student"):
        if isinstance(state, dict) and k in state and isinstance(state[k], dict):
            state = state[k]

    def strip_prefix(k: str) -> str:
        for p in ("module.", "backbone.", "model.", "teacher.", "student."):
            if k.startswith(p):
                return k[len(p):]
        return k

    return {strip_prefix(k): v for k, v in state.items()}


def _load_weights_strict_subset(model: nn.Module, state: dict, verbose: bool = True) -> float:
    model_sd = model.state_dict()
    state = _normalize_state_dict_keys(state)
    matched = {}
    skipped = []
    for k, v in state.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            matched[k] = v
        else:
            skipped.append(k)
    missing = [k for k in model_sd.keys() if k not in matched]
    model_sd.update(matched)
    model.load_state_dict(model_sd)
    ratio = len(matched) / max(1, len(model_sd))
    if verbose:
        print(f"[DINOv3] Loaded {len(matched)}/{len(model_sd)} tensors ({ratio*100:.2f}%). Skipped {len(skipped)} keys. Missing {len(missing)} keys.")
        if len(skipped) > 0:
            print(f"[DINOv3] Example skipped: {skipped[:5]}")
        if len(missing) > 0:
            print(f"[DINOv3] Example missing: {missing[:5]}")
    return ratio


def create_backbone_dinov3(method: str,
                           model_repo_path: str,
                           model_path: str,
                           dinov3_size: str = "base",
                           dinov3_weights_path: Optional[str] = None,
                           dinov3_rope_dtype: str = "bf16"):
    """
    Build a DINOv3 backbone and adapt its forward to return intermediate feature maps.
    """
    v3_vit_small, v3_vit_base, v3_vit_large = _safe_import_dinov3(model_repo_path)

    # Choose default filename patterns by size
    if dinov3_size == 'base':
        default_patterns = ["dinov3_vitb16_pretrain*.pth", "dinov3_vitb16*.pth"]
    elif dinov3_size == 'small':
        default_patterns = ["dinov3_vits16_pretrain*.pth", "dinov3_vits16*.pth"]
    elif dinov3_size == 'large':
        default_patterns = ["dinov3_vitl16_pretrain*.pth", "dinov3_vitl16*.pth"]
    else:
        raise ValueError(f"Unsupported DINOv3 size: '{dinov3_size}'. Please choose 'small', 'base' or 'large'.")

    # Load weights early to infer model options (e.g., storage tokens, layerscale)
    weight_path = dinov3_weights_path or _find_weight(model_path, default_patterns)
    if weight_path is None or (not os.path.exists(weight_path)):
        raise FileNotFoundError(
            f"DINOv3 weights not found. Looked for patterns {default_patterns} under '{model_path}'. "
            f"You can set 'dinov3_weights_path' to the exact file."
        )
    early_state = torch.load(weight_path, map_location='cpu')
    norm_state = _normalize_state_dict_keys(early_state if isinstance(early_state, dict) else {})

    # Infer storage tokens
    n_storage_tokens = 0
    if 'storage_tokens' in norm_state and hasattr(norm_state['storage_tokens'], 'shape'):
        try:
            n_storage_tokens = int(norm_state['storage_tokens'].shape[1])
        except Exception:
            n_storage_tokens = 0

    # Infer presence of LayerScale (gamma parameters)
    has_layerscale = any(('.ls1.gamma' in k) or ('.ls2.gamma' in k) for k in norm_state.keys())

    # Infer masked K-bias in attention (LinearKMaskedBias)
    has_mask_k_bias = any(k.endswith('.attn.qkv.bias_mask') for k in norm_state.keys())

    # Build the backbone with inferred options
    vit_kwargs = dict(pos_embed_rope_dtype=dinov3_rope_dtype,
                      n_storage_tokens=n_storage_tokens,
                      layerscale_init=(1e-5 if has_layerscale else None),
                      mask_k_bias=has_mask_k_bias)

    if dinov3_size == 'base':
        dino_backbone = v3_vit_base(patch_size=16, **vit_kwargs)
    elif dinov3_size == 'small':
        dino_backbone = v3_vit_small(patch_size=16, **vit_kwargs)
    else:  # large
        dino_backbone = v3_vit_large(patch_size=16, **vit_kwargs)

    # Now load weights robustly; require high match ratio to proceed
    state = early_state
    match_ratio = _load_weights_strict_subset(dino_backbone, state, verbose=True)
    if match_ratio < 0.85:
        raise RuntimeError(
            f"DINOv3 weights load ratio too low ({match_ratio*100:.2f}%). "
            f"Please verify 'dinov3_weights_path' matches the selected model size ('{dinov3_size}')."
        )

    # Select layers: spread4 for multilayer, last1 otherwise
    depth = getattr(dino_backbone, 'n_blocks', None) or len(getattr(dino_backbone, 'blocks'))
    if method == "multilayer":
        n = _select_spread_layers(depth, 4)
    else:
        n = _select_last_k_layers(depth, 1)

    # Wrap forward to return (tuple of) [B, C, H, W]
    dino_backbone.forward = partial(
        dino_backbone.get_intermediate_layers,
        n=n,
        reshape=True,
    )
    return dino_backbone

class DINO_linear(nn.Module):
    def __init__(self,
                 version,
                 method,
                 num_classes,
                 input_size,
                 model_repo_path,
                 model_path,
                 dinov2_size: str = "base",
                 dinov3_size: str = "base",
                 dinov3_weights_path: Optional[str] = None,
                 dinov3_rope_dtype: str = "bf16",
                 # FDM integration flags (single policy)
                 fdm_enable_apm: bool = False,
                 fdm_apm_mode: str = "S",
                 fdm_enable_acpa: bool = False):
        super().__init__()
        self.method = method
        self.version = version
        self.input_size = input_size
        # FDM flags
        self.fdm_enable_apm = bool(fdm_enable_apm)
        self.fdm_apm_mode = str(fdm_apm_mode or "S").upper()
        if self.fdm_apm_mode not in ("S", "M"):
            self.fdm_apm_mode = "S"
        self.fdm_enable_acpa = bool(fdm_enable_acpa)
        if self.version == 3:
            self.encoder = create_backbone_dinov3(method, model_repo_path, model_path, dinov3_size, dinov3_weights_path, dinov3_rope_dtype)
            if dinov3_size == 'base':
                self.in_channels = 768
            elif dinov3_size == 'small':
                self.in_channels = 384
            elif dinov3_size == 'large':
                self.in_channels = 1024
            else:
                raise ValueError(f"Unsupported DINOv3 size: '{dinov3_size}'. Please choose 'small', 'base' or 'large'.")
        elif self.version == 2 : 
            self.encoder = create_backbone_dinov2(method, model_repo_path, model_path, dinov2_size)
            if dinov2_size == 'base':
                self.in_channels = 768
            elif dinov2_size == 'small':
                self.in_channels = 384
            elif dinov2_size == 'large':
                self.in_channels = 1024
            else:
                raise ValueError(f"Unsupported DINOv2 size: '{dinov2_size}'. Please choose 'small', 'base' or 'large'.")
        else:
            raise ValueError(f"Unsupported DINO version: {self.version}. Only 2 and 3 are supported.")

        if method == "svf" : 
            self.encoder = resolver(self.encoder)
        if method == "lora" : 
            config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["qkv"],
                lora_dropout=0.1,
                bias="none",
            )
            self.encoder = get_peft_model(self.encoder, config)
        
        # Optional FDM modules (initialized lazily for spatial dims)
        self.apm = MaskModule(shape=None) if self.fdm_enable_apm else None
        self.acpa = PhaseAttention(self.in_channels) if self.fdm_enable_acpa else None

        # Decoder selection
        if method == "multilayer":
            # Configure per-layer out_channels aligned with module_segdino
            if self.version == 3:
                if dinov3_size in ('small', 'base'):
                    out_chs = [96, 192, 384, 768]
                elif dinov3_size == 'large':
                    out_chs = [192, 384, 768, 1024]
                else:
                    raise ValueError(f"Unsupported DINOv3 size: '{dinov3_size}'.")
            elif self.version == 2:
                if dinov2_size in ('small', 'base'):
                    out_chs = [96, 192, 384, 768]
                elif dinov2_size == 'large':
                    out_chs = [192, 384, 768, 1024]
                else:
                    raise ValueError(f"Unsupported DINOv2 size: '{dinov2_size}'.")
            else:
                raise ValueError(f"Unsupported DINO version: {self.version}.")

            self.decoder = DPTDecoder(in_channels=self.in_channels,
                                      num_classes=num_classes,
                                      features=128,
                                      out_channels=out_chs,
                                      use_bn=True)
            self.bn = None  # not used in multilayer DPT path
        else:
            self.decoder = nn.Conv2d(self.in_channels, num_classes, kernel_size=1)
            self.bn = nn.SyncBatchNorm(self.in_channels)

    def _ensure_apm_shape(self, feat: torch.Tensor) -> None:
        """Ensure APM parameters are initialized with batch-agnostic shape.
        S-mode: [1,1,H,W]; M-mode: [1,C,H,W].
        """
        if self.apm is None:
            return
        if not isinstance(feat, torch.Tensor):
            return
        _, c, h, w = feat.shape
        if self.fdm_apm_mode == "S":
            target = (1, 1, h, w)
        else:  # "M"
            target = (1, c, h, w)
        # Set target_shape so that MaskModule initializes without binding batch size
        if getattr(self.apm, "target_shape", None) != target:
            self.apm.target_shape = target
        # Trigger (re)initialization if needed
        self.apm._ensure_initialized(feat)

    def forward(self, x): 
        if self.version == 3:
            input_dim = int(self.input_size/16)*16
        elif self.version == 2 : 
            input_dim = int(self.input_size/14)*14
        else:
            raise ValueError(f"Unsupported DINO version: {self.version}. Only 2 and 3 are supported.")

        if self.method in ["linear", "multilayer"] : 
            with torch.no_grad():
                x = F.interpolate(x, size=[input_dim,input_dim], mode='bilinear', align_corners=False)
                x = self.encoder(x)
        else : 
            x = F.interpolate(x, size=[input_dim,input_dim], mode='bilinear', align_corners=False)
            x = self.encoder(x)
        
        if self.method == "multilayer":
            feats = list(x) if isinstance(x, (list, tuple)) else [x]
            # Apply FDM on selected layers if enabled
            if (self.apm is not None) or (self.acpa is not None):
                n = len(feats)
                # Single policy: apply only on the deeper two features (last two indices)
                apply_ids = list(range(max(0, n - 2), n))
                for i in apply_ids:
                    if self.apm is not None:
                        self._ensure_apm_shape(feats[i])
                        feats[i] = self.apm(feats[i])
                    if self.acpa is not None:
                        feats[i] = self.acpa(feats[i])
            return self.decoder(feats)
        else:
            x = torch.cat(x, dim=1)
            # Apply FDM for linear/svf path prior to BN/Conv
            if self.apm is not None:
                self._ensure_apm_shape(x)
                x = self.apm(x)
            if self.acpa is not None:
                x = self.acpa(x)
            x = self.bn(x)
            return self.decoder(x)
