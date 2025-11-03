import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from functools import partial
from typing import List, Sequence, Optional
import glob
from peft import LoraConfig, get_peft_model
from models.svf import *
from models.decoders.dpt import DPTDecoder
from modules.module_FDM.freq_masker import MaskModule
from modules.module_FDM.phase_attn import PhaseAttention

def create_backbone_dinov2(model_repo_path, model_path, dinov2_size: str = "base") :
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
    n = _select_spread_layers(depth, 4)
    dino_backbone.forward = partial(
            dino_backbone.get_intermediate_layers,
            n=n,
            reshape=True,
        )
    return dino_backbone

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


def create_backbone_dinov3(model_repo_path: str,
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

    # Select evenly spaced layers for multilayer decoding
    depth = getattr(dino_backbone, 'n_blocks', None) or len(getattr(dino_backbone, 'blocks'))
    n = _select_spread_layers(depth, 4)

    # Wrap forward to return (tuple of) [B, C, H, W]
    dino_backbone.forward = partial(
        dino_backbone.get_intermediate_layers,
        n=n,
        reshape=True,
        )
    return dino_backbone

class DINOMultilayer(nn.Module):
    def __init__(self,
                 version,
                 num_classes,
                 input_size,
                 model_repo_path,
                 model_path,
                 dinov2_size: str = "base",
                 dinov3_size: str = "base",
                 dinov3_weights_path: Optional[str] = None,
                 dinov3_rope_dtype: str = "bf16",
                 # Encoder adapters decoupled from decoder type
                 encoder_adapters: str = "auto",  # {auto|none|lora|svf}
                 # FDM integration flags (single policy)
                 fdm_enable_apm: bool = False,
                 fdm_apm_mode: str = "S",
                 fdm_enable_acpa: bool = False):
        super().__init__()
        self.version = version
        self.input_size = input_size
        # FDM flags
        self.fdm_enable_apm = bool(fdm_enable_apm)
        self.fdm_apm_mode = str(fdm_apm_mode or "S").upper()
        if self.fdm_apm_mode not in ("S", "M"):
            self.fdm_apm_mode = "S"
        self.fdm_enable_acpa = bool(fdm_enable_acpa)
        # Determine adapter mode
        enc_adapters = (encoder_adapters or "auto").lower()
        if enc_adapters not in ("auto", "none", "lora", "svf"):
            enc_adapters = "auto"
        if enc_adapters == "auto":
            enc_adapters = "none"
        self.encoder_adapters = enc_adapters
        if self.version == 3:
            self.encoder = create_backbone_dinov3(model_repo_path, model_path, dinov3_size, dinov3_weights_path, dinov3_rope_dtype)
            if dinov3_size == 'base':
                self.in_channels = 768
            elif dinov3_size == 'small':
                self.in_channels = 384
            elif dinov3_size == 'large':
                self.in_channels = 1024
            else:
                raise ValueError(f"Unsupported DINOv3 size: '{dinov3_size}'. Please choose 'small', 'base' or 'large'.")
        elif self.version == 2 : 
            self.encoder = create_backbone_dinov2(model_repo_path, model_path, dinov2_size)
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

        # Inject encoder adapters regardless of decoder type
        if self.encoder_adapters == "svf":
            self.encoder = resolver(self.encoder)
        elif self.encoder_adapters == "lora":
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

        # Encoder freeze policy: with adapters='none' we hard-freeze the backbone.
        self.encoder_frozen = self.encoder_adapters == "none"
        if self.encoder_frozen:
            for param in self.encoder.parameters():
                param.requires_grad_(False)
        self._refresh_encoder_grad_policy()

        # Decoder selection
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

    def _resize_to_backbone(self, x: torch.Tensor) -> torch.Tensor:
        if self.version == 3:
            input_dim = int(self.input_size/16)*16
        elif self.version == 2 : 
            input_dim = int(self.input_size/14)*14
        else:
            raise ValueError(f"Unsupported DINO version: {self.version}. Only 2 and 3 are supported.")
        return F.interpolate(x, size=[input_dim, input_dim], mode='bilinear', align_corners=False)

    def _forward_encoder(self, x: torch.Tensor, keep_encoder_grad: Optional[bool] = None) -> List[torch.Tensor]:
        resized = self._resize_to_backbone(x)
        if keep_encoder_grad is None:
            keep_encoder_grad = bool(getattr(self, "encoder_keep_grad_default", True))
        if not keep_encoder_grad:
            with torch.no_grad():
                feats = self.encoder(resized)
        else:
            feats = self.encoder(resized)
        feats_list = list(feats) if isinstance(feats, (list, tuple)) else [feats]
        return feats_list

    def _apply_fdm_multilayer(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply the optional FDM stack (APM → ACPA) to each multilayer feature map.
        """
        out = list(feats)
        if (self.apm is None) and (self.acpa is None):
            return out
        for idx, tensor in enumerate(out):
            if self.apm is not None:
                self._ensure_apm_shape(tensor)
                tensor = self.apm(tensor)
            if self.acpa is not None:
                tensor = self.acpa(tensor)
            out[idx] = tensor
        return out

    def _refresh_encoder_grad_policy(self) -> None:
        """Cache whether encoder outputs should keep gradients by default."""
        self.encoder_keep_grad_default = any(param.requires_grad for param in self.encoder.parameters())

    def forward(self, x):
        feats = self._forward_encoder(x, keep_encoder_grad=None)
        feats = self._apply_fdm_multilayer(feats)
        return self.decoder(feats)

    def forward_with_feature_maps(self, x: torch.Tensor, keep_encoder_grad: Optional[bool] = None):
        """
        Extended forward that surfaces intermediate feature maps for visualization.

        Returns:
            logits: decoder logits [B, num_classes, H, W]
            stage1: backbone feature map before FDM (deepest layer for multilayer)
            stage2: feature map after FDM (same spatial scale)
            stage3: decoder fused tensor prior to the classification head
        """
        feats_raw = self._forward_encoder(x, keep_encoder_grad=keep_encoder_grad)
        feats_fdm = self._apply_fdm_multilayer(feats_raw)
        logits, fused = self.decoder.forward_with_fused(feats_fdm)
        stage1 = feats_raw[-1]
        stage2 = feats_fdm[-1]
        stage3 = fused
        return logits, stage1, stage2, stage3

    def encoder_features(self, x: torch.Tensor, keep_encoder_grad: Optional[bool] = None) -> List[torch.Tensor]:
        """
        Public helper to run the ViT encoder while respecting the frozen/trainable policy.
        Passing keep_encoder_grad overrides the default behaviour.
        """
        return self._forward_encoder(x, keep_encoder_grad=keep_encoder_grad)
