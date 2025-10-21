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

    if method == "multilayer" :
        n = [8,9,10,11]
    else : 
        n = [11]
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

def create_backbone_dinov3(method: str,
                           model_repo_path: str,
                           model_path: str,
                           dinov3_size: str = "base",
                           dinov3_weights_path: Optional[str] = None):
    """
    Build a DINOv3 backbone and adapt its forward to return intermediate feature maps.
    """
    v3_vit_small, v3_vit_base, v3_vit_large = _safe_import_dinov3(model_repo_path)

    if dinov3_size == 'base':
        dino_backbone = v3_vit_base(patch_size=16)
        default_patterns = ["dinov3_vitb16_pretrain*.pth", "dinov3_vitb16*.pth"]
    elif dinov3_size == 'small':
        dino_backbone = v3_vit_small(patch_size=16)
        default_patterns = ["dinov3_vits16_pretrain*.pth", "dinov3_vits16*.pth"]
    elif dinov3_size == 'large':
        dino_backbone = v3_vit_large(patch_size=16)
        default_patterns = ["dinov3_vitl16_pretrain*.pth", "dinov3_vitl16*.pth"]
    else:
        raise ValueError(f"Unsupported DINOv3 size: '{dinov3_size}'. Please choose 'small', 'base' or 'large'.")

    # Load weights
    weight_path = dinov3_weights_path or _find_weight(model_path, default_patterns)
    if weight_path is None or (not os.path.exists(weight_path)):
        raise FileNotFoundError(
            f"DINOv3 weights not found. Looked for patterns {default_patterns} under '{model_path}'. "
            f"You can set 'dinov3_weights_path' to the exact file."
        )
    state = torch.load(weight_path, map_location='cpu')
    # Check whether state dict is directly a mapping or wrapped
    if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
        state = state['state_dict']
    dino_backbone.load_state_dict(state, strict=False)

    # Select layers: last or last 4 depending on method
    depth = getattr(dino_backbone, 'n_blocks', None) or len(getattr(dino_backbone, 'blocks'))
    if method == "multilayer":
        n: Union[int, Sequence[int]] = _select_last_k_layers(depth, 4)
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
                 dinov3_weights_path: Optional[str] = None):
        super().__init__()
        self.method = method
        self.version = version
        self.input_size = input_size
        if self.version == 3:
            self.encoder = create_backbone_dinov3(method, model_repo_path, model_path, dinov3_size, dinov3_weights_path)
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
        
        if method == "multilayer" : 
            if self.version in (2, 3):
                self.in_channels *= 4 # Concatenate 4 layers for multi-layer mode

        self.decoder = nn.Conv2d(self.in_channels, num_classes, kernel_size=1)
        self.bn = nn.SyncBatchNorm(self.in_channels)

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
    
        x = torch.cat(x,dim=1)

        x = self.bn(x)
        return self.decoder(x)
