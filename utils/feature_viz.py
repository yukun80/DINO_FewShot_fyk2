import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from typing import Optional, Tuple

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def denormalize_to_numpy(image: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized tensor image [C,H,W] back to RGB numpy array in [0,1].
    """
    if image.dim() != 3:
        raise ValueError(f"Expected image tensor with shape [C,H,W], got {tuple(image.shape)}")
    device = image.device
    mean = IMAGENET_MEAN.to(device).view(3, 1, 1)
    std = IMAGENET_STD.to(device).view(3, 1, 1)
    img = (image * std) + mean
    img = img.clamp(0.0, 1.0)
    img = img.permute(1, 2, 0).cpu().numpy()
    return img


def compute_grad_cam(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    target_size: tuple[int, int],
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap from activations/gradients and upsample to `target_size`.
    """
    if activations.shape != gradients.shape:
        raise ValueError(
            f"Activation/gradient shape mismatch: {tuple(activations.shape)} vs {tuple(gradients.shape)}"
        )

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=target_size, mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)
    return cam


def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    cmap_name: str = "inferno",
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Blend an RGB image in [0,1] with a heatmap in [0,1] using the provided colormap.
    Returns uint8 RGB array.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape [H,W,3], got {image_rgb.shape}")
    cmap = cm.get_cmap(cmap_name)
    colored = cmap(np.clip(heatmap, 0.0, 1.0))[..., :3]
    blended = (1.0 - alpha) * image_rgb + alpha * colored
    blended = np.clip(blended, 0.0, 1.0)
    return (blended * 255).astype(np.uint8)


def logits_to_probability_map(
    logits: torch.Tensor,
    class_index: int,
    reduce_batch: bool = True,
) -> torch.Tensor:
    """
    Convert raw logits [B, C, H, W] to a probability map for the given class.

    Args:
        logits: Tensor of raw logits.
        class_index: Target class channel to extract.
        reduce_batch: When True, squeezes the batch dimension (expects B==1).
    """
    if logits.dim() != 4:
        raise ValueError(f"Expected logits with shape [B,C,H,W], got {tuple(logits.shape)}")
    probs = torch.softmax(logits, dim=1)
    if class_index < 0 or class_index >= probs.shape[1]:
        raise ValueError(f"class_index {class_index} out of range for logits with {probs.shape[1]} channels")
    out = probs[:, class_index, :, :]
    if reduce_batch:
        if out.shape[0] != 1:
            raise ValueError("Batch size must be 1 when reduce_batch=True")
        out = out.squeeze(0)
    return out


def probability_to_heatmap_overlay(
    image_rgb: np.ndarray,
    prob_map: torch.Tensor | np.ndarray,
    cmap_name: str = "inferno",
    alpha: float = 0.6,
    value_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Render a probability map as a heatmap overlay on the RGB image.

    Args:
        image_rgb: RGB image in [0,1], shape [H,W,3].
        prob_map: Probability map tensor/array with shape [H,W].
        cmap_name: Matplotlib colormap name.
        alpha: Blend factor for overlay.
        value_range: Optional (min, max) clip range; defaults to data min/max.
    """
    if isinstance(prob_map, torch.Tensor):
        data = prob_map.detach().cpu().numpy()
    else:
        data = np.asarray(prob_map)
    if data.ndim != 2:
        raise ValueError(f"Expected probability map with shape [H,W], got {data.shape}")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape [H,W,3], got {image_rgb.shape}")
    if value_range is None:
        vmin, vmax = float(np.min(data)), float(np.max(data))
        if vmax - vmin < 1e-6:
            scaled = np.zeros_like(data)
        else:
            scaled = (data - vmin) / (vmax - vmin)
    else:
        vmin, vmax = value_range
        if vmax <= vmin:
            raise ValueError(f"Invalid value_range: ({vmin}, {vmax})")
        scaled = np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)
    return overlay_heatmap(image_rgb, scaled, cmap_name=cmap_name, alpha=alpha)


def summarize_probability_map(prob_map: torch.Tensor) -> Tuple[float, float, float]:
    """
    Return (min, mean, max) statistics for a probability map tensor.
    """
    if prob_map.ndim != 2:
        raise ValueError(f"Expected probability map with shape [H,W], got {tuple(prob_map.shape)}")
    return (
        float(prob_map.min().item()),
        float(prob_map.mean().item()),
        float(prob_map.max().item()),
    )
