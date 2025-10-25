import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm

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
