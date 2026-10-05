"""
Grad-CAM for the DR severity classifier (EfficientNet-B0 / timm).
The PS explicitly asks for Grad-CAM attention maps alongside lesion evidence
and confidence - this adds that, without replacing the lesion-evidence report,
which stays the primary explanation (see README for why).
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Hooks the last convolutional layer of a timm EfficientNet model and
    computes a Grad-CAM heatmap for a chosen (or predicted) class."""

    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None

        # timm EfficientNet: last conv block output, before global pooling.
        target_layer = model.conv_head if hasattr(model, "conv_head") else model.blocks[-1]

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        """x: 1 x C x H x W tensor, already normalized, on the right device.
        Returns (heatmap as H x W float32 in [0, 1], predicted/used class_idx, probs)."""
        self.model.zero_grad()
        out = self.model(x)
        probs = torch.softmax(out, dim=1)[0]

        if class_idx is None:
            class_idx = int(probs.argmax())

        out[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # global-avg-pool the gradients
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, class_idx, probs.detach().cpu().numpy()


def overlay_heatmap(image_rgb, heatmap, alpha=0.4):
    """image_rgb: H x W x 3 uint8. heatmap: H' x W' float32 in [0,1] (any size - resized to match).
    Returns an H x W x 3 uint8 RGB overlay."""
    h, w = image_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_u8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1 - alpha, colored, alpha, 0)


def compute_gradcam_overlay(model, image_rgb, transform, device, class_idx=None):
    """Convenience wrapper: image_rgb (H x W x 3 uint8) -> (overlay, class_idx, probs).
    `transform` must be the same val/eval-time Albumentations transform used for
    classification (resize + normalize + ToTensorV2), e.g. from stage6's get_transforms().
    Gradients must be enabled, so this must NOT be called inside torch.no_grad()."""
    cam_tool = GradCAM(model)
    x = transform(image=image_rgb)["image"].unsqueeze(0).to(device)
    x.requires_grad_(False)  # not needed on the input itself, only on activations

    model.eval()
    heatmap, used_idx, probs = cam_tool(x, class_idx=class_idx)
    overlay = overlay_heatmap(image_rgb, heatmap)
    return overlay, used_idx, probs
