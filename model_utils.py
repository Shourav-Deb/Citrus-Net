import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.cm as cm

# -------------------------------
# Load model
# -------------------------------
def load_model(ckpt_path, num_classes=4):
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


# -------------------------------
# Grad-CAM
# -------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, class_idx):
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        return cam[0].cpu().numpy()


# -------------------------------
# Heatmap overlay (NO OpenCV)
# -------------------------------
def overlay_cam(image, cam):
    cam = np.clip(cam, 0, 1)
    heatmap = cm.jet(cam)[..., :3]
    overlay = 0.5 * image + 0.5 * heatmap
    return np.clip(overlay, 0, 1)
