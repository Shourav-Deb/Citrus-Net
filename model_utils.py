import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.cm as cm

# -------------------------------
# Load uploaded model
# -------------------------------
def load_model(ckpt_path, num_classes):
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # Change backbone here if needed
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
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self):
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        return cam[0].cpu().numpy()


# -------------------------------
# Overlay heatmap
# -------------------------------
def overlay_cam(image, cam):
    cam = np.clip(cam, 0, 1)
    heatmap = cm.jet(cam)[..., :3]
    overlay = 0.5 * image + 0.5 * heatmap
    return np.clip(overlay, 0, 1)
