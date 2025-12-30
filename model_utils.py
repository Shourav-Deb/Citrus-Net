import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np

# -------------------------------
# Load model from checkpoint
# -------------------------------
def load_model(ckpt_path, num_classes=4):
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # Example: EfficientNet-B0 (best TL model)
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


# -------------------------------
# Grad-CAM helper
# -------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, class_idx):
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        return cam[0].cpu().numpy()


def overlay_cam(image, cam):
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap * 0.5 + image * 0.5
    return np.clip(overlay, 0, 1)
