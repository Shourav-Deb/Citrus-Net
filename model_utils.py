import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.cm as cm

# ==================================================
# Custom CNN (Stage B)
# ==================================================
class CitrusNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


# ==================================================
# Build model by name
# ==================================================
def build_model(model_name, num_classes):
    name = model_name.lower()

    if name == "custom_cnn":
        model = CitrusNet(num_classes)
        target_layer = model.features[-3]

    elif name == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        target_layer = model.layer4[-1].conv2

    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )
        target_layer = model.features[-1][0]

    elif name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(
            model.classifier.in_features, num_classes
        )
        target_layer = model.features[-1]

    elif name == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(
            model.classifier[2].in_features, num_classes
        )
        target_layer = model.features[-1][-1]

    elif name == "vit_b_16":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(
            model.heads.head.in_features, num_classes
        )
        target_layer = None

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model, target_layer


# ==================================================
# SAFE checkpoint loader (FIXED)
# ==================================================
def load_model(ckpt_path, model_name, num_classes):
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # Handle both checkpoint formats
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format")

    model, target_layer = build_model(model_name, num_classes)

    # Try strict load first
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        # Fallback: allow head mismatch
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model, target_layer


# ==================================================
# Grad-CAM
# ==================================================
class GradCAM:
    def __init__(self, target_layer):
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.activations = o.detach()

    def _bwd(self, m, gi, go):
        self.gradients = go[0].detach()

    def generate(self):
        w = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (w * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        return cam[0].cpu().numpy()


# ==================================================
# Overlay CAM
# ==================================================
def overlay_cam(image, cam):
    H, W, _ = image.shape
    cam = resize_cam(cam, H, W)
    heatmap = cm.jet(cam)[..., :3]
    return np.clip(0.5 * image + 0.5 * heatmap, 0, 1)


def resize_cam(cam, H, W):
    h, w = cam.shape
    y = (np.linspace(0, h - 1, H)).astype(int)
    x = (np.linspace(0, w - 1, W)).astype(int)
    return cam[y[:, None], x]
