import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.cm as cm

# ============================================================
# Stage B — Custom CNN (NO inplace ops to avoid Grad-CAM crash)
# ============================================================
class CitrusNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


# ============================================================
# Build model + select Grad-CAM target layer
# ============================================================
def build_model(model_name: str, num_classes: int):
    name = model_name.lower()

    if name == "custom_cnn":
        model = CitrusNet(num_classes)
        # pick the last conv layer (Conv2d(64->128)) as target
        target_layer = model.features[6]

    elif name == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        target_layer = model.layer4[-1].conv2

    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        target_layer = model.features[-1][0]

    elif name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        # last dense features layer norm/relu area varies; use features as target safely
        target_layer = model.features

    elif name == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        target_layer = model.features[-1][-1]

    elif name == "vit_b_16":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        target_layer = None  # Grad-CAM not applicable

    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    return model, target_layer


# ============================================================
# SAFE checkpoint loader (shape-filtered; DenseNet-proof)
# ============================================================
def load_model(ckpt_path: str, model_name: str, num_classes: int):
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format")

    model, target_layer = build_model(model_name, num_classes)
    model_dict = model.state_dict()

    # Filter by key + shape (prevents DenseNet and version mismatch crashes)
    filtered_state = {}
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered_state[k] = v

    model.load_state_dict(filtered_state, strict=False)
    model.eval()
    return model, target_layer


# ============================================================
# Grad-CAM (no inplace ops here either)
# ============================================================
class GradCAM:
    def __init__(self, target_layer):
        self.activations = None
        self.gradients = None

        self._h1 = target_layer.register_forward_hook(self._forward_hook)
        self._h2 = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self):
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)  # NOT inplace
        cam = cam / (cam.max() + 1e-8)
        return cam[0].cpu().numpy()

    def close(self):
        # optional cleanup
        try:
            self._h1.remove()
            self._h2.remove()
        except Exception:
            pass


# ============================================================
# Overlay CAM (no OpenCV)
# ============================================================
def overlay_cam(image: np.ndarray, cam: np.ndarray):
    H, W, _ = image.shape
    cam_resized = resize_cam(cam, H, W)
    heatmap = cm.jet(np.clip(cam_resized, 0, 1))[..., :3]
    overlay = 0.5 * image + 0.5 * heatmap
    return np.clip(overlay, 0, 1)


def resize_cam(cam: np.ndarray, H: int, W: int):
    h, w = cam.shape
    y_idx = (np.linspace(0, h - 1, H)).astype(int)
    x_idx = (np.linspace(0, w - 1, W)).astype(int)
    return cam[y_idx[:, None], x_idx]
