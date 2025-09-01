# utils.py
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import cv2
import numpy as np

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model(model_path):
    model = torch.load(model_path, map_location="cpu")
    model.eval()
    return model

def load_class_mapping(json_path):
    with open(json_path) as f:
        class_to_idx = json.load(f)
    return {v: k for k, v in class_to_idx.items()}

def predict_image(model, img, idx_to_class):
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)
    return idx_to_class[pred_idx.item()], conf.item()

def generate_gradcam(model, img, target_layer, idx_to_class):
    img_t = transform(img).unsqueeze(0)

    activations, gradients = [], []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)

    output = model(img_t)
    pred_class = output.argmax().item()
    model.zero_grad()
    output[0, pred_class].backward()

    handle_fwd.remove()
    handle_bwd.remove()

    grad = gradients[0].mean(dim=[2, 3], keepdim=True)
    cam = (grad * activations[0]).sum(dim=1).squeeze()
    cam = torch.relu(cam)
    cam = cam / cam.max()

    cam = cv2.resize(cam.cpu().numpy(), (img.size[0], img.size[1]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_cv, 0.5, heatmap, 0.5, 0)

    return overlay, idx_to_class[pred_class]
