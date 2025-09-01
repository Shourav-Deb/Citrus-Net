import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import numpy as np
sys.path.append(r'C:\Users\HP\OneDrive\Desktop\Streamlit')  # Ensure model.py is importable
from model import CitrusNet
import torchvision.models as tvm

# -----------------------------
# 1. Model selection and loading
# -----------------------------
model_options = {
    "CitrusNet": r"C:\Users\HP\OneDrive\Desktop\Streamlit\custom_cnn_best.pt",
    "ConvNeXt-Tiny": r"C:\Users\HP\OneDrive\Desktop\Streamlit\tl_convnext_tiny_best.pt",
    "DenseNet121": r"C:\Users\HP\OneDrive\Desktop\Streamlit\tl_densenet121_best.pt",
    "EfficientNet-B0": r"C:\Users\HP\OneDrive\Desktop\Streamlit\tl_efficientnet_b0_best.pt",
    "ResNet34": r"C:\Users\HP\OneDrive\Desktop\Streamlit\tl_resnet34_best.pt",
    "ViT": r"C:\Users\HP\OneDrive\Desktop\Streamlit\vit_best.pt"
}
selected_model_name = st.sidebar.selectbox("Choose Model", list(model_options.keys()))
selected_model_path = model_options[selected_model_name]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 128 if selected_model_name == "CitrusNet" else 224
class_names = ["murcott", "ponkan", "tangerine", "tankan"]

import torchvision.models as tvm

def get_model_and_layer(name):
    if name == "CitrusNet":
        model = CitrusNet(num_classes=4)
        target_layer = model.stage6.conv.pw
    elif name == "ResNet34":
        model = tvm.resnet34(weights=None)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, 4)
        target_layer = model.layer4[-1].conv2
    elif name == "DenseNet121":
        model = tvm.densenet121(weights=None)
        in_feats = model.classifier.in_features
        model.classifier = nn.Linear(in_feats, 4)
        target_layer = model.features.denseblock4.denselayer16.conv2
    elif name == "EfficientNet-B0":
        model = tvm.efficientnet_b0(weights=None)
        in_feats = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feats, 4)
        target_layer = model.features[-1][0]
    elif name == "ConvNeXt-Tiny":
        model = tvm.convnext_tiny(weights=None)
        in_feats = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_feats, 4)
        target_layer = model.features[-1][-1].block[2]
    elif name == "ViT":
        model = tvm.vit_b_16(weights=None)
        in_feats = model.heads.head.in_features
        model.heads.head = nn.Linear(in_feats, 4)
        # GradCAM for ViT is experimental; use last encoder block norm
        for n, m in reversed(list(model.named_modules())):
            if "encoder.layers" in n and ("ln_1" in n or "layernorm1" in n):
                target_layer = m
                break
        else:
            target_layer = list(model.modules())[-2]
    else:
        raise ValueError("Unknown model name")
    return model, target_layer

def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    return model

try:
    model, target_layer = get_model_and_layer(selected_model_name)
    model = load_checkpoint(model, selected_model_path)
    model.eval().to(device)
    st.success(f"✅ Model loaded: {selected_model_name}")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# -----------------------------
# 2. Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor()
])

# -----------------------------
# 3. XAI Methods
# -----------------------------
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np

def apply_gradcam(img_tensor, method="Grad-CAM"):
    if method == "Grad-CAM":
        cam = GradCAM(model=model, target_layers=[target_layer])
    elif method == "Grad-CAM++":
        cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    elif method == "Eigen-CAM":
        cam = EigenCAM(model=model, target_layers=[target_layer])
    elif method == "Ablation-CAM":
        cam = AblationCAM(model=model, target_layers=[target_layer])
    else:
        return None

    grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0, :]
    img_np = img_tensor.squeeze().permute(1,2,0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return cam_image

def apply_lime(pil_img):
    def batch_predict(images):
        model.eval()
        batch = torch.stack([
            transform(Image.fromarray(img).resize((input_size, input_size))) for img in images
        ], dim=0).to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        np.array(pil_img.resize((input_size, input_size))),
        batch_predict,
        labels=tuple(range(len(class_names))),
        hide_color=0,
        num_samples=1000
    )
    temp, mask = explanation.get_image_and_mask(
        label=explanation.top_labels[0],
        positive_only=True,
        hide_rest=False,
        num_features=10,
        min_weight=0.02
    )
    return mark_boundaries(temp, mask)

# -----------------------------
# 4. Streamlit UI
# -----------------------------
st.title("🍊 Citrus Fruit Classification + XAI")
st.write("Upload a fruit image → model predicts → choose XAI methods to explain it")

uploaded_file = st.file_uploader("Upload a fruit image", type=["jpg","jpeg","png"])

xai_methods = st.sidebar.multiselect(
    "Choose Explanation Methods",
    ["Grad-CAM", "Grad-CAM++", "Eigen-CAM", "Ablation-CAM", "LIME"],
    default=["Grad-CAM"]
)

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_label = class_names[pred_idx]

    st.subheader(f"Prediction: {pred_label} ({probs[pred_idx]*100:.2f}%)")
    st.image(pil_img, caption="Uploaded Image", width=250)

    st.markdown("#### Class Probabilities")
    prob_dict = {name: f"{probs[i]*100:.2f}%" for i, name in enumerate(class_names)}
    st.table(prob_dict)

    # Show explanation methods
    st.subheader("Explanations")
    cols = st.columns(len(xai_methods))
    for i, method in enumerate(xai_methods):
        try:
            if method == "LIME":
                explained = apply_lime(pil_img)
            else:
                explained = apply_gradcam(img_tensor, method)
            with cols[i]:
                st.image(explained, caption=method, use_container_width=True)
        except Exception as e:
            with cols[i]:
                st.error(f"{method} error: {e}")