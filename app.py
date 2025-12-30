import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from model_utils import load_model, GradCAM, overlay_cam

# -------------------------------
# App config
# -------------------------------
st.set_page_config(page_title="Citrus Fruit Classifier", layout="wide")
st.title("Citrus Fruit Classification (All Models) with Grad-CAM")

# Your class names (must match training)
CLASSES = ["murcott", "ponkan", "tangerine", "tankan"]

MODEL_OPTIONS = {
    "Custom CNN (Stage B)": "custom_cnn",
    "ResNet34 (Transfer Learning)": "resnet34",
    "EfficientNet-B0 (Transfer Learning)": "efficientnet_b0",
    "DenseNet121 (Transfer Learning)": "densenet121",
    "ConvNeXt-Tiny (Transfer Learning)": "convnext_tiny",
    "ViT-B/16 (Stage D)": "vit_b_16",
}

# -------------------------------
# Sidebar: model selection + uploads
# -------------------------------
st.sidebar.header("Model")
model_label = st.sidebar.selectbox("Select model type", list(MODEL_OPTIONS.keys()))
model_name = MODEL_OPTIONS[model_label]

model_file = st.sidebar.file_uploader("Upload matching model checkpoint (.pt)", type=["pt"])

st.sidebar.header("Images (batch supported)")
image_files = st.sidebar.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

show_gradcam = st.sidebar.toggle(
    "Show Grad-CAM (CNN models only)",
    value=True
)

# -------------------------------
# Preprocessing
# -------------------------------
# Note: Your ViT training used 0.5/0.5/0.5 normalization in your notebook.
# For simplicity and stability, we use ImageNet stats for all CNNs.
# For ViT, we use the 0.5 stats to match your training.
def get_transform(model_name: str):
    if model_name == "vit_b_16":
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

tfm = get_transform(model_name)

# -------------------------------
# Main logic
# -------------------------------
if not model_file or not image_files:
    st.info("Select a model type, upload its .pt file, then upload one or more images.")
    st.stop()

# Save uploaded model locally
MODEL_PATH = "temp_model.pt"
with open(MODEL_PATH, "wb") as f:
    f.write(model_file.read())

# Load model
try:
    model, target_layer = load_model(MODEL_PATH, model_name, num_classes=len(CLASSES))
except Exception as e:
    st.error("Failed to load the model. Make sure you selected the correct model type for the uploaded .pt file.")
    st.exception(e)
    st.stop()

# Build CAM engine only if supported and user wants it
cam_engine = None
can_cam = (target_layer is not None) and (model_name != "vit_b_16")
if show_gradcam and can_cam:
    try:
        cam_engine = GradCAM(target_layer)
    except Exception as e:
        st.warning("Grad-CAM could not be initialized for this model in the current environment.")
        st.exception(e)
        cam_engine = None

st.success(f"Loaded: {model_label}")

# Layout
left_col, right_col = st.columns([1, 1])

# -------------------------------
# Batch inference loop
# -------------------------------
st.header("Results")
grid = st.columns(2)

for idx, up in enumerate(image_files):
    try:
        img = Image.open(up).convert("RGB")
    except Exception:
        with grid[idx % 2]:
            st.error(f"Could not open image: {up.name}")
        continue

    img_tensor = tfm(img).unsqueeze(0)

    # -------------------------------
    # Prediction pass (NO gradients)
    # -------------------------------
    with torch.no_grad():
        logits_pred = model(img_tensor)
        probs = torch.softmax(logits_pred, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_name = CLASSES[pred_idx]
        pred_conf = float(probs[pred_idx])

    # Prepare display image for overlay
    img_disp = np.array(img.resize((224, 224)), dtype=np.float32) / 255.0

    # -------------------------------
    # Grad-CAM pass (WITH gradients)
    # -------------------------------
    cam_img = None
    if cam_engine is not None:
        try:
            with torch.enable_grad():
                model.zero_grad(set_to_none=True)

                # Fresh tensor for backward pass
                img_tensor_cam = img_tensor.clone().detach()
                logits_cam = model(img_tensor_cam)
                score = logits_cam[0, pred_idx]
                score.backward()

                cam = cam_engine.generate()
                cam_img = overlay_cam(img_disp, cam)
        except Exception as e:
            cam_img = None
            # Do not crash the whole app if one image CAM fails
            with grid[idx % 2]:
                st.warning(f"Grad-CAM failed for {up.name}. Prediction still shown.")
                st.exception(e)

    # -------------------------------
    # Render
    # -------------------------------
    with grid[idx % 2]:
        st.image(img, caption=up.name, use_column_width=True)
        st.markdown(
            f"""
            **Prediction:** {pred_name}  
            **Confidence:** {pred_conf * 100:.2f}%
            """
        )

        if model_name == "vit_b_16":
            st.info("Grad-CAM is not applicable to Vision Transformers. (Prediction shown only.)")
        elif cam_img is not None:
            st.image(cam_img, caption="Grad-CAM", use_column_width=True)
        elif show_gradcam:
            st.info("Grad-CAM not available for this run (prediction shown).")

# Cleanup
try:
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
except Exception:
    pass
