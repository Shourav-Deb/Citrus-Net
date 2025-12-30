import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from model_utils import load_model, GradCAM, overlay_cam

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Citrus Fruit Classifier", layout="centered")
st.title("🍊 Citrus Fruit Classification with Grad-CAM")

CLASSES = ["murcott", "ponkan", "tangerine", "tankan"]

# -------------------------------
# Upload model
# -------------------------------
st.header("1️⃣ Upload trained model (.pt)")
model_file = st.file_uploader("Upload model checkpoint", type=["pt"])

# -------------------------------
# Upload image
# -------------------------------
st.header("2️⃣ Upload image")
image_file = st.file_uploader("Upload fruit image", type=["jpg", "png", "jpeg"])

if model_file and image_file:
    # Save model temporarily
    with open("temp_model.pt", "wb") as f:
        f.write(model_file.read())

    model = load_model("temp_model.pt", num_classes=len(CLASSES))

    # Select target layer for Grad-CAM (EfficientNet)
    target_layer = model.features[-1][0]
    cam_engine = GradCAM(model, target_layer)

    # Image preprocessing
    img = Image.open(image_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img_tensor = tfm(img).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0].numpy()
        pred_idx = probs.argmax()

    st.subheader("🔍 Prediction")
    st.write(f"**Class:** {CLASSES[pred_idx]}")
    st.write(f"**Confidence:** {probs[pred_idx]*100:.2f}%")

    # Grad-CAM
    model.zero_grad()
    logits[0, pred_idx].backward()
    cam = cam_engine.generate(pred_idx)

    img_np = np.array(img.resize((224, 224))) / 255.0
    cam_img = overlay_cam(img_np, cam)

    st.subheader("🔥 Grad-CAM Visualization")
    st.image(cam_img, caption="Model Attention Map", use_column_width=True)
