import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from model_utils import load_model, GradCAM, overlay_cam

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Citrus Fruit Classifier", layout="wide")
st.title("Citrus Fruit Classification with Grad-CAM")

CLASSES = ["murcott", "ponkan", "tangerine", "tankan"]

# -------------------------------
# Sidebar uploads
# -------------------------------
st.sidebar.header("Upload Model")
model_file = st.sidebar.file_uploader("Upload trained model (.pt)", type=["pt"])

st.sidebar.header("Upload Images")
image_files = st.sidebar.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# -------------------------------
# Main logic
# -------------------------------
if model_file and image_files:
    with open("temp_model.pt", "wb") as f:
        f.write(model_file.read())

    model = load_model("temp_model.pt", num_classes=len(CLASSES))

    # EfficientNet target layer
    target_layer = model.features[-1][0]
    cam_engine = GradCAM(model, target_layer)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    st.header("Batch Predictions")

    cols = st.columns(2)

    for i, image_file in enumerate(image_files):
        img = Image.open(image_file).convert("RGB")
        img_tensor = tfm(img).unsqueeze(0)

        # -------------------------------
        # Prediction pass (NO gradients)
        # -------------------------------
        with torch.no_grad():
            logits_pred = model(img_tensor)
            probs = torch.softmax(logits_pred, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))

        # -------------------------------
        # Grad-CAM pass (WITH gradients)
        # -------------------------------
        model.zero_grad(set_to_none=True)
        logits_cam = model(img_tensor)
        logits_cam[0, pred_idx].backward()
        cam = cam_engine.generate()

        img_np = np.array(img.resize((224, 224))) / 255.0
        cam_img = overlay_cam(img_np, cam)

        with cols[i % 2]:
            st.image(img, caption=image_file.name, use_column_width=True)
            st.markdown(
                f"""
                **Prediction:** {CLASSES[pred_idx]}  
                **Confidence:** {probs[pred_idx]*100:.2f}%
                """
            )
            st.image(cam_img, caption="Grad-CAM", use_column_width=True)

else:
    st.info("Please upload a model file and at least one image to begin.")
