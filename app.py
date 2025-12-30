import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from model_utils import load_model, GradCAM, overlay_cam

st.set_page_config(page_title="Citrus Fruit Classifier", layout="wide")
st.title("Citrus Fruit Classification with Grad-CAM")

CLASSES = ["murcott", "ponkan", "tangerine", "tankan"]

# -------------------------------
# Upload model
# -------------------------------
st.sidebar.header("Upload Model")
model_file = st.sidebar.file_uploader("Upload trained .pt model", type=["pt"])

# -------------------------------
# Upload images (BATCH)
# -------------------------------
st.sidebar.header("Upload Images")
image_files = st.sidebar.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if model_file and image_files:
    with open("temp_model.pt", "wb") as f:
        f.write(model_file.read())

    model = load_model("temp_model.pt", num_classes=len(CLASSES))
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

    for idx, image_file in enumerate(image_files):
        img = Image.open(image_file).convert("RGB")
        img_tensor = tfm(img).unsqueeze(0)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)[0].numpy()
            pred_idx = probs.argmax()

        model.zero_grad()
        logits[0, pred_idx].backward()
        cam = cam_engine.generate(pred_idx)

        img_np = np.array(img.resize((224, 224))) / 255.0
        cam_img = overlay_cam(img_np, cam)

        with cols[idx % 2]:
            st.image(img, caption=f"Input: {image_file.name}", use_column_width=True)
            st.markdown(
                f"""
                **Prediction:** {CLASSES[pred_idx]}  
                **Confidence:** {probs[pred_idx]*100:.2f}%
                """
            )
            st.image(cam_img, caption="Grad-CAM", use_column_width=True)
