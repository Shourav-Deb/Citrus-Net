import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from model_utils import load_model, GradCAM, overlay_cam

st.set_page_config(page_title="Citrus Classifier", layout="wide")
st.title("Citrus Fruit Classification (All Models)")

CLASSES = ["murcott", "ponkan", "tangerine", "tankan"]

MODEL_OPTIONS = {
    "Custom CNN (Stage B)": "custom_cnn",
    "ResNet34 (TL)": "resnet34",
    "EfficientNet-B0 (TL)": "efficientnet_b0",
    "DenseNet121 (TL)": "densenet121",
    "ConvNeXt-Tiny (TL)": "convnext_tiny",
    "ViT-B/16 (Stage D)": "vit_b_16"
}

# ===============================
# Sidebar
# ===============================
model_label = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_OPTIONS.keys())
)
model_name = MODEL_OPTIONS[model_label]

model_file = st.sidebar.file_uploader(
    "Upload matching .pt model",
    type=["pt"]
)

image_files = st.sidebar.file_uploader(
    "Upload image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ===============================
# Main
# ===============================
if model_file and image_files:
    with open("temp_model.pt", "wb") as f:
        f.write(model_file.read())

    model, target_layer = load_model(
        "temp_model.pt",
        model_name,
        num_classes=len(CLASSES)
    )

    cam_engine = GradCAM(target_layer) if target_layer else None

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    cols = st.columns(2)

    for i, image_file in enumerate(image_files):
        img = Image.open(image_file).convert("RGB")
        img_tensor = tfm(img).unsqueeze(0)

        # Prediction
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))

        with cols[i % 2]:
            st.image(img, caption=image_file.name, use_column_width=True)
            st.markdown(
                f"""
                **Prediction:** {CLASSES[pred_idx]}  
                **Confidence:** {probs[pred_idx]*100:.2f}%
                """
            )

            # Grad-CAM only for CNNs
            if cam_engine:
                model.zero_grad(set_to_none=True)
                logits_cam = model(img_tensor)
                logits_cam[0, pred_idx].backward()
                cam = cam_engine.generate()

                img_np = np.array(img.resize((224, 224))) / 255.0
                cam_img = overlay_cam(img_np, cam)
                st.image(cam_img, caption="Grad-CAM", use_column_width=True)
            else:
                st.info("Grad-CAM is not applicable to Vision Transformers.")

else:
    st.info("Select a model, upload its .pt file, and upload images.")
