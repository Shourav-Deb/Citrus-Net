# 🍊 Introducing **Citrus-Net**  

An end-to-end **image classification pipeline** that combines 🧱 **Custom CNN**, 🔁 **Transfer Learning**, and 🔭 **Vision Transformer** models - enhanced with 🧩 **Explainable AI (XAI)** visualizations and deployed in a **Streamlit app**.  

---

![Project Banner](https://i.pinimg.com/1200x/78/35/de/7835dec62d0c7a980cf778ee4f305504.jpg)  

---

## 📌 Repository Description  
This repository contains the complete project for building and explaining a full **image classification pipeline**. The pipeline moves from a hand-crafted CNN baseline to state-of-the-art deep learning models, enriched with **Explainable AI (XAI)** methods, and deployed as an interactive **Streamlit web app**.  



## 🚀 Project Stages  
- 🔎 **Data Preparation** – Exploratory analysis, class balance checks, preprocessing, and augmentations.  
- 🧱 **Custom CNN (CitrusNet)** – Novel convolutional network designed and trained from scratch.  
- 🔁 **Transfer Learning** – Fine-tuning four pretrained CNN backbones (ImageNet weights).  
- 🔭 **Vision Transformer (ViT)** – Transformer-based classifier for advanced performance.  
- 🧩 **XAI (Explainability)** – Applying Grad-CAM, Score-CAM, LIME, and SHAP to interpret predictions.  
- 🌐 **Streamlit App** – User-friendly interface for image upload, prediction, and interactive XAI visualizations.



## ✅ Key Features  
- 🧱 Custom Made CNN [CitrusNet] with ≥65% accuracy (baseline target).  
- 🔁 Transfer learning with four unique CNN architectures.  
- 🔭 Vision Transformer experiment.  
- 📊 Comparative evaluation (accuracy, precision, recall, F1, confusion matrices).  
- 🧩 XAI overlays on ≥10 test images with detailed interpretation.  
- 🌐 Streamlit app for model selection, predictions, and side-by-side explanations.  



## 🛠️ Tech Stack  
- **Python** (PyTorch, torchvision, scikit-learn, Captum, pytorch-grad-cam, LIME)  
- **Streamlit** for deployment  
- **Matplotlib / Seaborn** for plots  
- **Google Colab / Kaggle GPU** for training  



## 📂 Dataset  
- I used the [**Citrus Fruit Dataset**](https://data.mendeley.com/datasets/bxfgvsn9kw/6) containing high-quality images of citrus fruits for training, validation, and testing.  



## 📑 Deliverables  (After Contact)
- 📄 Scientific Project Report (IEEE/ACM style).
- 📦 6 Runnable Jupiter Notebook Code.  
- 💾 6 Trained Model Weights.
  
     - CitrusNet_[custom_cnn].pt
     - Efficientnet_B0.pt
     - Resnet34.pt
     - Densenet121.pt
     - Convnext_Tiny.pt
     - VIT_Best.pt 

## 🚀 Quick Start

### 🌐 Use the Web App

1. Open the live application [No installation required]  
   👉 https://citrus-net.streamlit.app

2. From the left sidebar:
   - Select a model:
     - CitrusNet [Custom]
     - EfficientNet-B0 [Best]
     - ResNet34
     - or Upload your own `.pt` model
   - Upload one or more citrus fruit images (`.jpg`, `.png`)

3. View the results:
   - Predicted fruit class
   - Confidence score
   - Grad-CAM visual explanations
   - Optional LIME explanations


### 💻 Run Locally (Optional)

```bash
git clone https://github.com/Shourav-Deb/Citrus-Net.git
cd Citrus-Net
pip install -r requirements.txt
streamlit run app.py
```
---

## 📧 Contact
If you need the trained notebook, models or report file for academic purposes, feel free to [**Contact Anytime**](mailto:heyneeddev@gmail.com).

## ⭐ Acknowledgements
- Built with PyTorch, torchvision, scikit-learn, pytorch-grad-cam, LIME, and Streamlit.
- Inspired by cutting-edge research in Computer Vision & Explainable AI (XAI).


---


