# 🍊 Introducing **Citrus-Net**  

An end-to-end **image classification pipeline** that combines 🧱 **Custom CNN**, 🔁 **Transfer Learning**, and 🔭 **Vision Transformer** models — enhanced with 🧩 **Explainable AI (XAI)** visualizations and deployed in a 🌐 **Streamlit app**.  

---

![Project Banner](https://dummyimage.com/1200x400/000/fff&text=Citrus-Net+Project+Banner)  

---

## 📌 Repository Description  
This repository contains the complete project for building and explaining a full **image classification pipeline**. The pipeline moves from a hand-crafted CNN baseline to state-of-the-art deep learning models, enriched with **Explainable AI (XAI)** methods, and deployed as an interactive **Streamlit web app**.  



## 🚀 Project Stages  
- 🔎 **Data Preparation** – Exploratory analysis, class balance checks, preprocessing, and augmentations.  
- 🧱 **Custom CNN** – Novel convolutional network designed and trained from scratch.  
- 🔁 **Transfer Learning** – Fine-tuning four pretrained CNN backbones (ImageNet weights).  
- 🔭 **Vision Transformer (ViT)** – Transformer-based classifier for advanced performance.  
- 🧩 **XAI (Explainability)** – Applying Grad-CAM, Score-CAM, LIME, and SHAP to interpret predictions.  
- 🌐 **Streamlit App** – User-friendly interface for image upload, prediction, and interactive XAI visualizations.



## ✅ Key Features  
- 🧱 Custom CNN with ≥65% accuracy (baseline target).  
- 🔁 Transfer learning with four unique CNN architectures (not from demo code).  
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
We used the 🍊 [**Citrus Fruit Dataset**](https://data.mendeley.com/datasets/bxfgvsn9kw/6) containing high-quality images of citrus fruits for training, validation, and testing.  



## 📑 Deliverables  
- 📦 Public GitHub repo with runnable code.  
- 💾 Trained model weights (via Google Drive link after contacting).  
- 📄 Scientific project report (IEEE/ACM style).  
- 🌐 Streamlit demo folder.



## 🚀 Quick Start  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/Citrus-Net.git
cd Citrus-Net
```

### 2️⃣ Install Requirements
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App
```bash
streamlit run app.py
```
---

## 📸 XAI Heatmap
(Generated using Grad-CAM & LIME on random test images)

## 📧 Contact
If you need the trained models or datasets for academic purposes, feel free to [**Contact Anytime**](heyneeddev@gmail.com).

## ⭐ Acknowledgements
- Built with PyTorch, torchvision, scikit-learn, pytorch-grad-cam, LIME, and Streamlit.
- Inspired by cutting-edge research in Computer Vision & Explainable AI (XAI).


---


