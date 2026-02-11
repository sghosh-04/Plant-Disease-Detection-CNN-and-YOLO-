# 🌿 Plant Disease Detection System  
**Binary CNN Classifier + YOLOv8n Detector | Python | PyTorch | Computer Vision**

An end-to-end computer vision system designed to detect plant diseases under real-world variability.  
The pipeline combines a **lightweight CNN-based binary classifier (Healthy vs Diseased)** with a **YOLOv8n-based detection model**.

---

## 🚀 Project Overview

Traditional plant disease classifiers perform well on curated datasets but degrade in uncontrolled environments.  
This project focuses on:

- Robust classification under domain shift
- Efficiency for edge deployment
- Drone-specific visual variability
- Practical evaluation & failure analysis

--

## 🧠 System Architecture

**Pipeline Flow:**


<img width="512" height="600" alt="image" src="https://github.com/user-attachments/assets/70c1d75f-ab31-435e-8a3a-e75f0795abfa" />


---

## 🛠️ Technologies Used

- **Language:** Python  
- **Framework:** PyTorch  
- **Detection Model:** YOLOv8n (Ultralytics)  
- **Computer Vision:** OpenCV  
- **Augmentation:** Albumentations, TorchVision  
- **Evaluation:** Scikit-learn  

---

## 📂 Dataset

**Source:** Kaggle  
**Dataset:** New Plant Diseases Dataset (PlantVillage)  
**Original Classes:** 38  
**Converted To:** Binary Classification  

**Final Classes:**
- Healthy
- Diseased

---

## 🧠 CNN Model

Custom lightweight CNN (**FastBinaryPlantDetector**) designed for:

✔ Low computational overhead  
✔ Faster inference  
✔ Suitability for edge devices  

**Training Configuration:**

- Loss Function: BCELoss  
- Optimizer: AdamW  
- Scheduler: ReduceLROnPlateau  
- Epochs: 15  

---

## 📊 Evaluation Metrics

Model performance evaluated using:

✔ Accuracy  
✔ Precision  
✔ Recall  
✔ F1-score  
✔ Confusion Matrix  
✔ ROC Curve & AUC  
✔ Threshold Optimization  

> Observed performance degradation under live camera inputs due to domain shift.

---

## 🚁 YOLOv8n Detector

YOLO model trained for:

✔ Disease spot localization  
✔ Drone-specific augmentations  
✔ Robust detection under rotation & scale variations  
---

## ⚙️ Installation

```bash
git clone https://github.com/sghosh-04/Plant-Disease-Detection-CNN-and-YOLO-
cd Plant-Disease-Detection-CNN-and-YOLO-
pip install -r requirements.txt
