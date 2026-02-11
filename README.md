# 🌿 Plant Disease Detection System  
**CNN + YOLO | Python | Computer Vision**

An end-to-end computer vision pipeline designed to detect and classify plant diseases under real-world image variability.  
The system combines **YOLO-based leaf localization** with **CNN-based disease classification** to improve robustness and reduce background noise.

---

## 🚀 Project Overview

Plant disease classification models often perform well on curated datasets but degrade in uncontrolled environments.  
This project explores **practical deployment challenges**, including:

- Domain shift (dataset vs live camera inputs)
- Background noise
- Image variability (lighting, resolution, angles)

---

## 🧠 System Architecture

**Pipeline Flow:**


<img width="512" height="600" alt="image" src="https://github.com/user-attachments/assets/70c1d75f-ab31-435e-8a3a-e75f0795abfa" />



**Key Components:**

✔ YOLO model for leaf region localization  
✔ CNN classifier for disease prediction  
✔ Preprocessing pipeline for inference stability  

---

## 🛠️ Technologies Used

- **Language:** Python  
- **Computer Vision:** OpenCV  
- **Deep Learning:** CNN, YOLO  
- **Libraries:** PyTorch / TensorFlow (update based on your implementation)  
- **Tools:** NumPy, Pandas  

---

## 📂 Dataset

Model trained and evaluated on:

**Dataset:** PlantVillage *(or specify your dataset)*  
**Classes:** XX plant disease categories  
**Images:** ~XXXX samples  

> ⚠ Performance differences observed between curated datasets and live camera inputs due to domain shift.

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | XX% |
| Test Accuracy | XX% |
| Observations | Accuracy drop under live camera inputs |

**Insights:**

- Identified misclassification clusters via confusion matrix analysis  
- Observed domain shift effects impacting generalization  
- Highlighted deployment risks in uncontrolled conditions  

---

## ⚙️ Installation

Clone repository:

```bash
git clone https://github.com/yourusername/Plant-Disease-Detection
cd Plant-Disease-Detection
