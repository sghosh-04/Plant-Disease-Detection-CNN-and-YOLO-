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


<br>
<br>
📊 Model Details<br>
1. YOLO Model<br>
    Detects leaf regions with high precision<br>
    Filters background noise<br>
2. CNN Model<br>
    Trained on 38 plant disease classes<br>
    Includes healthy and diseased leaf categories<br>
    Achieved ~96% validation accuracy<br>
    
<br>
🌍 Applications<br>
Smart agriculture systems<br>
Drone-based crop monitoring<br>
Precision farming<br>
Early disease diagnosis<br>
Research and educational use<br>
<br>

⭐ Acknowledgements<br>
PlantVillage Dataset<br>
Ultralytics YOLO<br>
Open source deep learning community<br>
