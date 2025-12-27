# Plant-Disease-Detection (based on CNN and YOLO)

🌿 Plant Disease Detection System
An AI-powered computer vision system that detects plant leaf diseases using deep learning models. This project combines YOLO-based leaf detection and a CNN-based disease classifier to accurately identify diseases from images or live camera feeds.

📌 Overview
Plant diseases significantly impact agricultural productivity. Early and accurate detection is essential for timely intervention. This system automates the detection process by:
Identifying plant leaves using object detection
Classifying detected leaves into healthy or diseased categories
Supporting real time detection via webcam or image input
The system is designed to be modular, scalable, and deployable for real world agricultural applications.

🚀 Features
🌱 Automatic leaf detection using YOLO
🧠 Disease classification using Convolutional Neural Networks (CNN)
📷 Supports image input and live webcam detection
⚡ High accuracy with optimized preprocessing
🧩 Modular pipeline (easy to upgrade models)
🛠️ Can be integrated with drones or IoT systems

🧠 Tech Stack
Programming Language: Python
Deep Learning: PyTorch / TensorFlow
Object Detection: YOLO (Ultralytics)
Computer Vision: OpenCV
Data Processing: NumPy, PIL
Model Training: CNN (38-class plant disease dataset)

🏗️ System Architecture
Input Image / Webcam
        │
        ▼
YOLO Leaf Detection
        │
        ▼
Cropped Leaf Region
        │
        ▼
CNN Disease Classifier
        │
        ▼
Disease Prediction + Confidence

📊 Model Details
1. YOLO Model
    Detects leaf regions with high precision
    Filters background noise
2. CNN Model
    Trained on 38 plant disease classes
    Includes healthy and diseased leaf categories
    Achieved ~96% validation accuracy

🌍 Applications
Smart agriculture systems
Drone-based crop monitoring
Precision farming
Early disease diagnosis
Research and educational use

⭐ Acknowledgements
PlantVillage Dataset
Ultralytics YOLO
Open source deep learning community
