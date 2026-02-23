import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import warnings
from model_loader import PlantDiseaseCNN


warnings.filterwarnings("ignore")

# =====================================================
# PATHS
# =====================================================
LEAF_MODEL_PATH = "best_leaf_only.pt"
DISEASE_MODEL_PATH = "plant_cnn_model.pt"

# =====================================================
# CLASS NAMES (CNN OUTPUT ORDER)
# =====================================================
CLASS_NAMES = [
    "Apple_Apple_scab", "Apple_Black_rot", "Apple_Cedar_apple_rust", "Apple_healthy",
    "Blueberry_healthy", "Cherry_Powdery_mildew", "Cherry_healthy",
    "Corn_Cercospora_leaf_spot Gray_leaf_spot", "Corn_Common_rust",
    "Corn_Northern_Leaf_Blight", "Corn_healthy",
    "Grape_Black_rot", "Grape_Esca", "Grape_Leaf_blight", "Grape_healthy",
    "Orange_Haunglongbing", "Peach_Bacterial_spot", "Peach_healthy",
    "Pepper_bell_Bacterial_spot", "Pepper_bell_healthy",
    "Potato_Early_blight", "Potato_Late_blight", "Potato_healthy",
    "Raspberry_healthy", "Soybean_healthy",
    "Squash_Powdery_mildew", "Strawberry_Leaf_scorch", "Strawberry_healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites", "Tomato_Target_Spot",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus", "Tomato_healthy"
]

# =====================================================
# PREPROCESSING FOR CNN
# =====================================================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img)
    return img.unsqueeze(0)

# =====================================================
# LOAD MODELS
# =====================================================
print("🔄 Loading models...")

# Device (MUST be before torch.load)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙ Using device: {device}")

# Leaf detector (YOLO)
leaf_detector = YOLO(LEAF_MODEL_PATH)

# Disease classifier (FULL MODEL FILE)
disease_model = torch.load(DISEASE_MODEL_PATH, map_location=device)
disease_model.to(device)
disease_model.eval()

print("✅ Models loaded successfully")

# =====================================================
# MAIN LOOP
# =====================================================
cap = cv2.VideoCapture(0)
print("📷 Camera started | Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------------------------
    # 1️⃣ LEAF DETECTION
    # ---------------------------
    leaf_results = leaf_detector(frame, conf=0.4, verbose=False)

    if leaf_results and leaf_results[0].boxes is not None:
        boxes = leaf_results[0].boxes.xyxy.cpu().numpy()

        for (x1, y1, x2, y2) in boxes.astype(int):
            leaf_crop = frame[y1:y2, x1:x2]
            if leaf_crop.size == 0:
                continue

            # ---------------------------
            # 2️⃣ DISEASE CLASSIFICATION
            # ---------------------------
            input_tensor = preprocess(leaf_crop).to(device)

            with torch.no_grad():
                probs = torch.softmax(disease_model(input_tensor), dim=1)[0]
                conf, idx = torch.max(probs, 0)

            label = CLASS_NAMES[idx.item()]
            status = "HEALTHY" if "healthy" in label.lower() else "DISEASED"

            color = (0, 255, 0) if status == "HEALTHY" else (0, 0, 255)

            # ---------------------------
            # DRAW RESULTS
            # ---------------------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{status} ({conf:.2%})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    cv2.imshow("Leaf → Disease Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
