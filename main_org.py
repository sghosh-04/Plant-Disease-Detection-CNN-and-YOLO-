import cv2
import torch
import time
import numpy as np
import warnings
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from model_loader import PlantDiseaseCNN

warnings.filterwarnings("ignore")

LEAF_MODEL_PATH = "best_leaf_only.pt"
DISEASE_CNN_MODEL = "plant_cnn_model.pt"

CLASS_NAMES = [
    "Apple_Apple_scab","Apple_Black_rot","Apple_Cedar_apple_rust","Apple_healthy",
    "Blueberry_healthy","Cherry_Powdery_mildew","Cherry_healthy",
    "Corn_Cercospora_leaf_spot Gray_leaf_spot","Corn_Common_rust",
    "Corn_Northern_Leaf_Blight","Corn_healthy",
    "Grape_Black_rot","Grape_Esca","Grape_Leaf_blight","Grape_healthy",
    "Orange_Haunglongbing","Peach_Bacterial_spot","Peach_healthy",
    "Pepper_bell_Bacterial_spot","Pepper_bell_healthy",
    "Potato_Early_blight","Potato_Late_blight","Potato_healthy",
    "Raspberry_healthy","Soybean_healthy",
    "Squash_Powdery_mildew","Strawberry_Leaf_scorch","Strawberry_healthy",
    "Tomato_Bacterial_spot","Tomato_Early_blight","Tomato_Late_blight",
    "Tomato_Leaf_Mold","Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites","Tomato_Target_Spot",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus","Tomato_healthy"
]

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def preprocess(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img)
    return img.unsqueeze(0)

print("Loading Models...")

# Apple GPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS GPU 🚀")
else:
    device = torch.device("cpu")
    print("Using CPU")

leaf_detector = YOLO(LEAF_MODEL_PATH)

# pytorch 2.6 fix
import torch.serialization
torch.serialization.add_safe_globals([PlantDiseaseCNN])

try:
    cnn_model = torch.load(DISEASE_CNN_MODEL, map_location=device, weights_only=False)
except:
    cnn_model = PlantDiseaseCNN()
    cnn_model.load_state_dict(torch.load(DISEASE_CNN_MODEL, map_location=device))

cnn_model.to(device)
cnn_model.eval()

print("Models Loaded Successfully")

cap = cv2.VideoCapture(0)
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------- YOLO LEAF DETECTION ----------
    results = leaf_detector(frame, conf=0.70, imgsz=512, verbose=False)

    # No boxes detected
    if results is None or results[0].boxes is None or len(results[0].boxes)==0:
        cv2.putText(frame,"NO LEAF DETECTED",(40,60),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

        cv2.imshow("System",frame)
        if cv2.waitKey(1)&0xFF==ord("q"):
            break
        continue

    leaf_found = False

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = results[0].names[cls_id]

        # ONLY leaf allowed
        if cls_name.lower() != "leaf":
            continue

        leaf_found = True

        x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
        cv2.putText(frame,"LEAF DETECTED",(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        # ---------- ROI ----------
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # ---------- LEAF MASK ----------
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower = np.array([15, 10, 10])
        upper = np.array([110, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.medianBlur(mask, 7)

        leaf_only = cv2.bitwise_and(roi, roi, mask=mask)

        leaf_ratio = np.sum(mask > 0) / mask.size

        # STRICT BLOCK – DO NOT RUN CNN
        if leaf_ratio < 0.30:
            print("Leaf detected but mask too weak — CNN BLOCKED")
            continue

        crop = leaf_only

        # ---------- lighting normalize ----------
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        cv2.imshow("Leaf Only View", crop)

        # ---------- CNN ----------
        print("CNN RUNNING...")

        tensor = preprocess(crop).to(device)

        with torch.no_grad():
            probs = torch.softmax(cnn_model(tensor), dim=1)[0]
            conf, idx = torch.max(probs,0)

        label = CLASS_NAMES[idx.item()]
        conf_val = float(conf)

        # ---------- decision ----------
        if "healthy" in label.lower():
            status = "HEALTHY"
            color = (0,255,0)
        elif conf_val < 0.80:
            status = "UNCERTAIN"
            color = (0,165,255)
        else:
            status = "DISEASED"
            color = (0,0,255)

        # ---------- TERMINAL ----------
        print("Prediction Label  :", label)
        print("Confidence        :", round(conf_val*100,2), "%")
        print("Final Status      :", status)
        print("-----------------------------------")

        # ---------- SCREEN ----------
        cv2.putText(
            frame,
            f"{status} ({conf_val:.2%})",
            (x1, max(30, y1-20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            3
        )

    # ---------- If no valid leaf ----------
    if not leaf_found:
        cv2.putText(frame,"NO VALID LEAF",(40,60),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

    # ---------- FPS ----------
    now = time.time()
    fps = 1/(now-prev_time)
    prev_time = now
    cv2.putText(frame,f"FPS: {fps:.1f}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    cv2.imshow("System",frame)
    if cv2.waitKey(1)&0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()