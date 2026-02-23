import cv2
import time
import numpy as np
import warnings
import tensorflow.lite as tflite

warnings.filterwarnings("ignore")

# ---------------- PATHS ----------------
LEAF_MODEL_PATH = "best_leaf_only_float32.tflite"
DISEASE_CNN_MODEL = "model_fp32.tflite"

# ---------------- CLASSES ----------------
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
    "Squash_Powdery_mildew","Strawberry_Leaf_Scorch","Strawberry_healthy",
    "Tomato_Bacterial_spot","Tomato_Early_blight","Tomato_Late_blight",
    "Tomato_Leaf_Mold","Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites","Tomato_Target_Spot",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus","Tomato_healthy"
]

print("Loading TFLite models...")

# ---------------- YOLO TFLITE ----------------
leaf_interpreter = tflite.Interpreter(model_path=LEAF_MODEL_PATH)
leaf_interpreter.allocate_tensors()
leaf_in = leaf_interpreter.get_input_details()
leaf_out = leaf_interpreter.get_output_details()

YOLO_H = leaf_in[0]["shape"][1]
YOLO_W = leaf_in[0]["shape"][2]

print("Leaf YOLO TFLite loaded")
print("YOLO input size:", YOLO_W, "x", YOLO_H)

# ---------------- CNN TFLITE ----------------
cnn_interpreter = tflite.Interpreter(model_path=DISEASE_CNN_MODEL)
cnn_interpreter.allocate_tensors()
cnn_in = cnn_interpreter.get_input_details()
cnn_out = cnn_interpreter.get_output_details()

print("CNN TFLite loaded")
print("Models loaded successfully")

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0, cv2.CAP_ANY)

if not cap.isOpened():
    raise RuntimeError("❌ Webcam not detected")


def run_ai():
    

    ret, frame = cap.read()
    if not ret:
        return None

    H, W, _ = frame.shape

    # ---------- YOLO PREPROCESS ----------
    yolo_img = cv2.resize(frame, (YOLO_W, YOLO_H))
    yolo_img = cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB)
    yolo_img = yolo_img.astype(np.float32) / 255.0
    yolo_img = np.expand_dims(yolo_img, axis=0)

    leaf_interpreter.set_tensor(leaf_in[0]["index"], yolo_img)
    leaf_interpreter.invoke()

    detections = leaf_interpreter.get_tensor(leaf_out[0]["index"])[0]

    for det in detections:
        conf = det[4]
        if conf < 0.7:
            continue

        cx, cy, w, h = det[:4]

        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # ---------- MASK ----------
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (15, 10, 10), (110, 255, 255))
        mask = cv2.medianBlur(mask, 7)

        leaf_ratio = np.sum(mask > 0) / mask.size
        if leaf_ratio < 0.30:
            continue

        crop = cv2.bitwise_and(roi, roi, mask=mask)

        # ---------- CLAHE ----------
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(3.0, (8, 8))
        l = clahe.apply(l)
        crop = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        # ---------- CNN PREPROCESS ----------
        img = cv2.resize(crop, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        img = np.transpose(img, (0, 3, 1, 2))  # NHWC → NCHW


        cnn_interpreter.set_tensor(cnn_in[0]["index"], img)
        cnn_interpreter.invoke()
        probs = cnn_interpreter.get_tensor(cnn_out[0]["index"])[0]

        idx = int(np.argmax(probs))
        conf_val = float(probs[idx])
        label = CLASS_NAMES[idx]

        status = "HEALTHY" if "healthy" in label.lower() else "DISEASED"

        return {
            "status": status,
            "label": label,
            "confidence": conf_val
        }

    return None


# ---------------- STANDALONE RUN ----------------
if __name__ == "__main__":
    print("Running standalone AI inference...")
    while True:
        result = run_ai()
        if result:
            print(result)
        time.sleep(0.05)
