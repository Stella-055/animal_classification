import cv2
import time
import requests
from collections import deque
from ultralytics import YOLO

# === CONFIG ===
MODEL_PATH = "yolov8n.pt"   # pretrained model (small & fast)
DEVICE_URL = "http://192.168.1.101/receive"
CONF_THRESHOLD = 0.5        # YOLO confidences are usually lower than your TF model
COOLDOWN = 5                # seconds between alerts

# === LOAD YOLO MODEL ===
model = YOLO(MODEL_PATH)

# === Define target classes ===
# From COCO: person, cow, sheep, dog, cat, horse, bird
TARGET_CLASSES = ["person", "cow", "sheep", "elephant", "bear", "zebra", "giraffe"]

# === History smoothing ===
preds_history = deque(maxlen=5)

# === Predict Single Image ===
def predict_image(path):
    results = model(path)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if label in TARGET_CLASSES:
                print(f"Predicted: {label} ({conf:.2%})")

                # --- If animal (not human), send alert ---
                if label != "person" and conf >= CONF_THRESHOLD:
                    try:
                        requests.get(f"{DEVICE_URL}?animal={label}&conf={conf:.2f}")
                        print(f"✅ Alert sent: {label} ({conf:.2%})")
                    except:
                        print("⚠️ Could not reach WiFi device")

# === Predict Frame (for webcam) ===
def predict_frame(frame):
    results = model(frame, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return None, 0.0

    # Pick the most confident target detection
    best_label, best_conf = None, 0.0
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        if label in TARGET_CLASSES and conf > best_conf:
            best_label, best_conf = label, conf

    # Smooth predictions
    preds_history.append((best_label, best_conf))
    if preds_history:
        # Choose the most frequent label in history
        labels = [p[0] for p in preds_history if p[0] is not None]
        if labels:
            final_label = max(set(labels), key=labels.count)
            final_conf = sum(p[1] for p in preds_history if p[0] == final_label) / labels.count(final_label)
            return final_label, final_conf

    return best_label, best_conf

# === Live camera ===
def run_camera():
    cap = cv2.VideoCapture(0)
    last_sent = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label, conf = predict_frame(frame)

        if label:
            # === Send to WiFi receiver if animal detected (with cooldown) ===
            if label != "person" and conf >= CONF_THRESHOLD:
                if time.time() - last_sent > COOLDOWN:
                    try:
                        requests.get(f"{DEVICE_URL}?animal={label}&conf={conf:.2f}")
                        print(f"✅ Alert sent: {label} ({conf:.2%})")
                        last_sent = time.time()
                    except:
                        print("⚠️ Could not reach WiFi device")

            # === Display ===
            cv2.putText(frame, f"{label} ({conf:.2%})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        cv2.imshow("YOLO Live Detection", frame)

        # Press "q" or "ESC" to quit
        if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()

# === Example usage ===
if __name__ == "__main__":
    # Option 1: predict static image
    # predict_image("MTU.jpeg")

    # Option 2: run webcam
    run_camera()
