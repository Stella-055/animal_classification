import tensorflow as tf
import cv2
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import requests

# Load model
IMG_SIZE = (128, 128)
model = tf.keras.models.load_model("best_model.h5")

# Define your labels from training
class_labels = ["Cows", "Goats", "Hippo", "Humans", "Hyenas", "Moneys"]
DEVICE_URL = "http://192.168.1.101/receive"
CONF_THRESHOLD = 0.85
# Predict single image
def predict_image(path, class_labels):
    img = load_img(path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    idx = np.argmax(prediction)
    print(f"Predicted: {class_labels[idx]} ({prediction[0][idx]:.2%})")
     # --- If animal, send to Wi-Fi ---
    if class_labels[idx] != "Humans" and prediction[0][idx] >= CONF_THRESHOLD:
        try:
            requests.get(f"{DEVICE_URL}?animal={class_labels[idx]}&conf={prediction[0][idx]:.2f}")
            print(f" Alert sent: {class_labels[idx]} ({prediction[0][idx]:.2%})")
        except:
            print(" Could not reach WiFi device")
# === Predict Single Frame ===
def predict_frame(frame, model, class_labels):
    img = cv2.resize(frame, IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_index]
    confidence = prediction[0][predicted_index]
    return predicted_label, confidence


# Live camera
def run_camera(model, class_labels):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label, conf = predict_frame(frame, model, class_labels)

        # === Send to WiFi receiver if animal detected ===
        if label != "Humans" and conf >= CONF_THRESHOLD:
            try:
                requests.get(f"{DEVICE_URL}?animal={label}&conf={conf:.2f}")
                print(f" Alert sent: {label} ({conf:.2%})")
            except:
                print(" Could not reach WiFi device")

        # Display
        cv2.putText(frame, f"{label} ({conf:.2%})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.imshow("Live Camera Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
# Example usage
if __name__ == "__main__":
    # Option 1: predict a static image
   # predict_image("MTU.jpeg", class_labels)

    # Option 2: run live webcam
    run_camera(model, class_labels)
