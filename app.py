import tensorflow as tf
import cv2
import numpy as np
from keras.preprocessing.image import load_img, img_to_array


# Load model
IMG_SIZE = (128, 128)
model = tf.keras.models.load_model("best_model.h5")

# Define your labels from training
class_labels = ["Cows", "Goats", "Hippo", "Humans", "Hyenas", "Moneys"]


# Predict single image
def predict_image(path, class_labels):
    img = load_img(path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    idx = np.argmax(prediction)
    print(f"Predicted: {class_labels[idx]} ({prediction[0][idx]:.2%})")

# Live camera
def run_camera(model, class_labels):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.resize(frame, IMG_SIZE)
        img_array = np.expand_dims(img / 255.0, axis=0)
        prediction = model.predict(img_array)
        idx = np.argmax(prediction)
        label = class_labels[idx]
        conf = prediction[0][idx]
        cv2.putText(frame, f"{label} ({conf:.2%})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Live Camera Classification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Option 1: predict a static image
     predict_image("goat.jpeg", class_labels)

    # Option 2: run live webcam
   # run_camera(model, class_labels)
