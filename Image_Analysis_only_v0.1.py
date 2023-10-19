#v0.112/10/2023
import sys
import cv2
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox

# Load a pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the list of emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) == 0:
        return "Neutral"  # No face detected, consider it as neutral

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]

        # Resize the face_roi to (64, 64)
        face_roi = cv2.resize(face_roi, (64, 64))

        # Convert the grayscale image to RGB by replicating the single channel to three channels
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)

        # Expand dimensions to match model input shape (add batch dimension)
        face_rgb = np.expand_dims(face_rgb, axis=0)

        # Load the pre-trained emotion detection model from the "emotion_detection_model-master" folder
        emotion_model_path = 'C:/Users/dhars/OneDrive/Desktop/uwi/2007/stuff for 2007 code/emotion_detection_model-master/model.h5'
        emotion_model = tf.keras.models.load_model(emotion_model_path)

        emotion_predictions = emotion_model.predict(face_rgb)
        emotion_label = emotion_labels[np.argmax(emotion_predictions)]

        return emotion_label

class InputValidator(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sentient Analyser")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        # Picture input button
        self.picture_button = QPushButton("Select Picture", self)
        self.picture_button.clicked.connect(self.get_picture_input)
        layout.addWidget(self.picture_button)

        self.setLayout(layout)

    def get_picture_input(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Picture File", "", "Image Files (*.jpg *.png);;All Files (*)", options=options)

        if file_name:
            # Read the selected image file
            image = cv2.imread(file_name)

            if image is not None:
                emotion = detect_emotion(image)
                QMessageBox.information(self, "Emotion Analysis", f"Emotion in the image is: {emotion}")
            else:
                QMessageBox.warning(self, "Error", "Unable to read the selected image.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InputValidator()
    window.show()
    sys.exit(app.exec_())

