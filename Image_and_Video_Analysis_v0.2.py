import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk


# Load a pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the list of emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize a VideoWriter to save the analyzed frames as a video file
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec as needed
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (640, 480))  # Change the output file name and parameters as needed


# Global variable to hold the loaded model
emotion_model = None


def create_emotion_model():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(7, activation='softmax'))

    return model

def load_emotion_model():
    global emotion_model
    emotion_model_path = 'C:/Users/dhars/OneDrive/Desktop/uwi/2007/stuff for 2007 code/emotion_analysis_using_deep_learning/model.h5'  # Update with the path to your model.h5 file
    emotion_model = create_emotion_model()
    emotion_model.load_weights(emotion_model_path)

# Call this function at the start of your program
load_emotion_model()

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) == 0:
        return "No Face Detected"  # No face detected, consider it as neutral

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]

        # Resize the face_roi to (48, 48) as the model expects this input size
        face_roi = cv2.resize(face_roi, (48, 48))

        # Normalize the pixel values to be between 0 and 1
        face_roi = face_roi / 255.0

        # Expand dimensions to match model input shape (add batch dimension)
        face_roi = np.expand_dims(face_roi, axis=0)
        # The model expects an extra dimension for the color channel, add it
        face_roi = np.expand_dims(face_roi, axis=-1)

        # Now use the loaded model to make predictions
        global emotion_model
        emotion_predictions = emotion_model.predict(face_roi)
        emotion_label = emotion_labels[np.argmax(emotion_predictions)]

        return emotion_label

# Function to process video frames and return the average detected emotion
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    detected_emotions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        emotion = detect_emotion(frame)
        detected_emotions.append(emotion)

        frame_count += 1

    cap.release()
    out.release()

    # Calculate the average emotion
    if detected_emotions:
        emotion_counts = {emotion: detected_emotions.count(emotion) for emotion in emotion_labels}
        average_emotion = max(emotion_counts, key=emotion_counts.get)
    else:
        average_emotion = "Neutral"

    return average_emotion

class InputValidator(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sentient Analyser")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

       

        # Picture Input
        self.picture_button = QPushButton("Select Picture", self)
        layout.addWidget(self.picture_button)
        self.picture_button.clicked.connect(self.get_picture_input)

        self.setLayout(layout)

        # Video Input
        self.video_button = QPushButton("Select Video", self)
        layout.addWidget(self.video_button)
        self.video_button.clicked.connect(self.get_video_input)

        

    def get_video_input(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi);;All Files (*)", options=options)
        
        if file_name:
            average_emotion = process_video(file_name)  # Process the video and calculate the average emotion
            QMessageBox.information(self, "Average Emotion Detected", f"Average emotion detected in the video: {average_emotion}")

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

    def show_text_input(self, event):
        self.label.hide()
        self.text_input.show()
        self.text_input.setFocus()

    def display_submission_confirmation(self):
        text = self.text_input.text()
        if text:
            # Perform sentiment analysis
            sia = SentimentIntensityAnalyzer()
            sentiment = sia.polarity_scores(text)
            compound_score = sentiment['compound']

            # Determine mood based on compound score
            if compound_score >= 0.05:
                mood = "Positive"
            elif compound_score <= -0.05:
                mood = "Negative"
            else:
                mood = "Neutral"

            QMessageBox.information(self, "Submission Confirmed", f"Submission confirmed. Mood of the text: {mood}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InputValidator()
    window.show()
    sys.exit(app.exec_())