import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton,
                             QLineEdit, QFileDialog, QMessageBox)
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import numpy as np
from deepface import DeepFace
import cv2


# Initialize a VideoWriter to save the analyzed frames as a video file
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (640, 480))

def analyze_emotion(frame_or_path, is_video_frame=False):
    try:
        if is_video_frame:
            temp_img_path = 'temp_frame.jpg'
            cv2.imwrite(temp_img_path, frame_or_path)
            frame_or_path = temp_img_path

        results = DeepFace.analyze(img_path=frame_or_path, actions=['emotion'], enforce_detection=False)

        if is_video_frame:
            os.remove(temp_img_path)

        # Handle the possibility of multiple results being returned
        if isinstance(results, list):
            # Assuming we want the first result
            results = results[0]

        dominant_emotion = results.get('dominant_emotion', None)

        if not dominant_emotion:
            raise ValueError("Dominant emotion not found in the results.")

        if dominant_emotion in ["happy", "surprise"]:
            sentiment = "Positive"
        elif dominant_emotion in ["angry", "disgust", "fear", "sad"]:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return sentiment
    except Exception as e:
        print(f"An error occurred during emotion analysis: {e}")
        return "Error"


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    detected_sentiments = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        sentiment = analyze_emotion(frame, is_video_frame=True)
        detected_sentiments.append(sentiment)

        # Optional: Write the frame with detected sentiments to a video file
        out.write(frame)

    cap.release()
    out.release()

    # Calculate the average sentiment
    if detected_sentiments:
        sentiment_counts = {sentiment: detected_sentiments.count(sentiment) for sentiment in set(detected_sentiments)}
        average_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    else:
        average_sentiment = "Neutral"

    return average_sentiment






class InputValidator(QWidget):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("michellejieli/emotion_text_classifier")
        self.model = AutoModelForSequenceClassification.from_pretrained("michellejieli/emotion_text_classifier")

        self.setWindowTitle("Sentiment Analyser")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        self.label = QPushButton("Click here to enter text:", self)
        layout.addWidget(self.label)
        self.label.clicked.connect(self.show_text_input)

        # Picture Input
        self.picture_button = QPushButton("Select Picture", self)
        layout.addWidget(self.picture_button)
        self.picture_button.clicked.connect(self.get_picture_input)

        # Video Input
        self.video_button = QPushButton("Select Video", self)
        layout.addWidget(self.video_button)
        self.video_button.clicked.connect(self.get_video_input)

        self.setLayout(layout)

        self.text_input = QLineEdit(self)
        self.text_input.setPlaceholderText("Enter text and press Enter")
        self.text_input.returnPressed.connect(self.display_submission_confirmation)
        self.text_input.hide()
        layout.addWidget(self.text_input)

        self.select_pdf_button = QPushButton("Select PDF", self)
        self.select_pdf_button.hide()
        self.select_pdf_button.clicked.connect(self.process_pdf)
        layout.addWidget(self.select_pdf_button)

        self.select_document_button = QPushButton("Select Document", self)
        self.select_document_button.hide()
        self.select_document_button.clicked.connect(self.show_select_document)
        layout.addWidget(self.select_document_button)

        self.setLayout(layout)

    def show_text_input(self):
        self.text_input.show()
        self.select_document_button.show()
        self.label.hide()
        self.picture_button.hide()   # Hide the "Select Picture" button
        self.video_button.hide()     # Hide the "Select Video" button

    def show_select_document(self):
        file_dialog = QFileDialog()
        file_path, selected_filter = file_dialog.getOpenFileName(self, "Select a Document", "",
                                                                  "PDF Files (*.pdf);;Word Files (*.docx);;Text Files (*.txt)")
        if file_path:
            if selected_filter == "PDF Files (*.pdf)":
                self.process_pdf(file_path)
            elif selected_filter in ["Word Files (*.docx)", "Text Files (*.txt)"]:
                self.process_text_file(file_path)
        else:
            QMessageBox.warning(self, "Document Selection", "No document selected.")

    def process_pdf(self, file_path):
        if file_path:
            pdf_document = fitz.open(file_path)

            classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
            page_emotions_scores = []
            for page_number in range(len(pdf_document)):
                page = pdf_document.load_page(page_number)
                page_text = page.get_text()
                if page_text:
                    page_sentiments = []
                    for chunk in self.chunk_text(page_text, 500):
                        results = classifier(chunk)
                        page_sentiments.extend(results)

                    major_emotion, scaled_score = self.aggregate_page_emotions(page_sentiments)
                    if major_emotion is not None and scaled_score is not None:
                        page_emotions_scores.append((major_emotion, scaled_score))

            if page_emotions_scores:
                self.create_graph(page_emotions_scores, os.path.basename(file_path))
            else:
                QMessageBox.information(self, "PDF Sentiment Analysis", "No text found in the PDF.")

            pdf_document.close()

        else:
            QMessageBox.warning(self, "PDF Selection", "No PDF file selected.")

    def process_text_file(self, file_path):
        with open(file_path, 'r', encoding="utf-8") as file:
            content = file.read()

            classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
            results = classifier(content)

            major_emotion, scaled_score = self.aggregate_page_emotions(results)
            if major_emotion is not None and scaled_score is not None:
                page_emotions_scores = [(major_emotion, scaled_score)]

                self.create_graph(page_emotions_scores, os.path.basename(file_path))
            else:
                QMessageBox.information(self, "Document Sentiment Analysis", "No text found in the document.")

    def aggregate_page_emotions(self, sentiments):
        emotion_scores = {}
        for sentiment in sentiments:
            label = sentiment['label']
            score = sentiment['score']
            emotion_scores[label] = emotion_scores.get(label, 0) + score

        if not emotion_scores:  # Check if dictionary is empty
            return None, None  # Return None values if empty

        major_emotion = max(emotion_scores, key=emotion_scores.get)
        scaled_score = (emotion_scores[major_emotion] / len(sentiments)) * 100  # Scale from 0-1 to 0-100
        return major_emotion, scaled_score

    def chunk_text(self, text, max_tokens=500):
        words = text.split()
        chunk = ""
        for word in words:
            prospective_chunk = f"{chunk} {word}".strip()
            # Tokenize the prospective chunk and check its length
            tokenized_length = len(self.tokenizer.tokenize(prospective_chunk))
            if tokenized_length > max_tokens:
                yield chunk
                chunk = word
            else:
                chunk = prospective_chunk
        if chunk:
            yield chunk

    def create_graph(self, page_emotion_scores, pdf_name):
        emotion_colors = {
            'anger': 'red',
            'disgust': 'green',
            'fear': 'purple',
            'joy': 'yellow',
            'neutral': 'gray',
            'sadness': 'blue',
            'surprise': 'orange'
        }

        # Split the emotions and scores
        page_emotions, scores = zip(*page_emotion_scores)
        colors = [emotion_colors[emotion] for emotion in page_emotions]

        screen_res = QApplication.desktop().screenGeometry()
        screen_width, screen_height = screen_res.width(), screen_res.height()

        fig, ax = plt.subplots(figsize=(screen_width * 0.9 / 80, screen_height * 0.9 / 80))

        ax.bar(np.arange(len(page_emotions)), scores, color=colors, align='center')

        # Create a legend
        handles = [plt.Line2D([0], [0], color=emotion_colors[emotion], lw=4, label=emotion)
                   for emotion in emotion_colors.keys()]
        ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))

        # Adding dotted lines for y values
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

        ax.set_xticks(np.arange(len(page_emotions)))
        ax.set_xticklabels([str(i + 1) for i in range(len(page_emotions))])
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_ylim(0, 100)  # Assuming scores range from 0 to 100%

        ax.set_xlabel("Page Number")
        ax.set_ylabel("Emotion Intensity %")
        ax.set_title(f"Major Emotions by Page - {pdf_name}")

        plt.tight_layout()
        plt.show()

    def display_submission_confirmation(self):
        text = self.text_input.text()
        if text:
            classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
            result = classifier(text)
            emotion = result[0]['label']
            score = result[0]['score']
            scaled_score = score * 100  # Scale from 0-1 to 0-100
            QMessageBox.information(self, "Text Analysis Result", f"Emotion: {emotion}\nIntensity: {scaled_score:.2f}%", QMessageBox.Ok)
        else:
            QMessageBox.warning(self, "Empty Input", "Please enter some text before submitting.", QMessageBox.Ok)


    def get_picture_input(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Picture File", "", "Image Files (*.jpg *.png);;All Files (*)", options=options)

        if file_name:
            sentiment = analyze_emotion(file_name)
            QMessageBox.information(self, "Sentiment Analysis", f"Sentiment in the image is: {sentiment}")

    def get_video_input(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi);;All Files (*)", options=options)

        if file_name:
            average_sentiment = process_video(file_name)
            QMessageBox.information(self, "Average Sentiment Detected", f"Average sentiment detected in the video: {average_sentiment}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = InputValidator()
    window.show()
    sys.exit(app.exec_())
