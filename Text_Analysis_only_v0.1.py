import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox, QMenu, QAction
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')


class InputValidator(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sentient Analyser")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        self.label = QPushButton("Click here to enter text:", self)
        layout.addWidget(self.label)
        self.label.clicked.connect(self.show_text_input)

        self.text_input = QLineEdit(self)
        self.text_input.setPlaceholderText("Enter text and press Enter")
        self.text_input.returnPressed.connect(self.display_submission_confirmation)
        self.text_input.hide()
        layout.addWidget(self.text_input)

        # Three additional buttons to prompt the user
        self.input_sentence_button = QPushButton("Input Sentence", self)
        self.input_sentence_button.hide()
        self.input_sentence_button.clicked.connect(self.show_input_sentence)
        layout.addWidget(self.input_sentence_button)

        self.select_pdf_button = QPushButton("Select PDF", self)
        self.select_pdf_button.hide()
        self.select_pdf_button.clicked.connect(self.show_select_pdf)
        layout.addWidget(self.select_pdf_button)

        self.select_document_button = QPushButton("Select Document", self)
        self.select_document_button.hide()
        self.select_document_button.clicked.connect(self.show_select_document)
        layout.addWidget(self.select_document_button)

        self.setLayout(layout)

    def show_text_input(self):
        self.input_sentence_button.show()
        self.select_pdf_button.show()
        self.select_document_button.show()
        self.label.hide()

    def show_input_sentence(self):
        self.text_input.show()

    def show_select_pdf(self):
        QMessageBox.information(self, "Select PDF", "Button is working")

    def show_select_document(self):
        QMessageBox.information(self, "Select Document", "Button is working")

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
 