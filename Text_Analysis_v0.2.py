import sys
import os
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import fitz  # PyMuPDF

class InputValidator(QWidget):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("michellejieli/emotion_text_classifier")
        self.model = AutoModelForSequenceClassification.from_pretrained("michellejieli/emotion_text_classifier")

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
        self.select_pdf_button.show()
        self.select_document_button.show()
        self.label.hide()

    def show_select_document(self):
        QMessageBox.information(self, "Select Document", "Button is working")

    def process_pdf(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select a PDF file", "", "PDF Files (*.pdf)")
        if file_path:
            pdf_document = fitz.open(file_path)
            sentiments = []

            for page_number in range(len(pdf_document)):
                page = pdf_document.load_page(page_number)
                page_text = page.get_text()
                if page_text:
                    classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier") #need to only read 400 words
                    results = classifier(page_text)
                    sentiment = results[0]['label']
                    score = results[0]['score']
                    sentiments.append((sentiment, score))

            pdf_document.close()

            if sentiments:
                QMessageBox.information(self, "PDF Sentiment Analysis", "Sentiments for each page:\n" + "\n".join(
                    [f"Page {i+1}: Sentiment: {sentiment}, Score: {score:.2f}" for i, (sentiment, score) in
                     enumerate(sentiments)]))
            else:
                QMessageBox.information(self, "PDF Sentiment Analysis", "No text found in the PDF.")
        else:
            QMessageBox.warning(self, "PDF Selection", "No PDF file selected.")

    def display_submission_confirmation(self):
        text = self.text_input.text()
        if text:
            classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")
            results = classifier(text)
            sentiment = results[0]['label']
            score = results[0]['score']
            QMessageBox.information(self, "Submission Confirmed", f"Sentiment: {sentiment}, Score: {score:.2f}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InputValidator()
    window.show()
    sys.exit(app.exec_())
 
