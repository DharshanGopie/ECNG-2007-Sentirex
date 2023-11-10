import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QPlainTextEdit, QDialog, QTextEdit)
from PyQt5.QtCore import Qt
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import fitz  # PyMuPDF
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
import cv2
import mplcursors

# Initialize a VideoWriter to save the analyzed frames as a video file
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (640, 480))

class InputValidator(QWidget):
    def __init__(self):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained("michellejieli/emotion_text_classifier")
        self.model = AutoModelForSequenceClassification.from_pretrained("michellejieli/emotion_text_classifier")

        self.setWindowTitle("Sentiment Analyser")
        self.setGeometry(100, 100, 400, 200)

        self.layout = QVBoxLayout(self)

        # Button configuration
        buttons = [
            ("Select Text", self.show_text_input, True),
            ("Select Picture", self.get_picture_input, True),
            ("Select Video", self.get_video_input, True),
            ("Back", self.show_original_state, False),
            ("Select PDF", self.show_select_PDF, False),
            ("Select Text File", self.show_select_text_file, False),
            ("Help", self.help_button, True)
        ]

        self.buttons = {}
    
        # Create buttons based on the configuration
        for text, slot, visible in buttons:
            button = QPushButton(text, self)
            button.clicked.connect(slot)
            self.layout.addWidget(button)
            button.setVisible(visible)
            # Store the button reference
            self.buttons[text] = button

        # Create the text input field
        self.text_input = QPlainTextEdit(self)
        self.text_input.setPlaceholderText("Enter text here")
        self.text_input.setVisible(False)
        self.layout.addWidget(self.text_input)

        # Add the "Submit Text" button after the text field
        submit_text_button = QPushButton("Submit Text", self)
        submit_text_button.clicked.connect(self.process_text)
        self.layout.addWidget(submit_text_button)
        submit_text_button.setVisible(False)
        self.buttons["Submit Text"] = submit_text_button

        self.showMaximized()
        self.setLayout(self.layout)

        
    def help_button(self):
        # Create the dialog window
        help_dialog = QDialog(self)
        # Set window flags to remove the non-functional '?' button
        help_dialog.setWindowFlags(help_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        help_dialog.setWindowTitle('Help Information')
        dialog_layout = QVBoxLayout()

        # Add a label with help text to the dialog
        help_label = QLabel("This application allows you to analyze sentiments from 3 forms of media: Text, Image or Video.\n\n"
                    "Select Text: To enter text manually in a text box for sentiment analysis.\n\n"
                    "Select Text File: To copy text from a text file for analysis.\n\n"
                    "Select PDF: To select a PDF document for page by page analysis.\n\n"
                    "Submit Text: To submit the entered text for analysis.\n\n"
                    "For Text Analysis the user is given the option of viewing the sentiments in a heatmap.\n"
                    "For text box inputs, it is broken up into a maximum of 10 parts and displayed on the heatmap.\n"
                    "For PDF input, it is displayed per page.\n\n"
                    "Select Picture: To choose a picture for analysis.\n\n"
                    "Select Video: To choose a video for analysis.\n\n"
                    "Back: Return to the main menu.\n\n"
                    )
        dialog_layout.addWidget(help_label)

        # Set the layout and display the dialog
        help_dialog.setLayout(dialog_layout)
        help_dialog.exec_()

    def show_original_state(self):
        # Iterate over all stored buttons and set their visibility
        for text, button in self.buttons.items():
            if text in ["Select Text", "Select Picture", "Select Video", "Help"]:
                button.show()
            else:
                button.hide()
                
        self.text_input.hide()

    def show_text_input(self):
        # Hide initial buttons by iterating over all QPushButton instances
        for button in self.findChildren(QPushButton):
            button.hide()
    
        # Show the text input, its submit button, and the PDF and Document buttons
        self.text_input.setVisible(True)
        # The submit button, PDF, and Document buttons can be identified by their text
        for button in self.findChildren(QPushButton):
            if button.text() in ["Back", "Submit Text", "Select PDF", "Select Text File"]:
                button.setVisible(True)

    def show_select_PDF(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select a Document", "", "PDF Files (*.pdf)")
        if file_path:
            # Process the PDF file
            self.process_pdf(file_path)
        else:
            QMessageBox.warning(self, "Document Selection", "No document selected.")
            
    def process_pdf(self, file_path):
        try:
            # Open the PDF file
            pdf_document = fitz.open(file_path)
        
            classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, return_all_scores=True)
            all_page_sentiments = []
        
            # Calculate the total word count of the PDF
            total_word_count = sum(len(page.get_text("text").split()) for page in pdf_document)

            # Alert the user if the PDF has more than 20000 words
            if total_word_count > 20000:
                QMessageBox.warning(self, "Large Document Detected", "This PDF has more than 20,000 words and may result in slower response times.", QMessageBox.Ok)
        
            # Proceed to analyze each page
            for page_number in range(len(pdf_document)):
                page = pdf_document.load_page(page_number)
                page_text = page.get_text("text")
            
                # Analyze the page
                page_sentiments = []
                for chunk in self.chunk_text(page_text, 500):
                    chunk_results = classifier(chunk)
                    for result in chunk_results:
                        page_sentiments.append(result)

                page_emotion_scores = self.aggregate_page_emotions(page_sentiments)
                all_page_sentiments.append(page_emotion_scores)

            # After processing, if any pages were analyzed
            if all_page_sentiments:
                user_choice = QMessageBox.question(self, "Output Type", "Do you want a heatmap output for the PDF?", QMessageBox.Yes | QMessageBox.No)

                if user_choice == QMessageBox.Yes:
                    self.page_or_paragraph_map(all_page_sentiments, is_aggregated=False, pdf_name=os.path.basename(file_path))
                else:
                    self.results_dialog = PDFResultsDialog(all_page_sentiments, self)
                    self.results_dialog.exec_()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while processing the PDF file: {e}", QMessageBox.Ok)

    def show_select_text_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select a Text File", "", "Text Files (*.txt)")
        if file_path:
            # Process the text file
            self.process_text_file(file_path)
        else:
            QMessageBox.warning(self, "Text File Selection", "No text file selected.")
            
    def process_text_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            # Set the text to the text input field which is used by process_text
            self.text_input.setPlainText(text)
            # Now call process_text to handle the processing
            self.process_text()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while processing the text file: {e}", QMessageBox.Ok)
            
    def process_text(self):
        text = self.text_input.toPlainText()

        # Check if the word count exceeds 20,000 words
        word_count = len(text.split())
        if word_count <= 0:
            QMessageBox.warning(self, "No Text Detected", "Please enter some text before submitting.", QMessageBox.Ok)
            return 

        # Show a warning if the word count exceeds the limit but still proceed with the analysis
        if word_count > 20000:
            QMessageBox.warning(self, "Input Limit Exceeded", "Your input exceeds the 20,000 words limit. The analysis may take longer.", QMessageBox.Ok)
    
        # Check if the user wants a heatmap after sentiment analysis
        user_choice = QMessageBox.question(self, "Output Type", "Do you want a heatmap output for the text?", QMessageBox.Yes | QMessageBox.No)
        heatmap_required = user_choice == QMessageBox.Yes
   
        if text:
            classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, return_all_scores=True)
            aggregated_scores = {}

            # Split text into manageable chunks for sentiment analysis
            for chunk in self.chunk_text(text, 500):
                chunk_results = classifier(chunk)
                for sentiment_set in chunk_results:
                    for sentiment in sentiment_set:
                        label = sentiment['label']
                        score = sentiment['score']
                        if label in aggregated_scores:
                            aggregated_scores[label].append(score)
                        else:
                            aggregated_scores[label] = [score]

            # If heatmap is not required, just display the averaged emotions
            if not heatmap_required:
                averaged_emotions_with_scores = ""
                for emotion, scores in aggregated_scores.items():
                    average_score = sum(scores) / len(scores) * 100
                    averaged_emotions_with_scores += f"Emotion: {emotion}, Intensity: {average_score:.2f}%\n"
                QMessageBox.information(self, "Text File Analysis Result", averaged_emotions_with_scores.strip(), QMessageBox.Ok)
                
            # Generate the heatmap if required
            if heatmap_required:
                # Split the text into 10 parts
                text_parts = self.split_into_parts(text, 10)
                heatmap_scores = {emotion: [] for emotion in aggregated_scores.keys()}

                for part in text_parts:
                    part_score = {}
                    part_text_chunks = self.chunk_text(part, 500)
      
                    # Perform sentiment analysis on each chunk within the part
                    for chunk in part_text_chunks:
                        chunk_results = classifier(chunk)
                        for sentiment_set in chunk_results:
                            for sentiment in sentiment_set:
                                label = sentiment['label']
                                score = sentiment['score']
                                part_score[label] = part_score.get(label, 0) + score

                    # Normalize the scores so that they sum up to 100 for each part
                    total_score = sum(part_score.values())
                    normalized_scores = {emotion: (score / total_score * 100) if total_score else 0 for emotion, score in part_score.items()}
                    for emotion in heatmap_scores:
                        heatmap_scores[emotion].append(normalized_scores.get(emotion, 0))

                # Pass the normalized scores to the heatmap generation method
                self.page_or_paragraph_map(heatmap_scores, "Text Sentiment Heatmap")
        else:
            # Display the averaged emotions in a message box
            averaged_scores = {emotion: (sum(scores) / len(scores)) for emotion, scores in aggregated_scores.items()}
            display_scores = "\n".join(f"Emotion: {emotion}, Intensity: {average_score:.2f}%" for emotion, average_score in averaged_scores.items())
            QMessageBox.information(self, "Text Analysis Result", display_scores.strip(), QMessageBox.Ok)
            
    def split_into_parts(self, text, num_parts):
        # Split the text into paragraphs
        paragraphs = text.split('\n\n')
        # Calculate the number of paragraphs per part
        part_size = max(1, len(paragraphs) // num_parts)
        parts = [paragraphs[i:i + part_size] for i in range(0, len(paragraphs), part_size)]
        # Make sure we always have num_parts parts, combine any extras with the last part
        while len(parts) > num_parts:
            parts[-2].extend(parts[-1])
            parts.pop()
        # Combine paragraphs within each part back into strings
        return ['\n\n'.join(part) for part in parts]

    def aggregate_page_emotions(self, sentiments):
        emotion_scores = {}
        for sentiment_set in sentiments:  # sentiments is now a list of sentiment sets
            for sentiment in sentiment_set:  # Iterate over each sentiment dictionary
                label = sentiment['label']
                score = sentiment['score']
                if label in emotion_scores:
                    emotion_scores[label] += score
                else:
                    emotion_scores[label] = score

        # Normalize the scores to get percentages
        total_scores = sum(emotion_scores.values())
        if total_scores == 0:
            return {emotion: 0 for emotion in emotion_scores}  # Avoid division by zero
        scaled_emotions = {emotion: (score / total_scores) * 100 for emotion, score in emotion_scores.items()}
        return scaled_emotions

    def chunk_text(self, text, max_tokens=500):
        words = text.split()
        chunk = ""
        for word in words:
            if len(self.tokenizer.tokenize(f"{chunk} {word}".strip())) > max_tokens:
                if chunk:
                    yield chunk
                chunk = word
            else:
                chunk = f"{chunk} {word}".strip()
        if chunk:
            yield chunk

    def page_or_paragraph_map(self, scores, is_aggregated=False, title='', pdf_name=None):
        # Determine the labels and the matrix from scores
        if is_aggregated:
            labels = sorted(scores.keys())
            matrix = np.array([scores[label] for label in labels])
            xlabel = 'Paragraph'
            title = title or 'Emotion Intensity Heatmap'
        else:
            labels = sorted(set(emotion for page_data in scores for emotion in page_data))
            matrix = np.array([[page_data.get(emotion, 0) for emotion in labels] for page_data in scores]).T
            xlabel = 'Page Number'
            title = title or f'Emotion Intensity Heatmap for {pdf_name}'

        ylabel = 'Emotions'

        # Create the heatmap with the given parameters
        self.create_heatmap(matrix, labels, title, xlabel, ylabel)

    def create_heatmap(self, data, labels, title, xlabel, ylabel):
        # Determine the size of the figure dynamically based on the x-axis length
        fig_width = max(10, len(data[0]) * 0.6) # Increase the base width if there are more columns
        fig_height = max(6, len(labels) * 0.3) # Keep the height relatively static or adjust as needed
    
        _, ax = plt.subplots(figsize=(fig_width, fig_height))
        cax = ax.imshow(data, cmap='viridis', interpolation='nearest', vmin=0, vmax=100)

        # Set labels for axes
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=8) # Adjust fontsize if necessary
        ax.set_xticks(np.arange(len(data[0])))
        ax.set_xticklabels(np.arange(1, len(data[0]) + 1), rotation=90, fontsize=8) # Adjust fontsize if necessary

        # Add a color bar with label
        cbar = ax.figure.colorbar(cax, ax=ax)
        cbar.ax.set_ylabel("Intensity (%)", rotation=-90, va="bottom")

        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=20) # Increase pad for the title if necessary

        # Enable interactive data cursor
        mplcursors.cursor(hover=True)

        # Adjust the layout to make room for the x-axis labels and title
        plt.subplots_adjust(bottom=0.2, top=0.85) # Adjust bottom if x-labels are cut off and top if title is cut off
    
        # Show the plot with a layout that fits both axes and title
        plt.show()

    def get_picture_input(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Picture File", "", "Image Files (*.jpg *.png);;All Files (*)", options=options)

        if file_name:
            sentiment = self.analyze_emotion(file_name)
            QMessageBox.information(self, "Sentiment Analysis", f"Sentiment in the image is: {sentiment}")

    def get_video_input(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi);;All Files (*)", options=options)

        if file_name:
            average_sentiment = self.process_video(file_name)
            QMessageBox.information(self, "Average Sentiment Detected", f"Average sentiment detected in the video: {average_sentiment}")
            
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        # Calculate video length in seconds
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        video_length = frames / fps
    
        # Warn the user if video length is longer than 10 seconds
        if video_length > 10:
            QMessageBox.warning(self, "Input Limit Exceeded", "Your video exceeds the 5 second limit. The analysis may take longer.", QMessageBox.Ok)
            
        detected_sentiments = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            sentiment = self.analyze_emotion(frame, is_video_frame=True)
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
    
    def analyze_emotion(self, frame_or_path, is_video_frame=False):
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
            
class PDFResultsDialog(QDialog):
    def __init__(self, all_page_sentiments, parent=None):
        super(PDFResultsDialog, self).__init__(parent)
        self.all_page_sentiments = all_page_sentiments
        self.current_page = 0

        self.page_label = QLabel(self)
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.prev_button = QPushButton("Previous", self)
        self.next_button = QPushButton("Next", self)

        layout = QVBoxLayout(self)
        layout.addWidget(self.page_label)
        layout.addWidget(self.text_edit)
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        layout.addLayout(nav_layout)

        self.prev_button.clicked.connect(lambda: self.change_page(-1))
        self.next_button.clicked.connect(lambda: self.change_page(1))

        self.display_page()

    def display_page(self):
        sentiments = self.all_page_sentiments[self.current_page]
        output_text = "\n".join(f"Emotion: {emotion}, Intensity: {score:.2f}%" for emotion, score in sentiments.items())
        self.text_edit.setText(output_text)
        self.page_label.setText(f"Page {self.current_page + 1} of {len(self.all_page_sentiments)}")
        self.prev_button.setEnabled(self.current_page > 0)
        self.next_button.setEnabled(self.current_page < len(self.all_page_sentiments) - 1)

    def change_page(self, direction):
        self.current_page = max(0, min(self.current_page + direction, len(self.all_page_sentiments) - 1))
        self.display_page()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = InputValidator()
    window.show()
    sys.exit(app.exec_())