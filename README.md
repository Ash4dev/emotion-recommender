# Emotion Recommender

The Emotion Recommender is a web application built using Streamlit that detects facial expressions and recommends songs based on the detected emotion. It leverages a pre-trained Convolutional Neural Network (CNN) model to analyze facial expressions and provides song recommendations through YouTube search.

## Features

- **Facial Expression Detection**: Uses a pre-trained model to classify facial expressions into emotions such as anger, disgust, fear, happy, neutral, sad, and surprise.
- **Snapshot Capture**: Allows users to capture a snapshot using their webcam.
- **Emotion-Based Song Recommendation**: Recommends songs based on the detected emotion by searching on YouTube.

## Installation

To run the application locally, you need to set up the environment and install the required dependencies.

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Ash4dev/emotion-recommender.git
   cd emotion-recommender
   ```
2. **Create and Activate a Virtual Environment**
  ```
  python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`
  ```
3. **Install Dependencies**
  ```
  pip install -r requirements.txt
  ```

## Running the Application
1. **Start the streamlit app**
     ```
     streamlit run app.py
     ```
2. Open your web browser and go to http://localhost:8501 to use the application.

## Files
- app.py: Main script that deploys the Streamlit application.
- weights/: Contains the weights of the trained models for facial expression detection.
- haarcascade_frontalface_default.xml: Pre-trained model for face detection.
- requirements.txt: Lists the Python packages required for the project.

## Requirements
Make sure you have the following installed:

Python 3.6+
Streamlit
OpenCV
TensorFlow
Pillow
NumPy

## How It Works
- Model Loading: The application loads a pre-trained CNN model (CNNModel_fer_3emo.h5) that classifies facial expressions into different emotions.
- Face Detection: Uses a Haar Cascade classifier to detect faces in the captured image.
- Emotion Prediction: Processes the detected face to predict the emotion.
- Snapshot Capture: Allows users to take a photo using their webcam and displays the emotion detected.
- Song Recommendation: Provides YouTube search results based on the detected emotion, language, and singer inputs from the user.

# Troubleshooting
- If you encounter issues with the face cascade, ensure that haarcascade_frontalface_default.xml is in the same directory as app.py and that it is properly downloaded.
- Make sure all dependencies are correctly installed and compatible with your Python version.
