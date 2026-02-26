
## Speech Recognition and Emotion Detection

This project was collaboratively developed as part of our Final Year B.Tech Project.
Original development done along with:
Saurav Kumar Singh – https://github.com/SauravB210489CS/emotion-app

This project is a real-time **Vocal Emotion Detection Application** built using deep learning.  
It analyzes **Speech/Audio Input** and predicts the underlying human emotion using a Convolutional Neural Network (CNN).

The model extracts meaningful audio features (MFCC) and classifies the speaker's emotional state through a clean Streamlit web interface.

---

## Features

- Detects emotions from voice/audio input  
- CNN-based deep learning model  
- MFCC feature extraction  
- Streamlit web interface  
- Upload audio files for prediction  
- Fast and accurate results  

---

## Tech Stack

- Python  
- TensorFlow
- Librosa  
- NumPy  
- Streamlit  

---

## Project Structure

emotion-app/
│
├── model/ # Trained CNN model
├── app.py # Streamlit web app
├── requirements.txt # Required dependencies
└── README.md

---

## Installation & Setup

1. Clone the repository :-
  git clone https://github.com/SauravB210489CS/emotion-app.git cd emotion-app


2. Install dependencies :-
  pip install -r requirements.txt


3. Run the application :-
  streamlit run app.py


---

## Screenshots

screenshots/
├── home.png
├── upload.png
└── result.png

---

## Emotion Classes

- Angry  
- Calm  
- Happy  
- Sad  
- Fearful  
- Disgust  
- Neutral  

---

## Model Details

- CNN-based architecture  
- MFCC feature extraction  
- Trained on labeled speech dataset  
- High prediction accuracy  
- Optimized using TensorFlow  

---

## Future Improvements

- Real-time microphone input  
- Noise reduction pipeline  
- Multi-language emotion detection  
- Model optimization


---
## License

This project is for educational purposes only.
