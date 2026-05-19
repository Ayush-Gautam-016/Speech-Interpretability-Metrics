# 🎤 Speech Emotion Detection with LMAC Interpretability

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://YOUR-APP-NAME.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CNN--LSTM-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)

> A deep learning system that detects human emotions from speech audio using a CNN-LSTM model, with LMAC (Local Model-Agnostic Contributions) interpretability to explain **which part of the audio** triggered the prediction.

---

## 🚀 Live Demo

👉 **[Try it here → YOUR-APP-NAME.streamlit.app](https://speech-interpretability-metrics.streamlit.app/)**

Upload any `.wav` or `.mp3` speech file and instantly see:
- The predicted emotion with confidence scores
- A log-mel spectrogram of your audio
- A LMAC importance graph showing exactly which moment triggered the emotion

---

## 🎯 Emotions Detected

| Emotion | Emotion | Emotion | Emotion |
|---------|---------|---------|---------|
| 😡 Anger | 🤢 Disgust | 😨 Fear | 😊 Happy |
| 😐 Neutral | 😢 Sad | 😏 Sarcastic | 😲 Surprise |

---

## 🧠 How It Works

```
Audio Input (.wav / mic)
        ↓
Log-Mel Spectrogram  (64 mel bands)
        ↓
CNN Feature Extractor  (2x Conv2D + BatchNorm + MaxPool)
        ↓
LSTM Sequence Modeller  (128 hidden units)
        ↓
Emotion Classifier  (8 classes)
        ↓
LMAC Interpretability  (sliding mask → importance scores)
```

---

## 📁 Project Structure

```
Speech-Interpretability-Metrics/
├── model/
│   ├── best_emotion_model.pt     # Trained CNN-LSTM weights
│   └── label_encoder.pkl         # Emotion label encoder
├── script/
│   ├── app.py                    # Streamlit web app
│   ├── model_cnn_lstm.py         # CNN-LSTM architecture
│   ├── train_model.py            # Training script
│   ├── evaluate_model.py         # Evaluation + confusion matrix
│   ├── data_preprocessing.py     # Audio loading + mel extraction
│   ├── extract_features.py       # Feature extraction pipeline
│   ├── lmac_interpret.py         # LMAC interpretability script
│   ├── real_time_predict.py      # CLI real-time prediction
│   └── spectrogram_plot.py       # Spectrogram visualisation
├── confusion_matrix_eval.png     # Model evaluation result
├── emotions.csv                  # Dataset labels
├── requirements.txt              # Python dependencies
└── packages.txt                  # System dependencies (PortAudio)
```

---

## ⚙️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Ayush-Gautam-016/Speech-Interpretability-Metrics.git
cd Speech-Interpretability-Metrics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run script/app.py
```

Open your browser at `http://localhost:8501`

---

## 📊 Model Performance

The model was evaluated on a held-out test set across 8 emotion classes.

![Confusion Matrix](confusion_matrix_eval.png)

---

## 🔍 What is LMAC?

LMAC (Local Model-Agnostic Contributions) is an interpretability technique that works by:

1. Taking the original audio and getting a baseline emotion prediction
2. Sliding a 200ms mask across the audio, silencing each segment one at a time
3. Measuring how much the prediction confidence **drops** when each segment is silenced
4. The segments where the drop is biggest = the most important parts of the audio

This makes the model **explainable** — you can see exactly which moment in the speech caused the emotion to be detected.

---

## 🛠️ Tech Stack

- **PyTorch** — CNN-LSTM model training and inference
- **Librosa** — audio loading and mel spectrogram extraction
- **Streamlit** — web application interface
- **Scikit-learn** — label encoding and evaluation metrics
- **Matplotlib** — spectrogram and LMAC visualisation

---

## 👥 Team

| Name | GitHub |
|------|--------|
| Ayush Gautam | [@Ayush-Gautam-016](https://github.com/Ayush-Gautam-016) |
| Saurav B | [@SauravB210489CS](https://github.com/SauravB210489CS) |

---

## 📄 License

This project is for academic purposes.
