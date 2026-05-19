import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import joblib
import os
import sys
from scipy.io.wavfile import write

# --- Try to import sounddevice (only works locally, not on Streamlit Cloud) ---
try:
    import sounddevice as sd
    MICROPHONE_AVAILABLE = True
except OSError:
    MICROPHONE_AVAILABLE = False

# --- Path fix: works from any location ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(BASE_DIR, "..")
MODEL_PATH = os.path.join(REPO_DIR, "model", "best_emotion_model.pt")
ENCODER_PATH = os.path.join(REPO_DIR, "model", "label_encoder.pkl")
WAV_PATH = os.path.join(BASE_DIR, "live_input.wav")
SAMPLE_RATE = 16000
DURATION = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, os.path.join(REPO_DIR, "script"))
from model_cnn_lstm import CNNLSTMEmotionModel

@st.cache_resource
def load_encoder():
    return joblib.load(ENCODER_PATH)

@st.cache_resource
def load_model():
    enc = load_encoder()
    m = CNNLSTMEmotionModel(num_classes=len(enc.classes_))
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.to(DEVICE)
    m.eval()
    return m

label_encoder = load_encoder()
model = load_model()

def predict_emotion(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=64)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    x = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        output = model(x)
        prob = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(prob)
    return label_encoder.inverse_transform([pred_idx])[0], prob

def lmac_importance(y, baseline_prob):
    stride = int(0.1 * SAMPLE_RATE)
    win = int(0.2 * SAMPLE_RATE)
    pred_idx = np.argmax(baseline_prob)
    importance = []
    for start in range(0, len(y), stride):
        end = min(start + win, len(y))
        y_masked = y.copy()
        y_masked[start:end] = 0.0
        _, prob_masked = predict_emotion(y_masked)
        diff = baseline_prob[pred_idx] - prob_masked[pred_idx]
        importance.append(diff)
    return np.array(importance)

def process_audio(y):
    if len(y) < SAMPLE_RATE * DURATION:
        y = np.pad(y, (0, SAMPLE_RATE * DURATION - len(y)))
    else:
        y = y[:SAMPLE_RATE * DURATION]
    pred, prob = predict_emotion(y)

    emotion_emoji = {
        "anger": "😡", "disgust": "🤢", "fear": "😨",
        "happy": "😊", "neutral": "😐", "sad": "😢",
        "sarcastic": "😏", "surprise": "😲"
    }
    emoji = emotion_emoji.get(pred, "🎭")
    st.success(f"### {emoji} Predicted Emotion: **{pred.upper()}**")

    st.subheader("📊 Confidence Scores")
    prob_dict = {cls: float(p) for cls, p in zip(label_encoder.classes_, prob)}
    st.bar_chart(prob_dict)

    st.subheader("🌈 Log-Mel Spectrogram")
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=64)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.specshow(log_mel, sr=SAMPLE_RATE, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title("Log-Mel Spectrogram")
    st.pyplot(fig)
    plt.close()

    st.subheader("🔍 LMAC: Which part of audio triggered the emotion?")
    with st.spinner("Computing LMAC importance..."):
        importance = lmac_importance(y, prob)
    time_steps = np.linspace(0, DURATION, len(importance))
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.fill_between(time_steps, importance * 100, alpha=0.5, color="red")
    ax2.plot(time_steps, importance * 100, color="red", linewidth=2)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Importance (%)")
    ax2.set_title("Audio Segment Importance (LMAC)")
    st.pyplot(fig2)
    plt.close()

# --- UI ---
st.set_page_config(page_title="Speech Emotion Detector", page_icon="🎤")
st.title("🎤 Real-Time Speech Emotion Detection")
st.caption("CNN-LSTM Model | LMAC Interpretability")

tab1, tab2 = st.tabs(["🎙️ Record Audio", "📁 Upload Audio"])

with tab1:
    if not MICROPHONE_AVAILABLE:
        st.info(
            "🌐 **Running on Streamlit Cloud** — microphone recording is not available here.\n\n"
            "👉 Use the **Upload Audio** tab to analyse a .wav or .mp3 file.\n\n"
            "To use live recording, run the app locally with `streamlit run script/app.py`."
        )
    else:
        if st.button("🎙️ Record 3 Seconds & Predict"):
            with st.spinner("Recording..."):
                recording = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
                sd.wait()
                write(WAV_PATH, SAMPLE_RATE, recording)
            st.audio(WAV_PATH, format='audio/wav')
            y, _ = librosa.load(WAV_PATH, sr=SAMPLE_RATE)
            process_audio(y)

with tab2:
    st.write("Upload a speech audio file to detect the emotion.")
    uploaded = st.file_uploader("Choose a .wav or .mp3 file", type=["wav", "mp3"])
    if uploaded:
        tmp_path = os.path.join(BASE_DIR, "uploaded_audio.wav")
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())
        st.audio(tmp_path)
        y, _ = librosa.load(tmp_path, sr=SAMPLE_RATE)
        process_audio(y)
