import streamlit as st
import numpy as np
import librosa
import pickle
import matplotlib.pyplot as plt
import io
from tensorflow.keras.models import load_model
import librosa.display

# -------------------------------
# Load trained model and label encoder
# -------------------------------
MODEL_PATH = "ser_model_bilstm_attention.h5"
ENCODER_PATH = "label_encoder.pkl"

model = load_model(MODEL_PATH)
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# -------------------------------
# Feature extraction (same as training)
# -------------------------------
def extract_features(file_obj, sr=16000, n_mfcc=40, max_len=200, mode="mfcc+delta"):
    # Convert UploadedFile to BytesIO if needed
    if hasattr(file_obj, "read"):
        audio_bytes = file_obj.read()
        file_obj = io.BytesIO(audio_bytes)

    # Load audio
    y, sr = librosa.load(file_obj, sr=sr)
    y, _ = librosa.effects.trim(y)

    # Features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    log_mel = librosa.power_to_db(mel)

    if mode == "mfcc":
        combined = mfcc
    elif mode == "mfcc+delta":
        combined = np.vstack([mfcc, delta])
    elif mode == "full":
        combined = np.vstack([mfcc, delta, delta2, log_mel])
    else:
        combined = log_mel

    # Pad or truncate
    if combined.shape[1] < max_len:
        combined = np.pad(combined, ((0,0),(0,max_len - combined.shape[1])), mode='constant')
    else:
        combined = combined[:, :max_len]

    return combined.T, y, sr

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎤 Speech Emotion Recognition (BiLSTM + Attention)")
st.write("Upload a `.wav` file to predict the emotion.")

uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

if uploaded_file is not None:
    # Play audio
    st.audio(uploaded_file, format="audio/wav")

    # Extract features
    features, signal, sr = extract_features(uploaded_file)
    x_input = np.expand_dims(features, axis=0)

    # Predict
    prediction = model.predict(x_input)
    predicted_class = np.argmax(prediction, axis=1)
    emotion = label_encoder.inverse_transform(predicted_class)[0]

    # Show result
    st.success(f"Predicted Emotion: **{emotion}**")

    # -------------------------------
    # Confidence bar chart
    # -------------------------------
    st.write("### Confidence Scores")
    emotions = list(label_encoder.classes_)
    scores = prediction[0]

    fig, ax = plt.subplots()
    colors = ["skyblue" if e != emotion else "green" for e in emotions]
    ax.bar(emotions, scores, color=colors)
    ax.set_ylabel("Confidence")
    ax.set_title("Prediction Confidence per Emotion")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # -------------------------------
    # Spectrogram preview
    # -------------------------------
    st.write("### Spectrogram Preview")
    S = librosa.feature.melspectrogram(y=signal, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.set_title("Mel Spectrogram")
    st.pyplot(fig)
