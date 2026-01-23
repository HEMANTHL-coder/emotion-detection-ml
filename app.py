import streamlit as st
import pickle
import re
import numpy as np
import speech_recognition as sr
import tempfile

# ===================== EMOJI MAP ======================
emotion_emojis = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😲",
    "neutral": "😐",
    "stress": "😩",
    "disgust": "🤢",
    "boredom": "🥱",
    "love": "❤️",
    "gratitude": "🙏",
    "pride": "🏆",
    "hope": "🌈",
    "peace": "🕊️"
}

# ===================== UI STYLE ======================
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0f2027,#203a43,#2c5364); color:white;}
.title {font-size:40px;font-weight:bold;text-align:center;color:#00ffd5;}
.card {background:#1f2933;padding:25px;border-radius:15px;}
.emotion {background:#00c853;padding:15px;border-radius:10px;text-align:center;font-size:22px;}
.conf {background:#1565c0;padding:10px;border-radius:10px;text-align:center;margin-top:10px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🎤 Emotion Detection (Text + Voice)</div>", unsafe_allow_html=True)

# ===================== LOAD MODEL ======================
model = pickle.load(open("emotion_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# ===================== CLEAN TEXT ======================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    return text

# ===================== SPEECH TO TEXT ======================
def speech_to_text(audio_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio)
    except:
        return ""

# ===================== UI ======================
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("✍️ Enter text or speak")

    text_input = st.text_area("Type here:", height=120)

    audio_file = st.audio_input("🎤 Speak here")

    final_text = ""

    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_file.read())
            final_text = speech_to_text(f.name)
            st.success(f"🎙️ You said: {final_text}")

    if st.button("Predict Emotion"):
        if text_input.strip():
            final_text = text_input

        if final_text.strip() == "":
            st.warning("Please enter or speak something!")
        else:
            cleaned = clean_text(final_text)
            vector = tfidf.transform([cleaned])

            probs = model.predict_proba(vector)[0]
            emotion = model.classes_[np.argmax(probs)]
            confidence = np.max(probs) * 100

            emoji = emotion_emojis.get(emotion, "😐")

            st.markdown(
                f"<div class='emotion'>{emoji} Emotion: {emotion.upper()}</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='conf'>Confidence: {confidence:.2f}%</div>",
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)
