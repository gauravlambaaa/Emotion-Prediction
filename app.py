import streamlit as st
import numpy as np
import re
from nltk.stem import PorterStemmer
import pickle
import nltk

# ----------------------------- Setup -----------------------------
@st.cache_resource
def load_nltk_resources():
    nltk.download('stopwords')
    return set(nltk.corpus.stopwords.words('english'))

stopwords = load_nltk_resources()

# ------------------------ Load Trained Objects -------------------
@st.cache_resource
def load_model_and_tools():
    lg = pickle.load(open('logistic_regresion.pkl', 'rb'))
    tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    lb = pickle.load(open('label_encoder.pkl', 'rb'))
    return lg, tfidf_vectorizer, lb

lg, tfidf_vectorizer, lb = load_model_and_tools()

# ------------------------ Preprocessing --------------------------
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

# ------------------------- Prediction ----------------------------
def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])

    # Predict emotion and probability
    predicted_label = lg.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]

    probas = lg.predict_proba(input_vectorized)[0]
    confidence = np.max(probas)

    return predicted_emotion, confidence

# ------------------------- UI Setup ------------------------------
st.set_page_config(page_title="Emotion Detector", layout="centered")
st.title("🧠 Six Human Emotions Detector")
st.markdown("Detect the emotion behind any sentence!<br><br>**Supported Emotions**: `Joy`, `Fear`, `Anger`, `Love`, `Sadness`, `Surprise`", unsafe_allow_html=True)
st.markdown("---")

user_input = st.text_area("✏️ Enter your sentence below:", height=120)

if st.button("🔍 Predict Emotion"):
    if len(user_input.strip()) < 5:
        st.warning("⚠️ Please enter a longer sentence.")
    else:
        predicted_emotion, confidence = predict_emotion(user_input)
        emoji_map = {
            "joy": "😊", "fear": "😨", "anger": "😡",
            "love": "❤️", "sadness": "😢", "surprise": "😲"
        }
        emoji = emoji_map.get(predicted_emotion.lower(), "🙂")
        st.success(f"**Predicted Emotion**: {predicted_emotion.upper()} {emoji}")
        st.info(f"🔒 Confidence: `{confidence*100:.2f}%`")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
