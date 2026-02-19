import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.text_cleaner import clean_text
from utils.config import MAX_LEN

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Load model
@st.cache_resource
def load_artifacts():
    model = load_model("model/model.h5")
    with open("model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_artifacts()

# UI
st.title("📰 Fake News Detection System")
st.write("Enter a news article to check whether it is **Fake or Real**.")

news_text = st.text_area("Paste news content here", height=200)

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_text = clean_text(news_text)
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=MAX_LEN)

        prob = model.predict(padded)[0][0]

        if prob > 0.5:
            st.success(f"✅ Real News (Confidence: {prob:.2f})")
        else:
            st.error(f"❌ Fake News (Confidence: {1 - prob:.2f})")
