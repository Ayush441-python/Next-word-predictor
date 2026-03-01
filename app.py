import streamlit as st
from src.predict_pipeline import PredictionPipeline

# Load model once
@st.cache_resource
def load_pipeline():
    return PredictionPipeline()

pipeline = load_pipeline()

st.title("Next Word Predictor")
st.write("Word2Vec + LSTM based Language Model")

user_input = st.text_input("Enter a sentence:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        next_word = pipeline.predict_next_word(user_input)
        st.success(f"Predicted next word: {next_word}")