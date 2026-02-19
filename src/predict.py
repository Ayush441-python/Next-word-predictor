from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.config import MAX_LEN

def predict_news(text, model, tokenizer):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    prediction = model.predict(padded)[0][0]
    return "Real" if prediction > 0.5 else "Fake"