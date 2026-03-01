import os
import pickle
import numpy as np
from dataclasses import dataclass
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


@dataclass
class PredictionConfig:
    model_path: str = os.path.join("artifacts", "next_word_model.keras")
    tokenizer_path: str = os.path.join("artifacts", "tokenizer.pkl")
    max_seq_len_path: str = os.path.join("artifacts", "max_seq_len.pkl")


class PredictionPipeline:
    def __init__(self):
        self.config = PredictionConfig()
        self.model = load_model(self.config.model_path)
        with open(self.config.tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)
        with open(self.config.max_seq_len_path, "rb") as f:
            self.max_seq_len = pickle.load(f)

    def predict_next_word(self, text):
        token_list = self.tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences(
            [token_list],
            maxlen=self.max_seq_len-1,
            padding='pre'
        )

        predicted_probs = self.model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs)

        for word, index in self.tokenizer.word_index.items():
            if index == predicted_index:
                return word

        return ""