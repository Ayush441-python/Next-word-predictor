import os
import pickle
import numpy as np
from dataclasses import dataclass
from tensorflow.keras.preprocessing.text import Tokenizer


@dataclass
class DataTransformationConfig:
    input_file_path: str = os.path.join("artifacts", "cleaned_corpus.txt")
    tokenizer_path: str = os.path.join("artifacts", "tokenizer.pkl")
    sequence_data_path: str = os.path.join("artifacts", "sequence_data.npz")
    max_seq_len_path: str = os.path.join("artifacts", "max_seq_len.pkl")
    max_tokens: int = 50000   # safer limit


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def load_corpus(self):
        with open(self.config.input_file_path, "r", encoding="utf-8") as f:
            corpus = f.read()

        words = corpus.split()[:self.config.max_tokens]
        return " ".join(words)

    def generate_sequences(self, corpus):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([corpus])

        total_words = len(tokenizer.word_index) + 1
        token_list = tokenizer.texts_to_sequences([corpus])[0]

        window_size = 5  # fixed context size

        sequences = []
        labels = []

        for i in range(window_size, len(token_list)):
            seq = token_list[i-window_size:i]
            label = token_list[i]

            sequences.append(seq)
            labels.append(label)

        X = np.array(sequences)
        y = np.array(labels)

        return X, y, tokenizer, window_size + 1, total_words

    def save_artifacts(self, X, y, tokenizer, max_seq_len):
        os.makedirs("artifacts", exist_ok=True)

        np.savez(self.config.sequence_data_path, X=X, y=y)

        with open(self.config.tokenizer_path, "wb") as f:
            pickle.dump(tokenizer, f)

        with open(self.config.max_seq_len_path, "wb") as f:
            pickle.dump(max_seq_len, f)

        print("Data transformation completed.")

    def run(self):
        corpus = self.load_corpus()
        X, y, tokenizer, max_seq_len, total_words = self.generate_sequences(corpus)
        self.save_artifacts(X, y, tokenizer, max_seq_len)

        print("Vocabulary size:", total_words)
        print("X shape:", X.shape)
        print("y shape:", y.shape)

        return X, y, total_words, max_seq_len

