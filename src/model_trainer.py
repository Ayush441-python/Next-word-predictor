import os
import numpy as np
import pickle
from dataclasses import dataclass
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


@dataclass
class ModelTrainerConfig:
    sequence_data_path: str = os.path.join("artifacts", "sequence_data.npz")
    tokenizer_path: str = os.path.join("artifacts", "tokenizer.pkl")
    max_seq_len_path: str = os.path.join("artifacts", "max_seq_len.pkl")
    model_save_path: str = os.path.join("artifacts", "next_word_model.keras")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def load_data(self):
        data = np.load(self.config.sequence_data_path)
        X = data["X"]
        y = data["y"]

        with open(self.config.tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)

        vocab_size = len(tokenizer.word_index) + 1

        with open(self.config.max_seq_len_path, "rb") as f:
            max_seq_len = pickle.load(f)

        y = to_categorical(y, num_classes=vocab_size)

        return X, y, vocab_size, max_seq_len

    def build_model(self, vocab_size, max_seq_len):
        model = Sequential()
        model.add(Embedding(vocab_size, 100, input_length=max_seq_len-1))
        model.add(LSTM(150))
        model.add(Dropout(0.3))
        model.add(Dense(vocab_size, activation='softmax'))

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        return model

    def train(self):
        X, y, vocab_size, max_seq_len = self.load_data()

        model = self.build_model(vocab_size, max_seq_len)

        early_stop = EarlyStopping(
            monitor="loss",
            patience=2,
            restore_best_weights=True
        )

        model.fit(
            X,
            y,
            epochs=10,
            batch_size=256,
            callbacks=[early_stop]
        )

        model.save(self.config.model_save_path)

        print("Model training complete.")
        print("Model saved at:", self.config.model_save_path)

        return model
 