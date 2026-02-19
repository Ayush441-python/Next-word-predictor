import pickle
import os
from utils.config import BATCH_SIZE, EPOCHS

def train_model(model, X_train, y_train):
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2
    )

    return model


import pickle
import os

os.makedirs("model", exist_ok=True)

with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)





