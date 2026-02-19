from src import load_data, preprocess_data, tokenize_and_pad, build_bilstm_model
from src.train import train_model
from src.evaluate import evaluate_model

# Load
df = load_data("data/fake_or_real_news.csv")

# Preprocess
df = preprocess_data(df)

# Features
X, tokenizer = tokenize_and_pad(df['text'])
y = df['label'].values

# Model
model = build_bilstm_model()

# Train
model = train_model(model, X, y)

# Evaluate
evaluate_model(model, X, y)