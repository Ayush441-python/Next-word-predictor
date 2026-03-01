import os
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("notebook", "data", "cleaned_corpus.txt")
    processed_data_path: str = os.path.join("artifacts", "cleaned_corpus.txt")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def load_data(self):
        os.makedirs("artifacts", exist_ok=True)

        # Read plain text file
        with open(self.ingestion_config.raw_data_path, "r", encoding="utf-8") as f:
            corpus = f.read()

        # Save into artifacts
        with open(self.ingestion_config.processed_data_path, "w", encoding="utf-8") as f:
            f.write(corpus)

        print(f"Corpus saved at: {self.ingestion_config.processed_data_path}")

        return self.ingestion_config.processed_data_path


