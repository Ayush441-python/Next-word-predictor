from src.data_ingestion import DataIngestion
from src.data_transformation import DataTransformation
from src.model_trainer import ModelTrainer


if __name__ == "__main__":


    ingestion = DataIngestion()
    raw_data_path = ingestion.load_data()

    transformation = DataTransformation()
    transformation.run()

    trainer = ModelTrainer()
    trainer.train()