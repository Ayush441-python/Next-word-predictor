from src.data_ingestion import DataIngestion
from src.data_transformation import DataTransformation
from src.model_trainer import ModelTrainer


if __name__ == "__main__":

    # Step 1: Data Ingestion
    ingestion = DataIngestion()
    raw_data_path = ingestion.load_data()

    # Step 2: Data Transformation
    transformation = DataTransformation()
    transformation.run()

    # Step 3: Model Training
    trainer = ModelTrainer()
    trainer.train()