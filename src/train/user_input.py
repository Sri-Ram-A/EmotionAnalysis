import sys
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
from src import models
from src.utils import helper
from src.utils.paths import paths

MODEL_METHODS = {
    "rnn": ("1" , models.rnn)
}

def main():
    config = OmegaConf.load(paths.USER_CONFIG)
    run_number = helper.get_next_run_number(config.mlflow.experiment_name, config.mlflow.tracking_uri)
    MODEL_NAME = f"{config.dataset.name.lower()}_{config.model.architecture.lower()}_v{run_number}"
    MLFLOW_PERFORM = config.mlflow.perform.lower()
    RUN_NAME = f"{config.model.architecture.lower()}_{run_number}"
        
    if config.train.perform.lower() == "false":
        logger.warning("Skipping Training Pipeline")
        return
    
    # Minimal MLFlow logging only if enabled
    if MLFLOW_PERFORM == "true":
        import mlflow
        from mlflow.tensorflow import autolog , log_model
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)
        mlflow.start_run(run_name=RUN_NAME)
        autolog(
            log_models=True,
            log_input_examples=True,
            log_model_signatures=True
        )
        logger.info(f"MLFlow run started: {RUN_NAME} (Version {run_number})")
    
    # Load hdf5 dataset directly
    X, y = helper.load_h5(str(paths.DATA_H5_FILE))
    # Load tokenizer
    tokenizer = helper.load_tokenizer(str(paths.TOKENIZER_JSON_FILE))
    vocabulary_size = len(tokenizer.word_index) + 1
    timesteps = X.shape[1]
    total_classes = y.shape[1]
    embedding_dimension = config.model.embedding_dimension
    logger.info(f"Vocabulary size : {vocabulary_size} | Timesteps : {timesteps} | Model Embedding dimension : {embedding_dimension} | Classes : {total_classes}")
    
    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config.train.test_size, random_state=42)
    logger.info(f"Train size : {X_train.shape[0]} | Val size : {X_val.shape[0]}")
    
    # Train Model
    model = MODEL_METHODS[config.model.architecture][1].build_model(
        vocabulary_size, embedding_dimension, timesteps, total_classes
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.train.epochs,
        batch_size=config.train.batch_size
    )
    logger.success(f"Trained model for {config.train.epochs} epochs")

    # Save model in SavedModel format
    prod_num = MODEL_METHODS[config.model.architecture][0]
    model.export(paths.RECENT_MODEL_DIR / prod_num)
    logger.info(f"Model saved to: {paths.RECENT_MODEL_DIR / prod_num}")
    
    # End MLFlow run if enabled
    if MLFLOW_PERFORM == "true":
        mlflow.end_run()
        logger.info(f"MLFlow run completed - Run Name: {RUN_NAME}")

if __name__ == "__main__":
    main()