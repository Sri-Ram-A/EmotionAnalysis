import sys
import mlflow
import numpy as np
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf
from argparse import ArgumentParser
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec
from sklearn.model_selection import train_test_split
BASE_DIR =  Path(__file__).resolve().parents[2]
SRC_DIR =  Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src import models
from src.utils import helper

ARTIFACTS_DIR = SRC_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
MODEL_METHODS = {
    "rnn" : models.rnn
}
def get_arguments():
    parser = ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument( # Config file
        "--config_path",
        type=str,
        default="params.yaml",
        help="Path to configuration file"
    )
    return parser.parse_args()

def main():
    args = get_arguments()
    config = OmegaConf.load(BASE_DIR / args.config_path)
    
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
    mlflow.enable_system_metrics_logging()
    
    # Print connection information
    logger.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    logger.info(f"Active Experiment: {mlflow.get_experiment_by_name(config.mlflow.experiment_name)}")
    
    # Load Preprocessed Data
    X, y = helper.load_pkl(ARTIFACTS_DIR / "X.pkl") , helper.load_pkl(ARTIFACTS_DIR / "y.pkl")
    logger.info(f"Loaded X shape : {X.shape} and y shape : {y.shape}")
    
    # Train-Test Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config.train.test_size, random_state=42)
    
    # Start run 
    run =  mlflow.start_run(run_name="Twitter-Padding-NoEmbed")
    mlflow.tensorflow.autolog() # type: ignore
    
    # Train Model
    model = MODEL_METHODS[config.model.architecture].build_model(config)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.train.epochs,
        batch_size=config.train.batch_size
    )
    logger.success(f"Trained model for {config.train.epochs} epochs")
    
    # Evaluate Model
    loss, accuracy = model.evaluate(X_val,y_val)
    logger.info(f"Test Loss: {loss} | Test Accuracy: {accuracy}")
        
    # # Save the trained model : Get input , output shape from the model
    input_shape , output_shape = model.input_shape , model.output_shape
    logger.info(f"Model input shape : {input_shape} |  output shape : {output_shape}")
    signature = helper.get_model_signature(input_shape , output_shape)
    # Log model with signature
    mlflow.tensorflow.log_model( # type: ignore
        model,
        "model",
        signature=signature,
        registered_model_name=f"{config.model.architecture}_{config.dataset.name}_model"
    )
    
    # model_path = ARTIFACTS_DIR / "trained_model_weights.h5"
    # model.save(model_path)
    # logger.info(f"Model saved to: {model_path}")

    # Save the complete model (architecture + weights + optimizer state)
    complete_model_path = ARTIFACTS_DIR / "production_model.keras"
    model.save(complete_model_path)
    logger.info(f"Complete model saved to: {complete_model_path}")
    mlflow.end_run()
    
if __name__ == "__main__":
    main()
