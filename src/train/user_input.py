import sys
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
BASE_DIR =  Path(__file__).resolve().parents[2]
SRC_DIR =  Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src import models
from src.utils.helper import save_pkl , load_pkl

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
    
    # Load Preprocessed Data
    X, y = load_pkl(ARTIFACTS_DIR / "X.pkl") , load_pkl(ARTIFACTS_DIR / "y.pkl")
    logger.info(f"Loaded X shape : {X.shape} and y shape : {y.shape}")
    
    # Train-Test Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config.train.test_size, random_state=42)
    
    # Train Model
    model = MODEL_METHODS[config.model.architecture].build_model(config)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.train.epochs,
        batch_size=config.train.batch_size
    )
    logger.success(f"Trained model for {config.train.epochs} epochs")
    
    # Save the trained model
    model_path = ARTIFACTS_DIR / "trained_model_weights.h5"
    model.save(model_path)
    logger.info(f"Model saved to: {model_path}")

    # Save the complete model (architecture + weights + optimizer state)
    complete_model_path = ARTIFACTS_DIR / "production_model.keras"
    model.save(complete_model_path)
    logger.info(f"Complete model saved to: {complete_model_path}")
    
if __name__ == "__main__":
    main()
