import sys
import mlflow
from mlflow.tensorflow import load_model
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf
from argparse import ArgumentParser

BASE_DIR =  Path(__file__).resolve().parents[2]
SRC_DIR =  Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

ARTIFACTS_DIR = SRC_DIR / "artifacts"

def get_arguments():
    parser = ArgumentParser(description="Run training pipeline")
    parser.add_argument(
        "--config_path",
        type=str,
        default="params.yaml",
        help="Path to configuration file"
    )
    return parser.parse_args()

def main():
    args = get_arguments()
    config = OmegaConf.load(BASE_DIR / args.config_path)
    MODEL_NAME = f"{str(config.dataset.name).lower()}_{str(config.model.architecture).lower()}"
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    prod_model_uri = f"models:/{MODEL_NAME}@champion"
    prod_model = load_model(prod_model_uri)
    logger.success(f"Loaded Production Model")
    
    latest_model_uri = f"models:/{MODEL_NAME}@recent"
    latest_model = load_model(latest_model_uri)
    logger.success(f"Loaded Latest Model")

if __name__ == "__main__":
    main()
