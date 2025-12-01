import sys
import mlflow
from mlflow.tensorflow import load_model
from mlflow.tracking import MlflowClient

from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf
from argparse import ArgumentParser

BASE_DIR =  Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

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
    TRACKING_URI = config.mlflow.tracking_uri
    EXPERIMENT_NAME = config.mlflow.experiment_name
    mlflow.set_tracking_uri(TRACKING_URI)
    
    client = MlflowClient(TRACKING_URI)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found.")

    # Fetch latest FINISHED run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError("No finished runs found for this experiment.")

    latest_run = runs[0]
    latest_run_id = latest_run.info.run_id
    run_name = latest_run.info.run_name
    
    # Print key metrics prominently
    logger.success(f"Latest Run Name : {run_name} | Run ID : {latest_run_id}")
    metrics = latest_run.data.metrics
    logger.debug(f"Metrics : {metrics}")
    
    model_uri = Path(experiment._artifact_location) / "models" / str(latest_run.outputs.model_outputs[0].model_id) / "artifacts"
    loaded_model = mlflow.tensorflow.load_model(str(model_uri)) # Works
    logger.info(f"Model Loaded Successfully from : {model_uri}")

if __name__ == "__main__":
    main()