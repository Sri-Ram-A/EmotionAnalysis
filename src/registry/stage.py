import sys
import mlflow
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf
from mlflow.tracking import MlflowClient

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
from src.utils.paths import paths
from helper import print_yaml
def main():
    config = OmegaConf.load(paths.USER_CONFIG)
    TRACKING_URI = config.mlflow.tracking_uri
    EXPERIMENT_NAME = config.mlflow.experiment_name
    STAGED_ALIAS = "staged"
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(TRACKING_URI)
    
    # Fetch the registry
    try:
        client.get_registered_model(EXPERIMENT_NAME) # Ensure model registry entry exists
        logger.info(f"Registered model already exists - '{EXPERIMENT_NAME}'.")
    except Exception:
        logger.warning(f"Registered model not found. Creating it now - '{EXPERIMENT_NAME}'.")
        client.create_registered_model(EXPERIMENT_NAME)

    # Fetch the experiment
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment: 
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found.")
    logger.debug(f"Experiment Info: {experiment.__dict__}")
    
    # Fetch latest FINISHED run in the EXPERIMENT
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs: 
        raise ValueError("No finished runs found for this experiment.")
    
    latest_run = runs[0]
    LATEST_RUN_ID = latest_run.info.run_id
    LATEST_RUN_NAME = latest_run.info.run_name
    latest_acc = latest_run.data.metrics.get("val_accuracy") or latest_run.data.metrics.get("accuracy") or -float("inf")
    logger.success(f"Latest Run Name : {LATEST_RUN_NAME} | Run ID : {LATEST_RUN_ID} | Metrics : {latest_run.data.metrics}")
    
    # Get model artifact path
    model_uri = str(Path(experiment._artifact_location) / "models" / str(latest_run.outputs.model_outputs[0].model_id))

    # Try fetching staged model alias
    try:
        staged = client.get_model_version_by_alias(name=EXPERIMENT_NAME, alias=STAGED_ALIAS)
        logger.info(f"Found staged model version={staged.version}")
    except Exception:
        staged = None
        logger.warning("No staged alias found. Will register the current model as staged.")

    # If no staged model exists → stage the latest
    if staged is None:
        mv = client.create_model_version(
            name=EXPERIMENT_NAME,
            source=model_uri,
            run_id=LATEST_RUN_ID,
        )
        client.set_registered_model_alias(EXPERIMENT_NAME, STAGED_ALIAS, mv.version)
        logger.success(f"MODEL STAGED: version = {mv.version} with accuracy = {latest_acc}")

    else:
        # Get staged model accuracy
        staged_run = client.get_run(str(staged.run_id))
        staged_acc = staged_run.data.metrics.get("val_accuracy") or staged_run.data.metrics.get("accuracy") or -float("inf")
        logger.info(f"Staged model version={staged.version} | acc={staged_acc}")

        # Compare & promote if better
        if latest_acc > staged_acc:
            logger.success(f"Latest model is better (acc={latest_acc} > {staged_acc}). Promoting to staged.")
            
            # Check if this run already has a model version
            existing_versions = client.search_model_versions(f"run_id='{LATEST_RUN_ID}'")
            
            if existing_versions:
                # Use existing model version
                mv_version = existing_versions[0].version
                logger.info(f"Model version already exists for this run: version={mv_version}")
            else:
                # Create new model version
                mv = client.create_model_version(
                    name=EXPERIMENT_NAME,
                    source=model_uri,
                    run_id=LATEST_RUN_ID,
                )
                mv_version = mv.version
                logger.info(f"Created new model version: {mv_version}")
            
            # Update the staged alias to point to this version
            client.set_registered_model_alias(EXPERIMENT_NAME, STAGED_ALIAS, mv_version)
            logger.success(f"Updated staged alias → version {mv_version}")
        else:
            logger.info(f"Staged model is already better or equal (staged_acc={staged_acc} >= latest_acc={latest_acc}). Keeping current staging unchanged.")


if __name__ == "__main__":
    main()