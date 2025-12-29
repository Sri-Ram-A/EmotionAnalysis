import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient
from src.utils.paths import paths
from src.utils.schema import Config
from src.utils.helper import print_yaml

STAGED_ALIAS = "staged"

def main():
    config = Config.load(paths.USER_CONFIG)
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)

    # Get Experiment
    client = MlflowClient()
    experiment = client.get_experiment_by_name(config.mlflow.experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{config.mlflow.experiment_name}' not found")
    logger.info("Using experiment: {}", experiment.name)
    print_yaml(experiment, config.registry.debug)

    # Ensure registered model exists
    try:
        client.get_registered_model(experiment.name)
    except Exception:
        logger.warning("Registered model not found. Creating one.")
        client.create_registered_model(experiment.name)

    # Fetch latest finished run
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError("No finished runs found")
    run = runs[0]
    run_id = run.info.run_id
    run_name = run.info.run_name
    latest_acc = run.data.metrics.get("val_accuracy", -1)
    logger.info("Latest run | name={} | acc={}", run_name, latest_acc)
    print_yaml(runs[0],config.registry.debug)
    model_uri = f"runs:/{run_id}/model"
    # Get model artifact path model_uri = str(Path(experiment._artifact_location) / "models" / str(latest_run.outputs.model_outputs[0].model_id))

    # Fetch staged model (if any)
    try:
        staged = client.get_model_version_by_alias(experiment.name, STAGED_ALIAS)
        if staged.run_id:
            staged_run = client.get_run(staged.run_id)
            staged_acc = staged_run.data.metrics.get("val_accuracy", -1)
            logger.info("Staged model | version={} | acc={}", staged.version, staged_acc)
    except Exception:
        staged, staged_acc = None, -1
        logger.warning("No staged model found")

    # Promote if better
    if latest_acc > staged_acc:
        versions = client.search_model_versions(f"run_id='{run_id}'")
        mv = versions[0] if versions else client.create_model_version(
            name=experiment.name,
            source=model_uri,
            run_id=run_id,
        )
        client.set_registered_model_alias(experiment.name, STAGED_ALIAS, mv.version)
        logger.success("Model staged | version={} | acc={}", mv.version, latest_acc)
    else:
        logger.info("Staged model remains unchanged")

if __name__ == "__main__":
    main()
