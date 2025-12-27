import os
import mlflow
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from omegaconf import OmegaConf
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.entities import Run
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
from src.utils.paths import paths
import helper
import dockerize

def main():
    config = OmegaConf.load(paths.USER_CONFIG)
    TRACKING_URI = config.mlflow.tracking_uri
    EXPERIMENT_NAME = config.mlflow.experiment_name
    STAGED_ALIAS = "staged"
    PRODUCTION_ALIAS = "inprod"
    PROMOTE_TO_PROD = False
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    
    # Verify the model registry exists
    try:
        client.get_registered_model(EXPERIMENT_NAME)
        logger.info(f"Found registered model: '{EXPERIMENT_NAME}'")
    except Exception as e:
        logger.error(f"Registered model '{EXPERIMENT_NAME}' not found: {e}")
        sys.exit(1)
    
    # Try to get staged model
    try:
        staged: ModelVersion = client.get_model_version_by_alias(EXPERIMENT_NAME, STAGED_ALIAS)
        logger.info(f"Staged Model: version={staged.version}, run_id={staged.run_id}, status={staged.status}")
    except Exception as e:
        logger.error(f"No staged model exists - Run previous pipelines and save model. Error: {e}")
        sys.exit(1)
    
    # Get staged model run details
    try:
        staged_run = client.get_run(str(staged.run_id))
        staged_acc = staged_run.data.metrics.get("val_accuracy") or staged_run.data.metrics.get("accuracy") or -float("inf")
        logger.info(f"Staged Model - Run: {staged.run_id}, Accuracy: {staged_acc}")
    except Exception as e:
        logger.error(f"Failed to get staged model run details: {e}")
        sys.exit(1)

    # Try get production model
    try:
        prod: ModelVersion = client.get_model_version_by_alias(EXPERIMENT_NAME, PRODUCTION_ALIAS)
        logger.info(f"Production Model: version={prod.version}, run_id={prod.run_id}, status={prod.status}")
        prod_run: Run = client.get_run(str(prod.run_id))
        prod_acc = prod_run.data.metrics.get("val_accuracy") or prod_run.data.metrics.get("accuracy") or -float("inf")
        logger.info(f"Production Model - Run: {prod.run_id}, Accuracy: {prod_acc}")
        
        logger.info(f"Comparing: staged_acc={staged_acc} vs prod_acc={prod_acc}")
        if staged_acc > prod_acc:
            PROMOTE_TO_PROD = True
            logger.info("Staged model is better than production. Hence will promote.")
        else:
            logger.info("Production model already superior or equal. No changes made.")
    except Exception as e:
        logger.warning(f"No production model found: {e}. Promoting staged to production.")
        PROMOTE_TO_PROD = True

    if PROMOTE_TO_PROD:
        try:
            REPO_NAME = "starmagiciansr/mlops-tfx"
            VERSION_NO = f"{(float(staged.version)/10)}"
            IMAGE_NAME = f"tfserving/model:v{VERSION_NO}"

            logger.info("Building TF Serving Custom multi-model Docker Image 🐳")
            dockerize.build_custom_tfx_image(IMAGE_NAME)

            # Push to DockerHub
            helper.push_to_dockerhub(IMAGE_NAME, VERSION_NO, REPO_NAME)

            # 🔥 Trigger Render Deployment
            full_image_uri = f"docker.io/{REPO_NAME}:v{VERSION_NO}"
            logger.info(f"Triggering Render deployment for image: {full_image_uri}")
            helper.deploy_to_render(full_image_uri)

            # Set production alias
            client.set_registered_model_alias(EXPERIMENT_NAME, PRODUCTION_ALIAS, staged.version)
            logger.success(f"Promoted version = {staged.version} → PRODUCTION")

            # Remove staged alias
            client.delete_registered_model_alias(EXPERIMENT_NAME, STAGED_ALIAS)
            logger.success(f"Removed staged alias (version {staged.version} is now in production)")

        except Exception as e:
            logger.error(f"Failed to promote to production: {e}")
            sys.exit(1)

    else:
        logger.info("No promotion performed.")


if __name__ == "__main__":
    main()