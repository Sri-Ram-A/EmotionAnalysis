import mlflow
from loguru import logger
from pathlib import Path
from omegaconf import OmegaConf
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.entities import Run
BASE_DIR = Path(__file__).resolve().parents[2]
import sys
sys.path.append(str(BASE_DIR))
from src.utils.paths import paths
from helper import print_yaml , run_cmd

config = OmegaConf.load(paths.USER_CONFIG)
TRACKING_URI = config.mlflow.tracking_uri
EXPERIMENT_NAME = config.mlflow.experiment_name
STAGED_ALIAS = "staged"
PRODUCTION_ALIAS = "inprod"
PROMOTE_TO_PROD = False
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()
try:
    staged:ModelVersion = client.get_model_version_by_alias(EXPERIMENT_NAME,STAGED_ALIAS)
    logger.info("Staged:ModelVersion Info - Staged Model")
    print_yaml(staged.__dict__)
except Exception:
    logger.error("No staged model exists - Run previous pipelines and save model")
    sys.exit(1)
staged_run = client.get_run(str(staged.run_id))
staged_acc = staged_run.data.metrics.get("val_accuracy",-float("inf"))
logger.info("Staged:ModelVersion Info - Staged Model Run Details")
print_yaml(staged_run.__dict__)

# Try get production model
try:
    prod:ModelVersion = client.get_model_version_by_alias(EXPERIMENT_NAME, PRODUCTION_ALIAS)
    logger.info("Production:ModelVersion Info - Production Model")
    print_yaml(prod.__dict__)
    prod_run:Run = client.get_run(str(prod.run_id))
    prod_acc= prod_run.data.metrics.get("val_accuracy")
    logger.info("Staged:ModelVersion Info - Staged Model Run Details")
    print_yaml(prod_run.__dict__)
    
    logger.info(f"staged acc={staged_acc} vs prod acc={prod_acc}")
    if staged_acc > prod_acc:
        PROMOTE_TO_PROD = True
    else:
        logger.info("Production model already superior. No changes made.")
except Exception:
    logger.warning("No production model found. Promoting staged to production.")
    PROMOTE_TO_PROD = True

if PROMOTE_TO_PROD :
    
    IMAGE_NAME = "starmagiciansr/mlops-tfx"
    VERSION_TAG = f"v{(int(staged.version)/10)+1}"
    logger.info("Building TF Serving Custom multi-model Docker Image 🐳")
    run_cmd(f"docker build -t {IMAGE_NAME}:{VERSION_TAG} -t {IMAGE_NAME}:latest .")


    client.set_registered_model_alias(EXPERIMENT_NAME, PRODUCTION_ALIAS, staged.version)
    client.delete_registered_model_alias(EXPERIMENT_NAME , STAGED_ALIAS)
    logger.success(f"Promoted version = {staged.version} → PRODUCTION")
