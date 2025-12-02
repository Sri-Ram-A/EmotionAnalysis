import sys
from pathlib import Path
from loguru import logger
from helper import run

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
from src.utils.paths import paths

MODEL_PATHS = {
    "rnn": paths.RECENT_MODEL_DIR / "rnn",
    "lstm": paths.RECENT_MODEL_DIR / "lstm",
    "gru": paths.RECENT_MODEL_DIR / "gru",
}
CONFIG_FILE = paths.RECENT_MODEL_DIR / "model_config.config"
TEMP_CONTAINER = "temp_container"

FINAL_CONTAINER = "tfserving_multimodel"
TF_SERVING_IMAGE = "tensorflow/serving:latest"


def build_custom_tfx_image(FINAL_IMAGE):
    try:
        # Clean up any existing temp container first
        try:
            run(f"docker rm -f {TEMP_CONTAINER}", check=False)
            logger.info(f"Cleaned up existing {TEMP_CONTAINER}")
        except:
            pass
        # Clean up any existing final container
        try:
            run(f"docker rm -f {FINAL_CONTAINER}", check=False)
            logger.info(f"Cleaned up existing {FINAL_CONTAINER}")
        except:
            pass
        # 1. Run temporary container
        run(f"docker run -d --name {TEMP_CONTAINER} {TF_SERVING_IMAGE}")
        # 2. Copy only models that exist
        existing_models = {}
        for name, path in MODEL_PATHS.items():
            if path.exists():
                logger.info(f"Found model: {name} at {path}")
                run(f"docker cp {path} {TEMP_CONTAINER}:/models/{name}")
                existing_models[name] = path
            else:
                logger.warning(f"Model not found, skipping: {name} at {path}")
        if not existing_models:
            raise ValueError("No models found to package into Docker image!")
        logger.success(f"Successfully copied {len(existing_models)} model(s): {list(existing_models.keys())}")
        # 3. Copy config file (if exists)
        if CONFIG_FILE.exists():
            run(f"docker cp {CONFIG_FILE} {TEMP_CONTAINER}:/models/model_config.config")
            logger.info(f"Copied config file: {CONFIG_FILE}")
        else:
            logger.warning(f"Config file not found: {CONFIG_FILE}. Proceeding without it.")
        # 4. Commit container → final image
        run(
            f"docker commit "
            f"--change 'ENV MODEL_CONFIG_FILE /models/model_config.config' "
            f"{TEMP_CONTAINER} {FINAL_IMAGE}"
        )
        logger.success(f"Created custom multi-model TF Serving image: {FINAL_IMAGE}")
        # 5. Kill & remove temporary container
        run(f"docker kill {TEMP_CONTAINER}")
        run(f"docker rm {TEMP_CONTAINER}")
        logger.info(f"Cleaned up temporary container: {TEMP_CONTAINER}")
        # # 6. Optional: run the container immediately
        # run(
        #     f"docker run -d -p 8501:8501 "
        #     f"--name {FINAL_CONTAINER} "
        #     f"{FINAL_IMAGE} "
        #     f"--model_config_file=/models/model_config.config"
        # )
        # logger.success(
        #     f"Container '{FINAL_CONTAINER}' running on port 8501 with image: {FINAL_IMAGE}"
        # )
    except Exception as e:
        # Cleanup on failure
        logger.error(f"Failed to build Docker image: {e}")
        try:
            run(f"docker rm -f {TEMP_CONTAINER}", check=False)
            logger.info(f"Cleaned up temporary container after failure")
        except:
            pass
        raise