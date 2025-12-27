from loguru import logger
from pathlib import Path
from helper import run
from src.utils.paths import paths

TF_SERVING_IMAGE = "tensorflow/serving:latest"
TEMP_CONTAINER = "serving_base"
MODEL_NAME = "rnn_classifier"

def build_custom_tfx_image(image_name: str):
    """Builds a single-model TensorFlow Serving Docker image for RNN."""
    try:
        rnn_model_path = paths.RECENT_MODEL_DIR / "rnn"
        if not rnn_model_path.exists():
            raise FileNotFoundError(f"RNN SavedModel not found at: {rnn_model_path}")
        logger.info(f"Found RNN model at: {rnn_model_path}")

        # Cleanup if container exists
        run(f"docker rm -f {TEMP_CONTAINER}", check=False)

        # 1. Start base TF Serving container
        run(f"docker run -d --name {TEMP_CONTAINER} {TF_SERVING_IMAGE}")

        # 2. Copy SavedModel
        run(f"docker cp {rnn_model_path} {TEMP_CONTAINER}:/models/{MODEL_NAME}")

        # 3. Commit container → custom image
        run(f'docker commit --change "ENV MODEL_NAME {MODEL_NAME}" {TEMP_CONTAINER} {image_name}')

        logger.success(f"TF Serving image built: {image_name}")

        # 4. Cleanup
        run(f"docker kill {TEMP_CONTAINER}")
        run(f"docker rm {TEMP_CONTAINER}")

    except Exception as e:
        logger.error(f"Failed to build TF Serving image: {e}")
        run(f"docker rm -f {TEMP_CONTAINER}", check=False)
        raise
