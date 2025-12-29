import os
from loguru import logger
import subprocess
import requests
from urllib.parse import quote
from src.utils.paths import paths
from src.utils.paths import paths
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv(paths.BASE_DIR / ".env")

TF_SERVING_IMAGE = "tensorflow/serving:latest"
TEMP_CONTAINER = "serving_base"

def build_custom_tfx_image(image_name: str,model_name:str):
    """Builds a single-model TensorFlow Serving Docker image for."""
    try:
        model_path = paths.RECENT_MODEL_DIR / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"{model_name} SavedModel not found at: {model_path}")
        logger.info(f"Found {model_name} model at: {model_path}")

        # 0. Cleanup if container exists
        run(f"docker rm -f {TEMP_CONTAINER}", check=False)
        # 1. Start base TF Serving container
        run(f"docker run -d --name {TEMP_CONTAINER} {TF_SERVING_IMAGE}")
        # 2. Copy SavedModel
        run(f"docker cp {model_path} {TEMP_CONTAINER}:/models/{model_name}")
        # 3. Commit container → custom image
        run(f'docker commit --change "ENV MODEL_NAME {model_name}" {TEMP_CONTAINER} {image_name}')
        # 4. Cleanup
        run(f"docker kill {TEMP_CONTAINER}")
        run(f"docker rm {TEMP_CONTAINER}")

        logger.success(f"TF Serving image built: {image_name}")

    except Exception as e:
        logger.error(f"Failed to build TF Serving image: {e}")
        run(f"docker rm -f {TEMP_CONTAINER}", check=False)
        raise

def run(cmd, check=True):
    logger.info(f"Running: {cmd}")
    result = subprocess.run(cmd,shell=True,capture_output=True,text=True)
    # Print stdout if exists
    if result.stdout:
        print(result.stdout.strip())
    # Print stderr if exists
    if result.stderr:
        print(result.stderr.strip())
    # Check for errors if requested
    if check and result.returncode != 0:
        error_msg = f"Command failed: {cmd}"
        if result.stderr:
            error_msg += f"\nError: {result.stderr.strip()}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    return result.stdout

def push_to_dockerhub(IMAGE_NAME,REPO_NAME,VERSION_NO):
    logger.info("Running docker tag : ")
    run(f"docker tag {IMAGE_NAME} {REPO_NAME}:v{VERSION_NO}")
    logger.info("Pushing docker image to hub : ")
    run(f"docker push {REPO_NAME}:v{VERSION_NO}")
    logger.success("Pushed Image to DockerHUB")

def deploy_to_render(image_uri: str):
    """
    Trigger Render deployment of a Docker image.
    Parameters:
        image_uri (str): Full image URI like "docker.io/starmagiciansr/mlops-tfx:v1.0"
    """
    RENDER_DEPLOY_HOOK = os.getenv("RENDER_DEPLOY_HOOK")
    if not RENDER_DEPLOY_HOOK:
        raise ValueError("Set RENDER_DEPLOY_HOOK in your .env file!")
    # URL encode the image reference
    encoded_img = quote(image_uri, safe='')
    url = f"{RENDER_DEPLOY_HOOK}&imgURL={encoded_img}"
    logger.info(f"Triggering Render deploy → {url}")
    response = requests.get(url)
    if response.status_code == 200:
        logger.success("Render deployment triggered successfully!")
    else:
        logger.error(f"Render deploy failed: {response.status_code}, {response.text}")
