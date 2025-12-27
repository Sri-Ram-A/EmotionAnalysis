from pygments import highlight
from pygments.lexers import YamlLexer
from pygments.formatters import Terminal256Formatter
import yaml
import subprocess
import requests
from loguru import logger
import os
from pathlib import Path
from urllib.parse import quote
import sys
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv(BASE_DIR / ".env")
DEPLOY_HOOK = os.getenv("DEPLOY_HOOK")
def print_yaml(obj):
    """Pretty print YAML only if in DEBUG level"""
    if logger.level("DEBUG"):  # ensures print only if debug is active
        yaml_str = yaml.dump(obj, sort_keys=False, default_flow_style=False)
        formatter = Terminal256Formatter(style="monokai")
        logger.debug(highlight(yaml_str, YamlLexer(), formatter))


def run(cmd, check=True):
    logger.info(f"Running: {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )
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

def path_exists(path: Path, is_dir=False):
    if is_dir and not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not is_dir and not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    logger.success(f"Validated: {path}")
    
def push_to_dockerhub(IMAGE_NAME,VERSION_NO,REPO_NAME):
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
