from dataclasses import dataclass
from pathlib import Path
from loguru import logger
@dataclass(frozen=True)
class ProjectPaths:
    """Project directory paths in capital letters"""
    
    # Base directories
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    SRC_DIR: Path = Path(__file__).resolve().parents[1]
    
    # Configuration
    USER_CONFIG: Path = BASE_DIR / "params.yaml"
    
    # Artifacts directories
    ARTIFACTS_DIR: Path = SRC_DIR / "artifacts"
    RECENT_MODEL_DIR: Path = ARTIFACTS_DIR / "recent_model"
    
    # Data files
    DATA_H5_FILE: Path = ARTIFACTS_DIR / "data.h5"
    TOKENIZER_JSON_FILE: Path = ARTIFACTS_DIR / "tokenizer.json"
    
    # Model files
    CONFIG_JSON_FILE: Path = RECENT_MODEL_DIR / "config.json"
    
    
# Create instance for easy import
paths = ProjectPaths()
# Ensure directories exist
paths.ARTIFACTS_DIR.mkdir(exist_ok=True)
paths.RECENT_MODEL_DIR.mkdir(exist_ok=True)