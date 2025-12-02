from dataclasses import dataclass
from pathlib import Path
from loguru import logger
@dataclass(frozen=True)
class ProjectPaths:
    """Project directory paths in capital letters"""
    
    # Base directories
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    SRC_DIR: Path = Path(__file__).resolve().parents[1]
    
    # Artifacts directories
    ARTIFACTS_DIR: Path = SRC_DIR / "artifacts"
    RECENT_MODEL_DIR: Path = ARTIFACTS_DIR / "recent_model"
    
    # Data files
    DATA_H5_FILE: Path = ARTIFACTS_DIR / "data.h5"
    TOKENIZER_JSON_FILE: Path = ARTIFACTS_DIR / "tokenizer.json"
    
    # Model files
    CONFIG_JSON_FILE: Path = RECENT_MODEL_DIR / "config.json"
    
    # Configuration
    USER_CONFIG: Path = BASE_DIR / "params.yaml"
    
# Create instance for easy import
paths = ProjectPaths()
# Ensure directories exist
paths.ARTIFACTS_DIR.mkdir(exist_ok=True)
paths.RECENT_MODEL_DIR.mkdir(exist_ok=True)

# Create model directories properly
for dir_name in ["rnn", "lstm", "gru"]:
    model_dir = paths.RECENT_MODEL_DIR / dir_name / "1"
    model_dir.mkdir(parents=True, exist_ok=True)

model_config_path = paths.RECENT_MODEL_DIR / "model_config.config"
if not model_config_path.exists():
    content = """
    model_config_list {
    config {
        name: 'rnn'
        base_path: '/models/rnn/'
        model_platform: 'tensorflow'
    }
    config {
        name: 'lstm'
        base_path: '/models/lstm/'
        model_platform: 'tensorflow'
    }
    config {
        name: 'gru'
        base_path: '/models/gru/'
        model_platform: 'tensorflow'
    }
}
    """.strip()
    # Write to file
    with open(model_config_path, "w") as f:
        f.write(content)
    logger.info(f"Created model_config.config at: {model_config_path}")
    
paths.RECENT_MODEL_DIR.mkdir(exist_ok=True)