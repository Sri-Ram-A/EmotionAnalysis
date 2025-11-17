import pickle
from loguru import logger
from pathlib import Path

def save_pkl(file_path , x):
    file_path = Path(file_path)
    with open(file_path, "wb") as f:
        pickle.dump(x, f)
    logger.success(f"Saved at {file_path}")
    
def load_pkl(file_path):
    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)
    logger.success(f"Loaded pkl from {file_path}")
    return loaded_data