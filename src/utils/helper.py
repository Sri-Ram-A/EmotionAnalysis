import pickle
from loguru import logger
from pathlib import Path
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec
import numpy as np

def get_model_signature(input_shape, output_shape):
    input_schema = Schema([
        TensorSpec(
            type=np.dtype("float32"),          # <-- works for everyone
            shape=(-1, *input_shape[1:]),
            name="input"
        )
    ])

    output_schema = Schema([
        TensorSpec(
            type=np.dtype("float32"),
            shape=(-1, *output_shape[1:]),
            name="output"
        )
    ])

    return ModelSignature(inputs=input_schema, outputs=output_schema)

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