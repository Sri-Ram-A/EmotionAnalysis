import pickle
from loguru import logger
from pathlib import Path
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec
import numpy as np
import tensorflow as tf
import json
import h5py
from tensorflow.keras.preprocessing.text import tokenizer_from_json # pyright: ignore[reportMissingImports]
from typing import Any
import io

def save_h5(X, y, file_path):
    with h5py.File(file_path, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("y", data=y)
    logger.info(f"Dataset saved at {file_path} as hd5")
        
def load_h5(file_path):
    with h5py.File(file_path, "r") as f:
        X_ds: Any = f["X"]
        y_ds: Any = f["y"]
        X = X_ds[:]
        y = y_ds[:]
    logger.info(f"Dataset loaded from {file_path}")
    logger.info(f"Loaded data -> X: shape={X.shape}, dtype={X.dtype} ; y: shape={y.shape}, dtype={y.dtype}")
    return X, y

def save_tokenizer(tokenizer, file_path):
    tokenizer_json = tokenizer.to_json()
    with io.open(file_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))
    logger.info(f"Tokenizer saved at {file_path} as json")

def load_tokenizer(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        tok_dict = json.load(f)
    logger.info(f"Tokenizer loaded from {file_path}")
    return tokenizer_from_json(tok_dict)

def save_pkl(file_path , x):
    file_path = Path(file_path)
    # Create parent directory ONLY
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(x, f)
    logger.success(f"Saved at {file_path}")
    
def load_pkl(file_path):
    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)
    logger.success(f"Loaded pkl from {file_path}")
    return loaded_data

def get_model_signature(input_shape, output_shape):
    input_schema = Schema([
        TensorSpec(
            type=np.dtype("int32"),          # <-- works for everyone
            shape=(-1, *input_shape[1:]),
            name="input"
        )
    ])

    output_schema = Schema([
        TensorSpec(
            type=np.dtype("int32"),
            shape=(-1, *output_shape[1:]),
            name="output"
        )
    ])

    return ModelSignature(inputs=input_schema, outputs=output_schema)

def get_next_run_number(experiment_name, tracking_uri):
    """Get next run number for the experiment"""
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri)
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment:
            runs = client.search_runs(experiment_ids=[experiment.experiment_id])
            return len(runs)+1
    except:
        pass
    return 1