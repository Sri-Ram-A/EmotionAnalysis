import requests
import json
import numpy as np
import pickle
from loguru import logger
# ==========================================
# Load 3–4 samples from your validation data
# ==========================================
import tensorflow as tf
from pathlib import Path
SRC_DIR =  Path(__file__).resolve().parents[1] / "src"

def load_pkl(file_path):
    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)
    logger.success(f"Loaded pkl from {file_path}")
    return loaded_data

ARTIFACTS_DIR = SRC_DIR / "artifacts"

# Load preprocessed data
X = load_pkl(ARTIFACTS_DIR / "X.pkl").numpy()
y = load_pkl(ARTIFACTS_DIR / "y.pkl").numpy()

# Manual shuffle (same as training script)
n = X.shape[0]
idx = tf.random.shuffle(tf.range(n))
X = tf.gather(X, idx).numpy()
y = tf.gather(y, idx).numpy()

# Take validation split (consistent with training logic)
split = int(0.6 * n)
X_val, y_val = X[split:], y[split:]

# Select 4 samples
instances = X_val[:4]        # shape: (4, vocab_size)
true_labels = np.argmax(y_val[:4], axis=1)


# ==========================================
# TensorFlow Serving endpoint
# ==========================================
url = "http://localhost:8501/v1/models/rnn_classifier:predict"


def make_prediction(instances):
    data = json.dumps({
        "signature_name": "serving_default",
        "instances": instances.tolist()  # must be python lists
    })

    headers = {"content-type": "application/json"}
    response = requests.post(url, data=data, headers=headers)

    preds = json.loads(response.text)["predictions"]
    return np.array(preds)


# ==========================================
# Run inference on 4 validation samples
# ==========================================
preds = make_prediction(instances)

for i, pred in enumerate(preds):
    print(f"\nSample {i+1}")
    print(f"True Label:      {true_labels[i]}")
    print(f"Predicted Label: {np.argmax(pred)}")
    print(f"Raw Probabilities: {pred}")
