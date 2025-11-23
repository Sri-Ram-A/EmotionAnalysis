from pathlib import Path
import pandas as pd
from loguru import logger

def load_dataset(parameters,usecols = ["sentiment","cleaned_text"]):
    # CSV file must contain 2 columns - [ "sentiment" , "cleaned_text" ]
    csv_file_path = Path(parameters.path)
    df = pd.read_csv(csv_file_path,usecols=usecols,nrows=parameters.nrows)
    df["cleaned_text"] = df["cleaned_text"].astype(str)
    logger.success(f"Loaded dataset from {csv_file_path}")
    return df