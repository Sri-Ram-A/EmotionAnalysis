import sys
from pathlib import Path
BASE_DIR =  Path(__file__).resolve().parents[2]
SRC_DIR =  Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
print(SRC_DIR)
from argparse import ArgumentParser
import bow , padding , tfidf , dataset
from omegaconf import OmegaConf
from src.utils.helper import save_pkl

ARTIFACTS_DIR = BASE_DIR / "src" / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
PREPROCESSING_METHODS = {
    "bow":bow,
    "tfidf":tfidf,
    "padding":padding,
}

def get_arguments():
    parser = ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument( # Config file
        "--config_path",
        type=str,
        default="params.yaml",
        help="Path to configuration file"
    )
    return parser.parse_args()

def main():
    args = get_arguments()
    config = OmegaConf.load(BASE_DIR / args.config_path)
    df = dataset.load_dataset(config.dataset.path)
    X, y = PREPROCESSING_METHODS[config.preprocess.method].fit(df,config.preprocess)
    # Paths for outputs
    X_path , y_path = ARTIFACTS_DIR / "X.pkl" , ARTIFACTS_DIR / "y.pkl"
    # Save using pickle
    save_pkl(X_path , X)
    save_pkl(y_path , y)
    
if __name__ == "__main__":
    main()
