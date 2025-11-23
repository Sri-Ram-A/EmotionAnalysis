import sys
from pathlib import Path
BASE_DIR =  Path(__file__).resolve().parents[2]
SRC_DIR =  Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
print(SRC_DIR)
from omegaconf import OmegaConf
from argparse import ArgumentParser
from src.utils import helper
import padding , dataset

ARTIFACTS_DIR = BASE_DIR / "src" / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

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
    df = dataset.load_dataset(config.dataset)
    corpus,labels = df["cleaned_text"] , df["sentiment"]
    X, y , tokenizer = padding.fit(corpus,labels,config.preprocess)
    helper.save_h5(X, y, ARTIFACTS_DIR / "data.h5")
    helper.save_tokenizer(tokenizer,str(ARTIFACTS_DIR / "tokenizer.json"))

    
if __name__ == "__main__":
    main()
