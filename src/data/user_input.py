import sys
from pathlib import Path
BASE_DIR =  Path(__file__).resolve().parents[2]
SRC_DIR =  Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
print(SRC_DIR)
from omegaconf import OmegaConf
from src.utils import helper
from src.utils.paths import paths
import featurize , ingestion

def main():
    config = OmegaConf.load(paths.USER_CONFIG)
    df = ingestion.load_dataset(config)
    corpus,labels = df["cleaned_text"] , df["sentiment"]
    X, y , tokenizer = featurize.fit(corpus,labels,config.featurize)
    helper.save_h5(X, y, paths.DATA_H5_FILE)
    helper.save_tokenizer(tokenizer,str(paths.TOKENIZER_JSON_FILE))
    
if __name__ == "__main__":
    main()
