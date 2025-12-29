from . import featurize , ingestion
from src.utils import helper
from src.utils.paths import paths
from src.utils.schema import Config

def main():
    config = Config.load(paths.USER_CONFIG)
    df = ingestion.load_dataset(config)
    corpus,labels = df["cleaned_text"] , df["sentiment"]
    X, y , tokenizer = featurize.fit(corpus,labels,config.featurize)
    helper.save_h5(X, y, paths.DATA_H5_FILE)
    helper.save_tokenizer(tokenizer,str(paths.TOKENIZER_JSON_FILE))
    print('\u2500' * 175)
    
if __name__ == "__main__":
    main()
