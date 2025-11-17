from argparse import ArgumentParser
from src import preprocess , models

# Mapping user inputs → modules
PREPROCESSING_METHODS = {
    "bow": preprocess.bow,
    "tfidf": preprocess.tfidf,
    "padding": preprocess.padding,
}
MODEL_METHODS = {
    "rnn" : models.rnn
}
def get_arguments():
    parser = ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument( # Dataset Name
        "--data",
        type=str,
        required=True,
        help="Path to input CSV "
    )
    parser.add_argument( # Preprocessing
        "--preprocess",
        type=str,
        required=True,
        choices=PREPROCESSING_METHODS.keys(),
        help=f"Choose preprocessing technique - {PREPROCESSING_METHODS.keys()}"
    )
    parser.add_argument( # Preprocessing
        "--model",
        type=str,
        required=True,
        choices=MODEL_METHODS.keys(),
        help=f"Choose Fundamental Model Architecture - {MODEL_METHODS.keys()}"
    )
    return parser.parse_args()
