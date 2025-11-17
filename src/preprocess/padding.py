from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical # type: ignore
from loguru import logger

def fit(df : pd.DataFrame,parameters):
    parameters = parameters.padding
    
    tokenizer = Tokenizer(oov_token="None")
    tokenizer.fit_on_texts(df['cleaned_text'])
    vocab_size  = len(tokenizer.word_index) # 31774 unique words
    logger.debug(f"Vacobulary Size : {vocab_size} ")
    sequences  = tokenizer.texts_to_sequences(df['cleaned_text'])
    logger.debug(f"Total Sequences in dataset : {len(sequences)}" )# = total rows
    X = pad_sequences(
        sequences, 
        maxlen = parameters.maxlen,
        padding = parameters.padding
    )
    logger.debug(f"Padded Sequence shape (X) : {X.shape}")
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df['sentiment'])
    y = to_categorical(y_encoded)  # one-hot encode for multi-class
    logger.debug(f"Total rows , sentiment classes (y) : {y.shape}")
    
    return X, y