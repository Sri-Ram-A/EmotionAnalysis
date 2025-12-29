import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding,Input,Dropout
from loguru import logger

def build_model(vocabulary_size , embedding_dimension , timesteps , total_classes):
    # ---- Model ----
    model = Sequential()
    model.add(Input(shape=(timesteps,), dtype='int32'))
    model.add(Embedding(
        input_dim = vocabulary_size,
        output_dim = embedding_dimension,
        input_length = timesteps,
    ))
    model.add(SimpleRNN(256, return_sequences=False))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(total_classes, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    logger.success("Model built successfully")
    logger.debug(model.summary())
    
    return model