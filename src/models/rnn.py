# Suppress INFO and WARNING messages (shows lot of info in terminal lol)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import logging
tf_logger = tf.get_logger()
tf_logger.setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding, LSTM, GRU 
from loguru import logger

def build_model(config):
    # Extract parameters from config (matching your YAML structure)
    model_parameters = config.model.rnn
    compile_parameters = config.model.compile
    dataset_config = config.dataset
    
    # Build model
    model = Sequential()
    
    # RNN Layer
    model.add(SimpleRNN(
        units=model_parameters.simple_rnn.units, 
        input_shape=(
            model_parameters.simple_rnn.input_shape.timesteps, 
            model_parameters.simple_rnn.input_shape.input_nodes
        ),
        return_sequences=False
    ))
    
    # Dense Layer (using the units from dense section)
    model.add(Dense(
        units=model_parameters.dense.units,
        activation=model_parameters.dense.activation
    ))
    
    # Output Layer (using total_classes from dataset)
    model.add(Dense(
        units=dataset_config.total_classes,
        activation="softmax"
    ))
    
    # Compile model
    model.compile(
        loss=compile_parameters.loss,
        optimizer=compile_parameters.optimizer,
        metrics=[compile_parameters.metrics] if isinstance(compile_parameters.metrics, str) else list(compile_parameters.metrics)
    )
    
    logger.debug("Model built successfully:")
    logger.debug(model.summary())
    
    return model