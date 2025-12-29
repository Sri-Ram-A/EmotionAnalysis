from loguru import logger
from sklearn.model_selection import train_test_split
from src.models import rnn, lstm, gru
from src.utils import helper
from src.utils.paths import paths
from src.utils.schema import Config
import warnings
warnings.filterwarnings('ignore')

MODEL_METHODS = {"rnn": ("1", rnn), "lstm": ("1", lstm), "gru": ("1", gru)}

def main():
    config = Config.load(paths.USER_CONFIG)
    run_number = helper.get_next_run_number(config.mlflow.experiment_name, config.mlflow.tracking_uri)

    model_arch = config.model.architecture.lower()
    dataset_name = config.dataset.name.lower()
    mlflow_enabled = config.mlflow.perform
    run_name = f"{dataset_name}_{model_arch}_{run_number}"
    logger.info(f"Training pipeline started | run_name={run_name}")

    if not config.train.perform:
        logger.warning("Training pipeline skipped (train.perform=false)")
        return
    
    if mlflow_enabled:
        import mlflow
        from mlflow.tensorflow import autolog
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)
        mlflow.start_run(run_name=run_name,log_system_metrics=True)
        autolog(log_models=True, log_input_examples=True, log_model_signatures=True)
        logger.info(f"MLflow run started | experiment={config.mlflow.experiment_name} | run_name={run_name}")
    logger.info("Loading preprocessed dataset and tokenizer")

    X, y = helper.load_h5(str(paths.DATA_H5_FILE))
    tokenizer = helper.load_tokenizer(str(paths.TOKENIZER_JSON_FILE))
    vocabulary_size = len(tokenizer.word_index) + 1
    timesteps = X.shape[1]
    total_classes = y.shape[1]
    embedding_dimension = config.model.embedding_dimension
    logger.info(f"Dataset stats | vocab_size={vocabulary_size} | timesteps={timesteps} | embedding_dim={embedding_dimension} | classes={total_classes}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config.train.test_size, random_state=42)
    logger.info(f"Data split completed | train_size={X_train.shape[0]} | val_size={X_val.shape[0]}")
    logger.info(f"Building model | architecture={model_arch.upper()}")

    model = MODEL_METHODS[model_arch][1].build_model(vocabulary_size, embedding_dimension, timesteps, total_classes)
    logger.info(f"Training started | epochs={config.train.epochs} | batch_size={config.train.batch_size}")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=config.train.epochs, batch_size=config.train.batch_size)
    logger.success(f"Training completed successfully | epochs={config.train.epochs}")
    prod_num = MODEL_METHODS[model_arch][0]

    export_path = paths.RECENT_MODEL_DIR / model_arch / prod_num
    model.export(export_path)
    logger.info(f"Model exported | path={export_path}")
    if mlflow_enabled:
        mlflow.end_run()
        logger.info(f"MLflow run completed | run_name={run_name}")
    logger.success("Training pipeline finished successfully")

if __name__ == "__main__":
    main()
