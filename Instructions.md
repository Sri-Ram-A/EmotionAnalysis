1. How to configure the DVC storage?
    - https://dvc.org/doc/install/linux
    - https://dvc.org/doc/start
    - https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended

2. How to connect MLFlow?
    Option A: Database (Recommended)
    - https://mlflow.org/docs/latest/genai/getting-started/connect-environment/
    ```python
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
    ```
    - https://mlflow.org/docs/latest/api_reference/python_api/mlflow.tensorflow.html
    ## Metrics and Parameters
    Training and validation loss.
    User-specified metrics.
    Optimizer config, e.g., learning_rate, momentum, etc.
    Training configs, e.g., epochs, batch_size, etc.
    ## Artifacts
    Model summary on training start.
    Saved Keras model in MLflow Model format.
    TensorBoard logs on training end.