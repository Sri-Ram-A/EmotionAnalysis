1. How to configure the DVC storage?
    - https://dvc.org/doc/install/linux
    - https://dvc.org/doc/start
    - Below didnt work due to rate limits and stuff
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

3. DVC Pipelines ?
    ### Argparser
    - https://realpython.com/command-line-interfaces-python-argparse/
    ### Pipeline
    - Chatgpt meaning
    - https://dvc.org/doc/start/data-pipelines/data-pipelines
    Multiple .yaml files
    - https://discuss.dvc.org/t/multiple-params-yaml/720
    ### Error
        (tf-venv) srirama@LAPTOP-9S394MJD:~/sr_proj/EmotionAnalysis$ dvc repro -f
        ERROR: failed to reproduce 'preprocess': Unable to read RWLock-file '.dvc/tmp/rwlock'. JSON structure is corrupted: Expecting value: line 1 column 1 (char 0)
        (tf-venv) srirama@LAPTOP-9S394MJD:~/sr_proj/EmotionAnalysis$ rm -f .dvc/tmp/rwlock

4. Hyperparameter Tuning
    ### Ray-Tuner
    - I got OOM Error so left it
    ### Keras-Tuner
    https://www.tensorflow.org/tutorials/keras/keras_tuner
    - Below article for MLFLow + Keras-Tuner
    https://towardsdev.com/using-mlflow-with-keras-tuner-f6df5dd634bc
    https://keras.io/keras_tuner/api/hypermodels/base_hypermodel/
    https://keras.io/keras_tuner/api/tuners/hyperband/

5. Tensorflow serving
    ### Prerequisite 
    - To enable Docker in WSL
    https://docs.docker.com/desktop/features/wsl/
    - Navigate to Resources > WSL Integration.
    - Ensure that "Enable WSL 2 based engine" is checked.
    - Select the specific WSL distributions you want - Docker to integrate with (eg., Ubuntu).
    - Click "Apply & Restart" to save the changes and restart Docker Desktop.
    - In WSL Terminal run 
    ```python
    docker -v
    ```
    ### TFX
    - https://neptune.ai/blog/how-to-serve-machine-learning-models-with-tensorflow-serving-and-docker
    - Info from GPT :
    TensorFlow Serving expects the folder to look like:
    ```bash
    production_model/
        1/
            saved_model.pb
            variables/
    ```
    Not like this:
    ```bash
    production_model/
        saved_model.pb
    ```

    If your structure is missing version number directory (1/), TF-Serving won’t start.
    ```bash
    docker run -p 8501:8501   --name tfserving_classifier   --mount type=bind,source=/home/srirama/sr_proj/EmotionAnalysis/src/artifacts/production_model,target=/models/rnn_classifier   -e MODEL_NAME=rnn_classifier   -t tensorflow/serving:latest-gpu
    ```
    For running container from next time onwards:
    ```bash
        docker start tfserving_classifier
    ```

    

