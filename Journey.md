
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

6. TFX in Render
    ### This method bakes the model INTO the Docker image itself.
        https://www.tensorflow.org/tfx/serving/docker#creating_your_own_serving_image
    ```bash
    # 1. Run base serving container
    docker run -d --name serving_base tensorflow/serving
    # 2. Copy SavedModel into container
    docker cp /home/srirama/sr_proj/EmotionAnalysis/src/artifacts/production_model \
    serving_base:/models/rnn_classifier
    # 3. Commit container → create custom serving image
    docker commit --change "ENV MODEL_NAME rnn_classifier" \
    serving_base rnn_classifier_serving:latest
    # 4. Stop and remove temporary container
    docker kill serving_base
    docker rm serving_base
    # 5. Run your custom serving image
    docker run -p 8501:8501 --name tfserving_classifier_inbuilt \
    -t rnn_classifier_serving:latest
    # 6. Start next time
    docker start tfserving_classifier
    ```
    ### Push this model embedded docker image to Docker Hub
    Didnt Understand this much
        https://docs.docker.com/get-started/introduction/build-and-push-first-image/
    Skip to **Push Docker Image to Docker Hub**
        https://medium.com/@komalminhas.96/a-step-by-step-guide-to-build-and-push-your-own-docker-images-to-dockerhub-709963d4a8bc
    ```bash
    docker tag rnn_classifier_serving:latest starmagiciansr/mlops-tfx:v1.0
    docker push starmagiciansr/mlops-tfx:v1.0
    ```
    Will come back to this later:
        How to deploy multiple models in TFX:
            - https://medium.com/retina-ai-health-inc/tensorflow-serving-of-multiple-ml-models-simultaneously-to-a-rest-api-python-client-cd60ac6f71aa
            - https://www.tensorflow.org/tfx/serving/serving_config#reloading_model_server_configuration

        

7. MLFlow Registry for evaluation
    How to Load Model for evaluation?
    - https://mlflow.org/docs/latest/ml/model-registry/tutorial/#load-a-registered-model
    # Load the model from the Model Registry
    model_uri = f"models:/{model_name}/{model_version}"
    model_uri = f"models:/{model_name}@{model_version_alias}"
    model = mlflow.keras.load_model(f"runs:/{latest_run_id}/model") # Chatgpt

