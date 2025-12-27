
## 1. How to configure the DVC storage?

Links:

* [https://dvc.org/doc/install/linux](https://dvc.org/doc/install/linux)
* [https://dvc.org/doc/start](https://dvc.org/doc/start)
* Below didn’t work due to rate limits and stuff:
* [https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended)

---

## 2. How to connect MLFlow?

### Option A: Database (Recommended)
* [https://mlflow.org/docs/latest/genai/getting-started/connect-environment/](https://mlflow.org/docs/latest/genai/getting-started/connect-environment/)
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
* [https://mlflow.org/docs/latest/api_reference/python_api/mlflow.tensorflow.html](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.tensorflow.html)
---

## 3. DVC Pipelines

### Argparser

* [https://realpython.com/command-line-interfaces-python-argparse/](https://realpython.com/command-line-interfaces-python-argparse/)

### Pipeline

* ChatGPT meaning
* [https://dvc.org/doc/start/data-pipelines/data-pipelines](https://dvc.org/doc/start/data-pipelines/data-pipelines)
* Multiple `.yaml` files:
* [https://discuss.dvc.org/t/multiple-params-yaml/720](https://discuss.dvc.org/t/multiple-params-yaml/720)

### Error

```bash
(tf-venv) srirama@LAPTOP-9S394MJD:~/sr_proj/EmotionAnalysis$ dvc repro -f
ERROR: failed to reproduce 'preprocess': Unable to read RWLock-file '.dvc/tmp/rwlock'. JSON structure is corrupted: Expecting value: line 1 column 1 (char 0)
(tf-venv) srirama@LAPTOP-9S394MJD:~/sr_proj/EmotionAnalysis$ rm -f .dvc/tmp/rwlock
```

---

## 4. Hyperparameter Tuning

### Ray-Tuner

* I got OOM Error so left it

### Keras-Tuner

* [https://www.tensorflow.org/tutorials/keras/keras_tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner)
* MLFlow + Keras-Tuner

  * [https://towardsdev.com/using-mlflow-with-keras-tuner-f6df5dd634bc](https://towardsdev.com/using-mlflow-with-keras-tuner-f6df5dd634bc)
  * [https://keras.io/keras_tuner/api/hypermodels/base_hypermodel/](https://keras.io/keras_tuner/api/hypermodels/base_hypermodel/)
  * [https://keras.io/keras_tuner/api/tuners/hyperband/](https://keras.io/keras_tuner/api/tuners/hyperband/)

---

## 5. TensorFlow Serving

### Prerequisite

To enable Docker in WSL:

* [https://docs.docker.com/desktop/features/wsl/](https://docs.docker.com/desktop/features/wsl/)

Steps in Docker Desktop:

* Navigate to Resources > WSL Integration.
* Ensure that "Enable WSL 2 based engine" is checked.
* Select the specific WSL distributions you want Docker to integrate with (e.g., Ubuntu).
* Click "Apply & Restart" to save the changes.
* In WSL terminal:

```bash
docker -v
```

### TFX Info

* [https://neptune.ai/blog/how-to-serve-machine-learning-models-with-tensorflow-serving-and-docker](https://neptune.ai/blog/how-to-serve-machine-learning-models-with-tensorflow-serving-and-docker)

TensorFlow Serving expects folder structure:

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

If missing version number directory, TF-Serving won’t start.

Run docker:

```bash
docker run -p 8501:8501 \
--name tfserving_classifier \
--mount type=bind,source=/home/srirama/sr_proj/EmotionAnalysis/src/artifacts/production_model,target=/models/rnn_classifier \
-e MODEL_NAME=rnn_classifier \
-t tensorflow/serving:latest-gpu
```

Start container next time:

```bash
docker start tfserving_classifier
```

---

## 6. TFX in Render

This method bakes the model INTO the Docker image itself:

* [https://www.tensorflow.org/tfx/serving/docker#creating_your_own_serving_image](https://www.tensorflow.org/tfx/serving/docker#creating_your_own_serving_image)

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

### Push model-embedded Docker image to Docker Hub

Didn’t understand this much:

* [https://docs.docker.com/get-started/introduction/build-and-push-first-image/](https://docs.docker.com/get-started/introduction/build-and-push-first-image/)
  Skip to: Push Docker Image to Docker Hub
* [https://medium.com/@komalminhas.96/a-step-by-step-guide-to-build-and-push-your-own-docker-images-to-dockerhub-709963d4a8bc](https://medium.com/@komalminhas.96/a-step-by-step-guide-to-build-and-push-your-own-docker-images-to-dockerhub-709963d4a8bc)

```bash
docker tag rnn_classifier_serving:latest starmagiciansr/mlops-tfx:v1.0
docker push starmagiciansr/mlops-tfx:v1.0
```

Will come back to this later:

* How to deploy multiple models in TFX:
  * [https://www.scaler.com/topics/tensorflow/tensorflow-serving/](https://www.scaler.com/topics/tensorflow/tensorflow-serving/)
  * [https://medium.com/retina-ai-health-inc/tensorflow-serving-of-multiple-ml-models-simultaneously-to-a-rest-api-python-client-cd60ac6f71aa](https://medium.com/retina-ai-health-inc/tensorflow-serving-of-multiple-ml-models-simultaneously-to-a-rest-api-python-client-cd60ac6f71aa)
  * [https://www.tensorflow.org/tfx/serving/serving_config#reloading_model_server_configuration](https://www.tensorflow.org/tfx/serving/serving_config#reloading_model_server_configuration)
```bash
docker run -p 8501:8501 \
  --name tf-serving-multimodel \
  --mount type=bind,source=/home/srirama/sr_proj/EmotionAnalysis/src/artifacts/recent_model/rnn,target=/models/rnn \
  --mount type=bind,source=/home/srirama/sr_proj/EmotionAnalysis/src/artifacts/recent_model/lstm,target=/models/lstm \
  --mount type=bind,source=/home/srirama/sr_proj/EmotionAnalysis/src/artifacts/recent_model/gru,target=/models/gru \
  --mount type=bind,source=/home/srirama/sr_proj/EmotionAnalysis/src/artifacts/recent_model/model_config.config,target=/models/model_config.config \
  -t tensorflow/serving:latest-gpu \
  --model_config_file=/models/model_config.config

```
```bash
  # 1. Run a temporary serving container
  docker run -d --name serving_multi tensorflow/serving:latest-gpu

  # 2. Copy your models and config into the container
  docker cp /home/srirama/sr_proj/EmotionAnalysis/src/artifacts/recent_model/rnn \
  serving_multi:/models/rnn

  docker cp /home/srirama/sr_proj/EmotionAnalysis/src/artifacts/recent_model/lstm \
  serving_multi:/models/lstm

  docker cp /home/srirama/sr_proj/EmotionAnalysis/src/artifacts/recent_model/gru \
  serving_multi:/models/gru

  docker cp /home/srirama/sr_proj/EmotionAnalysis/src/artifacts/recent_model/model_config.config \
  serving_multi:/models/model_config.config

  # 3. Commit the container into a reusable image
  docker commit \
    --change "ENV MODEL_CONFIG_FILE /models/model_config.config" \
    serving_multi tfserving-multimodel:latest

  # 4. Delete temporary container
  docker kill serving_multi
  docker rm serving_multi

  # 5. Run your custom TF Serving image
  docker run -p 8501:8501 \
    --name tfserving_multimodel \
    -t tfserving-multimodel:latest \
    --model_config_file=/models/model_config.config

  # 6. Next time start it
  docker start tfserving_multimodel
```
---

## 7. MLFlow Registry for evaluation

How to Load Model for evaluation?

* [https://mlflow.org/docs/latest/ml/model-registry/tutorial/#load-a-registered-model](https://mlflow.org/docs/latest/ml/model-registry/tutorial/#load-a-registered-model)

### Load registered model example

```python
model_uri = f"models:/{model_name}/{model_version}"
model_uri = f"models:/{model_name}@{model_version_alias}"
model = mlflow.keras.load_model(f"runs:/{latest_run_id}/model") # Chatgpt (not working for me)
model_uri = str(Path(experiment._artifact_location) / "models" / str(latest_run.outputs.model_outputs[0].model_id)) # I found out on own using private variables

```

Found a different method to load the model:

* Refer to MLFlow docs + src/registry folder
* [https://mlflow.org/docs/latest/ml/model-registry/workflow/](https://mlflow.org/docs/latest/ml/model-registry/workflow/)

---

## 8. Deploy to Render using API
  - Used to automate docker image deploy : 
    https://render.com/docs/deploying-an-image
  - If you get permission denied error , use similar command
    sudo chown -R $USER:$USER /home/srirama/sr_proj/EmotionAnalysis/src/artifacts
  - Starting docker image problem in render
    ```bash
    docker history --no-trunc tensorflow/multimodel:v1.1
    ```


