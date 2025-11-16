# CivicSense
This project is about performing Sentiment Analysis on various Social Reforms and display insights on these sentiments . The final goal is making an web extension with the main focus on MLOPS
- Lets go

| Stage                        | What Happens                                 | Tools                               | MLflow Role                       |
| ---------------------------- | -------------------------------------------- | ----------------------------------- | --------------------------------- |
| **1. Data Versioning**       | Track datasets, preprocess updates           | DVC, Delta Lake, LakeFS             | Can store data artifacts          |
| **2. Experiment Tracking**   | Log metrics, params, artifacts               | **MLflow Tracking**                 | Core tracking                     |
| **3. Hyperparameter Tuning** | Automated search for best params             | Optuna, Hyperopt, Ray Tune          | Logs all runs & results           |
| **4. Model Packaging**       | Save model in reproducible format            | MLflow models, Docker               | Model flavors + environment       |
| **5. Model Registry**        | Version control for models                   | **MLflow Model Registry**           | Central model HUB                 |
| **6. Pipeline Automation**   | Orchestrate data → train → evaluate → deploy | Airflow, Prefect, Dagster, Kubeflow | Not native, but MLflow integrates |
| **7. CI/CD for ML**          | Automated testing & deployment               | GitHub Actions, GitLab CI           | Fetch & deploy MLflow models      |
| **8. Model Serving**         | Host model for real-time use                 | MLflow Serving, FastAPI, Kubernetes | Serve any MLflow model            |
| **9. Monitoring**            | Track drift, quality, perf                   | Evidently AI, WhyLogs, Prometheus   | Model metrics + logs              |
| **10. Retraining Loop**      | Auto-update model with new data              | Airflow/Prefect + MLflow            | Closing the MLops loop            |


https://docs.prefect.io/v3/get-started/install