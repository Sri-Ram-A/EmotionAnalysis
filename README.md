
# 🎭 CivicSense — Emotion & Sentiment Analysis for Social Media
**A production-ready MLOps system for multi-class emotion classification on social media content**

<div align="center">


[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6)](https://dvc.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Paul Ekman's 7 Emotions](public/emotion.png)

*Classifying emotions across Joy, Sadness, Anger, Fear, Surprise, Disgust, and Neutral categories*

[Features](#-key-features) • [Quick Start](#-quick-start) • [Documentation](#-project-structure) • [Contributing](#-contributing)

</div>

---

## 🎯 Overview

CivicSense is an end-to-end MLOps project that analyzes emotional content in social media text. The system classifies text based on emotions using deep learning models (RNN, LSTM, GRU, BiLSTM).

Unlike simple positive/negative sentiment analysis, CivicSense captures the nuanced emotional landscape of social media discourse, making it ideal for:

- 📊 **Content Analysis**: Understanding audience reactions to social content
- 🔍 **Trend Detection**: Identifying emotional patterns in public discourse
- 🎓 **Research**: Academic studies on emotion expression in digital communication
- 🛠️ **Product Development**: Building emotion-aware applications and features

**Current Status:** ✅ Active development with reproducible experiments and production-ready models

---

## ✨ Key Features

### Machine Learning
- 🧠 **Multiple Architectures**: RNN, LSTM, GRU, and Bidirectional LSTM models
- 🎯 **7-Class Classification**: Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral
- 📈 **72% Test Accuracy**: Strong performance on multi-class emotion detection
- 🔄 **Transfer Learning Ready**: Extensible architecture for fine-tuning

### MLOps Pipeline
- 📦 **DVC Integration**: Complete data version control and pipeline orchestration
- 📊 **MLflow Tracking**: Comprehensive experiment logging and model registry
- 🔁 **Reproducible Workflows**: Automated preprocessing, training, and evaluation
- 📝 **Model Governance**: Staging, promotion, and deployment workflows

### Data Engineering
- 🌐 **Multi-Source Datasets**: Twitter (1.6M), GoEmotions (58K), ISEAR (7K)
- 🧹 **Advanced Preprocessing**: Emoji handling, slang normalization, text cleaning
- 🔤 **Tokenization Pipeline**: Consistent vocabulary management with padding/truncating
- 💾 **Efficient Storage**: HDF5 format for fast batch loading

### Experimentation
- 📓 **Jupyter Notebooks**: Interactive exploration and visualization
- 🔬 **Baseline Comparisons**: BOW, TF-IDF, and deep learning benchmarks
- 📉 **Rich Visualizations**: Confusion matrices, training curves, embedding plots
- 🎛️ **Hyperparameter Tuning**: Keras Tuner integration for optimization

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Raw Datasets   │
│ (Twitter, Reddit│
│   ISEAR, etc.)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │ ◄─── slangs.txt, emoji mapping
│  & Cleaning     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Featurization  │ ◄─── tokenizer.json
│  (Tokenize +    │
│   Padding)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Training │ ◄─── params.yaml
│  (RNN/LSTM/GRU) │       
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MLflow         │
│  Experiment     │ ◄─── Metrics, artifacts
│  Tracking       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Registry │
│  (Stage/Promote)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Deployment     │
│  (TF Serving)   │
└─────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager
- Git (for cloning the repository)
- Optional: GPU with CUDA support for faster training

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/CivicSense.git
cd CivicSense
```

2. **Create and activate a virtual environment**
```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using conda
conda create -n civicsense python=3.10
conda activate civicsense
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Initialize DVC (if using data version control)**
```bash
dvc init
dvc pull  # Download data artifacts if available
```

### Running Your First Training

**Option 1: Using the main training script**
```bash
python src/main.py --config params.yaml
```

**Option 2: Using DVC pipeline (recommended for reproducibility)**
```bash
# Run the complete pipeline (preprocess → featurize → train)
dvc repro

# Or run specific stages
dvc repro train  # Only training stage
```

**Option 3: Interactive notebook**
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

### Starting MLflow UI

Monitor your experiments in real-time:
```bash
mlflow ui --port 5000
# Open http://localhost:5000 in your browser
```

---

## 🔄 Data Pipeline

### Data Sources

CivicSense combines multiple high-quality datasets:

| Dataset | Source | Size | Labels | Notes |
|---------|--------|------|--------|-------|
| **Twitter Sentiment** | Kaggle | 1.6M tweets | Binary  | Distant supervision via emoticons |
| **GoEmotions** | Google Research | 58K comments | 27  | Human-annotated Reddit comments |
| **ISEAR** | Academic | 7K narratives | Direct  | Cross-cultural emotional experiences |


### Running the Pipeline

**Via DVC (Recommended):**
```bash
# Run complete pipeline
dvc repro

# Run specific stage
dvc repro preprocess
dvc repro featurize
```

**Via Python scripts:**
```bash
# Preprocess raw data
python -m src.data.preprocess

# Featurize preprocessed data
python -m src.data.main
```

**Pipeline outputs:**
- `data/processed/train.csv` - Cleaned text with mapped labels
- `src/artifacts/data.h5` - Featurized sequences (padded/truncated to length 75)
- `src/artifacts/tokenizer.json` - Vocabulary and tokenization config

---

## 🎯 Model Training

### Available Models

| Architecture | Parameters | Training Time | Accuracy | Best For |
|--------------|------------|---------------|----------|----------|
| **Simple RNN** | ~160K | Fast (10 min) | ~64% | Quick baseline |
| **LSTM** | ~200K | Medium (20 min) | ~70% | Sequential patterns |
| **GRU** | ~180K | Medium (18 min) | ~70% | Efficiency + performance |
| **BiLSTM** | ~190K | Slower (25 min) | **~72%** | **Production (best accuracy)** |

*Training times on NVIDIA RTX 2050 GPU*

### Training a Model

**With custom hyperparameters:**
```bash
# Edit params.yaml, then:
dvc repro train
```

**Key hyperparameters in `params.yaml`:**
```yaml
train:
  epochs: 5
  batch_size: 400
  test_size: 0.4

model:
  architecture: bilstm  # rnn, lstm, gru, bilstm
  embedding_dimension: 32
  hidden_units: 64
  dropout: 0.5
```

### Training Output

After training completes, you'll find:
- **Model checkpoint**: `src/artifacts/recent/<architecture>/1/`
- **MLflow run**: Logged metrics, parameters, and artifacts
- **Training history**: Loss and accuracy curves

**Access results:**
```bash
# View in MLflow UI
mlflow ui --port 5000

# Or check artifacts directly
ls src/artifacts/recent/lstm/1/
# → saved_model.pb, variables/, ...
```

---

## 📊 Experiment Tracking

### MLflow Integration

Every training run automatically logs to MLflow:

**Starting MLflow UI:**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# Navigate to http://localhost:5000
```

**Key MLflow Features:**
- 🔍 **Compare runs**: Side-by-side metric comparison
- 📈 **Visualize metrics**: Interactive plots and charts
- 🏷️ **Tag experiments**: Organize by architecture, dataset, etc.
- 💾 **Model registry**: Version and stage models (staging/production)


### Deployment 

**TensorFlow Serving (Local/Docker)**
```bash
# Using Docker
docker run -p 8501:8501 \
  --mount type=bind,source=$(pwd)/src/artifacts/recent/bilstm,target=/models/emotion \
  -e MODEL_NAME=emotion \
  tensorflow/serving

# Test endpoint
curl -X POST http://localhost:8501/v1/models/emotion:predict \
  -H 'Content-Type: application/json' \
  -d '{"instances": [[1, 42, 15, 8, ...]]}'  # Tokenized sequence
```

---

## 📓 Notebooks & Exploration

The `notebooks/` directory contains interactive Jupyter notebooks for experimentation:

**Getting Started with Notebooks:**
```bash
# Install Jupyter if not already installed
pip install jupyter

# Launch notebook server
jupyter notebook

# Navigate to notebooks/ directory in the web UI
```

**Tips for Exploration:**
- Start with `01_exploratory_data_analysis.ipynb` to understand the data
- Use `04_model_training_comparison.ipynb` for quick architecture experiments
- Modify hyperparameters inline and re-run cells for rapid iteration

---

## 🤝 Contributing

We welcome contributions! Whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.

### How to Contribute

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub, then:
   git clone https://github.com/YOUR-USERNAME/CivicSense.git
   cd CivicSense
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clean, documented code
   - Add tests if applicable
   - Update documentation

4. **Run quality checks**
   ```bash
   # Format code
   black src/ notebooks/
   
   # Lint
   flake8 src/ --max-line-length=120
   
   # Run tests
   pytest tests/
   ```

5. **Commit and push**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   - Go to GitHub and click "New Pull Request"
   - Describe your changes clearly
   - Reference any related issues

---

## 🙏 Acknowledgments

This project builds upon and extends the excellent work from:

### Contributor
- **[MohithTP/SentimentAnalysis](https://github.com/MohithTP/SentimentAnalysis)** -  MLOps implementation and deployment architecture using AWS Sagemaker. 

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**You are free to:**
- ✅ Use commercially
- ✅ Modify
- ✅ Distribute
- ✅ Use privately

**Under the conditions:**
- 📄 License and copyright notice must be included
- ⚠️ No warranty or liability

---

### Community
- ⭐ Star this repository if you find it useful!
- 🔔 Watch for updates and new releases
- 🍴 Fork to create your own experiments

---


<div align="center">

### ⭐ Star this repository if you found it helpful!

**Made with ❤️ for the ML community**

[Report Bug](https://github.com/yourusername/CivicSense/issues) · [Request Feature](https://github.com/yourusername/CivicSense/issues) · [Contribute](CONTRIBUTING.md)

</div>

---

*Last Updated: January 2025*