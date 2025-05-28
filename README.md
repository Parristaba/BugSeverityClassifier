# Bug Severity Classifier using DistilBERT

This project provides an end-to-end pipeline for classifying software bug reports into severity categories using a transformer-based model (DistilBERT). It was developed for a Kaggle competition and makes use of both traditional NLP preprocessing and modern deep learning techniques.

## 📂 Repository Structure

```plaintext
.
├── dataset/                         # Contains bugs-train.csv and bugs-test.csv
├── predictions/                     # Contains the final submission CSV
├── bug_severity_pipeline.ipynb     # Jupyter notebook implementing the full pipeline
```

## 🔍 Problem

Given bug reports with summaries and severity labels, the goal is to classify the severity of new bug reports. The challenge lies in:
- Class imbalance across severity levels
- Noisy natural language input
- Need for generalization across unseen test data

## 🧠 Model

- **Base Model**: DistilBERT (`distilbert-base-uncased`)
- **Custom Classifier**: Extends `TFDistilBertForSequenceClassification` with custom `train_step`/`test_step`
- **Loss Handling**: Class weights are computed to address label imbalance
- **Regularization**: Uses dropout
- **Early Stopping**: Stops training when validation loss stops improving

## 🔁 Pipeline Overview

1. Load and clean the dataset using `nltk` (stopwords, lemmatization, punctuation removal)
2. Map severity classes to numerical labels
3. Tokenize with HuggingFace tokenizer
4. Create train/validation splits with `stratify`
5. Train with custom loss and optimizer
6. Predict on test set
7. Save predictions to `predictions/predicted_severity_final.csv`

## 📊 Dataset

- Located under `dataset/`
- Not included in the repo if restricted by Kaggle rules; please download from the competition page if needed

## 📁 Output

- Final CSV submission is saved under `predictions/`
- Format: `bug_id, severity`

## ⚠️ Note

Prediction values are not visible in this repo due to Kaggle competition restrictions. The code runs end-to-end and generates the CSV ready for submission.

## 🛠️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

### Key Packages:
- `tensorflow` (with GPU support)
- `transformers`
- `nltk`
- `pandas`
- `scikit-learn`

## ✍️ Author

Fevzi Kagan Becel
