# Phishing Website Detection Using MCMC-GAN and Stacking Ensemble

**Developed by: CS412 Group 9**  
**Academic Year: 2025**

This project presents a machine learning-based solution to phishing website detection. It addresses class imbalance using a custom MCMC-GAN and improves classification performance through a stacking ensemble model that combines multiple base classifiers.

---

## Problem Statement

Phishing detection datasets are typically imbalanced, with significantly fewer phishing examples compared to legitimate ones. This imbalance compromises the performance of conventional classifiers, often leading to poor generalization and high false negative rates. To address this challenge, we apply:

- A **Markov Chain Monte Carlo–based Generative Adversarial Network (MCMC-GAN)** for data augmentation.
- A **stacking ensemble model** to enhance the robustness and accuracy of phishing detection.

---

## System Architecture

```
Raw Dataset (CSV)
        ↓
Feature Selection (SelectKBest, Top 30 Features)
        ↓
MCMC-GAN Training (Generator + Discriminator + MCMC Sampler)
        ↓
Balanced Dataset (Real + Synthetic Phishing Samples)
        ↓
Model Training (Logistic Regression, Random Forest, XGBoost, CatBoost)
        ↓
Stacking Ensemble
        ↓
Evaluation and Model Export
```

---

## Environment Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/Kokanati/CS412-Group9.git
cd CS412-Group9
```

### Step 2: Create and Activate a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare Dataset

Ensure your dataset is saved at:

```
data/phishing_dataset.csv
```

The dataset must include a column named `status`:
- `0` represents legitimate websites
- `1` represents phishing websites

---

## Execution Instructions

### Train MCMC-GAN and Generate Balanced Dataset

```bash
python train.py
```

### Train and Evaluate Classifiers

```bash
python test.py
```

Outputs include:
- Synthetic data and balanced dataset
- Trained model files
- Evaluation metrics (Accuracy, Precision, Recall, F1 Score, ROC-AUC)

All outputs are stored in the `outputs/` directory.

---

## Key Components

- **Data Augmentation:** MCMC-GAN with Metropolis-Hastings sampling
- **Feature Selection:** Mutual Information via `SelectKBest`
- **Classifiers:** Logistic Regression, Random Forest, XGBoost, CatBoost
- **Ensemble Method:** StackingClassifier (scikit-learn)
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1 Score, ROC-AUC

---

## Project Structure

```
CS412-Group9/
├── data/                  # Input phishing dataset (CSV)
├── logs/                  # Optional logs directory
├── outputs/               # Augmented dataset and model outputs
├── mcmc_gan/              # GAN modules (generator, discriminator, sampler)
├── utils/                 # Feature selection utilities
├── train.py               # Training pipeline (MCMC-GAN + data balancing)
├── test.py                # Classifier training and evaluation
├── config.py              # Configuration settings
└── requirements.txt       # Python dependencies
```

---

## Acknowledgements

This system was developed as part of the CS412 – Artificial Intelligence course (2025) at the University of the South Pacific. It demonstrates the practical application of generative models and ensemble learning in addressing real-world cybersecurity challenges.
