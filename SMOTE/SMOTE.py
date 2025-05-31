# Install required packages
#!pip install catboost imblearn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import warnings
import pytz
from datetime import datetime
from IPython.display import display  # For displaying tables in Colab
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.mixture import GaussianMixture
from google.colab import drive

warnings.filterwarnings('ignore')

# Set timezone to UTC+12 (using Etc/GMT-12 for +12 offset)
timezone = pytz.timezone("Etc/GMT-12")
current_time = datetime.now(timezone)

# Logging setup
log_file = open("phishing_detection_run.log", "a")
def log(message):
    timestamp = datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S %Z")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    print(full_message, file=log_file)
    log_file.flush()

log(f"Debug: Current date and time: {current_time}")
log("Script started.")

# Mount Google Drive
log("Mounting Google Drive...")
drive.mount('/content/drive', force_remount=True)

# Check if the dataset folder exists
dataset_dir = "/content/drive/MyDrive/dataset"
if not os.path.exists(dataset_dir):
    log(f"ERROR: Directory {dataset_dir} does not exist.")
    log("Please follow these steps to set up the directory:")
    log("1. In Google Drive, go to 'My Drive' and create a folder named 'dataset'.")
    log("2. Go to 'Shared with me', find the dataset files (e.g., dataset80_20_top10.csv).")
    log("3. Right-click each file, select 'Add shortcut to Drive', and place it in 'My Drive/dataset/'.")
    log("4. Verify by running: !ls \"/content/drive/MyDrive/dataset/\" in a Colab cell.")
    log("Script cannot proceed without the datasets. Exiting.")
    log_file.close()
    raise FileNotFoundError(f"Directory {dataset_dir} not found. Please set up the directory and try again.")

# Debug: List contents of the directory to verify files
log("Listing contents of /content/drive/MyDrive/dataset/...")
!ls "/content/drive/MyDrive/dataset/"

# Dataset paths
DATASETS = [
    '/content/drive/MyDrive/dataset/dataset80_20_top10.csv',
    '/content/drive/MyDrive/dataset/dataset80_20_top20.csv',
    '/content/drive/MyDrive/dataset/dataset80_20_top30.csv',
    '/content/drive/MyDrive/dataset/dataset90_10_top10.csv',
    '/content/drive/MyDrive/dataset/dataset90_10_top20.csv',
    '/content/drive/MyDrive/dataset/dataset90_10_top30.csv',
    '/content/drive/MyDrive/dataset/dataset95_5_top10.csv',
    '/content/drive/MyDrive/dataset/dataset95_5_top20.csv',
    '/content/drive/MyDrive/dataset/dataset95_5_top30.csv'
]

# Load dataset
def load_dataset(path):
    try:
        df = pd.read_csv(path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        log(f"Loaded {path} shape={df.shape}")
        return X, y
    except Exception as e:
        log(f"Failed to load {path}: {e}")
        return None, None

# SMOTE
def apply_smote(X, y):
    try:
        sm = SMOTE(random_state=42)
        X_aug, y_aug = sm.fit_resample(X, y)
        log(f"SMOTE Augmentation: Before - Legitimate: {sum(y == 0)}, Phishing: {sum(y == 1)} | After - Legitimate: {sum(y_aug == 0)}, Phishing: {sum(y_aug == 1)}")
        return X_aug, y_aug
    except Exception as e:
        log(f"Error in SMOTE: {e}")
        return X, y

# MCMC using GMM
def apply_mcmc(X, y, n_components=5):
    try:
        minority_indices = np.where(y == 1)[0]
        majority_indices = np.where(y == 0)[0]
        if len(minority_indices) == 0:
            log("No minority class samples found for MCMC.")
            return X, y
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(X[minority_indices])
        n_synthetic = len(majority_indices) - len(minority_indices)
        if n_synthetic <= 0:
            log("No synthetic samples needed; classes already balanced.")
            return X, y
        X_synthetic = gmm.sample(n_samples=n_synthetic)[0]
        X_aug = np.vstack((X, X_synthetic))
        y_aug = np.concatenate((y, np.ones(n_synthetic)))
        log(f"MCMC Augmentation: Before - Legitimate: {sum(y == 0)}, Phishing: {sum(y == 1)} | After - Legitimate: {sum(y_aug == 0)}, Phishing: {sum(y_aug == 1)}")
        return X_aug, y_aug
    except Exception as e:
        log(f"Error in MCMC: {e}")
        return X, y

# Models
models = {
    'LogisticRegression': LogisticRegression(max_iter=2000, class_weight='balanced'),
    'RandomForest': RandomForestClassifier(n_jobs=-1, class_weight='balanced'),
    'XGBoost': XGBClassifier(tree_method='hist', eval_metric='logloss'),
    'CatBoost': CatBoostClassifier(task_type="CPU", verbose=0),
    'Stacking': StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=2000, class_weight='balanced')),
            ('rf', RandomForestClassifier(n_jobs=-1, class_weight='balanced')),
            ('xgb', XGBClassifier(tree_method='hist', eval_metric='logloss'))
        ],
        final_estimator=LogisticRegression(max_iter=2000),
        n_jobs=-1
    )
}

# Training & Evaluation
results = []
for path in DATASETS:
    dataset_name = os.path.basename(path)
    log(f"Processing dataset: {dataset_name}")
    X, y = load_dataset(path)
    if X is None or y is None:
        continue
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    for method_name, augment in {'SMOTE': apply_smote, 'MCMC': apply_mcmc}.items():
        X_aug, y_aug = augment(X_train, y_train)
        for model_name, model in models.items():
            log(f"Training {model_name} with {method_name} on {dataset_name}")
            try:
                start_time = time.time()
                model.fit(X_aug, y_aug)
                y_train_pred = model.predict(X_aug)
                y_test_pred = model.predict(X_test)
                y_test_prob = model.predict_proba(X_test)[:, 1]
                end_time = time.time()
                results.append({
                    'Dataset': dataset_name,
                    'Augmentation': method_name,
                    'Model': model_name,
                    'Train Accuracy': accuracy_score(y_aug, y_train_pred),
                    'Test Accuracy': accuracy_score(y_test, y_test_pred),
                    'Precision': precision_score(y_test, y_test_pred),
                    'Recall': recall_score(y_test, y_test_pred),
                    'F1 Score': f1_score(y_test, y_test_pred),
                    'ROC-AUC': roc_auc_score(y_test, y_test_prob),
                    'Runtime (s)': end_time - start_time
                })
            except Exception as e:
                log(f"Failed model {model_name} on {dataset_name} ({method_name}): {e}")

# Export results
if results:
    results_df = pd.DataFrame(results)
    # Round numerical columns for better readability
    numerical_cols = ['Train Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'Runtime (s)']
    results_df[numerical_cols] = results_df[numerical_cols].round(4)
    
    # Display results in tabular format in Colab
    log("Displaying model performance results in tabular format:")
    display(results_df)
    
    # Export to CSV
    results_df.to_csv("model_performance_results.csv", index=False)
    log("Saved results to model_performance_results.csv")

    # Determine the best augmentation technique for each metric
    metrics = ['Train Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    best_aug_summary = []
    best_model_summary = []
    for metric in metrics:
        best_aug = results_df.groupby('Augmentation')[metric].mean().idxmax()
        best_aug_score = results_df.groupby('Augmentation')[metric].mean().max()
        best_aug_summary.append({'Metric': metric, 'Best Augmentation': best_aug, 'Score': best_aug_score})
        
        best_model = results_df.groupby(['Model', 'Augmentation'])[metric].mean().idxmax()
        best_model_score = results_df.groupby(['Model', 'Augmentation'])[metric].mean().max()
        best_model_summary.append({'Metric': metric, 'Best Model (Augmentation)': f"{best_model[0]} ({best_model[1]})", 'Score': best_model_score})
    
    # Display best augmentation summary
    best_aug_df = pd.DataFrame(best_aug_summary)
    log("Best Augmentation Technique for Each Metric:")
    display(best_aug_df)
    
    # Display best model summary
    best_model_df = pd.DataFrame(best_model_summary)
    log("Best Model for Each Metric:")
    display(best_model_df)

    # Visualizations per dataset
    for metric in metrics:
        for dataset in results_df['Dataset'].unique():
            dataset_df = results_df[results_df['Dataset'] == dataset]
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Model', y=metric, hue='Augmentation', data=dataset_df)
            plt.title(f"Model Comparison on {metric} - {dataset}", fontsize=14)
            plt.ylabel(metric, fontsize=12)
            plt.xlabel('Model', fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            plt.tight_layout()
            filename = f"{metric.replace(' ', '_')}_comparison_{dataset.replace('.csv', '')}.png"
            plt.savefig(filename)
            plt.close()
            log(f"Saved plot: {filename}")

    # Consolidated plot: Average performance across all datasets
    for metric in metrics:
        avg_df = results_df.groupby(['Model', 'Augmentation'])[metric].mean().reset_index()
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Model', y=metric, hue='Augmentation', data=avg_df)
        plt.title(f"Average {metric} Across All Datasets", fontsize=14)
        plt.ylabel(f"Average {metric}", fontsize=12)
        plt.xlabel('Model', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout()
        filename = f"average_{metric.replace(' ', '_')}_across_datasets.png"
        plt.savefig(filename)
        plt.close()
        log(f"Saved consolidated plot: {filename}")
else:
    log("No results to save or plot. Check dataset paths and ensure datasets are available.")

log("Script completed.")
log_file.close()