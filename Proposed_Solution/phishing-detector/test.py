# Developed by: CS412 Group 9

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from joblib import dump
from config import OUTPUT_DIR, TEST_SIZE, RANDOM_STATE

def log(msg):
    print(f"[LOG] {msg}")

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)

    log(f"{name} Results:")
    log(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {roc:.4f}")
    return acc, prec, rec, f1, roc

def main():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "augmented_dataset.csv"))
    y = df["label"]
    X = df.drop(columns=["label"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "CatBoost": CatBoostClassifier(verbose=0)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        evaluate_model(name, model, X_test, y_test)
        dump(model, os.path.join(OUTPUT_DIR, f"{name}.joblib"))

    # Build Stacking Ensemble
    estimators = [(name, model) for name, model in models.items()]
    stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    stacking.fit(X_train, y_train)
    evaluate_model("StackingEnsemble", stacking, X_test, y_test)
    dump(stacking, os.path.join(OUTPUT_DIR, "StackingEnsemble.joblib"))

if __name__ == "__main__":
    main()
