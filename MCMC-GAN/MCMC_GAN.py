
# === MCMC-GAN Training & Evaluation Pipeline (Keras) ===

import numpy as np
import pandas as pd
import os
import sys
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, LeakyReLU, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.utils import shuffle
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Logging setup
csv_path = sys.argv[1]
dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
log_dir = os.path.join("logs", dataset_name)
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "training_log.txt")
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
def log(msg): print(msg); logging.info(msg)

# Define Generator and Discriminator
def build_generator(z_dim, output_dim):
    return Sequential([
        Input(shape=(z_dim,)),
        Dense(128), BatchNormalization(), ReLU(),
        Dense(256), BatchNormalization(), ReLU(),
        Dense(512), BatchNormalization(), ReLU(),
        Dense(output_dim, activation='sigmoid')
    ])

def build_discriminator(input_dim):
    return Sequential([
        Input(shape=(input_dim,)),
        Dense(512), LeakyReLU(0.2),
        Dense(256), LeakyReLU(0.2),
        Dense(1, activation='sigmoid')
    ])

# MCMC Sampling Function
def mcmc_sample_generator(generator, discriminator, z_dim, num_samples, n_steps=10, step_size=0.05):
    accepted = []
    for _ in range(num_samples):
        z = tf.random.normal([1, z_dim])
        z_current = tf.identity(z)
        g_current = generator(z_current, training=False)
        d_current = discriminator(g_current, training=False)

        for _ in range(n_steps):
            z_proposed = z_current + tf.random.normal([1, z_dim], stddev=step_size)
            g_proposed = generator(z_proposed, training=False)
            d_proposed = discriminator(g_proposed, training=False)

            p_current = tf.squeeze(d_current)
            p_proposed = tf.squeeze(d_proposed)
            accept_prob = tf.minimum(1.0, p_proposed / (p_current + 1e-8))

            if tf.random.uniform([]) < accept_prob:
                z_current = z_proposed
                d_current = d_proposed

        accepted.append(generator(z_current, training=False).numpy())
    return np.vstack(accepted)

# GAN training & augmentation
def augment_with_mcmc_gan(X_train, y_train, z_dim=64, epochs=100, batch_size=64):
    legit_count = np.sum(y_train == 0)
    phish_count = np.sum(y_train == 1)
    total = legit_count + phish_count
    log(f"Before augmentation: Legitimate={legit_count} ({(legit_count/total)*100:.2f}%), Phishing={phish_count} ({(phish_count/total)*100:.2f}%)")

    phishing_data = X_train[y_train == 1].astype('float32')
    input_dim = phishing_data.shape[1]
    generator = build_generator(z_dim, input_dim)
    discriminator = build_discriminator(input_dim)

    bce = BinaryCrossentropy()
    disc_opt = Adam(learning_rate=0.0002)
    gen_opt = Adam(learning_rate=0.0002)

    @tf.function
    def train_step(real_samples):
        noise = tf.random.normal([tf.shape(real_samples)[0], z_dim])
        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            generated = generator(noise)
            real_out = discriminator(real_samples)
            fake_out = discriminator(generated)
            d_loss = (bce(tf.ones_like(real_out), real_out) + bce(tf.zeros_like(fake_out), fake_out)) / 2
            g_loss = bce(tf.ones_like(fake_out), fake_out)
        disc_grads = disc_tape.gradient(d_loss, discriminator.trainable_variables)
        gen_grads = gen_tape.gradient(g_loss, generator.trainable_variables)
        disc_opt.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
        gen_opt.apply_gradients(zip(gen_grads, generator.trainable_variables))
        return d_loss, g_loss

    for epoch in range(epochs):
        idx = np.random.permutation(len(phishing_data))
        for i in range(0, len(phishing_data), batch_size):
            batch = phishing_data[idx[i:i+batch_size]]
            train_step(batch)
        if epoch % 10 == 0:
            log(f"Epoch {epoch+1}/{epochs} complete.")

    num_to_generate = legit_count - phish_count
    synthetic_samples = mcmc_sample_generator(generator, discriminator, z_dim, num_to_generate, n_steps=10)
    legitimate_data = X_train[y_train == 0]
    X_aug = np.vstack([legitimate_data, phishing_data, synthetic_samples])
    y_aug = np.concatenate([np.zeros(len(legitimate_data)), np.ones(len(phishing_data) + len(synthetic_samples))])
    return shuffle(X_aug, y_aug)

# Main pipeline
if __name__ == "__main__":
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    X_aug, y_aug = augment_with_mcmc_gan(X_train, y_train)
    aug_dir = f"outputs/{dataset_name}"; os.makedirs(aug_dir, exist_ok=True)
    pd.DataFrame(X_aug).assign(label=y_aug).to_csv(f"{aug_dir}/augmented_dataset.csv", index=False)
    log(f"Augmented dataset saved to '{aug_dir}/augmented_dataset.csv'")

    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_aug, y_aug, test_size=0.3, stratify=y_aug, random_state=42)

    models = {
        'Logistic_Regression': LogisticRegression(max_iter=1000),
        'Random_Forest': RandomForestClassifier(),
        'XGBoost': XGBClassifier(eval_metric='logloss'),
        'CatBoost': CatBoostClassifier(verbose=0),
        'Stacking_Ensemble': StackingClassifier(
            estimators=[
                ('lr', LogisticRegression(max_iter=1000)),
                ('rf', RandomForestClassifier()),
                ('xgb', XGBClassifier(eval_metric='logloss'))
            ],
            final_estimator=LogisticRegression()
        )
    }

    results = []
    for name, model in models.items():
        log(f"Training {name}...")
        model.fit(X_train_final, y_train_final)
        y_pred = model.predict(X_test_final)
        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test_final, y_pred),
            'Precision': precision_score(y_test_final, y_pred),
            'Recall': recall_score(y_test_final, y_pred),
            'F1 Score': f1_score(y_test_final, y_pred),
            'ROC-AUC': roc_auc_score(y_test_final, y_pred)
        }
        model_path = f"{aug_dir}/models/{name}.joblib"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        dump(model, model_path)
        log(f"Saved model to {model_path}")
        results.append(metrics)

    pd.DataFrame(results).to_csv(f"{aug_dir}/model_performance_summary.csv", index=False)
    log("Evaluation complete.")
