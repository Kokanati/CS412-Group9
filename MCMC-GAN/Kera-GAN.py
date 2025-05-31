# === GAN Training & Integration with Data Augmentation ===
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# === Logging Setup ===
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
csv_path = sys.argv[1]
dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
dataset_log_dir = os.path.join(log_dir, dataset_name)
os.makedirs(dataset_log_dir, exist_ok=True)
log_file = os.path.join(dataset_log_dir, "training_log.txt")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Console print wrapper
def log(msg):
    print(msg)
    logging.info(msg)

# === Generator and Discriminator Definitions (TensorFlow/Keras) ===
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, LeakyReLU, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf

def build_generator(z_dim, output_dim):
    model = Sequential([
        Input(shape=(z_dim,)),
        Dense(128), BatchNormalization(), ReLU(),
        Dense(256), BatchNormalization(), ReLU(),
        Dense(512), BatchNormalization(), ReLU(),
        Dense(output_dim, activation='sigmoid')
    ])
    return model

def build_discriminator(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512), LeakyReLU(0.2),
        Dense(256), LeakyReLU(0.2),
        Dense(1, activation='sigmoid')
    ])
    return model

def augment_with_gan(X_train, y_train, z_dim=64, epochs=100, batch_size=64):
    legit_count = np.sum(y_train == 0)
    phish_count = np.sum(y_train == 1)
    total = legit_count + phish_count
    log(f"Before augmentation: Legitimate={legit_count} ({(legit_count/total)*100:.2f}%), Phishing={phish_count} ({(phish_count/total)*100:.2f}%)")
    log("Initializing Keras-based GAN for data augmentation...")

    minority_class_label = 1
    phishing_data = X_train[y_train == minority_class_label].astype('float32')

    input_dim = phishing_data.shape[1]
    generator = build_generator(z_dim, input_dim)
    discriminator = build_discriminator(input_dim)

    bce = BinaryCrossentropy()
    disc_opt = Adam(learning_rate=0.0002)
    gen_opt = Adam(learning_rate=0.0002)

    @tf.function
    def train_step(real_samples):
        batch_size = tf.shape(real_samples)[0]
        noise = tf.random.normal([batch_size, z_dim])

        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            generated_samples = generator(noise)
            real_output = discriminator(real_samples)
            fake_output = discriminator(generated_samples)
            disc_loss = (bce(tf.ones_like(real_output), real_output) + bce(tf.zeros_like(fake_output), fake_output)) / 2
            gen_loss = bce(tf.ones_like(fake_output), fake_output)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

        disc_opt.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        gen_opt.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

        return disc_loss, gen_loss

    for epoch in range(epochs):
        idx = np.random.permutation(len(phishing_data))
        for i in range(0, len(phishing_data), batch_size):
            batch = phishing_data[idx[i:i+batch_size]]
            disc_loss, gen_loss = train_step(batch)

        if epoch % 10 == 0 or epoch == epochs - 1:
            log(f"Epoch {epoch+1}/{epochs} | D Loss: {disc_loss:.4f} | G Loss: {gen_loss:.4f}")

    # Generate enough synthetic phishing samples to balance the dataset
    num_fake_samples = legit_count - phish_count
    log("GAN training complete. Generating synthetic phishing samples...")
    noise = tf.random.normal([num_fake_samples, z_dim])
    synthetic_samples = generator.predict(noise)

    legitimate_data = X_train[y_train == 0]
    X_augmented = np.vstack((legitimate_data, phishing_data, synthetic_samples))
    y_augmented = np.concatenate((np.zeros(len(legitimate_data)), np.ones(len(phishing_data) + len(synthetic_samples))))

    total_aug = len(X_augmented)
    legit_aug = np.sum(y_augmented == 0)
    synth_phish = np.sum(y_augmented == 1)
    log(f"After augmentation: Legitimate={legit_aug} ({(legit_aug/total_aug)*100:.2f}%), Synthetic Phishing={synth_phish} ({(synth_phish/total_aug)*100:.2f}%)")
    log("Keras-based GAN augmentation complete.")
    return shuffle(X_augmented, y_augmented)

if __name__ == "__main__":
    log("\n=== Starting GAN-Augmented Model Evaluation Pipeline ===")

    df = pd.read_csv(csv_path)
    log(f"Dataset loaded from {csv_path}. Shape: {df.shape}")

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_aug, y_aug = augment_with_gan(X_train, y_train)


    # Save augmented dataset
    aug_dir = f"outputs/{dataset_name}"
    os.makedirs(aug_dir, exist_ok=True)
    augmented_df = pd.DataFrame(X_aug)
    augmented_df['label'] = y_aug
    augmented_df.to_csv(f"{aug_dir}/augmented_dataset.csv", index=False)
    log(f"Augmented dataset saved to '{aug_dir}/augmented_dataset.csv'")

    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_aug, y_aug, test_size=0.3, stratify=y_aug, random_state=42)

    from joblib import dump
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

    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    import time

    os.makedirs(f"{aug_dir}/confusion_matrices", exist_ok=True)

    results = []
    for name, model in models.items():
        log(f"===== Evaluating {name.replace('_', ' ')} =====")
        start_time = time.time()
        model.fit(X_train_final, y_train_final)
        y_pred_train = model.predict(X_train_final)
        y_pred_test = model.predict(X_test_final)

        metrics = {
            'Train Accuracy': accuracy_score(y_train_final, y_pred_train),
            'Test Accuracy': accuracy_score(y_test_final, y_pred_test),
            'Precision': precision_score(y_test_final, y_pred_test),
            'Recall': recall_score(y_test_final, y_pred_test),
            'ROC-AUC': roc_auc_score(y_test_final, y_pred_test),
            'F1 Score': f1_score(y_test_final, y_pred_test)
        }

        end_time = time.time()
        metrics['Runtime (s)'] = end_time - start_time
        results.append({"Model": name, **metrics})

        model_path = f"{aug_dir}/models/{name}.joblib"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        dump(model, model_path)
        log(f"Saved trained model to {model_path}")

        # Confusion Matrices
        plt.figure()
        ConfusionMatrixDisplay.from_predictions(y_test_final, y_pred_test, display_labels=["Legitimate", "Phishing"], cmap='Blues')
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(f"{aug_dir}/confusion_matrices/{name}_confusion_matrix.png")
        plt.close()

        plt.figure()
        ConfusionMatrixDisplay.from_predictions(y_test_final, y_pred_test, display_labels=["Legitimate", "Phishing"], cmap='Blues', normalize='true')
        plt.title(f"Normalized Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(f"{aug_dir}/confusion_matrices/{name}_confusion_matrix_normalized.png")
        plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{aug_dir}/model_performance_summary.csv", index=False)
    log(f"Model evaluation summary saved to '{aug_dir}/model_performance_summary.csv'")

    plt.figure(figsize=(10, 6))
    for metric in ['Test Accuracy', 'Precision', 'Recall', 'ROC-AUC', 'F1 Score']:
        plt.plot(results_df['Model'], results_df[metric], marker='o', label=metric)
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Comparison by Evaluation Metrics')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{aug_dir}/model_comparison_plot.png")
    plt.close()
    log(f"Performance comparison plot saved to '{aug_dir}/model_comparison_plot.png'")

    log("=== Pipeline execution complete. ===")
