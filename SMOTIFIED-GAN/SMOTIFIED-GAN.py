# === Install Required Libraries ===
#!pip install -q catboost xgboost scikit-learn pandas matplotlib tensorflow seaborn

# === Import Libraries ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.utils import shuffle
from google.colab import files

# === Upload dataset from local PC ===
uploaded = files.upload()
df = pd.read_csv(list(uploaded.keys())[0])

# === Split Features and Target ===
X = df.drop(columns=['phishing']).values
y = df['phishing'].values

# === Show Original Class Ratio ===
print("=== Class Ratio Before SMOTIFIED-GAN ===")
print(pd.Series(y).value_counts(normalize=True).rename({0: 'Legitimate', 1: 'Phishing'}).to_frame("Proportion"))

# === Initial Train/Test Split ===
X_train_base, X_test_final, y_train_base, y_test_final = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# === GAN Model Definitions ===
def build_generator(z_dim, output_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(z_dim,)),
        tf.keras.layers.Dense(128), tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(256), tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(512), tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(output_dim, activation='sigmoid')
    ])

def build_discriminator(input_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(512), tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(256), tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

# === Train and Apply SMOTIFIED-GAN ===
def augment_with_smotified_gan(X_train, y_train, z_dim=64, epochs=100, batch_size=64):
    phishing_data = X_train[y_train == 1].astype('float32')
    legit_count = np.sum(y_train == 0)
    phish_count = np.sum(y_train == 1)
    input_dim = phishing_data.shape[1]

    generator = build_generator(z_dim, input_dim)
    discriminator = build_discriminator(input_dim)

    bce = tf.keras.losses.BinaryCrossentropy()
    disc_opt = tf.keras.optimizers.Adam(0.0002)
    gen_opt = tf.keras.optimizers.Adam(0.0002)

    @tf.function
    def train_step(real_samples):
        noise = tf.random.normal([tf.shape(real_samples)[0], z_dim])
        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            fake_samples = generator(noise)
            real_output = discriminator(real_samples)
            fake_output = discriminator(fake_samples)
            disc_loss = (bce(tf.ones_like(real_output), real_output) + bce(tf.zeros_like(fake_output), fake_output)) / 2
            gen_loss = bce(tf.ones_like(fake_output), fake_output)
        disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
        disc_opt.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
        gen_opt.apply_gradients(zip(gen_grads, generator.trainable_variables))

    for epoch in range(epochs):
        idx = np.random.permutation(len(phishing_data))
        for i in range(0, len(phishing_data), batch_size):
            batch = phishing_data[idx[i:i+batch_size]]
            train_step(batch)

    synth_needed = legit_count - phish_count
    noise = tf.random.normal([synth_needed, z_dim])
    synthetic_phish = generator.predict(noise)
    final_X = np.vstack((X_train[y_train == 0], phishing_data, synthetic_phish))
    final_y = np.concatenate((np.zeros(legit_count), np.ones(phish_count + synth_needed)))
    return shuffle(final_X, final_y)

# === Apply GAN ===
X_bal, y_bal = augment_with_smotified_gan(X_train_base, y_train_base)

# === Class Distribution After SMOTIFIED-GAN ===
print("\n=== Class Ratio After SMOTIFIED-GAN ===")
print(pd.Series(y_bal).value_counts(normalize=True).rename({0: 'Legitimate', 1: 'Phishing'}).to_frame("Proportion"))

# === Standardization ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_bal)
X_test_scaled = scaler.transform(X_test_final)

# === Split Again for Training/Validation
X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_bal, test_size=0.2, stratify=y_bal, random_state=42)

# === Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=2000),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'CatBoost': CatBoostClassifier(verbose=0)
}
models['Stacking Ensemble'] = StackingClassifier(
    estimators=[('lr', models['Logistic Regression']), ('rf', models['Random Forest']), ('xgb', models['XGBoost'])],
    final_estimator=LogisticRegression()
)

# === Evaluation Metrics
results_test = []
results_train = []
cv_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test_scaled)

    # Train/test results
    results_train.append({
        "Model": name,
        "Accuracy": accuracy_score(y_train, y_pred_train),
        "Precision": precision_score(y_train, y_pred_train),
        "Recall": recall_score(y_train, y_pred_train),
        "F1 Score": f1_score(y_train, y_pred_train),
        "ROC-AUC": roc_auc_score(y_train, y_pred_train),
    })
    results_test.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test_final, y_pred_test),
        "Precision": precision_score(y_test_final, y_pred_test),
        "Recall": recall_score(y_test_final, y_pred_test),
        "F1 Score": f1_score(y_test_final, y_pred_test),
        "ROC-AUC": roc_auc_score(y_test_final, y_pred_test),
    })

    # Cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_score = cross_validate(model, X_train_scaled, y_bal, cv=cv, scoring=['accuracy', 'f1', 'roc_auc'], n_jobs=-1)
    cv_scores[name] = {
        'Accuracy': np.mean(cv_score['test_accuracy']),
        'F1 Score': np.mean(cv_score['test_f1']),
        'ROC-AUC': np.mean(cv_score['test_roc_auc']),
    }

# === Convert to DataFrames
df_train = pd.DataFrame(results_train)
df_test = pd.DataFrame(results_test)
df_cv = pd.DataFrame(cv_scores).T

# === Export CSV
df_test.to_csv("test_results.csv", index=False)
df_train.to_csv("train_results.csv", index=False)
df_cv.to_csv("cv_results.csv")

# === Confusion Matrices
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test_final, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# === ROC Curves
for name, model in models.items():
    RocCurveDisplay.from_estimator(model, X_test_scaled, y_test_final)
    plt.title(f"ROC Curve: {name}")
    plt.grid(True)
    plt.show()

# === Rankings and Summary
def rank_metrics(df, label):
    print(f"\n=== Model Ranking on {label} Set ===")
    for metric in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]:
        print(f"\nRanking by {metric}:")
        print(df.set_index("Model")[metric].sort_values(ascending=False))

rank_metrics(df_train, "Train")
rank_metrics(df_test, "Test")
print("\n=== Cross-Validation Scores ===")
print(df_cv)
