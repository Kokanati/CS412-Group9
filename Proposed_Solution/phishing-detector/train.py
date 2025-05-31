# Developed by: CS412 Group 9

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from mcmc_gan.generator import build_generator
from mcmc_gan.discriminator import build_discriminator
from mcmc_gan.mcmc_sampler import mcmc_sample_generator
from utils.feature_selector import select_top_k_features

from config import *

def log(msg):
    print(f"[LOG] {msg}")

def main():
    # Load data
    log("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = select_top_k_features(df, TARGET_COLUMN, k=TOP_K_FEATURES)

    X = df.drop(columns=[TARGET_COLUMN]).values
    y = df[TARGET_COLUMN].values

    X_train, _, y_train, _ = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    phishing_samples = X_train[y_train == 1]
    legit_samples = X_train[y_train == 0]
    n_to_generate = len(legit_samples) - len(phishing_samples)

    log(f"Generating {n_to_generate} synthetic phishing samples using MCMC-GAN...")

    input_dim = phishing_samples.shape[1]
    generator = build_generator(Z_DIM, input_dim)
    discriminator = build_discriminator(input_dim)

    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import BinaryCrossentropy

    bce = BinaryCrossentropy()
    disc_opt = Adam(0.0002)
    gen_opt = Adam(0.0002)

    @tf.function
    def train_step(real_batch):
        noise = tf.random.normal([real_batch.shape[0], Z_DIM])
        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            generated = generator(noise, training=True)
            real_out = discriminator(real_batch, training=True)
            fake_out = discriminator(generated, training=True)

            d_loss = (bce(tf.ones_like(real_out), real_out) + bce(tf.zeros_like(fake_out), fake_out)) / 2
            g_loss = bce(tf.ones_like(fake_out), fake_out)

        d_grads = disc_tape.gradient(d_loss, discriminator.trainable_variables)
        g_grads = gen_tape.gradient(g_loss, generator.trainable_variables)

        disc_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))
        gen_opt.apply_gradients(zip(g_grads, generator.trainable_variables))
        return d_loss, g_loss

    log("Training MCMC-GAN...")
    for epoch in range(EPOCHS):
        idx = np.random.permutation(len(phishing_samples))
        for i in range(0, len(phishing_samples), BATCH_SIZE):
            batch = phishing_samples[idx[i:i + BATCH_SIZE]]
            train_step(batch)
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            log(f"Epoch {epoch + 1}/{EPOCHS} completed.")

    synthetic_samples = mcmc_sample_generator(generator, discriminator, Z_DIM, n_to_generate, MCMC_STEPS, MCMC_STEP_SIZE)

    # Combine real and synthetic
    X_aug = np.vstack([legit_samples, phishing_samples, synthetic_samples])
    y_aug = np.concatenate([np.zeros(len(legit_samples)), np.ones(len(phishing_samples) + len(synthetic_samples))])
    X_aug, y_aug = shuffle(X_aug, y_aug, random_state=RANDOM_STATE)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    aug_path = os.path.join(OUTPUT_DIR, "augmented_dataset.csv")
    pd.DataFrame(X_aug).assign(label=y_aug).to_csv(aug_path, index=False)
    log(f"Balanced dataset saved to {aug_path}")

if __name__ == "__main__":
    main()
