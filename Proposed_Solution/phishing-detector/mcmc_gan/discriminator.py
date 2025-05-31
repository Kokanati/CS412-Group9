# developer: Reginald Hingano 
# date: 31/05/2025
# This code is part of a GAN implementation for generating synthetic data.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Input

def build_discriminator(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512),
        LeakyReLU(alpha=0.2),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')  # Output probability of "realness"
    ])
    return model
