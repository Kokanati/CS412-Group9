# developed by: Reginald Hingano

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Input

def build_generator(z_dim, output_dim):
    model = Sequential([
        Input(shape=(z_dim,)),
        Dense(128),
        BatchNormalization(),
        ReLU(),
        Dense(256),
        BatchNormalization(),
        ReLU(),
        Dense(512),
        BatchNormalization(),
        ReLU(),
        Dense(output_dim, activation='sigmoid')  
    ])
    return model
