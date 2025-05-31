#developed by: Reginald Hingano

import numpy as np
import tensorflow as tf

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
