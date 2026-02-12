import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np # Wird für tf.Variable Initialisierung benötigt

from .config import LEARNING_RATE
# Hypersphere wird aktuell nicht genutzt
class HypersphereNormalization(layers.Layer):
    def __init__(self, **kwargs):
        super(HypersphereNormalization, self).__init__(**kwargs)

    def call(self, inputs):
        norm = tf.norm(inputs, ord='euclidean', axis=-1, keepdims=True)
        # Add epsilon to prevent division by zero
        normalized_output = inputs / (norm + tf.keras.backend.epsilon())
        return normalized_output

class Sampling(layers.Layer):
    def call(self, inputs):
        mu, logvar = inputs
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * logvar) * eps

class VAE(Model):
    def __init__(self, input_shape, latent_dim, beta=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder: gleiche Conv/Pooling Struktur,
        # aber am Ende 2*latent_dim Kanäle -> split in mu/logvar
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv1D(64, 8, activation="relu", padding="same"),
            layers.MaxPooling1D(2, padding="same"),
            layers.Conv1D(32, 4, activation="relu", padding="same"),
            layers.MaxPooling1D(2, padding="same"),
            layers.Conv1D(2 * latent_dim, 3, activation=None, padding="same", name="encoder_mu_logvar"),
            layers.BatchNormalization(),
        ], name="encoder")

        self.sampling = Sampling()

        # Decoder bleibt wie bei dir
        self.decoder = tf.keras.Sequential([
            layers.Conv1DTranspose(32, 3, strides=2, activation="relu", padding="same"),
            layers.Conv1DTranspose(64, 4, strides=2, activation="relu", padding="same"),
            layers.Conv1D(input_shape[-1], 8, activation=None, padding="same"),
        ], name="decoder")

    def call(self, x, training=False):
        enc = self.encoder(x, training=training)  # (B, T, 2*latent_dim)
        mu, logvar = tf.split(enc, num_or_size_splits=2, axis=-1)

        z = self.sampling((mu, logvar))
        recon = self.decoder(z, training=training)

        # KL divergence (über alle dims gemittelt)
        kl = -0.5 * tf.reduce_mean(1.0 + logvar - tf.square(mu) - tf.exp(logvar))
        self.add_loss(self.beta * kl)

        return recon

    def get_latent_mu(self, x):
        enc = self.encoder(x, training=False)
        mu, _ = tf.split(enc, 2, axis=-1)
        return mu

# Die Funktion zum Trainieren und Evaluieren des VAE
def train_and_evaluate_vae(train_data, test_data, input_shape, latent_dim, beta=1.0,
                           epochs=10, batch_size=32, learning_rate=1e-4):
    vae = VAE(input_shape=input_shape, latent_dim=latent_dim, beta=beta)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    vae.compile(optimizer=optimizer, loss="mse")  # mse = Reconstruction; KL kommt via add_loss()

    print("\nStarte Training (VAE):")
    print(f"  Latent Dim: {latent_dim}")
    print(f"  beta (KL-Gewicht): {beta}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Epochs: {epochs}, Batch Size: {batch_size}")

    history = vae.fit(
        train_data, train_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(test_data, test_data),
        verbose=1
    )

    total_loss = vae.evaluate(test_data, test_data, verbose=0)
    print(f"Validierungs-Gesamtfehler (Rekonstruktion + beta*KL): {total_loss:.4f}")

    return vae, history, total_loss
