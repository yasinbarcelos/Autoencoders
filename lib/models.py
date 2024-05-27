import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.layers import Layer, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K

class AutoencoderModel(Model):
    def __init__(self, input_shape, latent_dim, activity_regularizer, **kwargs):
        super(AutoencoderModel, self).__init__(**kwargs)
        
        # Definindo o encoder
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(shape=input_shape),
            layers.Dense(224, activation='relu'),
            layers.Dense(84, activation='relu'),
            layers.Dense(latent_dim, activation='relu', activity_regularizer=activity_regularizer)
        ])
        
        # Definindo o decoder
        self.decoder = tf.keras.Sequential([
            layers.InputLayer(shape=(latent_dim,)),            
            layers.Dense(84, activation='relu'),
            layers.Dense(224, activation='relu'),
            layers.Dense(input_shape[0], activation='sigmoid')
        ])
    
    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
    
    


class KSparse(Layer):
    def __init__(self, k, **kwargs):
        super(KSparse, self).__init__(**kwargs)
        self.k = k

    def call(self, inputs):
        # Obter os valores top-k
        top_k_values, _ = tf.nn.top_k(inputs, k=self.k, sorted=False)
        # Obter o valor mínimo entre os top-k
        min_top_k_values = tf.reduce_min(top_k_values, axis=-1, keepdims=True)
        # Manter apenas os top-k valores e zerar o restante
        sparse_outputs = tf.where(inputs >= min_top_k_values, inputs, tf.zeros_like(inputs))
        return sparse_outputs
  
class SparseAutoencoderModel(Model):
    def __init__(self, input_shape, latent_dim, k, **kwargs):
        super(SparseAutoencoderModel, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            Input(shape=input_shape),
            Dense(latent_dim, activation='relu'),
            KSparse(k)  # Aplicando k-sparsity
        ])
        self.decoder = tf.keras.Sequential([
            Dense(input_shape[0], activation='sigmoid')
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    def get_encoder(self):
        input_img = Input(shape=(784,))
        encoded = self.encoder(input_img)
        return Model(input_img, encoded)

    def get_decoder(self):
        encoded_input = Input(shape=(self.latent_dim,))
        decoded = self.decoder(encoded_input)
        return Model(encoded_input, decoded)
    
    
    


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the latent vector."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

class VariationalAutoencoder(Model):
    def __init__(self, input_shape, latent_dim, activity_regularizer=None, **kwargs):
        super(VariationalAutoencoder, self).__init__(**kwargs)
        
        # Definindo o encoder
        self.encoder_input = layers.InputLayer(input_shape=input_shape)
        self.dense1 = layers.Dense(140, activation='relu')
        self.dense2 = layers.Dense(84, activation='relu')
        self.z_mean = layers.Dense(latent_dim, name='z_mean')
        self.z_log_var = layers.Dense(latent_dim, name='z_log_var')
        self.sampling = Sampling()
        
        # Definindo o decoder
        self.decoder_input = layers.InputLayer(input_shape=(latent_dim,))
        self.dense3 = layers.Dense(84, activation='relu')
        self.dense4 = layers.Dense(140, activation='relu')
        self.decoder_output = layers.Dense(input_shape[0], activation='sigmoid')

    def encode(self, inputs):
        x = self.encoder_input(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling([z_mean, z_log_var])
        return z_mean, z_log_var, z

    def decode(self, z):
        x = self.decoder_input(z)
        x = self.dense3(x)
        x = self.dense4(x)
        return self.decoder_output(x)
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encode(inputs)
        decoded = self.decode(z)
        kl_loss = -0.5 * tf.reduce_sum(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1, axis=-1)
        self.add_loss(tf.reduce_mean(kl_loss) / 784)
        return decoded

    def get_encoder(self):
        return Model(self.encoder_input, [self.z_mean, self.z_log_var, self.sampling([self.z_mean, self.z_log_var])])

    def get_decoder(self):
        return Model(self.decoder_input, self.decode(self.decoder_input))
    
def vae_loss(inputs, outputs):
    reconstruction_loss = mse(inputs, outputs) * 784
    return tf.reduce_mean(reconstruction_loss)